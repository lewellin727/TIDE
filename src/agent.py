import os
import json
import logging
from typing import List

from agentscope.agent import AgentBase
from agentscope.tool import Toolkit
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg, ToolUseBlock

from src.table import Table
from src.query_constraint import QueryConstraint
from src.tgr import rerank_paths, render_path_with_schema
from src.utils import PLANNING_SYS_PROMPT, OBSERVATION_STAGE_SYS_PROMPT, add_token_usage, LLM_BACKBONE
from src.operators import tool as tool_ctx
from src.operators.tool import (
    join_search, union_search, correlation_search,
    eq_ranker, range_ranker, metadata_seeker,
    combine, end_search,
)
from src.operators.ranker import CategoricalRanker, NumericalRanker


PLANNING_TOOL_GROUP = "planning_operators"
SEED_TOOL_GROUP = "seed_operator"  

logger = logging.getLogger("tide")


def _trim_for_log(obj, limit: int = 180) -> str:
    """Compact a tool input dict / arbitrary obj into a single line capped at `limit` chars."""
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    text = text.replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def _record_usage(response):
    """Pull input/output token counts out of an agentscope ChatResponse.usage and
    feed them into the process-wide counter. Returns (in_tok, out_tok) for logging."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    add_token_usage(in_tok, out_tok)
    return in_tok, out_tok


class TableAgent(AgentBase):


    def __init__(self, rel_graph, reasoning_tree, max_loop=8,
                 k_1=10, k_o=3, d_o=2, query_idx=None):
        super().__init__()

        self.rel_graph = rel_graph
        self.reasoning_tree = reasoning_tree
        self.max_loop = max_loop
        self.k_1 = k_1
        self.k_o = k_o
        self.d_o = d_o
        self.selected_node = None
        self.query_idx = query_idx
        self._loop_idx = 0

        tool_ctx.set_reasoning_tree(reasoning_tree)

        self.name = "tide"
        self.observation_sys_prompt = OBSERVATION_STAGE_SYS_PROMPT
        self.model = DashScopeChatModel(
            model_name=LLM_BACKBONE,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=False,
            enable_thinking=False,
        )
        self.formatter = DashScopeChatFormatter()
        self.toolkit = Toolkit()
        self.toolkit.create_tool_group(
            group_name=PLANNING_TOOL_GROUP,
            description="Operators for table discovery planning (relationships, value filters, combiner, termination).",
            active=True,
        )
        self.toolkit.create_tool_group(
            group_name=SEED_TOOL_GROUP,
            description="Seed operator: flat name-search to open a search direction on an empty node.",
            active=True,
        )

        for fn in (
            join_search, union_search, correlation_search,  
            eq_ranker, range_ranker,                         
            combine,                                     
            end_search,                               
        ):
            self.toolkit.register_tool_function(fn, group_name=PLANNING_TOOL_GROUP)
        self.toolkit.register_tool_function(metadata_seeker, group_name=SEED_TOOL_GROUP)

        self.planning_sys_prompt = PLANNING_SYS_PROMPT

    async def discovery(self, query) -> List[Table]:
        best_cnt = None   
        stagnant = 0
        for loop in range(self.max_loop):
            self._loop_idx = loop + 1
            self.select_node()
            if self.selected_node is None:
                logger.info("%s no activate leafs left → exiting loop early", self._tag())
                break
            depth = self.reasoning_tree.depth_of(self.selected_node)
            util = self.reasoning_tree.utility(self.selected_node)
            util_str = "inf" if util == float('inf') else f"{util:.2f}"
            logger.info(
                "%s SELECT node=%s depth=%d constraint_cnt=%d input_tables=%d utility=%s",
                self._tag(), self.selected_node.id, depth,
                self.selected_node.constraint_cnt, len(self.selected_node.input_tables),
                util_str,
            )
            self.observation()
            logger.info(
                "%s OBS paths=%d navi_tables=%d cand_tables=%d",
                self._tag(),
                len(self.selected_node.paths),
                len(self.selected_node.navi_tables),
                len(self.reasoning_tree.get_cand_tables(self.selected_node.id)),
            )

            # Observation Stage: LLM extracts structured evidence from col_constraints + paths;
            # Python then decorates with sketch-based value-check status so the Planner sees
            # which constraints are *actually* satisfied vs still needing a value ranker.
            obs_stage_prompt = self.reasoning_tree.gen_observation_stage_prompt(self.selected_node, query)
            evidence = await self.observe_llm(obs_stage_prompt)
            evidence = self._post_process_evidence(evidence)
            self._log_evidence(evidence)

            # Planning Stage: LLM picks next tool call(s) using the evidence.
            planning_prompt = self.reasoning_tree.gen_planning_prompt(
                query, self.selected_node, evidence,
            )
            msg_reasoning = await self.planning(planning_prompt)
            tool_names = [
                (b.get("name") if isinstance(b, dict) else getattr(b, "name", None))
                for b in msg_reasoning.content
                if (b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "tool_use"
            ]
            logger.info("%s PLAN tools=%s", self._tag(), tool_names)

            # Execution Stage: walk through the tool calls in msg_reasoning and update the search tree.
            await self.execution(msg_reasoning)

            # No-progress guard
            cur = min((n.constraint_cnt for n in self.reasoning_tree.nodes if not n.error), default=None)
            if cur is not None and (best_cnt is None or cur < best_cnt):
                best_cnt = cur
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= 3:
                    logger.info("%s no progress for %d loops (min_constraint_cnt=%s) → terminating",
                                self._tag(), stagnant, best_cnt)
                    break

        results = self.reasoning_tree.get_results()
        rt = self.reasoning_tree
        terminated_any = any(rt.adj_dict[n.id] == [] and n.terminated for n in rt.nodes)
        termination = "empty" if not results else ("end_search" if terminated_any else "degraded")
        logger.info(
            "%s DONE loops_used=%d total_nodes=%d candidates=%d termination=%s",
            self._tag(loop=False),
            self._loop_idx,
            len(self.reasoning_tree.nodes),
            len(results),
            termination,
        )
        return results

    def _tag(self, loop: bool = True) -> str:
        """Build a `[Q{idx} L{loop}]` log prefix for per-query/per-loop events."""
        q = self.query_idx if self.query_idx is not None else "?"
        if loop and self._loop_idx:
            return f"[Q{q} L{self._loop_idx}]"
        return f"[Q{q}]"

    def _log_evidence(self, evidence) -> None:
        """One log line: the ranked frontier directions + stuck constraints."""
        if evidence is None:
            logger.info("%s OBS_LLM evidence=none (all retries exhausted)", self._tag())
            return
        if not isinstance(evidence, dict) or not evidence:
            logger.info("%s OBS_LLM evidence=empty", self._tag())
            return
        dirs = evidence.get("directions") or []
        top = dirs[0] if dirs else {}
        logger.info(
            "%s OBS_LLM clusters=%d top_via=%s top_relates=%s stuck=%s raw=%s",
            self._tag(),
            len(dirs),
            top.get("via", "-"),
            top.get("relates_to", []),
            evidence.get("stuck") or "[]",
            _trim_for_log(evidence, limit=400),
        )


    def select_node(self):
        # Max utility U(v) = (|s_q^0| - |v.s_q|) / depth(v)
        leafs = self.reasoning_tree.get_activate_leafs()
        if not leafs:
            self.selected_node = None
            return
        def _key(node):
            u = self.reasoning_tree.utility(node)
            depth = self.reasoning_tree.depth_of(node)
            return (-u, node.constraint_cnt, -depth)
        self.selected_node = min(leafs, key=_key)


    def observation(self):
        leaf = self.selected_node

        target_tables = QueryConstraint.search(leaf.col_constraints)
        cand_tables = self.reasoning_tree.get_cand_tables(leaf.id)
        root_cons = self.reasoning_tree._root_constraints() or leaf.col_constraints
        seeds = [t for t in cand_tables if any(QueryConstraint._satisfies(c, t) for c in root_cons)]
        if not seeds:
            seeds = list(target_tables or [])
        seeds = seeds[: max(1, self.reasoning_tree.cand_top_k)]

        raw_paths = self.rel_graph.navigate(seeds, target_tables, d_o=self.d_o, k_1=self.k_1)
        final_paths = rerank_paths(
            raw_paths, leaf, self.reasoning_tree, self.rel_graph, k_o=self.k_o,
        )
        cand_names = {t.table_name for t in cand_tables}
        navi_table_names = list(
            {name for p in final_paths for name in p['tables']} - cand_names
        )
        tables = self.rel_graph.get_navi_tables(navi_table_names)

        name_to_schema = {t.table_name: t.to_schema() for t in tables}
        for t in cand_tables:
            name_to_schema.setdefault(t.table_name, t.to_schema())

        paths = [render_path_with_schema(p, name_to_schema) for p in final_paths]

        endpoint_names = {p['endpoint'] for p in final_paths}
        endpoint_tables = self.rel_graph.get_navi_tables(list(endpoint_names))
        self._value_hints = self._precheck_value_conditions(
            leaf.col_constraints, endpoint_tables
        )
        leaf.set_navi(tables, paths)

    def _precheck_value_conditions(self, constraints, endpoint_tables):
        hints = {}
        if not constraints or not endpoint_tables:
            return hints
        for i, c in enumerate(constraints):
            values = c.get('values') or []
            if not values:
                continue 
            cid = f"c{i + 1}"
            col_name = c['col_name']
            col_type = c['col_type']
            for table in endpoint_tables:
                try:
                    if col_type == 'Categorical':
                        verified = False
                        for v in values:
                            if v is None:
                                continue
                            ranked = CategoricalRanker.rank_frequence([table], col_name, v) or []
                            if ranked:
                                verified = True
                                break
                        hints[(cid, table.table_name)] = verified
                    elif col_type == 'Numerical' and len(values) == 2:
                        lo, hi = values[0], values[1]
                        ranked = NumericalRanker.rank_range([table], col_name, lo, hi) or []
                        hints[(cid, table.table_name)] = bool(ranked)
                except Exception:
                    continue
        return hints

    def _post_process_evidence(self, evidence):

        if not isinstance(evidence, dict):
            return evidence
        evidence["directions"] = list(evidence.get("directions") or [])
        evidence["stuck"] = list(evidence.get("stuck") or [])
        return evidence


    async def observe_llm(self, stage_prompt):

        self.toolkit.update_tool_groups([PLANNING_TOOL_GROUP, SEED_TOOL_GROUP], active=False)
        try:
            hint = (
                "\n\nNOTE: Your previous response was not valid JSON. "
                "Output ONLY the JSON object — no explanation, no markdown fences, nothing else."
            )
            for attempt in range(3):
                user_content = stage_prompt if attempt == 0 else (stage_prompt + hint)
                prompt = await self.formatter.format([
                    Msg("system", self.observation_sys_prompt, "system"),
                    Msg(name="user", content=user_content, role="user"),
                ])
                response = await self.model(
                    prompt,
                    tools=self.toolkit.get_json_schemas(),
                    tool_choice="none",
                )
                in_tok, out_tok = _record_usage(response)
                logger.info(
                    "%s LLM stage=obs attempt=%d in_tok=%d out_tok=%d",
                    self._tag(), attempt + 1, in_tok, out_tok,
                )
                text = self._extract_text(response)
                parsed = self._parse_json(text)
                if parsed is not None:
                    return parsed
            return None
        finally:
            self.toolkit.update_tool_groups([PLANNING_TOOL_GROUP], active=True)

    @staticmethod
    def _extract_text(response) -> str:
        parts = []
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()

    @staticmethod
    def _parse_json(text: str):
        if not text:
            return None
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            t = "\n".join(lines).strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        start = t.find("{")
        end = t.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None

    async def planning(self, planning_prompt):
        self.toolkit.update_tool_groups([PLANNING_TOOL_GROUP], active=True)
        self.toolkit.update_tool_groups(
            [SEED_TOOL_GROUP], active=not self.selected_node.input_tables)
        original_prompt = planning_prompt

        msg_reasoning = None
        retry_hint = (
            "\n\n--- RETRY NOTICE ---\n"
            "Your previous response contained no tool call. You MUST call at least one "
            "tool this turn. If the search is complete, call `end_search` (no arguments). "
            "Otherwise call the operator that best advances the unresolved Query State."
        )
        for attempt in range(3):
            current_prompt = original_prompt if attempt == 0 else (original_prompt + retry_hint)
            msg = Msg(name="user", content=current_prompt, role="user")
            prompt = await self.formatter.format([
                Msg("system", self.planning_sys_prompt, "system"),
                msg,
            ])
            response = await self.model(
                prompt,
                tools=self.toolkit.get_json_schemas(),
                tool_choice="auto",
            )
            in_tok, out_tok = _record_usage(response)
            logger.info(
                "%s LLM stage=plan attempt=%d in_tok=%d out_tok=%d",
                self._tag(), attempt + 1, in_tok, out_tok,
            )
            msg_reasoning = Msg(
                name=self.name,
                content=list(response.content),
                role="assistant",
            )
            if self.check(msg_reasoning):
                break
        return msg_reasoning


    async def execution(self, msg_reasoning) -> None:
        self.selected_node.s_p = [
            {"operator": b.get("name"), "input": b.get("input")}
            for b in msg_reasoning.content
            if (b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "tool_use"
        ]

        for block in msg_reasoning.content:
            btype = block.get('type') if isinstance(block, dict) else getattr(block, 'type', None)
            if btype != 'tool_use':
                continue

            tool_call = block
            _op = tool_call.get("name")
            if _op != "end_search" and self.reasoning_tree.already_tried(_op, tool_call.get("input")):
                logger.info("%s SKIPPED_REPEAT op=%s input=%s", self._tag(), _op,
                            _trim_for_log(tool_call.get("input")))
                continue

            if _op == "metadata_seeker" and self.selected_node.input_tables:
                logger.info("%s METADATA_BLOCKED (node already holds %d tables — must traverse) input=%s",
                            self._tag(), len(self.selected_node.input_tables),
                            _trim_for_log(tool_call.get("input")))
                continue
            try:
                tool_ctx.set_current_node(self.selected_node.id)
                output_blocks = await self.tool_use(tool_call)
                output = output_blocks[0] if output_blocks else None
                operator = tool_call["name"]

                output_text = ""
                if output is not None:
                    output_text = (
                        output.get("text", "") if isinstance(output, dict)
                        else getattr(output, "text", "") or ""
                    )
                is_tool_error = output_text.lstrip().lower().startswith("error:")

                if is_tool_error:
                    self.reasoning_tree.expand(
                        self.selected_node, operator, tool_call["input"], [],
                        error=True, error_msg=output_text.strip(),
                    )
                    new_child = self.reasoning_tree.nodes[-1]
                    logger.info(
                        "%s EXEC op=%s input=%s → ERROR child=%s msg=%s",
                        self._tag(), operator, _trim_for_log(tool_call.get("input")),
                        new_child.id, _trim_for_log(output_text.strip(), 120),
                    )
                elif operator == "end_search":
                    self.reasoning_tree.expand(
                        self.selected_node, "end_search", None,
                        self.selected_node.input_tables, valid=True,
                    )
                    new_child = self.reasoning_tree.nodes[-1]
                    logger.info(
                        "%s EXEC op=end_search → child=%s tables=%d (terminate)",
                        self._tag(), new_child.id, len(self.selected_node.input_tables),
                    )
                else:
                    input_param = tool_call["input"]
                    tables = tool_ctx.get_last_result()
                    self.reasoning_tree.expand(
                        self.selected_node, operator, input_param, tables,
                        valid=False,
                    )
                    new_child = self.reasoning_tree.nodes[-1]
                    logger.info(
                        "%s EXEC op=%s input=%s → child=%s tables=%d unresolved_constraints=%d",
                        self._tag(), operator, _trim_for_log(input_param),
                        new_child.id, len(tables), new_child.constraint_cnt,
                    )
            except Exception as e:
                self.reasoning_tree.expand(
                    self.selected_node,
                    tool_call.get('name', 'unknown'),
                    tool_call.get('input', {}),
                    [],
                    error=True,
                    error_msg=f"{type(e).__name__}: {e}",
                )
                logger.exception(
                    "%s EXEC op=%s raised exception",
                    self._tag(), tool_call.get('name', 'unknown'),
                )
        return None


    async def tool_use(self, tool_call: ToolUseBlock):
        tool_res = await self.toolkit.call_tool_function(tool_call)
        async for chunk in tool_res:
            return chunk.content
        return None


    def check(self, msg_reasoning):
        for content in msg_reasoning.content:
            btype = content.get('type') if isinstance(content, dict) else getattr(content, 'type', None)
            if btype == 'tool_use':
                return True
        return False
