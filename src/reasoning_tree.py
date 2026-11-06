import json
from copy import deepcopy
from typing import List, Optional
from src.table import Table
from src.utils import PLANNING_TEMPLATE, OBSERVATION_STAGE_TEMPLATE, render_evidence, COL_CONSTRAINTS_SATISFIED_HINT
from src.query_constraint import QueryConstraint
from src.col_name_matcher import ColNameMatcher


# Shared renderers for prompt text.

_CATEGORICAL_VALUE_CAP = 20  # Limit displayed categorical values.


def _render_constraint(con: dict, cid: Optional[str] = None) -> str:
    prefix = f"[{cid}] " if cid else ""
    col_name = con["col_name"]
    if con["col_type"] == "Categorical":
        if con["values"]:
            vals = [str(v) for v in con["values"]]
            val_str = ", ".join(vals[:_CATEGORICAL_VALUE_CAP])
            if len(vals) > _CATEGORICAL_VALUE_CAP:
                val_str += ", ..."
            return f"- {prefix}Column: '{col_name}' | Type: Categorical | Value Domain: {{{val_str}}}"
        return f"- {prefix}Column: '{col_name}' | Type: Categorical"
    elif con["values"]:
        min_val, max_val = con["values"][0], con["values"][1]
        return f"- {prefix}Column: '{col_name}' | Type: Numerical | Interval: [{min_val}, {max_val}]"
    return f"- {prefix}Column: '{col_name}' | Type: Numerical"


def _render_constraints(constraints, with_ids: bool = False) -> str:
    return "\n".join(
        _render_constraint(c, cid=f"c{i + 1}" if with_ids else None)
        for i, c in enumerate(constraints)
    )


def _render_table_line(table: Table) -> str:
    return f"- Table Name: {table.table_name} | Schema: {table.to_schema()}"


def _render_tables(tables) -> str:
    return "\n".join(_render_table_line(t) for t in tables)


def _render_paths(paths, with_ids: bool = False) -> str:
    """Render observation paths. Each path is now a multi-line block from
    `render_path_with_schema`, so we prefix the id on its own header line.
    """
    if not paths:
        return ""
    blocks = []
    for i, p in enumerate(paths):
        header = f"### [p{i + 1}]" if with_ids else "###"
        blocks.append(f"{header}\n{p}")
    return "\n\n".join(blocks)


class ReasoningTree:
    def __init__(self, query_table, init_constraints, lake_dir, cand_top_k: int = 10,
                 final_top_k: int = 20, final_min_constraint_match: int = 1,
                 enable_constraint_revision: bool = False):
        self.enable_constraint_revision = enable_constraint_revision
        self.curr_id = 0
        self.nodes = []
        self.adj_dict = {}         # Parent id -> child ids and operators.
        self.parent_dict = {}      # Node id -> direct parent id.
        self.all_parent_dict = {}  # Node id -> ancestor ids.
        self._id2node = {}
        self.lake_dir = lake_dir
        self.cand_top_k = max(1, int(cand_top_k))

        # Filter final candidates by the number of satisfied root constraints.
        # Retain final_top_k for API compatibility; return the full ranked list for evaluation cutoffs.
        self.final_top_k = max(1, int(final_top_k))
        self.final_min_constraint_match = max(0, int(final_min_constraint_match))

        self.init_node(query_table, init_constraints)

    def init_node(self, query_table, init_constraints):
        node = ReasoningNode(str(self.curr_id))
        node.input_constraints = init_constraints
        if self.enable_constraint_revision:
            node.full_constraints = deepcopy(init_constraints)
        node.col_constraints = init_constraints
        node.calc_constraint_cnt()
        node.input_tables = [query_table] if query_table else []

        self.nodes.append(node)
        self._id2node[node.id] = node
        self.adj_dict[node.id] = []
        self.parent_dict[node.id] = None
        self.all_parent_dict[node.id] = []

        # Snapshot the initial constraint count so utility does not depend on later root changes.
        self.initial_constraint_cnt = node.constraint_cnt

        self.curr_id += 1

    def depth_of(self, node) -> int:
        return len(self.all_parent_dict.get(node.id, []))

    def utility(self, node) -> float:
        """Per-leaf efficiency: U(v) = (|s_q^0| - |v.s_q|) / depth(v).

        Quantifies "progress per step": a trajectory that resolves constraints
        quickly stays attractive; a trajectory that keeps extending without
        reducing s_q sees its utility decay each step, naturally yielding
        breadth back to sibling branches. Root depth is 0 — we return +inf so
        the only-root case picks the root.

        Paper Algorithm 1 line 6 says "prioritize trajectories with the most
        progress on q"; this is a stricter (efficiency-aware) form of that.
        """
        d = self.depth_of(node)
        if d == 0:
            return float('inf')
        return (self.initial_constraint_cnt - node.constraint_cnt) / d

    def get_activate_leafs(self):
        leafs = []
        for node in self.nodes:
            if self.adj_dict[node.id] == [] and not node.end:
                leafs.append(node)
        return leafs

    def _root_constraints(self):
        """Original (unpruned) constraint list parsed from the user query at root.

        Used by the per-table output filter so the precision check applies the
        FULL query, not whatever residual was left on a particular leaf after
        loop-time pruning.
        """
        root = self._id2node.get("0")
        return root.input_constraints if root else []

    def _filter_and_rank(self, tables: List[Table]) -> List[Table]:
        """Per-table precision filter, then rank by CONSTRAINT-RELEVANCE first, score second. Returns
        the FULL ranked list (NOT capped): we persist every retrieved candidate in this order so the
        eval pipeline can slice it to any K (P@10, R@20, R@30, …) without re-running.

        Resolution is NON-destructive, so a frontier's tables still include look-alikes a seeker
        fetched but no ranker verified. We score each table by HOW MANY of the query's ORIGINAL
        constraints it satisfies (value constraints checked by VALUE via `_satisfies`):
          - filter: drop tables satisfying < `final_min_constraint_match` (pure noise);
          - rank: sort by (#constraints satisfied DESC, search score DESC). Because we now merge EVERY
            leaf's frontier for coverage, a single union/join hop can dump many raw look-alikes; ranking
            by relevance floats the true tables (which satisfy their hop's value constraint) above that
            raw-expansion noise so they land inside the eval cutoff instead of being buried.
        If the filter would wipe every table, fall back to the unfiltered set (never return empty).
        """
        if not tables:
            return []
        root_cons = self._root_constraints()
        sat = {t.table_name: sum(1 for c in root_cons if QueryConstraint._satisfies(c, t))
               for t in tables} if root_cons else {}
        need = self.final_min_constraint_match
        if root_cons and need > 0:
            kept = [t for t in tables if sat.get(t.table_name, 0) >= need]
            if kept:
                tables = kept
        return sorted(tables, key=lambda t: (sat.get(t.table_name, 0), t.score), reverse=True)

    def get_results(self) -> List[Table]:
        # Merge all non-error leaf frontiers, including unfinished branches, then filter and rank.
        # Return the full candidate list; evaluation applies its own cutoff.
        if self.enable_constraint_revision:
            return self._get_revised_results()
        return self._filter_and_rank(self._collect(lambda node: not node.error))

    def _get_revised_results(self) -> List[Table]:
        """Rank each occurrence using its branch's complete, revised interpretation."""
        best = {}
        for leaf in self.nodes:
            if self.adj_dict[leaf.id] or leaf.error:
                continue
            frontier = self._frontier_of(leaf)
            if frontier is None:
                continue
            constraints = leaf.full_constraints
            fmax = max((t.score for t in frontier.input_tables), default=0.0) or 1.0
            for t in frontier.input_tables:
                count = sum(QueryConstraint._satisfies(c, t) for c in constraints)
                score = t.score / fmax
                eligible = not constraints or count >= self.final_min_constraint_match
                rank = (eligible, count, score)
                old = best.get(t.table_name)
                if old is None or rank > old[0]:
                    best[t.table_name] = (rank, Table(t.table_name, t.df, score))
        rows = list(best.values())
        eligible = [r for r in rows if r[0][0]]
        rows = eligible or rows  # Preserve the existing all-empty filter fallback.
        rows.sort(key=lambda r: r[0][1:], reverse=True)
        return [table for _, table in rows]

    def _collect(self, leaf_ok) -> List[Table]:
        """MAX-merge the frontiers of every leaf for which `leaf_ok(node)` is True."""
        score_dict = {}
        proto_dict = {}
        seen = set()
        for node in self.nodes:
            if self.adj_dict[node.id]:
                continue
            if not leaf_ok(node):
                continue
            frontier = self._frontier_of(node)
            if frontier is None or frontier.id in seen:
                continue
            seen.add(frontier.id)
            # Normalize each frontier by its maximum score before merging; node-normalized scores have different scales.
            fmax = max((t.score for t in frontier.input_tables), default=0.0) or 1.0
            for table in frontier.input_tables:
                norm = table.score / fmax
                if table.table_name not in proto_dict:
                    proto_dict[table.table_name] = table
                    score_dict[table.table_name] = norm
                else:
                    score_dict[table.table_name] = max(score_dict[table.table_name], norm)
        return [Table(name, proto_dict[name].df, score_dict[name]) for name in proto_dict]

    def _frontier_of(self, leaf):
        """The usable frontier of a trajectory: the leaf if it carries tables,
        else the nearest non-empty ancestor. This rescues a trajectory whose
        last step filtered to ZERO tables (e.g. a ranker / relationship search
        that over-filtered, or correlation_search collapsing to 0) — instead of
        discarding the whole branch we fall back to the best candidates it held
        just before the collapse. Returns None if the whole path up to (and
        including) the leaf is empty, or only the root remains, or it hits an
        error node.
        """
        cur = leaf
        while cur is not None and not cur.input_tables:
            pid = self.parent_dict.get(cur.id)
            if pid is None:
                return None
            cur = self.id2node(pid)
        if cur is None or cur.error:
            return None
        if self.parent_dict.get(cur.id) is None:
            return None  # The root contains the query seed, not a discovered answer.
        return cur

    def get_cand_tables(self, node_id) -> List[Table]:
        node = self.id2node(node_id)
        if node is None:
            return []
        if node._cand_cache is not None:
            return node._cand_cache
        table_dict = {}
        for table in node.input_tables:
            if table.table_name not in table_dict:
                table_dict[table.table_name] = table
            else:
                if table.score > table_dict[table.table_name].score:
                    table_dict[table.table_name] = table
        result_tables = sorted(table_dict.values(), key=lambda t: t.score, reverse=True)
        node._cand_cache = result_tables[:self.cand_top_k] if result_tables else []
        return node._cand_cache


    def expand(self, selected_node, operator, input_param, tables, valid=False, error=False, error_msg=""):
        child_node = ReasoningNode(str(self.curr_id))
        self.curr_id += 1
        child_node.input_constraints = selected_node.col_constraints
        if self.enable_constraint_revision:
            child_node.input_constraints = deepcopy(selected_node.col_constraints)
            child_node.full_constraints = deepcopy(selected_node.full_constraints)
        child_node.input_tables = tables
        child_node.error_msg = error_msg
        self.nodes.append(child_node)
        self._id2node[child_node.id] = child_node
        self.adj_dict[child_node.id] = []
        self.parent_dict[child_node.id] = selected_node.id
        self.all_parent_dict[child_node.id] = self.all_parent_dict[selected_node.id] + [selected_node.id]

        if input_param is None:
            input_param = {}
        op = {'operator': operator, "input": input_param}
        self.adj_dict[selected_node.id].append({
            "child": child_node.id,
            "op": op
        })

        if error:
            child_node.end = True
            child_node.valid = False
            child_node.error = True
            return

        path_ops = self.get_history(selected_node.id)
        for path in path_ops:
            if path['operator'] == operator and path['input'] == input_param:
                child_node.end = True
                child_node.valid = False
                self._resolve_constraints(child_node, operator, input_param)
                return
        if operator == "end_search":
            child_node.end = True
            child_node.terminated = True   # Explicit termination preserves a valid result.
        elif tables == []:
            child_node.end = True

        child_node.valid = valid
        self._resolve_constraints(child_node, operator, input_param)

    def _op_sig(self, operator, input_param):
        try:
            return (operator, json.dumps(input_param, sort_keys=True, default=str))
        except Exception:
            return (operator, str(input_param))

    def already_tried(self, operator, input_param) -> bool:
        """True if this exact (operator, input) was already executed at any non-error node
        anywhere in the tree — tree-wide, so the planner can't re-run it by re-selecting a
        different node. (`end_search` is exempt — the caller should not pass it here.)"""
        sig = self._op_sig(operator, input_param)
        for edges in self.adj_dict.values():
            for edge in edges:
                op = edge.get("op", {})
                child = self.id2node(edge.get("child"))
                if child is not None and child.error:
                    continue
                if self._op_sig(op.get("operator"), op.get("input")) == sig:
                    return True
        return False

    # Map operators to the query columns they target.
    _OP_SINGLE_COLS = {
        "eq_ranker": ("column_name",), "range_ranker": ("column_name",),
        "join_search": ("join_column_name",),
        "correlation_search": ("join_column_name", "src_corr_column_name"),
    }
    _VALUE_RESOLVERS = {"eq_ranker", "range_ranker"}  # Only value rankers may resolve value constraints.
    # Verify constraints against the highest-scored outputs; numerical checks use the required row fraction.
    _CONF_TOP_K = 3
    _CONF_NUM_THRESHOLD = 0.5

    @staticmethod
    def _targeted_columns(operator, input_param):
        if not isinstance(input_param, dict):
            return []
        if operator == "union_search":
            return [c for c in (input_param.get("union_column_names") or []) if c]
        if operator == "metadata_seeker":
            return [c for c in (input_param.get("column_names") or []) if c]
        cols = []
        for k in ReasoningTree._OP_SINGLE_COLS.get(operator, ()):
            v = input_param.get(k)
            if v:
                cols.append(v)
        return cols

    def _resolve_constraints(self, node, operator, input_param):
        """Operator-bound, NON-destructive constraint resolution (replaces the global prune).

        A constraint is resolved ONLY by an operator that explicitly TARGETED its column AND
        whose output tables satisfy it:
          - VALUE constraints (eq/range) resolve only via a value ranker (eq_ranker/range_ranker);
            a relationship/seeker that merely carries the column does NOT resolve the value
            (this stops a single broad fetch from "completing" a chain it never traversed);
          - PRESENCE constraints (column existence) resolve via any operator targeting the column.
        input_tables are NOT pruned here (non-destructive): a no-progress operator simply leaves
        its constraints unresolved (→ low utility), without wiping the pool or sibling branches.
        Error nodes are skipped.
        """
        if node.error:
            return
        targeted = self._targeted_columns(operator, input_param)
        tables = node.input_tables or []
        top = sorted(tables, key=lambda t: getattr(t, "score", 0.0), reverse=True)[: self._CONF_TOP_K]
        remaining = []
        for con in (node.input_constraints or []):
            con = {k: v for k, v in con.items() if k != "coverage"}
            if con.get("values"):
                # Resolve value constraints only through a ranker targeting the matching column.
                hit = (operator in self._VALUE_RESOLVERS and bool(targeted)
                       and ColNameMatcher.find_match(con["col_name"], targeted) is not None)
                if hit and top and any(QueryConstraint._satisfies(con, t, self._CONF_NUM_THRESHOLD) for t in top):
                    continue
            else:
                # Resolve presence constraints when any top output contains the requested column.
                if top and any(QueryConstraint._satisfies(con, t) for t in top):
                    continue
            remaining.append(con)
        node.col_constraints = remaining
        node.calc_constraint_cnt()


    def get_history(self, child_id):
        """Walk from child_id's parent up to root. Each entry carries the child_id it produced."""
        path_ops = []
        current_id = child_id
        parent_node = self.id2node(self.parent_dict[current_id])
        while parent_node:
            for item in self.adj_dict[parent_node.id]:
                if item["child"] == current_id:
                    path_ops.append({
                        "child_id": current_id,
                        "operator": item["op"]["operator"],
                        "input": item["op"]["input"],
                    })
                    break
            current_id = parent_node.id
            parent_node = self.id2node(self.parent_dict[current_id])
        return path_ops

    def gen_observation_stage_prompt(self, selected_node, query=""):
        """Build the user prompt for the Observation Stage LLM.

        Inputs: the user Query (for intent judgement), Query State, and the ranked
        Observation Paths, with stable ids prefixed onto each entry.
        """
        col_constraints = _render_constraints(
            selected_node.col_constraints, with_ids=True
        )
        paths = _render_paths(selected_node.paths, with_ids=True)
        return OBSERVATION_STAGE_TEMPLATE.format(
            query=query if query else "(not provided)",
            col_constraints=col_constraints if col_constraints else "None.",
            paths=paths if paths else "None.",
        )

    def gen_planning_prompt(self, query, selected_node, observation_evidence):
        # Show only held tables as source tables; reachable neighbors belong in observation evidence.
        src_tables = _render_tables(list(self.get_cand_tables(selected_node.id)))

        col_constraints = _render_constraints(selected_node.col_constraints, with_ids=False)

        path_ops = self.get_history(selected_node.id)

        ops_lines = [
            f"{idx + 1}. [node {path['child_id']}] {path['operator']} "
            f"| Input: {json.dumps(path['input'])}"
            for idx, path in enumerate(reversed(path_ops))
        ]
        ops_history = "\n".join(ops_lines)

        # Include successful calls from other branches, deduplicated by operator and input.
        # Current-trajectory calls and failures are shown separately.
        path_child_ids = {p["child_id"] for p in path_ops}
        tree_op_seen = set()
        tree_op_lines = []
        for n in self.nodes:
            if n.error:
                continue
            if n.id in path_child_ids:
                continue
            parent_id = self.parent_dict.get(n.id)
            if parent_id is None:
                continue
            for edge in self.adj_dict.get(parent_id, []):
                if edge["child"] != n.id:
                    continue
                op = edge["op"]
                try:
                    sig = (op["operator"], json.dumps(op.get("input"), sort_keys=True, default=str))
                except Exception:
                    sig = (op["operator"], str(op.get("input")))
                if sig in tree_op_seen:
                    continue
                tree_op_seen.add(sig)
                tree_op_lines.append(
                    f"- [node {n.id}] {op['operator']} | Input: {json.dumps(op['input'])}"
                )
                break
        if tree_op_lines:
            ops_history = (
                ops_history
                + "\n\n### Tree-Wide Operations Already Tried (do NOT repeat any of these (operator, input) pairs)\n"
                + "\n".join(tree_op_lines[-20:])
            )

        # Include failed calls across the tree so the planner can correct invalid arguments.
        failed_lines = []
        for n in self.nodes:
            if not n.error or not n.error_msg:
                continue
            parent_id = self.parent_dict.get(n.id)
            if parent_id is None:
                continue
            for edge in self.adj_dict.get(parent_id, []):
                if edge["child"] != n.id:
                    continue
                op = edge["op"]
                msg = n.error_msg.replace("\n", " ").strip()
                if len(msg) > 200:
                    msg = msg[:200] + "..."
                failed_lines.append(
                    f"- {op['operator']}({json.dumps(op['input'])}) → {msg}"
                )
                break
        if failed_lines:
            ops_history = (
                ops_history
                + "\n\n### Failed Tool Calls (do not retry these inputs)\n"
                + "\n".join(failed_lines[-10:])  # Limit failure history to bound prompt size.
            )

        valid_ids = [selected_node.id] + list(self.all_parent_dict.get(selected_node.id, []))
        valid_ids_str = ", ".join(valid_ids)

        observation = PLANNING_TEMPLATE.format(
            query=query,
            src_tables=src_tables if src_tables else "None.",
            current_node_id=selected_node.id,
            valid_node_ids=valid_ids_str,
            ops_history=ops_history if ops_history else "None.",
            col_constraints=col_constraints if col_constraints else COL_CONSTRAINTS_SATISFIED_HINT,
            observation_evidence=render_evidence(observation_evidence),
        )

        return observation


    def id2node(self, node_id):
        if node_id is None:
            return None
        return self._id2node.get(str(node_id))


class ReasoningNode:
    """Node v ∈ V_T of the reasoning tree (paper Sec 4.1).

    State tuple v = (s_t, s_q, s_o, s_p):
      - s_t (input_tables): current candidate tables
      - s_q (col_constraints): unresolved query constraints
      - s_o: lake-grounded evidence surfaced this turn (navi_tables + paths)
      - s_p: planner-emitted operations for this turn

    Descriptive Python names remain primary; the s_* properties expose the
    paper symbols for readability cross-checks.
    """

    def __init__(self, id):
        self.id = str(id)
        self.end = False           # Terminal after explicit termination or a dead end.
        self.terminated = False    # Explicit end_search, distinct from a dead end.
        self.valid = False
        self.error = False
        self.input_constraints = []
        self.input_tables = []

        self.constraint_cnt = 0
        self.col_constraints = []
        self.navi_tables = []
        self.paths = []
        self.error_msg = ""
        self.s_p = []
        self._cand_cache = None

    @property
    def s_t(self):
        return self.input_tables

    @property
    def s_q(self):
        return self.col_constraints

    @property
    def s_o(self):
        return {"navi_tables": self.navi_tables, "paths": self.paths}

    def calc_constraint_cnt(self):
        self.constraint_cnt = QueryConstraint.count(self.col_constraints)

    def set_navi(self, navi_tables, paths):
        self.navi_tables = list(navi_tables)
        self.paths = paths
