import json
from typing import List, Optional
from src.table import Table
from src.utils import PLANNING_TEMPLATE, OBSERVATION_STAGE_TEMPLATE, render_evidence, COL_CONSTRAINTS_SATISFIED_HINT
from src.query_constraint import QueryConstraint
from src.col_name_matcher import ColNameMatcher

_CATEGORICAL_VALUE_CAP = 20 

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

    if not paths:
        return ""
    blocks = []
    for i, p in enumerate(paths):
        header = f"### [p{i + 1}]" if with_ids else "###"
        blocks.append(f"{header}\n{p}")
    return "\n\n".join(blocks)


class ReasoningTree:
    def __init__(self, query_table, init_constraints, lake_dir, cand_top_k: int = 10,
                 final_top_k: int = 20, final_min_constraint_match: int = 1):
        self.curr_id = 0
        self.nodes = []
        self.adj_dict = {}       
        self.parent_dict = {}     
        self.all_parent_dict = {} 
        self._id2node = {}
        self.lake_dir = lake_dir
        self.cand_top_k = max(1, int(cand_top_k))  

        self.final_top_k = max(1, int(final_top_k))
        self.final_min_constraint_match = max(0, int(final_min_constraint_match))

        self.init_node(query_table, init_constraints)

    def init_node(self, query_table, init_constraints):
        node = ReasoningNode(str(self.curr_id))
        node.input_constraints = init_constraints
        node.col_constraints = init_constraints
        node.calc_constraint_cnt()
        node.input_tables = [query_table] if query_table else []

        self.nodes.append(node)
        self._id2node[node.id] = node
        self.adj_dict[node.id] = []
        self.parent_dict[node.id] = None
        self.all_parent_dict[node.id] = []

        self.initial_constraint_cnt = node.constraint_cnt

        self.curr_id += 1

    def depth_of(self, node) -> int:
        return len(self.all_parent_dict.get(node.id, []))

    def utility(self, node) -> float:
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

        root = self._id2node.get("0")
        return root.input_constraints if root else []

    def _filter_and_rank(self, tables: List[Table]) -> List[Table]:

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
        return self._filter_and_rank(self._collect(lambda node: not node.error))

    def _collect(self, leaf_ok) -> List[Table]:
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
        cur = leaf
        while cur is not None and not cur.input_tables:
            pid = self.parent_dict.get(cur.id)
            if pid is None:
                return None  
            cur = self.id2node(pid)
        if cur is None or cur.error:
            return None
        if self.parent_dict.get(cur.id) is None:
            return None  
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
            child_node.terminated = True   
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


    _OP_SINGLE_COLS = {
        "eq_ranker": ("column_name",), "range_ranker": ("column_name",),
        "join_search": ("join_column_name",),
        "correlation_search": ("join_column_name", "src_corr_column_name"),
    }
    _VALUE_RESOLVERS = {"eq_ranker", "range_ranker"}  
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

        if node.error:
            return
        targeted = self._targeted_columns(operator, input_param)
        tables = node.input_tables or []
        top = sorted(tables, key=lambda t: getattr(t, "score", 0.0), reverse=True)[: self._CONF_TOP_K]
        remaining = []
        for con in (node.input_constraints or []):
            con = {k: v for k, v in con.items() if k != "coverage"}
            if con.get("values"):
                hit = (operator in self._VALUE_RESOLVERS and bool(targeted)
                       and ColNameMatcher.find_match(con["col_name"], targeted) is not None)
                if hit and top and any(QueryConstraint._satisfies(con, t, self._CONF_NUM_THRESHOLD) for t in top):
                    continue
            else:
                if top and any(QueryConstraint._satisfies(con, t) for t in top):
                    continue
            remaining.append(con)
        node.col_constraints = remaining
        node.calc_constraint_cnt()


    def get_history(self, child_id):
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

        src_tables = _render_tables(list(self.get_cand_tables(selected_node.id)))

        col_constraints = _render_constraints(selected_node.col_constraints, with_ids=False)

        path_ops = self.get_history(selected_node.id)

        ops_lines = [
            f"{idx + 1}. [node {path['child_id']}] {path['operator']} "
            f"| Input: {json.dumps(path['input'])}"
            for idx, path in enumerate(reversed(path_ops))
        ]
        ops_history = "\n".join(ops_lines)

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
                + "\n".join(failed_lines[-10:]) 
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


    def __init__(self, id):
        self.id = str(id)
        self.end = False          
        self.terminated = False    
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


