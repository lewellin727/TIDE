import contextvars
from typing import List, Literal, Annotated
from pydantic import Field
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from src.table import Table
from src.operators.seeker import ContentSeeker, ContextSeeker
from src.operators.ranker import CategoricalRanker, NumericalRanker


# ==== tool execution context (set by the agent before each tool call) ====
#
# Per-call output-cap cascade: VDB search (512) -> seekers (top_k) -> rankers (top_k=50)
# -> _cap (config.tools.result_cap) -> ReasoningTree.get_cand_tables (config.tree.cand_top_k).
#
# Rankers mutate `t.score` in place, but every call starts from a fresh Table list
# (names2table reads CSV; get_navi_tables wraps the cached df anew), so two nodes never
# share a Table instance and an in-place mutation cannot corrupt another node.
#
# The per-call state below is held in ContextVars so concurrent query tasks stay isolated;
# `_result_cap` is set once at startup and read-only afterwards.

_reasoning_tree_var: "contextvars.ContextVar" = contextvars.ContextVar(
    "tide_reasoning_tree", default=None
)
_current_node_id_var: "contextvars.ContextVar" = contextvars.ContextVar(
    "tide_current_node_id", default=None
)
_last_tool_result_var: "contextvars.ContextVar" = contextvars.ContextVar(
    "tide_last_tool_result", default=None
)

_result_cap: int = 20  # process-global startup config (read-only after init)


def set_reasoning_tree(reasoning_tree) -> None:
    _reasoning_tree_var.set(reasoning_tree)


def set_current_node(node_id) -> None:
    _current_node_id_var.set(str(node_id) if node_id is not None else None)
    _last_tool_result_var.set([])  # fresh slate for each tool call


def set_result_cap(cap: int) -> None:
    """Set the per-tool-call output cap (config.tools.result_cap), once at startup."""
    global _result_cap
    _result_cap = max(1, int(cap))


def get_last_result() -> List[Table]:
    return _last_tool_result_var.get() or []


def _stash(tables) -> List[Table]:
    out = list(tables) if tables else []
    _last_tool_result_var.set(out)
    return out


def _fetch_node_tables(node_id: str, in_trajectory: bool = True) -> List[Table]:
    """Return input_tables of a referenced node.

    `in_trajectory=True` (default, used by seekers/rankers): restrict to
    `ancestors + current` so a single trajectory's src_node_id stays linear.

    `in_trajectory=False` (used by combiners): allow any non-error node anywhere
    in the tree, so combiners can compose intermediate candidates across
    iterations (including sibling trajectories); restricting them to
    ancestors+current would make intersection/union/difference useless.
    """
    tree = _reasoning_tree_var.get()
    current_id = _current_node_id_var.get()
    if tree is None:
        raise RuntimeError("tool.set_reasoning_tree was never called")
    if current_id is None:
        raise RuntimeError("tool.set_current_node was never called")
    node_id = str(node_id)
    node = tree.id2node(node_id)
    if node is None:
        raise ValueError(f"Node '{node_id}' does not exist in search tree")
    if node.error:
        raise ValueError(f"Node '{node_id}' is an error node — its tables are invalid")
    if in_trajectory:
        valid_ids = {current_id} | set(tree.all_parent_dict.get(current_id, []))
        if node_id not in valid_ids:
            raise ValueError(
                f"Node id '{node_id}' is not on the current trajectory from node "
                f"'{current_id}'. Allowed ids (ancestors + current): "
                f"{sorted(valid_ids)}. For cross-trajectory composition, use a combiner "
                f"(intersection/union/difference), which accepts any non-error node id."
            )
    return list(node.input_tables)


def _summarize(operator: str, tables) -> str:
    n = len(tables) if tables else 0
    return f"{operator} completed. {n} tables produced."


def _merge_by_name(tables_groups, keep: Literal["max", "first", "left"] = "max") -> List[Table]:
    merged = {}
    for group in tables_groups:
        for t in group:
            existing = merged.get(t.table_name)
            if existing is None:
                merged[t.table_name] = t
            elif keep == "max" and t.score > existing.score:
                merged[t.table_name] = t
    return list(merged.values())


def _cap(tables) -> List[Table]:
    return list(tables)[:_result_cap] if tables else []


def _coerce_optional_float(x):
    """Coerce LLM-emitted numerical bounds back to float | None.

    LLMs often serialize `None` as the literal string `"None"` or `"null"` and
    numbers as `"41.0"` strings. Without coercion downstream numpy comparisons
    raise ufunc dtype errors. Returns None for None / blank / "none" / "null",
    else float(x). Raises ValueError on truly unparseable strings.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ("", "none", "null"):
            return None
        return float(t)
    return float(x)


# ==== seekers ====

def column_seeker(src_node_id: str, column_names: List[str]) -> ToolResponse:
    """Find tables that share cell values with the source node's `column_names`."""
    tables = _fetch_node_tables(src_node_id)
    result = ContentSeeker.search(tables, column_names) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("column_seeker", result))])


def metadata_seeker(column_names: List[str], column_types: List[str]) -> ToolResponse:
    """Open a NEW search direction: find lake tables whose schema has `column_names` (of `column_types`). No source node."""
    result = ContextSeeker.search(column_names, column_types, top_k=_result_cap) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("metadata_seeker", result))])


# ==== rankers ====

def eq_ranker(src_node_id: str, column_name: str, column_value: str) -> ToolResponse:
    """Exact-equality value filter: rank the source node's tables by how often `column_value` (a CELL value, not a column name) appears in `column_name`."""
    tables = _fetch_node_tables(src_node_id)
    result = CategoricalRanker.rank_frequence(tables, column_name, column_value, top_k=_result_cap) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("eq_ranker", result))])


def overlap_ranker(src_node_id: str, cand_node_id: str, join_column_name: str) -> ToolResponse:
    """Score the OVERLAP predicate: Jaccard similarity between `cand_node_id`'s `join_column_name` value set and `src_node_id`'s value set on the same column."""
    tables_1 = _fetch_node_tables(src_node_id)
    tables_2 = _fetch_node_tables(cand_node_id)
    result = CategoricalRanker.rank_overlap(tables_1, tables_2, join_column_name) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("overlap_ranker", result))])


def corr_ranker(src_node_id: str, cand_node_id: str, join_column_name: str,
                                    src_corr_column_name: str, cand_corr_column_name: str) -> ToolResponse:
    """Score the CORRELATION predicate: |Pearson| between `cand_node_id`'s `cand_corr_column_name` and `src_node_id`'s `src_corr_column_name`, over the `join_column_name` join."""
    tables_1 = _fetch_node_tables(src_node_id)
    tables_2 = _fetch_node_tables(cand_node_id)
    result = CategoricalRanker.rank_correlation(
        tables_1, tables_2, join_column_name, src_corr_column_name, cand_corr_column_name
    ) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("corr_ranker", result))])


def range_ranker(src_node_id: str, column_name: str, min_val: float | None, max_val: float | None) -> ToolResponse:
    """Numeric range value filter: rank the source node's tables by how many `column_name` cells fall in [min_val, max_val] (null = open bound)."""
    tables = _fetch_node_tables(src_node_id)
    try:
        lo = _coerce_optional_float(min_val)
        hi = _coerce_optional_float(max_val)
    except (TypeError, ValueError) as e:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Error: range_ranker `min_val` / `max_val` must be a number or null. "
                 f"Got min_val={min_val!r}, max_val={max_val!r}. ({e})"
        )])
    if lo is None and hi is None:
        return ToolResponse(content=[TextBlock(
            type="text",
            text="Error: range_ranker with [null, null] carries no actual filter — "
                 "use `union_search` (with src_node_id) or `metadata_seeker` for column-existence checks."
        )])
    if lo is not None and hi is not None and lo == hi:
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"Error: range_ranker degenerate range [{lo}, {hi}] (min == max). "
                 f"For exact equality use `eq_ranker` with `column_value=\"{lo}\"`, "
                 f"or widen the bounds."
        )])
    result = NumericalRanker.rank_range(tables, column_name, lo, hi, top_k=_result_cap) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("range_ranker", result))])


# ==== composite search (relationship-level convenience over the atomic seeker+ranker basis) ====

def join_search(src_node_id: str, join_column_name: str) -> ToolResponse:
    """Find tables joinable to the source node's tables on `join_column_name` (value overlap). Use when correlation isn't also needed — `correlation_search` already joins internally."""
    tables_1 = _fetch_node_tables(src_node_id)
    tables_2 = ContentSeeker.search(tables_1, [join_column_name], top_k=512) or []
    result = CategoricalRanker.rank_overlap(tables_1, tables_2, join_column_name, top_k=_result_cap) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("join_search", result))])


def union_search(src_node_id: str, union_column_names: List[str]) -> ToolResponse:
    """Find tables sharing `union_column_names` with the source node's tables (unionable schema)."""
    tables_1 = _fetch_node_tables(src_node_id)
    result = ContentSeeker.search(tables_1, union_column_names, top_k=_result_cap) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("union_search", result))])


def correlation_search(src_node_id: str, join_column_name: str, src_corr_column_name: str) -> ToolResponse:
    """Find tables that correlate on `src_corr_column_name` (numeric) after joining the source node's tables on `join_column_name`. Joins internally — don't pre-call join_search on the same key."""
    tables_1 = _fetch_node_tables(src_node_id)
    tables_2 = ContentSeeker.search(tables_1, [join_column_name], top_k=512) or []
    tables_2 = CategoricalRanker.rank_overlap(tables_1, tables_2, join_column_name, top_k=512) or []
    result = CategoricalRanker.rank_correlation(tables_1, tables_2, join_column_name, src_corr_column_name, None, top_k=_result_cap) or []
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize("correlation_search", result))])


# ==== combiners (set operations over two nodes' input_tables, keyed by table_name) ====

def combine(node_id_1: Annotated[str, Field(description="Any non-error node id (base set for 'difference').")],
            node_id_2: Annotated[str, Field(description="Any non-error node id (excluded set for 'difference').")],
            set_op: Annotated[str, Field(description="'intersection' (in BOTH), 'union' (in EITHER), or 'difference' (in node_id_1 but NOT node_id_2).")]) -> ToolResponse:
    """Set-combine two nodes' candidate tables (by name) via `set_op`, across any two non-error nodes (siblings included)."""
    t1 = _fetch_node_tables(node_id_1, in_trajectory=False)
    t2 = _fetch_node_tables(node_id_2, in_trajectory=False)
    op = (set_op or "intersection").strip().lower()
    if op == "union":
        result = _merge_by_name([t1, t2], keep="max")
    elif op == "difference":
        exclude = {t.table_name for t in t2}
        result = [t for t in t1 if t.table_name not in exclude]
    else:  # intersection (default)
        common = {t.table_name for t in t1} & {t.table_name for t in t2}
        result = _merge_by_name(
            [[t for t in t1 if t.table_name in common], [t for t in t2 if t.table_name in common]], keep="max")
    result = _cap(result)
    _stash(result)
    return ToolResponse(content=[TextBlock(type="text", text=_summarize(f"combine[{op}]", result))])


def end_search() -> ToolResponse:
    """Lock in the current node's tables as a RESULT and finish this branch. Call it as soon as the
    candidates answer the query — you do NOT need every constraint resolved. Only an ended branch is
    returned as an answer, so this call is REQUIRED to produce output. No arguments."""
    return ToolResponse(
        content=[TextBlock(type="text", text="<End></End>")]
    )
