import os
import contextvars
from openai import OpenAI
from src.col_filter import ColFilter
from src.col_name_matcher import ColNameMatcher
from src.operators.seeker import ContentSeeker, ContextSeeker
from src.operators.ranker import CategoricalRanker, NumericalRanker

def init_operators(vdb_searcher, col_match_threshold: float = 0.95, embedding_model=None,
                   ranker_cfg: dict | None = None):

    ContentSeeker.vdb_searcher = vdb_searcher
    ContextSeeker.vdb_searcher = vdb_searcher

    filter_ins_dict = {}
    for table_name, col_names in vdb_searcher.filter_dict.items():
        filter_ins_dict[table_name] = {}
        for col_name, filter_name in col_names.items():
            filter_ins_dict[table_name][col_name] = ColFilter(None, None, None, vdb_searcher.filter_dir, filter_name)
    CategoricalRanker.filter_ins_dict = filter_ins_dict
    NumericalRanker.filter_ins_dict = filter_ins_dict

    if ranker_cfg:
        if 'correlation_sign_mode' in ranker_cfg:
            CategoricalRanker.correlation_sign_mode = ranker_cfg['correlation_sign_mode']
        if 'correlation_score_threshold' in ranker_cfg:
            CategoricalRanker.correlation_score_threshold = float(ranker_cfg['correlation_score_threshold'])

    if embedding_model is None:
        from src.query_constraint import QueryConstraint  
        embedding_model = QueryConstraint.model
    if embedding_model is None:
        raise RuntimeError(
            "init_operators: no embedding model available. "
            "Call QueryConstraint.init(model_path) before init_operators, or pass embedding_model explicitly."
        )
    ColNameMatcher.init(model=embedding_model, threshold=col_match_threshold)

    all_col_names = []
    for table_name, col_map in vdb_searcher.filter_dict.items():
        all_col_names.extend(col_map.keys())
    ColNameMatcher.precompute(all_col_names)


def add_filter(query_table, filter_dir):
    if query_table.table_name in CategoricalRanker.filter_ins_dict:
        return 

    ColNameMatcher.precompute(list(query_table.df.columns))
    new_entries = {}
    for col_name in query_table.df.columns:
        new_entries[col_name] = ColFilter(
            query_table.df,
            col_name,
            query_table.get_column(col_name)['col_type'],
            filter_dir,
            f"{query_table.table_name}_{col_name}",
        )
    CategoricalRanker.filter_ins_dict[query_table.table_name] = new_entries
    NumericalRanker.filter_ins_dict[query_table.table_name] = new_entries


_token_counter_var: "contextvars.ContextVar" = contextvars.ContextVar(
    "tdagent_token_counter", default=None
)


def reset_token_counter() -> None:
    _token_counter_var.set({"input": 0, "output": 0, "calls": 0})


def add_token_usage(input_tokens: int, output_tokens: int) -> None:
    counter = _token_counter_var.get()
    if counter is None:
        return  
    if input_tokens:
        counter["input"] += int(input_tokens)
    if output_tokens:
        counter["output"] += int(output_tokens)
    counter["calls"] += 1


def get_token_counter() -> dict:
    counter = _token_counter_var.get()
    if counter is None:
        return {"input": 0, "output": 0, "calls": 0}
    return dict(counter) 


def llm_generate(prompt):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
        stream=False,
    )
    usage = getattr(completion, "usage", None)
    if usage is not None:
        add_token_usage(
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )
    return completion.choices[0].message.content



# ===================================================================================================================
# ========================================== PROMPT TEMPLATES =======================================================
# ===================================================================================================================

COL_CONSTRAINT_PROMPT = """# Role
A column-constraint extractor for table-discovery queries.

# Goal
Identify every column a candidate table MUST contain to satisfy the query (join, union, or filter columns).

# Inputs
- Query: a natural-language analytical request.
- Query Table Schema: column names of the user-provided query table (may be empty).

# Output
A JSON array of objects with keys:
- `col_name` (string)
- `col_type` ("Numerical" | "Categorical")
- `values` (array)

No prose, no markdown.

# Rules
1. EVERY column name explicitly mentioned in the Query (in single quotes, double quotes, or as a clearly named identifier / metric / entity) MUST appear as a constraint, even if no value is restricted on it. Do not omit a column just because it is only used for joining, correlating, or as a result field — if it is named, it is required.
2. Do NOT emit vague connective phrases that are not real columns, e.g. "another factor", "a related metric", "associated records", "some attribute", "related data". These are not columns — skip them.
3. A VALUE CONDITION is present whenever the Query restricts a column's magnitude — INCLUDING APPROXIMATE phrasing. Treat ALL of the following as value conditions and produce a NON-EMPTY numeric range (never `values=[]` when a number is tied to the column, and NEVER a zero-width `[N, N]`):
   - "around / about / approximately / roughly / near / close to / ~ N", or "reached/at around N" → a window `[N - 0.1*abs(N), N + 0.1*abs(N)]`.
   - "at least / over / more than / greater than / above / from N" → `[N, null]`.
   - "at most / under / less than / below / up to / no more than N" → `[null, N]`.
   - "between A and B" → `[A, B]`.
   If the column name carries a unit (e.g. ends in "thousands"/"millions"/"percent"), express N in THAT unit (e.g. "206 thousand" on a `..._thousands_...` column → 206, not 206000).
4. `values=[]` ONLY when the column is required but the Query restricts NO magnitude/category on it.
5. Categorical `values` lists the explicitly required cell values/entities (e.g. `Team = "Ferrari"` → `["Ferrari"]`). Do NOT put a phrase that merely identifies WHICH column (e.g. an age-bucket label like "50 to 54", a year like "2008") into `values` — that names a column, not a cell value.
6. Numerical `values` is `[min, max]`; one-sided ranges use `null` for the open side. Never output `[null, null]`.

# Query
{query}

# Query Table Schema
{table_schema}
"""


PLANNING_TEMPLATE = """## Operator History (DO NOT REPEAT any call with identical input)
{ops_history}

## Question
{query}

## Source Tables
{src_tables}

## Current Node
- id: {current_node_id}
- valid node ids for tool arguments: {valid_node_ids}

## Query State
{col_constraints}

## Observation Evidence
{observation_evidence}
"""

PLANNING_SYS_PROMPT = """# Role
You DECIDE the next operator(s) to extend the reasoning tree, using YOUR OWN state plus the
Observation Analyst's EVIDENCE (what is reachable around you — evidence, not an order; YOU decide).

# Reading the inputs
- Source Tables — the columns you ALREADY hold at the current node (look here before reaching out).
- Query State — the columns still UNRESOLVED. One with an `Interval`/`Value Domain` is a VALUE
  constraint (filter it by value); one without is a presence/structural constraint.
- Observation Evidence — reachable neighbour clusters: `reaches` (their columns), `via` (the bridge),
  `relates_to` (constraints they contain).
- Operator History + `Tree-Wide Operations Already Tried` — never repeat an identical (operator,input).

# Operators — what each does to your candidate tables
- EXPAND (brings NEW tables): `join_search` / `union_search` / `correlation_search` traverse from a
  source node's tables to CONNECTED tables (shared join key / shared columns / numeric correlation);
  `metadata_seeker` opens a fresh direction by schema, available ONLY on an empty node.
- FILTER (re-ranks the CURRENT tables, adds none): `range_ranker` (numeric range) / `eq_ranker` (exact category).
- `combine` — set-merge two branches (intersection / union / difference).
- `end_search` — LOCK IN the current node's tables as a RESULT leaf; takes no args. You do NOT need all
  constraints resolved. The final answer is the UNION of ALL end_search'd leaves.

# How to plan
FIRST, read the query as an ORDERED chain of logical STEPS and follow it step by step. A multi-step
query spells its steps out in sequence — e.g. "records from A linked to B ... among those, the ones
where C ... then the final attributes from D" is three ordered steps. Map ONE hop to ONE step and
execute them IN THE QUERY'S ORDER:
- Do the CURRENT step before the next — never skip ahead to a later step's columns.
- Advance each step over the EXACT link the query names for it (that step's bridge column / shared
  field), and match the operator to that link — do not substitute a different bridge or relationship
  kind because it looks easier.
- A step's value condition (e.g. "where C below X") is a FILTER on that step's tables, not a new step.

You may emit MULTIPLE operators in ONE turn — they run IN PARALLEL, each creating a new child node;
use this to grow a TREE, not a linear chain. Each step's tables live on a leaf, so at EACH step emit
TWO calls together: `end_search` to lock THIS step's tables in, AND a relationship operator to advance
a sibling branch to the NEXT step. Every step then contributes a leaf and the answer is their union —
so DON'T stop after one step, and DON'T wait until the end to start collecting.
Typical: rank step-1 value → [end_search + relate to step-2] → rank step-2 value → [end_search + relate] → …

# Rules
- NEVER repeat an `(operator,input)` already tried; if a direction is exhausted, expand a different way.
- `range_ranker.min_val`/`max_val` = numbers or JSON `null` for an open bound; never min==max (use `eq_ranker`).
- OR-queries ("either X or Y"): build each branch, then `combine(set_op='union')` in a later turn.
- The bare root (only the query_table) is never an answer.
"""

OBSERVATION_STAGE_SYS_PROMPT = """# Role
You are the Observation Analyst. You only OBSERVE table relationships: given the current candidate
tables and the (many, tangled) lake PATHS leading outward, report a SHORT ranked list of reachable
neighbour CLUSTERS as EVIDENCE for the Planner. Strict boundary: you do NOT choose operators, you do
NOT mention join/union/rank/filter, and you do NOT reason about values. You only describe WHAT is
reachable, HOW it connects, and WHICH constraints it relates to. The Planner decides what to do.

# Inputs
- Query — the user's need (use it to judge which neighbours matter).
- Unresolved Constraints — `[c{i}]`: a quoted query column + type (+ optional value condition, shown only so you can judge relevance — do NOT check values).
- Observed Paths — `[p{i}]`: route (nodes + bridge keys) + endpoint schema + connectivity score.

# Build the evidence
1. MERGE near-duplicate paths (same bridge / similar endpoint schema) into ONE cluster — never list duplicates.
2. Per cluster: `reaches` (the cluster's notable columns, one phrase); `via` (the bridge it connects through — the overlapping key, or the shared columns); `relates_to` (constraint ids whose column actually appears in the cluster's schema by name+type — never claim otherwise); `promise` (high/medium/low by connectivity + relevance); `why` (one sentence, query-grounded).
3. RANK most-relevant first; at most 4 clusters. Constraints reached by NO path → `stuck`.

# Output — JSON only, no prose/fences:
{"directions":[{"id":"r1","reaches":"<cols>","via":"<bridge key or shared cols>","relates_to":["c2","c3"],"promise":"high|medium|low","why":"<one sentence>"}],"stuck":["c5"]}
"""

OBSERVATION_STAGE_TEMPLATE = """## Query
{query}

## Unresolved Constraints
{col_constraints}

## Observed Paths
{paths}
"""

COL_CONSTRAINTS_SATISFIED_HINT = (
    "All column constraints are satisfied. Check whether the search is complete based on "
    "the user query and the operator history; if so, call `end_search` to terminate."
)


def render_evidence(evidence) -> str:

    if evidence is None:
        return "Observation stage failed; evidence unavailable."
    if not isinstance(evidence, dict) or not evidence:
        return "None."

    directions = evidence.get("directions") or []
    stuck = evidence.get("stuck") or []
    if not directions and not stuck:
        return "None."

    parts = []
    if directions:
        parts.append("### Reachable neighbours (EVIDENCE from the relationship graph — not instructions)")
        for d in directions:
            did = d.get("id", "?")
            reaches = d.get("reaches", "?")
            via = d.get("via", "?")
            rel = ", ".join(d.get("relates_to") or []) or "(none)"
            promise = d.get("promise", "?")
            parts.append(f"- **{did}** reaches {reaches} | via {via} | relates to {rel} | promise {promise}")
            why = d.get("why")
            if why:
                parts.append(f"    why: {why}")

    if stuck:
        if parts:
            parts.append("")
        parts.append(
            "### Unreached constraints (no path in the graph reaches these): "
            + ", ".join(str(s) for s in stuck)
        )

    if not parts:
        return "None."
    return "\n".join(parts)

