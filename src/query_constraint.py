import json
import math
from copy import deepcopy
from typing import List
from src.table import Table
from src.utils import COL_CONSTRAINT_PROMPT, llm_generate
from src.operators.seeker import ContextSeeker
from src.operators.ranker import CategoricalRanker, NumericalRanker
from src.col_name_matcher import ColNameMatcher


class QueryConstraint:
    # Each query constraint combines a column name, type and optional value condition.
    # The numerical threshold is the required fraction of non-null rows in range; zero requires at least one.
    numerical_predicate_threshold = 0.0

    @classmethod
    def init(cls, numerical_predicate_threshold: float = 0.0):
        cls.numerical_predicate_threshold = float(numerical_predicate_threshold)

    @classmethod
    def extract_constraints(cls, query, query_table):
        table_schema = query_table.to_schema() if query_table else []
        base_prompt = COL_CONSTRAINT_PROMPT.format(
            query=query,
            table_schema=json.dumps(table_schema),
        )
        last_error = None
        for attempt in range(3):
            # Give corrective feedback after parsing or schema validation fails.
            if attempt > 0 and last_error:
                prompt = (
                    base_prompt
                    + "\n\n--- RETRY NOTICE ---\n"
                    + f"Previous attempt failed: {last_error}\n"
                    + "Return ONLY a JSON array conforming exactly to the schema above. "
                    + "No prose, no code fences, no extra keys."
                )
            else:
                prompt = base_prompt
            try:
                res = llm_generate(prompt)
                res = json.loads(res[res.find("["):res.rfind("]") + 1])
                res = cls._normalize_constraints(res)
                if cls.validate(res):
                    return res
                last_error = "validation failed (wrong keys / type / value shape)"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
        raise RuntimeError(f"extract_constraints failed after 3 attempts. Last error: {last_error}")

    @classmethod
    def _normalize_constraints(cls, res):
        """Normalize LLM output: collapse `[None, None]` (forbidden by prompt but
        commonly produced) into `[]` for Numerical constraints."""
        if not isinstance(res, list):
            return res
        for r in res:
            if isinstance(r, dict) and r.get("col_type") == "Numerical":
                vals = r.get("values")
                if isinstance(vals, list) and len(vals) == 2 and vals[0] is None and vals[1] is None:
                    r["values"] = []
        return res

    @classmethod
    def validate(cls, res):
        for r in res:
            if set(r.keys()) != {"col_name", "col_type", "values"}:
                return False
            if not r["col_type"] in ["Categorical", "Numerical"]:
                return False
            if r["col_type"] == "Numerical" and len(r["values"]) > 0:
                if len(r["values"]) != 2:
                    return False
                if r["values"] == [None, None]:
                    return False
        return True

    @classmethod
    def count(cls, constraints):
        # A value condition belongs to its column constraint and is not counted separately.
        return len(constraints)

    @classmethod
    def _col_value_ok(cls, con: dict, cell: dict, num_threshold=None) -> bool:
        """Does this specific equivalent column's cell pass `con`'s value check?

        Empty `values` → presence-only (True). Categorical → every target value
        present. Numerical → fraction of valid rows in [lo, hi] meets the
        threshold; `num_threshold` overrides the class default (the output gate
        uses a stricter fraction than the permissive loop-time default).
        """
        col_values = con.get("values", [])
        if not col_values:
            return True

        # Check None explicitly; pandas Series have no scalar truth value.
        raw_values = cell.get("values")
        iterable = [] if raw_values is None else raw_values

        if con["col_type"] == "Categorical":
            try:
                table_set = set(iterable)
            except TypeError:
                return False  # Skip unhashable values.
            return all(v in table_set for v in col_values)

        if len(col_values) != 2:
            return False
        lo = float('-inf') if col_values[0] is None else col_values[0]
        hi = float('inf') if col_values[1] is None else col_values[1]
        threshold = cls.numerical_predicate_threshold if num_threshold is None else num_threshold
        n_in = n_total = 0
        for v in iterable:
            if v is None or (isinstance(v, float) and v != v):
                continue
            n_total += 1
            try:
                if lo <= v <= hi:
                    n_in += 1
            except TypeError:
                continue
        if n_total == 0:
            return False
        # A nonpositive threshold still requires at least one row in range.
        if threshold <= 0.0:
            return n_in > 0
        return (n_in / n_total) >= threshold

    @classmethod
    def _satisfies(cls, con: dict, table: Table, num_threshold=None) -> bool:
        """Per-table predicate: does `table` satisfy constraint `con`?

        Column bridging goes through ColNameMatcher (bge-large @ col_match_threshold) — the
        SAME matcher the seekers/rankers use — so a constraint column matches its aug-renamed
        lake column (e.g. Chinese↔chineses, Profit↔Profits). `find_match` returns the single
        best-matching column.

        Empty `values` → presence-only (an equivalent column suffices).
        Non-empty `values` → that column must also pass the value check (`num_threshold`
        overrides the loop-time numerical fraction; the output side passes a stricter one).
        """
        matched = ColNameMatcher.find_match(con["col_name"], list(table.df.columns))
        if not matched:
            return False
        try:
            cell = table.get_column(matched)
        except Exception:
            return False
        return cell is not None and cls._col_value_ok(con, cell, num_threshold)

    @classmethod
    def passes_value_gate(cls, value_cons, table: Table, num_threshold: float = 0.5) -> bool:
        """Output precision gate over the query's VALUE-bearing constraints.

        Keep the table iff it MATCHES ≥1 value constraint: some equivalent column whose
        VALUE check passes (strict numerical fraction `num_threshold`). A value_neg
        look-alike (right column, wrong value) matches no value constraint and is dropped,
        while a presence-only column match (the old, non-discriminative gate) no longer
        suffices.

        We deliberately do NOT reject a table merely for "violating" some other value
        constraint: column bridging is embedding-based (loose), so a GT table can incidentally
        share a column name with an unrelated constraint it was never meant to satisfy —
        punishing that dropped true positives. Requiring ≥1 real value match keeps the
        precision gain without that risk.
        """
        for con in value_cons:
            if cls._satisfies(con, table, num_threshold):
                return True
        return False

    # Use a wider schema candidate pool when value ranking can filter it.
    VALUE_POOL = 300   # Candidate pool before value ranking.
    EXIST_POOL = 10    # Candidate pool for column-presence queries.

    @classmethod
    def search(cls, constraints) -> List[Table]:
        if not constraints:
            return []

        all_tables = []
        for constraint in constraints:
            col_name = constraint["col_name"]
            col_type = constraint["col_type"]
            col_values = constraint["values"]
            has_value = bool(col_values) and (col_type != "Numerical" or len(col_values) == 2)
            try:
                if has_value:
                    pool = ContextSeeker.search([col_name], [col_type], top_k=cls.VALUE_POOL) or []
                    if col_type == "Categorical":
                        for value in col_values:
                            all_tables.extend(CategoricalRanker.rank_frequence(pool, col_name, value) or [])
                    else:
                        lo = col_values[0] if col_values[0] is not None else float('-inf')
                        hi = col_values[1] if col_values[1] is not None else float('inf')
                        all_tables.extend(NumericalRanker.rank_range(pool, col_name, lo, hi) or [])
                else:
                    all_tables.extend(ContextSeeker.search([col_name], [col_type], top_k=cls.EXIST_POOL) or [])
            except Exception:
                continue

        table_dict = {}
        for table in all_tables:
            if table.table_name not in table_dict:
                table_dict[table.table_name] = table

        if not table_dict:
            return []
        return list(table_dict.values())

    @classmethod
    def revision_prompt(cls, query, children):
        """One bounded batch for successful children of this execution stage."""
        payload = []
        for node, operator, inputs in children:
            tables = []
            for table in node.input_tables[:3]:
                df = table.df.iloc[:, :16]
                tables.append({"name": table.table_name,
                               "columns": {str(k): str(v) for k, v in df.dtypes.items()},
                               "sample_rows": json.loads(df.head(3).to_json(orient="records"))})
            payload.append({"node_id": node.id, "operator": operator, "inputs": inputs,
                            "constraints": node.full_constraints,
                            "unresolved_constraints": node.input_constraints, "tables": tables})
        return (
            "Review branch interpretations using the ORIGINAL query and returned evidence. "
            "Only add overlooked requirements or correct mistaken interpretations justified by the query. "
            "Never invent goals, weaken requirements to fit tables, or mark constraints satisfied. "
            "Samples are incomplete evidence. Prefer no change when unsure. "
            "Return JSON only: {\"nodes\":[{\"node_id\":\"id\",\"add\":[],\"revise\":[]}]}. "
            "Each revise item is {\"index\":0,\"constraint\":{...}}, using the zero-based index in "
            "the COMPLETE constraints list. Each constraint has exactly col_name (nonempty string), "
            "col_type (Categorical or Numerical), and values (array). Categorical values are literal "
            "cell values; Numerical values are [] or [lower,upper], with null for one open bound. "
            "Use [] for presence-only requirements. No remove operation. Omit unchanged nodes or use empty lists.\n"
            + json.dumps({"query": query, "branches": payload}, ensure_ascii=False, default=str)
        )


    @classmethod
    def apply_revision(cls, complete, unresolved, patch):
        """Validate add/replace atomically; retain satisfied requirements for ranking."""
        if not isinstance(patch, dict) or set(patch) != {"node_id", "add", "revise"}:
            raise ValueError("Revision requires node_id, add and revise")
        if not isinstance(patch["add"], list) or not isinstance(patch["revise"], list):
            raise ValueError("add and revise must be arrays")
        def checked(c):
            if not isinstance(c, dict) or set(c) != {"col_name", "col_type", "values"}:
                raise ValueError("Invalid constraint fields")
            if not isinstance(c["col_name"], str) or not c["col_name"].strip():
                raise ValueError("Empty column name")
            if c["col_type"] not in ("Categorical", "Numerical") or not isinstance(c["values"], list):
                raise ValueError("Invalid constraint type/values")
            for v in c["values"]:
                if v is not None and (type(v) not in (str, int, float, bool)
                                      or (type(v) is float and not math.isfinite(v))):
                    raise ValueError("Constraint values must be finite JSON scalars")
            c = cls._normalize_constraints([deepcopy(c)])[0]
            if not cls.validate([c]):
                raise ValueError("Invalid constraint schema")
            if c["col_type"] == "Numerical" and c["values"]:
                lo, hi = c["values"]
                if any(v is not None and type(v) not in (int, float) for v in (lo, hi)):
                    raise ValueError("Numerical bounds must be numbers or null")
                if lo is not None and hi is not None and lo > hi:
                    raise ValueError("Reversed numerical bounds")
            return c
        key = lambda c: json.dumps(c, sort_keys=True, ensure_ascii=False)
        full, pending = deepcopy(complete), deepcopy(unresolved)
        replacements, seen = {}, set()
        for item in patch["revise"]:
            if not isinstance(item, dict) or set(item) != {"index", "constraint"}:
                raise ValueError("Invalid revise item")
            i = item["index"]
            if type(i) is not int or not 0 <= i < len(complete) or i in seen:
                raise ValueError("Invalid or repeated constraint index")
            seen.add(i)
            new = checked(item["constraint"])
            if key(complete[i]) != key(new):
                replacements[key(complete[i])] = new
            full[i] = new
        pending = [deepcopy(replacements.get(key(c), c)) for c in pending]
        pending.extend(deepcopy(list(replacements.values())))
        for item in patch["add"]:
            new = checked(item)
            if key(new) not in {key(c) for c in full}:
                full.append(new)
                pending.append(deepcopy(new))
        pending = list({key(c): c for c in pending}.values())
        return full, pending
