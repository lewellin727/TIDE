import json
from typing import List
from src.table import Table
from src.utils import COL_CONSTRAINT_PROMPT, llm_generate
from src.operators.seeker import ContextSeeker
from src.operators.ranker import CategoricalRanker, NumericalRanker
from src.col_name_matcher import ColNameMatcher


class QueryConstraint:

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
        return len(constraints)

    @classmethod
    def _col_value_ok(cls, con: dict, cell: dict, num_threshold=None) -> bool:
        col_values = con.get("values", [])
        if not col_values:
            return True  

        raw_values = cell.get("values")
        iterable = [] if raw_values is None else raw_values

        if con["col_type"] == "Categorical":
            try:
                table_set = set(iterable)
            except TypeError:
                return False 
            return all(v in table_set for v in col_values)

        if len(col_values) != 2:
            return False
        lo = float('-inf') if col_values[0] is None else col_values[0]
        hi = float('inf') if col_values[1] is None else col_values[1]
        threshold = cls.numerical_predicate_threshold if num_threshold is None else num_threshold
        n_in = n_total = 0
        for v in iterable:
            if v is None or (isinstance(v, float) and v != v):  # skip None and NaN
                continue
            n_total += 1
            try:
                if lo <= v <= hi:
                    n_in += 1
            except TypeError:
                continue
        if n_total == 0:
            return False
        if threshold <= 0.0:
            return n_in > 0
        return (n_in / n_total) >= threshold

    @classmethod
    def _satisfies(cls, con: dict, table: Table, num_threshold=None) -> bool:

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

        for con in value_cons:
            if cls._satisfies(con, table, num_threshold):
                return True
        return False

    VALUE_POOL = 300   
    EXIST_POOL = 10   

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