import os
import heapq
import pandas as pd
from typing import List
from src.table import Table
from src.col_name_matcher import ColNameMatcher

def names2table(tables_dict, lakes_dir):
    tables = []
    for table_name, score in tables_dict.items():
        table_path = os.path.join(lakes_dir, table_name + ".csv")
        df = pd.read_csv(table_path, low_memory=False)
        tables.append(Table(table_name, df, score))
    return tables


def _aggregate_coverage(per_col_target_score: dict, n_queried: int) -> dict:
    target_tables = set()
    for per_target in per_col_target_score.values():
        target_tables.update(per_target.keys())

    table_2_score = {}
    for target in target_tables:
        matched_scores = [
            per_target[target]
            for per_target in per_col_target_score.values()
            if target in per_target
        ]
        if not matched_scores:
            continue
        coverage = len(matched_scores) / n_queried
        mean_score = sum(matched_scores) / len(matched_scores)
        table_2_score[target] = (coverage ** 2) * mean_score
    return table_2_score


def _take_top_k_normalized(table_2_score: dict, top_k: int) -> dict:
    top_k_keys = heapq.nlargest(top_k, table_2_score, key=table_2_score.get)
    top_k_items = {key: table_2_score[key] for key in top_k_keys}
    total = sum(top_k_items.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in top_k_items.items()}


class ContentSeeker:

    vdb_searcher = None

    @classmethod
    def search(cls, tables: List[Table], col_names: List[str], top_k: int = 20) -> List[Table]:

        per_col_target_score: dict = {col_name: {} for col_name in col_names}
        target_best_col: dict = {}
        target_best_score: dict = {}
        for table in tables:
            for col_name in col_names:
                matched = ColNameMatcher.find_match(col_name, list(table.df.columns))
                if not matched:
                    continue
                col = table.get_column(matched)
                if not col:
                    continue
                content_str = cls.vdb_searcher.col_formatter.col_format_content(
                    col['values'], col_name, col['col_type']
                )
                col_data = cls.vdb_searcher.search(content_str, "column")
                bucket = per_col_target_score[col_name]
                for cd in col_data:
                    target = cd['data']['table_name']
                    score = cd['score']
                    if score > bucket.get(target, 0.0):
                        bucket[target] = score
                    if score > target_best_score.get(target, -1.0):
                        target_best_score[target] = score
                        target_best_col[target] = cd['data'].get('column_name')

        table_2_score = _aggregate_coverage(per_col_target_score, len(col_names))
        if not table_2_score:
            return None
        tables_dict = _take_top_k_normalized(table_2_score, top_k)
        if not tables_dict:
            return None
        result = names2table(tables_dict, cls.vdb_searcher.lakes_dir)
        for t in result:
            t.matched_column = target_best_col.get(t.table_name)  
            t.predicate = 'sem'                            
        return result



class ContextSeeker:

    vdb_searcher = None

    @classmethod
    def search(cls, col_names: List[str], col_types: List[str], top_k: int = 20) -> List[Table]:
        per_col_target_score: dict = {col_name: {} for col_name in col_names}
        target_best_col: dict = {}
        target_best_score: dict = {}
        for (col_name, col_type) in zip(col_names, col_types):
            metadata_str = cls.vdb_searcher.col_formatter.col_format_metadata(col_name, col_type, None)
            col_data = cls.vdb_searcher.search(metadata_str, "metadata")
            bucket = per_col_target_score[col_name]
            for cd in col_data:
                target = cd['data']['table_name']
                score = cd['score']
                if score > bucket.get(target, 0.0):
                    bucket[target] = score
                if score > target_best_score.get(target, -1.0):
                    target_best_score[target] = score
                    target_best_col[target] = cd['data'].get('column_name')

        table_2_score = _aggregate_coverage(per_col_target_score, len(col_names))
        if not table_2_score:
            return None
        tables_dict = _take_top_k_normalized(table_2_score, top_k)
        if not tables_dict:
            return None
        result = names2table(tables_dict, cls.vdb_searcher.lakes_dir)
        for t in result:
            t.matched_column = target_best_col.get(t.table_name)  
            t.predicate = 'sem'                          
        return result

