from typing import List
from src.table import Table
from src.col_name_matcher import ColNameMatcher


class CategoricalRanker:
    filter_ins_dict = None
    correlation_sign_mode = 'abs'
    correlation_score_threshold = 0.5

    @classmethod
    def rank_frequence(cls, tables: List[Table], col_name: str, value: str, top_k=50) -> List[Table]:
        for table in tables:
            matched = ColNameMatcher.find_match(col_name, list(table.df.columns))
            if matched and table.get_column(matched)['col_type'] == 'Categorical':
                ranker = cls.filter_ins_dict[table.table_name][matched]
                score = ranker.filter.frequency_filter.get_frequency_score(value)
                table.score = score
                table.matched_column = matched   
                table.predicate = 'eq'        
            else:
                table.score = 0.0

        ranked_tables = [table for table in tables if table.score > 0]

        if not ranked_tables:
            return []

        total_score = sum(table.score for table in ranked_tables)
        for table in ranked_tables:
            table.score /= total_score

        ranked_tables.sort(key=lambda t: t.score, reverse=True)
        return ranked_tables[:top_k]

    @classmethod
    def rank_overlap(cls, tables_1: List[Table], tables_2: List[Table], join_col_name: str, top_k=50) -> List[Table]:
        filtered_1 = []
        for t_1 in tables_1:
            m1 = ColNameMatcher.find_match(join_col_name, list(t_1.df.columns))
            if m1 and t_1.get_column(m1)['col_type'] == 'Categorical':
                filtered_1.append((t_1, m1))

        filtered_2 = []
        for t_2 in tables_2:
            m2 = ColNameMatcher.find_match(join_col_name, list(t_2.df.columns))
            if m2 and t_2.get_column(m2)['col_type'] == 'Categorical':
                filtered_2.append((t_2, m2))

        for t_2, m2 in filtered_2:
            score = 0.0
            t_2_ranker = cls.filter_ins_dict[t_2.table_name][m2]
            for t_1, m1 in filtered_1:
                t_1_sketch = cls.filter_ins_dict[t_1.table_name][m1].filter.overlap_filter.sketch_df
                score += t_2_ranker.filter.overlap_filter.get_overlap_score(t_1_sketch)
            t_2.score = score
            t_2.matched_column = m2        
            t_2.predicate = 'overlap'     

        ranked_tables_2 = [t_2 for t_2, _ in filtered_2 if t_2.score > 0]

        if not ranked_tables_2:
            return []

        total_score = sum(table.score for table in ranked_tables_2)
        for table in ranked_tables_2:
            table.score /= total_score

        ranked_tables_2.sort(key=lambda t: t.score, reverse=True)
        return ranked_tables_2[:top_k]

    @classmethod
    def rank_correlation(cls, tables_1: List[Table], tables_2: List[Table],
                         join_col_name: str, corr_col_name_1: str, corr_col_name_2: str | None, top_k=50) -> List[Table]:
        filtered_1 = []
        for t_1 in tables_1:
            t_1_cols = list(t_1.df.columns)
            j1 = ColNameMatcher.find_match(join_col_name, t_1_cols)
            c1 = ColNameMatcher.find_match(corr_col_name_1, t_1_cols)
            if not j1 or not c1:
                continue
            if t_1.get_column(j1)['col_type'] != 'Categorical':
                continue
            if t_1.get_column(c1)['col_type'] != 'Numerical':
                continue
            filtered_1.append((t_1, j1, c1))

        filtered_2 = []
        for t_2 in tables_2:
            t_2_cols = list(t_2.df.columns)
            j2 = ColNameMatcher.find_match(join_col_name, t_2_cols)
            if not j2 or t_2.get_column(j2)['col_type'] != 'Categorical':
                continue
            if corr_col_name_2 is not None:
                c2 = ColNameMatcher.find_match(corr_col_name_2, t_2_cols)
                if c2 and t_2.get_column(c2)['col_type'] == 'Numerical':
                    filtered_2.append((t_2, j2, [c2]))
            else:
                num_cols = [
                    col for col in t_2_cols
                    if t_2.get_column(col)['col_type'] == 'Numerical' and col != j2
                ]
                if num_cols:
                    filtered_2.append((t_2, j2, num_cols))

        sign_mode = cls.correlation_sign_mode
        threshold = cls.correlation_score_threshold

        for t_2, j2, num_cols in filtered_2:
            max_score = 0.0
            best_col = None
            t_2_ranker = cls.filter_ins_dict[t_2.table_name][j2]
            for target_col in num_cols:
                col_score = 0.0
                for t_1, j1, c1 in filtered_1:
                    t_1_sketch = cls.filter_ins_dict[t_1.table_name][j1].filter.overlap_filter.sketch_df
                    col_score += t_2_ranker.filter.overlap_filter.get_corr_score(target_col, t_1_sketch, c1)
                score_compare = abs(col_score) if sign_mode == 'abs' else col_score
                if score_compare > max_score:
                    max_score = score_compare
                    best_col = target_col
            t_2.score = max_score
            t_2.matched_column = (best_col, j2) if best_col is not None else j2
            t_2.predicate = 'corr'

        ranked_tables_2 = [t_2 for t_2, _, _ in filtered_2 if t_2.score > threshold]
        if not ranked_tables_2:
            return []

        total_score = sum(table.score for table in ranked_tables_2)
        for table in ranked_tables_2:
            table.score /= total_score

        ranked_tables_2.sort(key=lambda t: t.score, reverse=True)
        return ranked_tables_2[:top_k]


class NumericalRanker:
    filter_ins_dict = None

    @classmethod
    def rank_range(cls, tables: List[Table], col_name: str, min_val: float, max_val: float, top_k=50) -> List[Table]:
        for table in tables:
            matched = ColNameMatcher.find_match(col_name, list(table.df.columns))
            if matched and table.get_column(matched)['col_type'] == 'Numerical':
                ranker = cls.filter_ins_dict[table.table_name][matched]
                score = ranker.filter.get_range_score(min_val, max_val)
                table.score = score
                table.matched_column = matched  
                table.predicate = 'range'        
            else:
                table.score = 0.0

        ranked_tables = [table for table in tables if table.score > 0]

        if not ranked_tables:
            return []

        total_score = sum(table.score for table in ranked_tables)
        for table in ranked_tables:
            table.score /= total_score

        ranked_tables.sort(key=lambda t: t.score, reverse=True)
        return ranked_tables[:top_k]
