import os
import joblib
from tqdm import tqdm
import pandas as pd
from typing import List
from src.table import Table
from src.operators.seeker import ContentSeeker
from src.operators.ranker import CategoricalRanker

class RelationshipGraph:

    TOP_K_NEIGHBORS = 10
    PER_COL_TOPK = 8          
    U_REL_AGG = os.environ.get("U_REL_AGG", "min")

    def __init__(self, vdb_searcher):
        self.vdb_searcher = vdb_searcher
        self.graph_path = os.path.join(
            self.vdb_searcher.vdb_dir,
            f"relationship_graph_pc{self.PER_COL_TOPK}_n{self.TOP_K_NEIGHBORS}.bin",
        )

        self._navi_table_cache: dict = {}
        if os.path.exists(self.graph_path):
            self.graph = joblib.load(self.graph_path)
        else:
            self.build()
            joblib.dump(self.graph, self.graph_path)

    def build(self):
        self.graph = {}

        tables = self.vdb_searcher.read_datalakes()
        for filename, df in tqdm(tables.items(), desc="Building relationship graph", total=len(tables)):
            table = Table(filename, df, 0.0)
            self.graph[table.table_name] = self._compute_outgoing_edges(table)

    def _compute_outgoing_edges(self, table) -> dict:

        union_tables_dict = {}
        join_tables_dict = {}
        for col_name in table.df.columns:
            top_k_tables = ContentSeeker.search([table], [col_name], top_k=self.PER_COL_TOPK)
            top_k_tables = [t for t in top_k_tables if t.table_name != table.table_name]

            for top_k_table in top_k_tables:
                if top_k_table.table_name not in union_tables_dict:
                    union_tables_dict[top_k_table.table_name] = top_k_table.score
                else:
                    union_tables_dict[top_k_table.table_name] += top_k_table.score

            if table.get_column(col_name)['col_type'] != 'Categorical':
                continue
            top_k_tables = CategoricalRanker.rank_overlap([table], top_k_tables, col_name)
            for top_k_table in top_k_tables:
                entry = join_tables_dict.setdefault(
                    top_k_table.table_name,
                    {'score': 0.0, 'col_score': {}}
                )
                entry['score'] = max(entry['score'], top_k_table.score)
                prev_col_score = entry['col_score'].get(col_name, 0.0)
                entry['col_score'][col_name] = max(prev_col_score, top_k_table.score)

        top_union = sorted(
            union_tables_dict.items(), key=lambda kv: kv[1], reverse=True,
        )[:self.TOP_K_NEIGHBORS]
        union_sum = sum(w for _, w in top_union)
        union_edges = {
            name: w / union_sum for name, w in top_union
        } if union_sum > 0 else {}

        top_join = sorted(
            join_tables_dict.items(), key=lambda kv: kv[1]['score'], reverse=True,
        )[:self.TOP_K_NEIGHBORS]
        join_sum = sum(entry['score'] for _, entry in top_join)
        if join_sum > 0:
            join_edges = {
                neighbor: {
                    'score': entry['score'] / join_sum,
                    'cols': sorted(
                        entry['col_score'].keys(),
                        key=lambda c, cs=entry['col_score']: cs[c],
                        reverse=True,
                    ),
                }
                for neighbor, entry in top_join
            }
        else:
            join_edges = {}

        return {'union': union_edges, 'join': join_edges}

    def add_query_table(self, query_table) -> None:

        if query_table is None:
            return
        name = query_table.table_name
        if name in self.graph:
            return  
        self.graph[name] = self._compute_outgoing_edges(query_table)

    def remove_query_table(self, table_name) -> None:

        self.graph.pop(table_name, None)
        self._navi_table_cache.pop(table_name, None)


    def _aggregate_u_rel(self, scores) -> float:

        if not scores:
            return 0.0
        agg = self.U_REL_AGG
        if agg == "min":
            return min(scores)
        if agg == "gmean":
            prod = 1.0
            for s in scores:
                prod *= max(s, 1e-9)
            return prod ** (1.0 / len(scores))
        return sum(scores) / len(scores)   

    def navigate(self, input_tables, target_tables, d_o: int = 2, k_1: int = 10):
        start_nodes = [t.table_name if hasattr(t, 'table_name') else t for t in input_tables]
        end_nodes = set([t.table_name if hasattr(t, 'table_name') else t for t in target_tables])

        valid_paths = []

        def dfs(current_node, current_path, current_scores, current_depth):
            if current_depth > d_o:
                return
            if current_node in end_nodes and current_depth > 0:
                u_rel = self._aggregate_u_rel(current_scores)
                tables = [step[1] for step in current_path]
                valid_paths.append({
                    'steps': list(current_path),
                    'tables': tables,
                    'endpoint': current_node,
                    'edge_scores': list(current_scores),
                    'U_rel': u_rel,
                })
                return

            if current_depth == d_o:
                return

            if current_node in self.graph:
                for op_type in ['union', 'join']:
                    edges = self.graph[current_node].get(op_type, {})
                    for neighbor, edge in edges.items():
                        if op_type == 'join':
                            edge_score = edge['score']
                            edge_cols = edge.get('cols', [])
                        else:
                            edge_score = edge
                            edge_cols = None
                        if not any(neighbor == step[1] for step in current_path):
                            current_path.append((op_type, neighbor, edge_cols))
                            current_scores.append(edge_score)
                            dfs(neighbor, current_path, current_scores, current_depth + 1)
                            current_path.pop()
                            current_scores.pop()


        for start_node in start_nodes:
            dfs(start_node, [(None, start_node, None)], [], 0)

        valid_paths.sort(key=lambda x: x['U_rel'], reverse=True)
        return valid_paths[:k_1]
    

    def get_navi_tables(self, table_names) -> List[Table]:
        lakes_dir = self.vdb_searcher.lakes_dir
        tables = []
        for table_name in table_names:
            df = self._navi_table_cache.get(table_name)
            if df is None:
                table_path = os.path.join(lakes_dir, f"{table_name}.csv")
                df = pd.read_csv(table_path, low_memory=False)
                self._navi_table_cache[table_name] = df
            tables.append(Table(table_name, df, 0.0))
        return tables
