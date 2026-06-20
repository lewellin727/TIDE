import os
import numpy as np
import pandas as pd
import math
import random
import joblib
import hashlib
import scipy.stats as stats

class CategoricalFrequencyFilter:
    def __init__(self, values, error_rate=0.01, confidence=0.99, seed=42, state_dict=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            self.total_count = 0
            self.width = int(math.ceil(math.e / error_rate))
            self.depth = int(math.ceil(math.log(1 / (1 - confidence))))
            self.table = np.zeros((self.depth, self.width), dtype=int)
            self.p = (1 << 31) - 1
            
            random.seed(seed) 
            self.hash_params = []
            for _ in range(self.depth):
                a = random.randint(1, self.p - 1)
                b = random.randint(0, self.p - 1)
                self.hash_params.append((a, b))
            for v in values:
                self.add(v)

    def _hash(self, item, row_idx):
        base_val = int.from_bytes(hashlib.md5(str(item).encode("utf-8")).digest()[:4], "big")
        a, b = self.hash_params[row_idx]
        return ((a * base_val + b) % self.p) % self.width

    def add(self, item):
        self.total_count += 1
        for i in range(self.depth):
            col = self._hash(item, i)
            self.table[i, col] += 1

    def get_frequency_score(self, item):
        if self.total_count == 0:
            return 0.0
        
        min_count = float('inf')
        for i in range(self.depth):
            col = self._hash(item, i)
            if self.table[i, col] < min_count:
                min_count = self.table[i, col]

        return min_count / self.total_count

    def state_dict(self):
        return {
            'total_count': self.total_count,
            'width': self.width,
            'depth': self.depth,
            'p': self.p,
            'hash_params': self.hash_params,
            'table': self.table 
        }
    
    def load_state_dict(self, state_dict):
        self.total_count = state_dict['total_count']
        self.width = state_dict['width']
        self.depth = state_dict['depth']
        self.p = state_dict['p']
        self.hash_params = state_dict['hash_params']
        self.table = state_dict['table']


class CategoricalOverlapFilter:
    def __init__(self, df, key_col, n=256, state_dict=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            self.n = n
            self.key_col = key_col
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                grouped_df = df[[key_col]].drop_duplicates().reset_index(drop=True)
            else:
                agg_dict = {col: "mean" for col in num_cols}
                defrag_df = df.copy()
                grouped_df = defrag_df.groupby(key_col).agg(agg_dict).reset_index()
            grouped_df = grouped_df.assign(
                h_k=grouped_df[key_col].apply(self._hash_h),
                hu_k=grouped_df[key_col].apply(self._hash_hu)
            )
            self.sketch_df = grouped_df.nsmallest(self.n, 'hu_k').copy()

    def _hash_h(self, key) -> int:
        hash_hex = hashlib.md5(str(key).encode('utf-8')).hexdigest()
        return int(hash_hex[:8], 16)  

    def _hash_hu(self, key) -> float:
        hash_hex = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
        return int(hash_hex[:16], 16) / (16**16 - 1)
    
    def state_dict(self):
        return {
            'n': self.n,
            'col_name': self.key_col,
            'sketch_df': self.sketch_df  
        }

    def load_state_dict(self, state_dict):
        self.n = state_dict['n']
        self.key_col = state_dict['col_name']
        self.sketch_df = state_dict['sketch_df'].copy() if state_dict['sketch_df'] is not None else None


    def get_overlap_score(self, sketch_2): 
        actual_n = min(self.n, len(self.sketch_df), len(sketch_2))
        hu_A = self.sketch_df['hu_k'].values[:self.n]
        hu_B = sketch_2['hu_k'].values[:self.n]
        merged = np.concatenate([hu_A, hu_B])
        _, counts = np.unique(merged, return_counts=True)
        top_n_counts = counts[:self.n]
        overlap_count = np.sum(top_n_counts == 2)

        jaccard = overlap_count / actual_n if actual_n > 0 else 0.0
        return jaccard
    
    def get_corr_score(self, col_1, sketch_2, col_2):
        df1 = self.sketch_df[['h_k', self.key_col, col_1]]
        df2 = sketch_2[['h_k', col_2]]
        merged_df = df1.merge(df2, on='h_k', how='inner', suffixes=('_left', '_right'))

        if col_1 == col_2:
            left_col, right_col = f'{col_1}_left', f'{col_2}_right'
        else:
            left_col, right_col = col_1, col_2

        if len(merged_df) > 1:
            pearson, _ = stats.pearsonr(merged_df[left_col], merged_df[right_col])
        else:
            pearson = 0.0

        return pearson


class CategoricalFilter:
    def __init__(self, df, col_name, state_dict=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            self.frequency_filter = CategoricalFrequencyFilter(df[col_name].dropna())
            self.overlap_filter = CategoricalOverlapFilter(df, col_name)

    def state_dict(self):
        return {
            'frequency_filter': self.frequency_filter.state_dict(),
            'overlap_filter': self.overlap_filter.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.frequency_filter = CategoricalFrequencyFilter(values=None, state_dict=state_dict['frequency_filter'])
        self.overlap_filter = CategoricalOverlapFilter(df=None, key_col=None, state_dict=state_dict['overlap_filter'])


class NumericalFilter:
    def __init__(self, df, col_name, num_bins=100, state_dict=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            values = df[col_name].dropna()  
            self.total_count = len(values)
            quantiles = np.linspace(0, 100, num_bins + 1)
            self.bin_edges = np.percentile(values, quantiles)
            self.counts, _ = np.histogram(values, bins=self.bin_edges)

    def get_range_score(self, min_val, max_val):
        min_val = float('-inf') if min_val is None else min_val
        max_val = float('inf') if max_val is None else max_val
        
        estimated_count = 0.0
        for i in range(len(self.counts)):
            left_edge = self.bin_edges[i]
            right_edge = self.bin_edges[i+1]
            bucket_count = self.counts[i]
            
            if bucket_count == 0:
                continue
            if right_edge < min_val or left_edge > max_val:
                continue
            elif left_edge >= min_val and right_edge <= max_val:
                estimated_count += bucket_count
            else:
                overlap_left = max(left_edge, min_val)
                overlap_right = min(right_edge, max_val)
                if right_edge > left_edge:
                    ratio = (overlap_right - overlap_left) / (right_edge - left_edge)
                    estimated_count += bucket_count * ratio
                else:
                    estimated_count += bucket_count
        return estimated_count / self.total_count
    
    def state_dict(self):
        return {
            'total_count': self.total_count,
            'bin_edges': self.bin_edges,
            'counts': self.counts
        }

    def load_state_dict(self, state_dict):
        self.total_count = state_dict['total_count']
        self.bin_edges = state_dict['bin_edges']
        self.counts = state_dict['counts']


class ColFilter:
    def __init__(self, df, col_name, col_type, filter_dir, filter_name):
        save_path = os.path.join(filter_dir, f"{filter_name}.bin")
        if os.path.exists(save_path):
            self.load_filter(save_path)
        else:
            self.build_col_filter(df, col_name, col_type, save_path)

    def build_col_filter(self, df, col_name, col_type, save_path):
        if col_type == "Categorical":
            self.filter = CategoricalFilter(df, col_name)
        else:
            self.filter = NumericalFilter(df, col_name)
        joblib.dump(self.filter.state_dict(), save_path)


    def load_filter(self, save_path):
        state_dict = joblib.load(save_path)
        if 'frequency_filter' in state_dict.keys():
            self.filter = CategoricalFilter(df=None, col_name=None, state_dict=state_dict)
        else:
            self.filter = NumericalFilter(df=None, col_name=None, state_dict=state_dict)
