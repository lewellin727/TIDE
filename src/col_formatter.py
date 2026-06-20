import math
from tqdm import tqdm
import pandas as pd
from collections import Counter
import re

class ColFormatter:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_tokens_len = tokenizer.num_special_tokens_to_add(pair=False)
        self.col_types = None
        self.df_counter = None
        self.total_categorical_columns = None

    def normalize_name(self, name):
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        name = name.replace('_', ' ').replace('-', ' ')
        return re.sub(r'\s+', ' ', name).strip().lower()

    def col_format_content(self, col_values, col_name, col_type, max_seq_length=512):
        col_name = self.normalize_name(col_name)
        prefix_str = f"Column: {col_name} | Type: {col_type} | Content: "
        prefix_tokens_len = len(self.tokenizer.tokenize(prefix_str))
        budget = max_seq_length - self.special_tokens_len - prefix_tokens_len
        if col_type == "Categorical":
            content_str = self.categorical_content_str(col_values, budget)
        else:
            content_str = self.numerical_content_str(col_values, budget)
        return prefix_str + content_str
    

    def col_format_metadata(self, col_name, col_type, table_col=None):
        col_name = self.normalize_name(col_name)
        return f"Column: {col_name} | Type: {col_type} | Description: None"

    def calc_col_types(self, tables):
        col_types = {}
        for filename, df in tables.items():
            col_types[filename] = {}
            for col_name, dtype in df.dtypes.items():
                col_types[filename][col_name] = "Numerical" if str(dtype).startswith(('int', 'float')) else "Categorical"
        self.col_types = col_types

    
    def calc_df(self, tables):
        df_counter = Counter()
        total_categorical_columns = 0
        for filename, df in tqdm(tables.items(), desc="Computing Global DF (Categorical Only)"):
            for col_name in df.columns:
                if self.col_types[filename][col_name] == "Numerical":
                    continue
                valid_cells = df[col_name].dropna()
                if len(valid_cells) == 0:
                    continue
                unique_vals = valid_cells.unique()
                df_counter.update(unique_vals)
                total_categorical_columns += 1
        self.df_counter = df_counter
        self.total_categorical_columns = total_categorical_columns


    def categorical_content_str(self, col_values, budget):
        valid_cells = col_values.dropna()
        tf_counter = Counter(valid_cells)
        tfidf_scores = {}
        for val, tf in tf_counter.items():
            df_val = self.df_counter.get(val, 0)
            idf = math.log((self.total_categorical_columns + 1) / (df_val + 1)) + 1
            tfidf_scores[val] = tf * idf
            
        sorted_vals = sorted(tf_counter.keys(), key=lambda x: tfidf_scores[x], reverse=True)
        selected_vals = []
        current_tokens_len = 0
        for val in sorted_vals:
            val_str = str(val)
            val_tokens_len = len(self.tokenizer.tokenize(val_str))
            if current_tokens_len + val_tokens_len + 1 <= budget:
                selected_vals.append(val_str)
                current_tokens_len += (val_tokens_len + 1)
            else:
                break
        content_str = ", ".join(selected_vals)
        return content_str


    def numerical_content_str(self, col_values, budget):
        nan_count = col_values.isna().sum()    
        valid_cells = col_values.dropna() 
        num_series = pd.to_numeric(valid_cells, errors='coerce').dropna()
        
        if not num_series.empty:
            unique_count = num_series.nunique()
            v_min = num_series.min()
            v_max = num_series.max()
            v_mean = num_series.mean()
            v_std = num_series.std()
            v_std = 0.0 if pd.isna(v_std) else v_std
            quantiles = num_series.quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
            content_str = (
                f"Unique: {unique_count}, NaN: {nan_count}, "
                f"Min: {v_min:.2f}, Max: {v_max:.2f}, Mean: {v_mean:.2f}, Std: {v_std:.2f}, "
                f"P20: {quantiles[0.2]:.2f}, P40: {quantiles[0.4]:.2f}, P60: {quantiles[0.6]:.2f}, P80: {quantiles[0.8]:.2f}"
            )
            if len(self.tokenizer.tokenize(content_str)) > budget:
                content_str = content_str[:budget*3] 
        else:
            content_str = "N/A"
        return  content_str
    

    def state_dict(self):
        return {
            "col_types": self.col_types,
            "df_counter": self.df_counter,
            "total_categorical_columns": self.total_categorical_columns
        }
    
    def load_state_dict(self, state_dict):
        self.col_types = state_dict["col_types"]
        self.df_counter = state_dict["df_counter"]
        self.total_categorical_columns = state_dict["total_categorical_columns"]

