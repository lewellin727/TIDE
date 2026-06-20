import os
import faiss
import joblib
import json
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.col_filter import ColFilter

class VDBSearcher:
    def __init__(self, vdb_dir, lakes_dir, emb_model_path, col_formatter, emb_batch_size=64,
                 score_threshold: float = 0.5):
        self.vdb_dir = vdb_dir
        self.lakes_dir = lakes_dir
        self.emb_model_path = emb_model_path
        self.metadata_index_path = os.path.join(vdb_dir, "metadata.index")
        self.column_index_path = os.path.join(vdb_dir, "column.index")
        self.src_file_path = os.path.join(vdb_dir, "src.json")
        self.formatter_path = os.path.join(vdb_dir, "formatter.bin")
        self.filter_dict_path = os.path.join(vdb_dir, "filter_dict.bin")
        self.filter_dir = os.path.join(vdb_dir, "filters")
        os.makedirs(self.filter_dir, exist_ok=True)
        self.filter_dict = {}
        self.col_formatter = col_formatter
        self.emb_batch_size = emb_batch_size
        self.score_threshold = float(score_threshold)  

        print("Loading embedding model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.emb_model_path)
        self.model = AutoModel.from_pretrained(self.emb_model_path)
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()

        self.build_vdb()
        
    def build_vdb(self):
        
        if os.path.exists(self.filter_dict_path) and os.path.exists(self.formatter_path) and os.path.exists(self.src_file_path):
            print("Loading existing filter, formatter, and source data ...")
            self.filter_dict = joblib.load(self.filter_dict_path)
            self.col_formatter.load_state_dict(joblib.load(self.formatter_path))
            with open(self.src_file_path, "r", encoding="utf-8") as f:
                self.cols_data = json.load(f)
        else:
            print("Extracting columns from data lakes...")
            tables = self.read_datalakes()
            self.col_formatter.calc_col_types(tables)
            self.col_formatter.calc_df(tables)
            cols = self.extract_cols(tables)
            self.cols_data = cols

            joblib.dump(self.filter_dict, self.filter_dict_path)
            joblib.dump(self.col_formatter.state_dict(), self.formatter_path)
            with open(self.src_file_path, "w", encoding="utf-8") as src_file:
                json.dump(cols, src_file, indent=4, ensure_ascii=False)
        
        if os.path.exists(self.metadata_index_path) and os.path.exists(self.column_index_path):
            print("Loading existing VDB indexes...")
        else:
            dim, M = 1024, 32
            index_metadata = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            index_column = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)

            index_metadata.hnsw.efConstruction = 256
            index_column.hnsw.efConstruction = 256

            meta_texts = [str(col.get("metadata", "")) for col in self.cols_data]
            col_texts = [str(col.get("column_content", "")) for col in self.cols_data]

            print("Embedding metadata in batches...")
            meta_embs = self.get_embeddings_batch(meta_texts)
            print("Embedding column content in batches...")
            col_embs = self.get_embeddings_batch(col_texts)

            index_metadata.add(meta_embs)
            index_column.add(col_embs)

            faiss.write_index(index_metadata, self.metadata_index_path)
            faiss.write_index(index_column, self.column_index_path)
            print("Databases built and saved successfully!")
        self.index_metadata = faiss.read_index(self.metadata_index_path)
        self.index_column = faiss.read_index(self.column_index_path)
        self.index_metadata.hnsw.efSearch = 1024
        self.index_column.hnsw.efSearch = 1024


    def read_datalakes(self):
        tables = {}
        for filename in tqdm(sorted(os.listdir(self.lakes_dir)), desc="Reading DataLakes"):
            if not filename.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(self.lakes_dir, filename), low_memory=False)
            if df.shape[0] > 0:
                tables[filename.split(".")[0]] = df
        return tables

    def extract_cols(self, tables):    
        column_representations = []
        for tabel_name, df in tqdm(tables.items(), desc="Formatting Columns"):
            table_columns =  df.columns.tolist()
            for col_name in df.columns:
                col_type = self.col_formatter.col_types[tabel_name][col_name]
                col_content = self.col_formatter.col_format_content(df[col_name], col_name, col_type)
                metadata = self.col_formatter.col_format_metadata(col_name, col_type)

                filter_name = f"{tabel_name}_{col_name}"
                filter = ColFilter(df, col_name, col_type, self.filter_dir, filter_name)
                self.filter_dict.setdefault(tabel_name, {})[col_name] = filter_name

                column_representations.append({
                    "table_name": tabel_name,
                    "column_name": col_name,
                    "column_type": col_type,
                    "column_content": col_content,
                    "metadata": metadata,
                    "filter": filter_name,
                    "table_columns": table_columns
                })

        return column_representations

    def get_embedding(self, text):
        if not text:
            text = ""
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            sentence_embeddings = outputs.last_hidden_state[:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.float().cpu().numpy().astype(np.float32)

    def get_embeddings_batch(self, texts, batch_size=None):
        if batch_size is None:
            batch_size = self.emb_batch_size
        texts = [t if t else "" for t in texts]
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[:, 0]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.float().cpu().numpy().astype(np.float32))
        return np.vstack(all_embs)


    def search(self, query: str, search_type: str = "metadata", top_k: int = 512):
        if search_type not in ["metadata", "column"]:
            raise ValueError("search_type must be either 'metadata' or 'column'")


        query_emb = self.get_embedding(query)
        target_index = self.index_metadata if search_type == "metadata" else self.index_column

        distances, indices = target_index.search(query_emb, top_k)
    
        results = []
        for i in range(top_k):
            if i >= len(indices[0]):
                break
            idx = int(indices[0][i])
            score = distances[0][i]
            if idx == -1 or idx >= len(self.cols_data) or score < self.score_threshold:
                continue
            matched_col_data = self.cols_data[idx]
            results.append({
                "score": float(score), 
                "data": matched_col_data
            })
        return results