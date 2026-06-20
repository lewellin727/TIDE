
import os
import json
import yaml
import pandas as pd
from tqdm import tqdm

# ---- Config (config.yaml) ----
_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_CFG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)


_CFG_MISSING = object()

def cfg(key, default=_CFG_MISSING):

    if key not in CONFIG:
        if default is _CFG_MISSING:
            raise KeyError(key)
        val = default
    else:
        val = CONFIG[key]
    env = os.environ.get(key.upper())
    if env is None:
        return val
    if isinstance(val, bool):
        return env.lower() in ("1", "true", "yes")
    if isinstance(val, int):
        return int(env)
    if isinstance(val, float):
        return float(env)
    return env

# ---- Column-name augmentation ----
import random
import inflection

def get_synonym(word):

    clean_word = inflection.underscore(word).replace('_', ' ')
    if not clean_word.replace(' ', '').isalpha():
        return word
    plural = inflection.pluralize(clean_word)
    singular = inflection.singularize(clean_word)
    variant = plural if plural.lower() != clean_word.lower() else (
        singular if singular.lower() != clean_word.lower() else None)
    if not variant:
        return word
    return variant.replace(' ', '_')

def format_transformation(word):
    choices = [
        lambda x: inflection.camelize(x, uppercase_first_letter=False), # camelCase
        lambda x: inflection.camelize(x, uppercase_first_letter=True),  # PascalCase
        lambda x: inflection.underscore(x),                             # snake_case
        lambda x: x.upper(),                                            # upper
        lambda x: x.lower()                                             # lower
    ]
    transformation = random.choice(choices)
    return transformation(word)

def augment_column_name(col_name):

    p = random.random()
    if p < 0.50:
        return col_name
    elif p < 0.90:
        new_name = get_synonym(col_name)
        if new_name == col_name:
            return format_transformation(col_name)
        return new_name
    else:
        return format_transformation(col_name)

def _force_augment_one(original_columns):
    idx = random.randrange(len(original_columns))
    forced = get_synonym(original_columns[idx])
    if forced == original_columns[idx]:
        forced = format_transformation(original_columns[idx])
    new_cols = list(original_columns)
    new_cols[idx] = forced
    return new_cols

def _build_rename_map(orig_cols):

    orig_set = set(orig_cols)
    m, targets = {}, set()
    for c in orig_cols:
        a = augment_column_name(c)

        if a == c or a in targets or (a in orig_set and a != c):
            a = c
        targets.add(a)
        m[c] = a
    return m


def slice_distractor_tables(df, src_idx, lakes_dir, n_slices, min_cols):

    cols = list(df.columns)
    if len(cols) < min_cols or len(df) < 2:
        return 0
    n_rows = len(df)
    written = 0
    for s in range(n_slices):
        k_cols = random.randint(min_cols, len(cols))
        sub_cols = random.sample(cols, k_cols)
        k_rows = random.randint(max(2, n_rows // 4), n_rows)
        rows = df.sample(n=k_rows) if k_rows < n_rows else df
        rows[sub_cols].reset_index(drop=True).to_csv(
            os.path.join(lakes_dir, f"noise_{src_idx}_{s}.csv"), index=False)
        written += 1
    return written


def augment_tables(input_dir, output_dir, colmap_path=None):

    from collections import defaultdict
    files = [f for f in os.listdir(input_dir)
             if f.endswith(".csv") and not f.startswith("query_table_")]
    by_qid = defaultdict(list)
    for f in files:

        key = "_".join(f.split("_")[:2]) if f.startswith("noise_") else f.split("_")[0]
        by_qid[key].append(f)

    def _sort_key(k):
        return (1, k) if k.startswith("noise_") else (0, f"{int(k):08d}")

    colmaps = {}
    for qid, fs in tqdm(sorted(by_qid.items(), key=lambda kv: _sort_key(kv[0])), desc="augment"):
        seen, order = set(), []
        frames = {}
        for f in fs:
            try:
                df = pd.read_csv(os.path.join(input_dir, f), low_memory=False)
            except Exception:
                continue
            frames[f] = df
            for c in df.columns:
                if c not in seen:
                    seen.add(c); order.append(c)
        m = _build_rename_map(order)
        colmaps[qid] = m
        for f, df in frames.items():
            df.columns = [m.get(c, c) for c in df.columns]
            df.to_csv(os.path.join(output_dir, f), index=False)

    if colmap_path:
        with open(colmap_path, "w") as fh:
            json.dump(colmaps, fh, indent=2)


# ---- Save files ----

def save_query(idx, query, tq_df, input_tables_dir, query_file_path, source_idx=None):
    with open(query_file_path, "r") as f:
        querys = json.load(f)
    input_table_name = None
    if tq_df is not None:                      
        tq_df.to_csv(os.path.join(input_tables_dir, f"query_table_{idx}.csv"), index=False)
        input_table_name = f"query_table_{idx}.csv"

    entry = {"id": idx, "query": query, "input_table": input_table_name}
    if source_idx is not None:
        entry["source_idx"] = source_idx
    querys.append(entry)
    with open(query_file_path, "w") as f:
        json.dump(querys, f, indent=4)


def dag_view(idx, record):

    def hop(e, i):
        h = {"step": i, "type": e["type"], "terminal": e["terminal"]}
        if e.get("branch"):
            h["branch"] = e["branch"]
        if e["key"]:
            h["join_key"] = e["key"]
        if e["union_cols"]:
            h["union_columns"] = e["union_cols"]
        if e["ref"]:
            h["correlation"] = {"measured": e["payload"], "reference": e["ref"]}
        if e["constraint"]:
            c = e["constraint"]
            h["value_filter"] = {"column": c["col"], "predicate": c["kind"], "target": c["target"]}
        return h
    return {
        "id": idx,
        "topology": record["topology"],
        "depth": record["depth"],
        "has_seed_table": record["has_tq"],
        "set_op": record["set_op"],
        "keys": record["keys"],
        "start_concepts": record["start"]["cols"],
        "hops": [hop(e, i) for i, e in enumerate(record["edges"], start=1)],
    }


def save_dag(idx, record, dag_file_path, source_idx=None):
    with open(dag_file_path, "r") as f:
        dags = json.load(f)
    entry = dag_view(idx, record)
    if source_idx is not None:
        entry["source_idx"] = source_idx
    dags.append(entry)
    with open(dag_file_path, "w") as f:
        json.dump(dags, f, indent=2)


def save_metadata(idx, tables, grounds, record, lakes_dir, matadata_file_path, source_idx=None):

    with open(matadata_file_path, "r") as f:
        metadatas = json.load(f)

    for nm, t in tables.items():
        t.to_csv(os.path.join(lakes_dir, nm), index=False)

    entry = {"id": idx, "grounds": grounds, "path": record,
             "n_tables": len(tables), "n_gt": len(grounds)}
    if source_idx is not None:
        entry["source_idx"] = source_idx
    metadatas.append(entry)
    with open(matadata_file_path, "w") as f:
        json.dump(metadatas, f, indent=4)

