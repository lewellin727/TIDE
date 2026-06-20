
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

curr_path = os.path.dirname(os.path.abspath(__file__))
base_dataset_dir = os.path.join(curr_path, "dataset")
raw_root = os.path.join(base_dataset_dir, "raw_data")
save_dir = os.path.join(base_dataset_dir, "processed_dataset")

DEFAULT_FILTERS = {
    "min_cols": 5,            # keep if df.shape[1] >  min_cols
    "min_rows": 100,          # keep if df.shape[0] >  min_rows
    "min_num": 2,             # keep if #numeric    >  min_num
    "min_cat": 2,             # keep if #categorical>  min_cat
    "drop_ratio": 0.8,        # drop columns whose missing ratio exceeds this
    "unique_cnt": 10,         # drop columns with fewer than this many unique values
    "text_len": 30,           # drop object columns whose avg string length exceeds this
    "col_overlap": 0.3,       # skip a table sharing > this fraction of column names already seen
}


DATASETS = {
    "SG":  {"raw_subdir": "datasets_SG",  "out_name": "OpenData_SG",
            "filters": {"min_cols": 3, "min_rows": 50, "min_num": 1, "min_cat": 1}},
    "UK":  {"raw_subdir": "datasets_UK",  "out_name": "OpenData_UK",
            "filters": {"min_cols": 3, "min_rows": 50, "min_num": 1, "min_cat": 1}},
    "USA": {"raw_subdir": "datasets_USA", "out_name": "OpenData_USA",
            "filters": {"min_cols": 3, "min_rows": 50, "min_num": 1, "min_cat": 1}},
    "CAN": {"raw_subdir": "datasets_CAN", "out_name": "OpenData_CAN",
            "filters": {"min_cols": 3, "min_rows": 50, "min_num": 1, "min_cat": 1}},
    "webtable": {"raw_subdir": "data/benchmark/webtable/large/split_1", "out_name": "WebTable"},
}


def process(dataset_dir, out_dir, dataset_name, f):
    """Clean every .csv in dataset_dir and write the survivors into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    cnt = 0
    col_sets = set()
    for filename in tqdm(sorted(os.listdir(dataset_dir))):
        if not filename.endswith(".csv"):
            continue
        table_path = os.path.join(dataset_dir, filename)
        try:
            df = pd.read_csv(table_path, low_memory=False)

            cols = set(df.columns)
            if cols and len(cols & col_sets) / len(cols) > f["col_overlap"]:
                continue

            # 1. drop columns with high missing ratio
            df = df.loc[:, df.isnull().mean() <= f["drop_ratio"]]
            # 2. drop columns with low unique count
            df = df[df.columns[df.nunique(dropna=True) >= f["unique_cnt"]]]
            # 3. drop columns with long text
            obj_cols = df.select_dtypes(exclude=[np.number]).columns
            cols_to_drop = []
            for col in obj_cols:
                vals = df[col].dropna()
                if len(vals) and vals.astype(str).str.len().mean() > f["text_len"]:
                    cols_to_drop.append(col)
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # 4. drop rows with missing values
            df = df.dropna()

            # 5. judge whether the table is too small
            if df.shape[1] <= f["min_cols"] or df.shape[0] <= f["min_rows"]:
                continue

            # 6. judge whether the table contains both numerical and categorical columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(num_cols) <= f["min_num"] or len(cat_cols) <= f["min_cat"]:
                continue
            df[num_cols] = df[num_cols].astype(float)
            df[cat_cols] = df[cat_cols].astype("category")

            # normalise illegal characters in column names
            df.columns = [str(col).replace('/', '_') for col in df.columns]

            save_name = f"{dataset_name}_{cnt:05d}.csv"
            df.to_csv(os.path.join(out_dir, save_name), index=False)
            cnt += 1
            col_sets.update(cols)
            del df
        except Exception:
            continue
    print(f"{dataset_name}: processed {cnt} tables -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="registry key of the raw dataset to process")
    for k in ("min_cols", "min_rows", "min_num", "min_cat"):
        parser.add_argument(f"--{k.replace('_', '-')}", type=int, default=None,
                            help=f"override filter '{k}'")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    filters = {**DEFAULT_FILTERS, **spec.get("filters", {})}
    for k in ("min_cols", "min_rows", "min_num", "min_cat"):
        if getattr(args, k) is not None:
            filters[k] = getattr(args, k)

    dataset_dir = os.path.join(raw_root, spec["raw_subdir"])
    out_dir = os.path.join(save_dir, spec["out_name"])
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"raw dir not found: {dataset_dir}")

    print(f"Processing {spec['out_name']} from {dataset_dir}")
    print(f"  filters: {filters}")
    process(dataset_dir, out_dir, spec["out_name"], filters)


if __name__ == "__main__":
    main()
