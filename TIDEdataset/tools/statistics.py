
import os
import sys
import glob
import json
import numpy as np
from collections import Counter
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cfg


def format_size(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024.0
    return f"{n:.1f}PB"


def kfmt(n):

    n = float(n)
    if abs(n) < 1000:
        return str(int(round(n)))
    for div, suf in ((1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.1f}".rstrip("0").rstrip(".") + suf
    return str(int(round(n)))


def rng(arr):

    return f"{kfmt(min(arr))}~{kfmt(max(arr))}"


def mean_std(arr):

    m, s = float(np.mean(arr)), float(np.std(arr))
    return f"{kfmt(m)}±{kfmt(s)}" if abs(m) >= 1000 else f"{m:.1f}±{s:.1f}"


def lake_stats(split_dir):

    aug = os.path.join(split_dir, "aug_lakes")
    files = glob.glob(os.path.join(aug, "*.csv"))
    if not files:
        return None
    import pandas as pd
    rows, cols, sizes = [], [], []
    for f in tqdm(files, desc="lake tables", leave=False):
        sizes.append(os.path.getsize(f))
        try:
            df = pd.read_csv(f, on_bad_lines="skip", low_memory=False)
            rows.append(df.shape[0]); cols.append(df.shape[1])
        except Exception:
            pass
    return {
        "total": len(files),
        "rows": rng(rows),
        "avg_rows": mean_std(rows),
        "cols": rng(cols),
        "avg_cols": mean_std(cols),
        "table_size": f"{format_size(min(sizes))}~{format_size(max(sizes))}",
    }


def query_stats(split_dir):
    """Query-side stats from DAG.json (structure) + metadata.json (grounds)."""
    dag = json.load(open(os.path.join(split_dir, "DAG.json")))
    meta = {m["id"]: m for m in json.load(open(os.path.join(split_dir, "metadata.json")))}

    topo = Counter(); step_dist = Counter()
    n_seed_table = 0
    gt_sizes, steps = [], []

    for d in dag:
        topo[d["topology"]] += 1
        if d["has_seed_table"]:
            n_seed_table += 1
        # step = #relationships + #set-ops + #value-constraints (identical to avg_step)
        s = len(d["hops"]) + (1 if d["set_op"] else 0) + sum(1 for h in d["hops"] if h.get("value_filter"))
        steps.append(s)
        step_dist[s] += 1
        gt_sizes.append(meta.get(d["id"], {}).get("n_gt", 0))

    hi = max(step_dist)
    step_dist_str = ":".join(str(step_dist.get(k, 0)) for k in range(2, hi + 1))

    return {
        "total": len(dag),
        "table": n_seed_table,
        "avg_gt": mean_std(gt_sizes),
        "avg_step": mean_std(steps),
        "step_dist": step_dist_str,                      # steps 2..max, colon-joined
        "topology": f"{topo.get('chain', 0)}:{topo.get('converge', 0)}",  # chain:converge
        "gt_total": int(np.sum(gt_sizes)),               # total positives, for Pos:Neg
    }


def main():
    curr = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    name = sys.argv[1] if len(sys.argv) > 1 else f"{cfg('dataset')}{cfg('output_suffix', '')}"
    split_dir = os.path.join(curr, "dataset", "datalakes", name)
    if not os.path.isdir(split_dir):
        print(f"split not found: {split_dir}"); return

    lake = lake_stats(split_dir)
    q = query_stats(split_dir)

    # Pos:Neg = total GT (answer set) : all other lake tables (distractors + negatives + noise)
    pos = q["gt_total"]
    neg = (lake["total"] - pos) if lake else 0
    pos_neg = f"1:{neg / pos:.1f}" if pos else "N/A"

    stats = {
        "split": name,
        "table_data_lakes": {
            "total": lake["total"],
            "rows": lake["rows"],
            "avg_rows": lake["avg_rows"],
            "cols": lake["cols"],
            "avg_cols": lake["avg_cols"],
            "table_size": lake["table_size"],
        } if lake else None,
        "natural_language_query": {
            "total": q["total"],
            "table": q["table"],
            "avg_gt": q["avg_gt"],
            "pos_neg": pos_neg,
            "avg_step": q["avg_step"],
            "step_dist": q["step_dist"],            # query counts for steps 2,3,...,max
            "topology": q["topology"],              # chain:converge
        },
    }
    out = os.path.join(curr, "dataset", "statistics", f"{name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
