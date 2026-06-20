import os
import random
import json
import time
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import sys
from tqdm import tqdm
from utils import cfg
from generator import Plan, Skip
from utils import save_query, save_metadata, save_dag, augment_tables, slice_distractor_tables
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def partition(df, valid_query, max_retries):

    MIN_GROUNDS = cfg("min_grounds")
    for attempt in range(1, max_retries + 1):
        plan = Plan(step=cfg("max_hops"))
        try:
            tables, grounds, tq_df, record = plan.realize(df, valid_query)
        except Skip:
            continue
        except Exception as e:
            print(f"  [realize err] {e}")
            continue
        if len(grounds) >= MIN_GROUNDS:
            return tables, grounds, tq_df, record, attempt
    return None, [], None, None, max_retries


if __name__ == '__main__':
    OUTPUT_SUFFIX = cfg("output_suffix", "")          # split name = dataset + suffix (suffix usually "")
    QUERY_TABLE_COUNT = cfg("query_table_count")      # source tables [0:N) -> queries
    DISTRACTOR_TABLE_COUNT = cfg("distractor_table_count")   # source tables [N:N+M) -> noise
    DISTRACTOR_SLICES = cfg("distractor_slices_per_table")
    FAIL_LIMIT = cfg("fail_limit")
    MIN_GROUNDS = cfg("min_grounds")
    MIN_COLS = cfg("min_cols")
    QUERIES_PER_TABLE = cfg("queries_per_table")
    MAX_RETRIES = cfg("partition_retries")

    datasets = [cfg("dataset")]
    random.seed(cfg("seed"))
    np.random.seed(cfg("seed"))
    for dataset in datasets:
        print(f"Processing {dataset}...")
        curr_path = os.path.dirname(os.path.abspath(__file__))
        datalakes_dir = os.path.join(curr_path, f'dataset/datalakes/{dataset}{OUTPUT_SUFFIX}')
        dataset_dir = os.path.join(curr_path, f'dataset/processed_dataset/{dataset}')
        lakes_dir = os.path.join(datalakes_dir, "lakes")
        input_tables_dir = os.path.join(datalakes_dir, 'input_tables')
        os.makedirs(input_tables_dir, exist_ok=True)
        os.makedirs(lakes_dir, exist_ok=True)

        query_file_path = os.path.join(datalakes_dir, "query.json")
        matadata_file_path = os.path.join(datalakes_dir, "metadata.json")
        dag_file_path = os.path.join(datalakes_dir, "DAG.json")     # Phase-A → Phase-B handoff

        begin_idx = 0
        valid_query = 0
        existing_queries = []
        if os.path.exists(query_file_path):
            try:
                with open(query_file_path, "r") as f:
                    existing_queries = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing_queries = []
        if existing_queries:
            valid_query = len(existing_queries)
            source_idxs = [q.get("source_idx") for q in existing_queries if isinstance(q.get("source_idx"), int)]
            if source_idxs:
                begin_idx = max(source_idxs) + 1
            print(f"[RESUME] found {valid_query} queries; resuming from csv idx={begin_idx}")
        else:
            for fp in (query_file_path, matadata_file_path, dag_file_path):
                with open(fp, "w") as f:
                    f.write("[]")

        fail_cnt = 0
        agg = {
            "csv_processed": 0, "saved": 0, "fail_partition": 0,
            "topology_counts": Counter(), "edge_type_counts": Counter(),
            "set_op_counts": Counter(), "depth_dist": Counter(), "value_counts": Counter(),
            "partition_attempts_dist": Counter(), "elapsed_total_s": 0.0, "grounds_sizes": [],
        }
        t_run_start = time.time()
        csv_files = sorted(list(Path(dataset_dir).rglob("*.csv")))

        # BY TABLE COUNT: the first QUERY_TABLE_COUNT source tables generate queries.
        query_files = csv_files[:QUERY_TABLE_COUNT]
        for idx, csv_file in tqdm(enumerate(query_files), total=len(query_files), desc="Query tables"):
            if idx < begin_idx:
                continue
            if fail_cnt >= FAIL_LIMIT:
                print(f"\n[STOP] reached fail limit ({FAIL_LIMIT}); aborting generation.")
                break

            df = pd.read_csv(csv_file, low_memory=False)
            df.columns = [str(col).replace('/', '_') for col in df.columns]
            csv_name = csv_file.name
            print(f"\nProcessing {idx}/{len(csv_files)} | csv={csv_name}")

            # QUERIES_PER_TABLE distinct queries per usable source table (boosts yield).
            for _ in range(QUERIES_PER_TABLE):
                t_q_start = time.time()
                agg["csv_processed"] += 1
                tables, grounds, tq_df, record, attempts = partition(df, valid_query, MAX_RETRIES)
                agg["partition_attempts_dist"][attempts] += 1

                if tables is None:
                    elapsed = time.time() - t_q_start
                    agg["elapsed_total_s"] += elapsed
                    agg["fail_partition"] += 1
                    fail_cnt += 1
                    print(f"[STATS] source_idx={idx} csv={csv_name} result=FAIL_PARTITION "
                          f"attempts={attempts} elapsed_s={elapsed:.2f}")
                    continue

                save_dag(valid_query, record, dag_file_path, source_idx=idx)
                save_query(valid_query, None, tq_df, input_tables_dir, query_file_path, source_idx=idx)
                save_metadata(valid_query, tables, grounds, record, lakes_dir, matadata_file_path, source_idx=idx)

                agg["saved"] += 1
                agg["grounds_sizes"].append(len(grounds))
                agg["topology_counts"][record["topology"]] += 1
                if record["topology"] == "chain":
                    agg["depth_dist"][record["depth"]] += 1
                if record["set_op"]:
                    agg["set_op_counts"][record["set_op"]] += 1
                for e in record["edges"]:
                    agg["edge_type_counts"][e["type"]] += 1
                    if e.get("constraint"):
                        agg["value_counts"][e["constraint"]["kind"]] += 1
                elapsed = time.time() - t_q_start
                agg["elapsed_total_s"] += elapsed
                print(f"[STATS] source_idx={idx} csv={csv_name} result=SAVED vid={valid_query} "
                      f"topology={record['topology']} depth={record['depth']} has_tq={record['has_tq']} "
                      f"attempts={attempts} grounds={len(grounds)} elapsed_s={elapsed:.2f}")
                valid_query += 1

        # End-of-dataset summary (grep-friendly).
        wall_total = time.time() - t_run_start
        gs = agg["grounds_sizes"]
        grounds_summary = f"min={min(gs)} mean={sum(gs)/len(gs):.1f} max={max(gs)}" if gs else "n/a"
        print("\n" + "=" * 80)
        print(f"[SUMMARY] dataset={dataset}")
        print(f"[SUMMARY] csv_processed={agg['csv_processed']} saved={agg['saved']} fail_partition={agg['fail_partition']}")
        print(f"[SUMMARY] topology={dict(agg['topology_counts'])} set_op={dict(agg['set_op_counts'])}")
        print(f"[SUMMARY] chain_depth_dist={dict(sorted(agg['depth_dist'].items()))}")
        print(f"[SUMMARY] edge_type_counts={dict(agg['edge_type_counts'])}")
        print(f"[SUMMARY] value_constraints={dict(agg['value_counts'])} (eq/range rankers)")
        print(f"[SUMMARY] partition_attempts_dist={dict(sorted(agg['partition_attempts_dist'].items()))}")
        print(f"[SUMMARY] elapsed_total={agg['elapsed_total_s']:.1f}s wall_clock_total={wall_total:.1f}s")
        print(f"[SUMMARY] grounds_per_query {grounds_summary}")
        print("=" * 80)


        for old in Path(lakes_dir).glob("noise_*.csv"):
            old.unlink()
        distractor_files = csv_files[QUERY_TABLE_COUNT: QUERY_TABLE_COUNT + DISTRACTOR_TABLE_COUNT]
        n_noise = 0
        for d_off, csv_file in enumerate(distractor_files):
            src_idx = QUERY_TABLE_COUNT + d_off
            try:
                ddf = pd.read_csv(csv_file, low_memory=False)
            except Exception as e:
                print(f"[DISTRACTOR] skip {csv_file.name}: {e}")
                continue
            ddf.columns = [str(c).replace('/', '_') for c in ddf.columns]
            n_noise += slice_distractor_tables(ddf, src_idx, lakes_dir, DISTRACTOR_SLICES, MIN_COLS)
        print(f"[DISTRACTOR] {len(distractor_files)} source tables -> {n_noise} noise tables in lakes/")

        aug_lakes_dir = os.path.join(datalakes_dir, "aug_lakes")
        os.makedirs(aug_lakes_dir, exist_ok=True)
        print(f"[AUGMENT] {lakes_dir} -> {aug_lakes_dir}")
        t_aug = time.time()
        augment_tables(lakes_dir, aug_lakes_dir, os.path.join(datalakes_dir, "aug_colmap.json"))
        print(f"[AUGMENT] done in {time.time() - t_aug:.1f}s")

        # Audit is not run here; run it manually with: python tools/run_audits.py
        print("[AUDIT] skipped (run tools/run_audits.py manually if needed)")
