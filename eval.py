import os
import json
import numpy as np

def calculate_metrics(ground_truth_tables, predicted_tables, k):
    """
    Calculate Precision@k, Recall@k, and NDCG@k
    """
    if not predicted_tables:
        return 0.0, 0.0, 0.0, 0.0

    # Ensure tables are treated without .csv extension for reliable comparison
    gt_set = set([t.replace('.csv', '') for t in ground_truth_tables])
    pred_list_k = [t.replace('.csv', '') for t in predicted_tables[:k]]

    if not gt_set:
        return 0.0, 0.0, 0.0

    # Precision@k (standard: relevant in top-k divided by the cutoff k)
    hits_k = sum(1 for p in pred_list_k if p in gt_set)
    precision = hits_k / k

    # Recall@k
    recall = hits_k / len(gt_set)

    recall_gt = min(k, len(gt_set)) / len(gt_set)

    # NDCG@k
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(pred_list_k) if p in gt_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(gt_set))))
    
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg, recall_gt

def eval_results(metadata_path, results_path, output_path, k_values=[1, 3, 5, 8, 10, 15, 20]):
    print("\n--- Starting Evaluation ---")
    
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found at {metadata_path}")
        return
    if not os.path.exists(results_path):
        print(f"Error: results file not found at {results_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    gt_dict = {}
    for meta in metadata:
        query_id = meta.get('id')
        gt_dict[query_id] = meta.get('grounds', [])

    query_json_path = os.path.join(os.path.dirname(metadata_path), "query.json")
    query_list = []
    if os.path.exists(query_json_path):
        with open(query_json_path, 'r') as f:
            query_list = json.load(f)

    overall_metrics = {k: {'P': [], 'R': [], 'NDCG': [], 'R_gt': []} for k in k_values}

    n_total = 0          # queries that have ground truth (the evaluable set)
    n_success = 0        # of those, the ones that returned a non-empty candidate list
    for res in results:
        res_idx = res['idx']
        candidates = res.get('candidates', [])

        # map idx to query id
        query_id = None
        if res_idx < len(query_list):
            query_id = query_list[res_idx].get('id')

        if query_id not in gt_dict:
            print(f"Warning: Could not find ground truth for query idx {res_idx} (id: {query_id})")
            continue

        n_total += 1
        if not candidates:
            print(f"Warning: No candidates for query idx {res_idx} (counted as 0)")
        else:
            n_success += 1

        gt_tables = gt_dict[query_id]

        for k in k_values:
            p, r, ndcg, r_gt = calculate_metrics(gt_tables, candidates, k)
            overall_metrics[k]['P'].append(p)
            overall_metrics[k]['R'].append(r)
            overall_metrics[k]['NDCG'].append(ndcg)
            overall_metrics[k]['R_gt'].append(r_gt)

    success_rate = n_success / n_total if n_total else 0.0
    print(f"Success rate: {n_success}/{n_total} = {success_rate:.4f}  "
          f"(P/R/NDCG averaged over ALL {n_total} queries; empty candidates count as 0)")

    reports = [{
        'n_total': n_total,
        'n_success': n_success,
        'success_rate': success_rate,
    }]
    for k in k_values:
        avg_p = np.mean(overall_metrics[k]['P']) if overall_metrics[k]['P'] else 0
        avg_r = np.mean(overall_metrics[k]['R']) if overall_metrics[k]['R'] else 0
        avg_ndcg = np.mean(overall_metrics[k]['NDCG']) if overall_metrics[k]['NDCG'] else 0
        avg_r_gt = np.mean(overall_metrics[k]['R_gt']) if overall_metrics[k]['R_gt'] else 0
        reports.append({
            'K': k,
            f'P@{k}': avg_p,
            f'R@{k}': avg_r,
            f'NDCG@{k}': avg_ndcg,
            f'R_gt@{k}': avg_r_gt
        })
        print(f"K={k}: P@{k}={avg_p:.4f}, R@{k}={avg_r:.4f}, NDCG@{k}={avg_ndcg:.4f}, R_gt@{k}={avg_r_gt:.4f}")
    with open(output_path, 'w') as f:
        json.dump(reports, f, indent=4)
    print(f"Evaluation report saved to {output_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Comma-separated dataset names; defaults to WebTable")
    parser.add_argument("--method", type=str, default="tide",
                        help="Comma-separated method names; defaults to tide")
    args = parser.parse_args()

    methods = args.method.split(",")
    datasets = args.dataset.split(",") if args.dataset else ["WebTable"]

    import yaml
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config = yaml.safe_load(open(os.path.join(curr_path, "config.yaml"), "r"))

    for method in methods:
        for dataset in datasets:
            metadata_path = os.path.join(config["datalake_dir"], dataset, "metadata.json")
            results_path = os.path.join(curr_path, f"results/{dataset}/{method}.json")
            output_path = os.path.join(curr_path, f"eval/{dataset}/{method}.json")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            eval_results(metadata_path, results_path, output_path, k_values=[1, 3, 5, 8, 10, 15, 17, 20, 30, 40, 50])