import os
import sys
import time
import yaml
import json
import asyncio
import argparse
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
import logging
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

# ====================================================================
# Logging
# ====================================================================

_log_fp = None


def _setup_logging(log_path: str) -> None:

    global _log_fp
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log_fp = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(_log_fp)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for _noisy in ("sentence_transformers", "httpx", "urllib3", "openai"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


    sys.stdout = _log_fp
    sys.stderr = _log_fp


# ====================================================================
# Imports of project modules (after logging is configured upstream)
# ====================================================================

from sentence_transformers import SentenceTransformer
from src.table import Table
from src.vdb_searcher import VDBSearcher
from src.col_formatter import ColFormatter
from src.utils import init_operators, add_filter, reset_token_counter, get_token_counter
from src.relationship_graph import RelationshipGraph
from src.query_constraint import QueryConstraint
from src.agent import TableAgent
from src.reasoning_tree import ReasoningTree
from src.operators import tool as tool_ctx


# ====================================================================
# Per-query execution
# ====================================================================

async def run_one_query(idx: int, query: dict, ctx: "RunContext") -> None:

    t_start = time.time()
    cand_tables = []
    error_msg = None

    async with ctx.sem:
        reset_token_counter()
        q = query["query"]
        query_table_name = query.get("input_table")
        q_excerpt = q[:200] + ("..." if len(q) > 200 else "")
        logging.info("[Q%d] START input_table=%s query=%s", idx, query_table_name, q_excerpt)

        query_table = None
        try:

            if query_table_name:
                df = pd.read_csv(os.path.join(ctx.input_tables_dir, query_table_name))
                query_table = Table(query_table_name.split(".")[0], df, 0.0)
                add_filter(query_table, ctx.vdb_searcher.filter_dir)

                ctx.rel_graph.add_query_table(query_table)
                logging.info(
                    "[Q%d] query_table=%s inserted into G_R (union_edges=%d, join_edges=%d)",
                    idx, query_table.table_name,
                    len(ctx.rel_graph.graph[query_table.table_name].get('union', {})),
                    len(ctx.rel_graph.graph[query_table.table_name].get('join', {})),
                )


            max_attempts = ctx.agent_cfg.get('max_attempts', 3)
            for attempt in range(1, max_attempts + 1):
                try:
                    init_constraints = await asyncio.to_thread(
                        QueryConstraint.extract_constraints, q, query_table
                    )
                    logging.info(
                        "[Q%d] attempt=%d/%d init_constraints=%d (%s)",
                        idx, attempt, max_attempts, len(init_constraints),
                        ",".join(c.get("col_name", "?") for c in init_constraints) or "none",
                    )
                    reasoning_tree = ReasoningTree(
                        query_table, init_constraints, ctx.lakes_dir,
                        cand_top_k=ctx.cand_top_k,
                        final_top_k=ctx.final_top_k,
                        final_min_constraint_match=ctx.final_min_constraint_match,
                    )
                    table_agent = TableAgent(
                        ctx.rel_graph, reasoning_tree,
                        max_loop=ctx.agent_cfg.get('max_loop', 8),
                        k_1=ctx.nav_cfg.get('k_1', 10),
                        k_o=ctx.nav_cfg.get('k_o', 3),
                        d_o=ctx.nav_cfg.get('d_o', 2),
                        query_idx=idx,
                    )
                    try:
                        cand_tables = await asyncio.wait_for(
                            table_agent.discovery(q), timeout=ctx.per_query_timeout_s
                        )
                    except asyncio.TimeoutError:
                        logging.warning("[Q%d] attempt=%d timed out after %ds", idx, attempt, ctx.per_query_timeout_s)
                        cand_tables = []
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    logging.exception("[Q%d] attempt=%d crashed", idx, attempt)
                    cand_tables = []
                if cand_tables:
                    error_msg = None
                    break
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logging.exception("[Q%d] crashed", idx)
        finally:

            if query_table is not None:
                ctx.rel_graph.remove_query_table(query_table.table_name)

        record = {
            "idx": idx,
            "query": q,
            "candidates": [t.table_name for t in cand_tables],
        }
        if error_msg is not None:
            record["error"] = error_msg

        async with ctx.result_lock:
            ctx.results = [r for r in ctx.results if r["idx"] != idx]
            ctx.results.append(record)
            ctx.results.sort(key=lambda r: r["idx"])
            tmp = ctx.res_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(ctx.results, f, indent=4)
            os.replace(tmp, ctx.res_file)

        elapsed = time.time() - t_start
        t = get_token_counter()
        logging.info(
            "[Q%d] END candidates=%d elapsed=%.1fs llm_calls=%d in_tok=%d out_tok=%d total_tok=%d%s",
            idx, len(cand_tables), elapsed,
            t["calls"], t["input"], t["output"], t["input"] + t["output"],
            f" error={error_msg}" if error_msg else "",
        )


# ====================================================================
# Async driver
# ====================================================================

@dataclass
class RunContext:

    vdb_searcher: Any
    rel_graph: Any
    lakes_dir: str
    input_tables_dir: str
    cand_top_k: int
    final_top_k: int
    final_min_constraint_match: int
    per_query_timeout_s: int
    agent_cfg: dict
    nav_cfg: dict
    res_file: str

    results: list = field(default_factory=list)

    sem: Optional[asyncio.Semaphore] = None
    result_lock: Optional[asyncio.Lock] = None


async def driver(queries: list, ctx: RunContext, selected_idxs: Optional[set], concurrency: int) -> None:
    ctx.sem = asyncio.Semaphore(concurrency)
    ctx.result_lock = asyncio.Lock()

    tasks = []
    for idx, query in enumerate(queries):
        if selected_idxs is not None and idx not in selected_idxs:
            continue

        tasks.append(asyncio.create_task(run_one_query(idx, query, ctx)))

    total = len(tasks)
    done = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        done += 1

        sys.stderr.write(f"  progress: {done}/{total} queries finished\n")
        sys.stderr.flush()


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.realpath(__file__))
    config = yaml.safe_load(open(os.path.join(curr_path, "config.yaml"), "r"))

    dataset = config["dataset"]

    parser = argparse.ArgumentParser(description="Run TIDE with optional dataset override")
    parser.add_argument("--dataset", type=str, default=None, help="The dataset to use (e.g. OpenData_SG)")
    parser.add_argument("--idxs", type=str, default=None, help="Comma-separated query indices to run (default: all). E.g. --idxs 9,13,17")
    parser.add_argument("--concurrency", type=int, default=None, help="Override agent.concurrency from config")
    parser.add_argument("--log", type=str, default=None, help="Path to main log file; default logs/{dataset}.log")
    parser.add_argument("--out", type=str, default=None, help="Path to results JSON; default results/{dataset}/tide.json")
    args = parser.parse_args()

    if args.dataset:
        dataset = args.dataset

    selected_idxs = (set(int(i) for i in args.idxs.split(",")) if args.idxs else None)

    log_path = args.log or os.path.join(curr_path, f"logs/{dataset}.log")
    _setup_logging(log_path)

    datalake_dir = os.path.join(config["datalake_dir"], dataset)
    result_dir = os.path.join(curr_path, f"results/{dataset}")
    os.makedirs(result_dir, exist_ok=True)

    lakes_dir = os.path.join(datalake_dir, "aug_lakes")
    input_tables_dir = os.path.join(datalake_dir, "input_tables")
    queries = json.load(open(os.path.join(datalake_dir, "query.json"), "r"))

    vdb_dir = os.path.join(config['vdb_dir'], f"{dataset}")
    os.makedirs(vdb_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config['emb_model_path'])
    col_formatter = ColFormatter(tokenizer)
    vdb_searcher = VDBSearcher(
        vdb_dir, lakes_dir, config['emb_model_path'], col_formatter,
        emb_batch_size=config.get('vdb_emb_batch_size', 64),
        score_threshold=config.get('vdb_score_threshold', 0.5),
    )

    ranker_cfg = config.get('ranker', {})
    QueryConstraint.init(
        numerical_predicate_threshold=ranker_cfg.get('numerical_predicate_threshold', 0.0),
    )

    col_match_model = SentenceTransformer(config['emb_model_path'])
    init_operators(
        vdb_searcher,
        col_match_threshold=config.get('col_match_threshold', 0.95),
        embedding_model=col_match_model,
        ranker_cfg=ranker_cfg,
    )
    rel_graph = RelationshipGraph(vdb_searcher)
    tool_ctx.set_result_cap(config.get('tools', {}).get('result_cap', 20))

    res_file = args.out or os.path.join(result_dir, "tide.json")
    valid_idxs = set(range(len(queries)))
    results = []
    if os.path.exists(res_file):
        results = json.load(open(res_file, "r"))

    results = list({r["idx"]: r for r in results if r.get("idx") in valid_idxs}.values())

    agent_cfg = config.get('agent', {})
    nav_cfg = config.get('navigation', {})
    tree_cfg = config.get('tree', {})
    result_cfg = config.get('result', {})

    concurrency = args.concurrency if args.concurrency else agent_cfg.get('concurrency', 1)

    ctx = RunContext(
        vdb_searcher=vdb_searcher,
        rel_graph=rel_graph,
        lakes_dir=lakes_dir,
        input_tables_dir=input_tables_dir,
        cand_top_k=tree_cfg.get('cand_top_k', 10),
        final_top_k=result_cfg.get('final_top_k', 20),
        final_min_constraint_match=result_cfg.get('final_min_constraint_match', 1),
        per_query_timeout_s=agent_cfg.get('per_query_timeout_s', 300),
        agent_cfg=agent_cfg,
        nav_cfg=nav_cfg,
        res_file=res_file,
        results=results,
    )

    n_selected = len([i for i in range(len(queries)) if selected_idxs is None or i in selected_idxs])
    logging.info(
        "Starting run: dataset=%s queries=%d concurrency=%d log=%s",
        dataset, n_selected, concurrency, log_path,
    )

    asyncio.run(driver(queries, ctx, selected_idxs, concurrency))

    logging.info("All queries finished.")
    if _log_fp:
        _log_fp.flush()
        _log_fp.close()
