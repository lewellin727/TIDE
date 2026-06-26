# 🗂️ TIDEdataset — A Benchmark for NL Multi-Step Table Discovery

`TIDEdataset` is the benchmark generator for **TIDE: Agentic Natural Language Table Discovery over
Data Lakes**. It builds a benchmark by *expansion*: one clean source table is sliced into a DAG of
many small lake tables connected by join / union / correlation relationships, the answer path is
surrounded by look-alike distractors, and an under-specified NL query is written whose execution plan
cannot be recovered from the query text alone — the answer requires observing the lake and traversing
inter-table relationships, not a single retrieval step.

## 💡 Design principles

A fair multi-step benchmark should let queries state only the analytical need, judge relevance by
executable data conditions, and evaluate every method from the same inputs. TIDEdataset enforces:

- **Plan-Free Queries.** A query states the high-level need but not the latent plan (relation types,
  connector columns, step order); these must be inferred from lake evidence, so the task cannot reduce
  to query→plan translation.
- **Verifiable Ground-Truth.** Relevance is defined by executable data conditions (relational
  compatibility, value predicates, statistical dependence), used only to label positives and
  **schema-identical hard negatives** that fail the required condition — methods stay free to retrieve
  by any strategy.
- **Solver-Neutral Protocol.** Every method sees only the lake and the query; construction artifacts
  (routes, connector columns, labels) are withheld, so scores reflect discovery from the common input.

## 📊 Benchmark composition

5 data lakes, 38,334 tables, 261 queries; each query targets ~21 ground-truth tables hidden among
structurally similar distractors, with plans of varied topology, relationship, and depth (2–6 steps).

| Split | Source corpus | Lake tables | Queries |
|---|---|--:|--:|
| GitTable | GitHub CSV files | 18,930 | 77 |
| WebTable | Web tables | 6,148 | 64 |
| OpenData_SG | Singapore open data | 2,072 | 21 |
| OpenData_USA | US open data | 6,391 | 51 |
| OpenWikiTable | Wikipedia tables | 4,793 | 48 |
| **Total** | | **38,334** | **261** |

## 📁 Repository structure

```
process_dataset.py   raw corpora -> cleaned source tables
main.py              Phase A — slice a source table into a lake (pandas only)
query_llm.py         Phase B — write the under-specified NL query for each DAG (LLM)
generator.py         core — Plan samples topology/hops; Plan.realize(S) slices S into the DAG
utils.py             config loader + data I/O + column-name augmentation
generate.sh          one command: Phase A + Phase B -> a single log
config.yaml          generation hyper-parameters (documented inline)
tools/               run_audits.py (gate) · audit_per_query.py · audit_reachability.py · statistics.py
```

## ⚙️ Installation

```bash
conda create -n tide python=3.10 -y && conda activate tide
pip install pandas numpy pyyaml tqdm inflection openai
export DASHSCOPE_API_KEY=<your-key>      # Phase B calls an OpenAI-compatible chat API
export TIDE_LLM_MODEL=qwen3.5-plus       # optional (default)
```

## 🚀 Quick start

Raw corpora and generated splits are **not** shipped. Provide cleaned source tables under
`dataset/processed_dataset/<Source>/` (register raw dumps in `process_dataset.py`, then run it):

```bash
python process_dataset.py --dataset USA  # (one-time) raw -> cleaned source tables
python main.py                           # Phase A: structure + augment  [set `dataset:` in config.yaml]
python query_llm.py                      # Phase B: NL queries  (--fill-only resumes, --limit N smoke-tests)
python tools/run_audits.py               # generation gate
python tools/statistics.py               # dataset statistics
# or:  bash generate.sh                  # Phase A + B, tee'd to log/<dataset>.log
```

Any config value is overridable by its UPPERCASE env var, e.g. `QUERY_TABLE_COUNT=10 python main.py`.

## 🔧 How it works

`process_dataset` cleans the raw corpora into source tables. **Phase A** (`main.py`, pure pandas, no
GPU) samples a topology + hops and slices a source table into connected lake tables, retrying until
the construction yields ≥ `min_grounds` ground-truth tables; it then mirrors the lake into `aug_lakes/`
with per-query column renames (≈50% keep / 40% inflectional synonym / 10% format variant) to require
semantic column matching. **Phase B** (`query_llm.py`) writes each query with an actor–critic loop —
the LLM drafts, a deterministic critic rejects text that leaks the connector key or step order.
`tools/run_audits.py` is the generation gate, run over both `lakes/` and `aug_lakes/`.

Construction details:

- **Relationships.** A *chain* is a sequence of hops — `overlap` (shared key values), `corr` (synthetic
  `_factor`/`_ref` columns correlated across a join, so the query cannot name them), or `union` (shared
  column set); a *converge* topology combines two branches by ∩ / ∪ / \\.
- **Value filters (`eq`/`range`).** A value hop re-fills a hop's positives so some satisfy the predicate
  (kept as GT) and the rest fail (value-level look-alike negatives) — no new tables are created.
- **Hard negatives & non-self-containment.** Negatives are schema-identical look-alikes that fail the
  relationship (or value); each slice keeps only its payload + the connector onward, so GT is reachable
  only by traversal.

## 📂 Output & schema

`dataset/datalakes/<Source>/`: `aug_lakes/` (searchable lake) · `input_tables/` (optional seed tables) ·
`query.json` `[{id, query, input_table, source_idx}]` · `DAG.json` (the true plan, below) ·
`metadata.json` `[{id, grounds, path, n_tables, n_gt}]` · `aug_colmap.json` (rename map, audit only).

```jsonc
// DAG.json — the true plan, deliberately NOT recoverable from query.json[].query
{
  "id": 18, "topology": "chain",           // "chain" | "converge"
  "depth": 2, "has_seed_table": false,
  "set_op": null, "keys": null,            // converge only
  "start_concepts": ["State_"],
  "hops": [
    {"step": 1, "type": "overlap", "join_key": "State_",
     "value_filter": {"column": "Rank_", "predicate": "range", "target": 76.0}},
    {"step": 2, "type": "union", "terminal": true, "union_columns": ["Land_area", "Population_density"]}
  ]
  // corr hops also carry "correlation": {"measured": "_factor2", "reference": "_ref2"}
}
```

## 🎛️ Configuration

All knobs live in `config.yaml`, documented inline and overridable by UPPERCASE env vars. The main
ones: `dataset`; `query_table_count` / `distractor_table_count` (source-table budget); `min_hops` /
`max_hops` (chain depth); `p_corr_hop` / `p_union_hop` / `p_combiner` (topology mix); `p_value`
(value-constraint rate); `pos_split_cnt` / `neg_split_cnt` (per-node positives / negatives); `seed`.

After regenerating a split, delete downstream caches built from the old lake (e.g. `vdb/<split>_*`).
