# 🗂️ TIDEdataset — A Benchmark for NL Multi-Step Table Discovery

`TIDEdataset` is the benchmark generator accompanying **TIDE: Agentic Natural Language Table
Discovery over Data Lakes**. Given a natural-language request (optionally with a seed table), the
task is to retrieve a set of ground-truth tables from a large data lake whose answer **cannot be
reached in a single retrieval step** — it requires observing the lake and traversing inter-table
relationships.

The generator builds a benchmark by **expansion**: it takes one clean source table, slices it into a
DAG of many small lake tables connected by join / union / correlation relationships, surrounds the
answer path with look-alike distractors, and writes an under-specified NL query whose execution plan
is deliberately *not* recoverable from the query text alone.

## 💡 Why this benchmark

Existing NL table-discovery benchmarks are single-step: the relevant tables can be matched directly
from the query. TIDEdataset is built so that two properties hold by construction:

- **C1 — the plan is not in the query.** Which relationship to follow, on which key, in what order is
  data-dependent and only discoverable by inspecting the lake. A query→plan translator cannot solve
  the task.
- **C2 — constraint-level verification is required.** Each answer is surrounded by **look-alike
  negatives** that share its structure but fail a value/relationship predicate, so single-step dense
  retrieval is insufficient.

Two design choices keep these honest:

- **Step-positive ground truth.** The ground truth is the union of every value-step's positives and
  the terminal positives — each step the true plan reaches counts as answered, while later hops still
  drop upstream columns so they remain reachable only by traversal.
- **Anti-circularity.** The benchmark is defined purely over *data properties* (relationships
  {overlap, corr, union}, value constraints {eq, range}, topology {chain, converge}) and never over
  any solver's operators or thresholds. Difficulty (e.g. column-name augmentation) is set by
  independent linguistic/structural rules.

## 📊 Benchmark composition

The released benchmark spans 5 data lakes (38,334 tables) and 261 queries. Each split is generated
from a different source corpus:

| Split | Source corpus | Lake tables | Queries |
|---|---|--:|--:|
| GitTable | GitHub CSV files | 18,930 | 77 |
| WebTable | Web tables | 6,148 | 64 |
| OpenData_SG | Singapore open data | 2,072 | 21 |
| OpenData_USA | US open data | 6,391 | 51 |
| OpenWikiTable | Wikipedia tables | 4,793 | 48 |
| **Total** | | **38,334** | **261** |

Each query targets ~21 ground-truth tables on average, hidden among structurally similar distractors,
with plans of varied topology, relationship type, and depth (2–6 steps).

## 📁 Repository structure

```
TIDEdataset/
├── process_dataset.py     # raw corpora  -> cleaned source tables
├── main.py                # Phase A: sample + realize a DAG plan into a lake (pandas only)
├── query_llm.py           # Phase B: write the under-specified NL query for each DAG (LLM)
├── generator.py           # core: Plan samples topology/hops; Plan.realize(S) slices S into the DAG
├── utils.py               # config loader + data I/O + column-name augmentation
├── generate.sh            # one command: Phase A + Phase B -> a single log
├── config.yaml            # all generation hyper-parameters
└── tools/
    ├── run_audits.py        # generation gate: runs both audits on lakes/ and aug_lakes/
    ├── audit_per_query.py   # per-query correctness (positives/negatives/values/GT)
    ├── audit_reachability.py# structural: every GT is reachable only by the true plan
    └── statistics.py        # dataset-level statistics (paper table format)
```

Generated data lands under `dataset/` (not shipped; see below).

## ⚙️ Installation

```bash
conda create -n tide python=3.10 -y && conda activate tide
pip install pandas numpy pyyaml tqdm inflection openai
```

Phase B calls an OpenAI-compatible chat API. Set the key (and optionally the model) in the
environment — they are read only at call time:

```bash
export DASHSCOPE_API_KEY=<your-key>
export TIDE_LLM_MODEL=qwen3.5-plus        # optional; this is the default
```

## 🗄️ Data

Raw corpora and generated splits are **not** included in this repository. To build a split you need
the cleaned source tables under `dataset/processed_dataset/<Source>/`:

- If you have the raw dumps, register them in `process_dataset.py` (`DATASETS`) and run
  `python process_dataset.py --dataset <key>` to clean them into `processed_dataset/`.
- Cleaning keeps only tables with enough rows/columns and **both** numeric and categorical columns,
  so relationships and value constraints are expressible.

```
dataset/
├── raw_data/                       # raw corpora (provide yourself)
├── processed_dataset/<Source>/     # cleaned source tables (process_dataset.py output)
├── datalakes/<Source>/             # a generated benchmark split (main.py + query_llm.py output)
└── statistics/<Source>.json        # statistics.py output
```

## 🚀 Quick start

```bash
# 0. (one-time) clean raw corpora into source tables
python process_dataset.py --dataset USA

# 1. Phase A — structure: slice source tables into a lake, augment columns, (audit)   [pandas, no GPU]
python main.py                       # set `dataset:` in config.yaml first

# 2. Phase B — language: write the under-specified NL queries                          [LLM]
python query_llm.py                  # resume with --fill-only; smoke-test with --limit 10

# 3. audit + statistics
python tools/run_audits.py
python tools/statistics.py
```

Or run Phase A + Phase B together, tee'd into one log:

```bash
bash generate.sh                     # -> log/<dataset>.log
```

`main.py` reads `dataset:` from `config.yaml`; any hyper-parameter can be overridden by the same-name
UPPERCASE env var, e.g. `QUERY_TABLE_COUNT=10 python main.py` for a quick test.

## 🔧 Generation pipeline

```
raw corpora
   │  process_dataset.py
   ▼
dataset/processed_dataset/<Source>/*.csv
   │  main.py  ── Phase A ──  (generator.py: sample a Plan, realize it into a DAG of slices)
   ▼
dataset/datalakes/<Source>/
   ├── lakes/                 the lake: GT-path slices + per-query negatives + noise_* distractors
   ├── input_tables/          seed tables query_table_<qid>.csv (only for has_seed queries)
   ├── DAG.json               the structural plan (Phase A→B handoff; `query` still empty)
   └── metadata.json          ground-truth table stems + full construction record
   │  utils.augment_tables    mirror the lake with per-query column renames
   ▼
   ├── aug_lakes/             column-renamed mirror — the DEPLOYMENT lake the method searches
   └── aug_colmap.json        per-query rename map (ground truth, used only by the audit)
   │  query_llm.py  ── Phase B ──  (LLM writes the under-specified NL query into query.json)
   ▼
   query.json                 [{id, query, input_table, source_idx}]
   │  tools/run_audits.py     two-layer audit on lakes/ and aug_lakes/ — the generation gate
```

**Phase A is pure pandas (no LLM, no GPU).** For each source table it samples a topology + hops,
slices the table into connected lake tables, and retries until the construction yields at least
`min_grounds` ground-truth tables. **Phase B** turns each DAG into a natural-language query with an
actor–critic loop (the LLM writes; a deterministic critic rejects any text that leaks the specific
connector key or step order). **Augmentation** mirrors the lake into `aug_lakes/` with per-query
injective column renames (≈50% keep / 40% inflectional synonym / 10% format variant) to require
semantic column matching; this mirror is what a discovery method is actually run against.

### How a lake is constructed

- **Relationship hops.** A *chain* is a sequence of hops; a hop is `overlap` (share key values),
  `corr` (a numeric column correlates with another across a join — the correlated columns are
  *synthetic* `_factor`/`_ref` columns so the query cannot name them), or `union` (share a column
  set). A *converge* topology runs two branches combined by a set op (∩ / ∪ / \\).
- **eq/range are value rankers, not table splits.** A value-constrained hop produces a set of
  positives from the same connected population, then re-fills the value column so some satisfy the
  predicate (kept as GT) and the rest fail (becoming value-level look-alike negatives). No new
  tables are created — the value is a fine-grained discriminator within the relationship-positive set.
- **Negatives.** Per node: relationship negatives (union → lacks the union columns; corr →
  joinable-but-uncorrelated; overlap → same schema but not joinable) plus the value-level negatives.
- **Non-self-containment.** Each slice keeps only its own payload plus the connector to the next hop;
  upstream columns are projected away, so the ground truth must be reached by traversal.

## 📂 Output layout & schema

`dataset/datalakes/<Source>/`:

| path | contents |
|---|---|
| `lakes/` | the lake — `<qid>_<n>.csv` (GT-path slices + per-query negatives) and `noise_<src>_<n>.csv` (distractor-only tables, never in any grounds) |
| `aug_lakes/` | column-renamed mirror of `lakes/` (the deployment lake) |
| `input_tables/` | seed tables `query_table_<qid>.csv` (only for `has_seed_table` queries) |
| `query.json` | `[{id, query, input_table, source_idx}]` (query empty until Phase B) |
| `DAG.json` | per-query structural plan (schema below) |
| `metadata.json` | `[{id, grounds:[<stem>.csv,...], path, n_tables, n_gt, source_idx}]` |
| `aug_colmap.json` | `{qid: {raw_col: aug_col}}` — rename ground truth (audit only) |

`DAG.json` is the **true plan**; the benchmark's premise is that it is *not* recoverable from
`query.json[].query`:

```jsonc
{
  "id": 18,
  "topology": "chain",              // "chain" | "converge"
  "depth": 2,
  "has_seed_table": false,          // true -> a seed table T_q is provided
  "set_op": null,                   // "intersection" | "union" | "difference" (converge only)
  "keys": null,                     // [Ka, Kb] branch keys (converge only)
  "start_concepts": ["State_"],     // entry columns the query starts from
  "hops": [
    { "step": 1, "type": "overlap", "terminal": false,
      "join_key": "State_",
      "value_filter": {"column": "Rank_", "predicate": "range", "target": 76.0} },
    { "step": 2, "type": "union", "terminal": true,
      "union_columns": ["Incorporated_place", "Land_area_mi_2_", "Population_density"] }
  ]
}
// corr hops additionally carry "correlation": {"measured": "_factor2", "reference": "_ref2"}
```

## 🎛️ Configuration

All hyper-parameters live in `config.yaml`; any value is overridable by the same-name UPPERCASE env
var (e.g. `QUERY_TABLE_COUNT=10 python main.py`).

| key | meaning |
|---|---|
| `dataset` | source under `processed_dataset/`; output split → `datalakes/<dataset>/` |
| `query_table_count` | first N source tables generate QUERIES |
| `distractor_table_count` | next M source tables become DISTRACTOR-only tables (no queries) |
| `distractor_slices_per_table` | random slices produced per distractor source table |
| `seed` | RNG seed (reproducible) |
| `queries_per_table` | queries per source table |
| `partition_retries` | re-plan attempts until `grounds >= min_grounds` |
| `pos_split_cnt` / `neg_split_cnt` | positives / relationship-negatives per node |
| `min_cols` / `min_grounds` | min columns per emitted table / min GT to keep a query |
| `min_hops` / `max_hops` | chain depth bounds |
| `p_query_table` | P(query ships a seed table T_q vs NL-only) |
| `p_corr_hop` / `p_union_hop` / `p_combiner` | P(corr edge) / P(union hop) / P(converge topology) |
| `p_value` / `value_pos_cnt` | P(hop carries an eq/range value) / positives a value hop produces |
| `min_corr_groups` / `anchor_k` / `anchor_core_k` / `intersection_gt` | corr group floor / joinable key-set size / forced-shared key count / GT count for ∩ |

After regenerating a split, delete any downstream caches built from the old lake (e.g. the method's
vector-index cache under `vdb/<split>_*`).
