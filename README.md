# 🌊 TIDE: Agentic Natural Language Table Discovery over Data Lakes

Official implementation of **TIDE**, an agentic framework for natural-language (NL) table discovery
over data lakes. TIDE reformulates discovery as a **lake-grounded, multi-step retrieval process**: an
LLM agent iteratively reasons over intermediate tables and data-dependent lake signals (schemas,
shared keys, value overlap, statistical dependence) to resolve high-level queries whose execution
plan cannot be inferred from the query text alone.

TIDE couples two components:

- **Constraint-level operators** — a composable action space of *seekers* (ground NL descriptors),
  *rankers* (verify value/relational predicates), and *combiners* (compose partial results), under a
  uniform binding-level contract. Classical join / union / correlation / NL search are recovered as
  fine-grained compositions of these operators.
- **Lake-grounded tree reasoning** — a constraint-guided discovery tree whose expansion is grounded
  in a precomputed table relationship graph via relationship-aware observation, then advanced by an
  observe–plan–execute loop with verified constraint updates.

On a new benchmark of 5 data lakes (38,334 tables, 261 queries), TIDE outperforms 5 state-of-the-art
baselines by up to **67% in precision** and **53% in recall**.

> The benchmark generator lives in [`TIDEdataset/`](TIDEdataset/) — see its README to build the data.

## 🧭 Method at a glance

```
                       ┌──────────────────────── offline (per lake) ───────────────────────┐
   data lake  ───────► │ column content/context indexes (HNSW) · per-column sketches        │
                       │ (Count-Min / equi-depth hist / KMV) · table relationship graph G_R │
                       └────────────────────────────────────────────────────────────────────┘
                                                   │
   NL query  ─► parse constraints s_q ─►  ┌──────── online: constraint-guided discovery tree ────────┐
                                          │  select node → relationship-aware observation (reliable + │
                                          │  novel paths over G_R) → plan operators → execute &       │
                                          │  verify constraints → expand tree                         │  ─►  top-k tables
                                          └────────────────────────────────────────────────────────────┘
```

The offline phase precomputes shared artifacts once; the online phase greedily expands the highest-
utility leaf each iteration until every branch terminates or the budget is exhausted, then aggregates
candidate tables from all non-error leaves.

## 📁 Repository structure

```
TIDE/
├── main.py                    # entry point: async per-query driver (offline build + online discovery)
├── eval.py                    # Precision@k / Recall@k / NDCG@k evaluation
├── config.yaml                # all runtime knobs (paths, thresholds, agent loop, navigation)
├── requirements.txt
├── src/
│   ├── agent.py               # TableAgent — the observe–plan–execute loop (Algorithm 1, online)
│   ├── reasoning_tree.py      # constraint-guided discovery tree (nodes, utility, result aggregation)
│   ├── relationship_graph.py  # table relationship graph G_R + reliability-based path search
│   ├── tgr.py                 # novelty-based path selection (submodular rerank) + path rendering
│   ├── query_constraint.py    # parse the query into structured column constraints s_q
│   ├── operators/
│   │   ├── seeker.py          # content / context seekers (ground NL descriptors)
│   │   ├── ranker.py          # eq / range / overlap / corr rankers (verify predicates)
│   │   └── tool.py            # the agent's action space (operators + combiner + terminate)
│   ├── col_filter.py          # per-column sketches: Count-Min, equi-depth histogram, KMV
│   ├── col_formatter.py       # column content / context serialization for embedding
│   ├── col_name_matcher.py    # semantic column-name matching (bridges renamed columns)
│   ├── vdb_searcher.py        # FAISS HNSW vector index over column/metadata embeddings
│   ├── table.py               # Table abstraction + binding fields ⟨matched column, predicate⟩
│   └── utils.py               # operator wiring, prompt templates, token accounting
└── TIDEdataset/               # benchmark generator (separate README)
```

## ⚙️ Installation

```bash
conda create -n tide python=3.10 -y && conda activate tide
pip install -r requirements.txt
```

Key dependencies: `agentscope` (agent framework), `faiss-cpu` (vector search), `sentence-transformers`
+ `transformers` + `torch` (embeddings), `openai` (LLM API), `pandas` / `numpy` / `scipy`.

**Embedding model.** TIDE embeds columns with `bge-large-en-v1.5`. `config.yaml` defaults to the
HuggingFace id `BAAI/bge-large-en-v1.5` (auto-downloaded); set it to a local path to run offline.

**LLM backend.** The agent calls an OpenAI-compatible chat endpoint (the paper uses Qwen3.6-Max and
DeepSeek-V4-Pro via Alibaba DashScope). Provide your key:

```bash
export DASHSCOPE_API_KEY=<your-key>
```

The model name and endpoint are set in [`src/agent.py`](src/agent.py) (planner/observer) and
[`src/utils.py`](src/utils.py) (constraint parser); change them there to use a different backbone.

## 🗄️ Data

TIDE runs on benchmark splits produced by [`TIDEdataset/`](TIDEdataset/). Point `config.yaml` at them:

```yaml
datalake_dir: TIDEdataset/dataset/datalakes   # parent of the generated splits
dataset: WebTable                             # the split to run on
```

Each split provides `aug_lakes/` (the searchable lake), `input_tables/` (optional seed tables),
`query.json`, and `metadata.json` (ground truth, used only for evaluation). To run on your own data
lake, match that layout.

## 🚀 Quick start

```bash
# 1. configure: set dataset + paths in config.yaml, and export DASHSCOPE_API_KEY

# 2. run discovery (first run also builds the indexes/graph cache under vdb_dir)
python main.py --dataset WebTable                 # -> results/WebTable/tdagent.json

#    options: --idxs 0,1,2 (subset)  --concurrency N  --out <path>  --log <path>

# 3. evaluate against ground truth
python eval.py --dataset WebTable                 # -> eval/WebTable/tdagent.json
```

`main.py` writes one record per query as `{idx, query, candidates}`; `eval.py` reports
Precision@k, Recall@k, and NDCG@k against each split's `metadata.json`.

## 🔍 How it works

**Constraint-level operators (paper §4).** Every operator returns a binding-level result set of
bindings ⟨host table, matched column, satisfied predicate, score⟩:

- *Seekers* ([`seeker.py`](src/operators/seeker.py)) ground an NL descriptor to candidate columns via
  two complementary signals — content (name + values) and context (description) — each embedded and
  indexed with HNSW ([`vdb_searcher.py`](src/vdb_searcher.py)).
- *Rankers* ([`ranker.py`](src/operators/ranker.py)) verify a predicate over candidates:
  `eq` (Count-Min), `range` (equi-depth histogram), `overlap` and `corr` (KMV sketch) —
  all in [`col_filter.py`](src/col_filter.py).
- *Combiners* ([`tool.py`](src/operators/tool.py)) compose partial results with set operations
  (intersection / union / difference).

**Lake-grounded tree reasoning (paper §5).** The agent ([`agent.py`](src/agent.py)) maintains a
constraint-guided discovery tree ([`reasoning_tree.py`](src/reasoning_tree.py)). Each iteration:

1. selects the highest-utility incomplete leaf;
2. runs **relationship-aware observation** — reliability-based path search over the precomputed
   relationship graph ([`relationship_graph.py`](src/relationship_graph.py)) followed by
   novelty-based, submodular path selection ([`tgr.py`](src/tgr.py)) — and summarizes the paths into
   lake-grounded evidence;
3. **plans** the next operator(s) from that evidence, then **executes** them, expanding child nodes
   and discharging query constraints only when a returned binding verifies them.

## 🎛️ Configuration

All knobs live in [`config.yaml`](config.yaml); paths are relative to the repo root.

| key | meaning |
|---|---|
| `datalake_dir`, `dataset` | parent dir of splits, and the split to run |
| `vdb_dir` | cache for vector indexes + sketches (built on first run) |
| `emb_model_path` | embedding model (HF id or local path) |
| `col_match_threshold`, `vdb_score_threshold` | cosine floors for column-name matching / vector search |
| `ranker.*` | correlation sign mode / threshold, numerical-predicate fraction |
| `agent.max_loop`, `agent.per_query_timeout_s`, `agent.concurrency` | planning budget, per-query timeout, query concurrency |
| `tree.cand_top_k` | per-node candidate cap shown to the planner |
| `tools.result_cap` | max tables returned by one operator call |
| `result.final_top_k`, `result.final_min_constraint_match` | output cap, per-table constraint filter |
| `navigation.k_1`, `navigation.k_o`, `navigation.d_o` | reliable path pool, paths surfaced to observation, BFS depth |

