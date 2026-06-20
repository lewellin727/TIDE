
import os
import re
import sys
import json
import argparse
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import cfg


def llm_generate(prompt, return_tokens=False, model=None):

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60.0,       
        max_retries=3,      
    )
    completion = client.chat.completions.create(
        model=model or os.getenv("TIDE_LLM_MODEL", "qwen3.5-plus"),
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
        stream=False,
    )
    u = completion.usage
    tokens = {"in": u.prompt_tokens, "out": u.completion_tokens, "total": u.total_tokens}
    res = completion.choices[0].message.content
    return (res, tokens) if return_tokens else res


_FORBIDDEN_OP = [
    r"\bshared?\s+(key|column|identifier|field|attribute)s?\b",
    r"\bcommon\s+(key|column|identifier|field)s?\b",
    r"\bsame\s+(keys?|schema|columns?|fields?)\b",
]
# Mechanical sequencing / DAG meta-vocabulary.
_FORBIDDEN_META = [
    r"\bfirst\b[^.]*\bthen\b", r"\bstep\s+(\d+|one|two|three|four|five)\b",
    r"\b(dag|node|operator|execution\s+plan|pipeline)\b",
]


_SANITIZE = [
    (re.compile(r"\bcorrelat(?:es|ed|ing)?\s+with\b", re.I), "appears to influence"),
    (re.compile(r"\b(?:moves?|vary|varies|tracks?|trends?|trending)\s+(?:with|together|alongside)\b", re.I), "appears to influence"),
    (re.compile(r"\bcorrelation(?:s)?\b", re.I), "apparent influence"),
    (re.compile(r'"_(?:ref|factor)[A-Za-z0-9_]*"|\b_(?:ref|factor)[A-Za-z0-9_]*\b', re.I), "a numeric indicator/metric"),
]


def _sanitize(q: str) -> str:
    for pat, repl in _SANITIZE:
        q = pat.sub(repl, q)
    return q


def _hop_value_targets(dag):
    out = []
    for h in dag.get("hops", []):
        vf = h.get("value_filter")
        if vf and vf.get("target") is not None:
            out.append(str(vf["target"]))
    return out


def critique(query, dag):

    q = query or ""
    ql = q.lower()
    issues = {}

    op = sorted({m.group(0) for pat in _FORBIDDEN_OP for m in re.finditer(pat, ql)})
    if op:
        issues["operator/relational vocabulary"] = (
            f"remove words that name the operation: {op}. Describe the relationship vaguely "
            f"(e.g. 'related to', 'associated with', 'that influence') so it could be a value "
            f"match, a statistical dependence, or a shared structure — the reader must inspect "
            f"the data to know which.")

    meta = sorted({m.group(0) for pat in _FORBIDDEN_META for m in re.finditer(pat, ql)})
    if meta:
        issues["mechanical/meta phrasing"] = f"remove step-enumeration / DAG meta words: {meta}"

    synth = sorted(set(re.findall(r"\b_(?:ref|factor)[A-Za-z0-9_]*\b", q)))
    if synth:
        issues["synthetic placeholder leaked"] = (
            f"do not mention synthetic placeholder columns {synth}; refer to such a synthesized "
            f"quantity generically as 'a numeric indicator/metric'.")

    def _is_num(t):
        try:
            float(str(t).replace(",", "")); return True
        except Exception:
            return False
    num_targets = [t for t in _hop_value_targets(dag) if _is_num(t)]
    if num_targets:
        nums = re.findall(r"\d[\d,.]*", q)
        if len(nums) < len(num_targets):
            issues["value omitted"] = (
                f"this query has {len(num_targets)} NUMERIC value condition(s) but only {len(nums)} "
                f"number(s) appear — state the concrete threshold for each (e.g. 'below 5000', "
                f"'less than 206'), never 'a certain range' / 'a specific threshold' / 'around'.")

    return (len(issues) == 0), issues


# ===================================================================================================================
# Actor — DAG → under-specified NL
# ===================================================================================================================
ACTOR_PROMPT = """# Role
You are a data analyst telling a colleague what tables you need from a large data lake. You
state your information NEED in plain language — never how a system should retrieve it.

# You receive (for YOUR understanding only)
One query's structure (JSON): `start_concepts`, ordered `hops` (each may carry a `value_filter`);
or for `topology: converge`, two branches joined by `set_op`.

# REVEAL — keep the request findable
- The start tables (`start_concepts`) in full, and the final attributes you want.
- Name every REAL column by its EXACT name in DOUBLE QUOTES, copied verbatim, never paraphrased
  — e.g. "Population_thousands", "_2001_Census". EXCEPTION: synthetic columns named `_ref…` /
  `_factor…` have no real name → call them "a numeric indicator/metric" (never quote them).
- EVERY value condition with its CONCRETE number/category, phrased to MATCH the filter's
  `predicate`: a `range` filter means the value is BELOW its target → write 'where "Population"
  is below/less than 5000' (NEVER "around" — a range is one-sided, not a neighbourhood); an `eq`
  filter is equality → write 'for "Team" = Ferrari'. NEVER write "a certain range/threshold".

# Keep it LIGHTLY under-specified (don't hand over the exact retrieval plan, but stay natural)
- Write ONE flowing request ("for those…", "and among those…"), not numbered "first… then…" steps.
- Do NOT spell out the SPECIFIC connector by name — never write "the shared key / common column /
  same column is X". Refer to the relationship by meaning, not by naming the exact bridge column.
- You MAY use natural relationship wording — "related to", "associated with", "that also include",
  "that influence / are correlated with", "but not". The KIND of relationship may read through;
  only the exact connector column and a literal step list must stay implicit.

# Converge (phrase naturally)
intersection → "…that are both … and …"; union → "…either … or …"; difference → "… but not …".

# Example
structure: start={{"Regional_sales","Region"}}; then a correlated factor where "Sales_2008" is below 5000 (range filter).
query: "I'm looking at "Regional_sales" by "Region" and want other sources that could help explain
those sales — records tied to the same regions carrying an attribute that appears to influence
"Sales_2008", focusing on where it is below 5000."

# Output
Return ONLY a JSON object, no prose, no markdown:
{{"query": "..."}}

# Structure
{DAG}
"""


def generate_query(dag, max_retries=6):

    brief = json.dumps(dag, ensure_ascii=False, separators=(",", ":"))
    feedback = ""
    stats = {"attempts": 0, "tokens": 0, "pass_attempt": None, "last_issues": []}
    last = ""
    for attempt in range(1, max_retries + 1):
        stats["attempts"] = attempt
        prompt = ACTOR_PROMPT.format(DAG=brief)
        if feedback:
            prompt += (f"\n\n# Your previous attempt was REJECTED\nprevious query: {last}\n"
                       f"fix these issues and rewrite:\n{feedback}")
        resp, tok = llm_generate(prompt, return_tokens=True)
        stats["tokens"] += tok["total"]
        try:
            obj = json.loads(resp[resp.find("{"):resp.rfind("}") + 1])
            last = _sanitize((obj.get("query") or "").strip())
            if not last:
                raise ValueError("no 'query' field")
        except (json.JSONDecodeError, ValueError):
            feedback = "output must be a JSON object {\"query\": \"...\"}."
            continue
        ok, issues = critique(last, dag)
        stats["last_issues"] = sorted(issues.keys())
        if ok:
            stats["pass_attempt"] = attempt
            return last, stats
        feedback = "\n".join(f"- {k}: {v}" for k, v in issues.items())
    return None, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default=cfg("output_suffix", ""))
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries (smoke test)")
    ap.add_argument("--fill-only", action="store_true",
                    help="skip queries that already have a non-empty NL (gap-fill; re-run to "
                         "converge on stochastic critic rejections)")
    a = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(root, "dataset", "datalakes", f"{cfg('dataset')}{a.suffix}")
    dag_path = os.path.join(d, "DAG.json")
    q_path = os.path.join(d, "query.json")
    dags = {x["id"]: x for x in json.load(open(dag_path))}
    queries = json.load(open(q_path))

    todo = [q for q in queries if (a.limit == 0 or q["id"] < a.limit)]
    tot_tok = 0; ok = 0; failed = []; rows = []
    for q in todo:
        dag = dags.get(q["id"])
        if dag is None:
            continue
        if a.fill_only and (q.get("query") or "").strip():
            ok += 1
            continue
        try:
            query, st = generate_query(dag)
        except Exception as e:                      # one query's API failure must not kill the run
            failed.append(q["id"])
            print(f"[Q{q['id']}] ERROR after retries: {type(e).__name__}: {e}", flush=True)
            continue
        tot_tok += st["tokens"]
        rows.append({"id": q["id"], "topology": dag["topology"], "attempts": st["attempts"],
                     "pass_attempt": st["pass_attempt"], "tokens": st["tokens"],
                     "issues": st["last_issues"], "query": query})
        if query:
            q["query"] = query; ok += 1
            print(f"[Q{q['id']}] ({dag['topology']}) attempt {st['pass_attempt']}/{st['attempts']} tok={st['tokens']}  {query}", flush=True)
        else:
            failed.append(q["id"])
            print(f"[Q{q['id']}] FAILED after {st['attempts']} (last issues: {st['last_issues']})", flush=True)
        json.dump(queries, open(q_path, "w"), indent=4, ensure_ascii=False)

    json.dump(queries, open(q_path, "w"), indent=4, ensure_ascii=False)
    print(f"\n[PHASE B] filled={ok}/{len(todo)}  failed={failed}  tokens={tot_tok}")
    print(f"[PHASE B] wrote -> {q_path}")

    from collections import Counter
    print(f"[PHASE B] model={os.getenv('TIDE_LLM_MODEL', 'qwen3.5-plus')} "
          f"attempts_dist={dict(sorted(Counter(r['attempts'] for r in rows).items()))}")


if __name__ == "__main__":
    main()
