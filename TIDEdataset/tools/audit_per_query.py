import os, sys, json, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cfg

POS = cfg("pos_split_cnt")
NEG = cfg("neg_split_cnt")
INTERSECTION_GT = cfg("intersection_gt")
_cache = {}
_REV = {}        
_LAKE = None     


def load(d, name):

    key = (d, name)
    if key not in _cache:
        fp = os.path.join(d, name)
        df = pd.read_csv(fp, low_memory=False) if os.path.exists(fp) else None
        if df is not None and _REV and d == _LAKE:
            rev = _REV.get(name.split("_")[0])
            if rev:
                df = df.rename(columns=rev)
        _cache[key] = df
    return _cache[key]


def svals(df, col):
    return set(df[col].dropna().astype(str)) if (df is not None and col is not None and col in df.columns) else set()


def ok(b):
    return "✓" if b else "✗"


def _start_pos(start, m, lake, inp):
    if start["kind"] == "query_table":
        return [load(inp, f"query_table_{m['id']}.csv")]
    return [load(lake, n) for n in start["pos"]]


def audit_chain(m, lake, inp):
    p = m["path"]; D = p["depth"]; edges = p["edges"]; grounds = m["grounds"]
    fails = []

    def hlabel(e):
        base = ("un[%s]" % ",".join(e.get("union_cols") or [])) if e["type"] == "union" \
            else (("corr[%s]" if e["type"] == "corr" else "ov[%s]") % e["key"])
        if e.get("constraint"):
            c = e["constraint"]
            base += "+%s(%s%s%s)" % (c["kind"], c["col"], "=" if c["kind"] == "eq" else "<", c["target"])
        return base

    tv = edges[-1].get("constraint")
    ops = "→".join(hlabel(e) for e in edges)
    print(f"\n[Q{m['id']}] CHAIN  depth={D}  has_tq={p['has_tq']}  {ops}" +
          (f"  +val({tv['col']}{'='if tv['kind']=='eq' else '<'}{tv['target']})" if tv else ""))
    print(f"   query: {(m.get('_query') or '(Phase A — DAG only; NL pending)')[:140]}")

    prev_pos = _start_pos(p["start"], m, lake, inp)

    def _passes(df, cj):
        col = cj["col"]
        if df is None or col not in df.columns:
            return None
        if cj["kind"] == "eq":
            return bool((df[col].astype(str) == str(cj["target"])).any())      # target present
        return bool((df[col].astype(float) < cj["target"]).all())              # all in range

    for i, e in enumerate(edges, start=1):
        pos, neg, vneg = e["pos"], e["neg"], e.get("value_neg") or []
        cj = e.get("constraint")
        c_pos = (len(pos) == POS); c_neg = (len(neg) == NEG + len(vneg))         # rel negs + value negs
        pt = load(lake, pos[0]) if pos else None
        nt = load(lake, neg[0]) if neg else None                                # neg[0] = a RELATIONSHIP neg
        pcols = set(pt.columns) if pt is not None else set()
        ncols = set(nt.columns) if nt is not None else set()
        upcols = set().union(*[set(t.columns) for t in prev_pos if t is not None]) if prev_pos else set()
        # --- relationship check (overlap / corr / union) ---
        if e["type"] == "union":
            U = set(e.get("union_cols") or [])
            c_link = U.issubset(pcols) and U.issubset(upcols)
            c_negfail = not U.issubset(ncols)
            c_schema = c_negfail
            lbl = "negDiffCols"
        else:
            jk = e["key"]
            c_schema = (pcols == ncols and len(pcols) > 0)
            upv = set().union(*[svals(t, jk) for t in prev_pos if t is not None])
            c_link = bool(upv & svals(pt, jk))
            c_negfail = bool(nt is not None and not (svals(nt, jk) & upv)) if e["type"] == "overlap" \
                else bool(nt is not None and jk in nt.columns)
            lbl = "sameSchema"
        # --- value check (eq/range ranker on this hop, if any): pos satisfies, value_neg fails ---
        c_posval = c_valneg = True
        if cj:
            c_posval = (_passes(pt, cj) is True)
            vt = load(lake, vneg[0]) if vneg else None
            c_valneg = (_passes(vt, cj) is False)
        all_ok = c_pos and c_neg and c_schema and c_link and c_negfail and c_posval and c_valneg
        vstr = f" posVal{ok(c_posval)} valNeg{ok(c_valneg)}" if cj else ""
        print(f"   edge{i} {hlabel(e)} pos={len(pos)}{ok(c_pos)} neg={len(neg)}{ok(c_neg)} "
              f"{lbl}{ok(c_schema)} linked{ok(c_link)} negFails{ok(c_negfail)}{vstr}" + ("" if all_ok else "  <-- CHECK"))
        checks = [("pos_cnt", c_pos), ("neg_cnt", c_neg), ("schema", c_schema), ("link", c_link), ("negfail", c_negfail)]
        if cj:
            checks += [("posval", c_posval), ("valneg", c_valneg)]
        for nm, c in checks:
            if not c:
                fails.append(f"edge{i}.{nm}")
        prev_pos = [load(lake, n) for n in pos]

    def _stem(n):
        return os.path.splitext(os.path.basename(n))[0]
    expected = set()
    for e in edges:
        if e.get("constraint") or e is edges[-1]:          # value hops + terminal
            expected |= {_stem(n) for n in e["pos"]}
    got = {_stem(g) for g in grounds}
    gtn = len(got)
    c_gtn = (gtn >= 5)
    c_optionA = (got == expected)
    extra, missing = got - expected, expected - got
    print(f"   GT (Option A): n={gtn}{ok(c_gtn)}  grounds==∪(value-hop+terminal positives){ok(c_optionA)}"
          + ("" if c_optionA else f"  missing={len(missing)} extra={len(extra)}"))
    for nm, c in [("gt_count", c_gtn), ("gt_optionA", c_optionA)]:
        if not c:
            fails.append(nm)
    return fails


def audit_converge(m, lake, inp):
    p = m["path"]; op = p["set_op"]; Ka, Kb = p["keys"]; grounds = m["grounds"]
    fails = []
    eA = p["edges"][0]
    print(f"\n[Q{m['id']}] CONVERGE {op}  keys=({Ka},{Kb})  has_tq={p['has_tq']}"
          + (f"  corrA[{eA['payload']}~{eA['ref']}]" if eA["type"] == "corr" else ""))
    print(f"   query: {(m.get('_query') or '(Phase A — DAG only; NL pending)')[:140]}")
    start = _start_pos(p["start"], m, lake, inp)
    st = start[0] if start else None
    va, vb = svals(st, Ka), svals(st, Kb)
    expect = {"intersection": INTERSECTION_GT, "difference": POS, "union": 2 * POS}[op]
    c_cnt = (len(grounds) == expect)
    good = gtn = 0
    for g in grounds:
        t = load(lake, g)
        if t is None:
            continue
        gtn += 1
        ina = bool(svals(t, Ka) & va); inb = bool(svals(t, Kb) & vb)
        if (op == "intersection" and ina and inb) or \
           (op == "difference" and ina and not inb) or \
           (op == "union" and (ina or inb)):
            good += 1
    c_join = (good == gtn)
    print(f"   GT: n={gtn} (expect {expect}){ok(c_cnt)}  join-correct={good}/{gtn}{ok(c_join)}")
    if not c_cnt:
        fails.append("gt_count")
    if not c_join:
        fails.append("gt_join")
    return fails


def main():
    global _REV, _LAKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--lake", default="lakes", choices=["lakes", "aug_lakes"])
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, f"dataset/datalakes/{cfg('dataset')}{a.suffix}")
    lake = os.path.join(d, a.lake); inp = os.path.join(d, "input_tables")
    _LAKE = lake
    if a.lake == "aug_lakes":
        cm = json.load(open(os.path.join(d, "aug_colmap.json")))
        _REV = {qid: {aug: orig for orig, aug in m.items()} for qid, m in cm.items()}
        print(f"[auditing aug_lakes — reverse-renaming columns via aug_colmap.json]")
    meta = json.load(open(os.path.join(d, "metadata.json")))
    qtext = {q["id"]: q["query"] for q in json.load(open(os.path.join(d, "query.json")))}
    for m in meta:
        m["_query"] = qtext.get(m["id"], "")

    all_fail = {}
    for m in meta:
        f = audit_converge(m, lake, inp) if m["path"]["topology"] == "converge" else audit_chain(m, lake, inp)
        if f:
            all_fail[m["id"]] = f
    print("\n" + "=" * 70)
    print(f"[AGGREGATE] {len(meta)} queries | fully-pass={len(meta)-len(all_fail)} | with-issues={len(all_fail)}")
    if all_fail:
        from collections import Counter
        c = Counter(x for v in all_fail.values() for x in v)
        print(f"[AGGREGATE] issue types: {dict(c)}")
        print(f"[AGGREGATE] queries with issues: {dict(all_fail)}")


if __name__ == "__main__":
    main()
