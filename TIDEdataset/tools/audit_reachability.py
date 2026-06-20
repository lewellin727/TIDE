"""Structural audit of a v4 (_tide_v4) split against the DESIGN_v4 Phase-A criteria.

Chains:
  (1) GT is NOT self-contained (missing upstream constraint columns).
  (2) Negatives are DISTRACTING (same column schema as positives, per key edge).
  (3) A per-table 1-hop filter baseline cannot reach GT (traversal required).
  (4) Cross-join predicate: corr GT correlates over the join; corr negs do not.
Converge (set-op) queries are audited structurally too (NOT skipped):
  (C) every GT table satisfies the set predicate (joinable via both/either/aOnly);
      corr branches correlate over the join.

Reads the UNIFIED metadata record + raw `lakes/`.
Usage: python tools/audit_reachability.py [--suffix _tide_v4]
"""
import os, json, argparse
import numpy as np, pandas as pd
from collections import Counter, defaultdict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cfg

_REV = {}        # qid -> {aug_name: orig_name}, set only when auditing aug_lakes
_LAKE = None     # lake dir whose tables get reverse-renamed (not input_tables)


def load_table(d, name):
    fp = os.path.join(d, name)
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, low_memory=False)
    if _REV and d == _LAKE:                       # reverse-rename aug cols → canonical
        rev = _REV.get(name.split("_")[0])
        if rev:
            df = df.rename(columns=rev)
    return df


def svals(df, col):
    return set(df[col].dropna().astype(str)) if (df is not None and col is not None and col in df.columns) else set()


def constraint_cols(edges):
    cc = set()
    for e in edges:
        for k in ("key", "ref", "payload"):
            if e.get(k):
                cc.add(e[k])
        cc |= set(e.get("union_cols") or [])
        if e.get("constraint") and e["constraint"].get("col"):
            cc.add(e["constraint"]["col"])
    return cc


def joincorr(up, tbl, jk, rc, pc):
    if up is None or tbl is None or rc not in up.columns or jk not in up.columns \
            or jk not in tbl.columns or pc not in tbl.columns:
        return None
    try:
        ga = tbl.assign(_k=tbl[jk].astype(str)).groupby("_k")[pc].mean()
        gb = up.assign(_k=up[jk].astype(str)).groupby("_k")[rc].mean()
        mm = pd.concat([ga, gb], axis=1, join="inner").dropna()
        return abs(mm.iloc[:, 0].astype(float).corr(mm.iloc[:, 1].astype(float))) if len(mm) >= 3 else None
    except Exception:
        return None


def main():
    global _REV, _LAKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--lake", default="lakes", choices=["lakes", "aug_lakes"])
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, f"dataset/datalakes/{cfg('dataset')}{args.suffix}")
    lake = os.path.join(d, args.lake); inp = os.path.join(d, "input_tables")
    _LAKE = lake
    if args.lake == "aug_lakes":
        cm = json.load(open(os.path.join(d, "aug_colmap.json")))
        _REV = {qid: {aug: orig for orig, aug in m.items()} for qid, m in cm.items()}
        print(f"[auditing aug_lakes — reverse-renaming columns via aug_colmap.json]")
    meta = json.load(open(os.path.join(d, "metadata.json")))

    depths = Counter()
    self_contained, gt_missing_first, neg_same_schema = [], [], []
    base_recall_by_depth = defaultdict(list)
    corr_pos_ok, corr_neg_ok = [], []
    conv_gt_ok, conv_corr_ok = [], []
    val_pos_ok, val_neg_ok = [], []
    n_conv = n_chain = 0

    # Exclude distractor (`noise_*`) tables: they are never in any query's grounds, so they cannot
    # affect the `reached & grounds` recall metric — but scanning/loading them per query makes the
    # base-recall O(queries × all-lake-files) and dominates runtime when `distractor_table_count` is
    # large. Skipping them keeps the audit fast regardless of how many distractors the lake holds.
    lake_files = [f for f in os.listdir(lake) if not f.startswith("noise_")]

    for m in meta:
        p = m["path"]; grounds = set(m["grounds"]); edges = p["edges"]; start = p["start"]

        if p["topology"] == "converge":
            n_conv += 1
            Ka, Kb = p["keys"]; op = p["set_op"]
            if start["kind"] == "query_table":
                st = load_table(inp, f"query_table_{m['id']}.csv")
            else:
                st = next((load_table(lake, n) for n in start["pos"]), None)
            va, vb = svals(st, Ka), svals(st, Kb)
            for g in grounds:
                t = load_table(lake, g)
                if t is None:
                    continue
                ina = bool(svals(t, Ka) & va); inb = bool(svals(t, Kb) & vb)
                conv_gt_ok.append((op == "intersection" and ina and inb) or
                                  (op == "difference" and ina and not inb) or
                                  (op == "union" and (ina or inb)))
            for e in edges:
                if e["type"] == "corr" and st is not None:
                    for g in sorted(grounds)[:3]:
                        r = joincorr(st, load_table(lake, g), e["key"], e["ref"], e["payload"])
                        if r is not None:
                            conv_corr_ok.append(r >= 0.5)
            continue

        # ---- chain ----
        n_chain += 1
        D = p["depth"]; depths[D] += 1
        cc = constraint_cols(edges)
        first = edges[0]
        first_cols = ({first["key"]} if first.get("key") else set()) | set(first.get("union_cols") or [])

        for g in sorted(grounds)[:5]:
            t = load_table(lake, g)
            if t is None:
                continue
            cols = set(t.columns)
            self_contained.append(cc.issubset(cols))
            if D >= 2 and first_cols:
                gt_missing_first.append(not (first_cols & cols))

        for e in edges:
            if e["type"] == "union":
                continue
            ps = [set(load_table(lake, n).columns) for n in e["pos"][:3] if load_table(lake, n) is not None]
            ns = [set(load_table(lake, n).columns) for n in e["neg"][:3] if load_table(lake, n) is not None]
            if ps and ns:
                neg_same_schema.append(all(s == ps[0] for s in ns))

        # value constraints (eq/range rankers on a hop): satisfying positives pass the value,
        # the value_neg tables (same hop, re-filled to fail) fail it.
        for e in edges:
            cj = e.get("constraint")
            if not cj:
                continue
            col = cj["col"]
            def vpass(df):
                if df is None or col not in df.columns:
                    return None
                return bool((df[col].astype(str) == str(cj["target"])).any()) if cj["kind"] == "eq" \
                    else bool((df[col].astype(float) < cj["target"]).all())
            vneg = e.get("value_neg") or []
            pv = vpass(load_table(lake, e["pos"][0])) if e["pos"] else None
            nv = vpass(load_table(lake, vneg[0])) if vneg else None
            if pv is not None:
                val_pos_ok.append(pv is True)
            if nv is not None:
                val_neg_ok.append(nv is False)

        # 1-hop filter baseline from the start, on the first key
        first_key = first.get("key")
        start_vals = set()
        if first_key:
            if start["kind"] == "query_table":
                qt = load_table(inp, f"query_table_{m['id']}.csv")
                start_vals = svals(qt, first_key)
            else:
                for n in start["pos"][:3]:
                    start_vals |= svals(load_table(lake, n), first_key)
        if start_vals and first_key:
            reached = {nm for nm in lake_files
                       if (lambda t: t is not None and bool(svals(t, first_key) & start_vals))(load_table(lake, nm))}
            base_recall_by_depth[D].append(len(reached & grounds) / max(1, len(grounds)))

        # corr cross-join on the terminal edge
        term = edges[-1]
        if term["type"] == "corr" and term.get("payload") and term.get("ref"):
            if D == 1:
                bridge = [load_table(inp, f"query_table_{m['id']}.csv")] if start["kind"] == "query_table" \
                    else [load_table(lake, n) for n in start["pos"]]
            else:
                bridge = [load_table(lake, n) for n in edges[-2]["pos"]]
            ups = [b for b in bridge if b is not None and term["ref"] in b.columns]
            up = pd.concat(ups, ignore_index=True) if ups else None      # verify vs the whole bridge SET
            for g in sorted(grounds)[:3]:
                r = joincorr(up, load_table(lake, g), term["key"], term["ref"], term["payload"])
                if r is not None:
                    corr_pos_ok.append(r >= 0.5)
            for n in term["neg"][:5]:
                r = joincorr(up, load_table(lake, n), term["key"], term["ref"], term["payload"])
                if r is not None:
                    corr_neg_ok.append(r < 0.5)

    pct = lambda x: f"{100*np.mean(x):.0f}%" if x else "n/a"
    print(f"=== v4 structural audit ({args.suffix}) | {len(meta)} queries "
          f"({n_chain} chain, {n_conv} converge) ===")
    print(f"chain depth dist: {dict(sorted(depths.items()))}")
    print(f"\n-- chains --")
    print(f"(1) GT self-contained (all constraint cols present)  : {pct(self_contained)}  (want LOW)")
    print(f"    D>=2 GT missing first-edge cols                  : {pct(gt_missing_first)}  (want ~100%)")
    print(f"(2) negatives share pos column-schema (distracting)  : {pct(neg_same_schema)}  (want ~100%)")
    print(f"(3) 1-hop filter baseline GT recall (by depth):")
    for D in sorted(base_recall_by_depth):
        v = base_recall_by_depth[D]
        print(f"      depth {D}: mean GT recall = {np.mean(v):.2f}   (D>=2 want <0.2)")
    print(f"(4) corr GT correlates over join                     : {pct(corr_pos_ok)}  (want ~100%)")
    print(f"    corr negatives do NOT correlate                  : {pct(corr_neg_ok)}  (want ~100%)")
    print(f"(5) value-hop positives satisfy the value            : {pct(val_pos_ok)}  (want ~100%)")
    print(f"    value-hop negatives fail the value               : {pct(val_neg_ok)}  (want ~100%)")
    print(f"\n-- converge --")
    print(f"(C) GT satisfies the set predicate                   : {pct(conv_gt_ok)}  (want ~100%)")
    print(f"    corr branch correlates over join                 : {pct(conv_corr_ok)}  (want ~100%)")


if __name__ == "__main__":
    main()
