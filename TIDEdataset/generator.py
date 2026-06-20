
import random
import pandas as pd
import numpy as np
from utils import cfg

POS_SPLIT_CNT = cfg("pos_split_cnt")
NEG_SPLIT_CNT = cfg("neg_split_cnt")
MAX_HOPS = cfg("max_hops")
MIN_HOPS = cfg("min_hops")
MIN_COLS = cfg("min_cols")
P_QUERY_TABLE = cfg("p_query_table")
P_CORR_HOP = cfg("p_corr_hop")
P_UNION_HOP = cfg("p_union_hop")
P_COMBINER = cfg("p_combiner")
P_VALUE = cfg("p_value")
VALUE_POS_CNT = cfg("value_pos_cnt")
INTERSECTION_GT = cfg("intersection_gt")
MIN_GROUNDS = cfg("min_grounds")
MIN_CORR_GROUPS = cfg("min_corr_groups")
ANCHOR_K = cfg("anchor_k")
ANCHOR_CORE_K = cfg("anchor_core_k")


class Skip(Exception):

    pass


def cats(df):
    return [c for c in df.columns if not str(df[c].dtype).startswith(("int", "float"))]


def nums(df):
    return [c for c in df.columns if str(df[c].dtype).startswith(("int", "float"))]


def repeating_keys(df):

    return [c for c in cats(df) if df[c].nunique() >= 2 and df[c].value_counts().iloc[0] >= 2]


_PLACEHOLDER = {"", "-", "–", "—", "——", "n/a", "na", "none", "no data", "unknown",
                "nan", "null", "?", "tbd", "n.a.", "."}


def meaningful_name(col):

    return any(ch.isalpha() for ch in str(col))


def is_label(v):

    s = str(v).strip()
    return bool(s) and s.lower() not in _PLACEHOLDER and any(ch.isalpha() for ch in s)


def eq_target(df, col):

    vc = df[col].dropna().astype(str).value_counts()
    for val in vc.index:
        if vc[val] >= 2 and is_label(val):
            return val
    return None


def row_sample(df, lo=0.5, hi=0.8):
    if len(df) == 0:
        return df
    n = max(1, int(len(df) * random.uniform(lo, hi)))
    return df.sample(n=min(n, len(df)))


def synth_corr_pair(S, jcol, rname, pname):

    groups = S[jcol].astype(str)
    z = {g: np.random.normal(0, 1) for g in groups.unique()}
    base = groups.map(z).astype(float).to_numpy()
    S[rname] = 50.0 + 10.0 * base + np.random.normal(0, 1.0, len(S))
    S[pname] = 100.0 + 20.0 * base + np.random.normal(0, 2.0, len(S))


def anchors(S, key):

    vc = S[key].dropna().value_counts()
    A = set(vc.index[: min(ANCHOR_K, max(1, len(vc) - 1))])
    if not (set(S[key].dropna().unique()) - A):
        raise Skip(f"key {key}: no foreign values for negatives")
    return A


def core_of(S, key):

    vc = S[key].dropna().value_counts()
    return set(vc.index[: min(ANCHOR_CORE_K, len(vc))])


def fixed_cols(all_cols, level_cols, filler_pool):

    need = max(0, MIN_COLS - len(level_cols))
    k = min(len(filler_pool), need + random.randint(0, 2))
    fill = random.sample(filler_pool, k) if filler_pool else []
    keep = set(level_cols) | set(fill)
    return [c for c in all_cols if c in keep]


def edge_cols(e):
    return list(e["union_cols"]) if e["type"] == "union" else [e["key"]]


def eq_refill(t, col, target, other_vals, satisfy):

    n = len(t)
    if n == 0 or not other_vals:
        return
    if satisfy:
        vals = [target if random.random() < 0.75 else random.choice(other_vals) for _ in range(n)]
        if target not in vals:                       # guarantee the target is present
            vals[random.randrange(n)] = target
        t[col] = vals
    else:
        t[col] = [random.choice(other_vals) for _ in range(n)]


def range_refill(t, col, target, lo, hi, satisfy):

    n = len(t)
    if n == 0:
        return
    t[col] = np.random.uniform(lo, target, n) if satisfy else np.random.uniform(target, hi, n)


class Emitter:

    def __init__(self, qid):
        self.qid = qid
        self.n = 0
        self.tables = {}

    def emit(self, df):
        nm = f"{self.qid}_{self.n}.csv"
        self.n += 1
        self.tables[nm] = df.reset_index(drop=True)
        return nm


class Plan:
    def __init__(self, step=4):
        self.has_tq = random.random() < P_QUERY_TABLE
        if random.random() < P_COMBINER:
            self.topology = "converge"
            self.set_op = random.choice(["intersection", "union", "difference"])
            self.depth = 1
        else:
            self.topology = "chain"
            self.set_op = None
            self.depth = random.randint(MIN_HOPS, min(MAX_HOPS, step))

    def realize(self, df, qid):

        S = df.reset_index(drop=True).copy()
        em = Emitter(qid)
        if self.topology == "chain":
            return self._realize_chain(S, em)
        return self._realize_converge(S, em)

    # ------------------------------------------------------------------ chain
    def _realize_chain(self, S, em):
        if len(S) < 50:
            raise Skip("too few rows")
        rkeys = repeating_keys(S)
        keys_avail = rkeys[:]
        random.shuffle(keys_avail)
        union_pool = [c for c in S.columns if c not in rkeys]
        nums_all = nums(S)
        all_cats = cats(S)
        used = set()                     # every column already spoken for

        eq_pool = [c for c in all_cats if meaningful_name(c) and eq_target(S, c) and S[c].nunique() >= 2]
        random.shuffle(eq_pool)
        eq_vals, range_vals = [], []
        def backbone_cap(drop_key=0, drop_union=()):
            ucols = len([x for x in union_pool if x not in used and x not in drop_union])
            return (len(keys_avail) - drop_key) + ucols // 2

        for c in eq_pool:
            if random.random() >= P_VALUE:
                continue
            if backbone_cap(drop_key=1) < MIN_HOPS:       # reserving this key would starve the chain
                break
            eq_vals.append(("eq", c, eq_target(S, c)))
            used.add(c)
            if c in keys_avail:
                keys_avail.remove(c)
        range_pool = [c for c in nums_all if c not in used and meaningful_name(c)
                      and float(S[c].min()) < float(S[c].quantile(0.6)) < float(S[c].max())]
        random.shuffle(range_pool)
        for c in range_pool:
            if random.random() >= P_VALUE:
                continue
            if backbone_cap(drop_union=(c,)) < MIN_HOPS:  # this numeric may be a union column
                continue
            range_vals.append(("range", c, float(round(S[c].quantile(0.6), 4))))
            used.add(c)

        def take_join_key():
            keys_avail.sort(key=lambda c: eq_target(S, c) is not None)  # plain keys first
            K = keys_avail.pop(0)
            used.add(K)
            return K

        # --- allocate RELATIONSHIP hops (overlap / corr / union): the traversal backbone ---
        edges = []
        for _ in range(self.depth):
            u_avail = [c for c in union_pool if c not in used]    # exclude reserved value cols
            want_union = (random.random() < P_UNION_HOP and len(u_avail) >= 2) or not keys_avail
            if want_union and len(u_avail) >= 2:
                u = random.sample(u_avail, random.randint(2, min(3, len(u_avail))))
                for c in u:
                    union_pool.remove(c)
                used.update(u)
                edges.append({"type": "union", "key": None, "union_cols": u, "ref": None, "payload": None, "value": None})
            elif keys_avail:
                is_corr = random.random() < P_CORR_HOP
                edges.append({"type": "corr" if is_corr else "overlap", "key": take_join_key(),
                              "union_cols": None, "ref": None, "payload": None, "value": None})
            else:
                break
        if len(edges) < MIN_HOPS:                     # a chain must traverse >= MIN_HOPS relationship
            raise Skip(f"chain depth {len(edges)} < MIN_HOPS (eq reservation left too few keys)")
        D = len(edges)

        random.shuffle(eq_vals); random.shuffle(range_vals)
        hop_idxs = list(range(D)); random.shuffle(hop_idxs)
        for (kind, col, target), hi in zip(eq_vals + range_vals, hop_idxs):
            edges[hi]["value"] = {"col": col, "kind": kind, "target": target}

        # anchors per key-bearing hop; demote thin corr hops; synth corr columns
        for e in edges:
            if e["key"]:
                e["anchor"] = anchors(S, e["key"])
        km = pd.Series(True, index=S.index)
        for e in edges:
            if e["key"]:
                km &= S[e["key"]].isin(e["anchor"])
        pop = S[km]
        for e in edges:
            if e["type"] == "corr" and pop[e["key"]].nunique() < MIN_CORR_GROUPS:
                e["type"] = "overlap"
        for i, e in enumerate(edges, start=1):
            if e["type"] == "corr":
                e["ref"], e["payload"] = f"_ref{i}", f"_factor{i}"
                synth_corr_pair(S, e["key"], e["ref"], e["payload"])

        refs = {i: e["ref"] for i, e in enumerate(edges, 1) if e["ref"]}
        pays = {i: e["payload"] for i, e in enumerate(edges, 1) if e["payload"]}
        vals = {i: e["value"] for i, e in enumerate(edges, 1) if e["value"]}
        A = {i: e.get("anchor") for i, e in enumerate(edges, 1) if e["key"]}
        C = {i: core_of(S, e["key"]) for i, e in enumerate(edges, 1) if e["key"]}
        constraint_cols = ({e["key"] for e in edges if e["key"]} |
                           {c for e in edges if e["union_cols"] for c in e["union_cols"]} |
                           set(refs.values()) | set(pays.values()) | {v["col"] for v in vals.values()})
        filler_pool = [c for c in S.columns if c not in constraint_cols]

        def node_cols(i):
            cols = []
            if i == 0:
                cols += edge_cols(edges[0])
                if 1 in refs:
                    cols.append(refs[1])
                return list(dict.fromkeys(cols))
            e = edges[i - 1]
            cols += edge_cols(e)                 # incoming join key (or union cols)
            if i in pays:
                cols.append(pays[i])             # corr payload (measured here)
            if i in vals:
                cols.append(vals[i]["col"])      # value column (verified here)
            if i < D:
                cols += edge_cols(edges[i])      # outgoing key to next hop
                if (i + 1) in refs:
                    cols.append(refs[i + 1])     # corr reference (lives upstream)
            return list(dict.fromkeys(cols))

        key_mask = pd.Series(True, index=S.index)
        for k, e in enumerate(edges, start=1):
            if e["key"]:
                key_mask &= S[e["key"]].isin(A[k])
        spine = S[key_mask]
        if len(spine) < 5:
            raise Skip("path-population too small")

        def pos_sample(cols):
            keep = pd.Series(False, index=spine.index)
            for hi, ed in enumerate(edges, start=1):
                if ed["key"] and ed["key"] in cols and hi in C:
                    keep |= spine[ed["key"]].isin(C[hi])
            forced = spine[keep]
            if len(forced) == 0:
                return row_sample(spine)[cols].copy()
            return pd.concat([forced, row_sample(spine[~keep])])[cols].copy()

        # --- start node: T_q or NL-only seed -------------------------------
        fc0 = fixed_cols(S.columns, node_cols(0), filler_pool)
        tq_df = None
        if self.has_tq:
            tq_df = pos_sample(fc0)
            start = {"kind": "query_table", "cols": list(tq_df.columns), "pos": [], "neg": []}
        else:
            seed_cols = node_cols(0)
            pos = [em.emit(pos_sample(fc0)) for _ in range(POS_SPLIT_CNT)]
            neg = []
            absent = [c for c in S.columns if c not in set(seed_cols)]
            if absent:
                for _ in range(NEG_SPLIT_CNT):
                    sub = row_sample(S)[absent]
                    neg.append(em.emit(sub.sample(frac=random.uniform(0.5, 0.9), axis=1) if sub.shape[1] > 1 else sub))
            start = {"kind": "seed", "cols": seed_cols, "pos": pos, "neg": neg}

        # --- hops 1..D -----------------------------------------------------
        rec_edges, grounds = [], []
        for i in range(1, D + 1):
            e = edges[i - 1]; is_term = (i == D); v = vals.get(i)
            fc = fixed_cols(S.columns, node_cols(i), filler_pool)

            value_neg = []
            if v:
                other_vals = ([x for x in S[v["col"]].dropna().astype(str).unique() if x != str(v["target"])]
                              if v["kind"] == "eq" else None)
                lo = float(S[v["col"]].min()) if v["kind"] == "range" else None
                hi = float(S[v["col"]].max()) if v["kind"] == "range" else None
                pos = []
                for idx in range(VALUE_POS_CNT):
                    t = pos_sample(fc)
                    sat = idx < POS_SPLIT_CNT
                    if v["col"] in t.columns:
                        if v["kind"] == "eq":
                            eq_refill(t, v["col"], v["target"], other_vals, sat)
                        else:
                            range_refill(t, v["col"], v["target"], lo, hi, sat)
                    (pos if sat else value_neg).append(em.emit(t))
            else:
                pos = [em.emit(pos_sample(fc)) for _ in range(POS_SPLIT_CNT)]

            if v or is_term:
                grounds.extend(pos)

            neg = []
            for j in range(NEG_SPLIT_CNT):
                if e["type"] == "union":
                    p = [x for x in S.columns if x not in e["union_cols"]]
                    k = random.randint(2, min(4, len(p))) if len(p) >= 2 else len(p)
                    t = row_sample(S)[random.sample(p, k)].copy()
                elif e["type"] == "corr" and (j % 5 != 0):
                    fb = S[~S[e["key"]].isin(A[i])]               # foreign key: can't out-join GT
                    t = row_sample(fb if len(fb) >= 3 else S)[fc].copy()
                    if pays[i] in t.columns:                      # scramble the payload too
                        t[pays[i]] = np.random.uniform(80, 120, size=len(t))
                else:
                    fb = S[~S[e["key"]].isin(A[i])]               # foreign on the incoming key
                    t = row_sample(fb)[fc].copy()
                neg.append(em.emit(t))
            neg += value_neg                                         # value-level distractors

            rec_edges.append({"type": e["type"], "branch": None, "key": e["key"],
                              "union_cols": e["union_cols"], "ref": e["ref"], "payload": e["payload"],
                              "constraint": v, "value_neg": value_neg, "terminal": is_term,
                              "cols": node_cols(i), "pos": pos, "neg": neg})

        if len(grounds) < MIN_GROUNDS:
            raise Skip(f"grounds {len(grounds)} < MIN_GROUNDS")
        record = {"topology": "chain", "depth": D, "has_tq": self.has_tq, "set_op": None,
                  "keys": None, "start": start, "edges": rec_edges}
        return em.tables, grounds, tq_df, record

    # --------------------------------------------------------------- converge
    def _realize_converge(self, S, em):
        if len(S) < 50:
            raise Skip("too few rows")
        cats = repeating_keys(S)
        if len(cats) < 2:
            raise Skip("converge needs >=2 repeating keys")
        Ka, Kb = random.sample(cats, 2)
        op = self.set_op
        branches = []
        for K, br in [(Ka, "A"), (Kb, "B")]:
            e = {"type": "overlap", "key": K, "union_cols": None, "ref": None, "payload": None, "branch": br,
                 "anchor": anchors(S, K)}
            if random.random() < P_CORR_HOP and len(e["anchor"]) >= MIN_CORR_GROUPS:
                e["type"] = "corr"
                e["ref"], e["payload"] = f"_ref{br}", f"_factor{br}"
                synth_corr_pair(S, K, e["ref"], e["payload"])
            branches.append(e)
        eA, eB = branches
        Aa, Ab = eA["anchor"], eB["anchor"]
        refs = {b["branch"]: b["ref"] for b in branches if b["ref"]}
        pays = {b["branch"]: b["payload"] for b in branches if b["payload"]}

        constraint_cols = {Ka, Kb} | set(refs.values()) | set(pays.values())
        filler = [c for c in S.columns if c not in constraint_cols]
        A_cols = [Ka] + ([pays["A"]] if "A" in pays else [])
        B_cols = [Kb] + ([pays["B"]] if "B" in pays else [])
        both_cols = list(dict.fromkeys(A_cols + B_cols))
        start_cols = [Ka, Kb] + list(refs.values())
        fcA = fixed_cols(S.columns, A_cols, filler)
        fcB = fixed_cols(S.columns, B_cols, filler)
        fcBoth = fixed_cols(S.columns, both_cols, filler)
        fcStart = fixed_cols(S.columns, start_cols, filler)

        inA, inB = S[Ka].isin(Aa), S[Kb].isin(Ab)
        both, aOnly, bOnly, neither = inA & inB, inA & ~inB, ~inA & inB, ~inA & ~inB

        def mk(mask, fc, n):
            fb = S[mask]
            return [em.emit(row_sample(fb)[fc].copy()) for _ in range(n)] if len(fb) >= 3 else []

        acov_parts = [S[S[Ka] == a].head(2) for a in Aa if (S[Ka] == a).any()] + \
                     [S[S[Kb] == b].head(2) for b in Ab if (S[Kb] == b).any()]
        acov = pd.concat(acov_parts).drop_duplicates() if acov_parts else S.iloc[0:0]
        base = S[inA | inB]
        if len(base) < 5:
            raise Skip("converge start spine too small")

        def start_table():
            t = pd.concat([acov, row_sample(base)]).drop_duplicates().copy()
            for K, Aset in [(Ka, Aa), (Kb, Ab)]:           # clamp start.Ka⊆Aa, start.Kb⊆Ab
                m = ~t[K].isin(Aset)
                if m.any():
                    t.loc[m, K] = np.random.choice(list(Aset), size=int(m.sum()))
            return t[fcStart].copy()

        tq_df = None
        if self.has_tq:
            tq_df = start_table()
            start = {"kind": "query_table", "cols": list(tq_df.columns), "pos": [], "neg": []}
        else:
            pos = [em.emit(start_table()) for _ in range(POS_SPLIT_CNT)]
            neg = []
            absent = [c for c in S.columns if c not in set(start_cols)]
            if absent:
                for _ in range(NEG_SPLIT_CNT):
                    sub = row_sample(S)[absent]
                    neg.append(em.emit(sub.sample(frac=random.uniform(0.5, 0.9), axis=1) if sub.shape[1] > 1 else sub))
            start = {"kind": "seed", "cols": start_cols, "pos": pos, "neg": neg}

        # GT + distractors per set op (GT names → grounds; distractors stay in the lake)
        if op == "intersection":
            grounds = mk(both, fcBoth, INTERSECTION_GT)              # reachable via BOTH
            mk(aOnly, fcBoth, 20); mk(bOnly, fcBoth, 20); mk(neither, fcBoth, 10)
        elif op == "union":
            grounds = mk(inA, fcA, POS_SPLIT_CNT) + mk(inB, fcB, POS_SPLIT_CNT)   # EITHER
            mk(~inA, fcA, 25); mk(~inB, fcB, 25)
        else:  # difference  A \ B : joinable on Ka but NOT Kb
            grounds = mk(aOnly, fcBoth, POS_SPLIT_CNT)
            mk(both, fcBoth, 30); mk(neither, fcBoth, 10)
        if not grounds:
            raise Skip("converge produced no GT")

        rec_edges = [{"type": b["type"], "branch": b["branch"], "key": b["key"], "union_cols": None,
                      "ref": b["ref"], "payload": b["payload"], "constraint": None, "terminal": True,
                      "cols": (A_cols if b["branch"] == "A" else B_cols), "pos": [], "neg": []}
                     for b in branches]
        record = {"topology": "converge", "depth": 1, "has_tq": self.has_tq, "set_op": op,
                  "keys": [Ka, Kb], "start": start, "edges": rec_edges}
        return em.tables, grounds, tq_df, record
