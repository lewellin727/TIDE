def build_tree_footprint(node_id, reasoning_tree):
    """I(v) per paper Eq. 9: tables already explored by ancestor or sibling branches.

    Implementation choice (see review note 2026-05-22): the paper's notation
    `Anc(v) ∪ SibTree(v)` is interpreted broadly as "every tree node except v
    itself and v's descendants" — i.e., for every ancestor `a` of `v`, all of
    `a`'s non-ancestor children and their subtrees are folded into the
    footprint, not just `v`'s direct siblings. This matches the algorithm's
    intent ("F_nov rewards paths visiting tables not yet captured anywhere in
    the explored tree") and the paper's wording "ancestor or sibling
    branches" (plural; the strict graph-theoretic "sibling" reading would
    miss cousins / great-uncles).

    Returns a `set[str]` of `table_name` values.
    """
    node_id = str(node_id)

    descendants = set()
    stack = [node_id]
    while stack:
        cur = stack.pop()
        for item in reasoning_tree.adj_dict.get(cur, []):
            child = str(item['child'])
            if child not in descendants:
                descendants.add(child)
                stack.append(child)

    # Exclude this node and its descendants from the tree footprint.
    footprint = set()
    for node in reasoning_tree.nodes:
        nid = str(node.id)
        if nid == node_id or nid in descendants:
            continue
        for t in node.input_tables:
            footprint.add(t.table_name)
    return footprint


def rerank_paths(raw_paths, node, reasoning_tree, rel_graph, k_o=3):
    """Paper Eq. 11: greedy submodular maximization of F_nov over P_1.

    Input `raw_paths` is P_1 — the top-k_1 reliable paths produced by
    `navigate` under U_rel (paper Eq. 8). This routine then picks the
    cardinality-constrained subset S ⊆ P_1, |S| ≤ k_o, that maximizes

        F_nov(S | v) = | ⋃_{p ∈ S} V(p) \\ I(v) |        (paper Eq. 10)

    by iteratively adding the path with the largest marginal coverage gain
    over I(v) and the already-selected union of path tables. Ties on
    marginal gain are broken by U_rel (higher first). Theorem 1 gives the
    classical (1 - 1/e) approximation guarantee for monotone submodular
    maximization under a cardinality constraint.
    """
    if not raw_paths:
        return []

    footprint = build_tree_footprint(node.id, reasoning_tree)

    selected = []
    selected_tables = set()
    candidates = list(raw_paths)

    while candidates and len(selected) < k_o:
        best = None
        best_key = (-1, -1.0)  # Break equal coverage gains by path reliability.
        for p in candidates:
            new_tables = set(p['tables']) - selected_tables - footprint
            gain = len(new_tables)
            key = (gain, p['U_rel'])
            if key > best_key:
                best_key = key
                best = p
        selected.append(best)
        selected_tables.update(best['tables'])
        candidates.remove(best)

    # Normalize reliability for prompt display.
    total = sum(p['U_rel'] for p in selected)
    for p in selected:
        p['norm_score'] = p['U_rel'] / total if total > 0 else 0.0

    return selected


def render_path(path):
    """Single-line summary kept for backward-compat / log lines."""
    segs = []
    for i, step in enumerate(path['steps']):
        op, name, cols = step
        if i == 0:
            segs.append(str(name))
        else:
            if op == 'join' and cols:
                col_desc = ", ".join(cols)
                segs.append(f" --JOIN({col_desc})--> {name}")
            else:
                segs.append(f" --{op.upper()}--> {name}")
    return "".join(segs) + f" (Score: {path.get('norm_score', 0.0):.4f})"


def render_path_with_schema(path, name_to_schema):
    """Multi-line structured block. Designed for the Observation prompt: separates
    nodes from edges, surfaces the bridge keys that were actually traversed, and
    keeps long (d_o ≥ 3) paths human / LLM parseable. Falls back gracefully when
    schemas are missing.
    """
    steps = path['steps']
    length = max(0, len(steps) - 1)
    score = path.get('norm_score', 0.0)
    endpoint = path.get('endpoint')

    node_lines = []
    for i, step in enumerate(steps):
        _op, name, _cols = step
        cols = name_to_schema.get(name)
        cols_str = (", ".join(str(c) for c in cols)) if cols else "(schema unavailable)"
        tag = ""
        if i == 0:
            tag = "  ⟵ start (in your current candidate set)"
        elif name == endpoint:
            tag = "  ⟵ endpoint"
        node_lines.append(f"    [{i}] {name}  cols: [{cols_str}]{tag}")

    edge_lines = []
    traversed_keys = []
    for i, step in enumerate(steps):
        if i == 0:
            continue
        op, name, cols = step
        src = steps[i - 1][1]
        if op == 'join' and cols:
            key_str = ", ".join(str(c) for c in cols)
            edge_lines.append(f"    [{i-1}]→[{i}]  JOIN  via column(s) [{key_str}]")
            traversed_keys.extend(cols)
        else:
            edge_lines.append(f"    [{i-1}]→[{i}]  {op.upper()}")

    bridge_summary = (
        ", ".join(sorted(set(str(k) for k in traversed_keys))) if traversed_keys else "(none)"
    )

    block = (
        f"length={length}, reliability={score:.4f}\n"
        f"  nodes:\n" + "\n".join(node_lines) + "\n"
        f"  edges:\n" + ("\n".join(edge_lines) if edge_lines else "    (no edges)") + "\n"
        f"  bridge_keys traversed: {bridge_summary}"
    )
    return block
    return base
