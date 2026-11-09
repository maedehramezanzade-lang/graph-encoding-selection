"""Graph + question generation (ER graphs, 5 tasks). Answers are computed by NetworkX."""
import random, json
import networkx as nx

TASKS = ["edge_existence", "node_degree", "node_count", "connected_nodes", "cycle_check"]


def make_er_graph(rng, n_min=5, n_max=20):
    """Erdos-Renyi graph as in Fatemi et al.: 5-20 nodes, edge probability sampled from [0,1].
    We resample until the graph has at least one edge."""
    while True:
        n = rng.randint(n_min, n_max)
        p = rng.random()
        g = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**9))
        if g.number_of_edges() >= 1:
            return g


def _graph_for(task, rng, want=None):
    """Return a graph (and optional wanted label) that suits the task."""
    for _ in range(10000):
        g = make_er_graph(rng)
        if task == "cycle_check":
            has_cycle = len(nx.cycle_basis(g)) > 0
            if has_cycle == want:
                return g
        elif task == "edge_existence":
            n = g.number_of_nodes()
            if want:  # need an edge
                return g
            if g.number_of_edges() < n * (n - 1) // 2:  # need a non-edge
                return g
        elif task == "connected_nodes":
            if any(d > 0 for _, d in g.degree()):
                return g
        else:
            return g
    raise RuntimeError("could not generate graph")


def make_question(task, rng, qid):
    want = rng.random() < 0.5 if task in ("cycle_check", "edge_existence") else None
    g = _graph_for(task, rng, want)
    nodes = list(g.nodes())
    rec = {"id": qid, "task": task, "n_nodes": g.number_of_nodes(),
           "n_edges": g.number_of_edges(), "edges": sorted(map(list, g.edges()))}
    if task == "edge_existence":
        if want:
            u, v = rng.choice(list(g.edges()))
            if rng.random() < 0.5: u, v = v, u
        else:
            while True:
                u, v = rng.sample(nodes, 2)
                if not g.has_edge(u, v): break
        rec["query"] = [u, v]; rec["answer"] = "yes" if g.has_edge(u, v) else "no"
    elif task == "node_degree":
        u = rng.choice(nodes); rec["query"] = [u]; rec["answer"] = g.degree(u)
    elif task == "node_count":
        rec["query"] = []; rec["answer"] = g.number_of_nodes()
    elif task == "connected_nodes":
        u = rng.choice([x for x, d in g.degree() if d > 0])
        rec["query"] = [u]; rec["answer"] = sorted(g.neighbors(u))
    elif task == "cycle_check":
        rec["query"] = []; rec["answer"] = "yes" if want else "no"
        assert (len(nx.cycle_basis(g)) > 0) == want
    return rec


def build(per_task, seed, start_id=0):
    rng = random.Random(seed)
    out, i = [], start_id
    for t in TASKS:
        for _ in range(per_task):
            out.append(make_question(t, rng, f"{t}-{i}")); i += 1
    return out


if __name__ == "__main__":
    import sys
    # main evaluation set (100/task) and a small held-out set (20/task) for the task-aware lookup baseline
    for name, per, seed in [("questions", 100, 7901), ("heldout", 20, 1234)]:
        data = build(per, seed)
        with open(f"data/{name}.jsonl", "w") as f:
            for r in data: f.write(json.dumps(r) + "\n")
        print(name, len(data))
