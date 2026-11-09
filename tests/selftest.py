import sys, json, collections
sys.path.insert(0, "src")
from encoders import ENCODINGS, build_prompt
from scoring import score, gold_text
import networkx as nx

recs = [json.loads(l) for l in open("data/questions.jsonl")]
bad = 0
for r in recs:
    g = nx.Graph(); g.add_nodes_from(range(r["n_nodes"])); g.add_edges_from(r["edges"])
    # re-verify gold answers independently
    t, q = r["task"], r["query"]
    exp = {"edge_existence": lambda: "yes" if g.has_edge(*q) else "no",
           "node_degree": lambda: g.degree(q[0]), "node_count": lambda: g.number_of_nodes(),
           "connected_nodes": lambda: sorted(g.neighbors(q[0])),
           "cycle_check": lambda: "yes" if len(nx.cycle_basis(g)) else "no"}[t]()
    if exp != r["answer"]: bad += 1
    for e in ENCODINGS:
        assert score(e, r, gold_text(e, r)) == 1, (e, r["id"])          # perfect answer scores 1
        assert score(e, r, "I don't know") == 0 or t == "x", (e, r["id"], "junk scored 1")
print("records", len(recs), "| gold mismatches", bad)
c = collections.Counter(r["task"] for r in recs); print(dict(c))
for t in ("edge_existence", "cycle_check"):
    print(t, collections.Counter(r["answer"] for r in recs if r["task"] == t))
ns = [r["n_nodes"] for r in recs]; print("nodes min/max", min(ns), max(ns))
r = next(x for x in recs if x["task"] == "connected_nodes" and x["n_nodes"] <= 6)
for e in ["adjacency", "incident", "friendship", "sp", "expert"]:
    print(f"\n--- {e} ---\n{build_prompt(e, r)}")
