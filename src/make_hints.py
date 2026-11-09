"""Build planner hints from fixed-run rows (top-K encodings per question type: accuracy %, mean input tokens).
  python3 src/make_hints.py results/X_heldout_fixed.jsonl [K=2] > data/hints.txt"""
import sys, json, collections, statistics as st
STYLE = {"edge_existence": "is there an edge between two nodes", "node_degree": "degree of a node",
         "node_count": "how many nodes", "connected_nodes": "list the nodes connected to a node",
         "cycle_check": "is there a cycle"}

def build_hints(rows, K=2):
    by = collections.defaultdict(list)
    for r in rows: by[(r["task"], r["encoding"])].append(r)
    out = ["Practice results per question type (accuracy %, tokens):"]
    for t in STYLE:
        sc = sorted(((st.mean(x["correct"] for x in by[(t, e)]) * 100, -st.mean(x["in_tokens"] for x in by[(t, e)]), e)
                     for e in {k[1] for k in by if k[0] == t}), reverse=True)[:K]
        out.append(f"{STYLE[t]}: " + ", ".join(f"{e} ({a:.0f}%, {-m:.0f})" for a, m, e in sc))
    return "\n".join(out)

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    print(build_hints(rows, int(sys.argv[2]) if len(sys.argv) > 2 else 2))
