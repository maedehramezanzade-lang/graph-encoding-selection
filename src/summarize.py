"""From fixed-run results: accuracy per encoding/task, mean tokens, and the free baselines
(best fixed, Always Incident, Always Shortest, task-aware lookup, cost-aware oracle)."""
import json, sys, collections, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0   # oracle: cheapest encoding that is correct
by = collections.defaultdict(list)
for r in rows: by[(r["encoding"], r["task"])].append(r)
encs = sorted({r["encoding"] for r in rows}); tasks = sorted({r["task"] for r in rows})
print("accuracy by encoding x task (%) | mean input tokens")
print(f"{'':13s}" + "".join(f"{t[:10]:>11s}" for t in tasks) + f"{'ALL':>8s}{'tokens':>8s}")
for e in encs:
    cells = [st.mean(x["correct"] for x in by[(e, t)]) * 100 if by[(e, t)] else float("nan") for t in tasks]
    allr = [x for t in tasks for x in by[(e, t)]]
    print(f"{e:13s}" + "".join(f"{c:11.1f}" for c in cells) +
          f"{st.mean(x['correct'] for x in allr)*100:8.1f}{st.mean(x['in_tokens'] for x in allr):8.0f}")
# per-question oracle: cheapest encoding that answers correctly (else cheapest overall)
q = collections.defaultdict(dict)
for r in rows: q[r["id"]][r["encoding"]] = r
corr = tok = 0
for qid, d in q.items():
    ok = [x for x in d.values() if x["correct"]]
    pick = min(ok or d.values(), key=lambda x: x["in_tokens"])
    corr += pick["correct"]; tok += pick["in_tokens"]
print(f"\ncost-aware oracle: acc {corr/len(q)*100:.1f}%  mean tokens {tok/len(q):.0f}  (n={len(q)})")
