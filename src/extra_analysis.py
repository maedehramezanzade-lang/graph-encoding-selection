"""Extra analyses added after review (all from stored results; no model is run).
  python3 src/extra_analysis.py --prefix Qwen2.5-7B-Instruct
1. hint following: how often the hinted planners chose an encoding named in their hints for that task
2. best encoding by graph size (terciles of n_nodes) within each task
3. chance level per task (majority class for yes/no tasks; most frequent answer for the others)
4. label balance of the test questions"""
import argparse, json, collections, statistics as st, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from planner import parse_choice
from make_hints import build_hints, STYLE
ap = argparse.ArgumentParser(); ap.add_argument("--prefix", required=True); ap.add_argument("--data", default="data/questions.jsonl"); ap.add_argument("--results", default="results")
a = ap.parse_args()
L = lambda p: [json.loads(l) for l in open(p)]
recs = {r["id"]: r for r in L(a.data)}
fixed = L(f"{a.results}/{a.prefix}_fixed.jsonl"); held = L(f"{a.results}/{a.prefix}_heldout_fixed.jsonl")
task = {q: r["task"] for q, r in recs.items()}
fold_of = {}
for t in set(task.values()):
    for i, q in enumerate(sorted(q for q in task if task[q] == t)): fold_of[q] = i % 5
def top2(rows):
    by = collections.defaultdict(list)
    for r in rows: by[(r["task"], r["encoding"])].append(r)
    out = {}
    for t in STYLE:
        sc = sorted(((st.mean(x["correct"] for x in by[(t, e)]) * 100, -st.mean(x["in_tokens"] for x in by[(t, e)]), e) for e in {k[1] for k in by if k[0] == t}), reverse=True)[:2]
        out[t] = {e for _, _, e in sc}
    return out
hint_all = top2(held); hint_fold = {k: top2([r for r in fixed if fold_of[r["id"]] != k]) for k in range(5)}
print(f"== {a.prefix}: 1. hint following (choice named in the hints for that task)")
for variant in ("hinted", "hinted_cv"):
    rows = L(f"{a.results}/{a.prefix}_planner_{variant}.jsonl"); tot = collections.Counter(); inn = collections.Counter(); ch = collections.defaultdict(collections.Counter)
    for r in rows:
        h = hint_all if variant == "hinted" else hint_fold[fold_of[r["id"]]]
        c = parse_choice(r["raw"])[0]; t = r["task"]; tot[t] += 1; inn[t] += int(c in h[t]); ch[t][c] += 1
    print(f"{variant}: overall {sum(inn.values())}/{sum(tot.values())} followed")
    for t in sorted(tot): print(f"   {t:16s} {inn[t]}/{tot[t]}  choices {dict(ch[t].most_common())}")
print("== 2. accuracy (%) by graph size tercile within task (rows: encoding with best accuracy in each tercile; sizes = n_nodes ranges)")
for t in sorted(STYLE):
    qs = sorted((q for q in recs if task[q] == t), key=lambda q: (recs[q]["n_nodes"], q)); n = len(qs)
    terc = [qs[:n // 3], qs[n // 3:2 * n // 3], qs[2 * n // 3:]]
    line = []
    for i, tq in enumerate(terc):
        ids = {q: 1 for q in tq}; acc = collections.defaultdict(list)
        for r in fixed:
            if r["id"] in ids: acc[r["encoding"]].append(r["correct"])
        best = max(acc, key=lambda e: st.mean(acc[e]))
        lo, hi = min(recs[q]["n_nodes"] for q in tq), max(recs[q]["n_nodes"] for q in tq)
        line.append(f"nodes {lo}-{hi}: {best} {st.mean(acc[best])*100:.0f}% (incident {st.mean(acc['incident'])*100:.0f}, adjacency {st.mean(acc['adjacency'])*100:.0f}, expert {st.mean(acc['expert'])*100:.0f})")
    print(f"   {t}:"); [print("      " + x) for x in line]
print("== 3. chance level (most frequent correct answer)")
for t in sorted(STYLE):
    c = collections.Counter(json.dumps(recs[q]["answer"]) for q in recs if task[q] == t); k, v = c.most_common(1)[0]
    print(f"   {t:16s} most frequent answer {k} = {v}%")
print("== 4. test-set label balance: edge_existence yes =", sum(recs[q]["answer"] == "yes" for q in recs if task[q] == "edge_existence"), "/ 100; cycle_check yes =", sum(recs[q]["answer"] == "yes" for q in recs if task[q] == "cycle_check"), "/ 100")
import networkx as nx
ncyc = [recs[q]["n_edges"] for q in recs if task[q] == "cycle_check" and recs[q]["answer"] == "no"]
ycyc = [recs[q]["n_edges"] for q in recs if task[q] == "cycle_check" and recs[q]["answer"] == "yes"]
print(f"   cycle_check mean edges: no-cycle graphs {st.mean(ncyc):.1f}, cycle graphs {st.mean(ycyc):.1f}")
