"""Compare strategies using fixed-run results (+ planner results, + optional held-out fixed run for the lookup baseline).

  python3 src/evaluate.py --fixed results/X_fixed.jsonl [--planner results/X_planner_balanced.jsonl ...] \
        [--heldout-fixed results/X_heldout_fixed.jsonl] [--csv results/X_summary.csv]

Cost columns:  in = input tokens (solver prompt + planner prompt) ; total = input + output tokens of solver and planner ;
sec = solver seconds + planner seconds. Planner prompts are counted in full (conservative). 'in*' counts only the
per-question part of the planner prompt (what remains if the fixed instructions are cached).
Every strategy is reported as mean with a 95% bootstrap CI for accuracy."""
import argparse, json, random, collections, statistics as st, csv, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from planner import parse_choice

ap = argparse.ArgumentParser()
ap.add_argument("--fixed", required=True); ap.add_argument("--planner", nargs="*", default=[])
ap.add_argument("--heldout-fixed"); ap.add_argument("--csv"); ap.add_argument("--paired", action="store_true", help="paired bootstrap and exact sign test (McNemar) on per-question differences between key strategies")
ap.add_argument("--fallback", default="incident", help="encoding used when the planner output names no encoding (default: the plan's Always-Incident encoding, fixed in advance)"); ap.add_argument("--tol", type=float, default=0.05,
    help="lookup baseline: cheapest encoding whose held-out accuracy is within tol of the best for that task")
a = ap.parse_args()
load = lambda p: [json.loads(l) for l in open(p)]
fixed = load(a.fixed)
R = collections.defaultdict(dict)                       # R[qid][enc] = row
for r in fixed: R[r["id"]][r["encoding"]] = r
ENC = sorted({r["encoding"] for r in fixed}); qids = [q for q in R if len(R[q]) == len(ENC)]
task = {q: next(iter(R[q].values()))["task"] for q in qids}
print(f"questions with all {len(ENC)} encodings: {len(qids)} of {len(R)}")

def row_cost(q, enc, plan=None):
    r = R[q][enc]
    d = dict(ok=r["correct"], inp=r["in_tokens"], inp_star=r["in_tokens"], tot=r["in_tokens"] + r["out_tokens"], sec=r["seconds"])
    if plan:
        d["inp"] += plan["plan_in_tokens"]; d["inp_star"] += plan["plan_suffix_tokens"]
        d["tot"] += plan["plan_in_tokens"] + plan["plan_out_tokens"]; d["sec"] += plan["plan_seconds"]
    return d

strategies = {}                                         # name -> {qid: list of cost dicts (averaged if several)}
for e in ENC: strategies[f"fixed:{e}"] = {q: [row_cost(q, e)] for q in qids}
strategies["always_incident"] = strategies.get("fixed:incident")
strategies["random_per_query"] = {q: [row_cost(q, e) for e in ENC] for q in qids}   # exact expectation
strategies["always_shortest_prompt"] = {q: [row_cost(q, min(ENC, key=lambda e: R[q][e]["in_tokens"]))] for q in qids}
def oracle(q):
    ok = [e for e in ENC if R[q][e]["correct"]] or ENC
    return min(ok, key=lambda e: R[q][e]["in_tokens"])
strategies["oracle_cheapest_correct"] = {q: [row_cost(q, oracle(q))] for q in qids}
if a.heldout_fixed:
    H = collections.defaultdict(list); tk = collections.defaultdict(list)
    for r in load(a.heldout_fixed): H[(r["task"], r["encoding"])].append(r["correct"]); tk[(r["task"], r["encoding"])].append(r["in_tokens"])
    pick = {}
    for t in set(task.values()):
        acc = {e: st.mean(H[(t, e)]) for e in ENC if H[(t, e)]}; best = max(acc.values())
        pick[t] = min((e for e in acc if acc[e] >= best - a.tol), key=lambda e: st.mean(tk[(t, e)]))
    print("task-aware lookup picks:", pick)
    strategies["task_aware_lookup"] = {q: [row_cost(q, pick[task[q]])] for q in qids}
# 5-fold cross-validated task-aware lookup on the 500 test questions: the encoding for each task is chosen on 4/5 of that
# task's questions and applied to the remaining 1/5 (no question is used to choose its own encoding).
def lookup_pick(cal_qids, tol):
    pick = {}
    for t in set(task.values()):
        qs = [q for q in cal_qids if task[q] == t]
        acc = {e: st.mean(R[q][e]["correct"] for q in qs) for e in ENC}; best = max(acc.values())
        pick[t] = min((e for e in ENC if acc[e] >= best - a.tol), key=lambda e: st.mean(R[q][e]["in_tokens"] for q in qs))
    return pick
fold_of = {}
for t in set(task.values()):
    for i, q in enumerate(sorted(q for q in qids if task[q] == t)): fold_of[q] = i % 5
cv = {}; picks_seen = collections.defaultdict(collections.Counter)
for f in range(5):
    pk = lookup_pick([q for q in qids if fold_of[q] != f], a.tol)
    for t, e in pk.items(): picks_seen[t][e] += 1
    for q in qids:
        if fold_of[q] == f: cv[q] = [row_cost(q, pk[task[q]])]
strategies["task_aware_lookup_cv5"] = cv
print("cv5 lookup picks per task (fold counts):", {t: dict(c) for t, c in picks_seen.items()})
for p in a.planner:
    P = {r["id"]: r for r in load(p)}
    for r in P.values(): r["choice"], r["parsed"] = parse_choice(r["raw"], a.fallback)   # re-parse so parser fixes apply to old runs
    name = "planner:" + p.split("/")[-1].replace(".jsonl", "")
    miss = [q for q in qids if q not in P]
    if miss: print(f"WARNING {name}: {len(miss)} questions missing, skipped")
    strategies[name] = {q: [row_cost(q, P[q]["choice"], P[q])] for q in qids if q in P}
    pr = [P[q] for q in qids if q in P]
    print(f"{name}: parsed ok {sum(x['parsed'] for x in pr)}/{len(pr)} (fallback '{a.fallback}' used {sum(not x['parsed'] for x in pr)}x); choice counts {dict(collections.Counter(x['choice'] for x in pr))}")

rng = random.Random(1)
def agg(qd, key, idx=None):
    qs = idx if idx is not None else list(qd)
    return st.mean(st.mean(c[key] for c in qd[q]) for q in qs)
def ci(qd, key="ok", B=1000):
    qs = list(qd); vals = []
    for _ in range(B): vals.append(agg(qd, key, [rng.choice(qs) for _ in qs]))
    vals.sort(); return vals[int(.025 * B)], vals[int(.975 * B)]

rows = []
print(f"\n{'strategy':44s}{'n':>5s}{'acc%':>7s}{'  95% CI':>14s}{'in':>7s}{'in*':>7s}{'total':>7s}{'sec':>7s}")
for name, qd in strategies.items():
    lo, hi = ci(qd)
    r = dict(strategy=name, n=len(qd), acc=agg(qd, "ok") * 100, lo=lo * 100, hi=hi * 100, inp=agg(qd, "inp"),
             inp_star=agg(qd, "inp_star"), tot=agg(qd, "tot"), sec=agg(qd, "sec")); rows.append(r)
    print(f"{name:44s}{r['n']:5d}{r['acc']:7.1f}  [{r['lo']:4.1f},{r['hi']:5.1f}]{r['inp']:7.0f}{r['inp_star']:7.0f}{r['tot']:7.0f}{r['sec']:7.2f}")
print("\nper-task accuracy (%):")
tasks = sorted(set(task.values())); print(f"{'strategy':44s}" + "".join(f"{t[:11]:>12s}" for t in tasks))
for name, qd in strategies.items():
    print(f"{name:44s}" + "".join(f"{agg(qd,'ok',[q for q in qd if task[q]==t])*100:12.1f}" for t in tasks))
if a.paired:
    from math import comb
    def okv(name): return {q: st.mean(c["ok"] for c in strategies[name][q]) for q in strategies[name]}
    best_fixed = max((n for n in strategies if n.startswith("fixed:")), key=lambda n: agg(strategies[n], "ok"))
    pairs = [("task_aware_lookup_cv5", "always_incident"), ("task_aware_lookup_cv5", best_fixed)]
    plan_names = [n for n in strategies if n.startswith("planner:")]
    pairs += [(n, "task_aware_lookup_cv5") for n in plan_names if "hinted" in n] + [(n, "always_incident") for n in plan_names if "hinted" in n]
    if "task_aware_lookup" in strategies:   # added after review: the realistic 20-questions-per-task lookup (appended last, so earlier random draws are unchanged)
        pairs += [("task_aware_lookup", "always_incident"), ("task_aware_lookup", best_fixed)]
        pairs += [(n, "task_aware_lookup") for n in plan_names if "hinted" in n and "cv" not in n]
    pairs += [(n, "always_incident") for n in plan_names if "hinted" not in n]   # descriptions-only planners vs Always Incident (appended last, earlier draws unchanged)
    print(f"\npaired comparison (A minus B, accuracy points on the same questions; best single wording picked on the test set = {best_fixed}, which favours the baseline)")
    print(f"{'A':46s}{'B':26s}{'diff':>7s}{'  95% CI':>16s}{'A only':>8s}{'B only':>8s}{'p (sign)':>10s}")
    rng2 = random.Random(2)
    for A, Bn in pairs:
        va, vb = okv(A), okv(Bn); qs = [q for q in qids if q in va and q in vb]
        d = {q: va[q] - vb[q] for q in qs}; boots = []
        for _ in range(2000): boots.append(st.mean(d[rng2.choice(qs)] for _ in qs))
        boots.sort(); n10 = sum(d[q] > 0 for q in qs); n01 = sum(d[q] < 0 for q in qs); n = n10 + n01
        p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n10, n01) + 1)) / 2 ** n)
        print(f"{A:46s}{Bn:26s}{st.mean(d.values())*100:7.1f}  [{boots[50]*100:5.1f},{boots[1949]*100:5.1f}]{n10:8d}{n01:8d}{p:10.4f}")

if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", a.csv)
