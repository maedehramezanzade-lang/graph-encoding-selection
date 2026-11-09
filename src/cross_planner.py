"""Separate the planner from the solver using stored results only (no model run).
Planner model P chose an encoding for each question (results/P_planner_<variant>.jsonl); the solver model S answered every
question in every encoding (results/S_fixed.jsonl). Accuracy of 'planner P + solver S' = S's stored correctness for P's choice.
Note: for the hinted variants the hints describe the planner model's own calibration results, not the solver's.
  python3 src/cross_planner.py"""
import json, random, statistics as st, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from planner import parse_choice
L = lambda p: [json.loads(l) for l in open(p)]
M = {"Qwen": "Qwen2.5-7B-Instruct", "Phi": "Phi-3.5-mini-instruct"}
fixed = {k: {(r["id"], r["encoding"]): r for r in L(f"results/{v}_fixed.jsonl")} for k, v in M.items()}
rng = random.Random(3)
print(f"{'planner':6s}{'solver':7s}{'variant':11s}{'acc%':>6s}{'  95% CI':>16s}{'own-planner acc%':>18s}{'distinct':>9s}{'top choice':>22s}")
for P in M:
    for S in M:
        for var in ["balanced", "cost", "shuffled", "hinted", "hinted_cv"]:
            rows = L(f"results/{M[P]}_planner_{var}.jsonl")
            ch = {r["id"]: parse_choice(r["raw"])[0] for r in rows}
            ok = [fixed[S][(q, e)]["correct"] for q, e in ch.items()]
            own = [fixed[P][(q, e)]["correct"] for q, e in ch.items()]
            bs = sorted(st.mean(rng.choice(ok) for _ in ok) for _ in range(1000))
            from collections import Counter
            c = Counter(ch.values()).most_common(1)[0]
            print(f"{P:6s}{S:7s}{var:11s}{st.mean(ok)*100:6.1f}  [{bs[25]*100:5.1f},{bs[975]*100:5.1f}]{st.mean(own)*100:18.1f}{len(set(ch.values())):9d}{c[0]+' ('+str(c[1])+')':>22s}")
