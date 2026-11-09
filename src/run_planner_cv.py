"""Cross-validated hinted planner: for each of 5 folds of the 500 test questions, the planner's hints are built from the
fixed-run results of the OTHER 4 folds (80 questions per task) and the planner is run only on the held-out fold.
No question's own results are used to build the hints it is planned with. Same fold assignment as evaluate.py.

  python3 src/run_planner_cv.py --model Qwen/Qwen2.5-7B-Instruct --fixed results/X_fixed.jsonl \
      --out results/X_planner_hinted_cv.jsonl --load4bit
Then add this file to evaluate.py --planner. Resumable."""
import argparse, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(__file__))
from planner import planner_parts, parse_choice
from make_hints import build_hints
from run_planner import DummyPlanner

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="dummy"); ap.add_argument("--data", default="data/questions.jsonl")
ap.add_argument("--fixed", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--backend", default="hf", choices=["hf", "dummy"]); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--load4bit", action="store_true"); ap.add_argument("--top", type=int, default=2)
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.data)]
fixed = [json.loads(l) for l in open(a.fixed)]
task = {r["id"]: r["task"] for r in recs}
fold_of = {}
for t in set(task.values()):
    for i, q in enumerate(sorted(q for q in task if task[q] == t)): fold_of[q] = i % 5
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
if a.backend == "dummy": llm = DummyPlanner()
else:
    from run_fixed import HF
    llm = HF(a.model, load4bit=a.load4bit)
f = open(a.out, "a")
for k in range(5):
    hints = build_hints([r for r in fixed if fold_of[r["id"]] != k], a.top)
    print(f"--- fold {k}: hints built from the other folds ---\n{hints}", flush=True)
    todo = [r for r in recs if fold_of[r["id"]] == k and r["id"] not in done]
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        parts = [planner_parts(r, "hinted", hints) for r in chunk]
        t0 = time.time(); outs = llm.generate([p + s for p, s in parts], [None] * len(chunk), max_new=8); dt = time.time() - t0
        for r, (pre, suf), (text, n_in, n_out) in zip(chunk, parts, outs):
            choice, ok = parse_choice(text)
            f.write(json.dumps({"model": a.model, "variant": "hinted_cv", "fold": k, "id": r["id"], "task": r["task"],
                                "raw": text, "choice": choice, "parsed": ok, "plan_in_tokens": n_in,
                                "plan_suffix_tokens": llm.count(suf), "plan_out_tokens": n_out,
                                "plan_seconds": dt / len(chunk)}) + "\n")
        f.flush()
print("done ->", a.out)
