"""Check that the planner prompts rebuilt from the saved data are exactly the prompts used in the original runs.
The original hint text was not saved, but each planner result row stores plan_in_tokens (prefix + hints + suffix). We rebuild every
hinted / hinted_cv prompt and compare its token count with the stored one for all 500 questions. A full match means the rebuilt
hints are the original hints (any change in wording or numbers would change the count for at least some questions).

  python3 src/verify_hints.py --model Qwen/Qwen2.5-7B-Instruct --prefix Qwen2.5-7B-Instruct      (needs `transformers`, no GPU)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from planner import planner_parts
from make_hints import build_hints

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--prefix", required=True)
ap.add_argument("--data", default="data/questions.jsonl"); ap.add_argument("--results", default="results")
ap.add_argument("--dummy", action="store_true", help="whitespace token count, only to test the script")
a = ap.parse_args()
if a.dummy: count = lambda t: len(t.split())
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    count = lambda t: len(tok(t, add_special_tokens=False).input_ids)
L = lambda p: [json.loads(l) for l in open(p)]
recs = {r["id"]: r for r in L(a.data)}
fixed = L(f"{a.results}/{a.prefix}_fixed.jsonl")
heldout = L(f"{a.results}/{a.prefix}_heldout_fixed.jsonl")
task = {q: r["task"] for q, r in recs.items()}
fold_of = {}
for t in set(task.values()):
    for i, q in enumerate(sorted(q for q in task if task[q] == t)): fold_of[q] = i % 5
hints_all = build_hints(heldout, 2)
hints_fold = {k: build_hints([r for r in fixed if fold_of[r["id"]] != k], 2) for k in range(5)}
for variant in ("hinted", "hinted_cv"):
    rows = L(f"{a.results}/{a.prefix}_planner_{variant}.jsonl"); ok = 0
    for r in rows:
        h = hints_all if variant == "hinted" else hints_fold[fold_of[r["id"]]]
        pre, suf = planner_parts(recs[r["id"]], "hinted", h)
        ok += int(count(pre + suf) == r["plan_in_tokens"])
    print(f"{a.prefix} {variant}: rebuilt prompt has the stored token count for {ok}/{len(rows)} questions")
