"""Run the planner on every question: record chosen encoding and the planner's own cost.
The solver answer for the chosen encoding is looked up from the fixed run (greedy decoding is deterministic),
so no extra solver calls are needed. Resumable.

  python3 src/run_planner.py --model Qwen/Qwen2.5-7B-Instruct --out results/qwen7b_planner_balanced.jsonl --variant balanced --load4bit
"""
import argparse, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(__file__))
from encoders import ENCODINGS
from planner import planner_parts, parse_choice


class DummyPlanner:
    def __init__(self): self.rng = random.Random(0)
    def count(self, t): return max(1, len(t) // 4)
    def generate(self, prompts, metas, max_new=None):
        return [(self.rng.choice(ENCODINGS), self.count(p), 2) for p in prompts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dummy"); ap.add_argument("--data", default="data/questions.jsonl")
    ap.add_argument("--out", required=True); ap.add_argument("--variant", default="balanced", choices=["balanced", "cost", "shuffled", "hinted"])
    ap.add_argument("--backend", default="hf", choices=["hf", "dummy"]); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hints", default="data/hints.txt"); ap.add_argument("--load4bit", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.data)]
    if a.limit: recs = recs[:a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    if a.backend == "dummy": llm = DummyPlanner()
    else:
        from run_fixed import HF
        llm = HF(a.model, load4bit=a.load4bit)
    hints = open(a.hints).read() if a.variant == "hinted" else ""
    todo = [r for r in recs if r["id"] not in done]
    f = open(a.out, "a")
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        parts = [planner_parts(r, a.variant, hints) for r in chunk]
        prompts = [p + s for p, s in parts]
        t0 = time.time(); outs = llm.generate(prompts, [None] * len(chunk), max_new=8); dt = time.time() - t0
        for r, (pre, suf), (text, n_in, n_out) in zip(chunk, parts, outs):
            choice, ok = parse_choice(text)
            f.write(json.dumps({"model": a.model, "variant": a.variant, "id": r["id"], "task": r["task"],
                                "raw": text, "choice": choice, "parsed": ok,
                                "plan_in_tokens": n_in, "plan_suffix_tokens": llm.count(suf),
                                "plan_out_tokens": n_out, "plan_seconds": dt / len(chunk)}) + "\n")
        f.flush()
    print(f"planner done: {len(todo)} new questions -> {a.out}")


if __name__ == "__main__":
    main()
