"""Fixed-encoding runs: every (model, encoding, question) -> output, correctness, tokens, time.
Resumable: results are appended to a JSONL file and finished items are skipped on restart.

  python3 src/run_fixed.py --model Qwen/Qwen2.5-7B-Instruct --data data/questions.jsonl \
      --out results/qwen7b_fixed.jsonl --backend hf --batch 8
  python3 src/run_fixed.py --backend dummy ...    # pipeline test, no GPU needed
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from encoders import ENCODINGS, build_prompt
from scoring import score, gold_text

SYSTEM = "You are a helpful assistant. Answer with only the final answer, no explanation."


class Dummy:
    """Returns the gold answer for even question ids and junk otherwise (tests the pipeline)."""
    def __init__(self, *a, **k): pass
    def count(self, text): return max(1, len(text) // 4)
    def generate(self, prompts, metas):
        return [(gold_text(m[0], m[1]) if hash(m[1]["id"]) % 2 == 0 else "I do not know", self.count(p), 5)
                for p, m in zip(prompts, metas)]


class HF:
    def __init__(self, model, max_new=64, load4bit=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch, self.max_new = torch, max_new
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map="auto")
        if load4bit:   # fits 7-8B models on a 16 GB T4
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_quant_type="nf4")
        else:          # T4 has no native bfloat16 -> use float16
            kw["torch_dtype"] = torch.float16
        self.m = AutoModelForCausalLM.from_pretrained(model, **kw).eval()

    def chat(self, p):
        return self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                            tokenize=False, add_generation_prompt=True)

    def count(self, text):  # input tokens of the user prompt itself (what the encoding costs)
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def generate(self, prompts, metas, max_new=None):
        """On CUDA out-of-memory, free the cache and retry the batch in two halves (until a single prompt fits)."""
        try:
            return self._generate(prompts, metas, max_new)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(prompts) == 1: raise
            h = len(prompts) // 2
            print(f"  OOM with batch {len(prompts)} -> retrying as {h}+{len(prompts) - h}", flush=True)
            return (self.generate(prompts[:h], metas[:h], max_new) + self.generate(prompts[h:], metas[h:], max_new))

    def _generate(self, prompts, metas, max_new=None):
        torch = self.torch
        enc = self.tok([self.chat(p) for p in prompts], return_tensors="pt", padding=True).to(self.m.device)
        with torch.no_grad():
            out = self.m.generate(**enc, max_new_tokens=max_new or self.max_new, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        res = []
        for i, p in enumerate(prompts):
            gen = out[i, enc.input_ids.shape[1]:]
            n_out = int((gen != self.tok.pad_token_id).sum())
            res.append((self.tok.decode(gen, skip_special_tokens=True), self.count(p), n_out))
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dummy"); ap.add_argument("--data", default="data/questions.jsonl")
    ap.add_argument("--out", required=True); ap.add_argument("--backend", default="hf", choices=["hf", "dummy"])
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N questions per task (quick test)")
    ap.add_argument("--encodings", default=",".join(ENCODINGS))
    ap.add_argument("--load4bit", action="store_true", help="4-bit weights (for 7-8B models on a T4)")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.data)]
    if a.limit:
        seen, keep = {}, []
        for r in recs:
            seen[r["task"]] = seen.get(r["task"], 0) + 1
            if seen[r["task"]] <= a.limit: keep.append(r)
        recs = keep
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            d = json.loads(l); done.add((d["encoding"], d["id"]))
    llm = Dummy() if a.backend == "dummy" else HF(a.model, load4bit=a.load4bit)
    f = open(a.out, "a")
    for enc in a.encodings.split(","):
        todo = [r for r in recs if (enc, r["id"]) not in done]
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            prompts = [build_prompt(enc, r) for r in chunk]
            t0 = time.time(); outs = llm.generate(prompts, [(enc, r) for r in chunk]); dt = time.time() - t0
            for r, (text, n_in, n_out) in zip(chunk, outs):
                f.write(json.dumps({"model": a.model, "encoding": enc, "id": r["id"], "task": r["task"],
                                    "output": text, "correct": score(enc, r, text), "in_tokens": n_in,
                                    "out_tokens": n_out, "seconds": dt / len(chunk)}) + "\n")
            f.flush()
        print(f"{enc}: {len(todo)} new", flush=True)


if __name__ == "__main__":
    main()
