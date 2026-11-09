# Placement 2 research records
Query-Aware Dynamic Selection of Graph Encodings for LLM Graph Reasoning
Maedeh Ramezanzadeh Tajanaki, FOSE7901 Placement 2, supervisor Dr. Venus Haghighi, Macquarie University, 2026.
Archive: https://github.com/maedehramezanzade-lang/graph-encoding-selection, DOI https://doi.org/10.5281/zenodo.24094353

## Folder layout
- `src/` code: graphs.py (questions), encoders.py (9 encodings), scoring.py (answer checking), run_fixed.py (solver runs),
  planner.py / run_planner.py / run_planner_cv.py (planner), make_hints.py, summarize.py, evaluate.py (all tables and tests).
- `data/` questions.jsonl (500 test questions, seed 7901), heldout.jsonl (100 calibration questions, seed 1234),
  hints_<model>.txt (practice-results block rebuilt from the calibration run with make_hints.py, K=2).
- `results/` raw outputs, one JSON line per question: `<model>_fixed.jsonl` (500 questions x 9 encodings = 4,500 lines),
  `<model>_heldout_fixed.jsonl` (900 lines), `<model>_planner_{balanced,cost,shuffled,hinted,hinted_cv}.jsonl` (500 lines each),
  `<model>_summary.csv` (summary table).
- `evaluation_outputs/` full printed output of evaluate.py --paired for each model (the source of every number in the report).
- `src/extra_analysis.py` (hint following, accuracy by graph size, chance level), `src/cross_planner.py` (planner of one model with the stored answers of the other; output evaluation_outputs/cross_planner.txt), `src/verify_hints.py`, `src/record_versions.py`; outputs in evaluation_outputs/.
- `tests/selftest.py` checks encoders, gold answers and scorer without a GPU.

## Reproduce
1. Questions: `python3 src/graphs.py` (needs `mkdir data`). The result is byte-identical to data/*.jsonl.
2. Solver runs (GPU, e.g. Colab T4):
   `python3 src/run_fixed.py --model Qwen/Qwen2.5-7B-Instruct --data data/questions.jsonl --out results/Qwen2.5-7B-Instruct_fixed.jsonl --batch 8 --load4bit`
   Same with `--data data/heldout.jsonl` for the calibration run. Phi: `microsoft/Phi-3.5-mini-instruct` without --load4bit.
3. Hints: `python3 src/make_hints.py results/<model>_heldout_fixed.jsonl 2 > data/hints.txt`
4. Planner: `python3 src/run_planner.py --model <model> --variant balanced|cost|shuffled|hinted --out results/...`; hinted-CV: `python3 src/run_planner_cv.py` (see its header).
5. Evaluation, no GPU needed:
   `python3 src/evaluate.py --fixed results/M_fixed.jsonl --heldout-fixed results/M_heldout_fixed.jsonl --planner results/M_planner_balanced.jsonl results/M_planner_cost.jsonl results/M_planner_shuffled.jsonl results/M_planner_hinted.jsonl results/M_planner_hinted_cv.jsonl --paired`
   (M = Qwen2.5-7B-Instruct or Phi-3.5-mini-instruct). Takes a few minutes because of the bootstraps. The paired block also tests the 20-question lookup table (task_aware_lookup) and the calibration-hinted planner against it.
   `python3 src/extra_analysis.py --prefix M` prints the hint-following counts, accuracy by graph size and chance levels used in Section 3 and Appendix E.

## Settings
Zero-shot, greedy decoding (do_sample=False), max 64 new tokens (solver) and 8 (planner). System message:
"You are a helpful assistant. Answer with only the final answer, no explanation."
Qwen2.5-7B-Instruct in 4-bit NF4 (bitsandbytes, float16 compute); Phi-3.5-mini-instruct in float16. Hugging Face Transformers, Colab T4.
Bootstrap: 1,000 resamples for intervals, 2,000 for paired tests; sign test is exact two-sided. CV5 folds: question index modulo 5 within each task.
Planner answers reuse the stored solver answer for the chosen encoding (valid because decoding is greedy). Fallback encoding: incident.

## Notes
- Software versions: see evaluation_outputs/environment_and_hint_verification.txt (Colab T4 runtime checked on 7 Nov 2026: Python 3.13.16, torch 2.14.1+cu130,
  transformers 5.22.1, accelerate 1.15.0, bitsandbytes 0.50.2, networkx 3.7, numpy 2.4.6, tokenizers 0.23.3, CUDA 13.0). The experiments ran on 22-24 Oct 2026
  with the same setup cell (pip install -U "bitsandbytes>=0.46.1"), which installs the latest versions; the versions listed were read on 7 Nov and were not saved on the run dates.
- The hints text used in the original planner runs was not saved as a file. data/hints_<model>.txt was rebuilt from the saved calibration run with make_hints.py.
  src/verify_hints.py rebuilds all hinted and hinted-CV prompts and compares their token count with the plan_in_tokens stored in the original runs:
  500/500 questions match for each of the four result files (Qwen and Phi, hinted and hinted-CV), so the rebuilt hints are the original hints.
- Small GPU numerical differences can change a few outputs on re-run.


## Licence and citation
