# Experiment log

One entry per run: date, command, results folder, findings. Newest at the bottom.

## 2026-10-09  interim report runs (kept in reference_results/)
- `python run_all.py` -> reference_results/results*.csv (4 languages, 32k, eng 4x; p = 1, 2, 4, inf)
- `python run_all.py --ps 1,2,4,8,16,32,inf` -> reference_results/sweep_summary.csv
- `python run_all.py --unit byte` -> reference_results/byte_unit_per_language.csv (parity rule inverted per byte)
- `python error_bars.py` (5 samples, hybrids 0.5/0.75/0.9) -> reference_results/errorbars/
  Findings: monotone front in 5/5 samples; p=8 cuts Gini 40% for 0.6% compression; hybrid lies on the same curve
  (differences ~0.001, both signs); no non-English language worse off than classical for any finite p.

## 2026-10-22  pilot of the 14-language natural-balance setting (kept in reference_results/l14_preview/)
- `python run_all.py --langs hin,mar,guj,ben,pan,ori,asm,san,tam,tel,kan,mal,urd,eng --natural --total-chars-m 400
   --vocab-size 64000 --ps 1,4,8,16,inf --hybrids 0.5,0.75,0.9 --worst-ns 3 --window 100`
  plus `--ps 32,64,128` in a second run, merged into one folder.
  Runtime: ~2 min per tokenizer on 2 cores, ~3 GB memory. Findings: monotone front; p-fair fairer than
  Hybrid-50 by 0.003 at equal compression, tie at 75%, hybrid fairer by 0.002 at 90%; every rule raises
  Bengali/Hindi (the cheapest); p-fair never raises an under-served language, the hybrid raises Gujarati
  and Malayalam. See the folder's README.
