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

## 2026-10-22  E0 setup and smoke test
- Machine: macOS (Apple Silicon, 10 cores, 16 GB). `cargo build --release` failed with the
  "tapi error: unknown architecture" linker error under the default SDK (MacOSX27.0); it builds with
  `export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk` (every command below is run
  with that variable set). numpy was missing from the miniconda python and was installed with pip.
- `python3 run_all.py` -> results/ (log: e0_run_all.log). `cmp` against reference_results/: results.csv,
  results_train_cr.csv, results_summary.csv, results_per_language.csv all byte-identical.
- `python3 check.py` -> e0_check.log: 29/29 PASS ("All checks passed.").
- No TeX installation on the machine; `brew install texlive` started in the background for the paper.

## 2026-10-22  E1 main result with all baselines -> results_main/
- `nohup python3 error_bars.py --samples 5 --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --run-args "--worst-ns 2,3 --window 100" --out results_main > e1_main.log 2>&1 &`  PID 31015
- Finished (5 samples x 13 tokenizers, about 40 min; log e1_main.log). results_main/results_errorbars.csv: the ten rows
  shared with reference_results/errorbars/ are identical to the reference, as expected.
- Findings: monotone front 5/5 samples (Gini and compression both lower at every step up in p). Matched compression
  (error_bars.py): Hybrid-50 -0.0012+-0.0005 (hybrid fairer 5/5), Hybrid-75 +0.0013+-0.0003 (p-fair fairer 5/5),
  Hybrid-90 -0.0014+-0.0004 (hybrid 5/5); WorstN-2 -0.0002+-0.0003 (defined in 2/5 samples), WorstN-3 -0.0006+-0.0001
  (4/5), Window-100 -0.0002+-0.0003 (4/5): Worst-N and the window sit at the parity end (Gini 0.0025-0.0029,
  compression 6.632-6.637 vs Parity-aware 0.0028 / 6.634), on the curve within the sample spread; in the samples where
  their compression is below Parity-aware's the comparison is undefined. They are NOT an interpolation at N=2,3 here.
- regress table (max over non-English of t/t_classical): p=2..32 0.982-0.995, Parity-aware 0.999, Hybrid-75 1.024,
  Hybrid-90 1.013, Worst-N/window 0.997-0.999. underserved table (languages above the classical mean = hin, guj):
  no rule raises either (max ratio 0.937-0.991), so at four languages the two tables differ only through Marathi.
- Caveat found while checking E2 folders: at 64k/128k/256k p=32 raises Marathi above its classical cost
  (ratios 1.006/1.035/1.048) and so does p=32 at 1x English (1.043) -- the "no Indian language worse off" claim
  holds in the main setting only (relevant to the E3 TODO items). Tables: results_main/paper_front.csv,
  paper_perlang.csv, paper_regress.csv, paper_underserved.csv, paper_sens.csv (E2 table).

## 2026-10-22  E4 (cost unit) and E2 (sensitivity), one background chain running alongside E1
- Chain script (E4 first, then E2 in the order of experiments.md): PID 31050, log e2_e4.log
  - `python3 run_all.py --ps 1,8,inf --unit byte --out results_unit_byte`
  - `python3 run_all.py --ps 1,8,inf --unit char --out results_unit_char`
  - `python3 run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --eng-chars-m 20 --data data_eng1 --out results_eng1`
  - `... --eng-chars-m 160 --data data_eng8 --out results_eng8`
  - `... --vocab-size 64000 --out results_64k`, `--vocab-size 128000 --out results_128k`, `--vocab-size 256000 --out results_256k`
- All seven runs finished in about 25 minutes on this machine (chain exit 0, log ends with E2_E4_CHAIN_DONE).
- E4 findings: per byte the parity rule gives English 98.9% of the merges (train_log.tsv lead_lang) and the Indic
  languages cost about twice their classical tokens per sentence (Hindi 63.6 vs 35.0, Marathi 70.6, Gujarati 66.5;
  English 26.6 vs 29.0) -- identical to the interim Table 4. Per character the inversion disappears (English leads 22%),
  but the per-sentence Gini of the char-unit Parity-aware tokenizer is 0.010 (vs 0.003 per sentence) with English now
  the most expensive language (32.9 vs 31.3 Gujarati), because sentence length in characters differs by language.
- E2 findings (single runs): Gini at 32k/64k/128k/256k: classical 0.0380/0.0378/0.0378/0.0383 (flat), Parity-aware
  0.0028/0.0032/0.0035/0.0026 (flat), p=8 0.0229/0.0255/0.0287/0.0316 (RISES). Cost of full parity grows with size:
  4.0% / 5.1% / 7.6% / 10.7%. Matched-compression vs hybrid (compare_hybrid.py): at 64k, 128k and 256k p-fair is fairer
  than all three hybrids (differences 0.0003-0.0075, growing with size); at 1x English p-fair fairer at 75/90% by
  0.004; at 8x English the hybrid is fairer at 75/90% by 0.002-0.004. The Lee et al. effect (Parity-aware losing
  equity with size) does NOT appear; instead intermediate p loses equity with size, contrary to the hypothesis in
  Section 6.2 of the draft -- reported as such in the paper.
