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

## 2026-10-22  Phase 2 launched (after STOP 1)
- E3: chain script (single run -> compare_hybrid.py -> 3-sample error bars), PID 36559, log e3_l14.log
  `python3 run_all.py --langs hin,mar,guj,ben,pan,ori,asm,san,tam,tel,kan,mal,urd,eng --natural --total-chars-m 400 --vocab-size 64000 --ps 1,2,4,8,16,32,64,128,inf --hybrids 0.5,0.75,0.9 --worst-ns 2,3,5 --window 100 --data data_l14 --out results_l14`
  then `python3 error_bars.py --samples 3 --ps 1,2,4,8,16,32,64,128,inf --hybrids 0.5,0.75,0.9 --run-args "--langs ... --natural --total-chars-m 400 --worst-ns 2,3,5 --window 100" --vocab-size 64000 --out results_l14_eb`
- E6 (3 scripts): `python3 allocate.py --clusters "Deva=hin,mar;Gujr=guj;Latn=eng" --ps 1,8 --vocab-size 32000 --out results_alloc`  PID 36560, log e6_alloc.log
- `pip install torch transformers` for E7 (PID 36561); E5 queued until the SuTRA gold files are provided.
- torch 2.14.1 (MPS available) and transformers 5.20.2 installed.
- E6 (3 scripts) finished in ~4 min (results_alloc/). Curve fits: RMSE 0.02-0.03 tokens/sentence (a=1214/1051/2333,
  b=0.5/0.5/0.6, d=21.3/20.4/20.9 for Deva/Gujr/Latn). Budgets (Deva/Gujr/Latn): uniform 11290/11288/11288,
  data-prop 9717/4706/19029, union-prop 16943/8471/8471, convex p=1 11201/6275/16091, convex p=8 13255/7937/12598.
  Results (Gini / train compression): joint p=1 0.0380/6.904, joint p=8 0.0233/6.862, uniform 0.0263/6.787,
  data-prop 0.0507/6.875, union-prop 0.0187/6.668, convex p=1 0.0379/6.901, convex p=8 0.0264/6.855.
  Findings: convex p=1 is the best-compressing split and within 0.05% of the joint tokenizer (the pilot's
  "splitting is costly" warning does not show at this scale: 6.901 vs 6.904); convex p=8 is NOT the fairest split
  (union-proportion has Gini 0.019 but 3.4% worse compression); convex p=8 dominates the uniform split (same Gini,
  1% better compression) and sits 0.003 in Gini / 0.1% compression behind joint p=8. Reported as such in Section 6.5.
- E3 single run finished (16 tokenizers, ~15 min). compare_hybrid.py: Hybrid-50 p-fair fairer by 0.0032, Hybrid-75
  hybrid by 0.0002, Hybrid-90 hybrid by 0.0022; WorstN-2/3/5 and Window-100 within 0.0003 of the curve at the parity
  end -- same picture as the pilot. Error-bar run (results_l14_eb) continues in the same chain.
- E6 (10 scripts): `python3 allocate.py --data data_l14 --vocab-size 64000 --ps 1,8 --out results_alloc_l14 --clusters "Deva=hin,mar,san;Gujr=guj;Beng=ben,asm;Guru=pan;Orya=ori;Taml=tam;Telu=tel;Knda=kan;Mlym=mal;Arab=urd;Latn=eng"`  PID 37484, log e6_alloc_l14.log
- E6 (10 scripts) first run: BUG FOUND. Every merged tokenizer cost about twice the joint one (English 68 tokens per
  sentence vs 32 for the Latin-only part). Cause: `pfair merge` identified tokens by their bytes, and the Bengali and
  Urdu training text contains the word-marker character U+2581 itself, so the alphabet entry overwrote the marker and
  every word-initial merge of every model was mis-mapped. Fix in src/main.rs (merge_models): tokens are now identified
  by their decomposition into merged base ids. Verified on the Latn+Beng+Arab parts (English 31.98 after the fix).
  The 3-script run was not affected (no U+2581 in those alphabets) but is rerun for consistency. The wrong folders are
  kept as results_alloc_premergefix/ and results_alloc_l14_premergefix/ (logs e6_alloc*_premergefix.log).
- After the fix: `cargo build --release` ok; chain PID 39018 (log check_e6_chain.log): `python3 check.py`
  (check_after_mergefix.log) then both allocate.py runs again (results_alloc/, results_alloc_l14/).
- check.py after the merge fix: 29/29 PASS (check_after_mergefix.log). E6 reruns: results_alloc/ byte-identical to the
  pre-fix run (no U+2581 in those alphabets); results_alloc_l14/ now sensible (~20 min; e6_alloc_l14.log).
- E6 findings (Gini / train bytes per token). 3 scripts @32k: joint p=1 0.0380/6.904, joint p=8 0.0233/6.862,
  uniform 0.0263/6.787, data-prop 0.0507/6.875, union-prop 0.0187/6.668, convex p=1 0.0379/6.901, convex p=8
  0.0264/6.855. 10 scripts @64k: joint p=1 0.0664/8.942, joint p=8 0.0373/8.848, uniform 0.0571/8.559, data-prop
  0.0843/8.884, union-prop 0.0475/8.775, convex p=1 0.0649/8.929, convex p=8 0.0515/8.847. Curve-fit RMSE <= 0.03
  (3 scripts) and <= 0.15 (10 scripts, Orya). Conclusions written in Section 6.5: the convex p=1 split is the
  best-compressing split, within 0.05% / 0.14% of the joint tokenizer (the pilot's "splitting is costly" warning
  does not hold: only the uniform split pays, 1.7% / 4.3%); convex p=8 lowers the Gini of the split (0.038->0.026,
  0.065->0.052) at the joint p=8 compression but the joint p=8 rule is fairer (0.023, 0.037): allocation is a
  weaker lever than the merge rule; union-proportion is fairer than convex p=8 but 2.7% / 0.8% less compressed;
  convex p=8 raises Assamese (42.4 -> 45.5) because its cluster cost is dominated by Bengali (25x the text).
  Budgets and fits in Appendix B (tab:alloc3, tab:alloc10).
- E3 error bars finished (results_l14_eb/, 3 samples x 16 tokenizers, ~1 h). Spread 0.0001-0.0007 in Gini.
  Matched compression: Hybrid-50 +0.0030+-0.0003 (p-fair fairer 3/3), Hybrid-75 -0.0002+-0.0001 (0/3),
  Hybrid-90 -0.0023+-0.0002 (0/3); WorstN-2/3/5 and Window-100 within 0.0002 of the curve (ties). Monotone front
  3/3 at every step; bootstrap: each step lowers Gini in 100% of resamples except p=128 -> Parity-aware (71.5%).
  The pilot picture holds with error bars: p-fair fairer at the fair end, hybrid fairer at the mild end, by amounts
  ~10x the spread but ~4% of the front's span; no headline dominance.

## 2026-10-23  E8 training-time table -> results_speed/
- `nohup /private/tmp/claude-501/-Users-siddhant-Desktop-pfair-bpe/f8f0011d-0444-42b2-8e1a-8d058149434b/scratchpad/e8.sh > e8_speed.log 2>&1 &` PID 41521: `python3 speed_table.py --out results_speed` (4 languages, 32k)
  then the same with the 14 languages / data_l14 / 64k; lazy and naive, p = 1, 8, inf. Machine otherwise idle.
- PAUSED on request (2026-10-23) after the 4-language rows below; the processes were killed and the temporary model
  folders removed. results_speed/timings.csv keeps the finished rows. To resume tomorrow run only the missing settings
  (speed_table.py appends rows): e.g. `python3 speed_table.py --out results_speed --ps inf` for a missing 4-language
  row, then `python3 speed_table.py --out results_speed --langs hin,mar,guj,ben,pan,ori,asm,san,tam,tel,kan,mal,urd,eng
  --train-dir data_l14/train --dev-dir data_l14/flores/dev --vocab-size 64000` for the 14-language rows (~2 h).
- Findings so far: lazy 4.6 s vs naive 120 s at p=1 (26x), 3.2 heap re-checks per merge (see timings.csv for the rest).
