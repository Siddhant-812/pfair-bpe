# Pilot: 14 languages at natural data balance (single run, 2026-10-22)

    python run_all.py --langs hin,mar,guj,ben,pan,ori,asm,san,tam,tel,kan,mal,urd,eng --natural --total-chars-m 400 \
        --vocab-size 64000 --ps 1,4,8,16,32,64,128,inf --hybrids 0.5,0.75,0.9 --worst-ns 3 --window 100

Text per language in proportion to its Sangraha split (SIZES.txt): Hindi 75M characters down to
Assamese 2M. About 2 minutes per tokenizer on two cores, 3 GB of memory. Single run, no error bars:
treat every number as provisional until E3 of experiments.md has been run with samples.

What it showed (results_summary.csv, paper_perlang.csv, paper_underserved.csv):

- The front is monotone again: Gini 0.066 (classical) -> 0.037 (p=8) -> 0.015 (p=32) -> 0.004 (p=inf);
  training-text compression 8.94 -> 8.85 -> 8.64 -> 8.33 bytes/token.
- Matched compression against the hybrid (compare_hybrid.py): p-fair fairer by 0.003 at the
  Hybrid-50 end, tie at 75%, hybrid fairer by 0.002 at the 90% end. Worst-N (N=3) and the moving
  window sit at the Parity-aware end, on the curve.
- The "no language worse off than classical" property of the 4-language setting does NOT hold:
  every fairness rule raises Bengali and Hindi, the cheapest languages (Bengali 30.4 -> 32.7 at p=8).
  What does hold: p-fair never raises a language that costs MORE than the classical mean
  (max ratio 0.96-0.98 at every p), whereas Hybrid-75 and Hybrid-90 raise Gujarati above its
  classical cost (37.1 -> 38.1) and Hybrid-90 raises Malayalam too. paper_underserved.csv has it.
- Compression on the balanced test set is best at p=16 (9.13 bytes/token vs 8.91 classical).
