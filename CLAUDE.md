# Working rules for this repository

Read `experiments.md` first: it is the plan for the paper, with one section per
experiment (goal, command, runtime, output, which table it fills, what to
conclude). Work through it in order unless told otherwise, and record what
was run and what came out in `experiments_log.md` (create it; one entry per
run with the command, the date, the results folder and two lines of findings).

## Rules

- Never edit or delete `reference_results/`. It is the record of the interim
  report. New results go in new folders (`results_<experiment>/`).
- After any change to `src/main.rs`, run `cargo build --release` and
  `python check.py`; all checks must pass before results are produced. If a
  check fails, fix the trainer, do not relax the check.
- Keep the default `python run_all.py` reproducing `reference_results/results*.csv`
  exactly (byte for byte). This is the regression test of the whole pipeline.
- Do not retype numbers into the paper. Generate tables with `paper_tables.py`
  (or a small script in the same spirit) and paste its output; keep the CSV it
  writes next to the results.
- Results folders are large (models, id files). Do not commit them; `.gitignore`
  already excludes `results*/`, `data*/`, `target/`.
- Long runs: launch with `nohup ... > log 2>&1 &`, note the PID in
  `experiments_log.md`, and check the log rather than re-running.
- The data download needs internet (Hugging Face and dl.fbaipublicfiles.com);
  everything else is offline.
- GPU: only `downstream/train_lm.py` needs one. Everything else is CPU and
  takes seconds to minutes per tokenizer.

## Conventions

- Languages are the three-letter names in `get_data.py` (`hin`, `mar`, `guj`,
  `eng`, `ben`, `tam`, ...). Files are `DIR/<lang>.txt`, one sentence per line.
- Method names in CSVs: `Classical`, `p=8`, `Parity-aware`, `Hybrid-75`,
  `WorstN-2`, `Window-100`, `p=8,lambda=0.5`. `compare_hybrid.is_pfair()` decides
  which rows form the p-fair curve.
- The parity measure is `gini_per_line`; the compression measure is
  `train_bytes_per_token` (bytes per token on the training text). Report both,
  plus `bytes_per_token` on the balanced test set, as in the paper.
- Error bars: mean and standard deviation over samples of training text
  (`error_bars.py`), plus the paired bootstrap over test sentences it prints.

## Paper

The LaTeX source is in `paper/` (ACL style, `main.tex` includes `sec/*.tex`).
Sections marked "(planned)" and cells marked "--" are waiting for results from
`experiments.md`. Replace them, then delete the "planned" markers and the
comments that say so. Compile with `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
