# p-fair BPE: one dial from compression to parity

Code for "One Dial from Compression to Parity: A Power-Mean View of Fair BPE
Tokenization". One Rust program trains every tokenizer in the paper, because
they are the same merge rule with different settings:

    score(a,b) = sum over languages l of  w_l * ( c_l(a,b) - lambda * cross_l(a,b) )

    w_l = t_l^(p-1)          t_l = current tokens per parallel sentence of language l
    c_l = pair count in l's training text;  cross_l = those occurrences that join two morphemes

| Setting | Tokenizer |
|---|---|
| `--p 1` | classical BPE (all weights 1) |
| `--p 2 .. 32` | p-fair BPE |
| `--p inf` | Parity-aware BPE, base rule (Foroutan et al., 2026) |
| `--p inf --hybrid 0.75` | hybrid baseline: first 75% of merges classical, rest parity |
| `--p inf --window 100 --alpha 2` | moving-window Parity-aware baseline |
| `--p inf --worst-n 2` | Worst-N parity baseline (Kanjirangat et al., 2026) |
| `--lambda 0.5 --morph-lexicon F` | any of the above plus the morphology penalty |
| `--pi-uniform` | equal language weights instead of text-share weights |

`experiments.md` is the full plan: every experiment of the paper, the command
that runs it, how long it takes, which table it fills, and what to conclude
from each outcome. `CLAUDE.md` holds the working rules for an AI coding
assistant in this repository.

## Setup

1. Rust: https://rustup.rs (one command), then open a new terminal.
   On macOS with a brand-new SDK and an older linker, `cargo build` may fail with
   "tapi error: unknown architecture"; then `export SDKROOT=$(xcrun --show-sdk-path)`
   or point `SDKROOT` at an older SDK in `/Library/Developer/CommandLineTools/SDKs/`.
2. `pip install pyarrow huggingface_hub numpy` (and `torch transformers` for the LM experiment).
3. `python run_all.py` builds the trainer, downloads the data and runs the
   default experiment (4 languages, 32k, classical / Parity-aware / p=2 / p=4).
4. `python check.py` runs the correctness checks (about 2 minutes).

## Scripts

| Script | What it does |
|---|---|
| `run_all.py` | one experiment: download data, train the requested tokenizers, evaluate (`python run_all.py --help`) |
| `error_bars.py` | repeat an experiment on several samples of training text; bootstrap over test sentences; compare baselines at matched compression (`--run-args "..."` passes anything on to run_all.py) |
| `compare_hybrid.py` | baselines against the p-fair front at equal compression, for one results folder |
| `get_data.py` | FLORES-200 dev/devtest and a Sangraha sample for any of 20 languages (`--sizes`, `--natural`, `--part`) |
| `morph_prepare.py` | split gold segmentations into a training lexicon and a held-out test set |
| `allocate.py` | vocabulary allocation across scripts: curve fitting, convex (KKT) split, baselines, merged tokenizers |
| `downstream/train_lm.py` | small GPT-2 style LM per tokenizer; bits per byte per language on FLORES devtest |
| `downstream/term_fertility.py` | tokens per technical term from term lists |
| `paper_tables.py` | LaTeX tables and pgfplots CSVs from the result folders |
| `check.py` | correctness checks of the trainer (fast == brute force for every rule, p=1 == textbook BPE, ...) |

## The trainer

    target/release/pfair help

    pfair train  --langs hin,mar,guj,eng --train-dir DIR --dev-dir DIR --unit line --p P --vocab-size N --out DIR
    pfair eval   --langs ... --model DIR --test-dir DIR --unit line [--summary F --detail F --csv-train F --per-line F --train-dir DIR]
    pfair morph-eval --model DIR --gold FILE [--lang L --name N --csv F]
    pfair encode / encode-ids --model DIR --input FILE
    pfair merge  --models A,B,C --out DIR

Text files are `DIR/<lang>.txt`, one sentence per line. Costs are measured in
tokens per parallel sentence on `--dev-dir` while training (FLORES dev) and
reported on the test directory (FLORES devtest), which training never sees.

## What comes out of `run_all.py`

| File | Contents |
|---|---|
| `results_summary.csv` | every metric, one row per tokenizer (Gini per line is the parity measure; `train_bytes_per_token` the compression measure) |
| `results_per_language.csv` | tokens per sentence of each language under each tokenizer |
| `results_morph.csv` | boundary F1 / MorphScore per language (with `--morph-gold`) |
| `results.csv`, `results_train_cr.csv` | `method,cr,gini` with test-set or training-text compression |
| `runs.json`, `config.json` | which model folder belongs to which name; the exact settings |
| `models/<name>/` | `model.txt`, `vocab.txt`, `train_log.tsv` (one row per merge: token, leading language, score, every language's cost), `train_stats.tsv`, `test_tokens_per_line.tsv` |

`reference_results/` holds the outputs of the runs reported in the interim
report, for comparison; the default `run_all.py` reproduces `reference_results/results*.csv`
exactly.

## Settings that matter

- **Cost unit.** `--unit line` (tokens per parallel sentence) is the paper's
  setting. Per byte, 3-byte Indic scripts look cheap and the parity rules send
  their merges to English (Section 6.1 of the paper); `--unit byte` reproduces that.
- **Data balance.** The default gives English four times the text of each other
  language (`--eng-chars-m 80 --chars-m 20`). `--sizes` and `--natural` set other mixes.
- **Morphology penalty.** `0 <= lambda <= 1`; a crossing occurrence of a pair
  counts `(1 - lambda)` instead of 1. Above 1 the pair scores are no longer
  monotone and only `--naive` is exact. Boundaries come from a lexicon of
  segmented words; words outside the lexicon contribute no crossings.
- **Determinism.** Training is deterministic given the data; ties between pairs
  go to the smaller pair id. Two runs on one machine give identical files.

## Checks

`python check.py` verifies that the lazy-evaluation trainer learns exactly the
same merges as brute-force rescoring for p = 1, 2, 4, 8, inf, for the hybrid,
Worst-N and moving-window baselines and for the morphology penalty; that p = 1
reproduces a textbook BPE written separately in Python; that the evaluator and
the trainer agree on token counts; and that merged per-script models behave
like their parts.
