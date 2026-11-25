# p-fair BPE: one dial from compression to parity

Code and reference results for **p-fair BPE**, a byte-pair-encoding trainer whose single
parameter `p` moves a multilingual vocabulary from classical BPE (`p = 1`, best total
compression) to Parity-aware BPE (`p = ∞`, equal token cost for every language), with the
intermediate tokenizers trained exactly and an order of magnitude faster than naive
rescoring. Everything in the paper's tables is produced by the scripts in this repository
from public data (Sangraha for training, FLORES-200 for evaluation).

![Compression–parity front, four languages](figures/front_4lang.png)

## The problem

Subword tokenizers learn their merges from frequency counts, so a language with less
training text gets fewer merges and pays more tokens for the same content. On parallel
FLORES sentences, classical BPE trained with four times as much English as Hindi, Marathi
or Gujarati spends 35 tokens on a Hindi sentence and 29 on its English translation. Tokens
are what API pricing, latency and context-window use are counted in, so this is a cost
borne by the speakers of under-served languages.

Parity-aware BPE (Foroutan et al., 2026) removes the disparity by always helping the
currently most expensive language, at a loss in overall compression. Between the two ends
the literature offers heuristics (switch from one rule to the other after a fixed share of
merges, take the worst N languages, a moving window) that state no objective. The question
this code answers: **is there one objective whose special cases are classical BPE and
Parity-aware BPE, and what do the tokenizers in between look like?**

## The method

Let `t_l` be the current cost of language `l`, measured as tokens per parallel sentence on
a development set. p-fair BPE minimises the power mean of the per-language costs,

    M_p(t) = ( Σ_l π_l t_l^p )^(1/p)        π_l = share of language l in the training text,

and its greedy step is a one-line change to BPE: the pair count of every language is
weighted by that language's current cost to the power `p − 1`,

    score(a, b) = Σ_l  t_l^(p−1) · c_l(a, b)         c_l = count of the pair in language l.

- `p = 1`: all weights are 1 and the rule is classical BPE.
- `p → ∞`: the weight of the most expensive language dominates and the rule becomes the
  base rule of Parity-aware BPE.
- `1 < p < ∞`: p-fair BPE. Every language keeps a positive weight, so no language is ever
  ignored, and the expensive ones count more.

Pair scores never increase under this rule, which permits **lazy evaluation**: scores are
stored in a heap and a pair is re-scored only when it reaches the top (Minoux, 1978). The
merge list is identical to full rescoring, and training is 26 to 67 times faster at
`p ∈ {1, 8}` (5 s instead of 120 s for four languages at 32k; 29 to 34 s instead of 381 to
1939 s for 14 languages at 64k).

One Rust program trains every tokenizer in the paper, because they are the same rule with
different settings:

| Setting | Tokenizer |
|---|---|
| `--p 1` | classical BPE |
| `--p 2 … 128` | p-fair BPE |
| `--p inf` | Parity-aware BPE, base rule (Foroutan et al., 2026) |
| `--p inf --hybrid 0.75` | hybrid baseline: first 75% of merges classical, rest parity |
| `--p inf --window 100 --alpha 2` | moving-window Parity-aware baseline (Foroutan et al., 2026) |
| `--p inf --worst-n 2` | Worst-N parity baseline (Kanjirangat et al., 2026) |
| `--lambda 0.5 --morph-lexicon F` | any of the above plus a morphology penalty (formulation only in the paper) |
| `--pi-uniform` | equal language weights instead of text-share weights |

## What comes out

**Raising `p` traces a monotone front** between compression (bytes per token on the training
text) and disparity (Gini coefficient of tokens per sentence across languages), in every
training sample. On Hindi, Marathi, Gujarati and English at a 32k vocabulary, `p = 8` cuts
the Gini from 0.038 to 0.023 (−40%) for a 0.6% loss in compression; full parity costs 4.0%.

![Tokens per sentence along the front](figures/per_language.png)

**Who pays.** p-fair BPE lowers the cost of the under-served languages and raises English;
it never makes a language that already costs more than the classical mean more expensive,
in any of our settings. The hybrid schedule reaches similar points on the front but does
raise some under-served languages (Malayalam, Gujarati, Kannada at 14 languages).

**14 languages in ten scripts at natural data balance** (64k vocabulary, text sizes spanning
a factor of 40) show the same front. The p-fair front and the hybrid front coincide to
within 0.003 in Gini at equal compression; p-fair is strictly fairer than every hybrid point
only at a 128k vocabulary (five of five samples).

![Compression–parity front, 14 languages](figures/front_14lang.png)

**The setting matters more than the merge rule.** The disparity, and the price of removing
it, depend on the data balance (parity costs 1.1% of compression with equal text, 4.0% with
four times as much English, 7.0% with eight times) and on the vocabulary size (intermediate
`p` loses equity as the vocabulary grows from 32k to 256k while Parity-aware BPE does not).
Measuring cost per byte instead of per parallel sentence inverts the disparity, because
Indic scripts take three UTF-8 bytes per character.

![Sensitivity to data balance and vocabulary size](figures/sensitivity.png)

**Model quality is unchanged.** A 36M-parameter GPT-2 style language model trained with each
tokenizer on identical text reaches 1.013 to 1.018 bits per byte on FLORES devtest for all
five tokenizers tested (classical, `p = 8`, `p = 32`, Hybrid-75, Parity-aware), while two
seeds of one tokenizer differ by 0.003. The cheaper tokenization of the Indian languages is
not paid for in model quality at this scale.

## Setup

1. **Rust**: install from https://rustup.rs (one command), open a new terminal. The trainer
   has no crate dependencies and builds in seconds. On macOS, if `cargo build` fails with
   "tapi error: unknown architecture", set `export SDKROOT=$(xcrun --show-sdk-path)` or point
   `SDKROOT` at an older SDK under `/Library/Developer/CommandLineTools/SDKs/`.
2. **Python 3.10+** with `pip install pyarrow huggingface_hub numpy`. Add `matplotlib` to
   regenerate the figures and `torch transformers` for the language-model experiment.
3. **Data** is downloaded on first use by `get_data.py` (called from `run_all.py`): FLORES-200
   dev and devtest from dl.fbaipublicfiles.com and a sample of the verified split of Sangraha
   from Hugging Face. Only the first part of one Sangraha file per language is streamed, about
   30 MB per language at the default size. Everything after the download runs offline.

```bash
git clone https://github.com/Siddhant-812/pfair-bpe && cd pfair-bpe
cargo build --release
python run_all.py          # downloads the data, trains 4 tokenizers, evaluates (about 5 minutes)
python check.py            # 29 correctness checks of the trainer (about 2 minutes)
```

## Replicating the numbers

Training is deterministic given the data (ties between pairs go to the smaller pair id), so
two runs on the same text give byte-identical models.

### Step 0: the regression test

`python run_all.py` trains classical, `p = 2`, `p = 4` and Parity-aware BPE on Hindi, Marathi,
Gujarati and English (20M characters each, 80M for English, 32k vocabulary) and writes
`results/`. Its `results.csv` must equal `reference_results/results.csv`:

| method | bytes per token (FLORES devtest) | Gini per sentence |
|---|---|---|
| Classical | 8.7199 | 0.0380 |
| p=2 | 8.7630 | 0.0347 |
| p=4 | 8.8129 | 0.0298 |
| Parity-aware | 8.8238 | 0.0030 |

```bash
diff results/results.csv reference_results/results.csv && echo identical
```

### The experiments of the paper

Each command writes one folder; `paper_tables.py` turns a folder into the LaTeX table and a
CSV next to it, so no number is ever retyped. Times are for one laptop core.

| Experiment | Command | Time | Output |
|---|---|---|---|
| Main front, 4 languages, 5 training samples, all baselines (Table 1, Fig. 1) | `python error_bars.py --samples 5 --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --run-args "--worst-ns 2,3 --window 100" --out results_main` | 40 min | `results_main/results_errorbars.csv`, matched-compression comparison printed |
| Tables from it | `python paper_tables.py front --dir results_main`, `… perlang`, `… regress`, `… underserved`, `… chart` | seconds | `results_main/paper_*.csv` |
| Cost unit (Table 6) | `python run_all.py --ps 1,8,inf --unit byte --out results_unit_byte` and `--unit char --out results_unit_char` | 1 min each | `results_unit_*/results_per_language.csv` |
| Data balance (Table 7) | `python run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --eng-chars-m 20 --data data_eng1 --out results_eng1` and `--eng-chars-m 160 --data data_eng8 --out results_eng8` | 5 min each | `results_eng*/results_summary.csv` |
| Vocabulary size (Table 7) | same with `--vocab-size 64000 --out results_64k`, `128000`, `256000`; 128k with error bars: `python error_bars.py --samples 5 --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --vocab-size 128000 --out results_128k_eb` | 3 to 60 min | `python paper_tables.py sens --dirs "1x=results_eng1,4x=results_main,8x=results_eng8,64k=results_64k,128k=results_128k_eb,256k=results_256k"` |
| 14 languages, natural balance (Table 3, Table 8, Fig. above) | `L=hin,mar,guj,ben,pan,ori,asm,san,tam,tel,kan,mal,urd,eng` then `python run_all.py --langs $L --natural --total-chars-m 400 --vocab-size 64000 --ps 1,2,4,8,16,32,64,128,inf --hybrids 0.5,0.75,0.9 --worst-ns 2,3,5 --window 100 --data data_l14 --out results_l14` and `python error_bars.py --samples 3 --ps 1,2,4,8,16,32,64,128,inf --hybrids 0.5,0.75,0.9 --run-args "--langs $L --natural --total-chars-m 400 --worst-ns 2,3,5 --window 100" --vocab-size 64000 --out results_l14_eb` | 15 min + 1 h | `results_l14_eb/results_errorbars.csv`; `python compare_hybrid.py results_l14` |
| Vocabulary allocation across scripts (Table 2, Tables 9–10) | `python allocate.py --clusters "Deva=hin,mar;Gujr=guj;Latn=eng" --ps 1,8 --vocab-size 32000 --out results_alloc` and `python allocate.py --data data_l14 --vocab-size 64000 --ps 1,8 --out results_alloc_l14 --clusters "Deva=hin,mar,san;Gujr=guj;Beng=ben,asm;Guru=pan;Orya=ori;Taml=tam;Telu=tel;Knda=kan;Mlym=mal;Arab=urd;Latn=eng"` | 4 min, 20 min | `results_alloc*/allocations.csv`, `results_summary.csv`; `python paper_tables.py alloc --dir results_alloc` |
| Training time, lazy vs full rescoring (Table 11) | `python speed_table.py --out results_speed` and `python speed_table.py --out results_speed --langs $L --train-dir data_l14/train --vocab-size 64000 --tag 14` | 5 min, 1 h | `results_speed/timings.csv` |
| Language-model bits per byte (GPU) | for each of `p1 p8 p32 hybrid0.75 pinf`: `python downstream/train_lm.py --tokenizer results_main/sample0/models/<tag>_v32000_line --name <Name> --langs hin,mar,guj,eng --train-dir data/train --test-dir data/flores/devtest --out results_lm/<tag> --epochs 2 --layers 6 --dim 512 --heads 8 --seq-len 512 --batch-seqs 32 --lr 6e-4 --warmup 200`; seeds 0 and 1 for `p1` and `p8` | 14 min per run on one GPU | `python paper_tables.py lm --dirs "Classical=results_lm/p1,p=8=results_lm/p8,p=32=results_lm/p32,Hybrid-75=results_lm/hybrid0.75,Parity-aware=results_lm/pinf" --seeds "Classical=results_lm/p1_seed1,p=8=results_lm/p8_seed1"` |
| Figures in this README | `python make_figures.py --main results_main --l14 results_l14_eb` | seconds | `figures/*.png` |

Expected headline values (mean over training samples, standard deviations in the CSVs):

| Setting | Tokenizer | Gini per sentence | Bytes per token (train) |
|---|---|---|---|
| 4 languages, 32k | Classical | 0.0380 | 6.910 |
| | p=8 | 0.0229 | 6.868 |
| | Parity-aware | 0.0028 | 6.634 |
| 14 languages, 64k | Classical | 0.0665 | 8.941 |
| | p=8 | 0.0370 | 8.847 |
| | Parity-aware | 0.0043 | 8.338 |

`reference_results/` holds the outputs of the interim runs (`errorbars/` is the
five-sample 4-language front without Worst-N and the window; `l14_preview/` a single
14-language run). New runs reproduce their rows exactly.

## Scripts

| Script | What it does |
|---|---|
| `src/main.rs` | the trainer and evaluator (`target/release/pfair help`): `train`, `eval`, `encode`, `encode-ids`, `merge`, `morph-eval` |
| `run_all.py` | one experiment with one command: download, train the requested tokenizers, evaluate (`python run_all.py --help` lists every setting) |
| `error_bars.py` | repeat an experiment on several non-overlapping samples of training text; paired bootstrap over test sentences; baselines against the p-fair front at matched compression |
| `compare_hybrid.py` | the matched-compression comparison for a single results folder; `is_pfair()` defines which rows form the front |
| `get_data.py` | FLORES-200 dev/devtest and a Sangraha sample for any of 20 languages (`--sizes`, `--natural`, `--part` for a different sample) |
| `allocate.py` | vocabulary allocation across scripts: cost curves from probe tokenizers, convex split by water-filling, baselines, merged tokenizers |
| `speed_table.py` | wall-clock time and heap re-checks per merge, lazy vs `--naive` |
| `paper_tables.py` | LaTeX tables and chart CSVs from the result folders (`front`, `perlang`, `regress`, `underserved`, `sens`, `alloc`, `lm`, `chart`, `morph`) |
| `make_figures.py` | the figures above |
| `check.py` | correctness checks: lazy trainer == brute force for every rule, `p = 1` == a textbook BPE written separately in Python, evaluator == trainer token counts, merged per-script models == their parts |
| `morph_prepare.py` | split gold segmentations into a training lexicon and a held-out test set (for the `--lambda` option) |
| `downstream/train_lm.py` | small GPT-2 style LM per tokenizer; bits per byte per language on FLORES devtest (needs a GPU) |
| `downstream/term_fertility.py` | tokens per technical term from term lists |

## Conventions

- Languages are three-letter names (`hin`, `mar`, `guj`, `eng`, `ben`, `tam`, …; the full list
  is in `get_data.py`). Text files are `DIR/<lang>.txt`, one sentence per line.
- `--unit line` (tokens per parallel sentence) is the paper's cost unit; `byte` and `char`
  reproduce the cost-unit ablation.
- Parity is `gini_per_line` and compression is `train_bytes_per_token` in every
  `results_summary.csv`; `bytes_per_token` is the compression on the balanced test set.
- Method names in CSVs: `Classical`, `p=8`, `Parity-aware`, `Hybrid-75`, `WorstN-2`, `Window-100`.
- A model folder holds `model.txt`, `vocab.txt`, `train_log.tsv` (one row per merge: token,
  leading language, score, every language's cost) and `train_stats.tsv`.

## License

MIT, see `LICENSE`.
