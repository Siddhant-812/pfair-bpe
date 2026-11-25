<h1 align="center">p-fair BPE</h1>

<p align="center">
  <b>One dial from compression to parity in multilingual tokenization.</b><br>
  <i>Rust trainer with exact lazy evaluation · evaluation and experiment scripts</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/trainer-Rust%2C%20no%20dependencies-orange" alt="Rust">
  <img src="https://img.shields.io/badge/scripts-Python%203.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/data-Sangraha%20%C2%B7%20FLORES--200-green" alt="Data">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT">
</p>

Classical BPE gives the best total compression but makes languages with less training text pay more tokens for the same content. Parity-aware BPE equalises the cost at a loss in compression. **p-fair BPE** connects the two with a single parameter $p$: $p = 1$ is classical BPE, $p \to \infty$ is Parity-aware BPE, and every value in between is a point on the compression–parity front, trained exactly and an order of magnitude faster than naive rescoring.

<p align="center">
  <img src="figures/front_4lang.png" width="720" alt="Compression–parity front on Hindi, Marathi, Gujarati and English">
</p>

<table align="center">
  <tr>
    <td align="center" width="50%"><img src="figures/per_language.png" alt="Tokens per sentence per language along the front"><br><sub>Raising <i>p</i> lowers the cost of the under-served languages and raises English.</sub></td>
    <td align="center" width="50%"><img src="figures/front_14lang.png" alt="Front with 14 languages in ten scripts"><br><sub>The same front with 14 languages in ten scripts at natural data balance.</sub></td>
  </tr>
</table>

<details>
<summary><b>How the merge rule works</b></summary>
<br>

With $t_\ell$ the current cost of language $\ell$ (tokens per parallel sentence on a development set) and $\pi_\ell$ its share of the training text, p-fair BPE minimises the power mean of the per-language costs,

$$M_p(t) = \Big(\sum_\ell \pi_\ell\, t_\ell^{\,p}\Big)^{1/p},$$

and the greedy step weights each language's pair counts by its current cost to the power $p-1$:

$$\mathrm{score}(a,b) = \sum_\ell t_\ell^{\,p-1}\, c_\ell(a,b).$$

At $p=1$ all weights are one (classical BPE); as $p\to\infty$ the most expensive language dominates (Parity-aware BPE). Scores never increase, so a lazy heap re-scores a pair only when it reaches the top and still produces the same merges as full rescoring.
</details>

## Installation

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh    # Rust, if missing
pip install pyarrow huggingface_hub numpy                             # matplotlib for figures, torch transformers for the LM
git clone https://github.com/Siddhant-812/pfair-bpe && cd pfair-bpe
cargo build --release
```

## Training

```bash
python get_data.py --langs hin,mar,guj,eng --eng-chars-m 80 --chars-m 20    # once: data/train, data/flores

target/release/pfair train --langs hin,mar,guj,eng \
    --train-dir data/train --dev-dir data/flores/dev \
    --unit line --p 8 --vocab-size 32000 --out models/p8
```

| Flag | Tokenizer |
|---|---|
| `--p 1` | classical BPE |
| `--p 2` … `--p 128` | p-fair BPE |
| `--p inf` | Parity-aware BPE |
| `--p inf --hybrid 0.75` | hybrid schedule: 75% classical merges, then parity |
| `--p inf --window 100 --alpha 2` | moving-window Parity-aware BPE |
| `--p inf --worst-n 2` | Worst-N parity |
| `--lambda 0.5 --morph-lexicon FILE` | morphology penalty on top of any rule |

`--unit line` measures cost in tokens per parallel sentence; `byte` and `char` are the alternatives. Text files are `DIR/<lang>.txt`, one sentence per line; the language codes are listed in `get_data.py`.

## Evaluation

```bash
target/release/pfair eval --langs hin,mar,guj,eng --model models/p8 \
    --test-dir data/flores/devtest --train-dir data/train --unit line \
    --summary summary.csv --detail per_language.csv

target/release/pfair encode --model models/p8 --input text.txt
```

`summary.csv` holds one row per tokenizer (Gini of per-language cost, bytes per token on the test set and on the training text); `per_language.csv` the tokens per sentence of each language.

## Experiments

<details open>
<summary><b>Sweep <i>p</i> and the baselines</b></summary>

```bash
python run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --worst-ns 2,3 --window 100 --out results_main
python run_all.py --vocab-size 128000 --out results_128k            # vocabulary size
python run_all.py --eng-chars-m 20 --data data_eng1 --out results_eng1   # data balance
python run_all.py --unit byte --out results_byte                     # cost unit
```
`run_all.py` downloads the data on first use, trains the requested tokenizers and evaluates them. `python run_all.py --help` lists every option.
</details>

<details>
<summary><b>Error bars</b></summary>

```bash
python error_bars.py --samples 5 --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 \
    --run-args "--worst-ns 2,3 --window 100" --out results_main
```
Repeats the run on five non-overlapping samples of training text, reports mean and spread, and compares every baseline with the p-fair front at matched compression.
</details>

<details>
<summary><b>14 languages at natural data balance</b></summary>

```bash
L=hin,mar,guj,ben,pan,ori,asm,san,tam,tel,kan,mal,urd,eng
python run_all.py --langs $L --natural --total-chars-m 400 --vocab-size 64000 \
    --ps 1,2,4,8,16,32,64,128,inf --hybrids 0.5,0.75,0.9 --worst-ns 2,3,5 --window 100 \
    --data data_l14 --out results_l14
python compare_hybrid.py results_l14
```
</details>

<details>
<summary><b>Vocabulary allocation across scripts</b></summary>

```bash
python allocate.py --clusters "Deva=hin,mar;Gujr=guj;Latn=eng" --ps 1,8 --vocab-size 32000 --out results_alloc
```
Fits a cost curve per script, splits the vocabulary budget by water-filling, trains one tokenizer per script and merges them.
</details>

<details>
<summary><b>Language-model check (GPU)</b></summary>

```bash
python downstream/train_lm.py --tokenizer models/p8 --name "p=8" --langs hin,mar,guj,eng \
    --train-dir data/train --test-dir data/flores/devtest --out results_lm/p8 \
    --epochs 2 --layers 6 --dim 512 --heads 8 --seq-len 512 --batch-seqs 32
```
Trains one small language model per tokenizer on identical text and reports bits per byte per language, which is comparable across tokenizers.
</details>

## Tables and figures

```bash
python paper_tables.py front --dir results_main        # also: perlang, regress, underserved, sens, alloc, lm
python make_figures.py --main results_main --l14 results_l14_eb
```
