# Experiment plan: from the interim results to a conference paper

This file is the plan for finishing the paper *One Dial from Compression to
Parity: A Power-Mean View of Fair BPE Tokenization* (LaTeX in `paper/`). It is
written to be followed step by step by a person or an AI coding assistant.
Each experiment says what it is for, the exact command, how long it takes,
what it produces, which table or figure in the paper it fills, and what to
conclude from each possible outcome. `CLAUDE.md` has the working rules.

Keep a running log in `experiments_log.md`: one entry per run (date, command,
results folder, two lines of findings). The paper is written from that log and
from the CSV files, never from memory.

## 0. Where the project stands and what the paper claims

**What exists.** A unified formulation (power mean of per-language token costs,
exponent `p`), the greedy rule it implies (`t_l^(p-1)`-weighted pair counts),
proofs that `p=1` is classical BPE and `p=inf` is Parity-aware BPE, an exact
lazy trainer, and a five-sample study on Hindi, Marathi, Gujarati and English
(`reference_results/`). The front is monotone and repeatable; a 40% Gini
reduction costs 0.6% compression; the front coincides with the hybrid baseline.

**The honest problem.** On four languages, `p`-fair BPE matches the hybrid
schedule and does not beat it. A paper whose only empirical claim is "a
principled formulation that recovers an existing trade-off" is a workshop
paper. To be competitive at a main conference it needs at least two of:

1. a setting where the formulation makes a measurable difference (many
   languages at natural, very unequal data balance: E3),
2. a downstream check that the intrinsic gains do not cost model quality (E7),
3. the extensions that the formulation enables and the baselines do not:
   the morphology penalty inside the same rule (E5) and the convex vocabulary
   allocation (E6),
4. a clear, reproducible negative result on the alternatives (Worst-N, moving
   window) and on the pitfalls (cost unit, data balance) that practitioners
   keep getting wrong (E1, E2, E4).

**Claims the paper can make today** (already supported): monotone front with
error bars; 40% Gini for 0.6% compression at p=8; p=1 and p=inf recover the
two existing methods exactly; lazy training is exact and ~20x faster; the cost
unit inverts the parity rule across scripts; the disparity depends on data
balance.

**A claim in the current draft that must change.** The draft says (abstract,
introduction, Section 5.5, conclusion) that `p`-fair BPE "never raises an
Indian language's cost above its classical cost". That is true with four
languages, where English is the only cheap language. A single-run pilot with
14 languages at natural balance (`reference_results/l14_preview/`) shows that
every fairness rule, `p`-fair included, raises the *cheapest* languages
(Bengali and Hindi, which have the most text): Bengali goes from 30.4 to 32.7
tokens per sentence at p=8. What survives, and is the better statement, is:
`p`-fair BPE never raises a language that costs *more than the classical mean*
(the under-served ones), at any `p`, whereas the hybrid schedule raises
Gujarati and Malayalam above their classical cost at 75% and 90%. The four
places in `paper/sec/*.tex` carry a `% TODO (E3 ...)` comment; reword them
after E3 with `paper_tables.py underserved`.

**Claims that need the experiments below:** anything about more than four
languages, vocabularies above 64k, the morphology axis, allocation, Worst-N,
the moving window, and downstream quality.

**Venue and timeline.** The draft uses the ACL style. Submit to ACL Rolling
Review (ARR) and commit to the next *ACL conference (check the current ARR
deadlines; cycles are roughly every two months), with EMNLP as the fallback.
Post to arXiv at the same time as the ARR submission (ARR allows it; use the
anonymity-period rules of the cycle). A realistic schedule for one person with
a laptop and occasional GPU access:

| Week | Work |
|---|---|
| 1 | E0 setup, E1 (rerun main with all baselines), E2 (sensitivity incl. 128k/256k), E4 (unit) |
| 2 | E3 (14 languages), E8 (speed table) |
| 3 | E5 (morphology, needs the SuTRA data), E6 (allocation) |
| 4 | E7 (LM, GPU), E9 optional (term lists) |
| 5 | write results, cut to 8 pages, internal review, arXiv + ARR |

Everything except E7 runs on a laptop CPU in seconds to minutes per tokenizer.

## 1. Setup and smoke test (E0)

```bash
# Rust (https://rustup.rs), then:
pip install pyarrow huggingface_hub numpy
pip install torch transformers          # only for E7
cargo build --release
python run_all.py                       # default run; ~2 min; downloads ~250 MB the first time
python check.py                         # ~2 min; everything must print PASS
```

`run_all.py` with no arguments must reproduce `reference_results/results*.csv`
byte for byte (compare with `cmp`). If it does not, stop and find out why
before running anything else.

Runtimes below are from a 2-core cloud machine; a modern laptop is faster.
Memory: the 14-language run (E3) needs about 3 GB.

## 2. Experiments

Conventions: every experiment writes to its own `results_<name>/` folder;
`paper_tables.py` turns a folder into LaTeX; the paper section that each
experiment fills is named. "Matched compression" means the comparison that
`error_bars.py` / `compare_hybrid.py` print: the baseline's Gini minus the Gini
of the `p`-fair curve interpolated at the same training-text compression.
Always include a dense sweep of `p` (`1,2,4,8,16,32,inf`) when that comparison
matters, because the interpolation is only as good as the curve's density.

### E1. Main result with all baselines (Sections 5.2-5.5, Tables 1-3, Fig. 2)

Purpose: the front with error bars, plus Worst-N and the moving window as
baselines next to the hybrid. Re-running the main setting is necessary because
`reference_results/errorbars` has no Worst-N or window rows.

```bash
python error_bars.py --samples 5 --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 \
    --run-args "--worst-ns 2,3 --window 100" --out results_main
python paper_tables.py front   --dir results_main      # Table 2 (front) incl. baselines
python paper_tables.py perlang --dir results_main      # Table 1 (per language)
python paper_tables.py regress --dir results_main      # worst regression vs classical (the "no language worse off" claim)
```

Runtime: ~3 min per sample x 5 samples (12 tokenizers per sample), ~20 min
total; downloads ~1 GB of extra text, deleted afterwards.

Outputs: `results_main/results_errorbars.csv`, `chart_pfair.csv`,
`chart_hybrid.csv` (pgfplots data for Fig. 2), `bootstrap_test_sentences.csv`,
the printed matched-compression comparison (copy it into Table 3 and Section 6.3).

Decision: the five-sample numbers should match `reference_results/errorbars`
for the rows that exist there (same code, same data; expect identical values).
Worst-N and the window are new rows: report where they fall relative to the
curve, in the same words as the hybrid comparison. If Worst-N with N=2,3 lies
below the curve (fairer at equal compression), say so plainly and look at its
per-language table: the question is whether it buys fairness by making some
language worse off than classical BPE (the regress table answers it).

### E2. Sensitivity: data balance and vocabulary size (Section 6.2, Table 5)

Purpose: fill the 128k row and add 256k, and have folders (not just numbers in
a chat) for the 1x and 8x English rows.

```bash
python run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --eng-chars-m 20  --data data_eng1 --out results_eng1
python run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --eng-chars-m 160 --data data_eng8 --out results_eng8
python run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --vocab-size 64000  --out results_64k
python run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --vocab-size 128000 --out results_128k
python run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --vocab-size 256000 --out results_256k
python paper_tables.py sens --dirs "1x=results_eng1,4x=results_main,8x=results_eng8,64k=results_64k,128k=results_128k,256k=results_256k"
```

Runtime: 10 tokenizers per run; 32k ~15 s each, 128k ~1 min, 256k ~2-3 min.
The 8x run downloads 160M characters of English.

Decision: Lee et al. (2026) report that Parity-aware BPE loses equity as the
vocabulary grows (45k to 135k). Compare the Gini of classical, p=8 and
Parity-aware across 32k/64k/128k/256k. (a) If Parity-aware's Gini rises with
size while p=8's does not, that is a result for the paper's hypothesis in
Section 6.2 ("every language keeps a positive weight"). (b) If neither moves,
report that the effect of Lee et al. does not appear in this setting and say
why it might (four languages; English share constant). (c) If both rise, say
so. In every case, add one sentence on whether the hybrid's matched-compression
difference changes with size.

### E3. Many languages at natural balance (new Section 6.6, Table 8) -- the key experiment

Purpose: the only setting in which `p`-fair BPE can plausibly separate from
the hybrid. With 14 languages whose text sizes span a factor of ~40, the
parity rule and the hybrid help one language at a time, Worst-N pools a few,
and `p`-fair weights all of them continuously.

```bash
L=hin,mar,guj,ben,pan,ori,asm,san,tam,tel,kan,mal,urd,eng
python run_all.py --langs $L --natural --total-chars-m 400 --vocab-size 64000 \
    --ps 1,2,4,8,16,32,64,128,inf --hybrids 0.5,0.75,0.9 --worst-ns 2,3,5 --window 100 \
    --data data_l14 --out results_l14
python compare_hybrid.py results_l14
python paper_tables.py single      --dir results_l14
python paper_tables.py underserved --dir results_l14
python paper_tables.py regress     --dir results_l14
# error bars (3 samples; ~2 min per tokenizer, 16 tokenizers per sample):
python error_bars.py --samples 3 --ps 1,2,4,8,16,32,64,128,inf --hybrids 0.5,0.75,0.9 \
    --run-args "--langs $L --natural --total-chars-m 400 --worst-ns 2,3,5 --window 100" \
    --vocab-size 64000 --out results_l14_eb
```

The dense `p` sweep up to 128 matters: with 14 languages the gap between
p=16 and p=inf is large (compression 8.75 vs 8.33), and a hybrid point in that
gap is compared against a chord of the curve, which is biased against `p`-fair.

**Pilot result (single run, in `reference_results/l14_preview/`, 2026-10-22):**
at the Hybrid-50 end `p`-fair was fairer by 0.003 in Gini at equal
compression, at 75% a tie, at 90% the hybrid was fairer by 0.002; Worst-N and
the window sat on the curve at the parity end. `p`-fair never raised an
under-served language; the hybrid raised Gujarati and Malayalam. Compression
on the balanced test set peaked at p=16. The error-bar run decides whether the
0.003 at the fair end is real (sample-to-sample spread was 0.0002-0.0005 at
four languages; expect somewhat more here).

`--natural` gives each language text in proportion to the size of its
Sangraha split (Hindi 75M characters, English 60M, Bengali 58M, Tamil 40M, ...,
Assamese 2M; see `data_l14/train/SIZES.txt`). The FLORES codes and Sangraha
folders of all 20 supported languages are in `get_data.py`.

Runtime: ~2 min per 64k tokenizer (960 MB of text, 6M distinct words), ~30 min
for the single run, ~1.5 h for the error bars; ~3 GB of memory.

What to look at, in order:

1. The matched-compression comparison (`compare_hybrid.py`). If `p`-fair is
   fairer than Hybrid-75/90 and Worst-N at equal compression by more than the
   sample-to-sample spread, this is the paper's headline result: write it into
   the abstract and introduction ("at natural balance with 14 languages,
   `p`-fair BPE dominates the hybrid schedule by ...").
2. The under-served table (`paper_tables.py underserved`): for the languages
   that cost more than the classical mean, does any rule make one of them more
   expensive? In the pilot, `p`-fair never did and the hybrid did. If the error
   bars confirm it, this replaces the four-language "no language worse off"
   claim everywhere in the paper (see Section 0). The `regress` table (all
   non-English languages) is the four-language version of the same question
   and will show every rule raising Bengali and Hindi; report it in the
   appendix so that the under-served statement is not read as "nobody pays".
3. Max/min ratio and the per-language table: which languages pay under each
   rule. Assamese (2M characters) and Sanskrit are the ones to watch.
4. Where the hybrid and `p`-fair differ most: at the 90% end (mild fairness)
   or the 50% end (strong fairness)?

If `p`-fair and the hybrid still coincide at 14 languages, the paper's
framing stays as it is in the current draft (a principled formulation that
recovers the trade-off and exposes its pitfalls), and Section 6.6 reports the
coincidence as a finding: the front is determined by the data, not by the rule.
That is still a result reviewers will not have seen.

### E4. The cost unit (Section 6.1, Table 4)

Purpose: have a results folder behind Table 4 (currently a single run from
the interim work).

```bash
python run_all.py --ps 1,8,inf --unit byte --out results_unit_byte
python run_all.py --ps 1,8,inf --unit char --out results_unit_char
```

Runtime: 1 min each. Report tokens per sentence (`results_per_language.csv`)
for the byte-unit Parity-aware tokenizer next to the sentence-unit one. The
char-unit run answers an obvious reviewer question ("is per-character enough?");
expect it to be close to per-sentence for these scripts but not identical,
because sentence length in characters differs by language.

### E5. Morphology penalty (Section 6.4, Table 6) -- needs the SuTRA data

Purpose: the third axis. The rule gains a term `-lambda * beta(a,b)` that
discounts pair occurrences whose junction is a morpheme boundary;
`0 <= lambda <= 1` keeps the lazy trainer exact (lambda = 1 ignores crossing
occurrences entirely).

Data: the SuTRA gold segmentations for Hindi, Marathi and Gujarati (you are an
author of SuTRA; the files are not in this repository). Any format with one
word per line and segments separated by `+`, `|`, `-` or whitespace works, or
`word<TAB>segments`. Lines whose segments do not spell the word (sandhi) are
skipped and listed in `morph/<lang>.skipped.txt`.

```bash
python morph_prepare.py --gold hin=/path/hindi.txt,mar=/path/marathi.txt,guj=/path/gujarati.txt --out morph
#   -> morph/lexicon_all.tsv (training penalty) and morph/<lang>.test.tsv (held-out evaluation)
python run_all.py --ps 1,8,inf --lambdas 0.25,0.5,0.75,1 \
    --morph-lexicon morph/lexicon_all.tsv \
    --morph-gold hin=morph/hin.test.tsv,mar=morph/mar.test.tsv,guj=morph/guj.test.tsv \
    --out results_morph
python paper_tables.py morph --dir results_morph
```

Runtime: 15 tokenizers x 15 s.

Important: `morph_prepare.py` splits the gold words 80/20 by word with a fixed
seed, and boundary F1 is measured on the 20% that the trainer never saw.
Never evaluate on words that were in the lexicon.

The trainer prints how much of the training text the lexicon covers ("they
cover X% of training word occurrences"). If X is small (a few percent), the
penalty can only act on those words and the F1 gain on held-out words will be
modest; say so in the paper and, if time allows, extend the lexicon with a
simple suffix-stripping rule derived from the gold data (not implemented;
describe it as future work otherwise).

Decision: report boundary F1 per language against lambda at p = 1, 8, inf,
with Gini and compression in the same rows. The expected shape is F1 up,
compression slightly down, Gini roughly unchanged (the penalty is applied
inside the same weighted score, so it should not disturb parity much). If F1
does not move, the lexicon coverage is the first suspect; the second is that
BPE boundaries on these scripts already align with morphemes about as well as
the gold data allows (SuTRA reports BPE boundary F1 of 0.47 on Marathi; check
that the lambda = 0 row reproduces a similar number).

Also run `morph-eval` on the hybrid and Parity-aware tokenizers from E1 to
show that fairness and morphology are orthogonal axes (Kanjirangat et al.
report the same for Worst-N):

```bash
for m in results_main/sample0/models/*; do target/release/pfair morph-eval --model $m --gold morph/hin.test.tsv --lang hin --name $(basename $m) --csv results_morph/morph_of_main.csv; done
```

### E6. Vocabulary allocation across scripts (Section 6.5, Table 7, Appendix B)

Purpose: the second level of the formulation. `allocate.py` fits a cost curve
per script from a few probe tokenizers, solves the convex budget split by
water-filling, builds a tokenizer per script at its budget, merges them, and
evaluates against uniform, data-proportional and union-proportion splits and
against a joint tokenizer without allocation.

```bash
# four languages, three scripts:
python allocate.py --clusters "Deva=hin,mar;Gujr=guj;Latn=eng" --ps 1,8 --vocab-size 32000 --out results_alloc
python paper_tables.py alloc --dir results_alloc
# fourteen languages, ten scripts (after E3 has downloaded data_l14):
python allocate.py --data data_l14 --vocab-size 64000 --ps 1,8 --out results_alloc_l14 \
   --clusters "Deva=hin,mar,san;Gujr=guj;Beng=ben,asm;Guru=pan;Orya=ori;Taml=tam;Telu=tel;Knda=kan;Mlym=mal;Arab=urd;Latn=eng"
```

Runtime: 4 languages ~3 min; 14 languages ~20 min (probe tokenizers at
2k-32k per script, then one tokenizer per script per allocation).

Outputs: `curves.csv` (measured points and fitted a, b, d with the fit error),
`allocations.csv` (budgets per method, predicted cost), `results_summary.csv`.

Decision and a warning from the pilot run: on a small sample the joint
tokenizer (no allocation) beat every allocation, because splitting by script
removes the sharing of digits, punctuation and Latin letters across scripts.
Expect the same at full scale. The honest framing is then: (i) among the ways
of splitting, the convex split at p=1 has the best compression and at p=8 the
best Gini, matching what the objective asks for; (ii) splitting by script is
itself costly, so allocation is useful only when per-script tokenizers are
wanted for other reasons (cluster-based vocabularies as in Brahma et al.,
XLM-V). Report the fit quality (RMSE in `curves.csv`) in the appendix; if the
curve does not fit (RMSE above ~0.3 tokens per sentence), add probe sizes.

### E7. Downstream: language-model bits per byte (Section 6.7, Table 9) -- needs a GPU

Purpose: show that the fairer tokenizers do not cost model quality in the
languages they help, with a tokenizer-independent measure.

```bash
for cfg in "Classical=p1" "p=8=p8" "p=32=p32" "Hybrid-75=hybrid0.75" "Parity-aware=pinf"; do
  name=${cfg%=*}; tag=${cfg##*=}
  python downstream/train_lm.py --tokenizer results_main/sample0/models/${tag}_v32000_line --name "$name" \
      --langs hin,mar,guj,eng --train-dir data/train --test-dir data/flores/devtest \
      --out results_lm/$tag --epochs 2 --layers 6 --dim 512 --heads 8 --seq-len 512 --batch-seqs 32
done
python paper_tables.py lm --dirs "Classical=results_lm/p1,p=8=results_lm/p8,p=32=results_lm/p32,Hybrid-75=results_lm/hybrid0.75,Parity-aware=results_lm/pinf"
```

Runtime: ~35M tokens per epoch at 32k; a 6-layer 512-dim model (~30M
parameters) runs at roughly 50-100k tokens/s on one modern GPU, so 2 epochs
take 10-25 minutes per tokenizer. On CPU it is hours; use `--max-lines` and
smaller `--layers/--dim` only for smoke tests, never for the paper.

Every model sees exactly the same text for the same number of epochs; only the
tokenizer differs. Report per-language bits per byte, their mean, and the
max/min ratio across languages. Run two seeds (`--seed 0`, `--seed 1`) for at
least Classical and p=8 to have an idea of the noise; report the spread.

Decision: (a) mean bits per byte within noise across tokenizers and a smaller
spread across languages at p=8: the intrinsic gains come for free; say so.
(b) p=8 better on Indic languages and worse on English by a similar amount:
the model's quality follows the token budget; also fine, report it. (c) mean
bits per byte clearly worse for the fair tokenizers: a real cost; report it
and move the "practical takeaways" paragraph accordingly. Do not drop the
experiment if it comes out as (c).

Optional (if time): Hindi-Marathi translation with a small transformer and
chrF++, as in SuTRA. It is a week of work and not required for the claim; the
LM experiment is the standard check in the tokenizer-fairness literature.

### E8. Training-time table (Appendix C, Table 10)

```bash
for p in 1 8 inf; do
  /usr/bin/time -f "%e s" target/release/pfair train --langs hin,mar,guj,eng --train-dir data/train --dev-dir data/flores/dev --unit line --p $p --vocab-size 32000 --out /tmp/speed_$p
  /usr/bin/time -f "%e s" target/release/pfair train --langs hin,mar,guj,eng --train-dir data/train --dev-dir data/flores/dev --unit line --p $p --vocab-size 32000 --out /tmp/speedn_$p --naive
done
```

Repeat with `--langs` set to the 14 languages and `--train-dir data_l14/train`
for the 14-language rows (the naive runs take tens of minutes there; one each is
enough). The trainer prints "heap re-checks per merge" at the end of every run.
On macOS `/usr/bin/time -f` is not available; use `time` or `gtime`.

### E9. Optional: tokens per technical term

If CSTT term lists are available (one term per line, per language):

```bash
python downstream/term_fertility.py --terms hin=terms/hin.txt,mar=terms/mar.txt,guj=terms/guj.txt \
    --models "Classical=results_main/sample0/models/p1_v32000_line,p=8=results_main/sample0/models/p8_v32000_line,Parity-aware=results_main/sample0/models/pinf_v32000_line" \
    --csv results_terms.csv
```

### E10. Optional: uniform language weights

`--pi-uniform` weights every language equally in the objective instead of by
its share of training text (the derivation in Section 4.1 gives text-share
weights). One run at 14 languages is enough for a sentence in the paper:

```bash
python run_all.py --langs $L --data data_l14 --skip-data --vocab-size 64000 --ps 1,4,8,16,inf --pi-uniform --out results_l14_piuniform
```

## 3. Writing plan

The draft is 13 pages compiled (8 of content allowed for review, 9 after
acceptance; unlimited references and appendix). Decide what moves to the
appendix once the new sections have numbers: candidates are the per-language
table (keep a 4-row version in the main text), the cost-unit table, and the
speed table.

| Paper section | Source of numbers | Script |
|---|---|---|
| 5.2-5.5, Tables 1-3, Fig. 2 | `results_main` (E1) | `paper_tables.py front/perlang/regress`, printed matched-compression comparison |
| 6.1 Table 4 | `results_unit_byte` (E4) | `results_per_language.csv` |
| 6.2 Table 5 | `results_eng1/eng8/64k/128k/256k` (E2) | `paper_tables.py sens` |
| 6.3 Worst-N / window | `results_main` (E1) | printed comparison |
| 6.4 Table 6 | `results_morph` (E5) | `paper_tables.py morph` |
| 6.5 Table 7 | `results_alloc` (E6) | `paper_tables.py alloc` |
| 6.6 Table 8 | `results_l14` (E3) | `paper_tables.py single/regress`, `compare_hybrid.py` |
| 6.7 Table 9 | `results_lm` (E7) | `paper_tables.py lm` |
| App. C Table 10 | E8 | by hand from the timings |

Order of writing: E3 and E7 decide the abstract and the introduction, so write
them last. Every "(planned)" marker and every "--" in `paper/sec/*.tex` must be
gone before submission; `grep -n "planned\|--" paper/sec/*.tex` finds them
(ignore the `--` inside `\cmidrule` and page ranges).

Figures: Fig. 2 (front) reads its coordinates from `chart_pfair.csv` and
`chart_hybrid.csv`; replace the inline coordinates in `sec/5_results.tex` with
the new values, or switch the plot to `table[col sep=comma]{...csv}` and upload
the CSVs with the source. Add a second front figure for the 14-language setting
if E3 separates the methods.

Reference check: every entry in `paper/main.bib` was verified against the
venue page in October 2026 except the page numbers of Kanjirangat et al.
(221-228, MeLLM 2026) and the SuTRA Interspeech pages, which should be checked
against the final proceedings. Do not cite the interim report.

## 4. Submission checklist

- `python check.py` passes on the final code; the default run reproduces `reference_results/results*.csv`.
- Every number in the paper traces to a CSV in a results folder named in `experiments_log.md`.
- Error bars: Tables 1-3 from five samples; Table 8 from three; LM from two seeds.
- Limitations section updated: re-implemented baselines (character-level), greedy first-order steps, data from one source (Sangraha), FLORES as the only parallel set, LM size.
- Anonymity: the ACL review version must not name the authors or say "our SuTRA"; cite Rathore et al. (2026) in the third person. `main.tex` uses `\usepackage[review]{acl}`; keep it for submission, switch to `[final]` for the camera-ready.
- Code release: this repository with `reference_results/` and a `RESULTS.md` listing the folders behind each table; licence file (MIT); no data redistribution (Sangraha and FLORES are downloaded by `get_data.py`).
- arXiv: the same PDF compiled with `\usepackage[final]{acl}` (author names), plus the appendix; add the arXiv id to the ARR form.
- Responsible-NLP checklist (ARR form): data licences (Sangraha CC-BY-4.0, FLORES-200 CC-BY-SA-4.0), compute used (CPU for everything but E7; one GPU-day for E7), no human subjects.

## 5. Pitfalls seen so far

- **Per-byte cost** sends the parity rule the wrong way across scripts. Never
  switch `--unit` away from `line` except in E4.
- **Matched-compression comparisons** need a dense `p` sweep; with three `p`
  values the interpolation is coarse and can flip the sign of a 0.001 difference.
- **Exact ties.** Worst-1 and Parity-aware differ only when two languages have
  exactly the same cost (it happens: at merge 216 in the small check). This is
  not a bug.
- **Lambda above 1** breaks the lazy bound; the trainer refuses it without
  `--naive`. Stay in [0, 1].
- **Allocation:** merged per-script vocabularies are smaller than the sum of
  budgets because alphabets overlap; `allocate.py` tops the budgets up so that
  every tokenizer has K entries. Compare allocations with each other and with
  the joint tokenizer at the same K.
- **Sangraha languages with one or two parquet files** (asm has 3, san 11, ori 11;
  mai/kas/mni/sat have 1) give overlapping samples for `error_bars.py`. Prefer
  three samples at 14 languages and say in the paper that the smallest
  languages' samples overlap.
- **macOS linker error** ("tapi error: unknown architecture") when building:
  `export SDKROOT=$(xcrun --show-sdk-path)` or point `SDKROOT` at an older SDK.
- **Disk:** every results folder keeps its models (a 64k model is ~2 MB; the
  `test_tokens_per_line.tsv` and the LM id files are larger). The 14-language
  data is ~1 GB per sample.
