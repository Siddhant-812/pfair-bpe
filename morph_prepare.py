#!/usr/bin/env python3
"""Prepare gold morphological segmentations for the morphology experiment.

    python morph_prepare.py --gold hin=path/to/hindi.txt,mar=...,guj=... [--out morph] [--test-share 0.2]

Input: one word per line, segments separated by '+', '|', '-' or whitespace, or
'word<TAB>segments'. Lines whose segments do not concatenate to the word
(sandhi, allomorphy) are kept in a separate file and not used, because a
boundary position cannot be placed in the surface form.

Output, per language, in --out:
    <lang>.lexicon.tsv   segmentations used during training (the beta penalty)
    <lang>.test.tsv      held-out words for boundary F1 / MorphScore  (never seen by the trainer)
    <lang>.skipped.txt   lines that could not be used
    lexicon_all.tsv      all languages' lexicon files concatenated (pass this to --morph-lexicon)

The split is by word, with a fixed seed, so the test words are never in the
lexicon that the trainer sees. Keep it that way: scoring on words that were
in the lexicon would measure memorisation, not alignment.
"""
import argparse
import os
import random
import re
import unicodedata


def parse_line(line):
    line = unicodedata.normalize("NFC", line.strip())
    if not line or line.startswith("#"):
        return None
    if "\t" in line:
        word, rest = line.split("\t", 1)
        segs = [s for s in re.split(r"[+|\-\s]+", rest.strip()) if s]
        return word.strip(), segs
    segs = [s for s in re.split(r"[+|\-]+", line) if s]
    if len(segs) < 2 and " " in line:  # "seg seg" without a tab
        segs = line.split()
    return "".join(segs), segs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="lang=FILE,lang=FILE,...")
    ap.add_argument("--out", default="morph")
    ap.add_argument("--test-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    all_lex = []
    for item in a.gold.split(","):
        lang, path = item.split("=")
        good, skipped, seen = [], [], set()
        with open(path, encoding="utf-8") as f:
            for raw in f:
                parsed = parse_line(raw)
                if parsed is None:
                    continue
                word, segs = parsed
                if len(segs) < 2 or "".join(segs) != word or any(c.isspace() for c in word):
                    skipped.append(raw.rstrip("\n"))
                    continue
                if word in seen:
                    continue
                seen.add(word)
                good.append((word, segs))
        rng.shuffle(good)
        n_test = int(round(len(good) * a.test_share))
        test, lex = good[:n_test], good[n_test:]
        for name, rows in [("lexicon", lex), ("test", test)]:
            with open(os.path.join(a.out, f"{lang}.{name}.tsv"), "w", encoding="utf-8") as f:
                for word, segs in rows:
                    f.write(f"{word}\t{' '.join(segs)}\n")
        with open(os.path.join(a.out, f"{lang}.skipped.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(skipped) + ("\n" if skipped else ""))
        all_lex += [f"{w}\t{' '.join(s)}" for w, s in lex]
        print(f"{lang}: {len(good)} usable words -> {len(lex)} lexicon, {len(test)} test; {len(skipped)} lines skipped (segments do not spell the word)")
    with open(os.path.join(a.out, "lexicon_all.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_lex) + "\n")
    print(f"\nTraining penalty: --morph-lexicon {a.out}/lexicon_all.tsv")
    print("Evaluation:       --morph-gold " + ",".join(f"{item.split('=')[0]}={a.out}/{item.split('=')[0]}.test.tsv" for item in a.gold.split(",")))


if __name__ == "__main__":
    main()
