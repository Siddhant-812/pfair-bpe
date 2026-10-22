#!/usr/bin/env python3
"""Tokens per term on a list of technical terms (for example CSTT glossaries).

    python downstream/term_fertility.py --terms hin=terms/hin.txt,mar=terms/mar.txt \
        --models "Classical=results/models/p1_v32000_line,p=8=results/models/p8_v32000_line" --csv results/term_fertility.csv

Each terms file has one term per line (multi-word terms are fine). The script
reports the mean number of tokens per term and per word of the term, so that a
tokenizer that keeps technical vocabulary whole scores lower.
"""
import argparse
import csv
import os
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(HERE, "target", "release", "pfair.exe" if os.name == "nt" else "pfair")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terms", required=True, help="lang=FILE,lang=FILE")
    ap.add_argument("--models", required=True, help="name=DIR,name=DIR")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    terms = dict(x.split("=", 1) for x in a.terms.split(","))
    models = dict(x.rsplit("=", 1) for x in a.models.split(","))
    rows = []
    for name, mdir in models.items():
        for lang, path in terms.items():
            out = subprocess.run([EXE, "encode", "--model", mdir, "--input", path], check=True, capture_output=True, text=True).stdout.splitlines()
            with open(path, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            n_terms = len(lines)
            n_words = sum(len(l.split()) for l in lines)
            n_tok = sum(len(o.split()) for o in out[: n_terms])
            rows.append({"method": name, "lang": lang, "terms": n_terms, "tokens_per_term": round(n_tok / max(1, n_terms), 3),
                         "tokens_per_word": round(n_tok / max(1, n_words), 3)})
            print(f"{name:<14} {lang}: {n_terms} terms, {rows[-1]['tokens_per_term']} tokens/term, {rows[-1]['tokens_per_word']} tokens/word")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
