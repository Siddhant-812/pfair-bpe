#!/usr/bin/env python3
"""Download the data for the tokenizer experiments.

  data/train/<lang>.txt           training text, one line per paragraph   (Sangraha, verified split)
  data/flores/dev/<lang>.txt      997 parallel sentences                  (FLORES-200 dev)
  data/flores/devtest/<lang>.txt  1012 parallel sentences                 (FLORES-200 devtest)

Only the first part of one Sangraha file per language is streamed (about 30 MB
each for the default size), not the whole 400-600 MB file.

Usage:
  pip install pyarrow huggingface_hub
  python get_data.py                   # 20 million characters per language
  python get_data.py --chars-m 5       # smaller and quicker
  python get_data.py --eng-chars-m 80  # give English 4x more text than the others
  python get_data.py --part 1 --out data_sample1   # a different sample of the training text
  python get_data.py --langs hin,mar,guj,ben,tam,tel,kan,mal,pan,ori,asm,san,urd,eng --sizes eng=80,hin=40 --chars-m 20
                                       # 14 languages; per-language sizes override --chars-m
  python get_data.py --langs ... --natural --total-chars-m 400
                                       # sizes in proportion to the size of each language's Sangraha split
"""
import argparse
import io
import os
import sys
import tarfile
import unicodedata
import urllib.request

# our short name -> (FLORES-200 code, Sangraha folder, relative size of the verified Sangraha
# split in parquet files of ~375 MB; English files are ~600 MB, counted as 1.6 each).
# The relative sizes are only used by --natural.
LANGS = {
    "hin": ("hin_Deva", "hin", 100),
    "mar": ("mar_Deva", "mar", 34),
    "guj": ("guj_Gujr", "guj", 27),
    "eng": ("eng_Latn", "eng", 80),
    "ben": ("ben_Beng", "ben", 77),
    "tam": ("tam_Taml", "tam", 53),
    "tel": ("tel_Telu", "tel", 41),
    "mal": ("mal_Mlym", "mal", 37),
    "kan": ("kan_Knda", "kan", 21),
    "urd": ("urd_Arab", "urd", 27),
    "nep": ("npi_Deva", "nep", 25),
    "pan": ("pan_Guru", "pan", 12),
    "ori": ("ory_Orya", "ori", 11),
    "san": ("san_Deva", "san", 11),
    "asm": ("asm_Beng", "asm", 3),
    "mai": ("mai_Deva", "mai", 1),
    "snd": ("snd_Arab", "snd", 2),
    "kas": ("kas_Arab", "kas", 1),
    "mni": ("mni_Beng", "mni", 1),
    "sat": ("sat_Olck", "sat", 1),
}
# script of each language (for the allocation experiment)
SCRIPTS = {
    "hin": "Deva", "mar": "Deva", "nep": "Deva", "san": "Deva", "mai": "Deva",
    "guj": "Gujr", "ben": "Beng", "asm": "Beng", "mni": "Beng", "tam": "Taml", "tel": "Telu",
    "mal": "Mlym", "kan": "Knda", "urd": "Arab", "snd": "Arab", "kas": "Arab", "pan": "Guru",
    "ori": "Orya", "sat": "Olck", "eng": "Latn",
}
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
MIN_LINE_CHARS = 20  # drop menu items, bylines and other fragments


def clean(line):
    # NFC so that the same visible text is always the same code points
    return unicodedata.normalize("NFC", " ".join(line.split()))


def unquote(line):
    # A few FLORES-200 files (mar, guj) have some lines left in CSV quoting:
    #   "He said, ""hello"""   ->   He said, "hello"
    # Undo it, otherwise those languages are charged for quote marks nobody wrote.
    if len(line) > 2 and line[0] == '"' and line[-1] == '"' and '""' in line:
        return line[1:-1].replace('""', '"')
    return line


def get_flores(out_dir, langs):
    done = all(
        os.path.exists(f"{out_dir}/flores/{split}/{l}.txt") for split in ("dev", "devtest") for l in langs
    )
    if done:
        print("FLORES-200: already there")
        return
    print("FLORES-200: downloading (25 MB) ...")
    with urllib.request.urlopen(FLORES_URL) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for split in ("dev", "devtest"):
            os.makedirs(f"{out_dir}/flores/{split}", exist_ok=True)
            for l in langs:
                member = f"./flores200_dataset/{split}/{LANGS[l][0]}.{split}"
                text = tar.extractfile(member).read().decode("utf-8")
                lines = [clean(unquote(x)) for x in text.splitlines()]
                with open(f"{out_dir}/flores/{split}/{l}.txt", "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                print(f"  {split}/{l}.txt  {len(lines)} sentences")


def get_sangraha(out_dir, lang, target_chars, part=0):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    path = f"{out_dir}/train/{lang}.txt"
    size_note = f"{out_dir}/train/{lang}.size"  # remembers how much text was asked for
    want = str(int(target_chars)) + (f" part{part}" if part else "")
    if os.path.exists(path) and os.path.exists(size_note) and open(size_note).read().strip() == want:
        print(f"Sangraha {lang}: already there ({os.path.getsize(path) / 1e6:.0f} MB)")
        return
    os.makedirs(f"{out_dir}/train", exist_ok=True)
    fs = HfFileSystem()
    folder = f"datasets/ai4bharat/sangraha/verified/{LANGS[lang][1]}"
    n_files = len([p for p in fs.ls(folder, detail=False) if p.endswith(".parquet")])
    # Sangraha is split into many files per language. `part` picks which file to
    # start from, so different parts are different, non-overlapping samples.
    if part >= n_files:
        print(f"Sangraha {lang}: only {n_files} file(s), so sample {part} reuses file {part % n_files} (samples overlap for this language)")
    seen, chars, lines, file_no = set(), 0, 0, 0
    with open(path + ".part", "w", encoding="utf-8") as out:
        while chars < target_chars and file_no < n_files:
            with fs.open(f"{folder}/data-{(part + file_no) % n_files}.parquet", "rb") as f:
                # buffer_size + pre_buffer=False: read the file piece by piece and stop early
                pf = pq.ParquetFile(f, buffer_size=4 << 20, pre_buffer=False)
                for batch in pf.iter_batches(batch_size=500, columns=["text"]):
                    for doc in batch.column(0).to_pylist():
                        for raw in (doc or "").split("\n"):
                            line = clean(raw)
                            if len(line) < MIN_LINE_CHARS:
                                continue
                            h = hash(line)
                            if h in seen:  # repeated boilerplate
                                continue
                            seen.add(h)
                            out.write(line + "\n")
                            chars += len(line)
                            lines += 1
                    print(f"\rSangraha {lang}: {chars / 1e6:6.1f} M characters", end="", flush=True)
                    if chars >= target_chars:
                        break
            file_no += 1
    os.replace(path + ".part", path)
    with open(size_note, "w") as f:
        f.write(want)
    print(f"\rSangraha {lang}: {chars / 1e6:.1f} M characters, {lines} lines, {os.path.getsize(path) / 1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--langs", default="hin,mar,guj,eng")
    ap.add_argument("--chars-m", type=float, default=20, help="million characters of training text per language")
    ap.add_argument("--eng-chars-m", type=float, default=None, help="override for English (default: same as the others)")
    ap.add_argument("--part", type=int, default=0, help="which sample of the training text to take (0, 1, 2, ...)")
    ap.add_argument("--sizes", default="", help="per-language sizes in million characters, e.g. eng=80,hin=40 (overrides --chars-m)")
    ap.add_argument("--natural", action="store_true", help="split --total-chars-m across the languages in proportion to their Sangraha size")
    ap.add_argument("--total-chars-m", type=float, default=400, help="total million characters for --natural")
    a = ap.parse_args()
    langs = a.langs.split(",")
    for l in langs:
        if l not in LANGS:
            sys.exit(f"unknown language {l}; add it to LANGS at the top of get_data.py")
    sizes = {l: a.chars_m for l in langs}
    if a.eng_chars_m is not None and "eng" in sizes:
        sizes["eng"] = a.eng_chars_m
    if a.natural:
        tot = sum(LANGS[l][2] for l in langs)
        sizes = {l: a.total_chars_m * LANGS[l][2] / tot for l in langs}
    for item in a.sizes.split(","):
        if item.strip():
            l, m = item.split("=")
            sizes[l.strip()] = float(m)
    get_flores(a.out, langs)
    for l in langs:
        get_sangraha(a.out, l, sizes[l] * 1e6, a.part)
    with open(f"{a.out}/train/SIZES.txt", "w") as f:
        for l in langs:
            f.write(f"{l}\t{sizes[l]:.2f}M chars requested\n")


if __name__ == "__main__":
    main()
