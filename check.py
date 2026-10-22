#!/usr/bin/env python3
"""Correctness checks. Run after the data is downloaded:   python check.py

1. The fast trainer and the brute-force trainer (--naive) must learn exactly
   the same merges, for every p.
2. p = 1 must equal classical BPE: compared against a textbook BPE written
   here in plain Python, which shares no code with the Rust trainer.
3. The evaluator must count the same number of tokens on the training text
   as the trainer itself ended with.
4. The hybrid baseline: fast == brute force, and its two extremes must equal
   Parity-aware BPE (0% classical) and classical BPE (100% classical).
5. The morphology penalty (lambda in [0,1]), Worst-N and the moving window:
   fast == brute force; lambda = 0 changes nothing; Worst-1 equals Parity-aware
   up to the first exact tie in costs; a model split at every character scores
   boundary recall 1 in morph-eval.
6. `merge` of per-script models evaluates each language like its own model.
"""
import collections
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["hin", "mar", "guj", "eng"]
EXE = os.path.join(HERE, "target", "release", "pfair.exe" if os.name == "nt" else "pfair")
WORK = os.path.join(HERE, "checks")
ok = True


def run(*args):
    subprocess.run([EXE, *args], check=True, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def report(name, passed):
    global ok
    ok = ok and passed
    print(("PASS  " if passed else "FAIL  ") + name, flush=True)


def sample(name, n_lines):
    d = os.path.join(WORK, name)
    os.makedirs(d, exist_ok=True)
    for l in LANGS:
        with open(os.path.join(HERE, "data", "train", f"{l}.txt"), encoding="utf-8") as f:
            lines = [next(f) for _ in range(n_lines)]
        with open(os.path.join(d, f"{l}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)
    return d


def read_model(d):
    L = open(os.path.join(d, "model.txt"), encoding="utf-8").read().split("\n")
    n_alpha = int(L[2].split()[1])
    alphabet = [chr(int(x)) for x in L[3:3 + n_alpha]]
    n_merges = int(L[3 + n_alpha].split()[1])
    merges = [tuple(map(int, x.split())) for x in L[4 + n_alpha:4 + n_alpha + n_merges]]
    return alphabet, merges


def textbook_bpe(train_dir, alphabet, n_merges):
    """Classical BPE with no shortcuts: recount every pair at every step."""
    cid = {c: 2 + i for i, c in enumerate(alphabet)}
    freq = collections.Counter()
    for l in LANGS:
        for line in open(os.path.join(train_dir, f"{l}.txt"), encoding="utf-8").read().split("\n"):
            freq.update(line.split())
    words = {w: [1] + [cid.get(c, 0) for c in w] for w in freq}
    merges, nxt = [], 2 + len(alphabet)
    for _ in range(n_merges):
        cnt = collections.Counter()
        for w, s in words.items():
            for x, y in zip(s, s[1:]):
                if x and y:
                    cnt[(x, y)] += freq[w]
        # most frequent pair; ties go to the pair with the smaller ids
        best = max(cnt.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        merges.append(best)
        for w, s in words.items():
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                    out.append(nxt)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            words[w] = out
        nxt += 1
    return merges


def main():
    if not os.path.exists(EXE):
        sys.exit("Build first:  cargo build --release")
    if not os.path.exists(os.path.join(HERE, "data", "train", "hin.txt")):
        sys.exit("Download the data first:  python get_data.py")
    langs = ",".join(LANGS)
    dev = os.path.join("data", "flores", "dev")

    # 1. fast == brute force
    small = sample("small", 1500)
    for unit in ["line", "byte"]:
        for p in ["1", "2", "4", "8", "inf"]:
            outs = []
            for mode in ["fast", "naive"]:
                out = os.path.join(WORK, f"{mode}_{unit}_p{p}")
                extra = ["--naive"] if mode == "naive" else []
                run("train", "--langs", langs, "--train-dir", small, "--dev-dir", dev, "--unit", unit,
                    "--p", p, "--vocab-size", "3000", "--out", out, *extra)
                outs.append(open(os.path.join(out, "model.txt"), encoding="utf-8").read())
            report(f"fast trainer == brute force   (p={p}, unit={unit})", outs[0] == outs[1])

    # 2. p = 1 == textbook classical BPE
    tiny = sample("tiny", 300)
    out = os.path.join(WORK, "tiny_p1")
    run("train", "--langs", langs, "--train-dir", tiny, "--p", "1", "--vocab-size", "1500", "--out", out)
    alphabet, merges = read_model(out)
    report(f"p=1 == textbook classical BPE ({len(merges)} merges)", textbook_bpe(tiny, alphabet, len(merges)) == merges)

    # 3. evaluator agrees with the trainer's own token totals
    for p in ["1", "4", "inf"]:
        model = os.path.join(WORK, f"fast_line_p{p}")
        detail = os.path.join(WORK, f"detail_p{p}.csv")
        if os.path.exists(detail):
            os.remove(detail)
        run("eval", "--langs", langs, "--model", model, "--test-dir", small, "--detail", detail)
        from_eval = [r.split(",")[5] for r in open(detail, encoding="utf-8").read().split("\n")[1:] if r]
        from_train = [r.split("\t")[4] for r in open(os.path.join(model, "train_stats.tsv"), encoding="utf-8").read().split("\n")[1:] if r]
        report(f"evaluator token counts == trainer token counts (p={p})", from_eval == from_train)

    # 4. hybrid baseline
    def hybrid(frac, mode):
        out = os.path.join(WORK, f"hybrid_{mode}_{frac}")
        run("train", "--langs", langs, "--train-dir", small, "--dev-dir", dev, "--unit", "line", "--p", "inf",
            "--hybrid", frac, "--vocab-size", "3000", "--out", out, *(["--naive"] if mode == "naive" else []))
        return open(os.path.join(out, "model.txt"), encoding="utf-8").read()

    def plain(p):
        return open(os.path.join(WORK, f"fast_line_p{p}", "model.txt"), encoding="utf-8").read()

    for frac in ["0.25", "0.5", "0.9"]:
        report(f"hybrid: fast trainer == brute force   ({frac} classical)", hybrid(frac, "fast") == hybrid(frac, "naive"))
    report("hybrid with 0% classical merges == Parity-aware BPE", hybrid("0", "fast") == plain("inf"))
    report("hybrid with 100% classical merges == classical BPE", hybrid("1", "fast") == plain("1"))

    # 5. morphology penalty, Worst-N, moving window
    lex = os.path.join(WORK, "lexicon.tsv")
    with open(lex, "w", encoding="utf-8") as f:  # synthetic segmentations: frequent words split in the middle
        import collections
        cnt = collections.Counter()
        for l in LANGS:
            with open(os.path.join(small, f"{l}.txt"), encoding="utf-8") as g:
                for line in g:
                    cnt.update(line.split())
        n = 0
        for w, _ in cnt.most_common():
            if len(w) >= 4 and w.isalpha():
                f.write(f"{w}\t{w[:len(w) // 2]} {w[len(w) // 2:]}\n")
                n += 1
            if n >= 2000:
                break

    def train_pair(tag, *extra):
        outs = []
        for mode in ["fast", "naive"]:
            out = os.path.join(WORK, f"{tag}_{mode}")
            run("train", "--langs", langs, "--train-dir", small, "--dev-dir", dev, "--unit", "line",
                "--vocab-size", "3000", "--out", out, *extra, *(["--naive"] if mode == "naive" else []))
            outs.append(open(os.path.join(out, "model.txt"), encoding="utf-8").read())
        return outs[0] == outs[1], outs[0]

    for p, lam in [("1", "0.5"), ("4", "1"), ("inf", "0.5")]:
        same, _ = train_pair(f"lambda_{p}_{lam}", "--p", p, "--lambda", lam, "--morph-lexicon", lex)
        report(f"morphology penalty: fast trainer == brute force   (p={p}, lambda={lam})", same)
    _, with_lex = train_pair("lambda0", "--p", "4", "--lambda", "0", "--morph-lexicon", lex)
    report("lambda = 0 with a lexicon == no lexicon", with_lex == plain("4"))
    for n in ["2", "3"]:
        same, _ = train_pair(f"worstn{n}", "--p", "inf", "--worst-n", n)
        report(f"Worst-N: fast trainer == brute force   (N={n})", same)
    _, w1 = train_pair("worstn1", "--p", "inf", "--worst-n", "1", "--worst-n-refresh", "1")
    # Worst-1 pools every language tied for the worst cost; Parity-aware takes the first one.
    # They must agree until the first exact tie between two costs.
    log = open(os.path.join(WORK, "fast_line_pinf", "train_log.tsv"), encoding="utf-8").read().split("\n")[1:]
    first_tie = len(log)
    for i, line in enumerate(log):
        costs = line.split("\t")[4:]
        if len(set(costs)) < len(costs):
            first_tie = i
            break
    m_pa = read_model(os.path.join(WORK, "fast_line_pinf"))[1]
    m_w1 = read_model(os.path.join(WORK, "worstn1_fast"))[1]
    report(f"Worst-1 == Parity-aware up to the first exact tie in costs (merge {first_tie})", m_pa[:first_tie] == m_w1[:first_tie])
    same, _ = train_pair("window", "--p", "inf", "--window", "50", "--alpha", "1.5")
    report("moving window: fast trainer == brute force   (W=50, alpha=1.5)", same)
    # morph-eval: a model with no merges puts a boundary after every character, so recall is 1
    out = os.path.join(WORK, "nomerge")
    run("train", "--langs", langs, "--train-dir", tiny, "--p", "1", "--vocab-size", "1000", "--out", out)
    alphabet, merges = read_model(out)
    with open(os.path.join(out, "model.txt"), "w", encoding="utf-8") as f:  # strip the merges
        f.write("pfair-bpe 1\nbase char\nalphabet %d\n" % len(alphabet) + "".join(f"{ord(c)}\n" for c in alphabet) + "merges 0\n")
    r = subprocess.run([EXE, "morph-eval", "--model", out, "--gold", lex], capture_output=True, text=True, cwd=HERE).stdout
    report("morph-eval: a character-level model has boundary recall 1", " R 1.0000 " in r)

    # 6. merge: each language costs the same under the merged model as under its own script model (+- a little
    #    sharing of digits and Latin letters), and the merged file loads and encodes
    parts = []
    for tag, ls in [("deva", "hin,mar"), ("gujr", "guj"), ("latn", "eng")]:
        out = os.path.join(WORK, f"part_{tag}")
        run("train", "--langs", ls, "--train-dir", small, "--dev-dir", dev, "--unit", "line", "--p", "1", "--vocab-size", "1500", "--out", out)
        parts.append(out)
    merged = os.path.join(WORK, "merged")
    run("merge", "--models", ",".join(parts), "--out", merged)
    def cost(model, ls):
        r = subprocess.run([EXE, "eval", "--langs", ls, "--model", model, "--test-dir", small, "--unit", "line"], capture_output=True, text=True, cwd=HERE).stdout
        return {l.split()[0]: float(l.split()[-1]) for l in r.splitlines() if l.split() and l.split()[0] in LANGS}
    c_merged = cost(merged, langs)
    c_parts = {}
    for out, ls in zip(parts, ["hin,mar", "guj", "eng"]):
        c_parts.update(cost(out, ls))
    report("merge: every language costs within 3% of its own script model", all(abs(c_merged[l] / c_parts[l] - 1) < 0.03 for l in LANGS))

    print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
