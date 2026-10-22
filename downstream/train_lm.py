#!/usr/bin/env python3
"""Downstream check: train one small causal language model per tokenizer on the
same training text and compare per-language bits per byte on FLORES devtest.

    python downstream/train_lm.py --tokenizer results/models/p8_v32000_line --name "p=8" \
        --langs hin,mar,guj,eng --train-dir data/train --test-dir data/flores/devtest \
        --out results_lm/p8  [--epochs 1] [--layers 6 --dim 512 --heads 8 --seq-len 512]

Every tokenizer gets exactly the same text (the same files, the same number of
epochs), so differences come from the tokenizer alone. The model is a GPT-2
style transformer built with HuggingFace transformers; the data goes through
`pfair encode-ids`, so no HuggingFace tokenizer is needed.

Outputs in --out:
    lm_results.csv   per language: tokens per sentence, NLL per sentence (nats),
                     bits per byte, bits per character
    train_log.csv    training loss every --log-every steps
    config.json      the settings, the parameter count, the vocabulary size

Bits per byte is the standard tokenizer-independent measure: a lower value means
the model predicts that language's text better, whatever the tokenizer did to it.
Needs: pip install torch transformers   (a GPU makes this minutes instead of hours).
"""
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(HERE, "target", "release", "pfair.exe" if os.name == "nt" else "pfair")


def vocab_size(model_dir):
    with open(os.path.join(model_dir, "vocab.txt"), encoding="utf-8") as f:
        return sum(1 for _ in f)


def encode_file(model_dir, path, out_path):
    if not os.path.exists(out_path):
        subprocess.run([EXE, "encode-ids", "--model", model_dir, "--input", path, "--output", out_path], check=True)
    with open(out_path, encoding="utf-8") as f:
        return [list(map(int, line.split())) for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="pfair model directory")
    ap.add_argument("--name", default=None, help="label in the results (default: the model directory)")
    ap.add_argument("--langs", default="hin,mar,guj,eng")
    ap.add_argument("--train-dir", default="data/train")
    ap.add_argument("--test-dir", default="data/flores/devtest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lines", type=int, default=0, help="use only the first N lines of each training file (0 = all)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = run the epochs)")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-seqs", type=int, default=32, help="sequences per step (tokens per step = batch-seqs * seq-len)")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    name = a.name or os.path.basename(a.tokenizer.rstrip("/"))
    langs = a.langs.split(",")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    # ---- data: same text for every tokenizer ----
    V = vocab_size(a.tokenizer)
    EOS = V  # one extra id marks line ends and starts every evaluation line
    stream = []
    n_lines = 0
    for l in langs:
        src = os.path.join(a.train_dir, f"{l}.txt")
        ids = encode_file(a.tokenizer, src, os.path.join(a.out, f"train_{l}.ids"))
        if a.max_lines:
            ids = ids[: a.max_lines]
        n_lines += len(ids)
        stream.extend(ids)
    rng.shuffle(stream)  # interleave the languages
    flat = [EOS]
    for ids in stream:
        flat.extend(ids)
        flat.append(EOS)
    del stream
    data = torch.tensor(flat, dtype=torch.long)
    del flat
    T = a.seq_len + 1
    n_chunks = (len(data) - 1) // T
    steps_per_epoch = max(1, n_chunks // a.batch_seqs)
    total_steps = int(round(a.epochs * steps_per_epoch))
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    print(f"{name}: vocab {V}, {n_lines} training lines, {len(data) / 1e6:.1f} M tokens, {n_chunks} chunks of {T}, "
          f"{steps_per_epoch} steps per epoch, {total_steps} steps on {a.device}", flush=True)

    # ---- model ----
    cfg = GPT2Config(vocab_size=V + 1, n_positions=T, n_embd=a.dim, n_layer=a.layers, n_head=a.heads,
                     bos_token_id=EOS, eos_token_id=EOS, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))
    use_amp = a.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- train ----
    order = list(range(n_chunks))
    log = open(os.path.join(a.out, "train_log.csv"), "w", newline="")
    logw = csv.writer(log)
    logw.writerow(["step", "loss", "lr", "seconds"])
    t0 = time.time()
    step, pos = 0, 0
    model.train()
    while step < total_steps:
        if pos == 0:
            rng.shuffle(order)
        idx = order[pos: pos + a.batch_seqs]
        pos = pos + a.batch_seqs if pos + a.batch_seqs < len(order) else 0
        x = torch.stack([data[i * T: i * T + T] for i in idx]).to(a.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = model(input_ids=x, labels=x).loss  # transformers shifts the labels internally
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % a.log_every == 0 or step == total_steps:
            logw.writerow([step, f"{loss.item():.4f}", f"{sched.get_last_lr()[0]:.2e}", f"{time.time() - t0:.0f}"])
            log.flush()
            print(f"  step {step}/{total_steps}  loss {loss.item():.3f}  [{time.time() - t0:.0f}s]", flush=True)
    log.close()

    # ---- evaluate: per-language NLL on FLORES devtest, each line started by EOS ----
    model.eval()
    rows = []
    for l in langs:
        src = os.path.join(a.test_dir, f"{l}.txt")
        ids = encode_file(a.tokenizer, src, os.path.join(a.out, f"test_{l}.ids"))
        with open(src, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f]
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_chars = sum(len(t) for t in texts)
        nll, n_tok = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ids), 16):
                batch = [[EOS] + s[: a.seq_len - 1] for s in ids[i: i + 16]]
                L = max(len(s) for s in batch)
                x = torch.full((len(batch), L), EOS, dtype=torch.long)
                mask = torch.zeros((len(batch), L), dtype=torch.bool)
                for j, s in enumerate(batch):
                    x[j, : len(s)] = torch.tensor(s)
                    mask[j, : len(s)] = True
                x, mask = x.to(a.device), mask.to(a.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(input_ids=x, attention_mask=mask).logits.float()
                lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None])[..., 0]
                m = mask[:, 1:]
                nll += float(-(lp * m).sum())
                n_tok += int(m.sum())
        rows.append({"method": name, "lang": l, "sentences": len(texts), "tokens_per_sentence": round(n_tok / len(texts), 3),
                     "nll_per_sentence": round(nll / len(texts), 3), "bits_per_byte": round(nll / math.log(2) / n_bytes, 4),
                     "bits_per_char": round(nll / math.log(2) / n_chars, 4), "nll_per_token": round(nll / max(1, n_tok), 4)})
        print(f"  {l}: {rows[-1]['tokens_per_sentence']} tok/sent, bits/byte {rows[-1]['bits_per_byte']}, nll/sent {rows[-1]['nll_per_sentence']}")
    with open(os.path.join(a.out, "lm_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bpb = [r["bits_per_byte"] for r in rows]
    json.dump({**vars(a), "name": name, "vocab": V, "params": n_params, "steps": total_steps, "train_tokens": len(data),
               "mean_bits_per_byte": sum(bpb) / len(bpb)}, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"done: {n_params / 1e6:.1f} M parameters, {total_steps} steps, mean bits/byte {sum(bpb) / len(bpb):.4f}; results in {a.out}/lm_results.csv")


if __name__ == "__main__":
    main()
