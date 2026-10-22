//! pfair: classical BPE, Parity-aware BPE and p-fair BPE in one fast trainer.
//!
//! Merge rule (one family, one knob p):
//!
//!     score(a,b) = sum over languages l of  t_l^(p-1) * c_l(a,b)
//!
//!   c_l(a,b) = how often the pair (a,b) occurs in language l's training text
//!   t_l      = current token cost of language l (tokens per byte / char / parallel line)
//!
//!   p = 1    every weight is 1, so score = total pair count  -> classical BPE
//!   p = inf  only the worst-off language counts              -> Parity-aware BPE (base variant)
//!   between  a blend of the two                              -> p-fair BPE
//!
//! Why it is fast: t_l only ever goes down and the count of an existing pair only
//! ever goes down, so an old score is always an upper bound on the current score.
//! That lets us keep scores in a max-heap and only re-check the few entries that
//! reach the top ("lazy greedy"), instead of rescoring every pair at every merge.
//! `--naive` switches to the brute-force rescoring, to prove both give identical merges.
//!
//! Optional extras on top of the rule above:
//!   --lambda L --morph-lexicon FILE   penalise merges that cross morpheme boundaries
//!   --worst-n N                       Worst-N parity baseline (pool the N worst-off languages)
//!   --window W --alpha A              moving-window Parity-aware baseline
//!   --pi uniform                      weight languages equally instead of by text share
//!
//! Subcommands: train, eval, encode, encode-ids, morph-eval, merge.   Run `pfair help`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufWriter, Write};
use std::time::Instant;

// ---------------------------------------------------------------------------
// A small, fast, deterministic hasher (std's default is slow and randomised).
// ---------------------------------------------------------------------------
#[derive(Default, Clone, Copy)]
struct Fx(u64);
impl Fx {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}
impl Hasher for Fx {
    fn write(&mut self, mut b: &[u8]) {
        while b.len() >= 8 {
            self.add(u64::from_le_bytes(b[..8].try_into().unwrap()));
            b = &b[8..];
        }
        if b.len() >= 4 {
            self.add(u32::from_le_bytes(b[..4].try_into().unwrap()) as u64);
            b = &b[4..];
        }
        for &x in b {
            self.add(x as u64);
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64)
    }
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64)
    }
    fn write_u64(&mut self, i: u64) {
        self.add(i)
    }
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64)
    }
    fn finish(&self) -> u64 {
        // final mix so that the low bits (used for the bucket) depend on all input bits
        let h = (self.0 ^ (self.0 >> 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        h ^ (h >> 29)
    }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

// ---------------------------------------------------------------------------
// Symbols.  id 0 = <unk>, id 1 = word-start marker, then the base alphabet,
// then one new id per merge.
// ---------------------------------------------------------------------------
const UNK: u32 = 0;
const MARK: u32 = 1;
const MARK_STR: &str = "\u{2581}"; // "▁", as in SentencePiece

#[inline]
fn pack(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

#[derive(Clone, Copy, PartialEq)]
enum Unit {
    Byte,
    Char,
    Line,
}
impl Unit {
    fn parse(s: &str) -> Unit {
        match s {
            "byte" => Unit::Byte,
            "char" => Unit::Char,
            "line" => Unit::Line,
            _ => die("--unit must be byte, char or line"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Unit::Byte => "byte",
            Unit::Char => "char",
            Unit::Line => "line",
        }
    }
}

/// Size of a text in the three units we can normalise token counts by.
#[derive(Default, Clone, Copy)]
struct Units {
    lines: u64,
    bytes: u64,
    chars: u64,
}
impl Units {
    fn get(&self, u: Unit) -> f64 {
        match u {
            Unit::Byte => self.bytes as f64,
            Unit::Char => self.chars as f64,
            Unit::Line => self.lines as f64,
        }
    }
}

fn read_text(path: &str) -> String {
    let b = fs::read(path).unwrap_or_else(|e| die(&format!("cannot read {path}: {e}")));
    match String::from_utf8(b) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
}

// ---------------------------------------------------------------------------
// Model = base alphabet + ordered list of merges.
// ---------------------------------------------------------------------------
struct Model {
    byte_base: bool,
    alphabet: Vec<char>, // unused when byte_base
    merges: Vec<(u32, u32)>,
    cmap: FxMap<char, u32>,
    ranks: FxMap<u64, (u32, u32)>, // pair -> (rank, new id)
}

impl Model {
    fn new(byte_base: bool, alphabet: Vec<char>, merges: Vec<(u32, u32)>) -> Model {
        let mut m = Model { byte_base, alphabet, merges, cmap: FxMap::default(), ranks: FxMap::default() };
        for (i, &c) in m.alphabet.iter().enumerate() {
            m.cmap.insert(c, 2 + i as u32);
        }
        let nb = m.n_base();
        for (i, &(a, b)) in m.merges.iter().enumerate() {
            m.ranks.insert(pack(a, b), (i as u32, nb + i as u32));
        }
        m
    }
    fn n_base(&self) -> u32 {
        2 + if self.byte_base { 256 } else { self.alphabet.len() as u32 }
    }
    fn vocab_size(&self) -> usize {
        self.n_base() as usize + self.merges.len()
    }

    /// word -> [MARK, base symbols...]
    #[inline]
    fn symbolize(&self, w: &str, out: &mut Vec<u32>) {
        out.clear();
        out.push(MARK);
        if self.byte_base {
            out.extend(w.bytes().map(|b| 2 + b as u32));
        } else {
            out.extend(w.chars().map(|c| *self.cmap.get(&c).unwrap_or(&UNK)));
        }
    }

    /// Apply the merges to one word (lowest-rank pair first, which is the same
    /// as replaying the merges in training order).
    fn encode_word(&self, w: &str, syms: &mut Vec<u32>) {
        self.symbolize(w, syms);
        loop {
            let mut best: Option<(u32, u32, u32, u32)> = None; // rank, new, a, b
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(r, n)) = self.ranks.get(&pack(syms[i], syms[i + 1])) {
                    if best.map_or(true, |x| r < x.0) {
                        best = Some((r, n, syms[i], syms[i + 1]));
                    }
                }
            }
            let Some((_, n, a, b)) = best else { break };
            let mut j = 0;
            let mut i = 0;
            while i < syms.len() {
                if i + 1 < syms.len() && syms[i] == a && syms[i + 1] == b {
                    syms[j] = n;
                    i += 2;
                } else {
                    syms[j] = syms[i];
                    i += 1;
                }
                j += 1;
            }
            syms.truncate(j);
        }
    }

    /// Length of every token in base units (characters, or bytes for a byte-level
    /// model). <unk> stands for one character; the word marker has length 0.
    fn token_lens(&self) -> Vec<u32> {
        let mut t: Vec<u32> = vec![1, 0];
        t.extend(std::iter::repeat(1).take(self.n_base() as usize - 2));
        for &(a, b) in &self.merges {
            let x = t[a as usize] + t[b as usize];
            t.push(x);
        }
        t
    }

    /// Bytes of every token, for printing.
    fn token_bytes(&self) -> Vec<Vec<u8>> {
        let mut t: Vec<Vec<u8>> = vec![b"<unk>".to_vec(), MARK_STR.as_bytes().to_vec()];
        if self.byte_base {
            for b in 0..=255u8 {
                t.push(vec![b]);
            }
        } else {
            for &c in &self.alphabet {
                t.push(c.to_string().into_bytes());
            }
        }
        for &(a, b) in &self.merges {
            let mut s = t[a as usize].clone();
            s.extend_from_slice(&t[b as usize]);
            t.push(s);
        }
        t
    }

    fn save(&self, dir: &str) {
        fs::create_dir_all(dir).unwrap_or_else(|e| die(&format!("cannot create {dir}: {e}")));
        let mut f = BufWriter::new(fs::File::create(format!("{dir}/model.txt")).unwrap());
        writeln!(f, "pfair-bpe 1").unwrap();
        writeln!(f, "base {}", if self.byte_base { "byte" } else { "char" }).unwrap();
        writeln!(f, "alphabet {}", self.alphabet.len()).unwrap();
        for &c in &self.alphabet {
            writeln!(f, "{}", c as u32).unwrap();
        }
        writeln!(f, "merges {}", self.merges.len()).unwrap();
        for &(a, b) in &self.merges {
            writeln!(f, "{a} {b}").unwrap();
        }
        let mut v = BufWriter::new(fs::File::create(format!("{dir}/vocab.txt")).unwrap());
        for (i, t) in self.token_bytes().iter().enumerate() {
            writeln!(v, "{i}\t{}", String::from_utf8_lossy(t)).unwrap();
        }
    }

    fn load(dir: &str) -> Model {
        let text = read_text(&format!("{dir}/model.txt"));
        let mut it = text.lines();
        if it.next() != Some("pfair-bpe 1") {
            die("not a pfair model file");
        }
        let byte_base = it.next() == Some("base byte");
        let n: usize = it.next().and_then(|l| l.strip_prefix("alphabet ")).and_then(|x| x.parse().ok()).unwrap_or_else(|| die("bad model file"));
        let mut alphabet = Vec::with_capacity(n);
        for _ in 0..n {
            let cp: u32 = it.next().and_then(|x| x.parse().ok()).unwrap_or_else(|| die("bad model file"));
            alphabet.push(char::from_u32(cp).unwrap_or_else(|| die("bad model file")));
        }
        let m: usize = it.next().and_then(|l| l.strip_prefix("merges ")).and_then(|x| x.parse().ok()).unwrap_or_else(|| die("bad model file"));
        let mut merges = Vec::with_capacity(m);
        for _ in 0..m {
            let l = it.next().unwrap_or_else(|| die("bad model file"));
            let (a, b) = l.split_once(' ').unwrap_or_else(|| die("bad model file"));
            merges.push((a.parse().unwrap(), b.parse().unwrap()));
        }
        Model::new(byte_base, alphabet, merges)
    }
}

// ---------------------------------------------------------------------------
// Word table: every distinct word once, with one frequency per column.
// Columns 0..L are the training text of each language; if a dev set is given,
// columns L..2L are the dev text of each language.
// ---------------------------------------------------------------------------
struct WordTable {
    ncol: usize,
    index: FxMap<String, u32>,
    freq: Vec<u64>,
}
impl WordTable {
    fn add_text(&mut self, text: &str, col: usize) -> Units {
        let mut u = Units::default();
        for line in text.lines() {
            let mut n = 0u64;
            for w in line.split_whitespace() {
                n += 1;
                u.bytes += w.len() as u64;
                u.chars += w.chars().count() as u64;
                let id = match self.index.get(w) {
                    Some(&i) => i as usize,
                    None => {
                        let i = self.index.len();
                        self.index.insert(w.to_string(), i as u32);
                        self.freq.resize(self.freq.len() + self.ncol, 0);
                        i
                    }
                };
                self.freq[id * self.ncol + col] += 1;
            }
            if n > 0 {
                u.lines += 1;
                u.bytes += n - 1; // the spaces between words
                u.chars += n - 1;
            }
        }
        u
    }
}

// ---------------------------------------------------------------------------
// Trainer state.
// ---------------------------------------------------------------------------
struct Trainer {
    nl: usize,
    ncol: usize,
    words: Vec<Vec<u32>>,
    freq: Vec<u64>,          // words.len() * ncol
    pair_ix: FxMap<u64, u32>, // packed pair -> pair id
    pairs: Vec<(u32, u32)>,
    pcount: Vec<i64>,        // pairs.len() * nl, training counts per language
    pwords: Vec<Vec<u32>>,   // words that contain (or once contained) the pair
    tokens: Vec<i64>,        // current token total per column
    // --- morphology penalty (only used when lambda > 0) ---
    lambda: f64,
    tok_len: Vec<u32>,       // length of every token id in base units
    bounds: Vec<Vec<u32>>,   // per word: sorted morpheme-boundary positions (boundary after unit j), empty if unknown
    pcross: Vec<i64>,        // pairs.len() * nl: occurrences of the pair whose junction is a morpheme boundary
}

/// Is there a morpheme boundary just before base-unit index `pos`?
#[inline]
fn crosses(bounds: &[u32], pos: u32) -> bool {
    pos > 0 && bounds.binary_search(&(pos - 1)).is_ok()
}

impl Trainer {
    #[inline]
    fn pid(&mut self, a: u32, b: u32, created: &mut Vec<u32>) -> usize {
        let key = pack(a, b);
        if let Some(&p) = self.pair_ix.get(&key) {
            return p as usize;
        }
        let p = self.pairs.len();
        self.pair_ix.insert(key, p as u32);
        self.pairs.push((a, b));
        self.pcount.resize(self.pcount.len() + self.nl, 0);
        self.pcross.resize(self.pcross.len() + self.nl, 0);
        self.pwords.push(Vec::new());
        created.push(p as u32);
        p
    }

    fn init_pairs(&mut self) {
        let mut created = Vec::new();
        for w in 0..self.words.len() {
            let syms = std::mem::take(&mut self.words[w]);
            let bounds = std::mem::take(&mut self.bounds[w]);
            let mut pos = 0u32;
            for i in 0..syms.len().saturating_sub(1) {
                let (a, b) = (syms[i], syms[i + 1]);
                pos += self.tok_len[a as usize];
                if a == UNK || b == UNK {
                    continue;
                }
                let p = self.pid(a, b, &mut created);
                let x = !bounds.is_empty() && crosses(&bounds, pos);
                for l in 0..self.nl {
                    self.pcount[p * self.nl + l] += self.freq[w * self.ncol + l] as i64;
                    if x {
                        self.pcross[p * self.nl + l] += self.freq[w * self.ncol + l] as i64;
                    }
                }
                if self.pwords[p].last() != Some(&(w as u32)) {
                    self.pwords[p].push(w as u32);
                }
            }
            self.bounds[w] = bounds;
            for c in 0..self.ncol {
                self.tokens[c] += (self.freq[w * self.ncol + c] * syms.len() as u64) as i64;
            }
            self.words[w] = syms;
        }
    }

    /// Replace every (a,b) by `new` in every word and update all the counts.
    /// Returns the ids of the pairs that came into existence.
    fn apply(&mut self, p: usize, new: u32) -> Vec<u32> {
        let (a, b) = self.pairs[p];
        let mut list = std::mem::take(&mut self.pwords[p]);
        list.sort_unstable();
        list.dedup();
        let mut created = Vec::new();
        let mut f = vec![0u64; self.ncol];
        let mut neu: Vec<u32> = Vec::new();
        for &w in &list {
            let w = w as usize;
            let old = std::mem::take(&mut self.words[w]);
            neu.clear();
            let mut k = 0u64;
            let mut i = 0;
            while i < old.len() {
                if i + 1 < old.len() && old[i] == a && old[i + 1] == b {
                    neu.push(new);
                    i += 2;
                    k += 1;
                } else {
                    neu.push(old[i]);
                    i += 1;
                }
            }
            if k == 0 {
                self.words[w] = old; // stale entry in the word list
                continue;
            }
            f.copy_from_slice(&self.freq[w * self.ncol..(w + 1) * self.ncol]);
            let bounds = std::mem::take(&mut self.bounds[w]);
            let morph = !bounds.is_empty();
            // take the word's old pairs out of the counts, put its new pairs in
            let mut pos = 0u32;
            for i in 0..old.len() - 1 {
                let (x, y) = (old[i], old[i + 1]);
                pos += self.tok_len[x as usize];
                if x == UNK || y == UNK {
                    continue;
                }
                let q = self.pair_ix[&pack(x, y)] as usize;
                let cr = morph && crosses(&bounds, pos);
                for l in 0..self.nl {
                    self.pcount[q * self.nl + l] -= f[l] as i64;
                    if cr {
                        self.pcross[q * self.nl + l] -= f[l] as i64;
                    }
                }
            }
            let mut pos = 0u32;
            for i in 0..neu.len().saturating_sub(1) {
                let (x, y) = (neu[i], neu[i + 1]);
                pos += self.tok_len[x as usize];
                if x == UNK || y == UNK {
                    continue;
                }
                let q = self.pid(x, y, &mut created);
                let cr = morph && crosses(&bounds, pos);
                for l in 0..self.nl {
                    self.pcount[q * self.nl + l] += f[l] as i64;
                    if cr {
                        self.pcross[q * self.nl + l] += f[l] as i64;
                    }
                }
                if (x == new || y == new) && self.pwords[q].last() != Some(&(w as u32)) {
                    self.pwords[q].push(w as u32);
                }
            }
            self.bounds[w] = bounds;
            for c in 0..self.ncol {
                self.tokens[c] -= (k * f[c]) as i64;
            }
            self.words[w] = neu.clone();
        }
        created
    }

    /// Weighted score of a pair: sum over languages of w_l * (c_l - lambda * cross_l).
    /// With lambda = 0 this is the plain weighted pair count.
    #[inline]
    fn score(&self, p: usize, wts: &[f64]) -> f64 {
        let c = &self.pcount[p * self.nl..(p + 1) * self.nl];
        let mut s = 0.0;
        if self.lambda > 0.0 {
            let x = &self.pcross[p * self.nl..(p + 1) * self.nl];
            for l in 0..self.nl {
                s += wts[l] * (c[l] as f64 - self.lambda * x[l] as f64);
            }
        } else {
            for l in 0..self.nl {
                s += wts[l] * c[l] as f64;
            }
        }
        s
    }

    /// Score restricted to one language (used by the Parity-aware rule).
    #[inline]
    fn score_lang(&self, p: usize, l: usize) -> f64 {
        let c = self.pcount[p * self.nl + l] as f64;
        if self.lambda > 0.0 {
            c - self.lambda * self.pcross[p * self.nl + l] as f64
        } else {
            c
        }
    }
}

// ---------------------------------------------------------------------------
// Morphological segmentation files.
//   seg1+seg2+seg3                 (also '|' as separator)
//   word<TAB>seg1 seg2 seg3        (segments separated by spaces, '+' or '|')
// A line whose segments do not concatenate to the word (sandhi) is skipped.
// Returns (word, boundary positions in base units) for the usable lines, and
// the number of skipped lines.
// ---------------------------------------------------------------------------
fn read_segmentations(path: &str, byte_base: bool) -> (Vec<(String, Vec<u32>)>, usize) {
    let text = read_text(path);
    let mut out = Vec::new();
    let mut skipped = 0usize;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (word, segs): (String, Vec<String>) = if let Some((w, rest)) = line.split_once('\t') {
            let segs: Vec<String> = rest.split(|c: char| c == '+' || c == '|' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();
            (w.trim().to_string(), segs)
        } else {
            let segs: Vec<String> = line.split(|c: char| c == '+' || c == '|').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();
            (segs.concat(), segs)
        };
        if segs.len() < 2 || segs.concat() != word || word.contains(char::is_whitespace) {
            skipped += 1;
            continue;
        }
        let mut b = Vec::new();
        let mut pos = 0u32;
        for seg in &segs[..segs.len() - 1] {
            pos += if byte_base { seg.len() as u32 } else { seg.chars().count() as u32 };
            b.push(pos - 1);
        }
        out.push((word, b));
    }
    (out, skipped)
}

type Entry = (u64, Reverse<u64>, u32, u32); // (key, tie-break: smaller pair first, stamp, pair id)

struct TrainOpts {
    langs: Vec<String>,
    train_dir: String,
    dev_dir: Option<String>,
    out: String,
    vocab_size: usize,
    p: f64, // f64::INFINITY for Parity-aware
    unit: Unit,
    byte_base: bool,
    min_char_freq: u64,
    naive: bool,
    /// Hybrid baseline from the Parity-aware BPE paper: this fraction of the
    /// merges is learned with the classical rule first, the rest with the
    /// Parity-aware rule. Only used together with p = inf.
    hybrid: Option<f64>,
    /// Morphology penalty: an occurrence of a pair whose junction is a morpheme
    /// boundary contributes w_l * (1 - lambda) instead of w_l to the score.
    /// For 0 <= lambda <= 1 every term stays non-negative and non-increasing, so
    /// the lazy heap remains exact; lambda > 1 is only allowed with --naive.
    lambda: f64,
    morph_lexicon: Vec<String>,
    /// Worst-N parity (Kanjirangat et al., 2026): pool the training counts of the
    /// languages whose cost is at least the mean cost of the N worst-off languages;
    /// the set is recomputed every `refresh` merges.
    worst_n: Option<(usize, u32)>,
    /// Moving-window Parity-aware (Foroutan et al., 2026): a language that led more
    /// than alpha*W/L of the last W merges cannot be selected.
    window: Option<(usize, f64)>,
    /// Language weights pi: by share of training text (the derivation in the paper)
    /// or uniform (every language counts the same whatever its amount of text).
    pi_uniform: bool,
}

fn train(o: &TrainOpts) {
    let t0 = Instant::now();
    let nl = o.langs.len();
    let has_dev = o.dev_dir.is_some();
    let ncol = if has_dev { 2 * nl } else { nl };
    let cost_off = if has_dev { nl } else { 0 }; // which columns the cost t_l is measured on
    if o.unit == Unit::Line && !has_dev {
        die("--unit line needs --dev-dir with parallel text (the same sentences in every language)");
    }

    // ---- read text into the word table ----
    let mut wt = WordTable { ncol, index: FxMap::default(), freq: Vec::new() };
    let mut units = vec![Units::default(); ncol];
    for (l, lang) in o.langs.iter().enumerate() {
        let path = format!("{}/{}.txt", o.train_dir, lang);
        units[l] = wt.add_text(&read_text(&path), l);
        eprintln!("  train {lang}: {} lines, {:.1} MB, {:.1} M chars", units[l].lines, units[l].bytes as f64 / 1e6, units[l].chars as f64 / 1e6);
    }
    if let Some(d) = &o.dev_dir {
        for (l, lang) in o.langs.iter().enumerate() {
            units[nl + l] = wt.add_text(&read_text(&format!("{d}/{lang}.txt")), nl + l);
        }
        let n0 = units[nl].lines;
        if o.unit == Unit::Line && units[nl..].iter().any(|u| u.lines != n0) {
            die("dev files have different numbers of lines, so they are not parallel");
        }
    }

    // ---- base alphabet ----
    let mut alphabet: Vec<char> = Vec::new();
    if !o.byte_base {
        let mut cf: FxMap<char, u64> = FxMap::default();
        for (w, &id) in &wt.index {
            let tf: u64 = wt.freq[id as usize * ncol..id as usize * ncol + nl].iter().sum();
            if tf > 0 {
                for c in w.chars() {
                    *cf.entry(c).or_insert(0) += tf;
                }
            }
        }
        let total: u64 = cf.values().sum();
        let mut v: Vec<(char, u64)> = cf.into_iter().filter(|x| x.1 >= o.min_char_freq).collect();
        v.sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
        let kept: u64 = v.iter().map(|x| x.1).sum();
        alphabet = v.into_iter().map(|x| x.0).collect();
        eprintln!("  alphabet: {} characters, covering {:.4}% of training characters", alphabet.len(), 100.0 * kept as f64 / total.max(1) as f64);
    }
    let mut model = Model::new(o.byte_base, alphabet, Vec::new());
    let n_base = model.n_base() as usize;
    if o.vocab_size <= n_base {
        die(&format!("--vocab-size {} is not larger than the base alphabet ({n_base})", o.vocab_size));
    }
    let n_merges = o.vocab_size - n_base;

    // ---- symbolise ----
    let nw = wt.index.len();
    let mut tr = Trainer {
        nl,
        ncol,
        words: vec![Vec::new(); nw],
        freq: std::mem::take(&mut wt.freq),
        pair_ix: FxMap::default(),
        pairs: Vec::new(),
        pcount: Vec::new(),
        pwords: Vec::new(),
        tokens: vec![0; ncol],
        lambda: o.lambda,
        tok_len: model.token_lens(),
        bounds: vec![Vec::new(); nw],
        pcross: Vec::new(),
    };
    for (w, &id) in &wt.index {
        let mut s = Vec::new();
        model.symbolize(w, &mut s);
        tr.words[id as usize] = s;
    }
    // ---- morpheme boundaries of the training words that are in the lexicon ----
    if !o.morph_lexicon.is_empty() {
        let (mut found, mut total, mut skipped, mut covered_tokens) = (0usize, 0usize, 0usize, 0u64);
        for path in &o.morph_lexicon {
            let (entries, sk) = read_segmentations(path, o.byte_base);
            skipped += sk;
            for (word, b) in entries {
                total += 1;
                if let Some(&id) = wt.index.get(&word) {
                    if tr.bounds[id as usize].is_empty() {
                        found += 1;
                        covered_tokens += tr.freq[id as usize * ncol..id as usize * ncol + nl].iter().sum::<u64>();
                        tr.bounds[id as usize] = b;
                    }
                }
            }
        }
        let all_tokens: u64 = (0..nw).map(|w| tr.freq[w * ncol..w * ncol + nl].iter().sum::<u64>()).sum();
        eprintln!(
            "  morphology: {found} of {total} lexicon words occur in the training text ({skipped} lines skipped); they cover {:.2}% of training word occurrences; lambda = {}",
            100.0 * covered_tokens as f64 / all_tokens.max(1) as f64,
            o.lambda
        );
        if o.lambda == 0.0 {
            eprintln!("  (lambda is 0, so the lexicon has no effect on training)");
        }
    } else if o.lambda > 0.0 {
        die("--lambda needs --morph-lexicon FILE (morpheme segmentations of training words)");
    }
    drop(wt);
    tr.init_pairs();
    eprintln!("  {} distinct words, {} distinct pairs, {} merges to learn  [{:.1}s]", tr.words.len(), tr.pairs.len(), n_merges, t0.elapsed().as_secs_f64());

    let denom: Vec<f64> = (0..nl).map(|l| units[cost_off + l].get(o.unit).max(1.0)).collect();
    let costs = |tr: &Trainer| -> Vec<f64> { (0..nl).map(|l| tr.tokens[cost_off + l] as f64 / denom[l]).collect() };
    // pi: by share of training text (factor 1) or uniform (divide by the language's size)
    let pi_factor: Vec<f64> = if o.pi_uniform {
        let sizes: Vec<f64> = (0..nl).map(|l| if o.byte_base { units[l].bytes as f64 } else { units[l].chars as f64 }.max(1.0)).collect();
        let mean = sizes.iter().sum::<f64>() / nl as f64;
        sizes.iter().map(|s| mean / s).collect()
    } else {
        vec![1.0; nl]
    };
    // Hybrid: the first `switch_at` merges use the classical rule, the rest the Parity-aware rule.
    let switch_at: Option<u32> = o.hybrid.map(|f| (f * n_merges as f64).round() as u32);
    let p_fin = if switch_at.is_some() { 1.0 } else { o.p }; // p used while the rule is the weighted-sum one
    let worst_n = o.worst_n;
    let is_inf = |step: u32| -> bool { worst_n.is_none() && switch_at.map_or(o.p.is_infinite(), |k| step >= k) };
    // Worst-N: the pooled set of languages, recomputed every `refresh` steps.
    let worst_set = |t: &[f64]| -> Vec<bool> {
        let (n, _) = worst_n.unwrap();
        let mut order: Vec<usize> = (0..nl).collect();
        order.sort_by(|&x, &y| t[y].partial_cmp(&t[x]).unwrap().then(x.cmp(&y)));
        let n = n.min(nl).max(1);
        let mean_n = order[..n].iter().map(|&l| t[l]).sum::<f64>() / n as f64;
        (0..nl).map(|l| t[l] >= mean_n - 1e-12).collect()
    };
    let mut wn_set: Vec<bool> = vec![true; nl];
    let weights = |t: &[f64], wn_set: &[bool]| -> Vec<f64> {
        if worst_n.is_some() {
            return wn_set.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        }
        t.iter().zip(&pi_factor).map(|(&x, &f)| f * if p_fin == 1.0 { 1.0 } else { x.powf(p_fin - 1.0) }).collect()
    };
    // Moving window: how often each language led one of the last W merges.
    let mut win: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut win_count = vec![0usize; nl];
    let win_limit: Option<f64> = o.window.map(|(w, alpha)| alpha * w as f64 / nl as f64);

    // ---- heaps: heaps[0] for finite p, one per language for the Parity-aware rule ----
    let mut heaps: Vec<BinaryHeap<Entry>> = vec![BinaryHeap::new(); nl.max(1)];
    let push_new = |tr: &Trainer, heaps: &mut Vec<BinaryHeap<Entry>>, ids: &[u32], wts: &[f64], stamp: u32, inf: bool| {
        for &q in ids {
            let q = q as usize;
            let key = Reverse(pack(tr.pairs[q].0, tr.pairs[q].1));
            if inf {
                for l in 0..nl {
                    let c = tr.score_lang(q, l);
                    if c > 0.0 {
                        heaps[l].push((c.to_bits(), key, stamp, q as u32));
                    }
                }
            } else {
                let s = tr.score(q, wts);
                if s > 0.0 {
                    heaps[0].push((s.to_bits(), key, stamp, q as u32));
                }
            }
        }
    };
    if worst_n.is_some() {
        wn_set = worst_set(&costs(&tr));
    }
    if !o.naive {
        let all: Vec<u32> = (0..tr.pairs.len() as u32).collect();
        let w0 = weights(&costs(&tr), &wn_set);
        push_new(&tr, &mut heaps, &all, &w0, 0, is_inf(0));
    }

    fs::create_dir_all(&o.out).unwrap_or_else(|e| die(&format!("cannot create {}: {e}", o.out)));
    let mut log = BufWriter::new(fs::File::create(format!("{}/train_log.tsv", o.out)).unwrap());
    write!(log, "step\ttoken\tlead_lang\tscore").unwrap();
    for l in &o.langs {
        write!(log, "\tcost_{l}").unwrap();
    }
    writeln!(log).unwrap();
    let mut tok_bytes = model.token_bytes();
    let mut lead = vec![0u64; nl];
    let mut refreshes: u64 = 0;

    for step in 0..n_merges as u32 {
        let t = costs(&tr);
        let inf = is_inf(step);
        let mut rebuild = !o.naive && step > 0 && switch_at == Some(step); // hybrid: the rule changes here
        if let Some((_, refresh)) = worst_n {
            if step > 0 && step % refresh.max(1) == 0 {
                let s = worst_set(&t);
                if s != wn_set {
                    wn_set = s;
                    rebuild = !o.naive; // the pooled set changed, so every stored score is stale
                }
            }
        }
        let wts = weights(&t, &wn_set);
        if rebuild {
            for h in heaps.iter_mut() {
                h.clear();
            }
            let all: Vec<u32> = (0..tr.pairs.len() as u32).collect();
            push_new(&tr, &mut heaps, &all, &wts, step, inf);
        }
        // languages from worst-off to best-off (only used for p = inf); a language
        // that led too many of the last W merges is skipped (moving window)
        let mut order: Vec<usize> = (0..nl).collect();
        order.sort_by(|&x, &y| t[y].partial_cmp(&t[x]).unwrap().then(x.cmp(&y)));
        if let Some(lim) = win_limit {
            order.retain(|&l| (win_count[l] as f64) <= lim);
        }

        // ---- pick the pair ----
        let mut chosen: Option<(usize, usize, f64)> = None; // pair id, lead language, score
        if inf {
            for &l in &order {
                if o.naive {
                    let mut best: Option<(u64, Reverse<u64>, usize)> = None;
                    for q in 0..tr.pairs.len() {
                        let c = tr.score_lang(q, l);
                        if c > 0.0 {
                            let cand = (c.to_bits(), Reverse(pack(tr.pairs[q].0, tr.pairs[q].1)), q);
                            if best.map_or(true, |b| (cand.0, cand.1) > (b.0, b.1)) {
                                best = Some(cand);
                            }
                        }
                    }
                    if let Some((_, _, q)) = best {
                        chosen = Some((q, l, tr.score_lang(q, l)));
                    }
                } else {
                    while let Some((_, key, stamp, q)) = heaps[l].pop() {
                        let c = tr.score_lang(q as usize, l);
                        if stamp == step {
                            chosen = Some((q as usize, l, c));
                            break;
                        }
                        refreshes += 1;
                        if c > 0.0 {
                            heaps[l].push((c.to_bits(), key, step, q));
                        }
                    }
                }
                if chosen.is_some() {
                    break;
                }
            }
        } else {
            let found = if o.naive {
                let mut best: Option<(u64, Reverse<u64>, usize)> = None;
                for q in 0..tr.pairs.len() {
                    let s = tr.score(q, &wts);
                    if s > 0.0 {
                        let cand = (s.to_bits(), Reverse(pack(tr.pairs[q].0, tr.pairs[q].1)), q);
                        if best.map_or(true, |b| (cand.0, cand.1) > (b.0, b.1)) {
                            best = Some(cand);
                        }
                    }
                }
                best.map(|b| b.2)
            } else {
                let mut r = None;
                while let Some((_, key, stamp, q)) = heaps[0].pop() {
                    if stamp == step {
                        r = Some(q as usize);
                        break;
                    }
                    refreshes += 1;
                    let s = tr.score(q as usize, &wts);
                    if s > 0.0 {
                        heaps[0].push((s.to_bits(), key, step, q));
                    }
                }
                r
            };
            if let Some(q) = found {
                // the language that contributes most to this merge's score
                let mut l_best = 0;
                for l in 1..nl {
                    if wts[l] * tr.pcount[q * nl + l] as f64 > wts[l_best] * tr.pcount[q * nl + l_best] as f64 {
                        l_best = l;
                    }
                }
                chosen = Some((q, l_best, tr.score(q, &wts)));
            }
        }
        let Some((q, l_lead, score)) = chosen else {
            eprintln!("  no pairs left after {step} merges; stopping early");
            break;
        };

        // ---- apply it ----
        let new_id = (n_base + model.merges.len()) as u32;
        let (a, b) = tr.pairs[q];
        model.merges.push((a, b));
        let mut tb = tok_bytes[a as usize].clone();
        tb.extend_from_slice(&tok_bytes[b as usize]);
        tok_bytes.push(tb);
        let len_new = tr.tok_len[a as usize] + tr.tok_len[b as usize];
        tr.tok_len.push(len_new);
        lead[l_lead] += 1;
        if let Some((w, _)) = o.window {
            win.push_back(l_lead);
            win_count[l_lead] += 1;
            if win.len() > w {
                let old = win.pop_front().unwrap();
                win_count[old] -= 1;
            }
        }
        let created = tr.apply(q, new_id);
        let t_after = costs(&tr);
        if !o.naive {
            push_new(&tr, &mut heaps, &created, &weights(&t_after, &wn_set), step + 1, is_inf(step + 1));
        }

        write!(log, "{}\t{}\t{}\t{:.6e}", step + 1, String::from_utf8_lossy(&tok_bytes[new_id as usize]), o.langs[l_lead], score).unwrap();
        for x in &t_after {
            write!(log, "\t{x:.6}").unwrap();
        }
        writeln!(log).unwrap();
        if (step + 1) % 4000 == 0 {
            eprintln!("  {:>6} merges  [{:.1}s]", step + 1, t0.elapsed().as_secs_f64());
        }
    }

    // ---- save ----
    let model = Model::new(model.byte_base, model.alphabet, model.merges);
    model.save(&o.out);
    let t = costs(&tr);
    let done = model.merges.len().max(1) as f64;
    eprintln!("  done: vocab {} ({} merges) in {:.1}s; heap re-checks per merge: {:.1}", model.vocab_size(), model.merges.len(), t0.elapsed().as_secs_f64(), refreshes as f64 / done);
    eprintln!("  share of merges led by each language, and its final cost (tokens per {} on the {} text):", o.unit.name(), if has_dev { "dev" } else { "training" });
    for l in 0..nl {
        eprintln!("    {:<6} {:>5.1}%   {:.4}", o.langs[l], 100.0 * lead[l] as f64 / done, t[l]);
    }
    // Size of the training text and its final token count, per language.
    // `eval` reads this to report compression on the training text as well.
    let mut f = fs::File::create(format!("{}/train_stats.tsv", o.out)).unwrap();
    writeln!(f, "lang\tlines\tbytes\tchars\ttokens").unwrap();
    for l in 0..nl {
        writeln!(f, "{}\t{}\t{}\t{}\t{}", o.langs[l], units[l].lines, units[l].bytes, units[l].chars, tr.tokens[l]).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Evaluation: token cost per language, global compression, Gini.
// ---------------------------------------------------------------------------
fn gini(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    if mean == 0.0 {
        return 0.0;
    }
    let mut d = 0.0;
    for a in x {
        for b in x {
            d += (a - b).abs();
        }
    }
    d / (2.0 * n * n * mean)
}

/// Returns (text size, tokens, unknown-character tokens) and fills
/// `per_line` with the token count of every non-empty line.
fn count_tokens(model: &Model, text: &str, per_line: &mut Vec<u32>) -> (Units, u64, u64) {
    let mut u = Units::default();
    let mut cache: FxMap<&str, (u32, u32)> = FxMap::default();
    let mut syms = Vec::new();
    let (mut toks, mut unks) = (0u64, 0u64);
    for line in text.lines() {
        let mut n = 0u64;
        let mut line_toks = 0u32;
        for w in line.split_whitespace() {
            n += 1;
            u.bytes += w.len() as u64;
            u.chars += w.chars().count() as u64;
            let (t, k) = *cache.entry(w).or_insert_with(|| {
                model.encode_word(w, &mut syms);
                (syms.len() as u32, syms.iter().filter(|&&s| s == UNK).count() as u32)
            });
            toks += t as u64;
            unks += k as u64;
            line_toks += t;
        }
        if n > 0 {
            u.lines += 1;
            u.bytes += n - 1;
            u.chars += n - 1;
            per_line.push(line_toks);
        }
    }
    (u, toks, unks)
}

/// Quote a CSV field if it contains a comma or a quote (method names such as "p=8,lambda=0.5").
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn append_csv(path: &str, header: &str, row: &str) {
    let exists = fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false);
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path).unwrap_or_else(|e| die(&format!("cannot write {path}: {e}")));
    if !exists {
        writeln!(f, "{header}").unwrap();
    }
    writeln!(f, "{row}").unwrap();
}

fn eval(args: &Args) {
    let model = Model::load(&args.req("model"));
    let langs = args.langs();
    let dir = args.req("test-dir");
    let name = args.get("name").unwrap_or_else(|| args.req("model"));
    let unit = Unit::parse(&args.get("unit").unwrap_or("byte".into()));
    let name_csv = csv_field(&name);

    let mut per: Vec<(Units, u64, u64)> = Vec::new();
    let mut per_line: Vec<Vec<u32>> = Vec::new(); // [language][sentence] -> tokens
    for l in &langs {
        let mut v = Vec::new();
        per.push(count_tokens(&model, &read_text(&format!("{dir}/{l}.txt")), &mut v));
        per_line.push(v);
    }
    let n0 = per[0].0.lines;
    let parallel = per.iter().all(|x| x.0.lines == n0);
    if unit == Unit::Line && !parallel {
        die("--unit line needs parallel test files (same number of lines in every language)");
    }
    let tot_tok: u64 = per.iter().map(|x| x.1).sum();
    let tot_bytes: u64 = per.iter().map(|x| x.0.bytes).sum();
    let tot_chars: u64 = per.iter().map(|x| x.0.chars).sum();
    let cost = |u: Unit| -> Vec<f64> { per.iter().map(|x| x.1 as f64 / x.0.get(u).max(1.0)).collect() };
    let (g_byte, g_char, g_line) = (gini(&cost(Unit::Byte)), gini(&cost(Unit::Char)), gini(&cost(Unit::Line)));
    let bpt = tot_bytes as f64 / tot_tok as f64;
    let cpt = tot_chars as f64 / tot_tok as f64;
    // compression on the training text itself (what classical BPE greedily maximises):
    // from the trainer's own totals if available, else by encoding --train-dir
    let mut train_bpt: Option<f64> = fs::read_to_string(format!("{}/train_stats.tsv", args.req("model"))).ok().map(|t| {
        let (mut b, mut k) = (0f64, 0f64);
        for line in t.lines().skip(1) {
            let c: Vec<&str> = line.split('\t').collect();
            b += c[2].parse::<f64>().unwrap_or(0.0);
            k += c[4].parse::<f64>().unwrap_or(0.0);
        }
        b / k.max(1.0)
    });
    if train_bpt.is_none() {
        if let Some(td) = args.get("train-dir") {
            let (mut b, mut k) = (0f64, 0f64);
            for l in &langs {
                let mut v = Vec::new();
                let (u, toks, _) = count_tokens(&model, &read_text(&format!("{td}/{l}.txt")), &mut v);
                b += u.bytes as f64;
                k += toks as f64;
            }
            train_bpt = Some(b / k.max(1.0));
        }
    }

    println!("\n== {name}   (vocab {}) ==", model.vocab_size());
    println!("{:<6} {:>7} {:>9} {:>9} {:>6} {:>12} {:>12} {:>12}", "lang", "lines", "bytes", "tokens", "unk", "tok/byte", "tok/char", "tok/line");
    for (l, x) in langs.iter().zip(&per) {
        println!(
            "{:<6} {:>7} {:>9} {:>9} {:>6} {:>12.4} {:>12.4} {:>12.2}",
            l, x.0.lines, x.0.bytes, x.1, x.2, x.1 as f64 / x.0.bytes as f64, x.1 as f64 / x.0.chars as f64, x.1 as f64 / x.0.lines as f64
        );
    }
    println!("global compression: {bpt:.4} bytes/token, {cpt:.4} chars/token");
    if let Some(x) = train_bpt {
        println!("compression of the training text: {x:.4} bytes/token");
    }
    println!("Gini of per-language cost:  per byte {g_byte:.4}   per char {g_char:.4}   per line {g_line:.4}{}", if parallel { "" } else { "  (per line is meaningless: files are not parallel)" });

    let g = match unit {
        Unit::Byte => g_byte,
        Unit::Char => g_char,
        Unit::Line => g_line,
    };
    if let Some(p) = args.get("csv") {
        append_csv(&p, "method,cr,gini", &format!("{name_csv},{bpt:.4},{g:.4}"));
    }
    if let Some(p) = args.get("per-line") {
        if !parallel {
            die("--per-line needs parallel test files");
        }
        let mut f = BufWriter::new(fs::File::create(&p).unwrap_or_else(|e| die(&format!("cannot write {p}: {e}"))));
        writeln!(f, "{}", langs.join("\t")).unwrap();
        for i in 0..n0 as usize {
            let row: Vec<String> = per_line.iter().map(|v| v[i].to_string()).collect();
            writeln!(f, "{}", row.join("\t")).unwrap();
        }
    }
    // same three columns, but with compression measured on the training text
    if let (Some(p), Some(x)) = (args.get("csv-train"), train_bpt) {
        append_csv(&p, "method,cr,gini", &format!("{name_csv},{x:.4},{g:.4}"));
    }
    if let Some(p) = args.get("summary") {
        append_csv(
            &p,
            "method,vocab,tokens,bytes_per_token,chars_per_token,gini_per_byte,gini_per_char,gini_per_line,train_bytes_per_token",
            &format!(
                "{name_csv},{},{tot_tok},{bpt:.4},{cpt:.4},{g_byte:.4},{g_char:.4},{g_line:.4},{}",
                model.vocab_size(),
                train_bpt.map_or(String::new(), |x| format!("{x:.4}"))
            ),
        );
    }
    if let Some(p) = args.get("detail") {
        for (l, x) in langs.iter().zip(&per) {
            append_csv(
                &p,
                "method,lang,lines,bytes,chars,tokens,unk,tokens_per_byte,tokens_per_char,tokens_per_line",
                &format!(
                    "{name_csv},{l},{},{},{},{},{},{:.5},{:.5},{:.3}",
                    x.0.lines, x.0.bytes, x.0.chars, x.1, x.2, x.1 as f64 / x.0.bytes as f64, x.1 as f64 / x.0.chars as f64, x.1 as f64 / x.0.lines as f64
                ),
            );
        }
    }
}

/// Print the tokens of each input line, separated by spaces (for looking at
/// the segmentation, and later for the morphology boundary score).
fn encode(args: &Args) {
    let model = Model::load(&args.req("model"));
    let text = read_text(&args.req("input"));
    let tb = model.token_bytes();
    let out = std::io::stdout();
    let mut out = BufWriter::new(out.lock());
    let mut syms = Vec::new();
    for line in text.lines() {
        let mut first = true;
        for w in line.split_whitespace() {
            model.encode_word(w, &mut syms);
            for &s in &syms {
                if !first {
                    out.write_all(b" ").unwrap();
                }
                first = false;
                out.write_all(String::from_utf8_lossy(&tb[s as usize]).as_bytes()).unwrap();
            }
        }
        out.write_all(b"\n").unwrap();
    }
}

/// Print the token ids of each input line, space separated (input for the
/// language-model experiments).
fn encode_ids(args: &Args) {
    let model = Model::load(&args.req("model"));
    let text = read_text(&args.req("input"));
    let out: Box<dyn Write> = match args.get("output") {
        Some(p) => Box::new(fs::File::create(&p).unwrap_or_else(|e| die(&format!("cannot write {p}: {e}")))),
        None => Box::new(std::io::stdout()),
    };
    let mut out = BufWriter::new(out);
    let mut cache: FxMap<&str, Vec<u32>> = FxMap::default();
    let mut syms = Vec::new();
    for line in text.lines() {
        let mut first = true;
        for w in line.split_whitespace() {
            let ids = cache.entry(w).or_insert_with(|| {
                model.encode_word(w, &mut syms);
                syms.clone()
            });
            for &s in ids.iter() {
                if !first {
                    out.write_all(b" ").unwrap();
                }
                first = false;
                write!(out, "{s}").unwrap();
            }
        }
        out.write_all(b"\n").unwrap();
    }
}

/// Morphological alignment of a tokenizer against gold segmentations:
/// boundary precision / recall / F1 (micro over all boundaries) and a
/// MorphScore-style recall over words that the tokenizer splits at all.
fn morph_eval(args: &Args) {
    let model = Model::load(&args.req("model"));
    let gold_path = args.req("gold");
    let name = args.get("name").unwrap_or_else(|| args.req("model"));
    let lang = args.get("lang").unwrap_or_default();
    let (entries, skipped) = read_segmentations(&gold_path, model.byte_base);
    let lens = model.token_lens();
    let mut syms = Vec::new();
    let (mut tp, mut n_pred, mut n_gold) = (0u64, 0u64, 0u64);
    let (mut ms_hits, mut ms_total, mut split_words) = (0u64, 0u64, 0u64);
    let mut tokens = 0u64;
    for (word, gold) in &entries {
        model.encode_word(word, &mut syms);
        tokens += syms.len() as u64;
        let mut pred: Vec<u32> = Vec::new();
        let mut pos = 0u32;
        for &s in &syms[..syms.len() - 1] {
            pos += lens[s as usize];
            if pos > 0 {
                pred.push(pos - 1);
            }
        }
        pred.dedup();
        let hits = gold.iter().filter(|g| pred.binary_search(g).is_ok()).count() as u64;
        tp += hits;
        n_pred += pred.len() as u64;
        n_gold += gold.len() as u64;
        if !pred.is_empty() {
            split_words += 1;
            ms_hits += hits;
            ms_total += gold.len() as u64;
        }
    }
    let prec = tp as f64 / n_pred.max(1) as f64;
    let rec = tp as f64 / n_gold.max(1) as f64;
    let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };
    let morphscore = ms_hits as f64 / ms_total.max(1) as f64;
    let tpw = tokens as f64 / entries.len().max(1) as f64;
    println!("{name} {lang}: {} words ({skipped} skipped), boundary P {prec:.4} R {rec:.4} F1 {f1:.4}, MorphScore {morphscore:.4} ({split_words} split words), tokens/word {tpw:.3}", entries.len());
    if let Some(p) = args.get("csv") {
        append_csv(&p, "method,lang,words,skipped,precision,recall,f1,morphscore,tokens_per_word", &format!("{},{lang},{},{skipped},{prec:.4},{rec:.4},{f1:.4},{morphscore:.4},{tpw:.4}", csv_field(&name), entries.len()));
    }
}

/// Merge several models (for example one per script, each with its own budget)
/// into one. Merges are interleaved by their relative rank within each model, and
/// a token that two models both create is kept once.
fn merge_models(args: &Args) {
    let paths: Vec<String> = args.req("models").split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if paths.len() < 2 {
        die("--models needs at least two model directories, comma separated");
    }
    let models: Vec<Model> = paths.iter().map(|p| Model::load(p)).collect();
    let byte_base = models[0].byte_base;
    if models.iter().any(|m| m.byte_base != byte_base) {
        die("cannot merge byte-level and character-level models");
    }
    let mut alphabet: Vec<char> = Vec::new();
    let mut seen: FxMap<char, ()> = FxMap::default();
    for m in &models {
        for &c in &m.alphabet {
            if seen.insert(c, ()).is_none() {
                alphabet.push(c);
            }
        }
    }
    let base = Model::new(byte_base, alphabet.clone(), Vec::new());
    let mut ids: FxMap<Vec<u8>, u32> = FxMap::default();
    for (i, t) in base.token_bytes().into_iter().enumerate() {
        ids.insert(t, i as u32);
    }
    // (relative rank, model index, rank, token a, token b)
    let mut all: Vec<(f64, usize, usize, Vec<u8>, Vec<u8>)> = Vec::new();
    for (i, m) in models.iter().enumerate() {
        let tb = m.token_bytes();
        let n = m.merges.len().max(1) as f64;
        for (r, &(a, b)) in m.merges.iter().enumerate() {
            all.push((r as f64 / n, i, r, tb[a as usize].clone(), tb[b as usize].clone()));
        }
    }
    all.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut merges: Vec<(u32, u32)> = Vec::new();
    let n_base = base.n_base();
    let mut dup = 0usize;
    for (_, _, _, ta, tb) in all {
        let (Some(&ia), Some(&ib)) = (ids.get(&ta), ids.get(&tb)) else { die("internal error: merge refers to an unknown token") };
        let mut t = ta.clone();
        t.extend_from_slice(&tb);
        if ids.contains_key(&t) {
            dup += 1;
            continue;
        }
        ids.insert(t, n_base + merges.len() as u32);
        merges.push((ia, ib));
    }
    let out = args.req("out");
    let merged = Model::new(byte_base, alphabet, merges);
    merged.save(&out);
    eprintln!("merged {} models into {out}: vocab {} ({} merges, {dup} duplicates dropped)", models.len(), merged.vocab_size(), merged.merges.len());
}

// ---------------------------------------------------------------------------
// Command line.
// ---------------------------------------------------------------------------
struct Args {
    kv: HashMap<String, String>,
    flags: Vec<String>,
}
impl Args {
    fn parse(v: &[String]) -> Args {
        let mut a = Args { kv: HashMap::new(), flags: Vec::new() };
        let mut i = 0;
        while i < v.len() {
            let k = v[i].strip_prefix("--").unwrap_or_else(|| die(&format!("unexpected argument '{}'", v[i])));
            if k == "naive" || k == "pi-uniform" {
                a.flags.push(k.to_string());
                i += 1;
            } else {
                let val = v.get(i + 1).unwrap_or_else(|| die(&format!("--{k} needs a value")));
                a.kv.insert(k.to_string(), val.clone());
                i += 2;
            }
        }
        a
    }
    fn get(&self, k: &str) -> Option<String> {
        self.kv.get(k).cloned()
    }
    fn req(&self, k: &str) -> String {
        self.get(k).unwrap_or_else(|| die(&format!("missing --{k}")))
    }
    fn langs(&self) -> Vec<String> {
        self.req("langs").split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    }
}

const HELP: &str = "\
pfair: classical, Parity-aware and p-fair BPE

  pfair train  --langs hin,mar,guj,eng --train-dir DIR --out DIR --p P [options]
      --p P              1 = classical BPE, inf = Parity-aware BPE, anything between = p-fair (e.g. 2, 4, 8)
      --vocab-size N     total vocabulary size (default 32000)
      --dev-dir DIR      measure each language's cost on this text instead of the training text
                         (DIR/<lang>.txt; use a parallel set such as FLORES dev)
      --unit U           cost unit: byte (default), char, or line (line needs a parallel --dev-dir)
      --base B           char (default) or byte
      --min-char-freq N  rarer characters become <unk> (default 5; char base only)
      --hybrid F         with --p inf: learn the first fraction F of the merges with the classical
                         rule, the rest with the Parity-aware rule (the paper's hybrid baseline)
      --window W --alpha A   with --p inf: moving-window Parity-aware (a language that led more than
                         A*W/L of the last W merges is skipped); the paper uses W=100, A=2
      --worst-n N [--worst-n-refresh R]   Worst-N parity: pool the counts of the languages whose cost
                         is at least the mean cost of the N worst-off ones; set recomputed every R merges (default 100)
      --lambda L --morph-lexicon F1[,F2..]   morphology penalty, 0 <= L <= 1: an occurrence of a pair whose
                         junction is a morpheme boundary (per the lexicon) counts (1-L) instead of 1
      --pi-uniform       weight every language equally instead of by its share of training text
      --naive            brute-force pair selection (slow; for checking the fast path)

  pfair eval   --langs hin,mar,guj,eng --model DIR --test-dir DIR [--name NAME] [--unit U]
               [--csv FILE] [--csv-train FILE] [--summary FILE] [--detail FILE] [--per-line FILE]
      --csv        method,cr,gini with cr = bytes per token on the test text
      --csv-train  the same, with cr = bytes per token on the training text
      --train-dir  training text to measure that compression on, for models without train_stats.tsv
      --per-line   token count of every test sentence in every language (for error bars)

  pfair encode     --model DIR --input FILE                 print the tokens of each line
  pfair encode-ids --model DIR --input FILE [--output FILE]  print the token ids of each line
  pfair morph-eval --model DIR --gold FILE [--lang L] [--name N] [--csv FILE]
                   boundary precision/recall/F1 and MorphScore against gold segmentations
                   (lines 'seg1+seg2' or 'word<TAB>seg1 seg2')
  pfair merge      --models A,B,C --out DIR                 merge per-script models into one

Text files are DIR/<lang>.txt, one sentence per line.";

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 || argv[1] == "help" || argv[1] == "--help" || argv[1] == "-h" {
        println!("{HELP}");
        return;
    }
    let args = Args::parse(&argv[2..]);
    match argv[1].as_str() {
        "train" => {
            let p_str = args.req("p");
            let p = match p_str.as_str() {
                "inf" | "infinity" | "Inf" => f64::INFINITY,
                s => s.parse::<f64>().unwrap_or_else(|_| die("--p must be a number >= 1, or inf")),
            };
            if !(p >= 1.0) {
                die("--p must be >= 1 (the objective is only convex, and the fast path only exact, for p >= 1)");
            }
            let o = TrainOpts {
                langs: args.langs(),
                train_dir: args.req("train-dir"),
                dev_dir: args.get("dev-dir"),
                out: args.req("out"),
                vocab_size: args.get("vocab-size").map_or(32000, |s| s.parse().unwrap_or_else(|_| die("bad --vocab-size"))),
                p,
                unit: Unit::parse(&args.get("unit").unwrap_or("byte".into())),
                byte_base: match args.get("base").as_deref() {
                    None | Some("char") => false,
                    Some("byte") => true,
                    _ => die("--base must be char or byte"),
                },
                min_char_freq: args.get("min-char-freq").map_or(5, |s| s.parse().unwrap_or_else(|_| die("bad --min-char-freq"))),
                naive: args.flags.iter().any(|f| f == "naive"),
                hybrid: args.get("hybrid").map(|s| s.parse::<f64>().unwrap_or_else(|_| die("--hybrid must be a number between 0 and 1"))),
                lambda: args.get("lambda").map_or(0.0, |s| s.parse::<f64>().unwrap_or_else(|_| die("--lambda must be a number >= 0"))),
                morph_lexicon: args.get("morph-lexicon").map_or(Vec::new(), |s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()),
                worst_n: args.get("worst-n").map(|s| {
                    let n = s.parse::<usize>().unwrap_or_else(|_| die("--worst-n must be a positive integer"));
                    let r = args.get("worst-n-refresh").map_or(100, |x| x.parse::<u32>().unwrap_or_else(|_| die("bad --worst-n-refresh")));
                    (n.max(1), r.max(1))
                }),
                window: args.get("window").map(|s| {
                    let w = s.parse::<usize>().unwrap_or_else(|_| die("--window must be a positive integer"));
                    let a = args.get("alpha").map_or(2.0, |x| x.parse::<f64>().unwrap_or_else(|_| die("bad --alpha")));
                    (w.max(1), a)
                }),
                pi_uniform: args.flags.iter().any(|f| f == "pi-uniform"),
            };
            if o.lambda < 0.0 {
                die("--lambda must be >= 0");
            }
            if o.lambda > 1.0 && !o.naive {
                die("--lambda above 1 makes pair scores non-monotone, so the lazy heap is not exact; use --naive with it, or lambda in [0, 1]");
            }
            if o.window.is_some() && !p.is_infinite() {
                die("--window goes with --p inf");
            }
            if o.worst_n.is_some() && (o.hybrid.is_some() || o.window.is_some()) {
                die("--worst-n cannot be combined with --hybrid or --window");
            }
            if let Some(f) = o.hybrid {
                if !(0.0..=1.0).contains(&f) {
                    die("--hybrid must be between 0 and 1");
                }
                if !p.is_infinite() {
                    die("--hybrid goes with --p inf (classical merges first, then Parity-aware merges)");
                }
            }
            eprintln!(
                "train: p={p_str}{}{}{}{}{}  vocab={}  unit={}  cost measured on {}  base={}{}",
                o.hybrid.map_or(String::new(), |f| format!("  hybrid (first {:.0}% of merges classical)", 100.0 * f)),
                o.window.map_or(String::new(), |(w, a)| format!("  window W={w} alpha={a}")),
                o.worst_n.map_or(String::new(), |(n, r)| format!("  worst-N N={n} (refresh every {r})")),
                if o.lambda > 0.0 { format!("  lambda={}", o.lambda) } else { String::new() },
                if o.pi_uniform { "  pi=uniform" } else { "" },
                o.vocab_size,
                o.unit.name(),
                if o.dev_dir.is_some() { "dev text" } else { "training text" },
                if o.byte_base { "byte" } else { "char" },
                if o.naive { "  [naive]" } else { "" }
            );
            train(&o);
        }
        "eval" => eval(&args),
        "encode" => encode(&args),
        "encode-ids" => encode_ids(&args),
        "morph-eval" => morph_eval(&args),
        "merge" => merge_models(&args),
        other => die(&format!("unknown command '{other}' (try: pfair help)")),
    }
}
