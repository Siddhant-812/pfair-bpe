#!/usr/bin/env python3
"""Is p-fair BPE better than the hybrid baseline?

Both methods sit between classical BPE and Parity-aware BPE, so the fair way
to compare them is at the SAME compression: for each hybrid tokenizer, find the
point on the p-fair curve with the same training-text compression and compare
the Gini there. Lower Gini at the same compression = better.

    python run_all.py --ps 1,2,4,8,16,32,inf --hybrids 0.5,0.75,0.9 --out results_hybrid
    python compare_hybrid.py results_hybrid
"""
import csv
import os
import sys


def read_points(folder):
    """method -> (training-text compression, Gini) from results_train_cr.csv."""
    with open(os.path.join(folder, "results_train_cr.csv"), encoding="utf-8") as f:
        return {r["method"]: (float(r["cr"]), float(r["gini"])) for r in csv.DictReader(f)}


def is_pfair(method):
    """Rows that form the p-fair curve: Classical, p=..., Parity-aware (without a morphology penalty)."""
    return (method == "Classical" or method == "Parity-aware" or method.startswith("p=")) and "lambda" not in method


def is_hybrid(method):
    """Any baseline that is compared against the p-fair curve at equal compression
    (Hybrid-*, WorstN-*, Window-*, ...)."""
    return not is_pfair(method)


def pfair_gini_at(points, cr):
    """Gini of the p-fair curve at compression `cr` (straight line between the
    two neighbouring p values). None if `cr` is outside the p-fair range."""
    curve = sorted(v for m, v in points.items() if is_pfair(m))
    for (c0, g0), (c1, g1) in zip(curve, curve[1:]):
        if c0 <= cr <= c1:
            return g0 if c1 == c0 else g0 + (g1 - g0) * (cr - c0) / (c1 - c0)
    return None


def compare(points):
    """[(hybrid name, cr, hybrid Gini, p-fair Gini at the same cr or None)]"""
    return [(m, cr, g, pfair_gini_at(points, cr)) for m, (cr, g) in points.items() if is_hybrid(m)]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rows = compare(read_points(sys.argv[1]))
    if not rows:
        sys.exit("No hybrid rows found. Run run_all.py with --hybrids first.")
    print(f"{'hybrid':<11} {'compression':>11} {'hybrid Gini':>12} {'p-fair Gini':>12}   better at this compression")
    for m, cr, g, pf in rows:
        if pf is None:
            print(f"{m:<11} {cr:>11.4f} {g:>12.4f} {'-':>12}   (outside the p-fair range)")
        else:
            who = "p-fair" if pf < g else "hybrid" if g < pf else "tie"
            print(f"{m:<11} {cr:>11.4f} {g:>12.4f} {pf:>12.4f}   {who}  (difference {abs(g - pf):.4f})")
    print("\nThis is one run. Use error_bars.py to see whether the differences are larger than the noise.")


if __name__ == "__main__":
    main()
