#!/usr/bin/env python3
"""Compare out-of-sample results across every walk-forward run of two or more runs.

Usage:
    python tools/plot_distribution.py output/random_walk output/trending \\
        --labels "random walk" "planted trend" --out docs/images/oos_distribution.png

Reads summary.csv from each output directory and draws the distribution of
out-of-sample annual return on capital over all schedule x selection-rule
runs with at least --min-windows traded windows. The whole distribution, not the best row, is
the honest summary of a walk-forward study. Requires matplotlib.
"""

import argparse
import csv
import os
import statistics
import sys


def load(path, min_windows):
    rows = []
    with open(os.path.join(path, "summary.csv"), newline="") as f:
        for r in csv.DictReader(f):
            # The same runs the CLI summarizes: sufficient, with enough windows.
            if int(r["viable"]) and int(r["total_windows"]) >= min_windows:
                rows.append(float(r["oos_annual_roc"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", nargs="+", help="run output directories")
    parser.add_argument("--labels", nargs="+", help="legend labels, one per directory")
    parser.add_argument("--min-windows", type=int, default=4)
    parser.add_argument("--out", default="oos_distribution.png")
    args = parser.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(p)) for p in args.outputs]
    if len(labels) != len(args.outputs):
        sys.exit("give one label per output directory")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    series = [(label, load(p, args.min_windows)) for label, p in zip(labels, args.outputs)]
    lo = min(min(v) for _, v in series if v)
    hi = max(max(v) for _, v in series if v)
    bins = [lo + (hi - lo) * i / 40 for i in range(41)]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for label, values in series:
        if not values:
            continue
        med = statistics.median(values)
        share = sum(v > 0 for v in values) / len(values)
        ax.hist(values, bins=bins, alpha=0.55,
                label=f"{label}: median {med:,.1f}%, {share:.0%} of {len(values)} runs positive")
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("out-of-sample annual return on capital, % (synthetic data)")
    ax.set_ylabel("runs")
    ax.set_title("Out-of-sample results across every schedule and selection rule")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
