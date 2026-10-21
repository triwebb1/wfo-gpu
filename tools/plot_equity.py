#!/usr/bin/env python3
"""Plot stitched out-of-sample equity curves from a walk-forward run.

Usage:
    python tools/plot_equity.py output/example              # top 5 by final PnL
    python tools/plot_equity.py output/example --top 10
    python tools/plot_equity.py output/example --match freq7d_lb91d

Reads <output>/equity/*.csv (timestamp_ns,cumulative_pnl) and writes
<output>/equity.png. Requires numpy and matplotlib.
"""

import argparse
import glob
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="the run's output directory ([data] output)")
    parser.add_argument("--top", type=int, default=5, help="curves to draw (default 5)")
    parser.add_argument("--match", default="", help="only files whose name contains this")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    # Rank by the final value (last line of each file), then load only the
    # curves that will be drawn: files can hold one row per bar.
    def final_pnl(path):
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 256))
            last = f.read().splitlines()[-1].decode()
        try:
            return float(last.split(",")[1])
        except (IndexError, ValueError):
            return None

    ranked = []
    for path in sorted(glob.glob(os.path.join(args.output, "equity", "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if args.match and args.match not in name:
            continue
        final = final_pnl(path)
        if final is not None:
            ranked.append((final, name, path))
    if not ranked:
        sys.exit(f"no equity curves found under {args.output}/equity")
    ranked.sort(key=lambda c: c[0], reverse=True)

    curves = []
    for final, name, path in ranked[: args.top]:
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        curves.append((final, name, data[:: max(1, len(data) // 4000)]))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for final, name, data in curves:
        days = (data[:, 0] - data[0, 0]) / 86400e9
        ax.plot(days, data[:, 1], linewidth=1.2, label=f"{name} ({final:,.0f})")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("days since first out-of-sample bar")
    ax.set_ylabel("cumulative net PnL")
    ax.set_title(f"Stitched out-of-sample equity, top {len(curves)} "
                 f"of {len(ranked)} runs (ranked on OOS data)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = os.path.join(args.output, "equity.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
