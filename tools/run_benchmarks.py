#!/usr/bin/env python3
"""Run wfo-gpu-bench and print Markdown rows for docs/BENCHMARKS.md.

Usage: python tools/run_benchmarks.py --bench build/wfo-gpu-bench [--repeat 5]

Build in Release first:
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", required=True, help="path to the wfo-gpu-bench executable")
    parser.add_argument("--days", type=int, default=30, help="synthetic sessions (default 30)")
    parser.add_argument("--strategies", type=int, default=20000,
                        help="distinct strategies per pass (default 20000)")
    parser.add_argument("--repeat", type=int, default=5, help="timed passes per device")
    args = parser.parse_args()

    cmd = [args.bench, "--days", str(args.days), "--strategies", str(args.strategies),
           "--repeat", str(args.repeat)]
    # Human-readable run first, so the console shows what was measured.
    subprocess.run(cmd, check=True)
    rows = subprocess.run(cmd + ["--markdown"], check=True, capture_output=True, text=True).stdout

    print("\nRows for docs/BENCHMARKS.md:\n")
    print("| Workload | Device | Hardware | OS, compiler | Strategies | Bars | Backtests/s "
          "| Bar-evals/s |")
    print("|---|---|---|---|---|---|---|---|")
    sys.stdout.write(rows)


if __name__ == "__main__":
    main()
