#!/usr/bin/env python3
"""Rebuild the committed test fixtures.

  1. tests/data/synthetic_30s.bin   via `wfo-gpu generate` (the parameters
     must match fixture_config() in tests/test_fixture.h)
  2. tests/reference/synthetic_indicators_reference.csv via the independent
     NumPy implementation in tools/gen_reference_indicators.py

Usage: python tools/make_fixtures.py --cli build/wfo-gpu
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", required=True, help="path to the wfo-gpu executable")
    args = parser.parse_args()

    fixture = os.path.join(ROOT, "tests", "data", "synthetic_30s.bin")
    subprocess.run(
        [args.cli, "generate", "--out", fixture, "--days", "3", "--seed", "20240108",
         "--start", "2024-01-08", "--symbol", "SYN", "--timeframe", "30"],
        check=True)
    subprocess.run(
        [sys.executable, "-I", os.path.join(ROOT, "tools", "gen_reference_indicators.py")],
        check=True)


if __name__ == "__main__":
    main()
