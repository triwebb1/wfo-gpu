#!/usr/bin/env bash
# End-to-end smoke test of the command-line pipeline on a small synthetic file:
# run, resume a finished run, reanalyze, and detail. Usage:
#
#   tools/pipeline_smoke.sh path/to/wfo-gpu [work-dir]
#
# Floors are relaxed so the screen keeps strategies and the walk-forward
# stage has something to do. Exits non-zero on the first failed check.
set -euo pipefail

exe=$1
work=${2:-$(mktemp -d)}
mkdir -p "$work"
cd "$work"

fail() { echo "FAIL: $*" >&2; exit 1; }

"$exe" --version
"$exe" generate --out bars.bin --days 45 --seed 11 --trend 0.03 > /dev/null

cat > smoke.toml <<'EOF'
[data]
bars = "bars.bin"
output = "out"

[screen]
end = 40000
max_strategies = 3000
robustness_floor = 0.0

[floors]
min_trades = 20
min_profit_factor = 0.0
min_annual_roc = -1000.0
max_negative_years = 10

[compute]
device = "native"
EOF

"$exe" run --config smoke.toml > run.log 2>&1 || { cat run.log; fail "run"; }
[ -s out/reduced_params.bin ] || fail "no reduced set written"
[ -s out/summary.csv ] || fail "no walk-forward summary written"
rows=$(wc -l < out/summary.csv)
[ "$rows" -gt 1 ] || fail "summary.csv has no runs"

# Resuming a finished run repeats nothing and rewrites the same outputs.
"$exe" run --config smoke.toml --resume > resume.log 2>&1 || { cat resume.log; fail "resume"; }
grep -q "Resumed the completed screen" resume.log || fail "screen was not resumed"
grep -q "Walk-forward already complete" resume.log || fail "walk-forward was re-run"
[ "$(wc -l < out/summary.csv)" -eq "$rows" ] || fail "resume changed the number of runs"

# --fresh discards the checkpoints and starts over; it must reach the same result.
"$exe" run --config smoke.toml --fresh > fresh.log 2>&1 || { cat fresh.log; fail "fresh"; }
[ "$(wc -l < out/summary.csv)" -eq "$rows" ] || fail "fresh run changed the number of runs"

"$exe" reanalyze --config smoke.toml --robustness-floor 0.5 > reanalyze.log 2>&1 ||
    { cat reanalyze.log; fail "reanalyze"; }

printf '0\n1\n# comment\n9216\n' > ids.txt
"$exe" detail --config smoke.toml --ids ids.txt > detail.log 2>&1 || { cat detail.log; fail "detail"; }
[ "$(wc -l < out/detail/detail_summary.csv)" -eq 4 ] || fail "detail summary should list 3 strategies"
[ -s out/detail/trades/strategy_9216.csv ] || fail "no trade log for strategy 9216"

# Ids outside the grid are rejected.
echo "-5" > bad_ids.txt
if "$exe" detail --config smoke.toml --ids bad_ids.txt > /dev/null 2>&1; then
    fail "detail accepted an id outside the grid"
fi

echo "pipeline smoke test passed ($((rows - 1)) walk-forward runs)"
