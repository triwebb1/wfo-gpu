#!/usr/bin/env python3
"""
gen_reference_indicators.py - Generate reference indicator values for cross-validation.

Loads the synthetic test fixture (tests/data/synthetic_30s.bin),
computes indicators with an independent NumPy implementation of the same
formulas the C++ code uses, and writes reference values to CSV. The C++
test suite (test_indicators_cross_validate.cpp) compares against it.

Two implementations written separately that agree to within float32
rounding is strong evidence that both are correct.

Usage: python tools/gen_reference_indicators.py

Indicators computed:
  - EMA-20 (seed=first close, alpha=2/(20+1))
  - RSI-14 (Wilder smoothing)
  - ATR-14 (Wilder smoothing)
  - SMA-50 (running window)
  - MACD(12,26,9) (EMA-based)
  - Bollinger(20,2) (population stddev)
"""

import csv
import math
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Binary bar format (see docs/DATA_FORMAT.md and src/market_data.h)
# ---------------------------------------------------------------------------
HEADER_SIZE = 64
BAR_SIZE = 32
HEADER_FMT = "<4sII8sIqq24s"
BAR_FMT = "<qffffII"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_DIR, "tests", "data", "synthetic_30s.bin")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "tests", "reference", "synthetic_indicators_reference.csv")


def load_binary(path):
    """Load bars from WFBT binary file. Returns (header_dict, bars_array)."""
    with open(path, "rb") as f:
        hdr_data = f.read(HEADER_SIZE)
        magic, version, bar_count, symbol, tf_sec, start_ts, end_ts, reserved = \
            struct.unpack(HEADER_FMT, hdr_data)

        assert magic == b"WFBT", f"Bad magic: {magic}"
        assert version == 1, f"Bad version: {version}"

        print(f"Loaded: {bar_count} bars, symbol={symbol.rstrip(b'\\x00').decode()}, "
              f"tf={tf_sec}s")

        # Read all bars
        bar_dtype = np.dtype([
            ("timestamp", "<i8"),
            ("open", "<f4"),
            ("high", "<f4"),
            ("low", "<f4"),
            ("close", "<f4"),
            ("volume", "<u4"),
            ("_pad", "<u4"),
        ])
        bars = np.fromfile(f, dtype=bar_dtype, count=bar_count)

    assert len(bars) == bar_count
    return {
        "bar_count": bar_count,
        "symbol": symbol.rstrip(b"\x00").decode(),
        "timeframe_sec": tf_sec,
    }, bars


def compute_ema(closes, period):
    """Compute EMA matching C++ EMASMACross::compute_ma for MAType::EMA.

    Seed: ema[0] = close[0]
    Update: ema[i] = alpha * close[i] + (1-alpha) * ema[i-1]
    alpha = 2.0 / (period + 1)
    Warmup: 3 * period (output NaN before that)

    Uses float32 arithmetic to match C++ precision exactly.
    """
    n = len(closes)
    alpha = np.float32(2.0 / (period + 1.0))
    one_minus_alpha = np.float32(1.0) - alpha
    warmup = 3 * period

    result = np.full(n, np.nan, dtype=np.float64)
    ema = np.float32(closes[0])

    for i in range(n):
        c = np.float32(closes[i])
        if i == 0:
            ema = c
        else:
            ema = alpha * c + one_minus_alpha * ema

        if i >= warmup:
            result[i] = float(ema)

    return result


def compute_sma(closes, period):
    """Compute SMA matching C++ EMASMACross::compute_ma for MAType::SMA.

    Running sum, valid from index period-1 onward.
    Warmup = period - 1.

    Uses float32 running sum to match C++ precision exactly.
    """
    n = len(closes)
    warmup = period - 1
    result = np.full(n, np.nan, dtype=np.float64)
    running_sum = np.float32(0.0)

    for i in range(n):
        c = np.float32(closes[i])
        running_sum += c

        if i >= period:
            running_sum -= np.float32(closes[i - period])

        if i >= period - 1 and i >= warmup:
            result[i] = float(running_sum / np.float32(period))

    return result


def compute_rsi(closes, period):
    """Compute RSI matching C++ RSI::compute.

    Wilder smoothing: avg = (avg * (period-1) + value) / period
    Initial avg_gain/avg_loss = simple mean of first 'period' changes.
    Warmup: 3 * period.

    Uses float32 arithmetic to match C++ precision exactly.
    """
    n = len(closes)
    warmup = 3 * period
    result = np.full(n, np.nan, dtype=np.float64)

    period_f = np.float32(period)
    period_minus_1 = np.float32(period - 1)

    avg_gain = np.float32(0.0)
    avg_loss = np.float32(0.0)
    initialized = False
    changes_counted = 0

    for i in range(n):
        if i == 0:
            continue

        diff = np.float32(closes[i]) - np.float32(closes[i - 1])
        gain = np.float32(max(np.float32(0.0), diff))
        loss = np.float32(max(np.float32(0.0), -diff))

        if not initialized:
            avg_gain += gain
            avg_loss += loss
            changes_counted += 1

            if changes_counted == period:
                avg_gain /= period_f
                avg_loss /= period_f
                initialized = True

                if i >= warmup:
                    if avg_gain == 0.0 and avg_loss == 0.0:
                        result[i] = 50.0
                    elif avg_loss == 0.0:
                        result[i] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        result[i] = float(np.float32(100.0) - np.float32(100.0) / (np.float32(1.0) + rs))
        else:
            avg_gain = (avg_gain * period_minus_1 + gain) / period_f
            avg_loss = (avg_loss * period_minus_1 + loss) / period_f

            if i >= warmup:
                if avg_gain == 0.0 and avg_loss == 0.0:
                    result[i] = 50.0
                elif avg_loss == 0.0:
                    result[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    result[i] = float(np.float32(100.0) - np.float32(100.0) / (np.float32(1.0) + rs))

    return result


def compute_atr(highs, lows, closes, period):
    """Compute ATR matching C++ ATR::compute.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    First bar: TR = high - low
    Initial ATR = simple mean of first 'period' TRs.
    Then Wilder smoothing.
    Warmup: 3 * period.

    Uses float32 arithmetic to match C++ precision exactly.
    """
    n = len(closes)
    warmup = 3 * period
    result = np.full(n, np.nan, dtype=np.float64)

    period_f = np.float32(period)
    period_minus_1 = np.float32(period - 1)

    atr_val = np.float32(0.0)
    tr_sum = np.float32(0.0)
    initialized = False

    for i in range(n):
        if i == 0:
            tr = np.float32(highs[i]) - np.float32(lows[i])
        else:
            prev_close = np.float32(closes[i - 1])
            hl = np.float32(highs[i]) - np.float32(lows[i])
            hpc = np.abs(np.float32(highs[i]) - prev_close)
            lpc = np.abs(np.float32(lows[i]) - prev_close)
            tr = max(hl, hpc, lpc)

        if not initialized:
            tr_sum += np.float32(tr)

            if i == period - 1:
                atr_val = tr_sum / period_f
                initialized = True
                if i >= warmup:
                    result[i] = float(atr_val)
        else:
            atr_val = (atr_val * period_minus_1 + np.float32(tr)) / period_f
            if i >= warmup:
                result[i] = float(atr_val)

    return result


def compute_macd(closes, fast_period, slow_period, signal_period):
    """Compute MACD matching C++ MACD::compute.

    Fast/slow EMAs seeded at first close.
    Signal EMA seeded at first MACD value.
    Warmup: 3 * slow_period + 3 * signal_period.
    Returns (macd_line, signal_line, histogram).

    Uses float32 arithmetic to match C++ precision exactly.
    """
    n = len(closes)
    warmup = 3 * slow_period + 3 * signal_period

    fast_alpha = np.float32(2.0 / (fast_period + 1.0))
    slow_alpha = np.float32(2.0 / (slow_period + 1.0))
    sig_alpha = np.float32(2.0 / (signal_period + 1.0))

    macd_line = np.full(n, np.nan, dtype=np.float64)
    signal_line = np.full(n, np.nan, dtype=np.float64)
    histogram = np.full(n, np.nan, dtype=np.float64)

    fast_ema = np.float32(0.0)
    slow_ema = np.float32(0.0)
    signal_ema = np.float32(0.0)
    signal_initialized = False

    for i in range(n):
        c = np.float32(closes[i])

        if i == 0:
            fast_ema = c
            slow_ema = c
        else:
            fast_ema = fast_alpha * c + (np.float32(1.0) - fast_alpha) * fast_ema
            slow_ema = slow_alpha * c + (np.float32(1.0) - slow_alpha) * slow_ema

        macd_val = fast_ema - slow_ema

        if not signal_initialized:
            signal_ema = macd_val
            signal_initialized = True
        else:
            signal_ema = sig_alpha * macd_val + (np.float32(1.0) - sig_alpha) * signal_ema

        hist_val = macd_val - signal_ema

        if i >= warmup:
            macd_line[i] = float(macd_val)
            signal_line[i] = float(signal_ema)
            histogram[i] = float(hist_val)

    return macd_line, signal_line, histogram


def compute_bollinger(closes, period, multiplier):
    """Compute Bollinger Bands matching C++ BollingerBands::compute.

    Uses running sum and running sum-of-squares for O(1) per bar.
    Population stddev: sqrt(E[X^2] - (E[X])^2)
    Warmup: period - 1.
    Returns (middle, upper, lower).

    C++ uses double for running_sum/running_sum_sq but float32 for close
    values and final output. We match this exactly.
    """
    n = len(closes)
    warmup = period - 1
    mult_f32 = np.float32(multiplier)

    middle = np.full(n, np.nan, dtype=np.float64)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)

    # C++ uses double for running sums (to avoid catastrophic cancellation)
    running_sum = 0.0
    running_sum_sq = 0.0

    for i in range(n):
        # C++ casts bar close (float32) to double
        c = float(np.float32(closes[i]))
        running_sum += c
        running_sum_sq += c * c

        if i >= period:
            old_c = float(np.float32(closes[i - period]))
            running_sum -= old_c
            running_sum_sq -= old_c * old_c

        if i >= period - 1:
            period_d = float(period)
            sma_d = running_sum / period_d
            sma = np.float32(sma_d)

            variance = running_sum_sq / period_d - sma_d * sma_d
            if variance < 0.0:
                variance = 0.0
            stddev = np.float32(math.sqrt(variance))

            if i >= warmup:
                middle[i] = float(sma)
                upper[i] = float(sma + mult_f32 * stddev)
                lower[i] = float(sma - mult_f32 * stddev)

    return middle, upper, lower


def main():
    print("=" * 70)
    print("  Generate Reference Indicator Values")
    print("=" * 70)

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}", file=sys.stderr)
        print("  Run tools/make_fixtures.py first.", file=sys.stderr)
        sys.exit(1)

    header, bars = load_binary(INPUT_PATH)
    n = header["bar_count"]

    # Extract OHLC arrays (as float32, matching the binary format)
    closes = bars["close"]
    highs = bars["high"]
    lows = bars["low"]

    print(f"\nComputing indicators on {n} bars...")

    # 1. EMA-20
    print("  EMA-20...")
    ema20 = compute_ema(closes, 20)

    # 2. RSI-14
    print("  RSI-14...")
    rsi14 = compute_rsi(closes, 14)

    # 3. ATR-14
    print("  ATR-14...")
    atr14 = compute_atr(highs, lows, closes, 14)

    # 4. SMA-50
    print("  SMA-50...")
    sma50 = compute_sma(closes, 50)

    # 5. MACD(12,26,9)
    print("  MACD(12,26,9)...")
    macd_line, macd_signal, macd_hist = compute_macd(closes, 12, 26, 9)

    # 6. Bollinger(20,2)
    print("  Bollinger(20,2)...")
    bb_middle, bb_upper, bb_lower = compute_bollinger(closes, 20, 2.0)

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"\nWriting reference CSV: {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "bar_index", "ema20", "rsi14", "atr14", "sma50",
            "macd_line", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower"
        ])

        for i in range(n):
            row = [i]
            for val in [ema20[i], rsi14[i], atr14[i], sma50[i],
                        macd_line[i], macd_signal[i], macd_hist[i],
                        bb_upper[i], bb_middle[i], bb_lower[i]]:
                if np.isnan(val):
                    row.append("NaN")
                else:
                    row.append(f"{val:.9g}")  # round-trips float32 exactly
            writer.writerow(row)

    # Summary: count valid (non-NaN) values per indicator
    print("\n  Valid value counts (non-NaN):")
    for name, arr in [("ema20", ema20), ("rsi14", rsi14), ("atr14", atr14),
                      ("sma50", sma50), ("macd_line", macd_line),
                      ("macd_signal", macd_signal), ("macd_hist", macd_hist),
                      ("bb_upper", bb_upper), ("bb_middle", bb_middle),
                      ("bb_lower", bb_lower)]:
        valid = np.count_nonzero(~np.isnan(arr))
        first_valid = np.argmax(~np.isnan(arr)) if valid > 0 else -1
        print(f"    {name}: {valid:,} valid (first at bar {first_valid})")

    # Print sample values for spot-checking
    print("\n  Sample values at bar 200:")
    i = 200
    print(f"    EMA-20:     {ema20[i]:.6f}")
    print(f"    RSI-14:     {rsi14[i]:.6f}")
    print(f"    ATR-14:     {atr14[i]:.6f}")
    print(f"    SMA-50:     {sma50[i]:.6f}")
    print(f"    MACD line:  {macd_line[i]:.6f}")
    print(f"    MACD sig:   {macd_signal[i]:.6f}")
    print(f"    MACD hist:  {macd_hist[i]:.6f}")
    print(f"    BB upper:   {bb_upper[i]:.6f}")
    print(f"    BB middle:  {bb_middle[i]:.6f}")
    print(f"    BB lower:   {bb_lower[i]:.6f}")

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\n  CSV file size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  DONE.")


if __name__ == "__main__":
    main()
