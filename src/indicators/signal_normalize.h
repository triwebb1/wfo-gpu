#pragma once
// ---------------------------------------------------------------------------
// Signal normalization: convert raw indicator output to [-1, +1] signal range.
//
// Technical indicators output values on incompatible scales:
//   RSI: 0-100, EMA: price levels, ATR: price units, CCI: unbounded, etc.
//
// The backtest engine's compute_composite_signal() assumes all values are in
// [-1, +1].  MAJORITY_VOTE checks val > 0 for long, val < 0 for short --
// this is completely broken for oscillators like RSI (always > 0, always
// votes long) and price-level indicators like EMA (votes based on sign of
// a price level, which is always positive for equities/futures).
//
// This module provides normalize_to_signal() which reads ALL raw series for
// one indicator and produces a single per-bar signal in [-1, +1].
//
// Convention: +1 = maximum bullish, -1 = maximum bearish, 0 = neutral.
// NaN is preserved for warmup bars.
// ---------------------------------------------------------------------------

#include "../common.h"
#include "../grids/indicator_grid.h"
#include "../market_data.h"
#include <cmath>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// normalize_to_signal -- convert raw indicator series to a single [-1, +1]
// signal per bar.
//
// Contract:
//   * Output lives in [-1, +1] (NaN passthrough for warmup / insufficient
//     history). Callers in the backtest engine assume this range; violating
//     it silently corrupts composite-signal voting.
//   * The function is pure: no state retained between calls.
//   * num_series MUST equal indicator_num_series(params) AND each
//     raw_series[s] MUST point to at least `count` valid floats aligned to
//     bar_start. Under-provisioning raw_series triggers UB, not a graceful
//     failure -- see the per-case asserts in the implementation.
//   * Series order within raw_series matches the producer order in
//     CLIndicators::compute_with_bars(); see signal_normalize.cpp and
//     its layout invariant comment.
//
// Parameters:
//   params      - IndicatorParams for this indicator (contains thresholds)
//   raw_series  - array of pointers to raw series data.
//                 raw_series[s][i] = value for series s at local bar index i.
//                 Number of pointers = indicator_num_series(params).
//   num_series  - number of raw series (== indicator_num_series(params))
//   bars        - full bar array aligned with series data (bars[bar_start+i])
//   bar_start   - absolute bar index of the first output element
//   count       - number of bars to process
//   signal_out  - output: count floats, each in [-1, +1] (or NaN for warmup)
// ---------------------------------------------------------------------------
void normalize_to_signal(const IndicatorParams& params, const float* const* raw_series,
                         int num_series, const Bar* bars, int bar_start, int count,
                         float* signal_out);

// ---------------------------------------------------------------------------
// normalize_all_indicators -- batch normalization for the full indicator grid.
//
// Takes the flat array of raw indicator series (one float* per raw series,
// as produced by CLIndicators::compute_with_bars) and produces one signal
// array per indicator in the IndicatorGrid.
//
// Parameters:
//   indicator_grid   - the grid defining all indicator combos
//   raw_series_ptrs  - flat array of raw series pointers (sum of num_series
//                      across all grid entries). Same layout as produced by
//                      compute_indicators().
//   bars             - bar data array (first element corresponds to bar_start=0)
//   num_bars         - number of bars in each series
//   signal_out       - output: IndicatorGrid::size() float* pointers.
//                      Each points to a pre-allocated buffer of num_bars floats.
//                      signal_out[indicator_id][i] = normalized signal for
//                      indicator_id at bar i.
// ---------------------------------------------------------------------------
void normalize_all_indicators(const IndicatorGrid&       indicator_grid,
                              const std::vector<float*>& raw_series_ptrs, const Bar* bars,
                              int num_bars, std::vector<float*>& signal_out);
