#include "signal_normalize.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Shared normalization primitives
// ---------------------------------------------------------------------------

// Oscillator (0-100): overbought/oversold with linear scaling.
// Used by RSI, Stochastic, ConnorsRSI.
static void normalize_oscillator(const float* raw, int count, float* out, float oversold,
                                 float overbought) {
    for (int i = 0; i < count; ++i) {
        const float v = raw[i];
        if (std::isnan(v)) {
            out[i] = v;
            continue;
        }

        if (v < oversold)
            out[i] = clamp_signal((oversold - v) / std::max(1e-6f, oversold));
        else if (v > overbought)
            out[i] = clamp_signal(-(v - overbought) / std::max(1e-6f, 100.0f - overbought));
        else
            out[i] = 0.0f;
    }
}

// Channel: price below lower -> buy, above upper -> sell.
// Used by Bollinger, Keltner, VWAP. Set strict=false for mean-reversion
// (price < lower) or strict=true for breakout (price <= lower).
static void normalize_channel(const float* upper, const float* lower, const Bar* bars,
                              int bar_start, int count, float* out, bool breakout = false) {
    for (int i = 0; i < count; ++i) {
        const float u = upper[i];
        const float l = lower[i];
        if (std::isnan(u) || std::isnan(l)) {
            out[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        // Strict-equality guard. Collapsed band (u == l) yields
        // an undefined signal under either mode; emit neutral (0). Preserves
        // breakout-mode early-warmup semantics (DONCHIAN/PIVOT seed both
        // series equal before accumulating extrema).
        if (u == l) {
            out[i] = 0.0f;
            continue;
        }
        // Malformed inversion (upper < lower) is a data-quality
        // fault -- e.g., PIVOT R1 < S1 produced by a bad prior-session
        // aggregate. Emit NaN rather than a silent directional signal.
        if (u < l) {
            out[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        const float price = bars[bar_start + i].close;
        if (breakout) {
            // Breakout: at-or-beyond the band
            if (price >= u)
                out[i] = 1.0f;
            else if (price <= l)
                out[i] = -1.0f;
            else
                out[i] = 0.0f;
        } else {
            // Mean-reversion: strictly beyond the band
            if (price < l)
                out[i] = 1.0f;
            else if (price > u)
                out[i] = -1.0f;
            else
                out[i] = 0.0f;
        }
    }
}

// Binary sign: positive -> +1, negative -> -1, zero -> 0. NaN preserved.
static void normalize_sign(const float* raw, int count, float* out) {
    for (int i = 0; i < count; ++i) {
        const float v = raw[i];
        if (std::isnan(v)) {
            out[i] = v;
            continue;
        }
        if (v > 0.0f)
            out[i] = 1.0f;
        else if (v < 0.0f)
            out[i] = -1.0f;
        else
            out[i] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Per-indicator normalization logic
// ---------------------------------------------------------------------------

// EMA/SMA Crossover: signal = sign(fast - slow).
static void normalize_ema_sma_cross(const float* fast_ma, const float* slow_ma, int count,
                                    float* out) {
    // Compute difference, then take sign
    for (int i = 0; i < count; ++i) {
        if (std::isnan(fast_ma[i]) || std::isnan(slow_ma[i])) {
            out[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        const float diff = fast_ma[i] - slow_ma[i];
        out[i]           = (diff > 0.0f) ? 1.0f : (diff < 0.0f) ? -1.0f : 0.0f;
    }
}

// ADX: 3 series -- ADX (0), +DI (1), -DI (2).
// If ADX > threshold (trend exists): +DI > -DI -> +1, +DI < -DI -> -1.
// If ADX <= threshold (no trend): signal = 0.
static void normalize_adx(const IndicatorParams& params, const float* adx_val, const float* plus_di,
                          const float* minus_di, int count, float* out) {
    const float threshold = params.adx_threshold;

    for (int i = 0; i < count; ++i) {
        const float a  = adx_val[i];
        const float pd = plus_di[i];
        const float md = minus_di[i];

        if (std::isnan(a) || std::isnan(pd) || std::isnan(md)) {
            out[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        if (a > threshold) {
            if (pd > md)
                out[i] = 1.0f;
            else if (pd < md)
                out[i] = -1.0f;
            else
                out[i] = 0.0f;
        } else {
            out[i] = 0.0f;
        }
    }
}

// Stochastic: same oscillator normalization as RSI with stochastic thresholds.

// CCI: 1 series, unbounded (typically -200 to +200).
// CCI > +threshold -> sell (mean-reversion), CCI < -threshold -> buy.
// Scale linearly: signal reaches +/-1 at +/-2*threshold.
// Note: Using mean-reversion interpretation consistent with oscillator logic.
static void normalize_cci(const IndicatorParams& params, const float* raw_cci, int count,
                          float* out) {
    const float THRESHOLD = params.cci_threshold;
    // Producer-side guard. cci_threshold == 0 would yield (v / 0) = +/-Inf
    // downstream, which clamp_signal would then saturate to +/-1 silently. Refuse
    // such configurations at the source rather than ship spurious extreme signals.
    assert(THRESHOLD > 0.0f && "normalize_cci: cci_threshold must be > 0");
    const float SCALE = THRESHOLD;  // signal reaches +/-1 at +/-2*threshold

    for (int i = 0; i < count; ++i) {
        const float v = raw_cci[i];
        if (std::isnan(v)) {
            out[i] = v;
            continue;
        }

        if (v > THRESHOLD) {
            // Overbought region: sell signal
            out[i] = clamp_signal(-(v - THRESHOLD) / SCALE);
        } else if (v < -THRESHOLD) {
            // Oversold region: buy signal
            out[i] = clamp_signal((-THRESHOLD - v) / SCALE);
        } else {
            out[i] = 0.0f;
        }
    }
}

// ROC: 1 series, percentage change (unbounded).
// ROC > 0 -> bullish (+1), ROC < 0 -> bearish (-1).
// Scale relative to the ROC lookback period: longer periods produce larger
// moves, so we scale with sqrt(period) (volatility ~ sqrt(time)).
// Base: 0.5% ROC over a 5-bar lookback = full signal.
static void normalize_roc(const IndicatorParams& params, const float* raw_roc, int count,
                          float* out) {
    constexpr float BASE_SCALE = 0.5f;  // 0.5% ROC = full signal at reference period
    constexpr float REF_PERIOD = 5.0f;  // reference lookback period
    const float     period     = static_cast<float>(std::max(1, params.roc_period));
    const float     scale      = BASE_SCALE * std::sqrt(period / REF_PERIOD);

    for (int i = 0; i < count; ++i) {
        const float v = raw_roc[i];
        if (std::isnan(v)) {
            out[i] = v;
            continue;
        }

        out[i] = clamp_signal(v / scale);
    }
}

// VWAP: channel normalization (mean-reversion).

// OBV: 1 series (raw or smoothed OBV).
// OBV is cumulative and unbounded. Compare current OBV against OBV from
// N bars ago to determine direction (slope-over-N). This smooths out the
// noise inherent in bar-to-bar OBV changes on short timeframes.
// Lookback equals the smoothing period when set (uncapped); otherwise
// default to 10. Capping the lookback at 10 would collapse
// the {10, 20, 50} grid to a single effective behavior for obv_smoothing>10.
static void normalize_obv(const IndicatorParams& params, const float* raw_obv, int count,
                          float* out) {
    // OBV smoothing of 1 is a degenerate slope-of-1-bar comparison
    // (noise dominated). Grid values are {10, 20, 50}; enforce > 1 so a
    // misconfigured spec fails loudly during CPU-path normalization.
    assert(params.obv_smoothing > 1 && "obv_smoothing must be > 1");
    // Use smoothing period as lookback if set, otherwise default to 10
    const int lookback = (params.obv_smoothing > 0) ? params.obv_smoothing : 10;

    for (int i = 0; i < count; ++i) {
        const float v = raw_obv[i];
        if (std::isnan(v)) {
            out[i] = v;
            continue;
        }

        if (i < lookback) {
            // Not enough history for full lookback: use earliest valid bar.
            int ref = -1;
            for (int j = 0; j < i; ++j) {
                if (!std::isnan(raw_obv[j])) {
                    ref = j;
                    break;
                }
            }
            if (ref < 0) {
                out[i] = 0.0f;
                continue;
            }
            const float ref_val = raw_obv[ref];
            if (v > ref_val)
                out[i] = 1.0f;
            else if (v < ref_val)
                out[i] = -1.0f;
            else
                out[i] = 0.0f;
            continue;
        }

        const float prev = raw_obv[i - lookback];
        if (std::isnan(prev)) {
            out[i] = 0.0f;
            continue;
        }

        if (v > prev)
            out[i] = 1.0f;
        else if (v < prev)
            out[i] = -1.0f;
        else
            out[i] = 0.0f;
    }
}

// Donchian: channel normalization (breakout mode).

// HMA (Hull Moving Average): 1 series -- price level.
// Trend direction signal: HMA rising -> +1 (bullish), falling -> -1 (bearish).
static void normalize_hma(const float* raw_hma, int count, float* out) {
    for (int i = 0; i < count; ++i) {
        const float v = raw_hma[i];
        if (std::isnan(v)) {
            out[i] = v;
            continue;
        }

        if (i == 0) {
            out[i] = 0.0f;
            continue;
        }

        const float prev = raw_hma[i - 1];
        if (std::isnan(prev)) {
            out[i] = 0.0f;
            continue;
        }

        if (v > prev)
            out[i] = 1.0f;
        else if (v < prev)
            out[i] = -1.0f;
        else
            out[i] = 0.0f;
    }
}

// Connors RSI: same oscillator normalization as RSI with Connors thresholds.

// ATR: non-directional (volatility measure). Always 0 signal.
static void normalize_atr(int count, float* out) {
    std::memset(out, 0, count * sizeof(float));
}

// ---------------------------------------------------------------------------
// Main dispatch function
// ---------------------------------------------------------------------------

void normalize_to_signal(const IndicatorParams& params, const float* const* raw_series,
                         int num_series, const Bar* bars, int bar_start, int count,
                         float* signal_out) {
    // Every indicator needs at least one raw series; the per-case
    // asserts below further constrain num_series against
    // indicator_num_series(params.type). These asserts are compiled out in
    // release; upstream call sites must still honor the contract documented
    // in signal_normalize.h.
    assert(num_series >= 1);
    (void)num_series;  // used for assertions only

    switch (params.type) {
        case IndicatorType::RSI:
            assert(num_series >= 1);
            normalize_oscillator(raw_series[0], count, signal_out, params.rsi_oversold,
                                 params.rsi_overbought);
            break;

        case IndicatorType::EMA_SMA_CROSS:
            assert(num_series >= 2);
            normalize_ema_sma_cross(raw_series[0], raw_series[1], count, signal_out);
            break;

        case IndicatorType::MACD:
            assert(num_series >= 3);
            normalize_sign(raw_series[2], count, signal_out);
            break;

        case IndicatorType::ADX:
            assert(num_series >= 3);
            // Series 0=ADX, 1=+DI, 2=-DI
            normalize_adx(params, raw_series[0], raw_series[1], raw_series[2], count, signal_out);
            break;

        case IndicatorType::BOLLINGER:
            assert(num_series >= 3);
            normalize_channel(raw_series[1], raw_series[2], bars, bar_start, count, signal_out);
            break;

        case IndicatorType::KELTNER:
            assert(num_series >= 3);
            normalize_channel(raw_series[1], raw_series[2], bars, bar_start, count, signal_out);
            break;

        case IndicatorType::STOCHASTIC:
            assert(num_series >= 2);
            // NOTE: raw_series[0] is always %K (IndicatorGrid's series order).
            // The indicator grid generates a separate %D slot (series index 1) but signal_normalize
            // unconditionally uses raw_series[0] here. Strategies assigned to the %D slot
            // effectively duplicate %K strategies -- known limitation.
            normalize_oscillator(raw_series[0], count, signal_out, params.stoch_oversold,
                                 params.stoch_overbought);
            break;

        case IndicatorType::CCI:
            assert(num_series >= 1);
            normalize_cci(params, raw_series[0], count, signal_out);
            break;

        case IndicatorType::ROC:
            assert(num_series >= 1);
            normalize_roc(params, raw_series[0], count, signal_out);
            break;

        case IndicatorType::VWAP:
            assert(num_series >= 3);
            normalize_channel(raw_series[1], raw_series[2], bars, bar_start, count, signal_out);
            break;

        case IndicatorType::OBV:
            assert(num_series >= 1);
            normalize_obv(params, raw_series[0], count, signal_out);
            break;

        case IndicatorType::DONCHIAN:
            assert(num_series >= 2);
            normalize_channel(raw_series[0], raw_series[1], bars, bar_start, count, signal_out,
                              true);
            break;

        case IndicatorType::PIVOT_POINTS:
            assert(num_series >= 7);
            // S1=series[1], R1=series[4]; use breakout mode (> R1 or < S1)
            normalize_channel(raw_series[4], raw_series[1], bars, bar_start, count, signal_out,
                              true);
            break;

        case IndicatorType::ATR:
            // ATR has 2 sub-series (short/long) but both are non-directional.
            // Output neutral signal.
            normalize_atr(count, signal_out);
            break;

        case IndicatorType::HMA:
            assert(num_series >= 1);
            normalize_hma(raw_series[0], count, signal_out);
            break;

        case IndicatorType::CONNORS_RSI:
            assert(num_series >= 1);
            normalize_oscillator(raw_series[0], count, signal_out, params.connors_oversold,
                                 params.connors_overbought);
            break;

        default:
            // Runtime check: every IndicatorType must have a
            // normalization case above; a new indicator type added without a
            // matching handler is a programmer error that would silently emit
            // neutral signals and drift strategies toward no-trade in release
            // builds. Throw std::logic_error so the failure is visible.
            throw std::logic_error(
                "unhandled IndicatorType in normalize_to_signal: "
                "indicator " +
                std::to_string(static_cast<int>(params.type)));
    }
}

// ---------------------------------------------------------------------------
// normalize_all_indicators -- batch normalization for the full indicator grid
// ---------------------------------------------------------------------------

void normalize_all_indicators(const IndicatorGrid&       indicator_grid,
                              const std::vector<float*>& raw_series_ptrs, const Bar* bars,
                              int num_bars, std::vector<float*>& signal_out) {
    const int num_indicators = indicator_grid.size();

    // Layout invariant: raw_series_ptrs is laid out contiguously in indicator-
    // grid order. For indicator i, its series occupy the slab
    //   raw_series_ptrs[raw_offset .. raw_offset + num_series(params_i))
    // where num_series is determined by indicator_num_series(params).
    // Series index within an indicator matches the producer order used by
    // CLIndicators::compute_with_bars() (e.g. MACD: 0=line, 1=signal,
    // 2=histogram; VWAP: 0=center, 1=upper, 2=lower; etc.). Any reshuffle
    // of that producer order MUST be mirrored in the normalize_* callers
    // below (see the switch in normalize_to_signal).
    int raw_offset = 0;

    for (int ind = 0; ind < num_indicators; ++ind) {
        const IndicatorParams& params = indicator_grid.params(ind);
        int                    ns     = indicator_num_series(params);

        // Gather pointers to this indicator's raw series
        std::vector<const float*> series_ptrs(ns);
        for (int s = 0; s < ns; ++s) {
            series_ptrs[s] = raw_series_ptrs[raw_offset + s];
        }

        // Normalize into the signal output for this indicator
        normalize_to_signal(params, series_ptrs.data(), ns, bars,
                            0,  // bar_start = 0 (bars pointer already offset by caller)
                            num_bars, signal_out[ind]);

        raw_offset += ns;
    }

    // Verify we consumed all raw series
    assert(raw_offset == static_cast<int>(raw_series_ptrs.size()));
}
