#include "indicator_warmup.h"
#include "cl_indicators.h"  // IndicatorSpec

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// warmup_for_params -- compute warmup for one indicator (with 2x safety)
// ---------------------------------------------------------------------------

static int warmup_for_params(const IndicatorParams& params) {
    int recommended = 0;

    switch (params.type) {
        case IndicatorType::EMA_SMA_CROSS: {
            // Each MA component has its own warmup
            int fast_wu = (params.fast_ma_type == MAType::EMA) ? 3 * params.fast_period
                                                               : (params.fast_period - 1);
            int slow_wu = (params.slow_ma_type == MAType::EMA) ? 3 * params.slow_period
                                                               : (params.slow_period - 1);
            recommended = std::max(fast_wu, slow_wu);
            break;
        }

        case IndicatorType::RSI:
            // Wilder smoothing: 3x period recommended
            recommended = 3 * params.rsi_period;
            break;

        case IndicatorType::ATR:
            // Wilder smoothing: 3x period recommended, take max of short and long
            recommended = std::max(3 * params.atr_short_period, 3 * params.atr_long_period);
            break;

        case IndicatorType::MACD:
            // Warmup = 3 * slow_period (exponentially weighted)
            recommended = 3 * params.macd_slow_period;
            break;

        case IndicatorType::ADX:
            // Warmup = 4 * period (double Wilder smoothing)
            recommended = 4 * params.adx_period;
            break;

        case IndicatorType::BOLLINGER:
            // Window-based: period bars
            recommended = params.bollinger_period;
            break;

        case IndicatorType::KELTNER: {
            // max(ma_warmup, atr_warmup)
            int atr_wu = 3 * params.keltner_atr_period;
            if (params.keltner_ma_type == MAType::EMA) {
                recommended = std::max(3 * params.keltner_ma_period, atr_wu);
            } else {
                recommended = std::max(params.keltner_ma_period, atr_wu);
            }
            break;
        }

        case IndicatorType::STOCHASTIC:
            // K_period + D_period
            recommended = params.stoch_k_period + params.stoch_d_period;
            break;

        case IndicatorType::CCI:
            // Window-based: period bars
            recommended = params.cci_period;
            break;

        case IndicatorType::ROC:
            // Window-based: period bars
            recommended = params.roc_period;
            break;

        case IndicatorType::VWAP:
            // Resets each session, no warmup needed
            recommended = 0;
            break;

        case IndicatorType::OBV:
            // Smoothing period
            recommended = params.obv_smoothing;
            break;

        case IndicatorType::DONCHIAN:
            // Window-based: period bars
            recommended = params.donchian_period;
            break;

        case IndicatorType::PIVOT_POINTS:
            // ~1 session at 30s bars
            recommended = 2640;
            break;

        case IndicatorType::HMA: {
            int sqrt_p =
                static_cast<int>(std::floor(std::sqrt(static_cast<float>(params.hma_period))));
            recommended = params.hma_period + sqrt_p;
            break;
        }

        case IndicatorType::CONNORS_RSI:
            recommended = std::max({3 * params.connors_rsi_period, 3 * params.connors_streak_period,
                                    params.connors_percentile_period});
            break;
    }

    // Apply 2x safety factor
    return recommended * 2;
}

// ---------------------------------------------------------------------------
// compute_max_warmup -- from parameter list
// ---------------------------------------------------------------------------

int max_indicator_warmup(const std::vector<IndicatorParams>& params_list) {
    int max_wu = 0;
    for (const auto& p : params_list) {
        max_wu = std::max(max_wu, warmup_for_params(p));
    }
    return max_wu;
}

// ---------------------------------------------------------------------------
// compute_max_warmup -- from IndicatorGrid
// ---------------------------------------------------------------------------

int max_indicator_warmup(const IndicatorGrid& grid) {
    int max_wu = 0;
    for (int i = 0; i < grid.size(); ++i) {
        max_wu = std::max(max_wu, warmup_for_params(grid.params(i)));
    }
    return max_wu;
}

// ---------------------------------------------------------------------------
// scale_warmup_for_mode -- scale native warmup to 30s bars
// ---------------------------------------------------------------------------

static int scale_warmup_for_mode(int native_warmup, WindowMode mode) {
    if (mode == WindowMode::ROLLING) return native_warmup;
    int tf_sec = window_mode_seconds(mode);
    int ratio  = tf_sec / 30;
    return native_warmup * ratio;
}

// ---------------------------------------------------------------------------
// compute_max_warmup -- from IndicatorSpec list (anchored-aware)
// ---------------------------------------------------------------------------

int max_indicator_warmup(const std::vector<IndicatorSpec>& specs) {
    int max_wu = 0;
    for (const auto& spec : specs) {
        // Build a minimal IndicatorParams to compute native warmup
        IndicatorParams p;
        p.type                      = spec.type;
        p.fast_ma_type              = spec.fast_ma_type;
        p.fast_period               = spec.fast_period;
        p.slow_ma_type              = spec.slow_ma_type;
        p.slow_period               = spec.slow_period;
        p.rsi_period                = spec.rsi_period;
        p.atr_short_period          = spec.atr_period;
        p.atr_long_period           = spec.atr_period;
        p.macd_fast_period          = spec.macd_fast_period;
        p.macd_slow_period          = spec.macd_slow_period;
        p.macd_signal_period        = spec.macd_signal_period;
        p.adx_period                = spec.adx_period;
        p.bollinger_period          = spec.bollinger_period;
        p.keltner_ma_type           = spec.keltner_ma_type;
        p.keltner_ma_period         = spec.keltner_ma_period;
        p.keltner_atr_period        = spec.keltner_atr_period;
        p.stoch_k_period            = spec.stoch_k_period;
        p.stoch_d_period            = spec.stoch_d_period;
        p.cci_period                = spec.cci_period;
        p.roc_period                = spec.roc_period;
        p.obv_smoothing             = spec.obv_smoothing;
        p.donchian_period           = spec.donchian_period;
        p.hma_period                = spec.hma_period;
        p.connors_rsi_period        = spec.connors_rsi_period;
        p.connors_streak_period     = spec.connors_streak_period;
        p.connors_percentile_period = spec.connors_percentile_period;

        int native_wu = warmup_for_params(p);
        int scaled_wu = scale_warmup_for_mode(native_wu, spec.window_mode);
        max_wu        = std::max(max_wu, scaled_wu);
    }
    return max_wu;
}
