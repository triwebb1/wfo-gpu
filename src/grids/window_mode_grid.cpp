#include "window_mode_grid.h"
#include <algorithm>

int WindowModeGrid::primary_period(const IndicatorParams& p) {
    switch (p.type) {
        case IndicatorType::EMA_SMA_CROSS:
            // The slow period is the dominant lookback
            return p.slow_period;
        case IndicatorType::RSI:
            return p.rsi_period;
        case IndicatorType::ATR:
            // The long period is the dominant lookback
            return p.atr_long_period;
        case IndicatorType::MACD:
            return p.macd_slow_period;
        case IndicatorType::ADX:
            return p.adx_period;
        case IndicatorType::BOLLINGER:
            return p.bollinger_period;
        case IndicatorType::KELTNER:
            return std::max(p.keltner_ma_period, p.keltner_atr_period);
        case IndicatorType::STOCHASTIC:
            return p.stoch_k_period;
        case IndicatorType::CCI:
            return p.cci_period;
        case IndicatorType::ROC:
            return p.roc_period;
        case IndicatorType::VWAP:
            return 1;  // VWAP resets each session, no lookback concern
        case IndicatorType::OBV:
            return p.obv_smoothing;
        case IndicatorType::DONCHIAN:
            return p.donchian_period;
        case IndicatorType::PIVOT_POINTS:
            return 1;  // Pivot points are session-based, no lookback concern
        case IndicatorType::HMA:
            return p.hma_period;
        case IndicatorType::CONNORS_RSI:
            return std::max(
                {p.connors_rsi_period, p.connors_streak_period, p.connors_percentile_period});
    }
    return 1;
}

bool WindowModeGrid::is_valid_combo(const IndicatorParams& p, WindowMode mode) {
    if (mode == WindowMode::ROLLING) return true;  // Always valid

    // Session-anchored indicators (VWAP, Pivot Points) are inherently
    // session-anchored and should NOT have window mode variants.
    // Only ROLLING mode is meaningful for these.
    if (p.type == IndicatorType::VWAP || p.type == IndicatorType::PIVOT_POINTS) {
        return false;  // non-ROLLING modes rejected
    }

    // Volume-dependent indicators (OBV) may not make sense on anchored TFs
    // where volume aggregation changes meaning.  Restrict to ROLLING and
    // ANCHORED_5M max (anything coarser distorts volume semantics).
    if (p.type == IndicatorType::OBV) {
        if (mode != WindowMode::ANCHORED_1M && mode != WindowMode::ANCHORED_2M &&
            mode != WindowMode::ANCHORED_5M) {
            return false;
        }
    }

    int period = primary_period(p);
    int tf_sec = window_mode_seconds(mode);

    // Effective lookback in hours = period * tf_sec / 3600
    float lookback_hours = static_cast<float>(period) * static_cast<float>(tf_sec) / 3600.0f;

    // Prune if lookback exceeds 7 days (168 hours).
    // The generous threshold allows regime-detection use cases such as
    // EMA-200 on 1h (200 hours) or ADX-28 on 1h (28 hours) while still
    // pruning nonsensical extremes.
    return lookback_hours <= MAX_LOOKBACK_HOURS;
}

WindowModeGrid::WindowModeGrid(const IndicatorGrid& ig) : m_indicator_grid(&ig) {
    constexpr WindowMode all_modes[] = {WindowMode::ROLLING,      WindowMode::ANCHORED_1M,
                                        WindowMode::ANCHORED_2M,  WindowMode::ANCHORED_5M,
                                        WindowMode::ANCHORED_15M, WindowMode::ANCHORED_30M,
                                        WindowMode::ANCHORED_1H};

    for (int i = 0; i < ig.size(); ++i) {
        const auto& params = ig.params(i);
        for (WindowMode mode : all_modes) {
            if (is_valid_combo(params, mode)) {
                WindowModeEntry e;
                e.indicator_id = i;
                e.mode         = mode;
                m_entries.push_back(e);
            }
        }
    }
}

int WindowModeGrid::find(const WindowModeEntry& e) const {
    for (int i = 0; i < static_cast<int>(m_entries.size()); ++i) {
        if (m_entries[i] == e) return i;
    }
    return -1;
}
