#include "volatility_regime.h"

// =============================================================================
// Volatility Regime Signals
// =============================================================================

VolatilityRegimeSignals::VolatilityRegimeSignals(int atr_short_lookback, int atr_long_lookback,
                                                 float expansion_threshold,
                                                 float contraction_threshold)
    : m_atr_short_lookback(atr_short_lookback),
      m_atr_long_lookback(atr_long_lookback),
      m_expansion_threshold(expansion_threshold),
      m_contraction_threshold(contraction_threshold) {
    assert(atr_short_lookback > 0);
    assert(atr_long_lookback > atr_short_lookback);
    assert(expansion_threshold > contraction_threshold);
}

int VolatilityRegimeSignals::num_signals() const {
    return 4;
}
const char* VolatilityRegimeSignals::name() const {
    return "VolatilityRegime";
}
int VolatilityRegimeSignals::warmup_days() const {
    return m_atr_long_lookback + 1;
}

std::vector<int32_t> VolatilityRegimeSignals::signal_available_after_et_seconds() const {
    // All vol regime signals use D-1 and earlier data only -> always available
    return {0, 0, 0, 0};
}

void VolatilityRegimeSignals::compute(const MarketData& md, int day_start, int day_end,
                                      float* output) const {
    const int num_days = day_end - day_start;
    const int NS       = num_signals();

    for (int i = 0; i < NS * num_days; ++i) output[i] = SIGNAL_NAN;

    for (int d = day_start; d < day_end; ++d) {
        const int di = d - day_start;

        if (d < m_atr_long_lookback + 1) continue;  // Need enough history for long ATR

        // Signal 0: ATR ratio (short vs long)
        float atr_short = compute_session_atr(md, d, m_atr_short_lookback);
        float atr_long  = compute_session_atr(md, d, m_atr_long_lookback);

        if (!std::isnan(atr_short) && !std::isnan(atr_long) && atr_long > 0.0f) {
            float ratio = atr_short / atr_long;
            // ratio > 1 = expanding volatility, < 1 = contracting
            // Scale: (ratio - 1.0) * 2, clamped to [-1, +1]
            output[0 * num_days + di] = clamp_signal((ratio - 1.0f) * 2.0f);
        }

        // Prev day range vs ATR
        SessionDayView prev = build_session_view(md, d - 1);
        if (prev.rth_count == 0) continue;

        float prev_range = prev.rth_range();

        // Use short-lookback ATR as the reference for expansion/contraction
        float atr_ref = compute_session_atr(md, d, m_atr_short_lookback);
        if (std::isnan(atr_ref) || atr_ref <= 0.0f) continue;

        float range_ratio = prev_range / atr_ref;

        // Signal 1: Prev day range expansion (> threshold x ATR)
        output[1 * num_days + di] = (range_ratio > m_expansion_threshold) ? 1.0f : 0.0f;

        // Signal 2: Prev day range contraction (< threshold x ATR)
        output[2 * num_days + di] = (range_ratio < m_contraction_threshold) ? 1.0f : 0.0f;

        // Signal 3: Consecutive expansion/contraction days
        {
            int expansion_streak   = 0;
            int contraction_streak = 0;

            for (int k = d - 1; k >= 1; --k) {
                float k_atr = compute_session_atr(md, k, m_atr_short_lookback);
                if (std::isnan(k_atr) || k_atr <= 0.0f) break;

                SessionDayView dv = build_session_view(md, k);
                if (dv.rth_count == 0) break;

                float k_ratio = dv.rth_range() / k_atr;

                if (k_ratio > m_expansion_threshold) {
                    if (contraction_streak > 0) break;  // streak broken
                    expansion_streak++;
                } else if (k_ratio < m_contraction_threshold) {
                    if (expansion_streak > 0) break;
                    contraction_streak++;
                } else {
                    break;  // Neither expansion nor contraction
                }
            }

            float streak_signal;
            if (expansion_streak > 0) {
                streak_signal = clamp_signal(static_cast<float>(expansion_streak) / 5.0f);
            } else if (contraction_streak > 0) {
                streak_signal = -clamp_signal(static_cast<float>(contraction_streak) / 5.0f);
            } else {
                streak_signal = 0.0f;
            }
            output[3 * num_days + di] = streak_signal;
        }
    }
}
