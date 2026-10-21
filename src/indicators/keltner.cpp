#include "keltner.h"
#include <algorithm>
#include <cassert>
#include <cmath>

KeltnerChannels::KeltnerChannels(MAType ma_type, int ma_period, int atr_period,
                                 float atr_multiplier)
    : m_ma_type(ma_type),
      m_ma_period(ma_period),
      m_atr_period(atr_period),
      m_atr_multiplier(atr_multiplier) {
    assert(ma_period > 0);
    assert(atr_period > 0);
    assert(atr_multiplier > 0.0f);
}

int KeltnerChannels::num_series() const {
    return 3;
}

int KeltnerChannels::warmup_bars() const {
    // Warmup:
    // EMA center: max(3*ma_period, 3*atr_period)
    // SMA center: max(ma_period, 3*atr_period)
    const int atr_warmup = 3 * m_atr_period;
    if (m_ma_type == MAType::EMA) {
        return std::max(3 * m_ma_period, atr_warmup);
    } else {
        return std::max(m_ma_period, atr_warmup);
    }
}

const char* KeltnerChannels::name() const {
    return "KeltnerChannels";
}

void KeltnerChannels::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = warmup_bars();

    // Series layout: center [0..N), upper [N..2N), lower [2N..3N)
    float* center_out = output;
    float* upper_out  = output + N;
    float* lower_out  = output + 2 * N;

    // ---- Compute center line (EMA or SMA of close) ----
    const float ma_period_f = static_cast<float>(m_ma_period);

    // EMA state
    const float ma_alpha = 2.0f / (ma_period_f + 1.0f);
    float       ema_val  = 0.0f;

    // SMA state
    double sma_sum = 0.0;

    // ---- Compute ATR (Wilder smoothing) ----
    const float atr_period_f       = static_cast<float>(m_atr_period);
    const float atr_period_minus_1 = atr_period_f - 1.0f;
    float       atr_val            = 0.0f;
    float       tr_sum             = 0.0f;
    bool        atr_initialized    = false;

    for (int i = 0; i < N; ++i) {
        const int   bar_idx = start + i;
        const float c       = bars[bar_idx].close;

        // ---- Center line ----
        float center = INDICATOR_NAN;
        if (m_ma_type == MAType::EMA) {
            if (i == 0) {
                ema_val = c;
            } else {
                ema_val = ma_alpha * c + (1.0f - ma_alpha) * ema_val;
            }
            center = ema_val;
        } else {
            // SMA with running sum
            sma_sum += static_cast<double>(c);
            if (i >= m_ma_period) {
                sma_sum -= static_cast<double>(bars[start + i - m_ma_period].close);
            }
            if (i >= m_ma_period - 1) {
                center = static_cast<float>(sma_sum / static_cast<double>(m_ma_period));
            }
        }

        // ---- ATR ----
        const float tr = (i == 0) ? bars[bar_idx].high - bars[bar_idx].low
                                  : compute_true_range(bars[bar_idx], bars[bar_idx - 1].close);

        float atr = INDICATOR_NAN;
        if (!atr_initialized) {
            tr_sum += tr;
            if (i == m_atr_period - 1) {
                atr_val         = tr_sum / atr_period_f;
                atr_initialized = true;
                atr             = atr_val;
            }
        } else {
            atr_val = (atr_val * atr_period_minus_1 + tr) / atr_period_f;
            atr     = atr_val;
        }

        // ---- Combine ----
        if (i < wu || std::isnan(center) || std::isnan(atr)) {
            center_out[i] = INDICATOR_NAN;
            upper_out[i]  = INDICATOR_NAN;
            lower_out[i]  = INDICATOR_NAN;
        } else {
            center_out[i] = center;
            upper_out[i]  = center + m_atr_multiplier * atr;
            lower_out[i]  = center - m_atr_multiplier * atr;
        }
    }
}
