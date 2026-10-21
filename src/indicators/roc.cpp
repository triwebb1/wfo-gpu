#include "roc.h"
#include <cassert>
#include <cmath>

ROC::ROC(int period, int smoothing_period)
    : m_period(period), m_smoothing_period(smoothing_period) {
    assert(period > 0);
    assert(smoothing_period >= 0);
}

int ROC::num_series() const {
    // Raw ROC only when no smoothing; raw + smoothed when smoothing > 0
    return (m_smoothing_period > 0) ? 2 : 1;
}

int ROC::warmup_bars() const {
    // Window-based: period bars for ROC
    // If smoothed, add 3*smoothing_period for EMA convergence
    if (m_smoothing_period > 0) {
        return m_period + 3 * m_smoothing_period;
    }
    return m_period;
}

const char* ROC::name() const {
    return "ROC";
}

void ROC::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = warmup_bars();

    // Series 0: raw ROC [0..N)
    float* roc_out = output;

    // Compute raw ROC
    for (int i = 0; i < N; ++i) {
        const int bar_idx = start + i;

        if (i >= m_period) {
            const float prev_close = bars[start + i - m_period].close;
            if (prev_close == 0.0f) {
                roc_out[i] = 0.0f;
            } else {
                roc_out[i] = ((bars[bar_idx].close - prev_close) / prev_close) * 100.0f;
            }
        } else {
            roc_out[i] = INDICATOR_NAN;
        }
    }

    // If smoothing is requested, compute EMA of raw ROC
    if (m_smoothing_period > 0) {
        float*      smooth_out      = output + N;
        const float alpha           = 2.0f / (static_cast<float>(m_smoothing_period) + 1.0f);
        const float one_minus_alpha = 1.0f - alpha;

        float ema             = 0.0f;
        bool  ema_initialized = false;

        for (int i = 0; i < N; ++i) {
            if (i >= m_period && !std::isnan(roc_out[i])) {
                if (!ema_initialized) {
                    ema             = roc_out[i];
                    ema_initialized = true;
                } else {
                    ema = alpha * roc_out[i] + one_minus_alpha * ema;
                }

                if (i < wu) {
                    smooth_out[i] = INDICATOR_NAN;
                } else {
                    smooth_out[i] = ema;
                }
            } else {
                smooth_out[i] = INDICATOR_NAN;
            }
        }
    }

    // Apply warmup NaN to raw ROC series as well when smoothing changes the warmup
    // Raw ROC values during warmup are needed for internal EMA computation
    // but the final output should respect the combined warmup
    for (int i = 0; i < std::min(wu, N); ++i) {
        roc_out[i] = INDICATOR_NAN;
    }
}
