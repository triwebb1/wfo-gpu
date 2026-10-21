#include "macd.h"
#include <algorithm>
#include <cassert>

MACD::MACD(int fast_period, int slow_period, int signal_period)
    : m_fast_period(fast_period), m_slow_period(slow_period), m_signal_period(signal_period) {
    assert(fast_period > 0);
    assert(slow_period > 0);
    assert(signal_period > 0);
    assert(fast_period < slow_period);
}

int MACD::num_series() const {
    return 3;
}

int MACD::warmup_bars() const {
    // MACD warmup = 3 * slow_period for MACD line convergence
    //             + 3 * signal_period for signal EMA convergence on top
    return 3 * m_slow_period + 3 * m_signal_period;
}

const char* MACD::name() const {
    return "MACD";
}

void MACD::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = warmup_bars();

    // Series layout: MACD line [0..N), signal line [N..2N), histogram [2N..3N)
    float* macd_out   = output;
    float* signal_out = output + N;
    float* hist_out   = output + 2 * N;

    // EMA multipliers
    const float fast_alpha = 2.0f / (static_cast<float>(m_fast_period) + 1.0f);
    const float slow_alpha = 2.0f / (static_cast<float>(m_slow_period) + 1.0f);
    const float sig_alpha  = 2.0f / (static_cast<float>(m_signal_period) + 1.0f);

    float fast_ema           = 0.0f;
    float slow_ema           = 0.0f;
    float signal_ema         = 0.0f;
    bool  signal_initialized = false;

    for (int i = 0; i < N; ++i) {
        const int   bar_idx = start + i;
        const float c       = bars[bar_idx].close;

        // Update fast and slow EMAs
        if (i == 0) {
            fast_ema = c;
            slow_ema = c;
        } else {
            fast_ema = fast_alpha * c + (1.0f - fast_alpha) * fast_ema;
            slow_ema = slow_alpha * c + (1.0f - slow_alpha) * slow_ema;
        }

        // MACD line = fast EMA - slow EMA
        const float macd_val = fast_ema - slow_ema;

        // Signal line = EMA of MACD line
        if (!signal_initialized) {
            signal_ema         = macd_val;
            signal_initialized = true;
        } else {
            signal_ema = sig_alpha * macd_val + (1.0f - sig_alpha) * signal_ema;
        }

        // Histogram = MACD - signal
        const float hist_val = macd_val - signal_ema;

        if (i < wu) {
            macd_out[i]   = INDICATOR_NAN;
            signal_out[i] = INDICATOR_NAN;
            hist_out[i]   = INDICATOR_NAN;
        } else {
            macd_out[i]   = macd_val;
            signal_out[i] = signal_ema;
            hist_out[i]   = hist_val;
        }
    }
}
