#include "atr.h"
#include <algorithm>
#include <cassert>
#include <cmath>

ATR::ATR(int period) : m_period(period) {
    assert(period > 0);
}

int ATR::num_series() const {
    return 1;
}

int ATR::warmup_bars() const {
    // Wilder smoothing warmup = 3 * period
    return 3 * m_period;
}

const char* ATR::name() const {
    return "ATR";
}

void ATR::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int   wu             = warmup_bars();
    const float period_f       = static_cast<float>(m_period);
    const float period_minus_1 = period_f - 1.0f;

    float atr_val     = 0.0f;
    float tr_sum      = 0.0f;
    bool  initialized = false;

    for (int i = 0; i < N; ++i) {
        const int bar_idx = start + i;

        // Compute True Range
        const float tr = (i == 0) ? bars[bar_idx].high - bars[bar_idx].low
                                  : compute_true_range(bars[bar_idx], bars[bar_idx - 1].close);

        if (!initialized) {
            tr_sum += tr;

            if (i == m_period - 1) {
                // First ATR value: simple average of first 'period' true ranges
                atr_val     = tr_sum / period_f;
                initialized = true;

                output[i] = (i < wu) ? INDICATOR_NAN : atr_val;
            } else {
                output[i] = INDICATOR_NAN;
            }
        } else {
            // Wilder smoothing update
            atr_val = (atr_val * period_minus_1 + tr) / period_f;

            output[i] = (i < wu) ? INDICATOR_NAN : atr_val;
        }
    }

    // If N < period, remaining outputs are already set to NaN in the loop
}
