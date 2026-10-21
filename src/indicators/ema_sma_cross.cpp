#include "ema_sma_cross.h"
#include <algorithm>
#include <cassert>

EMASMACross::EMASMACross(MAType fast_type, int fast_period, MAType slow_type, int slow_period)
    : m_fast_type(fast_type),
      m_fast_period(fast_period),
      m_slow_type(slow_type),
      m_slow_period(slow_period) {
    assert(fast_period > 0);
    assert(slow_period > 0);
    assert(fast_period < slow_period);
}

int EMASMACross::ma_warmup(MAType type, int period) {
    // EMA warmup = 3 * period
    // SMA first valid value is at index (period - 1), when the window is full
    if (type == MAType::EMA) {
        return 3 * period;
    }
    return period - 1;
}

int EMASMACross::warmup_bars() const {
    return std::max(ma_warmup(m_fast_type, m_fast_period), ma_warmup(m_slow_type, m_slow_period));
}

int EMASMACross::num_series() const {
    return 2;
}

const char* EMASMACross::name() const {
    return "EMASMACross";
}

void EMASMACross::compute_ma(const Bar* bars, int start, int end, MAType type, int period,
                             float* dst) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = ma_warmup(type, period);

    if (type == MAType::EMA) {
        const float alpha           = 2.0f / (static_cast<float>(period) + 1.0f);
        const float one_minus_alpha = 1.0f - alpha;

        // Seed EMA with the first close value in the range
        float ema = bars[start].close;

        for (int i = 0; i < N; ++i) {
            const int   bar_idx = start + i;
            const float c       = bars[bar_idx].close;

            if (i == 0) {
                ema = c;
            } else {
                ema = alpha * c + one_minus_alpha * ema;
            }

            // Mark warmup period as NaN
            dst[i] = (i < wu) ? INDICATOR_NAN : ema;
        }
    } else {
        // SMA with running sum for O(1) per bar
        double running_sum = 0.0;

        for (int i = 0; i < N; ++i) {
            const int    bar_idx = start + i;
            const double c       = static_cast<double>(bars[bar_idx].close);

            running_sum += c;

            // Subtract the element leaving the window
            if (i >= period) {
                running_sum -= static_cast<double>(bars[start + i - period].close);
            }

            // SMA is valid once we have 'period' bars (index >= period - 1)
            if (i >= period - 1 && i >= wu) {
                dst[i] = static_cast<float>(running_sum / static_cast<double>(period));
            } else {
                dst[i] = INDICATOR_NAN;
            }
        }
    }
}

void EMASMACross::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    // Series 0: fast MA values  [output + 0]
    // Series 1: slow MA values  [output + N]
    float* fast_out = output;
    float* slow_out = output + N;

    compute_ma(bars, start, end, m_fast_type, m_fast_period, fast_out);
    compute_ma(bars, start, end, m_slow_type, m_slow_period, slow_out);
}
