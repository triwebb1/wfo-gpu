#include "cci.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

CCI::CCI(int period) : m_period(period) {
    assert(period > 0);
}

int CCI::num_series() const {
    return 1;
}

int CCI::warmup_bars() const {
    // Window-based: first valid output at index m_period - 1
    return m_period - 1;
}

const char* CCI::name() const {
    return "CCI";
}

// Optimized: precompute all typical prices and use running sum for SMA.
// The MAD (Mean Absolute Deviation) inner loop is O(P) per bar and cannot
// be reduced to O(1) -- no running-sum trick exists for absolute deviations.
// However, precomputing typical prices avoids recomputing (H+L+C)/3 in the
// MAD inner loop, and caching the SMA from the running sum avoids redundant
// work. Overall complexity remains O(N*P) for the MAD pass but with a
// smaller constant factor.
// Produces identical output to the previous implementation.
void CCI::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    // warmup_bars() == m_period - 1; warmup is enforced by the
    // `i >= m_period - 1` branch below. No separate `wu` variable needed.
    const float period_f = static_cast<float>(m_period);
    const float constant = 0.015f;

    // Precompute all typical prices to avoid redundant (H+L+C)/3 in MAD loop
    std::vector<float> tp_arr(N);
    for (int i = 0; i < N; ++i) {
        const int bar_idx = start + i;
        tp_arr[i]         = (bars[bar_idx].high + bars[bar_idx].low + bars[bar_idx].close) / 3.0f;
    }

    // Running sum for SMA of typical prices (double to avoid precision loss
    // for high-priced instruments, matching bollinger.cpp pattern)
    double tp_sum = 0.0;

    for (int i = 0; i < N; ++i) {
        const float tp = tp_arr[i];

        tp_sum += tp;

        // Subtract the tp leaving the window
        if (i >= m_period) {
            tp_sum -= tp_arr[i - m_period];
        }

        // CCI valid once we have 'period' bars (index >= period - 1)
        if (i >= m_period - 1) {
            const float sma_tp = static_cast<float>(tp_sum / period_f);

            // Compute mean absolute deviation over the window using precomputed tp
            float mad_sum = 0.0f;
            for (int j = i - m_period + 1; j <= i; ++j) {
                mad_sum += std::abs(tp_arr[j] - sma_tp);
            }
            const float mean_dev = mad_sum / period_f;

            // The outer `i >= m_period - 1` branch already guarantees
            // we have enough window data to emit a CCI value. A further
            // `if (i < wu)` gate is unnecessary -- wu tracks the same warmup
            // condition and is satisfied by the outer check.
            if (mean_dev == 0.0f) {
                output[i] = 0.0f;
            } else {
                output[i] = (tp - sma_tp) / (constant * mean_dev);
            }
        } else {
            output[i] = INDICATOR_NAN;
        }
    }
}
