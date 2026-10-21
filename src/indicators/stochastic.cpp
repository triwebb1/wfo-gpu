#include "stochastic.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <vector>

Stochastic::Stochastic(int k_period, int d_period) : m_k_period(k_period), m_d_period(d_period) {
    assert(k_period > 0);
    assert(d_period > 0);
}

int Stochastic::num_series() const {
    return 2;
}

int Stochastic::warmup_bars() const {
    // Warmup: K_period + D_period
    return m_k_period + m_d_period;
}

const char* Stochastic::name() const {
    return "Stochastic";
}

// Optimized from O(N*K) to O(N) using monotonic deques for rolling max high
// and rolling min low over the K-period window. Each element is pushed and
// popped from the deque at most once, giving amortized O(1) per bar.
// Produces identical output to the previous inner-loop implementation.
void Stochastic::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int   wu         = warmup_bars();
    const float d_period_f = static_cast<float>(m_d_period);

    // Series layout: %K [0..N), %D [N..2N)
    float* k_out = output;
    float* d_out = output + N;

    // Compute raw %K values into a temporary buffer so the running sum for %D
    // is never corrupted by NaN masking.
    std::vector<float> raw_k(N, INDICATOR_NAN);

    // Monotonic deques for rolling max of highs and rolling min of lows
    // over K-period window. Stores indices into [0..N).
    std::deque<int> max_deque;  // front = index of max high in window
    std::deque<int> min_deque;  // front = index of min low in window

    for (int i = 0; i < N; ++i) {
        const float h = bars[start + i].high;
        const float l = bars[start + i].low;

        // Maintain monotonic deque for rolling max of highs
        while (!max_deque.empty() && max_deque.front() <= i - m_k_period) max_deque.pop_front();
        while (!max_deque.empty() && bars[start + max_deque.back()].high <= h) max_deque.pop_back();
        max_deque.push_back(i);

        // Maintain monotonic deque for rolling min of lows
        while (!min_deque.empty() && min_deque.front() <= i - m_k_period) min_deque.pop_front();
        while (!min_deque.empty() && bars[start + min_deque.back()].low >= l) min_deque.pop_back();
        min_deque.push_back(i);

        // %K requires at least k_period bars (index >= k_period - 1)
        if (i >= m_k_period - 1) {
            const float highest_high = bars[start + max_deque.front()].high;
            const float lowest_low   = bars[start + min_deque.front()].low;

            const float range = highest_high - lowest_low;
            if (range == 0.0f) {
                raw_k[i] = 50.0f;  // Neutral when range is zero
            } else {
                raw_k[i] = ((bars[start + i].close - lowest_low) / range) * 100.0f;
            }
        }
    }

    // Compute %D = SMA(%K, d_period) using raw_k values
    float d_sum         = 0.0f;
    int   valid_k_count = 0;

    for (int i = 0; i < N; ++i) {
        if (!std::isnan(raw_k[i])) {
            d_sum += raw_k[i];
            valid_k_count++;

            if (valid_k_count > m_d_period) {
                // Subtract the oldest %K in the D window
                d_sum -= raw_k[i - m_d_period];
            }

            if (i < wu) {
                k_out[i] = INDICATOR_NAN;
                d_out[i] = INDICATOR_NAN;
            } else {
                k_out[i] = raw_k[i];
                if (valid_k_count >= m_d_period) {
                    d_out[i] = d_sum / d_period_f;
                } else {
                    d_out[i] = INDICATOR_NAN;
                }
            }
        } else {
            k_out[i] = INDICATOR_NAN;
            d_out[i] = INDICATOR_NAN;
        }
    }
}
