#include "donchian.h"
#include <algorithm>
#include <cassert>
#include <deque>

DonchianChannel::DonchianChannel(int period) : m_period(period) {
    assert(period > 0);
}

int DonchianChannel::num_series() const {
    return 2;
}

int DonchianChannel::warmup_bars() const {
    // Window-based: first valid output at index m_period - 1
    return m_period - 1;
}

const char* DonchianChannel::name() const {
    return "DonchianChannel";
}

// Optimized from O(N*P) to O(N) using monotonic deques for rolling max high
// and rolling min low over the period window. Each element is pushed and
// popped at most once, giving amortized O(1) per bar.
// Produces identical output to the previous inner-loop implementation.
void DonchianChannel::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = warmup_bars();

    // Series layout: upper [0..N), lower [N..2N)
    float* upper_out = output;
    float* lower_out = output + N;

    // Monotonic deques for rolling max of highs and rolling min of lows.
    // Stores indices into [0..N).
    std::deque<int> max_deque;  // front = index of max high in window
    std::deque<int> min_deque;  // front = index of min low in window

    for (int i = 0; i < N; ++i) {
        const float h = bars[start + i].high;
        const float l = bars[start + i].low;

        // Maintain monotonic deque for rolling max of highs
        while (!max_deque.empty() && max_deque.front() <= i - m_period) max_deque.pop_front();
        while (!max_deque.empty() && bars[start + max_deque.back()].high <= h) max_deque.pop_back();
        max_deque.push_back(i);

        // Maintain monotonic deque for rolling min of lows
        while (!min_deque.empty() && min_deque.front() <= i - m_period) min_deque.pop_front();
        while (!min_deque.empty() && bars[start + min_deque.back()].low >= l) min_deque.pop_back();
        min_deque.push_back(i);

        // Valid once we have 'period' bars (index >= period - 1)
        if (i >= m_period - 1) {
            if (i < wu) {
                upper_out[i] = INDICATOR_NAN;
                lower_out[i] = INDICATOR_NAN;
            } else {
                upper_out[i] = bars[start + max_deque.front()].high;
                lower_out[i] = bars[start + min_deque.front()].low;
            }
        } else {
            upper_out[i] = INDICATOR_NAN;
            lower_out[i] = INDICATOR_NAN;
        }
    }
}
