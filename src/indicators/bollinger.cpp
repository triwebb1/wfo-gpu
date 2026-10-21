#include "bollinger.h"
#include <algorithm>
#include <cassert>
#include <cmath>

BollingerBands::BollingerBands(int period, float std_dev_multiplier)
    : m_period(period), m_multiplier(std_dev_multiplier) {
    assert(period > 0);
    assert(std_dev_multiplier > 0.0f);
}

int BollingerBands::num_series() const {
    return 3;
}

int BollingerBands::warmup_bars() const {
    // Window-based: first valid output at index m_period - 1
    return m_period - 1;
}

const char* BollingerBands::name() const {
    return "BollingerBands";
}

// Optimized from O(N*P) to O(N) using running sum and running sum-of-squares
// for stddev computation: variance = (sum_sq / P) - (sum / P)^2.
// Produces identical output to the previous inner-loop implementation.
void BollingerBands::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = warmup_bars();

    // Series layout: middle [0..N), upper [N..2N), lower [2N..3N)
    float* middle_out = output;
    float* upper_out  = output + N;
    float* lower_out  = output + 2 * N;

    // Running sum and sum-of-squares for O(1) SMA and stddev per bar.
    // Use double precision to avoid catastrophic cancellation when
    // sum_sq ~ sma^2 * period (e.g. ES ~5000 -> sum_sq ~ 500M).
    double running_sum    = 0.0;
    double running_sum_sq = 0.0;

    for (int i = 0; i < N; ++i) {
        const int   bar_idx = start + i;
        const float c       = bars[bar_idx].close;

        running_sum += static_cast<double>(c);
        running_sum_sq += static_cast<double>(c) * static_cast<double>(c);

        // Subtract the element leaving the window
        if (i >= m_period) {
            const double old_c = static_cast<double>(bars[start + i - m_period].close);
            running_sum -= old_c;
            running_sum_sq -= old_c * old_c;
        }

        // SMA valid once we have 'period' bars (index >= period - 1)
        if (i >= m_period - 1) {
            const double period_d = static_cast<double>(m_period);
            const double sma_d    = running_sum / period_d;
            const float  sma      = static_cast<float>(sma_d);

            // Compute stddev using running sums in double precision:
            // variance = E[X^2] - (E[X])^2 = sum_sq/P - (sum/P)^2
            double variance = running_sum_sq / period_d - sma_d * sma_d;
            // Guard against floating-point rounding producing tiny negatives
            if (variance < 0.0) variance = 0.0;
            const float stddev = static_cast<float>(std::sqrt(variance));

            if (i < wu) {
                middle_out[i] = INDICATOR_NAN;
                upper_out[i]  = INDICATOR_NAN;
                lower_out[i]  = INDICATOR_NAN;
            } else {
                middle_out[i] = sma;
                upper_out[i]  = sma + m_multiplier * stddev;
                lower_out[i]  = sma - m_multiplier * stddev;
            }
        } else {
            middle_out[i] = INDICATOR_NAN;
            upper_out[i]  = INDICATOR_NAN;
            lower_out[i]  = INDICATOR_NAN;
        }
    }
}
