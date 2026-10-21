#include "hma.h"
#include <cassert>
#include <cmath>
#include <vector>

HMA::HMA(int period) : m_period(period) {
    assert(period >= 2);  // Need period >= 2 for period/2 to be meaningful
}

int HMA::num_series() const {
    return 1;
}

int HMA::warmup_bars() const {
    // Full pipeline: period bars for the initial WMAs, plus floor(sqrt(period))
    // for the final WMA on the diff series.
    return m_period + static_cast<int>(std::floor(std::sqrt(static_cast<float>(m_period))));
}

const char* HMA::name() const {
    return "HMA";
}

// Compute WMA (Weighted Moving Average) on a data array.
// WMA(n) = sum(weight_i * data_i) / sum(weights), weights = 1,2,3,...,n
// Output is NaN for indices < wma_period - 1.
void HMA::compute_wma(const float* data, int count, int wma_period, float* out) {
    assert(wma_period > 0);

    // Weight denominator = n*(n+1)/2
    const float denom = static_cast<float>(wma_period) * static_cast<float>(wma_period + 1) / 2.0f;

    for (int i = 0; i < count; ++i) {
        if (i < wma_period - 1) {
            out[i] = INDICATOR_NAN;
            continue;
        }

        // Check if any input in the window is NaN
        bool  has_nan = false;
        float sum     = 0.0f;
        for (int j = 0; j < wma_period; ++j) {
            float val = data[i - wma_period + 1 + j];
            if (std::isnan(val)) {
                has_nan = true;
                break;
            }
            // Weight: j+1 (oldest gets weight 1, newest gets weight wma_period)
            sum += static_cast<float>(j + 1) * val;
        }

        if (has_nan) {
            out[i] = INDICATOR_NAN;
        } else {
            out[i] = sum / denom;
        }
    }
}

void HMA::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu          = warmup_bars();
    const int half_period = m_period / 2;
    const int sqrt_period = static_cast<int>(std::floor(std::sqrt(static_cast<float>(m_period))));

    // Ensure sqrt_period is at least 1
    const int final_period = (sqrt_period >= 1) ? sqrt_period : 1;

    // Step 1: Extract close prices
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = bars[start + i].close;
    }

    // Step 2: Compute WMA(period/2, close)
    std::vector<float> half_wma(N);
    compute_wma(closes.data(), N, half_period, half_wma.data());

    // Step 3: Compute WMA(period, close)
    std::vector<float> full_wma(N);
    compute_wma(closes.data(), N, m_period, full_wma.data());

    // Step 4: diff = 2 * half_wma - full_wma
    std::vector<float> diff(N);
    for (int i = 0; i < N; ++i) {
        if (std::isnan(half_wma[i]) || std::isnan(full_wma[i])) {
            diff[i] = INDICATOR_NAN;
        } else {
            diff[i] = 2.0f * half_wma[i] - full_wma[i];
        }
    }

    // Step 5: HMA = WMA(sqrt(period), diff)
    std::vector<float> hma_raw(N);
    compute_wma(diff.data(), N, final_period, hma_raw.data());

    // Step 6: Apply warmup masking
    for (int i = 0; i < N; ++i) {
        if (i < wu) {
            output[i] = INDICATOR_NAN;
        } else {
            output[i] = hma_raw[i];
        }
    }
}
