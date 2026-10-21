#pragma once
#include "indicator.h"

// Hull Moving Average (HMA)
//
// Reduces lag by using weighted moving averages of the WMA:
//   HMA(period) = WMA(sqrt(period), 2*WMA(period/2, price) - WMA(period, price))
//
// Steps:
//   1. half_wma = WMA(period/2, close)
//   2. full_wma = WMA(period, close)
//   3. diff = 2*half_wma - full_wma
//   4. HMA = WMA(sqrt(period), diff)
//
// Output: 1 series -- HMA value (price level).
//
// Warmup: period + floor(sqrt(period))
//   The full pipeline needs period bars for the WMAs plus sqrt(period) for
//   the final WMA.
//
// Parameters: period.
class HMA : public Indicator {
public:
    explicit HMA(int period);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int period() const { return m_period; }

private:
    int m_period;

    // Compute WMA on a data array. Writes to out[0..count).
    // out[i] = NaN for i < wma_period - 1, valid thereafter.
    static void compute_wma(const float* data, int count, int wma_period, float* out);
};
