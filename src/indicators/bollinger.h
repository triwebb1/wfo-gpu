#pragma once
#include "indicator.h"

// Bollinger Bands
//
// Computation:
//   middle = SMA(close, period)
//   stddev = sqrt(mean((close[i] - SMA)^2) over the period window)
//   upper = middle + multiplier * stddev
//   lower = middle - multiplier * stddev
//
// Output: 3 series -- middle (SMA), upper, lower.
//
// Warmup: period (window-based).
//
// Parameters: period, std_dev_multiplier.
class BollingerBands : public Indicator {
public:
    BollingerBands(int period, float std_dev_multiplier = 2.0f);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int   period() const { return m_period; }
    float multiplier() const { return m_multiplier; }

private:
    int   m_period;
    float m_multiplier;
};
