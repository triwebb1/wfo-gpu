#pragma once
#include "indicator.h"

// CCI (Commodity Channel Index)
//
// Computation:
//   typical_price = (high + low + close) / 3
//   CCI = (tp - SMA(tp, period)) / (0.015 * mean_deviation)
//   mean_deviation = mean(|tp[i] - SMA(tp)|) over the period window
//
// Output: 1 series -- CCI value (unbounded, typically -200 to +200).
//
// Warmup: period (window-based).
//
// Parameters: period.
class CCI : public Indicator {
public:
    explicit CCI(int period);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int period() const { return m_period; }

private:
    int m_period;
};
