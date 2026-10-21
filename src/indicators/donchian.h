#pragma once
#include "indicator.h"

// Donchian Channel
//
// Computation:
//   upper = highest high over past 'period' bars
//   lower = lowest low over past 'period' bars
//
// Output: 2 series -- upper, lower.
//
// Warmup: period (window-based).
//
// Parameters: period.
class DonchianChannel : public Indicator {
public:
    explicit DonchianChannel(int period);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int period() const { return m_period; }

private:
    int m_period;
};
