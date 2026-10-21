#pragma once
#include "indicator.h"

// ATR (Average True Range) using Wilder's smoothing.
//
// Computation:
//   True Range = max(high - low, |high - prev_close|, |low - prev_close|)
//   First bar: TR = high - low (no previous close available)
//   First 'period' bars: ATR = mean(TR[0..period-1])
//   Subsequent: ATR = (ATR * (period-1) + TR) / period  (Wilder smoothing)
//
// Output: 1 series -- ATR values.
//
// Warmup: 3 * period (Wilder smoothing).
//
// Parameters: period.
class ATR : public Indicator {
public:
    explicit ATR(int period);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int period() const { return m_period; }

private:
    int m_period;
};
