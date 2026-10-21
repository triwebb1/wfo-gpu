#pragma once
#include "indicator.h"

// ROC (Rate of Change)
//
// Computation:
//   ROC = (close - close[period_ago]) / close[period_ago] * 100
//   Optionally smoothed with EMA-3 or EMA-5.
//
// Output: 1 series (raw ROC) or 2 series (raw + smoothed) when smoothing > 0.
//
// Warmup: period (window-based). If smoothed, add 3*smoothing_period.
//
// Parameters: period, smoothing_period (0 = none, 3 = EMA-3, 5 = EMA-5).
class ROC : public Indicator {
public:
    ROC(int period, int smoothing_period = 0);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int period() const { return m_period; }
    int smoothing_period() const { return m_smoothing_period; }

private:
    int m_period;
    int m_smoothing_period;  // 0 = none, 3 = EMA-3, 5 = EMA-5
};
