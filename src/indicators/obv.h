#pragma once
#include "indicator.h"

// OBV (On Balance Volume)
//
// Computation:
//   If close > prev_close: OBV += volume
//   If close < prev_close: OBV -= volume
//   If close == prev_close: OBV unchanged
//
//   Optionally smoothed with SMA(smoothing_period) of OBV.
//   When smoothing_period > 0, output is the smoothed OBV.
//   When smoothing_period == 0, output is raw OBV.
//
// Output: 1 series -- OBV (raw or smoothed).
//
// Warmup: smoothing_period (for smoothed version), 0 for raw.
//
// Parameters: smoothing_ma_period (0 = raw OBV).
class OBV : public Indicator {
public:
    explicit OBV(int smoothing_ma_period = 0);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int smoothing_period() const { return m_smoothing_period; }

private:
    int m_smoothing_period;
};
