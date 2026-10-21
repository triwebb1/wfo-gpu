#pragma once
#include "indicator.h"

// ADX (Average Directional Index)
//
// Computation:
//   +DM = max(0, high[i] - high[i-1])       if > max(0, low[i-1] - low[i]), else 0
//   -DM = max(0, low[i-1] - low[i])          if > max(0, high[i] - high[i-1]), else 0
//   TR = max(high-low, |high-prev_close|, |low-prev_close|)
//   Smooth +DM, -DM, TR using Wilder smoothing over 'period'
//   +DI = (+DM_smooth / TR_smooth) * 100
//   -DI = (-DM_smooth / TR_smooth) * 100
//   DX = |+DI - -DI| / (+DI + -DI) * 100
//   ADX = Wilder smooth of DX over 'period'
//
// Output: 3 series -- ADX, +DI, -DI.
//
// Warmup: 4 * period.
//
// Parameters: period.
class ADX : public Indicator {
public:
    explicit ADX(int period);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int period() const { return m_period; }

private:
    int m_period;
};
