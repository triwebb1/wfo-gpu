#pragma once
#include "indicator.h"

// EMA/SMA Crossover Indicator
//
// Computes two moving averages (fast and slow) over bar close prices.
// Output: 2 series -- fast MA values and slow MA values.
// Signal generation (cross detection, [-1,+1] mapping) is handled by
// the backtest engine, not here.
//
// EMA: ema[0] = close[0], ema[i] = alpha * close[i] + (1-alpha) * ema[i-1]
//      where alpha = 2.0 / (period + 1)
// SMA: sma[i] = mean(close[i-period+1 .. i]) for i >= period-1
//      Uses running sum for O(1) per bar.
//
// Warmup: EMA = 3 * period, SMA = period. Cross = max(fast_warmup, slow_warmup).
class EMASMACross : public Indicator {
public:
    EMASMACross(MAType fast_type, int fast_period, MAType slow_type, int slow_period);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

private:
    MAType m_fast_type;
    int    m_fast_period;
    MAType m_slow_type;
    int    m_slow_period;

    // Compute a single MA series into dst[0..N) from bars[start..end)
    void compute_ma(const Bar* bars, int start, int end, MAType type, int period, float* dst) const;

    // Warmup for a single MA component
    static int ma_warmup(MAType type, int period);
};
