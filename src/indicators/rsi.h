#pragma once
#include "indicator.h"

// RSI (Relative Strength Index) using Wilder's smoothing method.
//
// Computation:
//   gain = max(0, close[i] - close[i-1])
//   loss = max(0, close[i-1] - close[i])
//   First 'period' bars: avg_gain = mean(gains), avg_loss = mean(losses)
//   Subsequent: avg_gain = (avg_gain * (period-1) + gain) / period  (Wilder smoothing)
//   RS = avg_gain / avg_loss  (if avg_loss == 0 -> RSI = 100)
//   RSI = 100 - 100 / (1 + RS)
//
// Output: 1 series -- RSI values (0-100 range).
//
// Warmup: 3 * period (Wilder smoothing).
//
// Parameters: period, overbought threshold, oversold threshold.
// Thresholds are stored but used by the backtest engine for signal generation,
// not by this indicator.
class RSI : public Indicator {
public:
    RSI(int period, float overbought = 70.0f, float oversold = 30.0f);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    // Accessors for thresholds (used by backtest engine)
    float overbought() const { return m_overbought; }
    float oversold() const { return m_oversold; }
    int   period() const { return m_period; }

private:
    int   m_period;
    float m_overbought;
    float m_oversold;
};
