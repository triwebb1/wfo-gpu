#pragma once
#include "indicator.h"

// MACD (Moving Average Convergence Divergence)
//
// Computation:
//   fast_ema = EMA(close, fast_period)
//   slow_ema = EMA(close, slow_period)
//   MACD line = fast_ema - slow_ema
//   Signal line = EMA(MACD line, signal_period)
//   Histogram = MACD line - Signal line
//
// Output: 3 series -- MACD line, signal line, histogram.
//
// Warmup: 3 * slow_period (exponentially-weighted).
//
// Parameters: fast_period, slow_period, signal_period.
class MACD : public Indicator {
public:
    MACD(int fast_period, int slow_period, int signal_period);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int fast_period() const { return m_fast_period; }
    int slow_period() const { return m_slow_period; }
    int signal_period() const { return m_signal_period; }

private:
    int m_fast_period;
    int m_slow_period;
    int m_signal_period;
};
