#pragma once
#include "indicator.h"

// Stochastic Oscillator
//
// Computation:
//   %K = (close - lowest_low(K_period)) / (highest_high(K_period) - lowest_low(K_period)) * 100
//   %D = SMA(%K, D_period)
//
// Output: 2 series -- %K, %D.
//
// Warmup: K_period + D_period.
//
// Parameters: k_period, d_period.
class Stochastic : public Indicator {
public:
    Stochastic(int k_period, int d_period);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int k_period() const { return m_k_period; }
    int d_period() const { return m_d_period; }

private:
    int m_k_period;
    int m_d_period;
};
