#pragma once
#include "indicator.h"

// Connors RSI -- composite oscillator combining three components:
//
//   1. RSI(close, rsi_period) -- short-term RSI of closing prices
//   2. StreakRSI: RSI(streak, streak_rsi_period) -- RSI of the up/down streak
//      length (consecutive up or down closes)
//   3. PercentRank: percentile rank of current close change over lookback period
//
// ConnorsRSI = (RSI + StreakRSI + PercentRank) / 3
//
// Output: 1 series -- ConnorsRSI value (0-100 range, like regular RSI).
//
// Warmup: max(3*rsi_period, 3*streak_rsi_period, percentile_period)
//   The RSI components use Wilder smoothing (3x period), and PercentRank
//   needs percentile_period bars of history.
//
// Parameters: rsi_period (default 3), streak_rsi_period (default 2),
//             percentile_period (default 100).
class ConnorsRSI : public Indicator {
public:
    ConnorsRSI(int rsi_period = 3, int streak_rsi_period = 2, int percentile_period = 100);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    int rsi_period() const { return m_rsi_period; }
    int streak_rsi_period() const { return m_streak_rsi_period; }
    int percentile_period() const { return m_percentile_period; }

private:
    int m_rsi_period;
    int m_streak_rsi_period;
    int m_percentile_period;

    // Compute Wilder-smoothed RSI on a data array (not necessarily close prices).
    // data[0..count) -> rsi_out[0..count).  rsi_out[i] = NaN until initialized.
    static void compute_rsi_on_data(const float* data, int count, int period, float* rsi_out);
};
