#pragma once
// =============================================================================
// Volatility Regime Signals
//
// 4 signals:
//   0: ATR ratio (5d vs 20d)              -- >1 expanding (+1 side), <1 contracting (-1 side)
//                                            scaled: clamp((ratio - 1.0) * 2, -1, 1)
//   1: Prev day range expansion           -- +1 if prev day range > 1.1x ATR, 0 otherwise
//   2: Prev day range contraction         -- +1 if prev day range < 0.7x ATR, 0 otherwise
//   3: Consecutive expansion/contraction  -- +clamp(streak/5) for expansion streaks,
//                                            -clamp(streak/5) for contraction streaks
//
// Availability: Available at session start.
// No lookahead: uses D-1 and earlier data only.
// =============================================================================

#include "session_signal_base.h"

class VolatilityRegimeSignals : public SessionSignal {
public:
    explicit VolatilityRegimeSignals(int atr_short_lookback = 5, int atr_long_lookback = 20,
                                     float expansion_threshold   = 1.1f,
                                     float contraction_threshold = 0.7f);

    void compute(const MarketData& md, int day_start, int day_end, float* output) const override;
    int  num_signals() const override;
    const char* name() const override;
    int         warmup_days() const override;

    // Time-availability: all 4 signals use D-1 and earlier data only,
    // available at session open (0 = always available).
    std::vector<int32_t> signal_available_after_et_seconds() const override;

private:
    int   m_atr_short_lookback;
    int   m_atr_long_lookback;
    float m_expansion_threshold;
    float m_contraction_threshold;
};
