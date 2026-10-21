#pragma once
// =============================================================================
// Gap Analysis Signals
//
// 6 signals:
//   0: Gap direction       -- RTH open vs prev RTH close: +1 up, -1 down, 0 flat
//   1: Gap size bucket     -- 0.0=tiny, 0.33=small, 0.66=medium, 1.0=large (magnitude)
//                             sign matches gap direction
//   2: Gap fill timing     -- +1 if gap filled by 11:00, 0.5 by 10:30, 0.25 by 10:00,
//                             0 if not filled. Sign: +1 means bullish reversal (gap down filled)
//   3: Gap into prev VA   -- +1 gap lands inside prev day range (value area), 0 outside
//   4: Gap in trend dir   -- +1 gap matches 3-day directional streak, -1 against, 0 no streak
//   5: Gap vs ETH VWAP    -- RTH open above ETH VWAP (+1) or below (-1), 0 if no ETH data
//
// Availability: Available after RTH open (but fill timing signal updates through 11:00).
// No lookahead: prev day data only.
// =============================================================================

#include "session_signal_base.h"

class GapAnalysisSignals : public SessionSignal {
public:
    explicit GapAnalysisSignals(int atr_lookback = 10);

    void compute(const MarketData& md, int day_start, int day_end, float* output) const override;
    int  num_signals() const override;
    const char* name() const override;
    int         warmup_days() const override;

    // Time-availability for each of the 6 signals:
    //   0: Gap direction       -> 34200 (09:30 ET, available at RTH open)
    //   1: Gap size bucket     -> 34200 (09:30 ET)
    //   2: Gap fill timing     -> 39600 (11:00 ET, fill scans through 11:00)
    //   3: Gap into prev VA   -> 34200 (09:30 ET)
    //   4: Gap in trend dir   -> 34200 (09:30 ET, uses prior day data only)
    //   5: Gap vs ETH VWAP    -> 34200 (09:30 ET, uses ETH data only)
    std::vector<int32_t> signal_available_after_et_seconds() const override;

private:
    int m_atr_lookback;
};
