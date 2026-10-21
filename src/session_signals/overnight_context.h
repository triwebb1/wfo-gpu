#pragma once
// =============================================================================
// Overnight (ETH) Context Signals
//
// 7 signals:
//   0: ETH direction           -- +1 up, -1 down, 0 flat
//   1: ETH range (ATR-relative)-- overnight range / daily ATR, clamped to [0,1]
//   2: ETH closed near extreme -- +1 near high (bullish), -1 near low (bearish), 0 mid
//   3: ETH new high late       -- +1 if session high in last 2 hours of ETH (bullish), 0 no
//   4: ETH trend steady        -- +1 if quiet overnight (<0.5 ATR range), 0 otherwise
//   5: RTH opens above/below ETH VWAP -- +1 above, -1 below, 0 at VWAP
//   6: RTH opens above/below ETH range -- +1 above, -1 below, 0 inside
//
// Availability: Available at RTH open.
// No lookahead: uses only ETH portion of current session day.
// =============================================================================

#include "session_signal_base.h"

class OvernightContextSignals : public SessionSignal {
public:
    explicit OvernightContextSignals(int atr_lookback = 10);

    void compute(const MarketData& md, int day_start, int day_end, float* output) const override;
    int  num_signals() const override;
    const char* name() const override;
    int         warmup_days() const override;

    // Time-availability: all 7 signals use ETH (overnight) data which
    // completes before RTH open.  Available at RTH open (09:30 ET).
    std::vector<int32_t> signal_available_after_et_seconds() const override;

private:
    int m_atr_lookback;
};
