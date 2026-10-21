#pragma once
// =============================================================================
// Opening Range (OR) & Initial Balance (IB) Signals
//
// 8 signals:
//   0: OR direction        -- first 30 min up/down/flat -> {-1, 0, +1}
//   1: OR range size       -- OR range as fraction of N-day ATR -> [0, 1]
//   2: OR breakout         -- price breaks above(+1) / below(-1) OR range, 0 if inside
//   3: OR false breakout   -- breaks then reverses: +1 if false bear break (bullish),
//                             -1 if false bull break (bearish), 0 otherwise
//   4: IB direction        -- first 60 min up/down/flat -> {-1, 0, +1}
//   5: IB range size       -- IB range as fraction of N-day ATR -> [0, 1]
//   6: IB tight-and-extend -- IB was tight (<0.5 ATR) then extended: +1 up, -1 down, 0 no
//   7: IB extension dir    -- first extension direction after IB end: +1 up, -1 down, 0 none
//
// Availability:
//   OR direction, range size (0-1): available after OR_END (10:00 ET)
//   OR breakout, false breakout (2-3): available after RTH_CLOSE (16:00 ET)
//     -- retrospective signals that scan OR_END through RTH_CLOSE
//   IB direction, range size (4-5): available after IB_END (10:30 ET)
//   IB tight-and-extend, extension dir (6-7): available after RTH_CLOSE (16:00 ET)
//     -- retrospective signals that scan IB_END through RTH_CLOSE
// No lookahead: ATR uses prior days only. Breakout/extension signals scan
// the full post-OR/IB period and are only available for the next session.
// =============================================================================

#include "session_signal_base.h"

class OpeningRangeSignals : public SessionSignal {
public:
    // atr_lookback: number of prior days for ATR normalization (default 10)
    // or_period_minutes: opening range duration (15 or 30 min, default 30)
    // false_breakout_threshold: fraction of OR range for false breakout reversal (default 0.25)
    // ib_tight_threshold: fraction of ATR below which IB is considered "tight" (default 0.5)
    explicit OpeningRangeSignals(int atr_lookback = 10, int or_period_minutes = 30,
                                 float false_breakout_threshold = 0.25f,
                                 float ib_tight_threshold       = 0.5f);

    void compute(const MarketData& md, int day_start, int day_end, float* output) const override;
    int  num_signals() const override;
    const char* name() const override;
    int         warmup_days() const override;

    // Time-availability for each of the 8 signals:
    //   0: OR direction        -> OR end time (parameterized by or_period_minutes)
    //   1: OR range size       -> OR end time
    //   2: OR breakout         -> 57600 (16:00 ET, retrospective -- scans full RTH)
    //   3: OR false breakout   -> 57600 (16:00 ET, retrospective -- scans full RTH)
    //   4: IB direction        -> 37800 (10:30 ET, IB period complete)
    //   5: IB range size       -> 37800 (10:30 ET)
    //   6: IB tight-and-extend -> 57600 (16:00 ET, retrospective -- scans full RTH)
    //   7: IB extension dir    -> 57600 (16:00 ET, retrospective -- scans full RTH)
    std::vector<int32_t> signal_available_after_et_seconds() const override;

private:
    int     m_atr_lookback;
    float   m_false_breakout_threshold;
    float   m_ib_tight_threshold;
    int32_t m_or_end_time;  // computed: RTH_OPEN + or_period_minutes * 60
};
