#pragma once
// =============================================================================
// Multi-Day Pattern Signals
//
// 12 signals:
//   0:  Inside day             -- +1 if today's range inside yesterday's, 0 otherwise
//   1:  Outside day            -- +1 if today's range exceeds yesterday's, 0 otherwise
//   2:  NR4                    -- +1 if narrowest range in 4 days, 0 otherwise
//   3:  NR7                    -- +1 if narrowest range in 7 days, 0 otherwise
//   4:  Poor high              -- +1 if prev close in upper 25% of prev range (weak buyers)
//   5:  Poor low               -- +1 if prev close in lower 25% of prev range (weak sellers)
//   6:  Running days           -- consecutive up closes: +clamp(count/5), down: -clamp(count/5)
//   7:  3-day range expansion  -- +1 if last 3 days show expanding ranges, 0 otherwise
//   8:  3-day range contraction-- +1 if last 3 days show contracting ranges, 0 otherwise
//   9:  Multiple inside days   -- +1 if 2+ consecutive inside days ending yesterday
//  10:  Prev day direction     -- +1 up, -1 down, 0 flat
//  11:  Prev day range bucket  -- 0=small, 0.33=normal, 0.66=large, 1.0=huge
//       Prev day close position and trending/choppy are folded into existing signals
//       (poor high/low covers close position; running days covers trend/chop)
//
// Availability: Available at start of session (uses only prior day data).
// No lookahead: all use D-1 and earlier.
// =============================================================================

#include "session_signal_base.h"

class MultidayPatternSignals : public SessionSignal {
public:
    // poor_threshold: close position fraction for poor high/low detection (default 0.25)
    //   Poor high = close in upper poor_threshold of range (weak buyers)
    //   Poor low  = close in lower poor_threshold of range (weak sellers)
    explicit MultidayPatternSignals(int atr_lookback = 10, float poor_threshold = 0.25f);

    void compute(const MarketData& md, int day_start, int day_end, float* output) const override;
    int  num_signals() const override;
    const char* name() const override;
    int         warmup_days() const override;

    // Time-availability: all 12 signals use only prior-day (D-1) data,
    // so they are available at session open (0 = always available).
    std::vector<int32_t> signal_available_after_et_seconds() const override;

private:
    int   m_atr_lookback;
    float m_poor_threshold;
};
