#pragma once
// =============================================================================
// Level Interaction (Bounce/Break) Signals
//
// 7 signals:
//   0: Bounced prev RTH high   -- +1 if price touched & reversed from prev high (bearish reject)
//                                  -1 if broke through (bullish break), 0 if not tested
//   1: Bounced prev RTH low    -- -1 if price touched & reversed from prev low (bullish reject)
//                                  +1 if broke through (bearish break), 0 if not tested
//   2: Bounced prev POC        -- +1 bounce up from POC, -1 bounce down, 0 not tested
//      POC = price with highest volume in prev session (volume profile, see compute_poc)
//   3: Bounced long EMA        -- +1 bounce up from 233-bar EMA, -1 bounce down, 0 not tested
//   4: Bounced VWAP            -- +1 bounce up from VWAP, -1 bounce down, 0 not tested
//   5: Prev high as support    -- +1 if price holds above prev high (support), 0 otherwise
//   6: Prev low as resistance  -- -1 if price holds below prev low (resistance), 0 otherwise
//
// Availability:
//   Bounce/break (0-4): available after RTH_CLOSE (16:00 ET) -- retrospective
//     signals that scan all RTH bars for level interaction with 2-bar confirm.
//   Support/resistance (5-6): available after RTH_CLOSE (16:00 ET) -- use
//     full-day rth_high/rth_low which aren't final until session close.
// No lookahead: prev day levels, current day values available next session.
// =============================================================================

#include "session_signal_base.h"

class LevelInteractionSignals : public SessionSignal {
public:
    // touch_ticks: how close price must get to a level to count as "touching"
    // tick_size: instrument minimum price increment (0.25 for ES/NQ)
    // ema_period: number of session closes for the long EMA (default 20 sessions
    //             approximates a ~233 bar EMA on 30s bars: 20 sessions * ~390 min
    //             / 0.5 min = ~15600 bars, but we use session closes as the
    //             EMA input so 20 is the appropriate period for that time scale)
    explicit LevelInteractionSignals(float touch_ticks = 2.0f, float tick_size = 0.25f,
                                     int ema_period = 20);

    void compute(const MarketData& md, int day_start, int day_end, float* output) const override;
    int  num_signals() const override;
    const char* name() const override;
    int         warmup_days() const override;
    bool        suppress_on_halt() const override { return true; }

    // Time-availability: all signals are retrospective (scan full RTH session),
    // so they are gated at RTH_CLOSE and available for the next session only.
    //   0-4: Bounce/break signals  -> 57600 (16:00 ET, retrospective -- scans full RTH)
    //   5-6: Support/resistance    -> 57600 (16:00 ET, retrospective -- uses full-day OHLC)
    std::vector<int32_t> signal_available_after_et_seconds() const override;

private:
    float m_touch_ticks;
    float m_tick_size;
    int   m_ema_period;
};
