#pragma once
// =============================================================================
// Intraday Structure Signals
//
// 10 signals:
//   0: Morning direction        -- 09:30-10:30 up/down/flat
//   1: Morning extreme timing   -- when session extreme was set (bucketed)
//                                  +1 = extreme in first 15 min (strong open drive)
//                                   0 = extreme mid-morning
//                                  -1 = extreme near 10:30 (trending morning)
//   2: Morning volatility collapse -- +1 if range narrows >75% after initial move, 0 otherwise
//   3: Morning trending vs rotational -- +1 trending (HH+HL or LL+LH), -1 rotational, 0 neutral
//   4: Morning first direction  -- +1 if first 15 min moved up, -1 down, 0 flat
//   5: Range consumed %         -- fraction of 10-day ATR used by 11:00 ET -> [0,1]
//   6: Lunch compression        -- +1 if 11:30-13:00 range < 0.3 ATR (compressed), 0 otherwise
//   7: Afternoon direction      -- 12:00-15:00 up/down/flat
//   8: Power hour direction     -- 15:00-15:50 up/down/flat
//   9: Opening drive quality    -- first 15 min magnitude / ATR, signed by direction
//
// Availability: Signals 0-4 available after 10:30.  Signal 5 after 11:00.
// Signal 6 after 13:00.  Signals 7,8 after their respective periods.
// For backtesting we compute all signals at end-of-day and the backtest engine
// gates availability by time.
// =============================================================================

#include "session_signal_base.h"

class IntradayStructureSignals : public SessionSignal {
public:
    // range_consumed_checkpoint_et: seconds since midnight ET for signal 5.
    // Supported checkpoint times: 36000 (10:00), 39600 (11:00), 43200 (12:00), 50400 (14:00).
    // lunch_compression_threshold: fraction of ATR below which lunch range is "compressed" (default 0.3)
    explicit IntradayStructureSignals(int     atr_lookback                 = 10,
                                      int32_t range_consumed_checkpoint_et = 39600,
                                      float   lunch_compression_threshold  = 0.3f);

    void compute(const MarketData& md, int day_start, int day_end, float* output) const override;
    int  num_signals() const override;
    const char* name() const override;
    int         warmup_days() const override;
    bool        suppress_on_halt() const override { return true; }

    // Time-availability for each of the 10 signals:
    //   0: Morning direction        -> 37800 (10:30 ET)
    //   1: Morning extreme timing   -> 37800 (10:30 ET)
    //   2: Morning volatility collapse -> 37800 (10:30 ET)
    //   3: Morning trending/rotational -> 37800 (10:30 ET)
    //   4: Morning first direction  -> 35100 (09:45 ET, first 15 min complete)
    //   5: Range consumed %         -> m_range_consumed_checkpoint_et (parameterized)
    //   6: Lunch compression        -> 46800 (13:00 ET, lunch period complete)
    //   7: Afternoon direction      -> 54000 (15:00 ET, afternoon period complete)
    //   8: Power hour direction     -> 57600 (16:00 ET, RTH close)
    //   9: Opening drive quality    -> 35100 (09:45 ET, first 15 min complete)
    std::vector<int32_t> signal_available_after_et_seconds() const override;

private:
    int     m_atr_lookback;
    int32_t m_range_consumed_checkpoint_et;
    float   m_lunch_compression_threshold;
};
