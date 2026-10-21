#pragma once
// =============================================================================
// Calendar & Timing Signals
//
// 4 signals:
//   0: Day of week     -- Mon=-1, Tue=-0.5, Wed=0, Thu=+0.5, Fri=+1
//   1: Week of month   -- scaled linearly: week 1=-1, week 5=+1
//   2: OPEX week       -- +1 if 3rd Friday week, 0 otherwise
//   3: Quarter signal  -- Q1=-1, Q2=-0.5, Q3=+0.5, Q4=+1
//
// Availability: Available at session start.
// No lookahead: purely calendar-derived.
// =============================================================================

#include "session_signal_base.h"

class CalendarSignals : public SessionSignal {
public:
    CalendarSignals() = default;

    void compute(const MarketData& md, int day_start, int day_end, float* output) const override;
    int  num_signals() const override;
    const char* name() const override;
    int         warmup_days() const override;

    // Time-availability: all 4 signals are purely calendar-derived,
    // available at session open (0 = always available).
    std::vector<int32_t> signal_available_after_et_seconds() const override;
};
