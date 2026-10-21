#include "opening_range.h"

// =============================================================================
// Opening Range (OR) & Initial Balance (IB) Signals
// =============================================================================

OpeningRangeSignals::OpeningRangeSignals(int atr_lookback, int or_period_minutes,
                                         float false_breakout_threshold, float ib_tight_threshold)
    : m_atr_lookback(atr_lookback),
      m_false_breakout_threshold(false_breakout_threshold),
      m_ib_tight_threshold(ib_tight_threshold),
      m_or_end_time(session_time::RTH_OPEN + or_period_minutes * 60) {
    assert(atr_lookback > 0);
    assert(or_period_minutes == 15 || or_period_minutes == 30);
    assert(false_breakout_threshold > 0.0f && false_breakout_threshold <= 1.0f);
    assert(ib_tight_threshold > 0.0f && ib_tight_threshold <= 1.0f);
}

int OpeningRangeSignals::num_signals() const {
    return 8;
}
const char* OpeningRangeSignals::name() const {
    return "OpeningRange";
}
int OpeningRangeSignals::warmup_days() const {
    return m_atr_lookback + 1;
}

std::vector<int32_t> OpeningRangeSignals::signal_available_after_et_seconds() const {
    return {
        m_or_end_time,  // 0: OR direction        (parameterized OR end)
        m_or_end_time,  // 1: OR range size       (parameterized OR end)
        session_time::
            RTH_CLOSE,  // 2: OR breakout         (57600 = 16:00, retrospective -- scans full RTH)
        session_time::
            RTH_CLOSE,  // 3: OR false breakout   (57600 = 16:00, retrospective -- scans full RTH)
        session_time::IB_END,  // 4: IB direction        (37800 = 10:30)
        session_time::IB_END,  // 5: IB range size       (37800 = 10:30)
        session_time::
            RTH_CLOSE,  // 6: IB tight-and-extend (57600 = 16:00, retrospective -- scans full RTH)
        session_time::
            RTH_CLOSE  // 7: IB extension dir    (57600 = 16:00, retrospective -- scans full RTH)
    };
}

void OpeningRangeSignals::compute(const MarketData& md, int day_start, int day_end,
                                  float* output) const {
    const int num_days = day_end - day_start;
    const int NS       = num_signals();

    // Initialize all to NaN
    for (int i = 0; i < NS * num_days; ++i) output[i] = SIGNAL_NAN;

    for (int d = day_start; d < day_end; ++d) {
        const int di = d - day_start;  // day offset into output

        SessionDayView view = build_session_view(md, d);
        if (view.rth_count == 0) continue;

        float atr = compute_session_atr(md, d, m_atr_lookback);
        if (std::isnan(atr) || atr <= 0.0f) continue;

        // --- OR: parameterized opening range (09:30 - OR end) ---
        auto [or_first, or_count] =
            bars_in_time_range(md, view, session_time::RTH_OPEN, m_or_end_time);

        float or_high = -1e30f, or_low = 1e30f;
        float or_open = SIGNAL_NAN, or_close = SIGNAL_NAN;

        if (or_count > 0) {
            or_open  = md.bar(or_first).open;
            or_close = md.bar(or_first + or_count - 1).close;
            for (int i = or_first; i < or_first + or_count; ++i) {
                or_high = std::max(or_high, md.bar(i).high);
                or_low  = std::min(or_low, md.bar(i).low);
            }
        } else {
            continue;  // No OR bars -> can't compute
        }

        float or_range = or_high - or_low;

        // Signal 0: OR direction
        output[0 * num_days + di] = direction_signal(or_open, or_close, atr * 0.02f);

        // Signal 1: OR range size (as fraction of ATR, clamped to [0,1])
        output[1 * num_days + di] = clamp_signal(or_range / atr);

        // --- IB: first 60 minutes (09:30 - 10:30 ET) ---
        auto [ib_first, ib_count] =
            bars_in_time_range(md, view, session_time::RTH_OPEN, session_time::IB_END);

        float ib_high = -1e30f, ib_low = 1e30f;
        float ib_open = SIGNAL_NAN, ib_close = SIGNAL_NAN;

        if (ib_count > 0) {
            ib_open  = md.bar(ib_first).open;
            ib_close = md.bar(ib_first + ib_count - 1).close;
            for (int i = ib_first; i < ib_first + ib_count; ++i) {
                ib_high = std::max(ib_high, md.bar(i).high);
                ib_low  = std::min(ib_low, md.bar(i).low);
            }
        } else {
            continue;
        }

        float ib_range = ib_high - ib_low;

        // Signal 4: IB direction
        output[4 * num_days + di] = direction_signal(ib_open, ib_close, atr * 0.02f);

        // Signal 5: IB range size
        output[5 * num_days + di] = clamp_signal(ib_range / atr);

        // --- Post-IB analysis: bars after 10:30 ---
        auto [post_first, post_count] =
            bars_in_time_range(md, view, session_time::IB_END, session_time::RTH_CLOSE);

        // Signal 2: OR breakout (using bars after OR through end of RTH)
        {
            auto [post_or_first, post_or_count] =
                bars_in_time_range(md, view, m_or_end_time, session_time::RTH_CLOSE);
            float breakout = 0.0f;
            for (int i = post_or_first; i < post_or_first + post_or_count && post_or_count > 0;
                 ++i) {
                if (md.bar(i).high > or_high) {
                    breakout = 1.0f;
                    break;
                }
                if (md.bar(i).low < or_low) {
                    breakout = -1.0f;
                    break;
                }
            }
            output[2 * num_days + di] = breakout;
        }

        // Signal 3: OR false breakout (price breaks then reverses back inside)
        {
            auto [post_or_first, post_or_count] =
                bars_in_time_range(md, view, m_or_end_time, session_time::RTH_CLOSE);
            float false_breakout = 0.0f;
            bool  broke_high = false, broke_low = false;
            for (int i = post_or_first; i < post_or_first + post_or_count && post_or_count > 0;
                 ++i) {
                if (md.bar(i).high > or_high) broke_high = true;
                if (md.bar(i).low < or_low) broke_low = true;

                // Check for reversal: if we broke high but close back in range
                if (broke_high &&
                    md.bar(i).close < or_high - or_range * m_false_breakout_threshold) {
                    false_breakout = -1.0f;  // false bull break -> bearish
                    break;
                }
                if (broke_low && md.bar(i).close > or_low + or_range * m_false_breakout_threshold) {
                    false_breakout = 1.0f;  // false bear break -> bullish
                    break;
                }
            }
            output[3 * num_days + di] = false_breakout;
        }

        // Signal 6: IB tight-and-extend
        {
            bool  ib_tight   = (ib_range < atr * m_ib_tight_threshold);
            float extend_dir = 0.0f;
            if (ib_tight && post_count > 0) {
                for (int i = post_first; i < post_first + post_count; ++i) {
                    if (md.bar(i).high > ib_high) {
                        extend_dir = 1.0f;
                        break;
                    }
                    if (md.bar(i).low < ib_low) {
                        extend_dir = -1.0f;
                        break;
                    }
                }
            }
            output[6 * num_days + di] = extend_dir;
        }

        // Signal 7: IB extension direction (first break of IB range after 10:30)
        {
            float ext_dir = 0.0f;
            if (post_count > 0) {
                for (int i = post_first; i < post_first + post_count; ++i) {
                    if (md.bar(i).high > ib_high) {
                        ext_dir = 1.0f;
                        break;
                    }
                    if (md.bar(i).low < ib_low) {
                        ext_dir = -1.0f;
                        break;
                    }
                }
            }
            output[7 * num_days + di] = ext_dir;
        }
    }
}
