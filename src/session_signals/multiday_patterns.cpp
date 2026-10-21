#include "multiday_patterns.h"

// =============================================================================
// Multi-Day Pattern Signals
// =============================================================================

MultidayPatternSignals::MultidayPatternSignals(int atr_lookback, float poor_threshold)
    : m_atr_lookback(atr_lookback), m_poor_threshold(poor_threshold) {
    assert(atr_lookback > 0);
    assert(poor_threshold > 0.0f && poor_threshold < 0.5f);
}

int MultidayPatternSignals::num_signals() const {
    return 12;
}
const char* MultidayPatternSignals::name() const {
    return "MultidayPatterns";
}
int MultidayPatternSignals::warmup_days() const {
    return std::max(m_atr_lookback + 1, 8);
}

std::vector<int32_t> MultidayPatternSignals::signal_available_after_et_seconds() const {
    // All multi-day signals use D-1 and earlier data only -> available at session open
    return {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

void MultidayPatternSignals::compute(const MarketData& md, int day_start, int day_end,
                                     float* output) const {
    const int num_days = day_end - day_start;
    const int NS       = num_signals();

    for (int i = 0; i < NS * num_days; ++i) output[i] = SIGNAL_NAN;

    for (int d = day_start; d < day_end; ++d) {
        const int di = d - day_start;

        if (d < 2) continue;  // Need at least 2 prior days for most signals

        SessionDayView today = build_session_view(md, d);
        SessionDayView prev  = build_session_view(md, d - 1);
        if (today.rth_count == 0 || prev.rth_count == 0) continue;

        float atr        = compute_session_atr(md, d, m_atr_lookback);
        float prev_range = prev.rth_range();

        // Signal 0: Inside day (today's range inside yesterday's)
        // Uses prev (D-1) vs D-2 for no-lookahead: is "yesterday" inside "day before"
        if (d >= 2) {
            SessionDayView prev2 = build_session_view(md, d - 2);
            if (prev2.rth_count > 0) {
                bool inside = (prev.rth_high <= prev2.rth_high && prev.rth_low >= prev2.rth_low);
                output[0 * num_days + di] = inside ? 1.0f : 0.0f;
            }
        }

        // Signal 1: Outside day (yesterday's range exceeds day-before's)
        if (d >= 2) {
            SessionDayView prev2 = build_session_view(md, d - 2);
            if (prev2.rth_count > 0) {
                bool outside = (prev.rth_high > prev2.rth_high && prev.rth_low < prev2.rth_low);
                output[1 * num_days + di] = outside ? 1.0f : 0.0f;
            }
        }

        // Signal 2: NR4 (narrowest range in 4 days, looking at D-1..D-4)
        if (d >= 4) {
            bool nr4 = true;
            for (int k = d - 4; k < d - 1; ++k) {
                SessionDayView dv = build_session_view(md, k);
                if (dv.rth_count > 0 && dv.rth_range() <= prev_range) {
                    nr4 = false;
                    break;
                }
            }
            output[2 * num_days + di] = nr4 ? 1.0f : 0.0f;
        }

        // Signal 3: NR7 (narrowest range in 7 days, looking at D-1..D-7)
        if (d >= 7) {
            bool nr7 = true;
            for (int k = d - 7; k < d - 1; ++k) {
                SessionDayView dv = build_session_view(md, k);
                if (dv.rth_count > 0 && dv.rth_range() <= prev_range) {
                    nr7 = false;
                    break;
                }
            }
            output[3 * num_days + di] = nr7 ? 1.0f : 0.0f;
        }

        // Signal 4: Poor high (prev close in upper poor_threshold of prev range -> weak buyers)
        if (prev_range > 0.0f) {
            float close_pos           = (prev.rth_close - prev.rth_low) / prev_range;
            output[4 * num_days + di] = (close_pos >= 1.0f - m_poor_threshold) ? 1.0f : 0.0f;
        }

        // Signal 5: Poor low (prev close in lower poor_threshold of prev range -> weak sellers)
        if (prev_range > 0.0f) {
            float close_pos           = (prev.rth_close - prev.rth_low) / prev_range;
            output[5 * num_days + di] = (close_pos <= m_poor_threshold) ? 1.0f : 0.0f;
        }

        // Signal 6: Running days (consecutive up/down closes)
        {
            int streak = 0;
            int dir    = 0;
            for (int k = d - 1; k >= 1; --k) {
                SessionDayView dv      = build_session_view(md, k);
                SessionDayView dv_prev = build_session_view(md, k - 1);
                if (dv.rth_count == 0 || dv_prev.rth_count == 0) break;

                int this_dir = (dv.rth_close > dv_prev.rth_close)   ? 1
                               : (dv.rth_close < dv_prev.rth_close) ? -1
                                                                    : 0;

                if (this_dir == 0) break;
                if (dir == 0) dir = this_dir;
                if (this_dir != dir) break;
                streak++;
            }
            // Scale: 1 day = 0.2, 5+ days = 1.0
            float scaled              = clamp_signal(static_cast<float>(streak) / 5.0f);
            output[6 * num_days + di] = static_cast<float>(dir) * scaled;
        }

        // Signal 7: 3-day range expansion (D-1, D-2, D-3 show increasing ranges)
        if (d >= 3) {
            SessionDayView d1 = build_session_view(md, d - 1);
            SessionDayView d2 = build_session_view(md, d - 2);
            SessionDayView d3 = build_session_view(md, d - 3);
            if (d1.rth_count > 0 && d2.rth_count > 0 && d3.rth_count > 0) {
                bool expanding =
                    (d1.rth_range() > d2.rth_range() && d2.rth_range() > d3.rth_range());
                output[7 * num_days + di] = expanding ? 1.0f : 0.0f;
            }
        }

        // Signal 8: 3-day range contraction
        if (d >= 3) {
            SessionDayView d1 = build_session_view(md, d - 1);
            SessionDayView d2 = build_session_view(md, d - 2);
            SessionDayView d3 = build_session_view(md, d - 3);
            if (d1.rth_count > 0 && d2.rth_count > 0 && d3.rth_count > 0) {
                bool contracting =
                    (d1.rth_range() < d2.rth_range() && d2.rth_range() < d3.rth_range());
                output[8 * num_days + di] = contracting ? 1.0f : 0.0f;
            }
        }

        // Signal 9: Multiple inside days (2+ consecutive inside days ending at D-1)
        if (d >= 3) {
            int inside_count = 0;
            for (int k = d - 1; k >= 2; --k) {
                SessionDayView dv      = build_session_view(md, k);
                SessionDayView dv_prev = build_session_view(md, k - 1);
                if (dv.rth_count == 0 || dv_prev.rth_count == 0) break;
                if (dv.rth_high <= dv_prev.rth_high && dv.rth_low >= dv_prev.rth_low) {
                    inside_count++;
                } else {
                    break;
                }
            }
            output[9 * num_days + di] = (inside_count >= 2) ? 1.0f : 0.0f;
        }

        // Signal 10: Prev day direction
        if (d >= 2) {
            SessionDayView prev2 = build_session_view(md, d - 2);
            if (prev2.rth_count > 0) {
                output[10 * num_days + di] = direction_signal(prev2.rth_close, prev.rth_close);
            }
        }

        // Signal 11: Prev day range bucket (ATR-relative)
        if (!std::isnan(atr) && atr > 0.0f) {
            float ratio = prev_range / atr;
            float bucket;
            if (ratio < 0.5f)
                bucket = 0.0f;  // small
            else if (ratio < 1.0f)
                bucket = 0.33f;  // normal
            else if (ratio < 1.3f)
                bucket = 0.66f;  // large
            else
                bucket = 1.0f;  // huge
            output[11 * num_days + di] = bucket;
        }
    }
}
