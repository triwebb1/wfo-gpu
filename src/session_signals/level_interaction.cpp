#include "level_interaction.h"

// =============================================================================
// Level Interaction (Bounce/Break) Signals
// =============================================================================

LevelInteractionSignals::LevelInteractionSignals(float touch_ticks, float tick_size, int ema_period)
    : m_touch_ticks(touch_ticks), m_tick_size(tick_size), m_ema_period(ema_period) {
    assert(touch_ticks >= 0.0f);
    assert(tick_size > 0.0f);
    assert(ema_period > 0);
}

int LevelInteractionSignals::num_signals() const {
    return 7;
}
const char* LevelInteractionSignals::name() const {
    return "LevelInteraction";
}
int LevelInteractionSignals::warmup_days() const {
    return std::max(3, m_ema_period);
}

std::vector<int32_t> LevelInteractionSignals::signal_available_after_et_seconds() const {
    // All signals are retrospective -- they scan the full RTH session and are
    // gated at RTH_CLOSE (16:00 ET), available for the next session only.
    // Bounce/break signals scan all RTH bars for level interaction with 2-bar
    // confirmation.  Support/resistance signals use full-day rth_high/rth_low.
    return {
        session_time::
            RTH_CLOSE,  // 0: Bounced prev RTH high  (57600 = 16:00, retrospective -- scans full RTH)
        session_time::
            RTH_CLOSE,  // 1: Bounced prev RTH low   (57600 = 16:00, retrospective -- scans full RTH)
        session_time::
            RTH_CLOSE,  // 2: Bounced prev POC       (57600 = 16:00, retrospective -- scans full RTH)
        session_time::
            RTH_CLOSE,  // 3: Bounced long EMA       (57600 = 16:00, retrospective -- scans full RTH)
        session_time::
            RTH_CLOSE,  // 4: Bounced VWAP           (57600 = 16:00, retrospective -- scans full RTH)
        session_time::
            RTH_CLOSE,  // 5: Prev high as support   (57600 = 16:00, retrospective -- uses full-day rth_low)
        session_time::
            RTH_CLOSE  // 6: Prev low as resistance (57600 = 16:00, retrospective -- uses full-day rth_high)
    };
}

// Helper: determine if price touched a level and whether it bounced or broke through
// Returns: +1 bounced up, -1 bounced down, 0 not tested
static float bounce_or_break(const MarketData& md, const SessionDayView& view, float level,
                             float touch_dist, bool check_from_above) {
    // Scan all RTH bars for level interaction (retrospective -- gated at RTH_CLOSE)
    const int rth_last = view.rth_first + view.rth_count - 1;

    for (int i = view.rth_first; i < view.rth_first + view.rth_count; ++i) {
        int32_t tod = md.session(i).time_of_day_sec;
        if (tod >= session_time::RTH_CLOSE) break;

        if (check_from_above) {
            // Level is below; price coming down to it
            if (md.bar(i).low <= level + touch_dist) {
                // Need 2 bars after touch for genuine confirmation;
                // skip if not enough bars remain in the session.
                if (i + 2 > rth_last) continue;
                float subsequent_close = md.bar(i + 2).close;
                if (subsequent_close > level + touch_dist) return 1.0f;   // bounced up
                if (subsequent_close < level - touch_dist) return -1.0f;  // broke down
            }
        } else {
            // Level is above; price coming up to it
            if (md.bar(i).high >= level - touch_dist) {
                // Need 2 bars after touch for genuine confirmation;
                // skip if not enough bars remain in the session.
                if (i + 2 > rth_last) continue;
                float subsequent_close = md.bar(i + 2).close;
                if (subsequent_close < level - touch_dist) return -1.0f;  // bounced down
                if (subsequent_close > level + touch_dist) return 1.0f;   // broke up
            }
        }
    }

    return 0.0f;
}

void LevelInteractionSignals::compute(const MarketData& md, int day_start, int day_end,
                                      float* output) const {
    const int num_days = day_end - day_start;
    const int NS       = num_signals();

    for (int i = 0; i < NS * num_days; ++i) output[i] = SIGNAL_NAN;

    // Compute EMA incrementally instead of O(D^2) recomputation.
    // Warm up the EMA over sessions [0, day_start) once, then update
    // incrementally within the main loop.
    const float alpha     = 2.0f / (static_cast<float>(m_ema_period) + 1.0f);
    float       ema       = SIGNAL_NAN;
    int         ema_count = 0;
    for (int k = 0; k < day_start; ++k) {
        SessionDayView dv = build_session_view(md, k);
        if (dv.rth_count == 0) continue;
        if (ema_count == 0) {
            ema = dv.rth_close;  // seed with first available close
        } else {
            ema = alpha * dv.rth_close + (1.0f - alpha) * ema;
        }
        ema_count++;
    }

    for (int d = day_start; d < day_end; ++d) {
        const int di = d - day_start;

        if (d < 1) {
            // Update EMA with this day's close before skipping
            SessionDayView dv = build_session_view(md, d);
            if (dv.rth_count > 0) {
                if (ema_count == 0)
                    ema = dv.rth_close;
                else
                    ema = alpha * dv.rth_close + (1.0f - alpha) * ema;
                ema_count++;
            }
            continue;
        }

        SessionDayView today = build_session_view(md, d);
        SessionDayView prev  = build_session_view(md, d - 1);
        if (today.rth_count == 0 || prev.rth_count == 0) {
            // Update EMA with this day's close before skipping
            if (today.rth_count > 0) {
                if (ema_count == 0)
                    ema = today.rth_close;
                else
                    ema = alpha * today.rth_close + (1.0f - alpha) * ema;
                ema_count++;
            }
            continue;
        }

        float touch_dist = m_touch_ticks * m_tick_size;

        // Signal 0: Bounced prev RTH high
        // If price approaches from below and reverses -> bearish rejection
        output[0 * num_days + di] = bounce_or_break(md, today, prev.rth_high, touch_dist, false);

        // Signal 1: Bounced prev RTH low
        // If price approaches from above and reverses -> bullish rejection
        output[1 * num_days + di] = bounce_or_break(md, today, prev.rth_low, touch_dist, true);

        // Signal 2: Bounced prev POC (volume profile; see compute_poc)
        {
            // True volume-profile POC: 1-tick bins across full session (ETH+RTH),
            // highest cumulative volume bin, tie-break by VWAP proximity.
            float prev_poc = compute_poc(md, prev.first_bar, prev.bar_count, m_tick_size);
            if (!std::isnan(prev_poc)) {
                bool from_above = (today.rth_open > prev_poc);
                output[2 * num_days + di] =
                    bounce_or_break(md, today, prev_poc, touch_dist, from_above);
            } else {
                output[2 * num_days + di] = 0.0f;
            }
        }

        // Signal 3: Bounced long EMA of session closes
        // EMA is maintained incrementally across loop iterations.
        // At this point, ema covers sessions [0, d-1].
        {
            if (ema_count > 0 && !std::isnan(ema)) {
                bool from_above           = (today.rth_open > ema);
                output[3 * num_days + di] = bounce_or_break(md, today, ema, touch_dist, from_above);
            } else {
                output[3 * num_days + di] = 0.0f;
            }
        }

        // Signal 4: Bounced VWAP (session VWAP computed up to current point)
        // Use yesterday's closing VWAP as the reference level for today
        {
            float prev_vwap = compute_vwap(md, prev.rth_first, prev.rth_count);
            if (!std::isnan(prev_vwap)) {
                bool from_above = (today.rth_open > prev_vwap);
                output[4 * num_days + di] =
                    bounce_or_break(md, today, prev_vwap, touch_dist, from_above);
            } else {
                output[4 * num_days + di] = 0.0f;
            }
        }

        // Signal 5: Prev high as support
        // +1 if today's RTH low stays above prev RTH high (old resistance = new support)
        {
            bool holds                = (today.rth_low >= prev.rth_high - touch_dist);
            bool tested               = (today.rth_low <= prev.rth_high + touch_dist * 3.0f);
            output[5 * num_days + di] = (holds && tested) ? 1.0f : 0.0f;
        }

        // Signal 6: Prev low as resistance
        // -1 if today's RTH high stays below prev RTH low (old support = new resistance)
        {
            bool holds                = (today.rth_high <= prev.rth_low + touch_dist);
            bool tested               = (today.rth_high >= prev.rth_low - touch_dist * 3.0f);
            output[6 * num_days + di] = (holds && tested) ? -1.0f : 0.0f;
        }

        // Update EMA with today's close for the next iteration.
        // After this, ema covers sessions [0, d].
        if (ema_count == 0) {
            ema = today.rth_close;
        } else {
            ema = alpha * today.rth_close + (1.0f - alpha) * ema;
        }
        ema_count++;
    }
}
