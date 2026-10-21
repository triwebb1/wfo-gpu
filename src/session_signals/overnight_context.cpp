#include "overnight_context.h"

// =============================================================================
// Overnight (ETH) Context Signals
// =============================================================================

OvernightContextSignals::OvernightContextSignals(int atr_lookback) : m_atr_lookback(atr_lookback) {
    assert(atr_lookback > 0);
}

int OvernightContextSignals::num_signals() const {
    return 7;
}
const char* OvernightContextSignals::name() const {
    return "OvernightContext";
}
int OvernightContextSignals::warmup_days() const {
    return m_atr_lookback + 1;
}

std::vector<int32_t> OvernightContextSignals::signal_available_after_et_seconds() const {
    // All overnight context signals use only ETH data -> available at RTH open
    return {
        session_time::RTH_OPEN,  // 0: ETH direction
        session_time::RTH_OPEN,  // 1: ETH range
        session_time::RTH_OPEN,  // 2: ETH closed near extreme
        session_time::RTH_OPEN,  // 3: ETH new high late
        session_time::RTH_OPEN,  // 4: ETH trend steady
        session_time::RTH_OPEN,  // 5: RTH opens above/below ETH VWAP
        session_time::RTH_OPEN   // 6: RTH opens above/below ETH range
    };
}

void OvernightContextSignals::compute(const MarketData& md, int day_start, int day_end,
                                      float* output) const {
    const int num_days = day_end - day_start;
    const int NS       = num_signals();

    for (int i = 0; i < NS * num_days; ++i) output[i] = SIGNAL_NAN;

    for (int d = day_start; d < day_end; ++d) {
        const int di = d - day_start;

        SessionDayView view = build_session_view(md, d);
        if (view.eth_count == 0) continue;  // No ETH data

        float atr = compute_session_atr(md, d, m_atr_lookback);
        if (std::isnan(atr) || atr <= 0.0f) continue;

        float eth_range = view.eth_high - view.eth_low;

        // Signal 0: ETH direction
        output[0 * num_days + di] = direction_signal(view.eth_open, view.eth_close, atr * 0.01f);

        // Signal 1: ETH range (ATR-relative), clamped to [0, 1]
        output[1 * num_days + di] = clamp_signal(eth_range / atr);

        // Signal 2: ETH closed near extreme
        // Near high (upper 25%) -> +1 (bullish), near low (lower 25%) -> -1, middle -> 0
        if (eth_range > 0.0f) {
            float close_pos = (view.eth_close - view.eth_low) / eth_range;
            if (close_pos >= 0.75f)
                output[2 * num_days + di] = 1.0f;
            else if (close_pos <= 0.25f)
                output[2 * num_days + di] = -1.0f;
            else
                output[2 * num_days + di] = 0.0f;
        } else {
            output[2 * num_days + di] = 0.0f;
        }

        // Signal 3: ETH new high late (session high in last 2 hours of overnight)
        // Overnight runs from ~18:00 to 09:30; last 2 hours = 07:30-09:30
        {
            int32_t late_start = session_time::RTH_OPEN - 2 * 3600;  // 07:30 = 27000
            float   late_high  = -1e30f;
            for (int i = view.eth_first; i < view.eth_first + view.eth_count; ++i) {
                int32_t tod = md.session(i).time_of_day_sec;
                if (tod >= late_start && tod < session_time::RTH_OPEN) {
                    late_high = std::max(late_high, md.bar(i).high);
                }
            }
            bool new_high_late = (late_high >= view.eth_high - atr * 0.01f && late_high > -1e29f);
            output[3 * num_days + di] = new_high_late ? 1.0f : 0.0f;
        }

        // Signal 4: ETH trend steady (quiet overnight, range < 0.5 ATR)
        output[4 * num_days + di] = (eth_range < atr * 0.5f) ? 1.0f : 0.0f;

        // Signal 5: RTH opens above/below ETH VWAP
        if (view.rth_count > 0) {
            float eth_vwap = compute_vwap(md, view.eth_first, view.eth_count);
            if (!std::isnan(eth_vwap)) {
                output[5 * num_days + di] = direction_signal(eth_vwap, view.rth_open, atr * 0.01f);
            } else {
                output[5 * num_days + di] = 0.0f;
            }
        }

        // Signal 6: RTH opens above/below ETH range
        if (view.rth_count > 0) {
            if (view.rth_open > view.eth_high)
                output[6 * num_days + di] = 1.0f;  // gapped above ETH range
            else if (view.rth_open < view.eth_low)
                output[6 * num_days + di] = -1.0f;  // gapped below ETH range
            else
                output[6 * num_days + di] = 0.0f;  // inside ETH range
        }
    }
}
