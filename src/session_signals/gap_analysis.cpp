#include "gap_analysis.h"

// =============================================================================
// Gap Analysis Signals
// =============================================================================

GapAnalysisSignals::GapAnalysisSignals(int atr_lookback) : m_atr_lookback(atr_lookback) {
    assert(atr_lookback > 0);
}

int GapAnalysisSignals::num_signals() const {
    return 6;
}
const char* GapAnalysisSignals::name() const {
    return "GapAnalysis";
}
int GapAnalysisSignals::warmup_days() const {
    return std::max(m_atr_lookback + 1, 4);
}

std::vector<int32_t> GapAnalysisSignals::signal_available_after_et_seconds() const {
    return {
        session_time::RTH_OPEN,  // 0: Gap direction     (34200 = 09:30)
        session_time::RTH_OPEN,  // 1: Gap size bucket   (34200 = 09:30)
        39600,                   // 2: Gap fill timing   (39600 = 11:00)
        session_time::RTH_OPEN,  // 3: Gap into prev VA  (34200 = 09:30)
        session_time::RTH_OPEN,  // 4: Gap in trend dir  (34200 = 09:30)
        session_time::RTH_OPEN   // 5: Gap vs ETH VWAP   (34200 = 09:30)
    };
}

void GapAnalysisSignals::compute(const MarketData& md, int day_start, int day_end,
                                 float* output) const {
    const int num_days = day_end - day_start;
    const int NS       = num_signals();

    for (int i = 0; i < NS * num_days; ++i) output[i] = SIGNAL_NAN;

    for (int d = day_start; d < day_end; ++d) {
        const int di = d - day_start;

        if (d < 1) continue;  // Need at least one prior day

        SessionDayView today = build_session_view(md, d);
        SessionDayView prev  = build_session_view(md, d - 1);
        if (today.rth_count == 0 || prev.rth_count == 0) continue;

        float atr = compute_session_atr(md, d, m_atr_lookback);
        if (std::isnan(atr) || atr <= 0.0f) continue;

        float gap     = today.rth_open - prev.rth_close;
        float gap_abs = std::abs(gap);

        // Signal 0: Gap direction
        output[0 * num_days + di] = direction_signal(prev.rth_close, today.rth_open, atr * 0.01f);

        // Signal 1: Gap size bucket (ATR-relative, signed)
        // tiny < 0.1 ATR, small < 0.25 ATR, medium < 0.5 ATR, large >= 0.5 ATR
        {
            float bucket;
            float ratio = gap_abs / atr;
            if (ratio < 0.1f)
                bucket = 0.0f;
            else if (ratio < 0.25f)
                bucket = 0.33f;
            else if (ratio < 0.5f)
                bucket = 0.66f;
            else
                bucket = 1.0f;

            float sign                = (gap >= 0.0f) ? 1.0f : -1.0f;
            output[1 * num_days + di] = sign * bucket;
        }

        // Signal 2: Gap fill timing
        // Gap up filled = price drops to prev close; gap down filled = price rises to prev close
        {
            float fill_signal = 0.0f;
            bool  gap_up      = (gap > atr * 0.01f);
            bool  gap_down    = (gap < -atr * 0.01f);

            if (gap_up || gap_down) {
                // Check bars at various times
                int32_t check_times[] = {
                    session_time::OR_END,  // 10:00
                    session_time::IB_END,  // 10:30
                    11 * 3600              // 11:00 = 39600
                };
                float fill_values[] = {0.25f, 0.5f, 1.0f};

                for (int t = 0; t < 3; ++t) {
                    auto [tf, tc] =
                        bars_in_time_range(md, today, session_time::RTH_OPEN, check_times[t]);
                    for (int i = tf; i < tf + tc && tc > 0; ++i) {
                        if (gap_up && md.bar(i).low <= prev.rth_close) {
                            fill_signal = fill_values
                                [t];  // gap down filled -> bullish reversal sign absorbed below
                            break;
                        }
                        if (gap_down && md.bar(i).high >= prev.rth_close) {
                            fill_signal = fill_values[t];
                            break;
                        }
                    }
                    if (fill_signal > 0.0f) break;
                }
                // Sign: gap up that fills = bearish (-), gap down that fills = bullish (+)
                if (gap_up) fill_signal = -fill_signal;
                // gap_down fills are already positive (bullish reversal)
            }
            output[2 * num_days + di] = fill_signal;
        }

        // Signal 3: Gap into prev value area (RTH open inside prev day range)
        {
            bool inside = (today.rth_open >= prev.rth_low && today.rth_open <= prev.rth_high);
            output[3 * num_days + di] = inside ? 1.0f : 0.0f;
        }

        // Signal 4: Gap in trend direction (matches 3-day directional streak)
        // NO LOOKAHEAD: Checks days d-3, d-2, d-1 (prior completed sessions).
        // At RTH_OPEN of session d, all three sessions are fully closed
        // (CME sessions close at 17:00 ET, RTH opens at 09:30 ET next day).
        // Day d's data is not referenced in this loop.
        {
            float trend_signal = 0.0f;
            if (d >= 4) {
                int up_count = 0, down_count = 0;
                for (int k = d - 3; k < d; ++k) {
                    SessionDayView dv      = build_session_view(md, k);
                    SessionDayView dv_prev = build_session_view(md, k - 1);
                    if (dv.rth_count > 0 && dv_prev.rth_count > 0) {
                        if (dv.rth_close > dv_prev.rth_close)
                            up_count++;
                        else if (dv.rth_close < dv_prev.rth_close)
                            down_count++;
                    }
                }
                if (up_count >= 3 && gap > 0)
                    trend_signal = 1.0f;
                else if (down_count >= 3 && gap < 0)
                    trend_signal = 1.0f;
                else if (up_count >= 3 && gap < 0)
                    trend_signal = -1.0f;
                else if (down_count >= 3 && gap > 0)
                    trend_signal = -1.0f;
            }
            output[4 * num_days + di] = trend_signal;
        }

        // Signal 5: Gap vs ETH VWAP
        {
            float eth_vwap = compute_vwap(md, today.eth_first, today.eth_count);
            if (!std::isnan(eth_vwap) && today.eth_count > 0) {
                output[5 * num_days + di] = direction_signal(eth_vwap, today.rth_open, atr * 0.01f);
            } else {
                output[5 * num_days + di] = 0.0f;
            }
        }
    }
}
