#include "intraday_structure.h"

// =============================================================================
// Intraday Structure Signals
// =============================================================================

IntradayStructureSignals::IntradayStructureSignals(int     atr_lookback,
                                                   int32_t range_consumed_checkpoint_et,
                                                   float   lunch_compression_threshold)
    : m_atr_lookback(atr_lookback),
      m_range_consumed_checkpoint_et(range_consumed_checkpoint_et),
      m_lunch_compression_threshold(lunch_compression_threshold) {
    assert(atr_lookback > 0);
    assert(range_consumed_checkpoint_et >= session_time::RTH_OPEN &&
           range_consumed_checkpoint_et <= session_time::RTH_CLOSE);
    assert(lunch_compression_threshold > 0.0f && lunch_compression_threshold <= 1.0f);
}

int IntradayStructureSignals::num_signals() const {
    return 10;
}
const char* IntradayStructureSignals::name() const {
    return "IntradayStructure";
}
int IntradayStructureSignals::warmup_days() const {
    return m_atr_lookback + 1;
}

std::vector<int32_t> IntradayStructureSignals::signal_available_after_et_seconds() const {
    return {
        session_time::IB_END,            // 0: Morning direction        (37800 = 10:30)
        session_time::IB_END,            // 1: Morning extreme timing   (37800 = 10:30)
        session_time::IB_END,            // 2: Morning vol collapse     (37800 = 10:30)
        session_time::IB_END,            // 3: Morning trending/rot     (37800 = 10:30)
        session_time::FIRST_15_END,      // 4: Morning first direction  (35100 = 09:45)
        m_range_consumed_checkpoint_et,  // 5: Range consumed %      (parameterized checkpoint)
        session_time::LUNCH_END,         // 6: Lunch compression        (46800 = 13:00)
        session_time::POWER_HOUR,        // 7: Afternoon direction      (54000 = 15:00)
        session_time::RTH_CLOSE,         // 8: Power hour direction     (57600 = 16:00)
        session_time::FIRST_15_END       // 9: Opening drive quality    (35100 = 09:45)
    };
}

void IntradayStructureSignals::compute(const MarketData& md, int day_start, int day_end,
                                       float* output) const {
    const int num_days = day_end - day_start;
    const int NS       = num_signals();

    for (int i = 0; i < NS * num_days; ++i) output[i] = SIGNAL_NAN;

    for (int d = day_start; d < day_end; ++d) {
        const int di = d - day_start;

        SessionDayView view = build_session_view(md, d);
        if (view.rth_count == 0) continue;

        float atr = compute_session_atr(md, d, m_atr_lookback);
        if (std::isnan(atr) || atr <= 0.0f) continue;

        // --- Morning: 09:30-10:30 ---
        auto [morn_first, morn_count] =
            bars_in_time_range(md, view, session_time::RTH_OPEN, session_time::IB_END);

        if (morn_count > 0) {
            float morn_open  = md.bar(morn_first).open;
            float morn_close = md.bar(morn_first + morn_count - 1).close;
            float morn_high = -1e30f, morn_low = 1e30f;
            int   high_bar_offset = 0, low_bar_offset = 0;

            for (int i = 0; i < morn_count; ++i) {
                int bi = morn_first + i;
                if (md.bar(bi).high > morn_high) {
                    morn_high       = md.bar(bi).high;
                    high_bar_offset = i;
                }
                if (md.bar(bi).low < morn_low) {
                    morn_low       = md.bar(bi).low;
                    low_bar_offset = i;
                }
            }

            (void)(morn_high - morn_low);  // range used implicitly via high/low

            // Signal 0: Morning direction
            output[0 * num_days + di] = direction_signal(morn_open, morn_close, atr * 0.02f);

            // Signal 1: Morning extreme timing
            // Which extreme was set and when?
            // Use the bar offset of the session extreme within the morning period
            // Early (first quarter) = +1, late (last quarter) = -1, middle = 0
            {
                // Determine if the key extreme is the high or low
                // Trending up -> session extreme is the high; trending down -> the low
                if (morn_count <= 1) {
                    // Single bar in morning: no meaningful timing information
                    output[1 * num_days + di] = 0.0f;
                } else {
                    bool  extreme_is_high = (morn_close > morn_open);
                    int   extreme_offset  = extreme_is_high ? high_bar_offset : low_bar_offset;
                    float frac =
                        static_cast<float>(extreme_offset) / static_cast<float>(morn_count - 1);
                    // Map: 0.0 (early) -> +1.0, 0.5 (mid) -> 0.0, 1.0 (late) -> -1.0
                    output[1 * num_days + di] = clamp_signal(1.0f - 2.0f * frac);
                }
            }

            // Signal 2: Morning volatility collapse
            // Check if range in second half of morning is < 25% of first half
            {
                int half = morn_count / 2;
                if (half > 0) {
                    float first_half_high = -1e30f, first_half_low = 1e30f;
                    float second_half_high = -1e30f, second_half_low = 1e30f;
                    for (int i = 0; i < half; ++i) {
                        int bi          = morn_first + i;
                        first_half_high = std::max(first_half_high, md.bar(bi).high);
                        first_half_low  = std::min(first_half_low, md.bar(bi).low);
                    }
                    for (int i = half; i < morn_count; ++i) {
                        int bi           = morn_first + i;
                        second_half_high = std::max(second_half_high, md.bar(bi).high);
                        second_half_low  = std::min(second_half_low, md.bar(bi).low);
                    }
                    float first_range  = first_half_high - first_half_low;
                    float second_range = second_half_high - second_half_low;
                    bool  collapse     = (first_range > 0 && second_range < first_range * 0.25f);
                    output[2 * num_days + di] = collapse ? 1.0f : 0.0f;
                } else {
                    output[2 * num_days + di] = 0.0f;
                }
            }

            // Signal 3: Morning trending vs rotational
            // Trending: HH+HL (up) or LL+LH (down)
            // Rotational: alternating direction
            {
                if (morn_count >= 4) {
                    // Split morning into 4 quarters and check swing pattern
                    int   q = morn_count / 4;
                    float qh[4], ql[4];
                    for (int qi = 0; qi < 4; ++qi) {
                        int qs = morn_first + qi * q;
                        int qe =
                            (qi == 3) ? (morn_first + morn_count) : (morn_first + (qi + 1) * q);
                        qh[qi] = -1e30f;
                        ql[qi] = 1e30f;
                        for (int i = qs; i < qe; ++i) {
                            qh[qi] = std::max(qh[qi], md.bar(i).high);
                            ql[qi] = std::min(ql[qi], md.bar(i).low);
                        }
                    }
                    // Uptrend: HH + HL
                    bool hh = (qh[1] > qh[0]) && (qh[2] > qh[1] || qh[3] > qh[2]);
                    bool hl = (ql[1] > ql[0]) && (ql[2] > ql[1] || ql[3] > ql[2]);
                    // Downtrend: LL + LH
                    bool ll = (ql[1] < ql[0]) && (ql[2] < ql[1] || ql[3] < ql[2]);
                    bool lh = (qh[1] < qh[0]) && (qh[2] < qh[1] || qh[3] < qh[2]);

                    if (hh && hl)
                        output[3 * num_days + di] = 1.0f;
                    else if (ll && lh)
                        output[3 * num_days + di] = -1.0f;
                    else
                        output[3 * num_days + di] = 0.0f;
                } else {
                    output[3 * num_days + di] = 0.0f;
                }
            }

            // Signal 4: Morning first direction (first 15 min)
            {
                auto [f15_first, f15_count] = bars_in_time_range(md, view, session_time::RTH_OPEN,
                                                                 session_time::FIRST_15_END);
                if (f15_count > 0) {
                    float f15_open            = md.bar(f15_first).open;
                    float f15_close           = md.bar(f15_first + f15_count - 1).close;
                    output[4 * num_days + di] = direction_signal(f15_open, f15_close, atr * 0.01f);
                } else {
                    output[4 * num_days + di] = 0.0f;
                }
            }
        }

        // Signal 5: Range consumed % at checkpoint (fraction of ATR used by then)
        {
            auto [r_first, r_count] = bars_in_time_range(md, view, session_time::RTH_OPEN,
                                                         m_range_consumed_checkpoint_et);
            if (r_count > 0) {
                float high = -1e30f, low = 1e30f;
                for (int i = r_first; i < r_first + r_count; ++i) {
                    high = std::max(high, md.bar(i).high);
                    low  = std::min(low, md.bar(i).low);
                }
                float consumed            = (high - low) / atr;
                output[5 * num_days + di] = clamp_signal(consumed);
            }
        }

        // Signal 6: Lunch compression (11:30-13:00 range < 0.3 ATR)
        {
            auto [lunch_first, lunch_count] =
                bars_in_time_range(md, view, session_time::LUNCH_START, session_time::LUNCH_END);
            if (lunch_count > 0) {
                float high = -1e30f, low = 1e30f;
                for (int i = lunch_first; i < lunch_first + lunch_count; ++i) {
                    high = std::max(high, md.bar(i).high);
                    low  = std::min(low, md.bar(i).low);
                }
                float lunch_range = high - low;
                output[6 * num_days + di] =
                    (lunch_range < atr * m_lunch_compression_threshold) ? 1.0f : 0.0f;
            }
        }

        // Signal 7: Afternoon direction (12:00-15:00)
        {
            int32_t noon = 12 * 3600;
            auto [aft_first, aft_count] =
                bars_in_time_range(md, view, noon, session_time::POWER_HOUR);
            if (aft_count > 0) {
                float aft_open            = md.bar(aft_first).open;
                float aft_close           = md.bar(aft_first + aft_count - 1).close;
                output[7 * num_days + di] = direction_signal(aft_open, aft_close, atr * 0.02f);
            }
        }

        // Signal 8: Power hour direction (15:00-15:50)
        {
            auto [ph_first, ph_count] = bars_in_time_range(md, view, session_time::POWER_HOUR,
                                                           session_time::POWER_HOUR_END);
            if (ph_count > 0) {
                float ph_open             = md.bar(ph_first).open;
                float ph_close            = md.bar(ph_first + ph_count - 1).close;
                output[8 * num_days + di] = direction_signal(ph_open, ph_close, atr * 0.02f);
            }
        }

        // Signal 9: Opening drive quality (first 15 min magnitude / ATR, signed)
        {
            auto [od_first, od_count] =
                bars_in_time_range(md, view, session_time::RTH_OPEN, session_time::FIRST_15_END);
            if (od_count > 0) {
                float od_open  = md.bar(od_first).open;
                float od_close = md.bar(od_first + od_count - 1).close;
                float move     = od_close - od_open;
                float quality  = move / atr;
                output[9 * num_days + di] =
                    clamp_signal(quality * 5.0f);  // scale so 0.2 ATR move = 1.0
            }
        }
    }
}
