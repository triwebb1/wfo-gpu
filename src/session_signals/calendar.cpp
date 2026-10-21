#include "calendar.h"

// =============================================================================
// Calendar & Timing Signals
// =============================================================================

int CalendarSignals::num_signals() const {
    return 4;
}
const char* CalendarSignals::name() const {
    return "Calendar";
}
int CalendarSignals::warmup_days() const {
    return 0;
}

std::vector<int32_t> CalendarSignals::signal_available_after_et_seconds() const {
    // Calendar signals are date-derived -> always available
    return {0, 0, 0, 0};
}

// ---------------------------------------------------------------------------
// Calendar helpers (same approach as market_data.cpp)
// ---------------------------------------------------------------------------
static constexpr int64_t NS_PER_SEC_CAL = 1'000'000'000LL;
static constexpr int64_t NS_PER_DAY_CAL = 86400LL * NS_PER_SEC_CAL;

// Convert UTC nanosecond timestamp to (year, month, day) -- Howard Hinnant algorithm
static void ts_to_ymd(int64_t ts_ns, int& year, int& month, int& day) {
    int64_t total_days = ts_ns / NS_PER_DAY_CAL;
    if (ts_ns < 0) total_days -= 1;

    int64_t z   = total_days + 719468;
    int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    int64_t doe = z - era * 146097;
    int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int64_t y   = yoe + era * 400;
    int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int64_t mp  = (5 * doy + 2) / 153;
    day         = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    month       = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    year        = static_cast<int>(y + (month <= 2 ? 1 : 0));
}

// Tomohiko Sakamoto's day-of-week algorithm (0=Sunday ... 6=Saturday)
static int day_of_week(int y, int m, int d) {
    static constexpr int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (m < 3) y -= 1;
    return (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;
}

void CalendarSignals::compute(const MarketData& md, int day_start, int day_end,
                              float* output) const {
    const int num_days = day_end - day_start;
    const int NS       = num_signals();

    for (int i = 0; i < NS * num_days; ++i) output[i] = SIGNAL_NAN;

    for (int d = day_start; d < day_end; ++d) {
        const int di = d - day_start;

        SessionDayView view = build_session_view(md, d);
        if (view.bar_count == 0) continue;

        // Get timestamp and derive ET offset from session info (DST-aware).
        // Use RTH open bar if available (its time_of_day_sec is known = 34200 for 09:30 ET).
        // ET offset = time_of_day_sec - UTC_seconds_since_midnight.
        int     ref_bar                = (view.rth_first >= 0) ? view.rth_first : view.first_bar;
        int64_t ts                     = md.bar(ref_bar).timestamp;
        int32_t tod_et                 = md.session(ref_bar).time_of_day_sec;
        int64_t utc_sec_since_midnight = (ts / NS_PER_SEC_CAL) % 86400;
        if (ts < 0) utc_sec_since_midnight = ((ts / NS_PER_SEC_CAL) % 86400 + 86400) % 86400;
        int64_t et_offset_sec = static_cast<int64_t>(tod_et) - utc_sec_since_midnight;
        int64_t et_ts         = ts + et_offset_sec * NS_PER_SEC_CAL;

        int year, month, day;
        ts_to_ymd(et_ts, year, month, day);

        int dow = day_of_week(year, month, day);

        // Signal 0: Day of week
        // Mon(1)=-1, Tue(2)=-0.5, Wed(3)=0, Thu(4)=0.5, Fri(5)=1
        // dow: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
        {
            float dow_signal = 0.0f;
            switch (dow) {
                case 1:
                    dow_signal = -1.0f;
                    break;  // Monday
                case 2:
                    dow_signal = -0.5f;
                    break;  // Tuesday
                case 3:
                    dow_signal = 0.0f;
                    break;  // Wednesday
                case 4:
                    dow_signal = 0.5f;
                    break;  // Thursday
                case 5:
                    dow_signal = 1.0f;
                    break;  // Friday
                default:
                    dow_signal = 0.0f;
                    break;  // Weekend (shouldn't happen for RTH)
            }
            output[0 * num_days + di] = dow_signal;
        }

        // Signal 1: Week of month (1-5, scaled: 1=-1, 5=+1)
        {
            int   week        = (day - 1) / 7 + 1;                    // 1-5
            float week_signal = static_cast<float>(week - 3) / 2.0f;  // maps 1->-1, 3->0, 5->+1
            output[1 * num_days + di] = clamp_signal(week_signal);
        }

        // Signal 2: OPEX week (3rd Friday week of the month)
        {
            // Find the 3rd Friday of this month
            int first_dow = day_of_week(year, month, 1);
            // First Friday: if first_dow <= 5, first_fri = 1 + (5 - first_dow) % 7
            int first_fri = 1 + (5 - first_dow + 7) % 7;
            int third_fri = first_fri + 14;

            // Check if current day is in the same week as third_fri
            // Week containing third_fri: [third_fri - day_of_week(third_fri) ... +6]
            // Simplify: within 3 business days of the third Friday
            bool opex_week            = (day >= third_fri - 4 && day <= third_fri + 1);
            output[2 * num_days + di] = opex_week ? 1.0f : 0.0f;
        }

        // Signal 3: Quarter signal -- Q1=-1, Q2=-0.5, Q3=+0.5, Q4=+1
        {
            float q_signal = 0.0f;
            if (month <= 3)
                q_signal = -1.0f;
            else if (month <= 6)
                q_signal = -0.5f;
            else if (month <= 9)
                q_signal = 0.5f;
            else
                q_signal = 1.0f;
            output[3 * num_days + di] = q_signal;
        }
    }
}
