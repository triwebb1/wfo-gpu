#include "session_model.h"

#include "civil_time.h"
#include "market_data.h"

SessionModel SessionModel::us_futures() {
    return SessionModel{};
}

SessionModel SessionModel::continuous() {
    SessionModel m;
    m.name              = "continuous";
    m.clock             = Clock::UTC;
    m.session_open_sec  = 0;
    m.session_close_sec = -1;
    m.flat_at_close     = false;
    m.core_open_sec     = 13 * 3600 + 30 * 60;
    m.core_split_sec    = 16 * 3600 + 30 * 60;
    m.core_close_sec    = 20 * 3600;
    m.detect_halts      = false;
    m.early_close_calendar.clear();
    m.session_signals = false;
    return m;
}

bool SessionModel::from_name(const std::string& name, SessionModel& out) {
    if (name == "us_futures") {
        out = us_futures();
        return true;
    }
    if (name == "continuous") {
        out = continuous();
        return true;
    }
    return false;
}

int32_t SessionModel::utc_offset_sec(int64_t utc_sec) const {
    return clock == Clock::US_EASTERN ? civil::us_eastern_offset_sec(utc_sec) : 0;
}

uint8_t SessionModel::hours_flags(int32_t tod) const {
    uint8_t f = 0;
    if (tod >= core_open_sec && tod < core_close_sec) f |= SESSION_IN_CORE;
    if (tod >= core_open_sec && tod < core_split_sec) f |= SESSION_IN_MORNING;
    if (tod >= core_split_sec && tod < core_close_sec) f |= SESSION_IN_AFTERNOON;
    return f;
}

bool SessionModel::validate(std::string* error) const {
    auto fail = [&](const char* msg) {
        if (error) *error = msg;
        return false;
    };
    constexpr int32_t DAY    = 86400;
    auto              in_day = [](int32_t t) { return t >= 0 && t < DAY; };
    if (!in_day(session_open_sec)) return fail("session open must be within the day");
    if (session_close_sec >= DAY) return fail("session close must be within the day");
    if (flat_at_close && session_close_sec < 0)
        return fail("flat_at_close needs a session close time");
    if (!in_day(core_open_sec) || core_close_sec <= 0 || core_close_sec > DAY)
        return fail("core hours must be within the day");
    if (!(core_open_sec <= core_split_sec && core_split_sec <= core_close_sec))
        return fail("core hours need open <= split <= close");
    if (detect_halts && halt_gap_sec <= 0) return fail("halt_gap must be positive");
    return true;
}

void stamp_session_flags(SessionInfo* sessions, int count, const SessionModel& model,
                         int base_tf_sec) {
    const int32_t open = model.session_open_sec;
    // At or past the daily close of its own session, before the next open.
    auto after_close = [&](const SessionInfo& s) {
        return model.flat_at_close && s.session_close_sec >= 0 &&
               seconds_since_open(s.time_of_day_sec, open) >=
                   seconds_since_open(s.session_close_sec, open);
    };

    for (int i = 0; i < count; ++i) {
        SessionInfo& s = sessions[i];
        uint8_t      f = 0;
        if (after_close(s)) {
            f = SESSION_AFTER_CLOSE;
        } else {
            f = model.hours_flags(s.time_of_day_sec);
            if (model.flat_at_close && s.session_close_sec >= 0) {
                const bool final_slot = s.time_of_day_sec < s.session_close_sec &&
                                        s.time_of_day_sec + base_tf_sec >= s.session_close_sec;
                const bool next_is_past_close =
                    i + 1 < count &&
                    (sessions[i + 1].session_day != s.session_day || after_close(sessions[i + 1]));
                if (final_slot || next_is_past_close) f |= SESSION_CLOSE_BAR;
            }
        }
        if (s.halt_flag) f |= SESSION_HALTED;
        s.flags = f;
    }
}
