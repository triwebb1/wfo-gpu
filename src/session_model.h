#pragma once
// ---------------------------------------------------------------------------
// SessionModel -- how a market's trading day is laid out.
//
// MarketData uses the model to assign bars to session days, to compute each
// bar's local time of day, and to stamp per-bar flags (core hours, last bar
// before the daily close, halted session). The backtest engines read only
// those flags, so the same CPU and GPU code serves any market the model can
// describe.
//
// Built-in models:
//   us_futures  US Eastern clock with DST. Sessions run 18:00 -> 17:00 with
//               a one-hour break, flat at the close, core hours 09:30-16:00
//               split at 12:00, halt detection, and the early-close calendar
//               (config/cme_early_closes.csv).
//   continuous  24/7 markets on the UTC clock. Session days are UTC days,
//               there is no daily close, and core hours default to
//               13:30-20:00 UTC (the US equity session), split at 16:30.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>

struct SessionInfo;

// Bits of SessionInfo::flags.
enum SessionFlag : uint8_t {
    SESSION_IN_CORE      = 1u << 0,  // inside core hours ("RTH")
    SESSION_IN_MORNING   = 1u << 1,  // core open .. core split
    SESSION_IN_AFTERNOON = 1u << 2,  // core split .. core close
    SESSION_CLOSE_BAR    = 1u << 3,  // last bar before the daily close
    SESSION_HALTED       = 1u << 4,  // session has an abnormal core-hours gap
    SESSION_AFTER_CLOSE  = 1u << 5,  // between the daily close and the next open
};

struct SessionModel {
    enum class Clock : uint8_t {
        UTC        = 0,
        US_EASTERN = 1,  // America/New_York, US DST rules since 2007
    };

    std::string name  = "us_futures";
    Clock       clock = Clock::US_EASTERN;

    // Local time (seconds after midnight) at which a new session day begins.
    int32_t session_open_sec = 18 * 3600;
    // Local daily close; positions are flattened on the last bar before it
    // when flat_at_close is set. Negative means no daily close.
    int32_t session_close_sec = 17 * 3600;
    bool    flat_at_close     = true;

    // Core hours drive the RTH / MORNING / AFTERNOON trading-hours filters.
    int32_t core_open_sec  = 9 * 3600 + 30 * 60;
    int32_t core_split_sec = 12 * 3600;
    int32_t core_close_sec = 16 * 3600;

    // Flag a session as halted when two consecutive core-hours bars are more
    // than halt_gap_sec apart.
    bool    detect_halts = true;
    int32_t halt_gap_sec = 15 * 60;

    // File name of the early-close calendar ("YYYY-MM-DD,HH:MM" rows), looked
    // up next to the bar file and in config/. Empty disables it.
    std::string early_close_calendar = "cme_early_closes.csv";

    // Whether the futures session-signal module (opening range, gaps,
    // overnight context, ...) applies to this market.
    bool session_signals = true;

    static SessionModel us_futures();
    static SessionModel continuous();

    // Look up a built-in model by name. Returns false if unknown.
    static bool from_name(const std::string& name, SessionModel& out);

    // Offset to add to UTC seconds to get local clock seconds.
    int32_t utc_offset_sec(int64_t utc_sec) const;

    // Core-hours flag bits for a local time of day.
    uint8_t hours_flags(int32_t time_of_day_sec) const;

    // Check invariants; on failure returns false and describes the problem.
    bool validate(std::string* error = nullptr) const;
};

// Seconds elapsed since the session opened, for a local time of day. Times of
// day cannot be compared directly: a session that opens at 18:00 sees 19:00
// before it sees 10:00 the next morning.
inline int32_t seconds_since_open(int32_t time_of_day_sec, int32_t session_open_sec) {
    const int32_t d = time_of_day_sec - session_open_sec;
    return d < 0 ? d + 86400 : d;
}

// Recompute SessionInfo::flags for bars built by hand (tests, custom data
// pipelines) from session_day, time_of_day_sec, session_close_sec and
// halt_flag. base_tf_sec is the bar length used to find the last bar before
// the close.
//
// With flat_at_close, CLOSE_BAR marks the last bar before each daily close:
// the bar in the final slot, or, when that slot is missing, the bar after
// which the session ends or the close passes. Bars stamped after the close
// (an early close, or stray data between the close and the next open) get
// AFTER_CLOSE and no core-hours bits.
void stamp_session_flags(SessionInfo* sessions, int count, const SessionModel& model,
                         int base_tf_sec);
