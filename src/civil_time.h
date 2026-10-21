#pragma once
// ---------------------------------------------------------------------------
// Civil-time helpers: proleptic Gregorian date arithmetic and the US Eastern
// daylight-saving rule (second Sunday of March 02:00 local -> first Sunday of
// November 02:00 local, in force since 2007).
//
// Everything is integer arithmetic, so results are identical on every
// platform and compiler. Date algorithms follow Howard Hinnant's
// days_from_civil / civil_from_days.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace civil {

constexpr int64_t NS_PER_SEC  = 1'000'000'000LL;
constexpr int64_t SEC_PER_DAY = 86'400LL;

// Days since 1970-01-01 for a proleptic Gregorian date.
constexpr int64_t days_from_civil(int y, int m, int d) {
    y -= m <= 2 ? 1 : 0;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const int64_t yoe = y - era * 400;
    const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

struct Date {
    int year;
    int month;  // 1..12
    int day;    // 1..31
};

constexpr Date civil_from_days(int64_t z) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const int64_t doe = z - era * 146097;
    const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const int64_t mp  = (5 * doy + 2) / 153;
    const int     d   = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    const int     m   = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    const int     y   = static_cast<int>(yoe + era * 400 + (m <= 2 ? 1 : 0));
    return Date{y, m, d};
}

// Day of week for a day count since the epoch: 0 = Sunday .. 6 = Saturday.
constexpr int weekday_from_days(int64_t z) {
    return static_cast<int>(z >= -4 ? (z + 4) % 7 : (z + 5) % 7 + 6);
}

// Floor division for possibly negative numerators.
constexpr int64_t floor_div(int64_t a, int64_t b) {
    return a / b - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0);
}

// Day count of the n-th (1-based) given weekday in a month.
constexpr int64_t nth_weekday(int y, int m, int weekday, int n) {
    const int64_t first  = days_from_civil(y, m, 1);
    const int     offset = (weekday - weekday_from_days(first) + 7) % 7;
    return first + offset + 7 * (n - 1);
}

constexpr int32_t US_EASTERN_EST = -5 * 3600;
constexpr int32_t US_EASTERN_EDT = -4 * 3600;

// UTC offset in seconds of US Eastern time at a UTC instant.
constexpr int32_t us_eastern_offset_sec(int64_t utc_sec) {
    const int y = civil_from_days(floor_div(utc_sec, SEC_PER_DAY)).year;
    // 02:00 EST = 07:00 UTC; 02:00 EDT = 06:00 UTC.
    const int64_t dst_start = nth_weekday(y, 3, 0, 2) * SEC_PER_DAY + 7 * 3600;
    const int64_t dst_end   = nth_weekday(y, 11, 0, 1) * SEC_PER_DAY + 6 * 3600;
    return (utc_sec >= dst_start && utc_sec < dst_end) ? US_EASTERN_EDT : US_EASTERN_EST;
}

// UTC instant (seconds) of a US Eastern wall-clock time. Ambiguous fall-back
// times resolve to the first (EDT) occurrence; nonexistent spring-forward
// times are read as EST.
constexpr int64_t us_eastern_to_utc_sec(int y, int m, int d, int32_t local_sec) {
    const int64_t local  = days_from_civil(y, m, d) * SEC_PER_DAY + local_sec;
    const int64_t as_edt = local - US_EASTERN_EDT;
    if (us_eastern_offset_sec(as_edt) == US_EASTERN_EDT) return as_edt;
    return local - US_EASTERN_EST;
}

}  // namespace civil
