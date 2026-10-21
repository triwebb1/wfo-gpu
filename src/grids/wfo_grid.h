#pragma once
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// WFO Re-optimization trigger mode
//
// FIXED_INTERVAL: re-optimize every F bars.
// Adaptive triggers: step forward at 1-week intervals but only re-optimize
// when the trigger condition fires. Between triggers, reuse the last winner.
// ---------------------------------------------------------------------------
enum class WFOTrigger : uint8_t {
    FIXED_INTERVAL   = 0,  // re-optimize every F bars
    DD_TRIGGER_5PCT  = 1,  // re-optimize when OOS equity drops 5% from peak
    DD_TRIGGER_10PCT = 2,  // re-optimize when OOS equity drops 10% from peak
    LOSERS_5         = 3,  // re-optimize after 5 consecutive losing trades
    LOSERS_10        = 4   // re-optimize after 10 consecutive losing trades
};

inline const char* trigger_name(WFOTrigger t) {
    switch (t) {
        case WFOTrigger::FIXED_INTERVAL:
            return "fixed";
        case WFOTrigger::DD_TRIGGER_5PCT:
            return "dd5pct";
        case WFOTrigger::DD_TRIGGER_10PCT:
            return "dd10pct";
        case WFOTrigger::LOSERS_5:
            return "losers5";
        case WFOTrigger::LOSERS_10:
            return "losers10";
    }
    return "?";
}

inline float trigger_dd_pct(WFOTrigger t) {
    switch (t) {
        case WFOTrigger::DD_TRIGGER_5PCT:
            return 0.05f;
        case WFOTrigger::DD_TRIGGER_10PCT:
            return 0.10f;
        default:
            return 0.0f;
    }
}

inline int trigger_losers_count(WFOTrigger t) {
    switch (t) {
        case WFOTrigger::LOSERS_5:
            return 5;
        case WFOTrigger::LOSERS_10:
            return 10;
        default:
            return 0;
    }
}

// ---------------------------------------------------------------------------
// WFO Schedule frequency as calendar period
// ---------------------------------------------------------------------------
enum class WFOFrequency : uint8_t {
    WEEK_1    = 0,  // 1 week
    WEEK_2    = 1,  // 2 weeks
    MONTH_1   = 2,  // 1 month
    QUARTER_1 = 3   // 1 quarter
};

// Approximate calendar days for each frequency
inline int frequency_approx_days(WFOFrequency f) {
    switch (f) {
        case WFOFrequency::WEEK_1:
            return 7;
        case WFOFrequency::WEEK_2:
            return 14;
        case WFOFrequency::MONTH_1:
            return 30;
        case WFOFrequency::QUARTER_1:
            return 91;
    }
    return 7;
}

inline const char* frequency_name(WFOFrequency f) {
    switch (f) {
        case WFOFrequency::WEEK_1:
            return "1w";
        case WFOFrequency::WEEK_2:
            return "2w";
        case WFOFrequency::MONTH_1:
            return "1mo";
        case WFOFrequency::QUARTER_1:
            return "1q";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// WFO Lookback period
// ---------------------------------------------------------------------------
enum class WFOLookback : uint8_t {
    MONTH_1 = 0,
    MONTH_2 = 1,
    MONTH_3 = 2,
    MONTH_6 = 3,
    YEAR_1  = 4,
    YEAR_2  = 5,
    YEAR_3  = 6,
    YEAR_5  = 7
};

// Approximate calendar days for each lookback
inline int lookback_approx_days(WFOLookback lb) {
    switch (lb) {
        case WFOLookback::MONTH_1:
            return 30;
        case WFOLookback::MONTH_2:
            return 61;
        case WFOLookback::MONTH_3:
            return 91;
        case WFOLookback::MONTH_6:
            return 183;
        case WFOLookback::YEAR_1:
            return 365;
        case WFOLookback::YEAR_2:
            return 730;
        case WFOLookback::YEAR_3:
            return 1095;
        case WFOLookback::YEAR_5:
            return 1826;
    }
    return 30;
}

inline const char* lookback_name(WFOLookback lb) {
    switch (lb) {
        case WFOLookback::MONTH_1:
            return "1mo";
        case WFOLookback::MONTH_2:
            return "2mo";
        case WFOLookback::MONTH_3:
            return "3mo";
        case WFOLookback::MONTH_6:
            return "6mo";
        case WFOLookback::YEAR_1:
            return "1y";
        case WFOLookback::YEAR_2:
            return "2y";
        case WFOLookback::YEAR_3:
            return "3y";
        case WFOLookback::YEAR_5:
            return "5y";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// WFOScheduleConfig -- one WFO schedule configuration
// ---------------------------------------------------------------------------
struct WFOScheduleConfig {
    WFOFrequency frequency = WFOFrequency::WEEK_1;
    WFOLookback  lookback  = WFOLookback::MONTH_1;
    WFOTrigger   trigger   = WFOTrigger::FIXED_INTERVAL;

    bool operator==(const WFOScheduleConfig& o) const {
        return frequency == o.frequency && lookback == o.lookback && trigger == o.trigger;
    }
    bool operator!=(const WFOScheduleConfig& o) const { return !(*this == o); }
};

// ---------------------------------------------------------------------------
// WFOGrid -- enumerates valid WFO schedule configurations
//
// Fixed-interval: 4 freq x 8 lookbacks = 32 raw, pruned to 30.
// Adaptive: 4 triggers x 8 lookbacks x 1 freq (1 week) = 32.
// Total: ~62 configs.
// ---------------------------------------------------------------------------
class WFOGrid {
public:
    WFOGrid();

    int size() const { return static_cast<int>(m_entries.size()); }

    const WFOScheduleConfig& config(int id) const { return m_entries[id]; }

    // Reverse lookup
    int find(const WFOScheduleConfig& c) const;

private:
    std::vector<WFOScheduleConfig> m_entries;
};
