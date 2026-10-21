#pragma once
#include <cstdint>
#include <vector>
#include "indicator_grid.h"

// ---------------------------------------------------------------------------
// WindowMode -- rolling vs anchored computation mode
// ---------------------------------------------------------------------------
enum class WindowMode : uint8_t {
    ROLLING      = 0,  // Standard computation on 30s bars
    ANCHORED_1M  = 1,  // Compute on 1-minute bars
    ANCHORED_2M  = 2,  // Compute on 2-minute bars
    ANCHORED_5M  = 3,  // Compute on 5-minute bars
    ANCHORED_15M = 4,  // Compute on 15-minute bars
    ANCHORED_30M = 5,  // Compute on 30-minute bars
    ANCHORED_1H  = 6   // Compute on 1-hour bars
};

// Timeframe in seconds for each window mode
inline int window_mode_seconds(WindowMode mode) {
    switch (mode) {
        case WindowMode::ROLLING:
            return 30;
        case WindowMode::ANCHORED_1M:
            return 60;
        case WindowMode::ANCHORED_2M:
            return 120;
        case WindowMode::ANCHORED_5M:
            return 300;
        case WindowMode::ANCHORED_15M:
            return 900;
        case WindowMode::ANCHORED_30M:
            return 1800;
        case WindowMode::ANCHORED_1H:
            return 3600;
    }
    return 30;
}

inline const char* window_mode_name(WindowMode mode) {
    switch (mode) {
        case WindowMode::ROLLING:
            return "rolling";
        case WindowMode::ANCHORED_1M:
            return "anchored_1m";
        case WindowMode::ANCHORED_2M:
            return "anchored_2m";
        case WindowMode::ANCHORED_5M:
            return "anchored_5m";
        case WindowMode::ANCHORED_15M:
            return "anchored_15m";
        case WindowMode::ANCHORED_30M:
            return "anchored_30m";
        case WindowMode::ANCHORED_1H:
            return "anchored_1h";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// WindowModeEntry -- one (indicator_id, window_mode) combo
// ---------------------------------------------------------------------------
struct WindowModeEntry {
    int        indicator_id = 0;  // index into IndicatorGrid
    WindowMode mode         = WindowMode::ROLLING;

    bool operator==(const WindowModeEntry& o) const {
        return indicator_id == o.indicator_id && mode == o.mode;
    }
    bool operator!=(const WindowModeEntry& o) const { return !(*this == o); }
};

// ---------------------------------------------------------------------------
// WindowModeGrid -- enumerates valid (indicator, window_mode) combinations
//
// Prunes nonsensical combos such as 1h anchored with period 200
// (which would mean 200 hours of lookback).
// ---------------------------------------------------------------------------
class WindowModeGrid {
public:
    // Builds the grid from an existing IndicatorGrid.
    explicit WindowModeGrid(const IndicatorGrid& ig);

    int size() const { return static_cast<int>(m_entries.size()); }

    const WindowModeEntry& entry(int id) const { return m_entries[id]; }

    // Reverse lookup: returns ID or -1
    int find(const WindowModeEntry& e) const;

    // Access the underlying indicator grid.
    // Lifetime requirement: the IndicatorGrid passed to the constructor must
    // outlive this WindowModeGrid. The pointer is non-owning.
    const IndicatorGrid& indicator_grid() const { return *m_indicator_grid; }

private:
    const IndicatorGrid*         m_indicator_grid;  // non-owning; must outlive this object
    std::vector<WindowModeEntry> m_entries;

    // Maximum effective lookback in hours before a combo is pruned.
    // Set to 7 days (168 hours) to allow long-period indicators on higher
    // timeframes for regime detection (e.g., EMA-200 on 1h = 200 hours,
    // ADX-28 on 1h = 28 hours, EMA-233 on 5m = 19.4 hours).
    static constexpr float MAX_LOOKBACK_HOURS = 168.0f;

    // Get the primary period for an indicator (used for pruning)
    static int primary_period(const IndicatorParams& p);

    // Check if (indicator, window_mode) is sensible
    static bool is_valid_combo(const IndicatorParams& p, WindowMode mode);
};
