#pragma once
// =============================================================================
// Session Signal Base -- common interface and constants for session-structure
// signals.
//
// Each signal module computes one or more float signals in [-1.0, +1.0] per
// session day.  Output convention:
//   +1.0  = maximum bullish
//   -1.0  = maximum bearish
//    0.0  = neutral
//   NaN   = signal not yet available (warmup period)
//
// No lookahead: signals on session day D use only completed bars within D
// (for intraday signals) or data from D-1 and earlier (for multi-day).
// =============================================================================

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <cassert>
#include "../common.h"
#include "../market_data.h"

// Sentinel for unavailable / warmup signals
constexpr float SIGNAL_NAN = std::numeric_limits<float>::quiet_NaN();

// ---------------------------------------------------------------------------
// Session timing constants (ET seconds since midnight)
// ---------------------------------------------------------------------------
namespace session_time {
constexpr int32_t RTH_OPEN       = 9 * 3600 + 30 * 60;   // 09:30 = 34200
constexpr int32_t OR_END         = 10 * 3600;            // 10:00 = 36000
constexpr int32_t IB_END         = 10 * 3600 + 30 * 60;  // 10:30 = 37800
constexpr int32_t LUNCH_START    = 11 * 3600 + 30 * 60;  // 11:30 = 41400
constexpr int32_t LUNCH_END      = 13 * 3600;            // 13:00 = 46800
constexpr int32_t AFTERNOON      = 14 * 3600;            // 14:00 = 50400
constexpr int32_t POWER_HOUR     = 15 * 3600;            // 15:00 = 54000
constexpr int32_t RTH_CLOSE      = 16 * 3600;            // 16:00 = 57600
constexpr int32_t SESSION_CLOSE  = 17 * 3600;            // 17:00 = 61200
constexpr int32_t SESSION_OPEN   = 18 * 3600;            // 18:00 = 64800
constexpr int32_t FIRST_15_END   = 9 * 3600 + 45 * 60;   // 09:45
constexpr int32_t POWER_HOUR_END = 15 * 3600 + 50 * 60;  // 15:50 = 57000
}  // namespace session_time

// ---------------------------------------------------------------------------
// SessionDayView -- provides a lightweight view into bars for one session day.
// Constructed from MarketData + session_day index.
// ---------------------------------------------------------------------------
struct SessionDayView {
    const Bar*         bars;       // pointer to full bar array
    const SessionInfo* sessions;   // pointer to full session info array
    int                first_bar;  // global index of first bar in this session
    int                bar_count;  // number of bars in this session

    // RTH sub-range (bars with time_of_day_sec in [RTH_OPEN, RTH_CLOSE))
    int rth_first;  // global index of first RTH bar (-1 if none)
    int rth_count;  // number of RTH bars

    // ETH sub-range (bars before RTH_OPEN)
    int eth_first;  // global index of first ETH bar (-1 if none)
    int eth_count;

    // Session OHLC (over all bars in the session)
    float session_open;
    float session_high;
    float session_low;
    float session_close;

    // RTH OHLC
    float rth_open;
    float rth_high;
    float rth_low;
    float rth_close;

    // ETH OHLC
    float eth_open;
    float eth_high;
    float eth_low;
    float eth_close;

    // Day range (RTH high - RTH low)
    float rth_range() const { return rth_high - rth_low; }
};

// Build a SessionDayView for a given session day.
// Returns a view with bar_count=0 if session_day is out of range or has no bars.
// NOTE: Callers that access the same session day multiple times should cache the
// returned view to avoid redundant computation.
inline SessionDayView build_session_view(const MarketData& md, int session_day) {
    SessionDayView v{};
    v.bars         = &md.bar(0);
    v.sessions     = &md.session(0);
    v.first_bar    = -1;
    v.bar_count    = 0;
    v.rth_first    = -1;
    v.rth_count    = 0;
    v.eth_first    = -1;
    v.eth_count    = 0;
    v.session_open = v.session_high = v.session_low = v.session_close = 0.0f;
    v.rth_open = v.rth_high = v.rth_low = v.rth_close = 0.0f;
    v.eth_open = v.eth_high = v.eth_low = v.eth_close = 0.0f;

    if (session_day < 0 || session_day >= md.session_count()) return v;

    const int total = md.bar_count();

    // O(1) lookup via precomputed reverse mapping
    int start = md.first_bar_of_session(session_day);
    if (start < 0) return v;

    // Use first_bar_of_session(session_day + 1) for O(1) end-boundary
    // lookup instead of a linear scan.  Fall back to linear scan only for the
    // last session day (where no next session exists).
    int end;
    if (session_day + 1 < md.session_count()) {
        int next_start = md.first_bar_of_session(session_day + 1);
        end            = (next_start > start) ? next_start : total;
    } else {
        end = start;
        while (end < total && md.session(end).session_day == session_day) ++end;
    }

    v.first_bar = start;
    v.bar_count = end - start;

    // Session OHLC
    v.session_open  = md.bar(start).open;
    v.session_close = md.bar(end - 1).close;
    v.session_high  = md.bar(start).high;
    v.session_low   = md.bar(start).low;
    for (int i = start; i < end; ++i) {
        v.session_high = std::max(v.session_high, md.bar(i).high);
        v.session_low  = std::min(v.session_low, md.bar(i).low);
    }

    // RTH / ETH splits
    for (int i = start; i < end; ++i) {
        int32_t tod = md.session(i).time_of_day_sec;
        if (tod >= session_time::RTH_OPEN && tod < session_time::RTH_CLOSE) {
            if (v.rth_first < 0) {
                v.rth_first = i;
                v.rth_open  = md.bar(i).open;
                v.rth_high  = md.bar(i).high;
                v.rth_low   = md.bar(i).low;
            }
            v.rth_high  = std::max(v.rth_high, md.bar(i).high);
            v.rth_low   = std::min(v.rth_low, md.bar(i).low);
            v.rth_close = md.bar(i).close;
            v.rth_count++;
        } else if (tod < session_time::RTH_OPEN || tod >= session_time::SESSION_OPEN) {
            // ETH = pre-market (00:00-09:30) + evening (18:00-midnight).
            // Post-RTH bars (16:00-18:00) are intentionally excluded -- they
            // fall between RTH close and session open/maintenance window.
            if (v.eth_first < 0) {
                v.eth_first = i;
                v.eth_open  = md.bar(i).open;
                v.eth_high  = md.bar(i).high;
                v.eth_low   = md.bar(i).low;
            }
            v.eth_high  = std::max(v.eth_high, md.bar(i).high);
            v.eth_low   = std::min(v.eth_low, md.bar(i).low);
            v.eth_close = md.bar(i).close;
            v.eth_count++;
        }
    }

    return v;
}

// ---------------------------------------------------------------------------
// Helper: compute daily ATR from prior session day views (no lookahead).
// Uses true range = max(H-L, |H-prev_close|, |L-prev_close|) as a simple
// average over `lookback` sessions.  This is intentionally a simple average
// (not Wilder/EMA smoothing) because session-level ATR for regime
// classification benefits from equal weighting of all sessions in the window.
// Returns SIGNAL_NAN if not enough history.
// ---------------------------------------------------------------------------
inline float compute_session_atr(const MarketData& md, int session_day, int lookback) {
    if (session_day < lookback) return SIGNAL_NAN;

    // Collect daily true ranges for [session_day - lookback, session_day - 1]
    // (no lookahead: we do NOT include session_day itself)
    float atr        = 0.0f;
    float prev_close = SIGNAL_NAN;
    int   count      = 0;

    for (int d = session_day - lookback; d < session_day; ++d) {
        SessionDayView dv = build_session_view(md, d);
        if (dv.rth_count == 0) continue;

        float tr;
        if (std::isnan(prev_close)) {
            tr = dv.rth_range();
        } else {
            float hl  = dv.rth_range();
            float hpc = std::abs(dv.rth_high - prev_close);
            float lpc = std::abs(dv.rth_low - prev_close);
            tr        = std::max({hl, hpc, lpc});
        }

        // Simple running average over all sessions in the window
        atr = (atr * static_cast<float>(count) + tr) / static_cast<float>(count + 1);
        ++count;
        prev_close = dv.rth_close;
    }

    if (count < lookback / 2) return SIGNAL_NAN;  // not enough data
    return atr;
}

// clamp_signal is provided by common.h

// ---------------------------------------------------------------------------
// Helper: direction sign (-1, 0, +1) with a neutral dead-zone.
// ---------------------------------------------------------------------------
inline float direction_signal(float from, float to, float dead_zone = 0.0f) {
    float diff = to - from;
    if (std::abs(diff) <= dead_zone) return 0.0f;
    return diff > 0.0f ? 1.0f : -1.0f;
}

// ---------------------------------------------------------------------------
// SessionSignal -- base class for all session-structure signal modules.
// ---------------------------------------------------------------------------
class SessionSignal {
public:
    virtual ~SessionSignal() = default;

    // Compute signals for a range of session days [day_start, day_end).
    // Output layout: output[signal_idx * num_days + day_offset] where
    //   num_days = day_end - day_start
    //   day_offset = day - day_start
    virtual void compute(const MarketData& md, int day_start, int day_end, float* output) const = 0;

    // Number of signal series produced.
    virtual int num_signals() const = 0;

    // Name for debugging.
    virtual const char* name() const = 0;

    // Minimum number of prior session days needed for warmup.
    virtual int warmup_days() const = 0;

    // -----------------------------------------------------------------------
    // Halt suppression flag.
    //
    // Returns true if this signal module's outputs should be suppressed
    // (set to 0.0 / neutral) on sessions flagged with halt_flag == 1.
    //
    // Default: false (signals derived from session-level aggregates are
    // unaffected by halts).  Intraday structure and level interaction
    // signals override this to return true.
    // -----------------------------------------------------------------------
    virtual bool suppress_on_halt() const { return false; }

    // -----------------------------------------------------------------------
    // Time-availability metadata for each signal output.
    //
    // Returns a vector of size num_signals().  Each element is the ET
    // seconds-since-midnight at which that signal becomes available during
    // a session day.  Bars whose time_of_day_sec is BEFORE this value must
    // treat the signal as unavailable (0.0 / neutral).
    //
    // Convention:
    //   0         = available at session open (18:00 ET prior day / ETH start)
    //   34200     = available at RTH open (09:30 ET)
    //   36000     = available after OR end (10:00 ET)
    //   37800     = available after IB end (10:30 ET)
    //   57600     = available after RTH close (16:00 ET) -- retrospective signals
    //
    // This metadata is used by broadcast_session_signals_to_bars() and the
    // backtest engine to prevent lookahead: a bar at 09:35 must not see a
    // signal whose available_after is 10:00.
    // -----------------------------------------------------------------------
    virtual std::vector<int32_t> signal_available_after_et_seconds() const = 0;
};

// ---------------------------------------------------------------------------
// broadcast_session_signal_to_bars -- expand one per-session-day signal
// series into a per-bar array without lookahead.
//
// A bar sees its own session's value once the session has reached the
// signal's availability time, and 0 (neutral) before that. Signals that only
// become available at or after the core close (RTH_CLOSE) describe a whole
// regular session; they are meant for the next session, so before their
// availability time a bar sees the previous session's value instead.
//
// Parameters:
//   md                - market data (for session info lookups)
//   signal_values     - per-day signal array (layout: signal_values[day_offset])
//                       for ONE signal series, covering [day_start, day_end)
//   day_start         - first session day index in signal_values
//   day_end           - one past last session day index
//   available_after   - ET seconds-since-midnight when this signal becomes valid
//   bar_start         - first bar index for output
//   bar_end           - one past last bar index for output
//   output            - pre-allocated float array of size (bar_end - bar_start)
//   suppress_on_halt  - zero the value when its source session was halted
// ---------------------------------------------------------------------------
inline void broadcast_session_signal_to_bars(const MarketData& md, const float* signal_values,
                                             int day_start, int day_end, int32_t available_after,
                                             int bar_start, int bar_end, float* output,
                                             bool suppress_on_halt = false) {
    const int     num_bars     = bar_end - bar_start;
    const int32_t open         = md.session_model().session_open_sec;
    const bool    carries_over = available_after >= session_time::RTH_CLOSE;

    for (int i = 0; i < num_bars; ++i) {
        output[i] = 0.0f;

        const SessionInfo& si  = md.session(bar_start + i);
        const int          day = si.session_day;
        if (day < day_start || day >= day_end) continue;

        // Available today: the session has reached the availability time,
        // comparing elapsed time since the open rather than raw clock time
        // (an 18:00 open sees 19:00 before 10:00). On an early-close session
        // a signal timed after the close never becomes available.
        bool available_today = available_after <= 0;
        if (!available_today) {
            const bool closed_before = si.early_close && available_after > si.session_close_sec;
            available_today = !closed_before && seconds_since_open(si.time_of_day_sec, open) >=
                                                    seconds_since_open(available_after, open);
        }

        int  source_day = day;
        bool halted     = si.halt_flag;  // set from a session's halt onward
        if (!available_today) {
            if (!carries_over || day - 1 < day_start) continue;
            source_day = day - 1;
            // The previous session is complete; its last bar carries its halt flag.
            const int prev_last = md.first_bar_of_session(day) - 1;
            halted              = prev_last >= 0 && md.session(prev_last).halt_flag;
        }

        // Intraday-structure and level-interaction signals are meaningless
        // on a halted session; aggregate signals (gaps, calendar, overnight
        // context) pass suppress_on_halt = false.
        if (suppress_on_halt && halted) continue;

        const float val = signal_values[source_day - day_start];
        if (!std::isnan(val)) output[i] = val;  // NaN marks warmup: neutral
    }
}

// ---------------------------------------------------------------------------
// Helper: find bars in a time window within a session.
// Returns (first_global_idx, count). count=0 if nothing found.
// ---------------------------------------------------------------------------
inline std::pair<int, int> bars_in_time_range(const MarketData& md, const SessionDayView& view,
                                              int32_t time_start_sec, int32_t time_end_sec) {
    int first = -1;
    int count = 0;
    for (int i = view.first_bar; i < view.first_bar + view.bar_count; ++i) {
        int32_t t = md.session(i).time_of_day_sec;
        if (t >= time_start_sec && t < time_end_sec) {
            if (first < 0) first = i;
            count++;
        }
    }
    return {first, count};
}

// ---------------------------------------------------------------------------
// Helper: compute volume-weighted average price over a bar range.
// ---------------------------------------------------------------------------
inline float compute_vwap(const MarketData& md, int first_bar, int bar_count) {
    if (bar_count <= 0) return SIGNAL_NAN;
    double pv_sum = 0.0;
    double v_sum  = 0.0;
    for (int i = first_bar; i < first_bar + bar_count; ++i) {
        double typical = (static_cast<double>(md.bar(i).high) + static_cast<double>(md.bar(i).low) +
                          static_cast<double>(md.bar(i).close)) /
                         3.0;
        double vol = static_cast<double>(md.bar(i).volume);
        pv_sum += typical * vol;
        v_sum += vol;
    }
    if (v_sum <= 0.0) return SIGNAL_NAN;
    return static_cast<float>(pv_sum / v_sum);
}

// ---------------------------------------------------------------------------
// Helper: compute Point of Control (volume profile) over a bar range.
//
// Bins each bar's volume at its close price using
// 1-tick bins, finds the bin with highest cumulative volume. Ties broken
// by proximity to session VWAP.
//
// tick_size = minimum price increment (0.25 for ES/NQ).
// Scope should be the full session (18:00 ET open to 17:00 ET close).
// ---------------------------------------------------------------------------
inline float compute_poc(const MarketData& md, int first_bar, int bar_count, float tick_size) {
    if (bar_count <= 0 || tick_size <= 0.0f) return SIGNAL_NAN;

    // Find price range to determine bin count
    float min_price = md.bar(first_bar).close;
    float max_price = min_price;
    for (int i = first_bar; i < first_bar + bar_count; ++i) {
        float c = md.bar(i).close;
        if (c < min_price) min_price = c;
        if (c > max_price) max_price = c;
    }

    // Create bins: one per tick across the session's close-price range
    int num_bins = static_cast<int>((max_price - min_price) / tick_size) + 1;
    if (num_bins <= 0 || num_bins > 100000) return SIGNAL_NAN;  // sanity check

    std::vector<uint64_t> bin_volume(num_bins, 0);

    // Assign each bar's volume to its close-price bin
    for (int i = first_bar; i < first_bar + bar_count; ++i) {
        float c   = md.bar(i).close;
        int   bin = static_cast<int>((c - min_price) / tick_size);
        if (bin >= 0 && bin < num_bins) {
            bin_volume[bin] += md.bar(i).volume;
        }
    }

    // Find bin with highest volume; tie-break by proximity to VWAP
    uint64_t max_vol   = 0;
    int      best_bin  = 0;
    float    vwap      = compute_vwap(md, first_bar, bar_count);
    float    best_dist = std::numeric_limits<float>::max();

    for (int b = 0; b < num_bins; ++b) {
        if (bin_volume[b] > max_vol) {
            max_vol   = bin_volume[b];
            best_bin  = b;
            best_dist = std::abs((min_price + b * tick_size) - vwap);
        } else if (bin_volume[b] == max_vol) {
            float dist = std::abs((min_price + b * tick_size) - vwap);
            if (dist < best_dist) {
                best_bin  = b;
                best_dist = dist;
            }
        }
    }

    if (max_vol == 0) return SIGNAL_NAN;
    return min_price + best_bin * tick_size;
}
