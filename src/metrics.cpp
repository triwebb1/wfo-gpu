#include "metrics.h"
#include "backtest_engine.h"
#include "common.h"
#include "cost_model.h"
#include "market_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <limits>
#include <numeric>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int64_t NS_PER_SEC            = 1'000'000'000LL;
static constexpr int64_t NS_PER_DAY            = 86400LL * NS_PER_SEC;
static constexpr double  TRADING_DAYS_PER_YEAR = 252.0;

// ---------------------------------------------------------------------------
// Calendar helpers -- convert UTC nanosecond timestamps to date components
// ---------------------------------------------------------------------------

// Convert UTC nanoseconds to a tm struct (UTC).
static std::tm ns_to_utc_tm(int64_t ts_ns) {
    const time_t secs = static_cast<time_t>(ts_ns / NS_PER_SEC);
    std::tm      result{};
#ifdef _WIN32
    gmtime_s(&result, &secs);
#else
    gmtime_r(&secs, &result);
#endif
    return result;
}

// ISO week number (Monday-based) and year for weekly bucketing.
// Returns (iso_year, iso_week) pair encoded as year * 100 + week.
static int iso_year_week(int64_t ts_ns) {
    const std::tm t = ns_to_utc_tm(ts_ns);

    // Day of year (0-based)
    const int yday = t.tm_yday;

    // Day of week: Monday=0 .. Sunday=6  (tm_wday: Sunday=0 .. Saturday=6)
    int dow = (t.tm_wday + 6) % 7;

    // ISO week calculation
    int week = (yday - dow + 10) / 7;
    int year = t.tm_year + 1900;

    if (week < 1) {
        // Belongs to last week of previous year.
        year -= 1;
        week = iso_weeks_in_year(year);
    } else if (week > 52) {
        // Check if it belongs to week 1 of next year.
        // ISO 8601: week 53 exists only if Dec 31 is Thursday or later.
        // Approximate: if week > 53, or if week == 53 and Dec 31 is Mon-Wed,
        // then this is actually week 1 of the next year.
        // Compute day-of-week for Dec 31 of this year:
        // Dec 31 is day 364 or 365 (leap year) from Jan 1.
        // We can derive from yday and wday: Dec 31 wday = (t.tm_wday + (364 - yday)) % 7
        // Convert to Monday=0: dec31_dow_mon = (dec31_wday_sun + 6) % 7
        const int days_to_dec31 = 364 + (is_leap_year(year) ? 1 : 0) - yday;
        const int dec31_wday    = (t.tm_wday + days_to_dec31) % 7;  // Sunday=0 based
        const int dec31_dow_mon = (dec31_wday + 6) % 7;             // Monday=0 based
        if (week > 53 || (week == 53 && dec31_dow_mon < 3)) {
            year += 1;
            week = 1;
        }
    }

    return year * 100 + week;
}

// (year, month) from timestamp
static int year_month(int64_t ts_ns) {
    const std::tm t = ns_to_utc_tm(ts_ns);
    return (t.tm_year + 1900) * 100 + (t.tm_mon + 1);
}

// Year from timestamp
static int year_of(int64_t ts_ns) {
    const std::tm t = ns_to_utc_tm(ts_ns);
    return t.tm_year + 1900;
}

// ---------------------------------------------------------------------------
// ET offset helper -- derive ET offset from SessionInfo + bar timestamp
// ---------------------------------------------------------------------------

// Compute the UTC-to-ET offset in seconds for a bar, using the SessionInfo's
// time_of_day_sec (ET seconds since midnight) and the bar's UTC timestamp.
// Returns a negative value (e.g. -18000 for EST, -14400 for EDT).
// This lets us shift a UTC timestamp to ET for calendar date extraction.
static int32_t compute_et_offset_sec(int64_t bar_ts_ns, int32_t et_time_of_day_sec) {
    const int64_t utc_secs = bar_ts_ns / NS_PER_SEC;
    const int32_t utc_sod  = static_cast<int32_t>(
        ((utc_secs % 86400) + 86400) % 86400);  // UTC seconds since midnight, always positive
    int32_t offset = et_time_of_day_sec - utc_sod;
    // The offset should be in the range [-18000, -14400] (ET is behind UTC).
    // Handle wrap-around: if ET midnight is in a different UTC day, the raw
    // difference can be off by +/-86400.
    if (offset > 43200) offset -= 86400;
    if (offset < -43200) offset += 86400;
    return offset;
}

// Shift a UTC nanosecond timestamp to ET using the derived offset, then
// extract calendar info via ns_to_utc_tm() (which treats the shifted value
// as if it were UTC, giving us ET calendar dates).
static int64_t utc_ns_to_et_ns(int64_t bar_ts_ns, int32_t et_offset_sec) {
    return bar_ts_ns + static_cast<int64_t>(et_offset_sec) * NS_PER_SEC;
}

// Overloads of iso_year_week / year_month / year_of that accept SessionInfo
// for ET-correct calendar extraction.
static int iso_year_week_et(int64_t bar_ts_ns, const SessionInfo& si) {
    const int32_t offset = compute_et_offset_sec(bar_ts_ns, si.time_of_day_sec);
    return iso_year_week(utc_ns_to_et_ns(bar_ts_ns, offset));
}

static int year_month_et(int64_t bar_ts_ns, const SessionInfo& si) {
    const int32_t offset = compute_et_offset_sec(bar_ts_ns, si.time_of_day_sec);
    return year_month(utc_ns_to_et_ns(bar_ts_ns, offset));
}

static int year_of_et(int64_t bar_ts_ns, const SessionInfo& si) {
    const int32_t offset = compute_et_offset_sec(bar_ts_ns, si.time_of_day_sec);
    return year_of(utc_ns_to_et_ns(bar_ts_ns, offset));
}

// ---------------------------------------------------------------------------
// bucket_weekly_pnl
// ---------------------------------------------------------------------------
std::vector<WeeklyPnL> bucket_weekly_pnl(const std::vector<Trade>& trades, const Bar* bars,
                                         int bar_start, int bar_end, const SessionInfo* sessions,
                                         const int* sd_to_week, int sd_range_first,
                                         int sd_range_last) {
    if (trades.empty()) return {};

    // Map from ISO year-week key to accumulated PnL
    std::unordered_map<int, float>   week_pnl;
    std::unordered_map<int, int64_t> week_start_ts;

    for (const auto& t : trades) {
        // Use exit bar timestamp to assign the trade to a week
        const int     exit_bar = std::min(t.exit_bar, bar_end - 1);
        const int64_t ts       = bars[exit_bar].timestamp;
        int           key;
        if (sd_to_week && sessions) {
            // Fast path: O(1) lookup from precomputed table
            key = sd_to_week[sessions[exit_bar].session_day];
        } else {
            key = sessions ? iso_year_week_et(ts, sessions[exit_bar]) : iso_year_week(ts);
        }

        week_pnl[key] += t.net_pnl;
        if (week_start_ts.find(key) == week_start_ts.end() || ts < week_start_ts[key]) {
            week_start_ts[key] = ts;
        }
    }

    // Also include weeks in the evaluation range that had no trades (PnL = 0).
    if (sd_to_week && sessions && sd_range_first <= sd_range_last) {
        // Fast path: O(session_days) scan using precomputed mappings
        int prev_week_key = -1;
        for (int sd = sd_range_first; sd <= sd_range_last; ++sd) {
            const int key = sd_to_week[sd];
            if (key != prev_week_key) {
                if (week_pnl.find(key) == week_pnl.end()) {
                    week_pnl[key] = 0.0f;
                    // Use a placeholder timestamp (not critical for
                    // consistency metrics -- only ordering matters)
                    week_start_ts[key] = bars[bar_start].timestamp;
                }
                prev_week_key = key;
            }
        }
    } else {
        // Original O(bars) scan for all unique weeks present in the data.
        int prev_week_key = -1;
        for (int i = bar_start; i < bar_end; ++i) {
            const int key = sessions ? iso_year_week_et(bars[i].timestamp, sessions[i])
                                     : iso_year_week(bars[i].timestamp);
            if (key != prev_week_key) {
                if (week_pnl.find(key) == week_pnl.end()) {
                    week_pnl[key]      = 0.0f;
                    week_start_ts[key] = bars[i].timestamp;
                }
                prev_week_key = key;
            }
        }
    }

    // Convert to sorted vector -- sort by ISO year-week key (not timestamp)
    // to maintain chronological order even when fast-path uses placeholder ts.
    std::vector<std::pair<int, float>> key_pnl_pairs;
    key_pnl_pairs.reserve(week_pnl.size());
    for (const auto& [key, pnl] : week_pnl) {
        key_pnl_pairs.push_back({key, pnl});
    }
    std::sort(key_pnl_pairs.begin(), key_pnl_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<WeeklyPnL> result;
    result.reserve(key_pnl_pairs.size());
    for (const auto& [key, pnl] : key_pnl_pairs) {
        WeeklyPnL wp;
        wp.week_start_ts =
            week_start_ts.count(key) ? week_start_ts[key] : bars[bar_start].timestamp;
        wp.pnl = pnl;
        result.push_back(wp);
    }

    return result;
}

// ---------------------------------------------------------------------------
// bucket_monthly_pnl
// ---------------------------------------------------------------------------
std::vector<MonthlyPnL> bucket_monthly_pnl(const std::vector<Trade>& trades, const Bar* bars,
                                           int bar_start, int bar_end, const SessionInfo* sessions,
                                           const int* sd_to_month, int sd_range_first,
                                           int sd_range_last) {
    if (trades.empty()) return {};

    std::unordered_map<int, float> month_pnl;

    for (const auto& t : trades) {
        const int exit_bar = std::min(t.exit_bar, bar_end - 1);
        int       key;
        if (sd_to_month && sessions) {
            key = sd_to_month[sessions[exit_bar].session_day];
        } else {
            key = sessions ? year_month_et(bars[exit_bar].timestamp, sessions[exit_bar])
                           : year_month(bars[exit_bar].timestamp);
        }
        month_pnl[key] += t.net_pnl;
    }

    // Include months in range with no trades
    if (sd_to_month && sessions && sd_range_first <= sd_range_last) {
        // Fast path: derive start/end months from precomputed table
        int start_ym = sd_to_month[sd_range_first];
        int end_ym   = sd_to_month[sd_range_last];
        int y        = start_ym / 100;
        int m        = start_ym % 100;

        while (y * 100 + m <= end_ym) {
            const int key = y * 100 + m;
            if (month_pnl.find(key) == month_pnl.end()) {
                month_pnl[key] = 0.0f;
            }
            ++m;
            if (m > 12) {
                m = 1;
                ++y;
            }
        }
    } else {
        // Use ET-shifted timestamps for range boundaries when sessions available
        std::tm start_tm, end_tm;
        if (sessions) {
            const int32_t s_off = compute_et_offset_sec(bars[bar_start].timestamp,
                                                        sessions[bar_start].time_of_day_sec);
            const int32_t e_off = compute_et_offset_sec(bars[bar_end - 1].timestamp,
                                                        sessions[bar_end - 1].time_of_day_sec);
            start_tm            = ns_to_utc_tm(utc_ns_to_et_ns(bars[bar_start].timestamp, s_off));
            end_tm              = ns_to_utc_tm(utc_ns_to_et_ns(bars[bar_end - 1].timestamp, e_off));
        } else {
            start_tm = ns_to_utc_tm(bars[bar_start].timestamp);
            end_tm   = ns_to_utc_tm(bars[bar_end - 1].timestamp);
        }
        int       y      = start_tm.tm_year + 1900;
        int       m      = start_tm.tm_mon + 1;
        const int end_ym = (end_tm.tm_year + 1900) * 100 + (end_tm.tm_mon + 1);

        while (y * 100 + m <= end_ym) {
            const int key = y * 100 + m;
            if (month_pnl.find(key) == month_pnl.end()) {
                month_pnl[key] = 0.0f;
            }
            ++m;
            if (m > 12) {
                m = 1;
                ++y;
            }
        }
    }

    std::vector<MonthlyPnL> result;
    result.reserve(month_pnl.size());
    for (const auto& [key, pnl] : month_pnl) {
        MonthlyPnL mp;
        mp.year  = key / 100;
        mp.month = key % 100;
        mp.pnl   = pnl;
        result.push_back(mp);
    }
    std::sort(result.begin(), result.end(), [](const MonthlyPnL& a, const MonthlyPnL& b) {
        return (a.year * 100 + a.month) < (b.year * 100 + b.month);
    });

    return result;
}

// ---------------------------------------------------------------------------
// bucket_yearly_pnl
// ---------------------------------------------------------------------------
std::vector<YearlyPnL> bucket_yearly_pnl(const std::vector<Trade>& trades, const Bar* bars,
                                         int bar_start, int bar_end, const SessionInfo* sessions,
                                         const int* sd_to_year, int sd_range_first,
                                         int sd_range_last) {
    if (trades.empty()) return {};

    std::unordered_map<int, float> year_pnl;

    for (const auto& t : trades) {
        const int exit_bar = std::min(t.exit_bar, bar_end - 1);
        int       y;
        if (sd_to_year && sessions) {
            y = sd_to_year[sessions[exit_bar].session_day];
        } else {
            y = sessions ? year_of_et(bars[exit_bar].timestamp, sessions[exit_bar])
                         : year_of(bars[exit_bar].timestamp);
        }
        year_pnl[y] += t.net_pnl;
    }

    // Include years in range
    if (sd_to_year && sessions && sd_range_first <= sd_range_last) {
        const int start_y = sd_to_year[sd_range_first];
        const int end_y   = sd_to_year[sd_range_last];
        for (int y = start_y; y <= end_y; ++y) {
            if (year_pnl.find(y) == year_pnl.end()) {
                year_pnl[y] = 0.0f;
            }
        }
    } else {
        const int start_y = sessions ? year_of_et(bars[bar_start].timestamp, sessions[bar_start])
                                     : year_of(bars[bar_start].timestamp);
        const int end_y = sessions ? year_of_et(bars[bar_end - 1].timestamp, sessions[bar_end - 1])
                                   : year_of(bars[bar_end - 1].timestamp);
        for (int y = start_y; y <= end_y; ++y) {
            if (year_pnl.find(y) == year_pnl.end()) {
                year_pnl[y] = 0.0f;
            }
        }
    }

    std::vector<YearlyPnL> result;
    result.reserve(year_pnl.size());
    for (const auto& [y, pnl] : year_pnl) {
        YearlyPnL yp;
        yp.year = y;
        yp.pnl  = pnl;
        result.push_back(yp);
    }
    std::sort(result.begin(), result.end(),
              [](const YearlyPnL& a, const YearlyPnL& b) { return a.year < b.year; });

    return result;
}

// ---------------------------------------------------------------------------
// compute_metrics
// ---------------------------------------------------------------------------

BacktestResult compute_metrics(const std::vector<Trade>& trades, const Bar* bars, int bar_start,
                               int bar_end, const CostModel& cost, const SessionInfo* sessions,
                               int timeframe_sec, const int* sd_to_week, const int* sd_to_month,
                               const int* sd_to_year, int sd_range_first, int sd_range_last) {
    BacktestResult r;

    // -- Guard: zero-bar range is invalid ----------------------------------
    if (bar_end <= bar_start) return r;

    // -- Time period ------------------------------------------------------
    r.start_ts = bars[bar_start].timestamp;
    r.end_ts   = bars[bar_end - 1].timestamp;

    const double duration_ns    = static_cast<double>(r.end_ts - r.start_ts);
    const double duration_days  = duration_ns / static_cast<double>(NS_PER_DAY);
    const double duration_weeks = duration_days / 7.0;
    const double duration_years = duration_days / 365.25;

    r.total_trades = static_cast<int>(trades.size());

    if (r.total_trades == 0) {
        // No trades: all metrics remain at zero/default. metrics_complete
        // intentionally left false -- downstream robustness center-guard
        // short-circuits on green_weeks_pct==0 / avg_win==0 anyway.
        return r;
    }

    // -- Trades per week --------------------------------------------------
    r.trades_per_week =
        (duration_weeks > 0.0) ? static_cast<float>(r.total_trades / duration_weeks) : 0.0f;

    // -- Win/loss stats ---------------------------------------------------
    int   win_count       = 0;
    float sum_wins        = 0.0f;
    float sum_losses      = 0.0f;
    float total_hold_bars = 0.0f;
    float total_net_pnl   = 0.0f;

    // Per-trade daily PnL accumulator for Sharpe computation
    // We collect per-trade net PnL and assign to "trade days" for Sharpe.
    // A more precise approach uses actual daily bucketing, but per-trade
    // is a reasonable approximation when trades don't overlap (which they
    // don't, due to the single-position constraint).

    // Equity curve for drawdown
    float equity      = 0.0f;
    float peak_equity = 0.0f;
    float max_dd      = 0.0f;

    // Max live drawdown (worst per-trade unrealized M2M)
    float worst_min_unrealized = 0.0f;

    // Accumulators for avg_trade_dd and avg_trade_peak
    float sum_min_unrealized = 0.0f;
    float sum_max_unrealized = 0.0f;

    for (const auto& t : trades) {
        // Guard: reject NaN/Inf trade PnL -- indicates upstream bug.
        // Return default (zero) metrics rather than propagating poison.
        if (std::isnan(t.net_pnl) || std::isinf(t.net_pnl)) {
            return BacktestResult{};
        }
        // Also reject NaN/Inf in per-trade extremes (min/max unrealized).
        // These feed avg_trade_dd and avg_trade_peak; poisoning propagates into
        // Pareto dimensions (TRR) and selection metrics. Match net_pnl behavior.
        if (!std::isfinite(t.min_unrealized) || !std::isfinite(t.max_unrealized)) {
            return BacktestResult{};
        }
        total_net_pnl += t.net_pnl;
        total_hold_bars += static_cast<float>(t.exit_bar - t.entry_bar);

        if (t.net_pnl > 0.0f) {
            ++win_count;
            sum_wins += t.net_pnl;
        } else {
            sum_losses += t.net_pnl;  // negative
        }

        // Equity curve drawdown (realized)
        equity += t.net_pnl;
        if (equity > peak_equity) peak_equity = equity;
        const float dd = peak_equity - equity;
        if (dd > max_dd) max_dd = dd;

        // Per-trade worst unrealized
        if (t.min_unrealized < worst_min_unrealized) {
            worst_min_unrealized = t.min_unrealized;
        }

        // Accumulate per-trade extremes for avg_trade_dd and avg_trade_peak
        sum_min_unrealized += t.min_unrealized;
        sum_max_unrealized += t.max_unrealized;
    }

    const int loss_count = r.total_trades - win_count;

    r.win_rate      = static_cast<float>(win_count) / static_cast<float>(r.total_trades);
    r.avg_win       = (win_count > 0) ? (sum_wins / static_cast<float>(win_count)) : 0.0f;
    r.avg_loss      = (loss_count > 0) ? (sum_losses / static_cast<float>(loss_count)) : 0.0f;
    r.avg_hold_bars = total_hold_bars / static_cast<float>(r.total_trades);

    r.total_pnl    = total_net_pnl;
    r.gross_wins   = sum_wins;
    r.gross_losses = sum_losses;

    r.max_drawdown      = max_dd;
    r.max_live_drawdown = -worst_min_unrealized;  // stored as positive magnitude
    r.avg_trade_dd =
        -(sum_min_unrealized / static_cast<float>(r.total_trades));  // positive magnitude
    r.avg_trade_peak = sum_max_unrealized / static_cast<float>(r.total_trades);

    // -- Profit factor ----------------------------------------------------
    r.profit_factor = compute_profit_factor(sum_wins, sum_losses);

    // -- Payoff ratio -----------------------------------------------------
    r.payoff_ratio = compute_payoff_ratio(r.avg_win, r.avg_loss);

    // -- Sharpe ratio (annualized) ----------------------------------------
    // Compute daily PnL by bucketing trades into trading sessions.
    // When session_days is provided (from MarketData session index), use
    // the DST-correct session day assignment directly. Otherwise fall back
    // to the fixed UTC timestamp shift (approximate, off by ~1 hour during EST).
    {
        std::unordered_map<int64_t, float> daily_pnl;

        if (sessions) {
            // DST-correct path: use session_day index from MarketData's
            // session index. Each bar's SessionInfo::session_day is assigned
            // using ET-aware boundary detection, so this correctly handles
            // the DST transition regardless of whether we're in EST or EDT.
            for (const auto& t : trades) {
                const int     exit_bar = std::min(t.exit_bar, bar_end - 1);
                const int64_t day_key  = static_cast<int64_t>(sessions[exit_bar].session_day);
                daily_pnl[day_key] += t.net_pnl;
            }

            // Include all session days in the range (0 PnL for no-trade days)
            {
                int32_t prev_day = -1;
                for (int i = bar_start; i < bar_end; ++i) {
                    const int32_t sd = sessions[i].session_day;
                    if (sd != prev_day) {
                        const int64_t day_key = static_cast<int64_t>(sd);
                        if (daily_pnl.find(day_key) == daily_pnl.end()) {
                            daily_pnl[day_key] = 0.0f;
                        }
                        prev_day = sd;
                    }
                }
            }
        } else {
            // Fallback: fixed UTC timestamp shift (approximate, for callers
            // without session index access, e.g. test harnesses)
            static constexpr int64_t SESSION_SHIFT_NS = 22LL * 3600LL * NS_PER_SEC;

            for (const auto& t : trades) {
                const int     exit_bar = std::min(t.exit_bar, bar_end - 1);
                const int64_t day_key  = (bars[exit_bar].timestamp - SESSION_SHIFT_NS) / NS_PER_DAY;
                daily_pnl[day_key] += t.net_pnl;
            }

            {
                int64_t prev_day = -1;
                for (int i = bar_start; i < bar_end; ++i) {
                    const int64_t day_key = (bars[i].timestamp - SESSION_SHIFT_NS) / NS_PER_DAY;
                    if (day_key != prev_day) {
                        if (daily_pnl.find(day_key) == daily_pnl.end()) {
                            daily_pnl[day_key] = 0.0f;
                        }
                        prev_day = day_key;
                    }
                }
            }
        }

        if (daily_pnl.size() >= 2) {
            const int n_days = static_cast<int>(daily_pnl.size());
            double    sum    = 0.0;
            for (const auto& [_, pnl] : daily_pnl) {
                sum += static_cast<double>(pnl);
            }
            const double mean = sum / n_days;

            double var = 0.0;
            for (const auto& [_, pnl] : daily_pnl) {
                const double diff = static_cast<double>(pnl) - mean;
                var += diff * diff;
            }
            var /= (n_days - 1);  // sample variance
            const double std_dev = std::sqrt(var);

            r.sharpe_ratio =
                (std_dev > 0.0)
                    ? static_cast<float>(mean / std_dev * std::sqrt(TRADING_DAYS_PER_YEAR))
                    : 0.0f;
        }
    }

    // -- Calmar ratio -----------------------------------------------------
    if (duration_years > 0.0 && r.max_drawdown > 0.0f) {
        const float annualized_pnl =
            static_cast<float>(static_cast<double>(r.total_pnl) / duration_years);
        r.calmar_ratio = annualized_pnl / r.max_drawdown;
    }

    // -- Weekly consistency -----------------------------------------------
    {
        auto weeks = bucket_weekly_pnl(trades, bars, bar_start, bar_end, sessions, sd_to_week,
                                       sd_range_first, sd_range_last);
        if (!weeks.empty()) {
            int green_weeks    = 0;
            int max_red_streak = 0;
            int cur_red_streak = 0;

            for (const auto& w : weeks) {
                if (w.pnl > 0.0f) {
                    ++green_weeks;
                    cur_red_streak = 0;
                } else {
                    ++cur_red_streak;
                    if (cur_red_streak > max_red_streak) max_red_streak = cur_red_streak;
                }
            }

            r.green_weeks_pct =
                static_cast<float>(green_weeks) / static_cast<float>(weeks.size()) * 100.0f;
            r.max_red_weeks = max_red_streak;
        }
    }

    // -- Monthly consistency ----------------------------------------------
    {
        auto months = bucket_monthly_pnl(trades, bars, bar_start, bar_end, sessions, sd_to_month,
                                         sd_range_first, sd_range_last);
        if (!months.empty()) {
            int green_months = 0;
            int neg_months   = 0;
            for (const auto& m : months) {
                if (m.pnl > 0.0f) ++green_months;
                if (m.pnl < 0.0f) ++neg_months;
            }
            r.green_months_pct =
                static_cast<float>(green_months) / static_cast<float>(months.size()) * 100.0f;
            r.negative_months = neg_months;
        }
    }

    // -- Yearly consistency -----------------------------------------------
    {
        auto years       = bucket_yearly_pnl(trades, bars, bar_start, bar_end, sessions, sd_to_year,
                                             sd_range_first, sd_range_last);
        r.negative_years = 0;
        for (const auto& y : years) {
            if (y.pnl < 0.0f) ++r.negative_years;
        }
    }

    // -- Capital efficiency -----------------------------------------------
    const float capital       = cost.capital_required;
    const float avg_trade_pnl = r.total_pnl / static_cast<float>(r.total_trades);

    if (duration_years > 0.0 && capital > 0.0f) {
        const double final_equity = static_cast<double>(capital) + static_cast<double>(r.total_pnl);
        if (final_equity <= 0.0) {
            // Total wipeout: sentinel values.
            r.annual_roc     = -100.0f;
            r.time_to_double = -1.0f;
        } else {
            const double cagr =
                (std::pow(final_equity / static_cast<double>(capital), 1.0 / duration_years) -
                 1.0) *
                100.0;
            r.annual_roc = static_cast<float>(cagr);
            if (cagr > 0.0) {
                r.time_to_double =
                    static_cast<float>((std::log(2.0) / std::log(1.0 + cagr / 100.0)) * 365.25);
            } else {
                // No positive growth: never doubles.
                r.time_to_double = -1.0f;
            }
        }
    }

    if (capital > 0.0f) {
        r.reward_risk_trade = avg_trade_pnl / capital;
    }

    // Daily profit rate: avg_trade_pnl / capital / avg_hold_days * 100
    // avg_hold_days = avg_hold_bars * (timeframe_sec / 86400 sec per day)
    {
        const float bar_duration_days = static_cast<float>(timeframe_sec) / 86400.0f;
        const float avg_hold_days     = r.avg_hold_bars * bar_duration_days;
        if (avg_hold_days > 0.0f && capital > 0.0f) {
            r.daily_profit_rate = avg_trade_pnl / capital / avg_hold_days * 100.0f;
        }
    }

    // -- Trade Recovery Ratio (TRR) ---------------------------------------
    // TRR = (avg_win * win_rate) / avg_trade_dd
    // Sentinel: when avg_trade_dd == 0 (no trade ever went negative), TRR = 100.0
    if (r.avg_trade_dd > 0.0f) {
        r.trade_recovery_ratio = (r.avg_win * r.win_rate) / r.avg_trade_dd;
    } else if (r.total_trades > 0) {
        r.trade_recovery_ratio = 100.0f;
    }

    // -- Viability check --------------------------------------------------
    // avg per-trade PnL >= 2x spread (round_trip_cost)
    r.viable = (avg_trade_pnl >= 2.0f * cost.round_trip_cost());

    // Every metric is populated. Robustness scoring only uses neighbors
    // with this flag, so a CPU result must always set it.
    r.metrics_complete = true;
    return r;
}
