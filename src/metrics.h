#pragma once
#include <cstdint>
#include <vector>

// Forward declarations
struct Trade;
struct CostModel;
struct Bar;
struct SessionInfo;

// ---------------------------------------------------------------------------
// BacktestResult -- all required metrics (see docs/METHODOLOGY.md)
//
// Populated by compute_metrics() after a backtest produces a vector of Trades.
// ---------------------------------------------------------------------------
struct BacktestResult {
    // Time period
    int64_t start_ts = 0;  // first bar timestamp (ns UTC)
    int64_t end_ts   = 0;  // last bar timestamp (ns UTC)

    // Trade statistics
    int   total_trades    = 0;
    float trades_per_week = 0.0f;
    float win_rate        = 0.0f;  // 0..1
    float avg_win         = 0.0f;  // dollars (positive)
    float avg_loss        = 0.0f;  // dollars (negative)
    float avg_hold_bars   = 0.0f;  // average trade duration in bars

    // PnL
    float total_pnl    = 0.0f;  // net, after costs
    float gross_wins   = 0.0f;  // sum of positive net_pnl trades
    float gross_losses = 0.0f;  // sum of negative net_pnl trades (negative value)

    // Drawdown
    float max_drawdown      = 0.0f;  // realized equity curve peak-to-trough (positive)
    float max_live_drawdown = 0.0f;  // worst per-trade unrealized M2M (positive)
    float avg_trade_dd      = 0.0f;  // average per-trade worst unrealized (positive magnitude)
    float avg_trade_peak    = 0.0f;  // average per-trade peak unrealized (positive)

    // Risk-adjusted
    float sharpe_ratio  = 0.0f;  // annualized: mean(daily_pnl)/std(daily_pnl)*sqrt(252)
    float calmar_ratio  = 0.0f;  // annualized_pnl / max_drawdown
    float profit_factor = 0.0f;  // gross_wins / |gross_losses|
    float payoff_ratio  = 0.0f;  // avg_win / |avg_loss|

    // Consistency
    float green_weeks_pct  = 0.0f;  // % of weeks with positive PnL
    float green_months_pct = 0.0f;  // % of months with positive PnL
    int   max_red_weeks    = 0;     // longest consecutive losing week streak
    int   negative_years   = 0;     // count of calendar years with negative PnL
    int   negative_months  = 0;     // count of calendar months with negative PnL

    // Capital efficiency
    float annual_roc = 0.0f;  // CAGR %: ((capital+total_pnl)/capital)^(1/years) - 1, times 100
    float reward_risk_trade = 0.0f;  // avg_pnl_per_trade / capital_required
    float daily_profit_rate = 0.0f;  // avg_trade_pnl / capital / avg_hold_days * 100
    float time_to_double =
        0.0f;  // days to double at CAGR: log(2)/log(1+cagr) * 365.25; -1 if never

    // Trade Recovery Ratio
    float trade_recovery_ratio = 0.0f;  // (avg_win * win_rate) / avg_trade_dd

    // Strategy identification
    int  strategy_id = -1;     // grid index for this strategy config
    bool viable      = false;  // avg per-trade PnL >= 2x spread

    // True once compute_metrics() has fully populated the CPU-only
    // fields (green_*_pct, max_red_weeks, sharpe_ratio, annual_roc, etc.).
    // GPU results have this false until the screen re-runs them on the CPU
    // engine (Orchestrator's refinement pass). Robustness scoring skips
    // neighbors with metrics_complete == false to avoid contaminating
    // neighborhood stability scores with zero/default placeholders.
    bool metrics_complete = false;
};

// ---------------------------------------------------------------------------
// Time-bucketed PnL for consistency metrics
// ---------------------------------------------------------------------------
struct WeeklyPnL {
    int64_t week_start_ts = 0;  // timestamp of first bar in this week (ns UTC)
    float   pnl           = 0.0f;
};

struct MonthlyPnL {
    int   year  = 0;
    int   month = 0;  // 1..12
    float pnl   = 0.0f;
};

struct YearlyPnL {
    int   year = 0;
    float pnl  = 0.0f;
};

// ---------------------------------------------------------------------------
// Compute all metrics from trades + bar data
//
// Parameters:
//   trades         - completed trades from the backtest engine
//   bars           - the bar array (for timestamps)
//   bar_start      - index of first evaluation bar
//   bar_end        - index one past the last evaluation bar
//   cost           - cost model (for viability check)
//   sessions       - optional per-bar SessionInfo array (same indexing as bars[]).
//                    When non-null, daily PnL bucketing (for Sharpe ratio) uses
//                    sessions[bar_index].session_day instead of the fixed UTC
//                    timestamp shift, providing DST-correct session day boundaries.
//                    Obtained from &MarketData::session(0).
//
// Returns a fully populated BacktestResult.
// ---------------------------------------------------------------------------
BacktestResult compute_metrics(const std::vector<Trade>& trades, const Bar* bars, int bar_start,
                               int bar_end, const CostModel& cost,
                               const SessionInfo* sessions = nullptr, int timeframe_sec = 30,
                               const int* sd_to_week = nullptr, const int* sd_to_month = nullptr,
                               const int* sd_to_year = nullptr, int sd_range_first = 0,
                               int sd_range_last = 0);

// ---------------------------------------------------------------------------
// Helper: bucket trade PnL into weekly/monthly/yearly periods
// Uses bar timestamps to determine calendar boundaries.
// When sessions is non-null, timestamps are shifted to ET before extracting
// calendar dates, so trades near midnight UTC are bucketed to the correct
// CME session week/month/year.
//
// Performance optimization: when sd_to_week / sd_to_month / sd_to_year
// arrays are provided (from MarketData::session_day_to_week() etc.), the
// O(bars) scan for zero-trade periods is replaced with an O(session_days)
// scan using precomputed lookups. sd_range_first/sd_range_last define the
// session day range to scan. Pass nullptr + 0/0 to use the existing logic.
// ---------------------------------------------------------------------------
std::vector<WeeklyPnL> bucket_weekly_pnl(const std::vector<Trade>& trades, const Bar* bars,
                                         int bar_start, int bar_end,
                                         const SessionInfo* sessions = nullptr,
                                         const int* sd_to_week = nullptr, int sd_range_first = 0,
                                         int sd_range_last = 0);

std::vector<MonthlyPnL> bucket_monthly_pnl(const std::vector<Trade>& trades, const Bar* bars,
                                           int bar_start, int bar_end,
                                           const SessionInfo* sessions = nullptr,
                                           const int* sd_to_month = nullptr, int sd_range_first = 0,
                                           int sd_range_last = 0);

std::vector<YearlyPnL> bucket_yearly_pnl(const std::vector<Trade>& trades, const Bar* bars,
                                         int bar_start, int bar_end,
                                         const SessionInfo* sessions = nullptr,
                                         const int* sd_to_year = nullptr, int sd_range_first = 0,
                                         int sd_range_last = 0);
