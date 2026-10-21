#pragma once
// ---------------------------------------------------------------------------
// DetailRunner -- Run specific strategy configurations and produce detailed
// output for deep-dive analysis of promising strategies.
//
// Input:  A list of strategy IDs (from the IndicatorGrid x StrategyGrid space)
//         or a text file with one strategy ID per line.
// Output per strategy:
//   1. Full BacktestResult metrics
//   2. Per-trade CSV: entry/exit bar, timestamp, price, direction, gross/net PnL,
//      min unrealized, hold bars
//   3. Quarterly PnL breakdown
//   4. Summary CSV: one row per strategy with all metrics
// ---------------------------------------------------------------------------

#include <string>
#include <vector>
#include "backtest_engine.h"
#include "cost_model.h"
#include "market_data.h"
#include "metrics.h"
#include "quarterly_pnl.h"

namespace detail {

// ---------------------------------------------------------------------------
// DetailResult -- full results for one strategy configuration
// ---------------------------------------------------------------------------
struct DetailResult {
    int                       strategy_id = -1;
    StrategyConfig            config;  // the strategy the id decodes to
    BacktestResult            metrics;
    std::vector<Trade>        trades;
    std::vector<QuarterlyPnL> quarterly;
};

// ---------------------------------------------------------------------------
// Run a list of specific strategy IDs on a bar range and produce detailed results.
//
// Parameters:
//   bars           - the full bar array
//   num_bars       - total bars
//   eval_start     - first evaluation bar index
//   eval_end       - one past last evaluation bar index
//   strategy_ids   - list of strategy IDs to evaluate
//   cost           - cost model
//   entry_lag      - 1 (standard) or 2 (lag-2 validation)
//   sessions       - optional per-bar SessionInfo for DST-correct quarterly bucketing.
//                    Pass &MarketData::session(0) when available.
//   md             - optional MarketData that `bars` belongs to. When its session
//                    model enables session signals, they are computed and the
//                    strategy grid spans them, exactly as in the screen. Without
//                    it the grid spans the technical indicators only.
//
// Returns one DetailResult per strategy ID; an id outside the grid gets an
// empty result.
// ---------------------------------------------------------------------------
std::vector<DetailResult> run_detail(const Bar* bars, int num_bars, int eval_start, int eval_end,
                                     const std::vector<int>& strategy_ids, const CostModel& cost,
                                     int entry_lag = 1, const SessionInfo* sessions = nullptr,
                                     const MarketData* md = nullptr);

// ---------------------------------------------------------------------------
// Load strategy IDs from a text file (one integer per line, # comments allowed)
// ---------------------------------------------------------------------------
std::vector<int> load_strategy_ids(const std::string& filepath);

// ---------------------------------------------------------------------------
// Output writers
// ---------------------------------------------------------------------------

// Write per-trade CSV for one strategy.
// Columns: trade_idx, entry_bar, exit_bar, entry_ts, exit_ts, entry_price,
//          exit_price, direction, gross_pnl, net_pnl, min_unrealized, max_unrealized, hold_bars
bool write_trade_csv(const std::string& filepath, const DetailResult& result, const Bar* bars);

// Write quarterly PnL CSV for one strategy.
// Columns: year, quarter, pnl, trades
bool write_quarterly_csv(const std::string& filepath, const DetailResult& result);

// Write summary CSV for all strategies (one row each).
// Columns: strategy_id, total_trades, trades_per_week, win_rate, avg_win, avg_loss,
//          total_pnl, max_drawdown, max_live_drawdown, sharpe, calmar, profit_factor,
//          green_weeks_pct, max_red_weeks, annual_roc, trade_recovery_ratio, viable
bool write_detail_summary_csv(const std::string&               filepath,
                              const std::vector<DetailResult>& results);

// Write all detailed outputs to a directory.
// Creates:
//   output_dir/detail_summary.csv
//   output_dir/trades/strategy_{id}.csv
//   output_dir/quarterly/strategy_{id}.csv
bool write_all_detail_outputs(const std::string&               output_dir,
                              const std::vector<DetailResult>& results, const Bar* bars);

}  // namespace detail
