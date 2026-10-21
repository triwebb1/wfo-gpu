#pragma once
// ---------------------------------------------------------------------------
// Results -- walk-forward output serialization
//
// Writes three types of output files:
//   1. Per-WFO-config result file (CSV with window-by-window data)
//   2. Summary CSV (one row per WFO config with aggregate metrics)
//   3. Equity curve data (timestamp, cumulative PnL)
// ---------------------------------------------------------------------------

#include <string>
#include <vector>

// Forward declarations to avoid circular includes
struct WFORunResult;
struct WFOWindowResult;
struct WFOConfig;
struct BacktestResult;

namespace results {

// ---------------------------------------------------------------------------
// Write a per-WFO-config detail CSV with window-by-window data.
//
// Columns: window_index, is_start_bar, is_end_bar, oos_start_bar, oos_end_bar,
//          strategy_id, archetype, skipped,
//          is_total_pnl, is_sharpe, is_annual_roc, is_profit_factor,
//          oos_lag1_total_pnl, oos_lag1_sharpe, oos_lag1_annual_roc,
//          oos_lag2_total_pnl, oos_lag2_sharpe, oos_lag2_annual_roc,
//          pnl_decomp_bar_body_pct, pnl_decomp_gap_pct
//
// Returns true on success.
// ---------------------------------------------------------------------------
bool write_window_csv(const std::string& filepath, const WFORunResult& run);

// ---------------------------------------------------------------------------
// Write summary CSV: one row per WFO config with aggregate metrics.
//
// Columns: schedule_freq_days, schedule_lookback_days, selection_strategy,
//          total_windows, windows_skipped, viable, parameter_stability,
//          oos_total_pnl, oos_total_trades, oos_annual_roc, oos_sharpe,
//          oos_profit_factor, oos_max_drawdown
//
// Truncates and rewrites the file with header row and all rows.
// Returns true on success.
// ---------------------------------------------------------------------------
bool write_summary_csv(const std::string& filepath, const std::vector<WFORunResult>& runs);

// ---------------------------------------------------------------------------
// Write equity curve CSV for a single WFO run.
//
// Columns: timestamp_ns, cumulative_pnl
//
// Returns true on success.
// ---------------------------------------------------------------------------
bool write_equity_csv(const std::string& filepath, const WFORunResult& run);

// ---------------------------------------------------------------------------
// Write all walk-forward outputs to a directory.
//
// Creates:
//   output_dir/summary.csv          -- summary of all WFO runs
//   output_dir/details/<name>.csv   -- per-config window detail
//   output_dir/equity/<name>.csv    -- per-config equity curve
//   output_dir/meta_analysis.csv    -- best selection strategy analysis
//
// Where <name> = "freq{F}d_lb{L}d_{selection_strategy}"
//
// Returns true if all writes succeeded.
// ---------------------------------------------------------------------------
bool write_walk_forward_results(const std::string&               output_dir,
                                const std::vector<WFORunResult>& runs,
                                const std::string&               best_selection_strategy    = "",
                                float                            best_selection_avg_oos_roc = 0.0f);

}  // namespace results
