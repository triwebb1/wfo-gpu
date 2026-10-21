#pragma once
#include <vector>
#include "market_data.h"

// Forward declaration
struct Trade;

// ---------------------------------------------------------------------------
// PnLDecomposition -- lag-2 validation analysis
//
// When lag-2 results degrade vs lag-1, decompose PnL
// into bar-body vs gap components to determine if the edge is real or an
// artifact of sub-second execution assumptions.
//
//   bar_body_pnl = direction * (close[fill_bar] - open[fill_bar])
//   gap_pnl      = direction * (open[fill_bar] - close[fill_bar - 1])
//
// Interpretation:
//   bar_body_pct > 80%  -> real short-horizon edge (signal predicts bar direction)
//   gap_pct      > 50%  -> sub-second artifact (PnL comes from the overnight/inter-bar gap)
// ---------------------------------------------------------------------------
struct PnLDecomposition {
    float total_bar_body_pnl = 0.0f;  // sum of bar-body PnL across all trades (dollars)
    float total_gap_pnl      = 0.0f;  // sum of gap PnL across all trades (dollars)
    float total_gross_pnl    = 0.0f;  // total gross PnL for reference

    float bar_body_pct = 0.0f;  // bar_body_pnl / total_gross_pnl * 100
    float gap_pct      = 0.0f;  // gap_pnl / total_gross_pnl * 100

    bool is_real_edge    = false;  // bar_body_pct > 80%
    bool is_gap_artifact = false;  // gap_pct > 50%

    // Per-trade decomposition (optional, for detailed analysis)
    struct PerTrade {
        int   trade_index   = 0;
        float bar_body_pnl  = 0.0f;
        float gap_pnl       = 0.0f;
        float remaining_pnl = 0.0f;  // PnL from bars after the fill bar
    };
    std::vector<PerTrade> per_trade;
};

// ---------------------------------------------------------------------------
// Decompose PnL into bar-body and gap components.
//
// Parameters:
//   trades - completed trade list (from BacktestEngine::last_trades())
//   bars   - the bar array
//   cost   - cost model (for price_to_pnl conversion)
//
// Each trade's fill bar is trade.entry_bar.  The decomposition examines:
//   gap     = open[entry_bar] - close[entry_bar - 1]
//   body    = close[entry_bar] - open[entry_bar]
//
// Contract: trade.entry_bar is LOCAL to the `bars` pointer passed
// here (i.e. `bars[trade.entry_bar]` is the fill bar). Callers that
// generate trades on a sliced bar range MUST pass the matching `bars`
// slice (same base pointer as used during backtest), NOT the full-file
// bar array, or the gap/body decomposition will reference the wrong bars.
//
// Both are multiplied by trade.direction and converted to dollars.
// ---------------------------------------------------------------------------
struct CostModel;

PnLDecomposition decompose_pnl(const std::vector<Trade>& trades, const Bar* bars,
                               const CostModel& cost);
