#pragma once

#include <cmath>

// ---------------------------------------------------------------------------
// CostModel -- instrument-specific trading cost parameters
//
// Spread-inclusive cost model with per-instrument
// round-trip costs.  Every trade deducts round_trip_cost() from gross PnL.
//
// UNITS:
//   tick_size   = minimum price increment (price units, NOT dollars)
//   tick_value  = dollar value of one tick_size move per contract
//   point_value = dollar value of a 1.00 price move per contract
//
// Reference table:
//
//   Instrument   tick_size  tick_value  point_value  Notes
//   -----------  ---------  ----------  -----------  -------------------------
//   ES  (mini)   0.25 pt    $12.50      $50.00/pt    S&P 500 index points
//   NQ  (mini)   0.25 pt     $5.00      $20.00/pt    Nasdaq-100 index points
//   MES (micro)  0.25 pt     $1.25       $5.00/pt    1/10th of ES
//   MNQ (micro)  0.25 pt     $0.50       $2.00/pt    1/10th of NQ
//   GC  (gold)   0.10 $/oz  $10.00     $100.00/pt    Gold $/troy oz, 100 oz/contract
//
// A "tick" is one tick_size of price movement.  ES slipping 1 tick means
// the fill was 0.25 pts worse than expected, costing $12.50 per contract.
// It does NOT mean $1 of slippage.
// ---------------------------------------------------------------------------

struct CostModel {
    float tick_size;         // minimum price increment (price units, NOT dollars)
    float tick_value;        // dollar value per tick per contract
    float point_value;       // dollar value per 1.00 price move = tick_value / tick_size
    int   round_trip_ticks;  // bid-ask spread in ticks (e.g. 2 = 1 tick each side)
    float slippage_ticks;    // additional slippage per side, in ticks (NOT dollars)
    float capital_required;  // capital to survive worst drawdowns (NOT exchange margin)

    // Total round-trip cost in dollars for one contract.
    // Example ES with slippage: (2 spread + 2*1 slip) ticks * $12.50/tick = $50.00
    float round_trip_cost() const {
        return (static_cast<float>(round_trip_ticks) + 2.0f * slippage_ticks) * tick_value;
    }

    // Convert a price movement to dollar PnL for 1 contract.
    // price_diff is in price units (same as bar OHLC values).
    // Example: ES moves 10.00 pts -> 10.0 * $50/pt = $500.
    float price_to_pnl(float price_diff) const { return price_diff * point_value; }

    // Structural invariants. Callers (e.g. BacktestEngine::run_impl)
    // check this once per engine-construction boundary and abort / refuse
    // the run if it returns false. Kept pure so it is cheap to call from
    // hot paths for defense-in-depth assertions.
    bool is_valid() const {
        if (!(tick_size > 0.0f)) return false;
        if (!(tick_value > 0.0f)) return false;
        if (!(point_value > 0.0f)) return false;
        if (round_trip_ticks < 0) return false;
        if (!(slippage_ticks >= 0.0f)) return false;
        if (!std::isfinite(slippage_ticks)) return false;
        if (!(capital_required > 0.0f)) return false;
        return true;
    }

    // Pre-built models for common instruments
    static CostModel es() {
        return CostModel{
            .tick_size        = 0.25f,   // 0.25 index points per tick
            .tick_value       = 12.50f,  // $12.50 per 0.25-pt tick
            .point_value      = 50.0f,   // $50.00 per 1.00-pt move
            .round_trip_ticks = 2,       // 1 tick spread each side
            .slippage_ticks   = 2.0f,    // 2 ticks (0.50 pts) slip per side = $25.00/side
            .capital_required = 15000.0f
            // Total RT cost = (2 + 2*2) * $12.50 = 6 * $12.50 = $75.00
        };
    }

    static CostModel nq() {
        return CostModel{
            .tick_size        = 0.25f,  // 0.25 index points per tick
            .tick_value       = 5.00f,  // $5.00 per 0.25-pt tick
            .point_value      = 20.0f,  // $20.00 per 1.00-pt move
            .round_trip_ticks = 2,      // 1 tick spread each side
            .slippage_ticks   = 2.0f,   // 2 ticks (0.50 pts) slip per side = $10.00/side
            .capital_required = 20000.0f
            // Total RT cost = (2 + 2*2) * $5.00 = 6 * $5.00 = $30.00
        };
    }
};
