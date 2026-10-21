#include "pnl_decomposition.h"
#include "backtest_engine.h"
#include "cost_model.h"

#include <cmath>

// ---------------------------------------------------------------------------
// decompose_pnl
//
// For each trade, the fill bar is trade.entry_bar.  We decompose the gross
// PnL of each trade into three components:
//
//   gap_pnl  = direction * (open[fill_bar] - close[fill_bar - 1])
//              This is the close-to-open gap that happens between the signal
//              bar's close and the fill bar's open.  If this dominates PnL,
//              the strategy is exploiting a gap artifact.
//
//   bar_body_pnl = direction * (close[fill_bar] - open[fill_bar])
//              This is the bar body of the fill bar itself.  If this dominates,
//              the strategy has a real short-horizon directional edge.
//
//   remaining_pnl = gross_pnl - gap_pnl - bar_body_pnl
//              PnL from subsequent bars (position held beyond the fill bar).
//
// All components are converted to dollars via cost.price_to_pnl().
// ---------------------------------------------------------------------------

PnLDecomposition decompose_pnl(const std::vector<Trade>& trades, const Bar* bars,
                               const CostModel& cost) {
    PnLDecomposition result;

    if (trades.empty()) return result;

    result.per_trade.reserve(trades.size());

    float total_bar_body = 0.0f;
    float total_gap      = 0.0f;
    float total_gross    = 0.0f;

    for (int ti = 0; ti < static_cast<int>(trades.size()); ++ti) {
        const Trade&               t = trades[static_cast<size_t>(ti)];
        PnLDecomposition::PerTrade pt;
        pt.trade_index = ti;

        const int   fill_bar = t.entry_bar;
        const float dir      = static_cast<float>(t.direction);

        // Gap component: open[fill_bar] - close[fill_bar - 1]
        // fill_bar - 1 must be valid.  If fill_bar == 0, gap is undefined (skip).
        if (fill_bar > 0) {
            const float gap_price_diff = bars[fill_bar].open - bars[fill_bar - 1].close;
            pt.gap_pnl                 = cost.price_to_pnl(dir * gap_price_diff);
        } else {
            pt.gap_pnl = 0.0f;
        }

        // Bar body component: close[fill_bar] - open[fill_bar]
        // When a LEVEL_EXIT fires on the same bar as entry, the
        // trade's gross_pnl is (dir * (level - open)), not derived from the
        // bar close. Using close-based body would misattribute the intrabar
        // path and leave a non-zero "remaining" that did not exist. Instead
        // assign body = gross - gap (the intrabar realized move) and zero
        // the remaining component. SESSION_CLOSE and FORCED_CLOSE fill at
        // the NEXT bar's open / last bar's close respectively; they only hit
        // this same-bar branch when entry and that final bar coincide, for
        // which the same gross-minus-gap attribution is the most faithful
        // decomposition.
        if (t.entry_bar == t.exit_bar &&
            (t.fill_type == FillType::LEVEL_EXIT || t.fill_type == FillType::SESSION_CLOSE ||
             t.fill_type == FillType::FORCED_CLOSE)) {
            pt.bar_body_pnl  = t.gross_pnl - pt.gap_pnl;
            pt.remaining_pnl = 0.0f;
        } else {
            const float body_price_diff = bars[fill_bar].close - bars[fill_bar].open;
            pt.bar_body_pnl             = cost.price_to_pnl(dir * body_price_diff);
            // Remaining: everything else (PnL from bars after the fill bar)
            pt.remaining_pnl = t.gross_pnl - pt.gap_pnl - pt.bar_body_pnl;
        }

        total_bar_body += pt.bar_body_pnl;
        total_gap += pt.gap_pnl;
        total_gross += t.gross_pnl;

        result.per_trade.push_back(pt);
    }

    result.total_bar_body_pnl = total_bar_body;
    result.total_gap_pnl      = total_gap;
    result.total_gross_pnl    = total_gross;

    // Compute percentages and clamp to [-100, 200] to handle cases where
    // components exceed total_gross in magnitude (e.g., offsetting components).
    const float abs_total = std::abs(total_gross);
    if (abs_total > 0.0f) {
        result.bar_body_pct = (total_bar_body / abs_total) * 100.0f;
        result.gap_pct      = (total_gap / abs_total) * 100.0f;

        result.bar_body_pct = std::max(-100.0f, std::min(200.0f, result.bar_body_pct));
        result.gap_pct      = std::max(-100.0f, std::min(200.0f, result.gap_pct));
    }

    // Classify using clamped values
    result.is_real_edge    = (result.bar_body_pct > 80.0f);
    result.is_gap_artifact = (result.gap_pct > 50.0f);

    return result;
}
