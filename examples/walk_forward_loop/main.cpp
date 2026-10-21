// Example: a hand-written walk-forward loop on library primitives.
//
// The built-in pipeline (`wfo-gpu run`) searches a large grid with screening,
// robustness and Pareto selection. This example shows the core idea in a few
// dozen lines so the pieces are easy to reuse:
//
//   for each window:
//     in-sample   evaluate every candidate with CLBacktest (GPU or CPU)
//     select      the best viable candidate by return on capital
//     out-of-sample  trade it with BacktestEngine at lag 1, then lag 2
//   stitch the out-of-sample trades and report
//
// Candidates combine two signals: a built-in EMA crossover and the sign of a
// short-term return (momentum), with a few aggregation modes, thresholds and
// exits.
//
// Run: wfo-gpu-example-walk-forward [bar_file]
// Without an argument it generates a synthetic series with planted trends.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "backtest_engine.h"
#include "bar_io.h"
#include "common.h"
#include "compute/cl_backtest.h"
#include "compute/cl_context.h"
#include "indicators/ema_sma_cross.h"
#include "market_data.h"
#include "metrics.h"
#include "synthetic.h"

int main(int argc, char** argv) {
    // ---- Data ----------------------------------------------------------------
    const std::string path = argc > 1 ? argv[1] : "example_trending.bin";
    if (argc <= 1) {
        SyntheticConfig cfg;
        cfg.num_days            = 120;
        cfg.seed                = 5;
        cfg.trend_strength      = 0.04;  // a known, planted edge for trend followers
        cfg.trend_duration_bars = 4000;
        std::string err;
        if (!write_bar_file(path, cfg.symbol, cfg.timeframe_sec, generate_synthetic_bars(cfg),
                            &err)) {
            std::cerr << "cannot write " << path << ": " << err << "\n";
            return 1;
        }
    }
    MarketData md;
    if (!md.load(path)) return 1;
    const int  n    = md.bar_count();
    const Bar* bars = md.bars();

    // ---- Signals in [-1, +1] --------------------------------------------------
    // 0: EMA(20) vs EMA(120) crossover, scaled by the gap in units of price.
    // 1: sign of the 60-bar return.
    std::vector<float> ma(static_cast<size_t>(n) * 2);
    EMASMACross(MAType::EMA, 20, MAType::EMA, 120).compute(bars, 0, n, ma.data());
    std::vector<float> cross(n), mom(n);
    for (int i = 0; i < n; ++i) {
        const float fast = ma[i], slow = ma[n + i];
        cross[i] = std::isnan(fast) || std::isnan(slow)
                       ? INDICATOR_NAN
                       : clamp_signal((fast - slow) / (0.001f * slow));
        mom[i]   = i < 60 ? INDICATOR_NAN
                          : (bars[i].close > bars[i - 60].close   ? 1.0f
                             : bars[i].close < bars[i - 60].close ? -1.0f
                                                                  : 0.0f);
    }
    const float* signals[] = {cross.data(), mom.data()};

    // ---- Candidates ------------------------------------------------------------
    std::vector<StrategyConfig> candidates;
    for (int combo = 0; combo < 3; ++combo) {
        for (float threshold : {0.25f, 0.5f, 0.75f}) {
            for (int exit = 0; exit < 3; ++exit) {
                StrategyConfig sc;
                sc.entry_threshold = threshold;
                if (combo < 2) {  // one signal on its own
                    sc.num_indicators       = 1;
                    sc.indicator_indices[0] = combo;
                } else {  // both must agree
                    sc.num_indicators       = 2;
                    sc.indicator_indices[0] = 0;
                    sc.indicator_indices[1] = 1;
                    sc.aggregation          = StrategyConfig::UNANIMOUS;
                }
                if (exit == 0) sc.exit_mode = StrategyConfig::SIGNAL_REVERSAL;
                if (exit == 1) {
                    sc.exit_mode         = StrategyConfig::TRAILING_STOP;
                    sc.trailing_atr_mult = 4.0f;
                }
                if (exit == 2) {
                    sc.exit_mode         = StrategyConfig::BAR_TIMEOUT;
                    sc.exit_timeout_bars = 240;
                }
                candidates.push_back(sc);
            }
        }
    }

    // ---- Walk forward -----------------------------------------------------------
    CLContext ctx;
    ctx.initialize(DeviceType::AUTO);
    CLBacktest bt;
    CostModel  cost       = CostModel::es();
    cost.capital_required = 100000.0f;

    const int          warmup   = 200;
    const int          lookback = 20 * 2760;  // ~20 sessions in-sample
    const int          step     = 5 * 2760;   // ~5 sessions out-of-sample
    std::vector<Trade> oos_lag1, oos_lag2;

    std::printf("device: %s\n\n",
                ctx.is_available() ? ctx.device_name().c_str() : "native C++ path");
    std::printf("%-6s %-12s %-10s %8s %8s %10s %10s\n", "window", "IS bars", "winner", "IS PnL$",
                "IS PF", "OOS PnL$", "lag2 PnL$");
    int window = 0;
    for (int t = warmup + lookback; t + step <= n; t += step, ++window) {
        // In-sample: evaluate every candidate on [t - lookback, t).
        std::vector<BacktestResult> is;
        bt.evaluate(ctx, bars, 0, signals, 2, n, t - lookback, t, candidates, cost, 1, is,
                    EarlyTermination{}, md.sessions());

        // Select: highest net PnL among viable candidates with enough trades.
        int best = -1;
        for (int k = 0; k < static_cast<int>(is.size()); ++k) {
            if (!is[k].viable || is[k].total_trades < 20) continue;
            if (best < 0 || is[k].total_pnl > is[best].total_pnl) best = k;
        }
        if (best < 0) {
            std::printf("%-6d %-12d %-10s %8s %8s %10s %10s\n", window, lookback, "none", "-", "-",
                        "flat", "flat");
            continue;
        }

        // Out-of-sample: trade the winner on [t, t + step), then again one bar late.
        const float*         oos_signals[] = {cross.data() + t, mom.data() + t};
        BacktestEngine       e1, e2;
        const BacktestResult r1 = e1.run(bars, md.sessions(), t, t + step, oos_signals, 2, nullptr,
                                         candidates[best], cost, 1);
        const BacktestResult r2 = e2.run(bars, md.sessions(), t, t + step, oos_signals, 2, nullptr,
                                         candidates[best], cost, 2);
        oos_lag1.insert(oos_lag1.end(), e1.last_trades().begin(), e1.last_trades().end());
        oos_lag2.insert(oos_lag2.end(), e2.last_trades().begin(), e2.last_trades().end());

        // CLBacktest on a GPU returns raw accumulators; re-run the winner's
        // in-sample window on the CPU for its full metrics.
        const float*   is_signals[] = {cross.data() + t - lookback, mom.data() + t - lookback};
        BacktestEngine eis;
        const BacktestResult is_full = eis.run(bars, md.sessions(), t - lookback, t, is_signals, 2,
                                               nullptr, candidates[best], cost, 1);
        std::printf("%-6d %-12d #%-9d %8.0f %8.2f %10.0f %10.0f\n", window, lookback, best,
                    is_full.total_pnl, std::min(is_full.profit_factor, 99.0f), r1.total_pnl,
                    r2.total_pnl);
    }

    // ---- Stitched out-of-sample result -------------------------------------------
    const int            first_oos = warmup + lookback;
    const BacktestResult agg1 = compute_metrics(oos_lag1, bars, first_oos, n, cost, md.sessions());
    const BacktestResult agg2 = compute_metrics(oos_lag2, bars, first_oos, n, cost, md.sessions());
    std::printf("\nStitched out-of-sample (%d windows)\n", window);
    std::printf("  %-6s %7s %10s %8s %7s %6s %10s %9s\n", "lag", "trades", "net PnL$", "ROC%",
                "Sharpe", "PF", "MaxDD$", "GreenWk%");
    auto print_row = [](int lag, const BacktestResult& r) {
        std::printf("  %-6d %7d %10.0f %8.2f %7.2f %6.2f %10.0f %9.1f\n", lag, r.total_trades,
                    r.total_pnl, r.annual_roc, r.sharpe_ratio, std::min(r.profit_factor, 99.0f),
                    r.max_drawdown, r.green_weeks_pct);
    };
    print_row(1, agg1);
    print_row(2, agg2);
    std::cout << "\nThe series has planted trend regimes, so trend-following candidates should\n"
                 "hold up out of sample. Rerun on a plain random walk (wfo-gpu generate with\n"
                 "--trend 0, then pass the file) and the edge should disappear.\n";
    return 0;
}
