// Example: plug a custom signal into the batch backtester.
//
// 1. Implement the Indicator interface (here: a z-score of price against its
//    rolling mean, a classic mean-reversion signal).
// 2. Map its output into [-1, +1]: the engines treat any such float series as
//    a signal.
// 3. Sweep entry thresholds, exits and trading hours over that one signal
//    with CLBacktest, which runs on an OpenCL device when one is present and
//    on the multithreaded C++ path otherwise, with the same results.
//
// Run: wfo-gpu-example-custom-signal [bar_file]
// Without an argument it generates a synthetic series first.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "bar_io.h"
#include "common.h"
#include "backtest_engine.h"
#include "compute/cl_backtest.h"
#include "compute/cl_context.h"
#include "indicators/indicator.h"
#include "market_data.h"
#include "synthetic.h"

// A custom indicator: (close - SMA(n)) / stdev(n), one output series.
class ZScore : public Indicator {
public:
    explicit ZScore(int period) : m_period(period) {}

    void compute(const Bar* bars, int start, int end, float* out) const override {
        double sum = 0.0, sum2 = 0.0;
        for (int i = start; i < end; ++i) {
            const double x = bars[i].close;
            sum += x;
            sum2 += x * x;
            if (i - start >= m_period) {
                const double old = bars[i - m_period].close;
                sum -= old;
                sum2 -= old * old;
            }
            const int k = i - start;
            if (k + 1 < m_period) {
                out[k] = INDICATOR_NAN;  // warming up
                continue;
            }
            const double mean = sum / m_period;
            const double var  = std::max(0.0, sum2 / m_period - mean * mean);
            out[k]            = var > 0.0 ? static_cast<float>((x - mean) / std::sqrt(var)) : 0.0f;
        }
    }
    int         num_series() const override { return 1; }
    int         warmup_bars() const override { return m_period; }
    const char* name() const override { return "zscore"; }

private:
    int m_period;
};

int main(int argc, char** argv) {
    // ---- Data ----------------------------------------------------------------
    std::string path = argc > 1 ? argv[1] : "example_bars.bin";
    if (argc <= 1) {
        SyntheticConfig cfg;
        cfg.num_days     = 120;
        cfg.seed         = 3;
        const auto  bars = generate_synthetic_bars(cfg);
        std::string err;
        if (!write_bar_file(path, cfg.symbol, cfg.timeframe_sec, bars, &err)) {
            std::cerr << "cannot write " << path << ": " << err << "\n";
            return 1;
        }
    }
    MarketData md;
    if (!md.load(path)) return 1;
    const int n = md.bar_count();

    // ---- Signal --------------------------------------------------------------
    // Mean reversion: a high z-score is bearish. Divide by 3 and clamp so the
    // signal spans [-1, +1].
    const ZScore       z(240);  // two hours of 30-second bars
    std::vector<float> raw(n), signal(n);
    z.compute(md.bars(), 0, n, raw.data());
    for (int i = 0; i < n; ++i) signal[i] = clamp_signal(-raw[i] / 3.0f);
    const float* series[] = {signal.data()};

    // ---- Strategy variants over the one signal --------------------------------
    std::vector<StrategyConfig> variants;
    for (float threshold : {0.3f, 0.5f, 0.7f}) {
        for (int hours = StrategyConfig::ETH; hours <= StrategyConfig::AFTERNOON; ++hours) {
            for (int exit = 0; exit < 4; ++exit) {
                StrategyConfig sc;
                sc.num_indicators       = 1;
                sc.indicator_indices[0] = 0;
                sc.entry_threshold      = threshold;
                sc.trading_hours        = static_cast<StrategyConfig::TradingHours>(hours);
                switch (exit) {
                    case 0:
                        sc.exit_mode = StrategyConfig::SIGNAL_REVERSAL;
                        break;
                    case 1:
                        sc.exit_mode         = StrategyConfig::BAR_TIMEOUT;
                        sc.exit_timeout_bars = 60;
                        break;
                    case 2:
                        sc.exit_mode         = StrategyConfig::TRAILING_STOP;
                        sc.trailing_atr_mult = 3.0f;
                        break;
                    case 3:
                        sc.exit_mode   = StrategyConfig::TP_SL_ATR;
                        sc.tp_atr_mult = 3.0f;
                        sc.sl_atr_mult = 2.0f;
                        break;
                }
                variants.push_back(sc);
            }
        }
    }

    // ---- Batch backtest ---------------------------------------------------------
    CLContext ctx;
    ctx.initialize(DeviceType::AUTO);
    std::cout << "device: " << (ctx.is_available() ? ctx.device_name() : "native C++ path") << "\n";

    CostModel cost                         = CostModel::es();
    cost.capital_required                  = 100000.0f;
    const int                   eval_start = z.warmup_bars();
    CLBacktest                  bt;
    std::vector<BacktestResult> results;
    bt.evaluate(ctx, md.bars(), 0, series, 1, n, eval_start, n, variants, cost, 1, results,
                EarlyTermination{}, md.sessions());

    // ---- Report -----------------------------------------------------------------
    // The batch path returns trade counts, PnL, drawdown and profit factor for
    // every variant. Rank on those, then re-run the leaders on the CPU engine
    // for the full metric set (calendar-time return, Sharpe, consistency).
    std::vector<int> order(results.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return results[a].total_pnl > results[b].total_pnl; });
    order.resize(std::min<size_t>(order.size(), 10));

    static const char* HOURS[] = {"all", "core", "morning", "afternoon"};
    static const char* EXITS[] = {"reversal", "timeout", "trail", "", "", "", "tp/sl"};
    std::printf("\nTop %zu of %zu variants by net PnL\n", order.size(), variants.size());
    std::printf("%-6s %-10s %-9s %7s %9s %8s %7s %6s %9s %7s\n", "thresh", "hours", "exit",
                "trades", "net PnL$", "ROC%", "Sharpe", "PF", "GreenWk%", "viable");
    const float* offset[] = {signal.data() + eval_start};
    for (int idx : order) {
        const StrategyConfig& sc = variants[idx];
        BacktestEngine        engine;
        const BacktestResult  r =
            engine.run(md.bars(), md.sessions(), eval_start, n, offset, 1, nullptr, sc, cost, 1);
        std::printf("%-6.1f %-10s %-9s %7d %9.0f %8.2f %7.2f %6.2f %9.1f %7s\n", sc.entry_threshold,
                    HOURS[sc.trading_hours], EXITS[sc.exit_mode], r.total_trades, r.total_pnl,
                    r.annual_roc, r.sharpe_ratio, std::min(r.profit_factor, 99.99f),
                    r.green_weeks_pct, r.viable ? "yes" : "no");
    }
    std::cout << "\nOn a synthetic random walk no variant should hold up after costs. Feed\n"
                 "the same signal to the full pipeline with `wfo-gpu run` to test it\n"
                 "walk-forward, or see examples/walk_forward_loop.\n";
    return 0;
}
