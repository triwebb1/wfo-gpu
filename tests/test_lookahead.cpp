// No-lookahead regression tests.
//
// A value computed for bar t may depend only on bars <= t. Each test builds
// two datasets that are identical up to a cut bar and different afterwards,
// recomputes, and requires every output at or before the cut to match bit
// for bit. Any leak of future data shows up as a mismatch.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "backtest_engine.h"
#include "bar_io.h"
#include "civil_time.h"
#include "compute/cl_context.h"
#include "compute/cl_indicators.h"
#include "grids/indicator_grid.h"
#include "grids/session_signal_grid.h"
#include "grids/strategy_grid.h"
#include "grids/window_mode_grid.h"
#include "indicators/signal_normalize.h"
#include "indicators/ema_sma_cross.h"
#include "market_data.h"
#include "synthetic.h"

namespace {

// Two bar files that agree on bars [0, cut] and diverge after it.
struct PerturbedPair {
    MarketData base;
    MarketData perturbed;
    int        cut = 0;
};

std::vector<Bar> perturb_after(std::vector<Bar> bars, int cut) {
    for (size_t i = static_cast<size_t>(cut) + 1; i < bars.size(); ++i) {
        // Shift and stretch later prices so every derived value changes.
        const float k = 1.03f + 0.0001f * static_cast<float>(i % 17);
        bars[i].open *= k;
        bars[i].high *= k * 1.002f;
        bars[i].low *= k * 0.998f;
        bars[i].close *= k;
        bars[i].volume = bars[i].volume * 3 + 7;
    }
    return bars;
}

void load_pair(PerturbedPair& pair, const std::string& tag, int cut) {
    SyntheticConfig cfg;
    cfg.seed        = 7;
    cfg.num_days    = 12;
    const auto bars = generate_synthetic_bars(cfg);
    pair.cut        = cut;

    const auto dir = std::filesystem::temp_directory_path() / "wfo_gpu_lookahead";
    std::filesystem::create_directories(dir);
    const std::string a = (dir / (tag + "_a.bin")).string();
    const std::string b = (dir / (tag + "_b.bin")).string();
    std::string       err;
    ASSERT_TRUE(write_bar_file(a, "SYN", cfg.timeframe_sec, bars, &err)) << err;
    ASSERT_TRUE(write_bar_file(b, "SYN", cfg.timeframe_sec, perturb_after(bars, cut), &err)) << err;
    ASSERT_TRUE(pair.base.load(a));
    ASSERT_TRUE(pair.perturbed.load(b));
}

bool same_value(float x, float y) {
    if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y);
    return std::memcmp(&x, &y, sizeof(float)) == 0;
}

// Compute one indicator spec over all bars through the production dispatcher.
std::vector<float> compute_spec(const MarketData& md, const IndicatorSpec& spec) {
    const int          n = md.bar_count();
    std::vector<float> o(n), h(n), l(n), c(n);
    for (int i = 0; i < n; ++i) {
        o[i] = md.bar(i).open;
        h[i] = md.bar(i).high;
        l[i] = md.bar(i).low;
        c[i] = md.bar(i).close;
    }
    std::vector<float>  out(n);
    std::vector<float*> outputs = {out.data()};
    CLContext           ctx;  // not initialized: CPU reference path
    CLIndicators        ind;
    ind.compute_with_bars_anchored(ctx, md.bars(), 0, o.data(), h.data(), l.data(), c.data(), n, md,
                                   {spec}, outputs);
    return out;
}

// Every signal the screen and walk-forward stages backtest on: each
// technical indicator in the grid, normalized, then each gated session signal.
std::vector<std::vector<float>> full_signal_set(const MarketData& md) {
    const int          n = md.bar_count();
    std::vector<float> o(n), h(n), l(n), c(n);
    for (int i = 0; i < n; ++i) {
        o[i] = md.bar(i).open;
        h[i] = md.bar(i).high;
        l[i] = md.bar(i).low;
        c[i] = md.bar(i).close;
    }
    const IndicatorGrid        grid;
    std::vector<IndicatorSpec> specs;
    for (int g = 0; g < grid.size(); ++g)
        for (int s = 0; s < indicator_num_series(grid.params(g)); ++s)
            specs.push_back(IndicatorSpec::from_params(grid.params(g), s));
    std::vector<std::vector<float>> raw(specs.size(), std::vector<float>(n));
    std::vector<float*>             raw_ptrs;
    for (auto& r : raw) raw_ptrs.push_back(r.data());
    CLContext    ctx;
    CLIndicators ind;
    ind.compute_with_bars(ctx, md.bars(), 0, o.data(), h.data(), l.data(), c.data(), n, specs,
                          raw_ptrs);
    std::vector<std::vector<float>> out(grid.size(), std::vector<float>(n));
    std::vector<float*>             out_ptrs;
    for (auto& r : out) out_ptrs.push_back(r.data());
    normalize_all_indicators(grid, raw_ptrs, md.bars(), n, out_ptrs);

    const SessionSignalGrid ss;
    const int               day_start = md.first_session_day_in_range(0);
    const int               day_end   = md.last_session_day_in_range(n) + 1;
    for (int g = 0; g < ss.size(); ++g) {
        auto sig = create_session_signal(ss.params(g));
        if (!sig) continue;
        const int          ns   = sig->num_signals();
        const int          days = day_end - day_start;
        std::vector<float> day_values(static_cast<size_t>(ns) * days);
        sig->compute(md, day_start, day_end, day_values.data());
        const auto avail = sig->signal_available_after_et_seconds();
        for (int s = 0; s < ns; ++s) {
            std::vector<float> bar_signal(n, 0.0f);
            broadcast_session_signal_to_bars(md, &day_values[static_cast<size_t>(s) * days],
                                             day_start, day_end, avail[s], 0, n, bar_signal.data(),
                                             sig->suppress_on_halt());
            out.push_back(std::move(bar_signal));
        }
    }
    return out;
}

const WindowMode kAllModes[] = {
    WindowMode::ROLLING,     WindowMode::ANCHORED_1M,  WindowMode::ANCHORED_2M,
    WindowMode::ANCHORED_5M, WindowMode::ANCHORED_15M, WindowMode::ANCHORED_30M,
    WindowMode::ANCHORED_1H,
};

std::vector<IndicatorSpec> representative_specs() {
    std::vector<IndicatorSpec> specs;
    IndicatorSpec              s;
    s.type        = IndicatorType::EMA_SMA_CROSS;
    s.fast_period = 5;
    s.slow_period = 12;
    specs.push_back(s);
    s            = IndicatorSpec{};
    s.type       = IndicatorType::RSI;
    s.rsi_period = 7;
    specs.push_back(s);
    s            = IndicatorSpec{};
    s.type       = IndicatorType::ATR;
    s.atr_period = 7;
    specs.push_back(s);
    s             = IndicatorSpec{};
    s.type        = IndicatorType::MACD;
    s.macd_series = 2;
    specs.push_back(s);
    s                  = IndicatorSpec{};
    s.type             = IndicatorType::BOLLINGER;
    s.bollinger_period = 10;
    s.bollinger_series = 1;
    specs.push_back(s);
    s                = IndicatorSpec{};
    s.type           = IndicatorType::STOCHASTIC;
    s.stoch_k_period = 9;
    specs.push_back(s);
    s                 = IndicatorSpec{};
    s.type            = IndicatorType::DONCHIAN;
    s.donchian_period = 10;
    specs.push_back(s);
    s               = IndicatorSpec{};
    s.type          = IndicatorType::OBV;
    s.obv_smoothing = 5;
    specs.push_back(s);
    return specs;
}

}  // namespace

// Every indicator type above, in every window mode, ignores bars after t.
// The cut sits mid-way through an hour so it falls inside a forming
// higher-timeframe bar for every anchored mode.
TEST(Lookahead, IndicatorsIgnoreFutureBarsInEveryWindowMode) {
    PerturbedPair pair;
    load_pair(pair, "indicators", 8 * 2760 + 37);
    ASSERT_FALSE(HasFatalFailure());

    for (const WindowMode mode : kAllModes) {
        for (IndicatorSpec spec : representative_specs()) {
            spec.window_mode = mode;
            const auto a     = compute_spec(pair.base, spec);
            const auto b     = compute_spec(pair.perturbed, spec);
            int        valid = 0;
            for (int i = 0; i <= pair.cut; ++i) {
                ASSERT_TRUE(same_value(a[i], b[i]))
                    << "indicator " << static_cast<int>(spec.type) << " mode "
                    << static_cast<int>(mode) << " bar " << i << " changed when only bars after "
                    << pair.cut << " changed";
                if (!std::isnan(a[i])) ++valid;
            }
            EXPECT_GT(valid, 0) << "indicator " << static_cast<int>(spec.type) << " mode "
                                << static_cast<int>(mode) << " never warmed up";
            // The perturbation must actually reach later outputs, otherwise the
            // test proves nothing.
            bool changed_later = false;
            for (int i = pair.cut + 1; i < pair.base.bar_count() && !changed_later; ++i)
                changed_later = !same_value(a[i], b[i]);
            EXPECT_TRUE(changed_later)
                << "perturbation had no effect for indicator " << static_cast<int>(spec.type);
        }
    }
}

// The completed map only ever points at higher-timeframe bars whose last
// base bar has already closed.
TEST(Lookahead, CompletedMapPointsAtFinishedBars) {
    SyntheticConfig cfg;
    cfg.num_days    = 2;
    const auto bars = generate_synthetic_bars(cfg);
    const auto path =
        (std::filesystem::temp_directory_path() / "wfo_gpu_completed_map.bin").string();
    ASSERT_TRUE(write_bar_file(path, "SYN", 30, bars));
    MarketData md;
    ASSERT_TRUE(md.load(path));

    for (const int tf : {60, 120, 300, 900, 1800, 3600}) {
        const auto& t = md.timeframe(tf);
        ASSERT_EQ(static_cast<int>(t.completed_map.size()), md.bar_count());
        // Last base bar index of each TF bar.
        std::vector<int> last_child(t.bars.size(), -1);
        for (int i = 0; i < md.bar_count(); ++i) last_child[t.parent_map[i]] = i;
        for (int i = 0; i < md.bar_count(); ++i) {
            const int k = t.completed_map[i];
            ASSERT_LE(k, t.parent_map[i]);
            ASSERT_GE(k, t.parent_map[i] - 1);
            if (k >= 0) {
                ASSERT_LE(last_child[k], i) << "tf " << tf << " bar " << i;
            }
        }
    }
}

// The anchored broadcast must not hand base bar i the value of the TF bar
// that is still forming. With parent_map it would; with completed_map it
// lags by exactly one TF bar inside a block.
TEST(Lookahead, AnchoredBroadcastUsesOnlyCompletedBars) {
    const int        ratio   = 10;
    const int        tf_bars = 30;
    const int        n       = ratio * tf_bars;
    std::vector<Bar> tf(tf_bars);
    for (int t = 0; t < tf_bars; ++t) {
        tf[t]       = Bar{};
        tf[t].close = 100.0f + static_cast<float>(t);
        tf[t].open = tf[t].high = tf[t].low = tf[t].close;
    }
    std::vector<int> completed(n);
    for (int i = 0; i < n; ++i) {
        const int t  = i / ratio;
        completed[i] = (i % ratio == ratio - 1) ? t : t - 1;
    }
    EMASMACross        ema(MAType::EMA, 1, MAType::EMA, 2);  // EMA(1) == close
    std::vector<float> out(static_cast<size_t>(n) * 2);
    compute_anchored(ema, tf.data(), tf_bars, completed.data(), 0, n, out.data());
    for (int i = ratio * 5; i < n; ++i) {
        const int k = completed[i];
        if (std::isnan(out[i])) continue;
        EXPECT_FLOAT_EQ(out[i], 100.0f + static_cast<float>(k)) << "bar " << i;
        EXPECT_LE(k, i / ratio);
    }
}

// A halt (a long gap inside core hours) only blocks entries from the gap
// onward. Flagging the whole session would use knowledge, at 09:30, of a gap
// that only happens at 11:00.
TEST(Lookahead, HaltFlagStartsAtTheGap) {
    SyntheticConfig cfg;
    cfg.num_days = 2;
    auto bars    = generate_synthetic_bars(cfg);

    // Drop 20 minutes of bars starting 11:00 ET on the first session's day
    // (2024-01-02, EST, so 16:00 UTC).
    const int64_t gap_start =
        (civil::days_from_civil(2024, 1, 2) * 86400 + 16 * 3600) * civil::NS_PER_SEC;
    const int64_t    gap_end = gap_start + 20 * 60 * civil::NS_PER_SEC;
    std::vector<Bar> holed;
    for (const Bar& b : bars)
        if (b.timestamp < gap_start || b.timestamp >= gap_end) holed.push_back(b);
    ASSERT_EQ(holed.size(), bars.size() - 40);

    const auto path = (std::filesystem::temp_directory_path() / "wfo_gpu_halt.bin").string();
    ASSERT_TRUE(write_bar_file(path, "SYN", 30, holed));
    MarketData md;
    ASSERT_TRUE(md.load(path));

    int before = 0, after = 0, other_session = 0;
    for (int i = 0; i < md.bar_count(); ++i) {
        const auto& s       = md.session(i);
        const bool  flagged = s.halt_flag != 0;
        EXPECT_EQ(flagged, (s.flags & SESSION_HALTED) != 0) << i;
        if (s.session_day != 0) {
            other_session += flagged;
        } else if (md.bar(i).timestamp < gap_start) {
            before += flagged;
        } else {
            after += !flagged;
        }
    }
    EXPECT_EQ(before, 0) << "bars before the gap must not be halted";
    EXPECT_EQ(after, 0) << "bars from the gap to the close must be halted";
    EXPECT_EQ(other_session, 0);
}

// The full production signal path (every technical indicator in the grid,
// normalized to [-1, +1], followed by every session signal broadcast with
// its availability gate) ignores bars after t. This is the exact signal set
// the screen and walk-forward stages backtest on.
class FullPipelineLookahead : public ::testing::TestWithParam<int> {};

// Cuts at three times of day (offsets from an 18:00 ET session open): the
// evening, the overnight session, and inside regular trading hours. Gates
// that compare raw clock times break differently at each.
INSTANTIATE_TEST_SUITE_P(Lookahead, FullPipelineLookahead,
                         ::testing::Values(8 * 2760 + 150, 8 * 2760 + 1337, 8 * 2760 + 2100));

TEST_P(FullPipelineLookahead, SignalsIgnoreFutureBars) {
    PerturbedPair pair;
    load_pair(pair, "pipeline" + std::to_string(GetParam()), GetParam());
    ASSERT_FALSE(HasFatalFailure());

    const auto a = full_signal_set(pair.base);
    const auto b = full_signal_set(pair.perturbed);
    ASSERT_EQ(a.size(), b.size());
    int leaking = 0;
    for (size_t k = 0; k < a.size(); ++k) {
        for (int i = 0; i <= pair.cut; ++i) {
            if (!same_value(a[k][i], b[k][i])) {
                ADD_FAILURE() << "signal " << k << " bar " << i << ": " << a[k][i] << " vs "
                              << b[k][i] << " (cut at " << pair.cut << ")";
                ++leaking;
                break;
            }
        }
        if (leaking >= 20) break;
    }
}

// Trades, not just signals: a sample of strategies drawn from the whole grid
// (one to four signals, every aggregation, exit and trading-hours option,
// session signals included) is backtested on the base and perturbed data.
// Every trade that closes by the cut must be identical in both runs.
TEST(Lookahead, TradesClosedBeforeTheCutIgnoreFutureBars) {
    PerturbedPair pair;
    load_pair(pair, "trades", 9 * 2760 + 611);
    ASSERT_FALSE(HasFatalFailure());
    const auto                a = full_signal_set(pair.base);
    const auto                b = full_signal_set(pair.perturbed);
    std::vector<const float*> pa, pb;
    for (const auto& v : a) pa.push_back(v.data());
    for (const auto& v : b) pb.push_back(v.data());
    const int n_signals = static_cast<int>(a.size());

    const StrategyGrid grid(n_signals);
    const CostModel    cost            = CostModel::es();
    const int64_t      total           = std::min<int64_t>(grid.total_count(), 2'000'000'000LL);
    int                compared_trades = 0, strategies = 0;
    for (int64_t id = 7; id < total && strategies < 300; id += total / 20000) {
        const StrategyGridEntry e = grid.entry_at(id);
        if (e.is_redundant()) continue;
        ++strategies;
        const StrategyConfig sc = e.to_strategy_config();
        BacktestEngine       ea, eb;
        ea.run(pair.base.bars(), pair.base.sessions(), 0, pair.base.bar_count(), pa.data(),
               n_signals, nullptr, sc, cost, 1);
        eb.run(pair.perturbed.bars(), pair.perturbed.sessions(), 0, pair.perturbed.bar_count(),
               pb.data(), n_signals, nullptr, sc, cost, 1);
        std::vector<Trade> ta, tb;
        for (const Trade& t : ea.last_trades())
            if (t.exit_bar <= pair.cut) ta.push_back(t);
        for (const Trade& t : eb.last_trades())
            if (t.exit_bar <= pair.cut) tb.push_back(t);
        ASSERT_EQ(ta.size(), tb.size()) << "strategy " << id;
        for (size_t k = 0; k < ta.size(); ++k) {
            ASSERT_EQ(ta[k].entry_bar, tb[k].entry_bar) << "strategy " << id;
            ASSERT_EQ(ta[k].exit_bar, tb[k].exit_bar) << "strategy " << id;
            ASSERT_EQ(ta[k].exit_price, tb[k].exit_price) << "strategy " << id;
        }
        compared_trades += static_cast<int>(ta.size());
    }
    EXPECT_GE(strategies, 200);
    EXPECT_GT(compared_trades, 1000) << "the sample should actually trade before the cut";
}
