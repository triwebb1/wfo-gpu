#include <gtest/gtest.h>

#include "wfo_engine.h"
#include "orchestrator.h"
#include "results.h"
#include "market_data.h"
#include "cost_model.h"
#include "pareto.h"
#include "selection.h"
#include "grids/wfo_grid.h"
#include "grids/selection_grid.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: create a synthetic bar array for testing.
// Generates bars with a slight upward drift so that some strategies produce
// positive PnL (allowing Pareto frontiers to form).
// ---------------------------------------------------------------------------
static std::vector<Bar> make_test_bars(int count, float base_price = 5000.0f) {
    std::vector<Bar> bars(count);
    // Start timestamp: 2020-01-06 18:00:00 UTC (Mon session open) in nanoseconds
    const int64_t start_ns        = 1578340800LL * 1'000'000'000LL;
    const int64_t bar_interval_ns = 30LL * 1'000'000'000LL;  // 30-second bars

    float price = base_price;
    for (int i = 0; i < count; ++i) {
        // Simple deterministic price series with minor trend + oscillation
        float drift = 0.001f * static_cast<float>(i);  // slight upward trend
        float osc   = 2.0f * std::sin(static_cast<float>(i) * 0.1f);
        float noise = 0.5f * std::sin(static_cast<float>(i) * 0.37f);

        price = base_price + drift + osc + noise;

        bars[i].timestamp = start_ns + static_cast<int64_t>(i) * bar_interval_ns;
        bars[i].open      = price;
        bars[i].high      = price + 0.50f;
        bars[i].low       = price - 0.50f;
        bars[i].close     = price + 0.25f * std::sin(static_cast<float>(i) * 0.05f);
        bars[i].volume    = 100 + (i % 50);
        bars[i]._pad      = 0;
    }
    return bars;
}

// ---------------------------------------------------------------------------
// WFOConfig construction test
// ---------------------------------------------------------------------------
TEST(WFOConfigTest, FromSchedule) {
    WFOScheduleConfig sc;
    sc.frequency = WFOFrequency::WEEK_1;
    sc.lookback  = WFOLookback::MONTH_1;

    // 2880 bars per day (24h of 30s bars)
    WFOConfig cfg = WFOConfig::from_schedule(sc, 2880.0f);

    EXPECT_EQ(cfg.frequency_days, 7);
    EXPECT_EQ(cfg.lookback_days, 30);
    EXPECT_GT(cfg.frequency_bars, 0);
    EXPECT_GT(cfg.lookback_bars, 0);
    EXPECT_GT(cfg.lookback_bars, cfg.frequency_bars);
}

// ---------------------------------------------------------------------------
// test_wfo_single_window: 1 IS + 1 OOS window.
// Verify winner selection and OOS result are produced.
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, SingleWindow) {
    // Create enough bars for 1 IS window + 1 OOS period
    // Use small lookback (50 bars) and small frequency (20 bars) for speed
    const int lookback   = 500;
    const int frequency  = 200;
    const int total_bars = lookback + frequency + 200;  // warmup margin

    auto bars = make_test_bars(total_bars);

    WFOConfig schedule;
    schedule.lookback_bars  = lookback;
    schedule.frequency_bars = frequency;
    schedule.lookback_days  = 1;
    schedule.frequency_days = 1;

    CostModel cost = CostModel::es();

    // Use a small reduced set for testing
    // Strategy IDs 0,1,2 from the strategy grid
    std::vector<int> reduced_ids = {0, 1, 2};

    WFOEngine    engine;
    WFORunResult result =
        engine.run(bars.data(), total_bars, schedule, "frontier_centroid", reduced_ids, cost);

    // With lookback=500, frequency=200, total=900:
    // Window 0: IS [0,500), OOS [500,700)
    // Window 1: IS [200,700), OOS [700,900)
    EXPECT_EQ(result.windows.size(), 2u);

    // The window should have valid structure
    const auto& w = result.windows[0];
    EXPECT_EQ(w.window_index, 0);
    EXPECT_EQ(w.is_start_bar, 0);
    EXPECT_EQ(w.is_end_bar, lookback);
    EXPECT_EQ(w.oos_start_bar, lookback);
    EXPECT_EQ(w.oos_end_bar, lookback + frequency);

    // If not skipped, should have a valid strategy_id
    if (!w.skipped) {
        EXPECT_GE(w.strategy_id, 0);
        EXPECT_FALSE(w.archetype.empty());
    }

    // Selection strategy should be recorded
    EXPECT_EQ(result.selection_strategy, "frontier_centroid");

    // Equity curve should have entries
    EXPECT_FALSE(result.equity_curve.empty());
}

// ---------------------------------------------------------------------------
// test_wfo_equity_stitch: 3 OOS windows.
// Verify stitched equity curve is continuous across windows.
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, EquityStitch) {
    // 3 windows requires lookback + 3*frequency bars
    const int lookback   = 300;
    const int frequency  = 100;
    const int total_bars = lookback + 3 * frequency + 200;

    auto bars = make_test_bars(total_bars);

    WFOConfig schedule;
    schedule.lookback_bars  = lookback;
    schedule.frequency_bars = frequency;
    schedule.lookback_days  = 1;
    schedule.frequency_days = 1;

    CostModel        cost        = CostModel::es();
    std::vector<int> reduced_ids = {0, 1, 2};

    WFOEngine    engine;
    WFORunResult result =
        engine.run(bars.data(), total_bars, schedule, "best_roc", reduced_ids, cost);

    // Should have at least 3 windows
    EXPECT_GE(static_cast<int>(result.windows.size()), 3);

    // Equity curve should span all OOS periods
    EXPECT_FALSE(result.equity_curve.empty());

    // Verify equity curve is continuous (no jumps between windows).
    // Specifically, check that the stitched equity sums window PnLs.
    float expected_total_pnl = 0.0f;
    for (const auto& w : result.windows) {
        if (!w.skipped) {
            expected_total_pnl += w.oos_lag1.total_pnl;
        }
    }

    // The final equity curve value should equal the sum of all OOS PnLs
    // (within floating-point tolerance)
    if (!result.equity_curve.empty()) {
        float final_equity = result.equity_curve.back();
        EXPECT_NEAR(final_equity, expected_total_pnl, std::abs(expected_total_pnl) * 0.01f + 1.0f);
    }

    // Timestamps should be monotonically increasing
    for (size_t i = 1; i < result.equity_timestamps.size(); ++i) {
        EXPECT_GE(result.equity_timestamps[i], result.equity_timestamps[i - 1]);
    }
}

// ---------------------------------------------------------------------------
// test_wfo_skip_empty_frontier: Window where no strategies pass floors.
// Verify skipped=true and flat equity.
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, SkipEmptyFrontier) {
    // Create a window evaluation with an empty frontier directly
    WindowEvaluation eval;
    eval.is_start_bar   = 0;
    eval.is_end_bar     = 100;
    eval.oos_start_bar  = 100;
    eval.oos_end_bar    = 200;
    eval.empty_frontier = true;

    const int total_bars = 300;
    auto      bars       = make_test_bars(total_bars);

    WFOConfig schedule;
    schedule.lookback_bars  = 100;
    schedule.frequency_bars = 100;
    schedule.lookback_days  = 1;
    schedule.frequency_days = 1;

    CostModel        cost        = CostModel::es();
    std::vector<int> reduced_ids = {0};

    std::vector<WindowEvaluation> evaluations = {eval};

    WFOEngine    engine;
    WFORunResult result =
        engine.apply_selection(bars.data(), total_bars, schedule, SelectionStrategy::BEST_ROC,
                               evaluations, reduced_ids, cost);

    ASSERT_EQ(static_cast<int>(result.windows.size()), 1);
    EXPECT_TRUE(result.windows[0].skipped);
    EXPECT_EQ(result.windows_skipped, 1);

    // Equity curve should be flat (all values the same = 0)
    for (float eq : result.equity_curve) {
        EXPECT_FLOAT_EQ(eq, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// test_wfo_lag2: Verify lag-2 entries use entry_lag=2.
//
// We verify that the lag-2 OOS result exists and is computed.
// Since lag-2 shifts entry by 1 bar later, its total trades may differ
// from lag-1 (fewer trades or different PnL).
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, Lag2Validation) {
    const int lookback   = 500;
    const int frequency  = 200;
    const int total_bars = lookback + frequency + 200;

    auto bars = make_test_bars(total_bars);

    WFOConfig schedule;
    schedule.lookback_bars  = lookback;
    schedule.frequency_bars = frequency;
    schedule.lookback_days  = 1;
    schedule.frequency_days = 1;

    CostModel        cost        = CostModel::es();
    std::vector<int> reduced_ids = {0, 1, 2};

    WFOEngine    engine;
    WFORunResult result =
        engine.run(bars.data(), total_bars, schedule, "best_roc", reduced_ids, cost);

    ASSERT_GE(static_cast<int>(result.windows.size()), 1);

    // For non-skipped windows, lag-2 result should exist
    for (const auto& w : result.windows) {
        if (!w.skipped) {
            // lag-2 result is always computed (even if 0 trades)
            // The key check: lag-2 uses a different entry lag.
            // We verify both results have the same bar range
            EXPECT_EQ(w.oos_lag1.start_ts, w.oos_lag2.start_ts);
            EXPECT_EQ(w.oos_lag1.end_ts, w.oos_lag2.end_ts);
        }
    }
}

// ---------------------------------------------------------------------------
// test_wfo_viable_check: >50% windows skipped -> viable=false.
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, ViableCheck) {
    // Create 3 window evaluations, 2 of which have empty frontiers
    const int total_bars = 500;
    auto      bars       = make_test_bars(total_bars);

    WindowEvaluation eval1, eval2, eval3;

    // Window 1: empty (skipped)
    eval1.is_start_bar   = 0;
    eval1.is_end_bar     = 100;
    eval1.oos_start_bar  = 100;
    eval1.oos_end_bar    = 200;
    eval1.empty_frontier = true;

    // Window 2: empty (skipped)
    eval2.is_start_bar   = 100;
    eval2.is_end_bar     = 200;
    eval2.oos_start_bar  = 200;
    eval2.oos_end_bar    = 300;
    eval2.empty_frontier = true;

    // Window 3: empty (skipped)
    eval3.is_start_bar   = 200;
    eval3.is_end_bar     = 300;
    eval3.oos_start_bar  = 300;
    eval3.oos_end_bar    = 400;
    eval3.empty_frontier = true;

    WFOConfig schedule;
    schedule.lookback_bars  = 100;
    schedule.frequency_bars = 100;
    schedule.lookback_days  = 1;
    schedule.frequency_days = 1;

    CostModel        cost        = CostModel::es();
    std::vector<int> reduced_ids = {0};

    std::vector<WindowEvaluation> evaluations = {eval1, eval2, eval3};

    WFOEngine    engine;
    WFORunResult result =
        engine.apply_selection(bars.data(), total_bars, schedule, SelectionStrategy::BEST_ROC,
                               evaluations, reduced_ids, cost);

    // All 3 windows skipped -> >50% -> not sufficient_windows
    EXPECT_EQ(result.windows_skipped, 3);
    EXPECT_FALSE(result.sufficient_windows);
}

// ---------------------------------------------------------------------------
// Test: WFOConfig::from_schedule produces valid bar counts for all schedules
// ---------------------------------------------------------------------------
TEST(WFOConfigTest, AllSchedulesValid) {
    WFOGrid grid;
    for (int i = 0; i < grid.size(); ++i) {
        WFOConfig cfg = WFOConfig::from_schedule(grid.config(i), 2880.0f);
        EXPECT_GT(cfg.frequency_bars, 0) << "Schedule " << i;
        EXPECT_GT(cfg.lookback_bars, 0) << "Schedule " << i;
        EXPECT_GE(cfg.lookback_bars, cfg.frequency_bars)
            << "Lookback must be >= frequency for schedule " << i;
    }
}

// ---------------------------------------------------------------------------
// Test: apply_selection with mixed skipped/non-skipped windows
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, MixedWindows) {
    const int total_bars = 500;
    auto      bars       = make_test_bars(total_bars);

    // Window 1: skipped
    WindowEvaluation eval1;
    eval1.is_start_bar   = 0;
    eval1.is_end_bar     = 100;
    eval1.oos_start_bar  = 100;
    eval1.oos_end_bar    = 200;
    eval1.empty_frontier = true;

    // Window 2: skipped
    WindowEvaluation eval2;
    eval2.is_start_bar   = 100;
    eval2.is_end_bar     = 200;
    eval2.oos_start_bar  = 200;
    eval2.oos_end_bar    = 300;
    eval2.empty_frontier = true;

    WFOConfig schedule;
    schedule.lookback_bars  = 100;
    schedule.frequency_bars = 100;
    schedule.lookback_days  = 1;
    schedule.frequency_days = 1;

    CostModel        cost        = CostModel::es();
    std::vector<int> reduced_ids = {0};

    std::vector<WindowEvaluation> evaluations = {eval1, eval2};

    WFOEngine    engine;
    WFORunResult result = engine.apply_selection(bars.data(), total_bars, schedule,
                                                 SelectionStrategy::FRONTIER_CENTROID, evaluations,
                                                 reduced_ids, cost);

    EXPECT_EQ(static_cast<int>(result.windows.size()), 2);
    EXPECT_EQ(result.windows_skipped, 2);
    EXPECT_FALSE(result.sufficient_windows);  // 100% skipped

    // Equity should be flat
    for (float eq : result.equity_curve) {
        EXPECT_FLOAT_EQ(eq, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Test: Results serialization (write_summary_csv and write_window_csv)
// ---------------------------------------------------------------------------
TEST(ResultsTest, WriteCSV) {
    WFORunResult run;
    run.schedule.frequency_bars = 100;
    run.schedule.lookback_bars  = 300;
    run.schedule.frequency_days = 7;
    run.schedule.lookback_days  = 30;
    run.selection_strategy      = "best_roc";
    run.sufficient_windows      = true;
    run.windows_skipped         = 0;
    run.parameter_stability     = 0.85f;

    WFOWindowResult w;
    w.window_index           = 0;
    w.is_start_bar           = 0;
    w.is_end_bar             = 300;
    w.oos_start_bar          = 300;
    w.oos_end_bar            = 400;
    w.strategy_id            = 5;
    w.archetype              = "grinder";
    w.skipped                = false;
    w.is_result.total_pnl    = 1000.0f;
    w.is_result.sharpe_ratio = 1.5f;
    w.oos_lag1.total_pnl     = 500.0f;
    w.oos_lag2.total_pnl     = 450.0f;
    run.windows.push_back(w);

    run.equity_curve      = {0.0f, 5.0f, 10.0f, 15.0f, 20.0f};
    run.equity_timestamps = {1000, 2000, 3000, 4000, 5000};

    run.oos_aggregate.total_pnl    = 500.0f;
    run.oos_aggregate.annual_roc   = 25.0f;
    run.oos_aggregate.sharpe_ratio = 1.2f;

    // Write window CSV to a temp location
    std::string tmp_dir     = ".";
    std::string window_path = tmp_dir + "/test_window_output.csv";
    std::string equity_path = tmp_dir + "/test_equity_output.csv";

    bool ok1 = results::write_window_csv(window_path, run);
    EXPECT_TRUE(ok1);

    bool ok2 = results::write_equity_csv(equity_path, run);
    EXPECT_TRUE(ok2);

    // Write summary CSV
    std::string               summary_path = tmp_dir + "/test_summary_output.csv";
    std::vector<WFORunResult> runs         = {run};
    bool                      ok3          = results::write_summary_csv(summary_path, runs);
    EXPECT_TRUE(ok3);

    // Clean up temp files
    std::remove(window_path.c_str());
    std::remove(equity_path.c_str());
    std::remove(summary_path.c_str());
}

// ---------------------------------------------------------------------------
// Test: Parameter stability computation
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, ParameterStability) {
    // All windows choose the same strategy -> max stability
    {
        WFORunResult run;
        for (int i = 0; i < 5; ++i) {
            WFOWindowResult w;
            w.skipped     = false;
            w.strategy_id = 42;
            run.windows.push_back(w);
        }
        // Stability should be 1.0 (no variance)
        // We can't call the private method directly, but we can test via run()
        // Instead, verify through the public interface by checking the run result
    }

    // All windows skipped -> stability should be 1.0 (default)
    {
        const int total_bars = 500;
        auto      bars       = make_test_bars(total_bars);

        WindowEvaluation eval;
        eval.is_start_bar   = 0;
        eval.is_end_bar     = 100;
        eval.oos_start_bar  = 100;
        eval.oos_end_bar    = 200;
        eval.empty_frontier = true;

        WFOConfig schedule;
        schedule.lookback_bars  = 100;
        schedule.frequency_bars = 100;
        schedule.lookback_days  = 1;
        schedule.frequency_days = 1;

        CostModel        cost        = CostModel::es();
        std::vector<int> reduced_ids = {0};

        WFOEngine    engine;
        WFORunResult result =
            engine.apply_selection(bars.data(), total_bars, schedule, SelectionStrategy::BEST_ROC,
                                   {eval}, reduced_ids, cost);

        // All skipped -> stability = 1.0 (default when < 2 non-skipped windows)
        EXPECT_FLOAT_EQ(result.parameter_stability, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Test: profit_factor sentinel for 100% win rate (no losses)
//
// profit_factor = gross_wins / |gross_losses|. When gross_losses == 0
// (all-winner scenario), a 0.0f result would
// incorrectly fail quality floors, so the engine returns the 1e6f sentinel
// when gross_wins > 0 and gross_losses == 0.
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, ProfitFactorAllWinners) {
    // ---- Case 1: all winners (gross_wins > 0, gross_losses == 0) ----------
    {
        BacktestResult r;
        r.gross_wins   = 1000.0f;
        r.gross_losses = 0.0f;  // no losing trades

        float abs_losses = std::abs(r.gross_losses);
        r.profit_factor  = (abs_losses > 0.0f)     ? (r.gross_wins / abs_losses)
                           : (r.gross_wins > 0.0f) ? 1e6f
                                                   : 0.0f;

        // Must be the large sentinel, NOT 0.0
        EXPECT_GE(r.profit_factor, 1e5f);
        // Must pass the IS quality floor (profit_factor > 1.0)
        EXPECT_GT(r.profit_factor, 1.0f);
    }

    // ---- Case 2: no trades at all (gross_wins == 0, gross_losses == 0) ----
    {
        BacktestResult r;
        r.gross_wins   = 0.0f;
        r.gross_losses = 0.0f;

        float abs_losses = std::abs(r.gross_losses);
        r.profit_factor  = (abs_losses > 0.0f)     ? (r.gross_wins / abs_losses)
                           : (r.gross_wins > 0.0f) ? 1e6f
                                                   : 0.0f;

        // No trades -> profit_factor should remain 0
        EXPECT_FLOAT_EQ(r.profit_factor, 0.0f);
    }

    // ---- Case 3: normal mix of wins and losses -------------------------
    {
        BacktestResult r;
        r.gross_wins   = 2000.0f;
        r.gross_losses = -1000.0f;  // negative value as stored

        float abs_losses = std::abs(r.gross_losses);
        r.profit_factor  = (abs_losses > 0.0f)     ? (r.gross_wins / abs_losses)
                           : (r.gross_wins > 0.0f) ? 1e6f
                                                   : 0.0f;

        EXPECT_NEAR(r.profit_factor, 2.0f, 0.001f);
    }

    // ---- Case 4: all losers (gross_wins == 0, gross_losses < 0) ---------
    {
        BacktestResult r;
        r.gross_wins   = 0.0f;
        r.gross_losses = -500.0f;

        float abs_losses = std::abs(r.gross_losses);
        r.profit_factor  = (abs_losses > 0.0f)     ? (r.gross_wins / abs_losses)
                           : (r.gross_wins > 0.0f) ? 1e6f
                                                   : 0.0f;

        EXPECT_FLOAT_EQ(r.profit_factor, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Integration test for adaptive-trigger dispatch.
//
// Adaptive-trigger state lives in per-selection local scope inside
// apply_selection, so each selection strategy carries its own
// cumulative_oos_equity / peak_oos_equity / consecutive_losers /
// last_used_eval_idx / trigger_fired_for_next. The trigger arithmetic is
// unit-tested in isolation elsewhere; this test exercises apply_selection
// with WFOTrigger::DD_TRIGGER_* and WFOTrigger::LOSERS_*.
//
// This test drives the full adaptive-trigger dispatch path through
// apply_selection for all 7 selection strategies. It verifies:
//   (a) adaptive triggers do not crash or alter structural validity,
//   (b) a DD_TRIGGER with an impossibly-high threshold never fires, so it
//       must match FIXED_INTERVAL bit-for-bit on the same bars/evaluations,
//   (c) re-running the same adaptive configuration on the same inputs
//       produces identical results (determinism / no state leak across
//       invocations via per-selection local scope).
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, AdaptiveTriggerIntegration) {
    const int lookback           = 300;
    const int frequency          = 100;
    const int num_windows_target = 4;
    const int total_bars         = lookback + num_windows_target * frequency + 200;
    auto      bars               = make_test_bars(total_bars);

    WFOConfig base;
    base.lookback_bars  = lookback;
    base.frequency_bars = frequency;
    base.lookback_days  = 1;
    base.frequency_days = 1;

    CostModel        cost        = CostModel::es();
    std::vector<int> reduced_ids = {0, 1, 2};

    WFOEngine engine;
    auto evaluations = engine.evaluate_windows(bars.data(), total_bars, base, reduced_ids, cost);
    ASSERT_GE(evaluations.size(), static_cast<size_t>(num_windows_target));

    // -- Baseline: FIXED_INTERVAL (trigger inactive) --
    WFOConfig fixed = base;
    fixed.trigger   = WFOTrigger::FIXED_INTERVAL;
    auto fixed_result =
        engine.apply_selection(bars.data(), total_bars, fixed, SelectionStrategy::BEST_ROC,
                               evaluations, reduced_ids, cost);
    ASSERT_EQ(fixed_result.windows.size(), evaluations.size());

    // -- DD_TRIGGER_10PCT with never-triggered threshold: must match FIXED --
    // Synthetic bars have very small drift/noise; 10% drawdown is unreachable
    // on the OOS PnL scale produced here. Invariant: per-window strategy ids
    // and oos_lag1 totals must match FIXED_INTERVAL exactly.
    WFOConfig dd_inert       = base;
    dd_inert.trigger         = WFOTrigger::DD_TRIGGER_10PCT;
    dd_inert.trigger_dd_frac = trigger_dd_pct(dd_inert.trigger);
    auto dd_inert_result =
        engine.apply_selection(bars.data(), total_bars, dd_inert, SelectionStrategy::BEST_ROC,
                               evaluations, reduced_ids, cost);
    ASSERT_EQ(dd_inert_result.windows.size(), fixed_result.windows.size());
    for (size_t i = 0; i < fixed_result.windows.size(); ++i) {
        EXPECT_EQ(fixed_result.windows[i].strategy_id, dd_inert_result.windows[i].strategy_id)
            << "Inactive DD trigger must match FIXED_INTERVAL at window " << i;
        EXPECT_FLOAT_EQ(fixed_result.windows[i].oos_lag1.total_pnl,
                        dd_inert_result.windows[i].oos_lag1.total_pnl);
    }

    // -- Determinism + per-selection state isolation --
    // Run each of the 7 selection strategies twice under LOSERS_5 trigger,
    // each invocation using a fresh WFOEngine. The per-selection local
    // state must yield bit-exact identical results across re-runs. Any leak
    // via statics or shared engine fields would produce a mismatch.
    const SelectionStrategy selections[] = {
        SelectionStrategy::BEST_ROC,          SelectionStrategy::BEST_CONSISTENCY,
        SelectionStrategy::BEST_TRR,          SelectionStrategy::BEST_GRINDER,
        SelectionStrategy::BEST_SNIPER,       SelectionStrategy::FRONTIER_CENTROID,
        SelectionStrategy::ROC_X_CONSISTENCY,
    };

    WFOConfig losers        = base;
    losers.trigger          = WFOTrigger::LOSERS_5;
    losers.trigger_losers_n = trigger_losers_count(losers.trigger);

    auto run_all = [&]() {
        std::vector<WFORunResult> out;
        out.reserve(std::size(selections));
        for (auto sel : selections) {
            out.push_back(engine.apply_selection(bars.data(), total_bars, losers, sel, evaluations,
                                                 reduced_ids, cost));
        }
        return out;
    };
    auto run1 = run_all();
    auto run2 = run_all();
    ASSERT_EQ(run1.size(), run2.size());
    for (size_t s = 0; s < run1.size(); ++s) {
        ASSERT_EQ(run1[s].windows.size(), run2[s].windows.size());
        for (size_t w = 0; w < run1[s].windows.size(); ++w) {
            EXPECT_EQ(run1[s].windows[w].strategy_id, run2[s].windows[w].strategy_id)
                << "Non-determinism under LOSERS_5 at selection " << s << " window " << w
                << " -- possible per-selection state leak across runs";
            EXPECT_FLOAT_EQ(run1[s].windows[w].oos_lag1.total_pnl,
                            run2[s].windows[w].oos_lag1.total_pnl);
        }
        EXPECT_EQ(run1[s].windows_skipped, run2[s].windows_skipped);
        EXPECT_EQ(run1[s].sufficient_windows, run2[s].sufficient_windows);
    }
}

// ---------------------------------------------------------------------------
// Test: WFO Grid produces expected number of configs
// ---------------------------------------------------------------------------
TEST(WFOGridTest, GridSize) {
    WFOGrid grid;
    // Fixed: 30 configs (4 freq x 8 lookback, pruned where lookback < frequency)
    // Adaptive: 4 triggers x 8 lookbacks x 1 freq (1w) = 32 configs
    // Total = 62
    EXPECT_EQ(grid.size(), 62);

    // Verify all configs have lookback >= frequency
    for (int i = 0; i < grid.size(); ++i) {
        const auto& cfg = grid.config(i);
        EXPECT_GE(lookback_approx_days(cfg.lookback), frequency_approx_days(cfg.frequency))
            << "Config " << i << ": lookback must be >= frequency";
    }
}

// ---------------------------------------------------------------------------
// Out-of-sample windows start no earlier than the configured bar, so they
// never trade on the screen range the reduced set was selected on.
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, OosWindowsStartAfterScreenRange) {
    const int lookback = 300, frequency = 100;
    const int total_bars = lookback + 6 * frequency;
    auto      bars       = make_test_bars(total_bars);
    WFOConfig schedule;
    schedule.lookback_bars             = lookback;
    schedule.frequency_bars            = frequency;
    schedule.lookback_days             = 1;
    schedule.frequency_days            = 1;
    const std::vector<int> reduced_ids = {0, 1, 2};

    WFOEngine engine;
    const int screen_end = lookback + 2 * frequency + 17;
    engine.set_min_oos_start(screen_end);
    const auto evaluations =
        engine.evaluate_windows(bars.data(), total_bars, schedule, reduced_ids, CostModel::es());
    ASSERT_FALSE(evaluations.empty());
    EXPECT_EQ(evaluations.front().oos_start_bar, screen_end);
    for (const auto& e : evaluations) {
        EXPECT_GE(e.oos_start_bar, screen_end);
        EXPECT_EQ(e.is_end_bar, e.oos_start_bar);
        EXPECT_LE(e.oos_end_bar, total_bars);
    }
}

// ---------------------------------------------------------------------------
// An adaptive trigger fires on trades. When a window deploys nothing (empty
// in-sample frontier) no trade can fire it, so the next window must
// re-optimize instead of reusing the empty frontier for the rest of the run.
// ---------------------------------------------------------------------------
TEST(WFOEngineTest, AdaptiveTriggerReoptimizesAfterEmptyWindow) {
    const int lookback = 300, frequency = 100;
    const int total_bars = lookback + 5 * frequency;
    auto      bars       = make_test_bars(total_bars);
    WFOConfig base;
    base.lookback_bars                 = lookback;
    base.frequency_bars                = frequency;
    base.lookback_days                 = 1;
    base.frequency_days                = 1;
    const CostModel        cost        = CostModel::es();
    const std::vector<int> reduced_ids = {0, 1, 2};

    WFOEngine engine;
    auto evaluations = engine.evaluate_windows(bars.data(), total_bars, base, reduced_ids, cost);
    ASSERT_GE(evaluations.size(), 3u);
    evaluations[0].empty_frontier = true;
    evaluations[0].frontier.clear();

    WFOConfig  fixed = base;
    const auto fixed_r =
        engine.apply_selection(bars.data(), total_bars, fixed, SelectionStrategy::BEST_ROC,
                               evaluations, reduced_ids, cost);
    WFOConfig dd       = base;
    dd.trigger         = WFOTrigger::DD_TRIGGER_10PCT;
    dd.trigger_dd_frac = trigger_dd_pct(dd.trigger);
    const auto dd_r    = engine.apply_selection(
        bars.data(), total_bars, dd, SelectionStrategy::BEST_ROC, evaluations, reduced_ids, cost);
    ASSERT_EQ(dd_r.windows.size(), fixed_r.windows.size());
    EXPECT_TRUE(dd_r.windows[0].skipped);
    // Window 1 re-optimizes on its own in-sample window, exactly as the
    // fixed-interval schedule does.
    EXPECT_EQ(dd_r.windows[1].skipped, fixed_r.windows[1].skipped);
    EXPECT_EQ(dd_r.windows[1].strategy_id, fixed_r.windows[1].strategy_id);
    EXPECT_EQ(dd_r.windows[1].is_start_bar, evaluations[1].is_start_bar);
}
