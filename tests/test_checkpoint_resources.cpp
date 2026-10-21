#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>

#include "checkpoint.h"
#include "resource_monitor.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// test_checkpoint_save_load
//
// Save a checkpoint with populated screen and walk-forward fields.
// Load it back and verify every field matches.
// ---------------------------------------------------------------------------
TEST(CheckpointTest, SaveLoad) {
    // Build a checkpoint with non-trivial data
    Checkpoint cp;
    cp.run_id = "test_run_42";
    cp.phase  = 1;

    // Screen progress
    cp.strategies_evaluated = 150;
    cp.total_strategies     = 500;

    // Add some completed BacktestResults
    for (int i = 0; i < 5; ++i) {
        BacktestResult r;
        r.start_ts             = 1000000 + i;
        r.end_ts               = 2000000 + i;
        r.total_trades         = 42 + i;
        r.trades_per_week      = 3.5f + static_cast<float>(i);
        r.win_rate             = 0.55f + 0.01f * static_cast<float>(i);
        r.avg_win              = 200.0f + static_cast<float>(i);
        r.avg_loss             = -150.0f - static_cast<float>(i);
        r.avg_hold_bars        = 10.0f;
        r.total_pnl            = 1000.0f * static_cast<float>(i + 1);
        r.gross_wins           = 5000.0f;
        r.gross_losses         = -3000.0f;
        r.max_drawdown         = 500.0f;
        r.max_live_drawdown    = 300.0f;
        r.sharpe_ratio         = 1.5f;
        r.calmar_ratio         = 2.0f;
        r.profit_factor        = 1.67f;
        r.payoff_ratio         = 1.33f;
        r.green_weeks_pct      = 0.65f;
        r.green_months_pct     = 0.70f;
        r.max_red_weeks        = 3;
        r.negative_years       = 0;
        r.annual_roc           = 25.0f;
        r.reward_risk_trade    = 0.02f;
        r.daily_profit_rate    = 0.1f;
        r.time_to_double       = 365.0f;
        r.trade_recovery_ratio = 0.24f;
        r.strategy_id          = i;
        r.viable               = true;
        cp.completed_results.push_back(r);
    }

    // Walk-forward progress
    cp.current_schedule_idx = 3;
    cp.current_window_idx   = 7;

    // Add a completed WFO run
    WFORunResult run;
    run.schedule.frequency_bars = 1000;
    run.schedule.lookback_bars  = 5000;
    run.schedule.frequency_days = 7;
    run.schedule.lookback_days  = 30;
    run.selection_strategy      = "best_roc";
    run.parameter_stability     = 0.85f;
    run.windows_skipped         = 1;
    run.sufficient_windows      = true;

    WFOWindowResult w;
    w.window_index               = 0;
    w.is_start_bar               = 0;
    w.is_end_bar                 = 5000;
    w.oos_start_bar              = 5000;
    w.oos_end_bar                = 6000;
    w.strategy_id                = 42;
    w.archetype                  = "grinder";
    w.skipped                    = false;
    w.is_result.total_pnl        = 10000.0f;
    w.is_result.sharpe_ratio     = 2.0f;
    w.oos_lag1.total_pnl         = 3000.0f;
    w.oos_lag2.total_pnl         = 2500.0f;
    w.pnl_decomp.bar_body_pct    = 85.0f;
    w.pnl_decomp.gap_pct         = 15.0f;
    w.pnl_decomp.is_real_edge    = true;
    w.pnl_decomp.is_gap_artifact = false;
    run.windows.push_back(w);

    run.equity_curve      = {0.0f, 100.0f, 200.0f, 150.0f, 300.0f};
    run.equity_timestamps = {1000, 2000, 3000, 4000, 5000};

    run.oos_aggregate.total_pnl  = 3000.0f;
    run.oos_aggregate.annual_roc = 20.0f;

    cp.completed_runs.push_back(run);

    // Save to a temp file
    std::string tmp_path = "test_checkpoint_save_load.wfck";
    ASSERT_TRUE(cp.save(tmp_path));

    // Verify the file is valid
    EXPECT_TRUE(Checkpoint::is_valid(tmp_path));

    // Load it back (returns std::optional)
    auto loaded_opt = Checkpoint::load(tmp_path);
    ASSERT_TRUE(loaded_opt.has_value());
    Checkpoint& loaded = *loaded_opt;

    // Verify all fields match
    EXPECT_EQ(loaded.run_id, cp.run_id);
    EXPECT_EQ(loaded.phase, cp.phase);

    // Screen
    EXPECT_EQ(loaded.strategies_evaluated, cp.strategies_evaluated);
    EXPECT_EQ(loaded.total_strategies, cp.total_strategies);
    ASSERT_EQ(loaded.completed_results.size(), cp.completed_results.size());

    for (size_t i = 0; i < cp.completed_results.size(); ++i) {
        const auto& orig = cp.completed_results[i];
        const auto& copy = loaded.completed_results[i];
        EXPECT_EQ(copy.start_ts, orig.start_ts) << "Result " << i;
        EXPECT_EQ(copy.end_ts, orig.end_ts) << "Result " << i;
        EXPECT_EQ(copy.total_trades, orig.total_trades) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.trades_per_week, orig.trades_per_week) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.win_rate, orig.win_rate) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.avg_win, orig.avg_win) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.avg_loss, orig.avg_loss) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.total_pnl, orig.total_pnl) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.sharpe_ratio, orig.sharpe_ratio) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.profit_factor, orig.profit_factor) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.green_weeks_pct, orig.green_weeks_pct) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.annual_roc, orig.annual_roc) << "Result " << i;
        EXPECT_FLOAT_EQ(copy.trade_recovery_ratio, orig.trade_recovery_ratio) << "Result " << i;
        EXPECT_EQ(copy.strategy_id, orig.strategy_id) << "Result " << i;
        EXPECT_EQ(copy.viable, orig.viable) << "Result " << i;
    }

    // Walk-forward
    EXPECT_EQ(loaded.current_schedule_idx, cp.current_schedule_idx);
    EXPECT_EQ(loaded.current_window_idx, cp.current_window_idx);
    ASSERT_EQ(loaded.completed_runs.size(), cp.completed_runs.size());

    const auto& orig_run = cp.completed_runs[0];
    const auto& copy_run = loaded.completed_runs[0];
    EXPECT_EQ(copy_run.schedule.frequency_bars, orig_run.schedule.frequency_bars);
    EXPECT_EQ(copy_run.schedule.lookback_bars, orig_run.schedule.lookback_bars);
    EXPECT_EQ(copy_run.schedule.frequency_days, orig_run.schedule.frequency_days);
    EXPECT_EQ(copy_run.schedule.lookback_days, orig_run.schedule.lookback_days);
    EXPECT_EQ(copy_run.selection_strategy, orig_run.selection_strategy);
    EXPECT_FLOAT_EQ(copy_run.parameter_stability, orig_run.parameter_stability);
    EXPECT_EQ(copy_run.windows_skipped, orig_run.windows_skipped);
    EXPECT_EQ(copy_run.sufficient_windows, orig_run.sufficient_windows);

    ASSERT_EQ(copy_run.windows.size(), orig_run.windows.size());
    EXPECT_EQ(copy_run.windows[0].window_index, orig_run.windows[0].window_index);
    EXPECT_EQ(copy_run.windows[0].strategy_id, orig_run.windows[0].strategy_id);
    EXPECT_EQ(copy_run.windows[0].archetype, orig_run.windows[0].archetype);
    EXPECT_EQ(copy_run.windows[0].skipped, orig_run.windows[0].skipped);
    EXPECT_FLOAT_EQ(copy_run.windows[0].is_result.total_pnl,
                    orig_run.windows[0].is_result.total_pnl);
    EXPECT_FLOAT_EQ(copy_run.windows[0].oos_lag1.total_pnl, orig_run.windows[0].oos_lag1.total_pnl);
    EXPECT_FLOAT_EQ(copy_run.windows[0].oos_lag2.total_pnl, orig_run.windows[0].oos_lag2.total_pnl);
    EXPECT_FLOAT_EQ(copy_run.windows[0].pnl_decomp.bar_body_pct,
                    orig_run.windows[0].pnl_decomp.bar_body_pct);
    EXPECT_EQ(copy_run.windows[0].pnl_decomp.is_real_edge,
              orig_run.windows[0].pnl_decomp.is_real_edge);

    ASSERT_EQ(copy_run.equity_curve.size(), orig_run.equity_curve.size());
    for (size_t i = 0; i < orig_run.equity_curve.size(); ++i) {
        EXPECT_FLOAT_EQ(copy_run.equity_curve[i], orig_run.equity_curve[i]);
    }

    ASSERT_EQ(copy_run.equity_timestamps.size(), orig_run.equity_timestamps.size());
    for (size_t i = 0; i < orig_run.equity_timestamps.size(); ++i) {
        EXPECT_EQ(copy_run.equity_timestamps[i], orig_run.equity_timestamps[i]);
    }

    EXPECT_FLOAT_EQ(copy_run.oos_aggregate.total_pnl, orig_run.oos_aggregate.total_pnl);
    EXPECT_FLOAT_EQ(copy_run.oos_aggregate.annual_roc, orig_run.oos_aggregate.annual_roc);

    // Cleanup
    std::remove(tmp_path.c_str());
}

// ---------------------------------------------------------------------------
// test_checkpoint_invalid_file
//
// Verify that loading a non-existent or corrupt file returns nullopt
// without crashing.
// ---------------------------------------------------------------------------
TEST(CheckpointTest, InvalidFile) {
    // Non-existent file: load returns nullopt.
    auto cp1 = Checkpoint::load("nonexistent_file.wfck");
    EXPECT_FALSE(cp1.has_value());

    // Write a file with invalid magic bytes
    {
        std::string   bad_path = "test_bad_magic.wfck";
        std::ofstream out(bad_path, std::ios::binary);
        out.write("XXXX", 4);
        out.close();

        EXPECT_FALSE(Checkpoint::is_valid(bad_path));
        auto cp2 = Checkpoint::load(bad_path);
        EXPECT_FALSE(cp2.has_value());

        std::remove(bad_path.c_str());
    }

    // Write a file with wrong version
    {
        std::string   bad_path = "test_bad_version.wfck";
        std::ofstream out(bad_path, std::ios::binary);
        out.write("WFCK", 4);
        uint32_t bad_version = 999;
        out.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
        out.close();

        EXPECT_FALSE(Checkpoint::is_valid(bad_path));
        auto cp3 = Checkpoint::load(bad_path);
        EXPECT_FALSE(cp3.has_value());

        std::remove(bad_path.c_str());
    }
}

// ---------------------------------------------------------------------------
// test_checkpoint_empty
//
// Save and load a checkpoint with no results.
// ---------------------------------------------------------------------------
TEST(CheckpointTest, Empty) {
    Checkpoint cp;
    cp.run_id               = "empty_test";
    cp.phase                = 2;
    cp.current_schedule_idx = 10;
    cp.current_window_idx   = 5;

    std::string tmp_path = "test_checkpoint_empty.wfck";
    ASSERT_TRUE(cp.save(tmp_path));

    auto loaded_opt = Checkpoint::load(tmp_path);
    ASSERT_TRUE(loaded_opt.has_value());
    Checkpoint& loaded = *loaded_opt;
    EXPECT_EQ(loaded.run_id, "empty_test");
    EXPECT_EQ(loaded.phase, 2);
    EXPECT_EQ(loaded.current_schedule_idx, 10);
    EXPECT_EQ(loaded.current_window_idx, 5);
    EXPECT_TRUE(loaded.completed_results.empty());
    EXPECT_TRUE(loaded.completed_runs.empty());

    std::remove(tmp_path.c_str());
}

// ---------------------------------------------------------------------------
// test_checkpoint_default_directory
// ---------------------------------------------------------------------------
TEST(CheckpointTest, DefaultDirectory) {
    std::string dir = Checkpoint::default_directory();
    EXPECT_FALSE(dir.empty());
    // Should contain "checkpoints"
    EXPECT_NE(dir.find("checkpoints"), std::string::npos);
}

// ---------------------------------------------------------------------------
// test_checkpoint_trigger_roundtrip
//
// Verify that WFOConfig trigger fields (trigger, trigger_dd_frac,
// trigger_losers_n) survive a save/load cycle.
// ---------------------------------------------------------------------------
TEST(CheckpointTest, TriggerRoundTrip) {
    Checkpoint cp;
    cp.run_id = "trigger_test";
    cp.phase  = 2;

    // Build a WFO run with an adaptive trigger config
    WFORunResult run;
    run.schedule.frequency_bars   = 20160;  // 1 week
    run.schedule.lookback_bars    = 86400;  // 1 month
    run.schedule.frequency_days   = 7;
    run.schedule.lookback_days    = 30;
    run.schedule.trigger          = WFOTrigger::DD_TRIGGER_5PCT;
    run.schedule.trigger_dd_frac  = 0.05f;
    run.schedule.trigger_losers_n = 0;
    run.selection_strategy        = "best_roc";
    run.parameter_stability       = 0.9f;
    run.windows_skipped           = 0;
    run.sufficient_windows        = true;
    run.oos_aggregate.total_pnl   = 500.0f;
    cp.completed_runs.push_back(run);

    // Second run with losers trigger
    WFORunResult run2;
    run2.schedule.frequency_bars   = 20160;
    run2.schedule.lookback_bars    = 86400;
    run2.schedule.frequency_days   = 7;
    run2.schedule.lookback_days    = 30;
    run2.schedule.trigger          = WFOTrigger::LOSERS_10;
    run2.schedule.trigger_dd_frac  = 0.0f;
    run2.schedule.trigger_losers_n = 10;
    run2.selection_strategy        = "frontier_centroid";
    run2.parameter_stability       = 0.8f;
    run2.windows_skipped           = 0;
    run2.sufficient_windows        = true;
    run2.oos_aggregate.total_pnl   = 300.0f;
    cp.completed_runs.push_back(run2);

    // Save
    std::string tmp_path = "test_checkpoint_trigger_roundtrip.wfck";
    ASSERT_TRUE(cp.save(tmp_path));

    // Load back
    auto loaded_opt = Checkpoint::load(tmp_path);
    ASSERT_TRUE(loaded_opt.has_value());
    Checkpoint& loaded = *loaded_opt;

    ASSERT_EQ(loaded.completed_runs.size(), 2u);

    // First run: DD_TRIGGER_5PCT
    const auto& r1 = loaded.completed_runs[0];
    EXPECT_EQ(r1.schedule.trigger, WFOTrigger::DD_TRIGGER_5PCT);
    EXPECT_FLOAT_EQ(r1.schedule.trigger_dd_frac, 0.05f);
    EXPECT_EQ(r1.schedule.trigger_losers_n, 0);

    // Second run: LOSERS_10
    const auto& r2 = loaded.completed_runs[1];
    EXPECT_EQ(r2.schedule.trigger, WFOTrigger::LOSERS_10);
    EXPECT_FLOAT_EQ(r2.schedule.trigger_dd_frac, 0.0f);
    EXPECT_EQ(r2.schedule.trigger_losers_n, 10);

    // Also verify FIXED_INTERVAL round-trips correctly (default)
    Checkpoint cp3;
    cp3.run_id = "fixed_test";
    cp3.phase  = 2;
    WFORunResult run3;
    run3.schedule.frequency_bars   = 100;
    run3.schedule.lookback_bars    = 500;
    run3.schedule.frequency_days   = 1;
    run3.schedule.lookback_days    = 5;
    run3.schedule.trigger          = WFOTrigger::FIXED_INTERVAL;
    run3.schedule.trigger_dd_frac  = 0.0f;
    run3.schedule.trigger_losers_n = 0;
    run3.selection_strategy        = "best_roc";
    run3.sufficient_windows        = true;
    cp3.completed_runs.push_back(run3);

    std::string tmp_path2 = "test_checkpoint_trigger_fixed.wfck";
    ASSERT_TRUE(cp3.save(tmp_path2));
    auto loaded3_opt = Checkpoint::load(tmp_path2);
    ASSERT_TRUE(loaded3_opt.has_value());
    Checkpoint& loaded3 = *loaded3_opt;
    ASSERT_EQ(loaded3.completed_runs.size(), 1u);
    EXPECT_EQ(loaded3.completed_runs[0].schedule.trigger, WFOTrigger::FIXED_INTERVAL);

    // Cleanup
    std::remove(tmp_path.c_str());
    std::remove(tmp_path2.c_str());
}

// ---------------------------------------------------------------------------
// test_resource_monitor_threads
//
// Verify max_threads() returns hardware_concurrency() / 2.
// ---------------------------------------------------------------------------
TEST(ResourceMonitorTest, MaxThreads) {
    ResourceMonitor monitor;

    int hw       = ResourceMonitor::hardware_concurrency();
    int expected = static_cast<int>(static_cast<float>(hw) * monitor.max_cpu_pct() / 100.0f);
    if (expected < 1) expected = 1;

    EXPECT_EQ(monitor.max_threads(), expected);
    EXPECT_GE(monitor.max_threads(), 1);
}

// ---------------------------------------------------------------------------
// test_resource_monitor_ram
//
// Verify current_ram_gb() returns a reasonable value.
// ---------------------------------------------------------------------------
TEST(ResourceMonitorTest, CurrentRAM) {
    ResourceMonitor monitor;

    float ram = monitor.current_ram_gb();
    // Should be positive (we're a running process, must use SOME RAM)
    EXPECT_GT(ram, 0.0f);
    // Should be less than 128 GB (reasonable upper bound)
    EXPECT_LT(ram, 128.0f);
}

// ---------------------------------------------------------------------------
// test_resource_monitor_hardware_concurrency
//
// Verify hardware_concurrency() returns at least 1.
// ---------------------------------------------------------------------------
TEST(ResourceMonitorTest, HardwareConcurrency) {
    int hw = ResourceMonitor::hardware_concurrency();
    EXPECT_GE(hw, 1);
    // On modern systems, typically 2-128 cores
    EXPECT_LE(hw, 1024);
}

// ---------------------------------------------------------------------------
// test_resource_monitor_within_limits
//
// With generous limits, we should be within bounds.
// ---------------------------------------------------------------------------
TEST(ResourceMonitorTest, WithinLimits) {
    // Very generous limits -- should always pass
    ResourceMonitor generous(128.0f, 100.0f);
    EXPECT_TRUE(generous.within_limits());

    // Zero RAM limit -- should fail (we use some RAM)
    ResourceMonitor strict(0.0f, 100.0f);
    EXPECT_FALSE(strict.within_limits());
}

// ---------------------------------------------------------------------------
// test_resource_monitor_custom_limits
//
// Verify custom limits are stored correctly.
// ---------------------------------------------------------------------------
TEST(ResourceMonitorTest, CustomLimits) {
    ResourceMonitor monitor(16.0f, 25.0f);
    EXPECT_FLOAT_EQ(monitor.max_ram_gb(), 16.0f);
    EXPECT_FLOAT_EQ(monitor.max_cpu_pct(), 25.0f);
}

// ---------------------------------------------------------------------------
// test_resource_monitor_default_limits
//
// The default constructor picks up the process-wide limits, and
// set_default_limits() changes them. The thread count follows the CPU share.
// ---------------------------------------------------------------------------
TEST(ResourceMonitorTest, DefaultLimits) {
    const ResourceMonitor::Limits saved = ResourceMonitor::default_limits();

    ResourceMonitor monitor;
    EXPECT_FLOAT_EQ(monitor.max_ram_gb(), saved.max_ram_gb);
    EXPECT_FLOAT_EQ(monitor.max_cpu_pct(), saved.max_cpu_pct);
    EXPECT_GT(monitor.max_ram_gb(), 0.0f);
    EXPECT_GE(monitor.max_cpu_pct(), 1.0f);
    EXPECT_LE(monitor.max_cpu_pct(), 100.0f);

    ResourceMonitor::set_default_limits({8.0f, 25.0f});
    ResourceMonitor limited;
    ResourceMonitor::set_default_limits(saved);

    // Environment overrides (WFO_GPU_MAX_*) take precedence when present.
    if (std::getenv("WFO_GPU_MAX_RAM_GB") == nullptr) EXPECT_FLOAT_EQ(limited.max_ram_gb(), 8.0f);
    if (std::getenv("WFO_GPU_MAX_CPU_PCT") == nullptr) {
        EXPECT_FLOAT_EQ(limited.max_cpu_pct(), 25.0f);
        const int hw = ResourceMonitor::hardware_concurrency();
        EXPECT_EQ(limited.max_threads(), std::max(1, hw / 4));
    }
}

TEST(ResourceMonitorTest, PhysicalRam) {
    EXPECT_GT(ResourceMonitor::physical_ram_gb(), 0.0f);
}
