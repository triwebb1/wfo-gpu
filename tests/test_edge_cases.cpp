// =============================================================================
// Edge-Case Tests for Metrics, Checkpoints, Selection, and Grids
//
// Boundary and degenerate-input coverage:
//   - NaN trade PnL rejection in compute_metrics
//   - Inf trade PnL rejection in compute_metrics
//   - Zero-bar range guard in compute_metrics
//   - Checkpoint save/load round-trip correctness
//   - Pareto frontier edge cases (single entry, all-identical, empty)
//   - Selection on single-entry frontier
//   - Drawdown pause edge cases (no trades, immediate trigger, threshold=0)
//   - Robustness with zero center metric
//   - PnL decomposition edge cases
//   - Strategy grid boundary (first and last index)
//
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>
#include "test_helpers.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#include "backtest_engine.h"
#include "checkpoint.h"
#include "cost_model.h"
#include "grids/strategy_grid.h"
#include "landscape.h"
#include "market_data.h"
#include "metrics.h"
#include "pareto.h"
#include "pnl_decomposition.h"
#include "robustness.h"
#include "selection.h"

// =============================================================================
// Helpers
// =============================================================================

// Create a simple trade at given bar indices with specified PnL.
static Trade make_trade(int entry_bar, int exit_bar, float net_pnl, int dir = 1,
                        float entry_price = 4500.0f) {
    Trade t;
    t.entry_bar      = entry_bar;
    t.exit_bar       = exit_bar;
    t.entry_price    = entry_price;
    t.exit_price     = entry_price + (net_pnl > 0 ? 1.0f : -1.0f);
    t.direction      = dir;
    t.gross_pnl      = net_pnl + 75.0f;  // approximate: add round-trip cost back
    t.net_pnl        = net_pnl;
    t.min_unrealized = (net_pnl < 0) ? net_pnl : 0.0f;
    t.max_unrealized = (net_pnl > 0) ? net_pnl : 0.0f;
    return t;
}

// =============================================================================
// NaN / Inf Guard in compute_metrics
// =============================================================================

TEST(MetricsEdge, RejectsNaNTradePnl) {
    auto               bars = make_daily_bars(5);
    std::vector<Trade> trades;
    trades.push_back(make_trade(0, 1, 100.0f));
    trades.push_back(make_trade(2, 3, std::numeric_limits<float>::quiet_NaN()));

    BacktestResult result = compute_metrics(trades, bars.data(), 0, 5, es_cost());

    // Should return default (zero) metrics, not propagate NaN
    EXPECT_EQ(result.total_trades, 0)
        << "NaN trade PnL should cause early return with zero metrics";
    EXPECT_FALSE(std::isnan(result.sharpe_ratio)) << "Sharpe must never be NaN";
    EXPECT_FALSE(std::isnan(result.profit_factor)) << "Profit factor must never be NaN";
}

TEST(MetricsEdge, RejectsInfTradePnl) {
    auto               bars = make_daily_bars(5);
    std::vector<Trade> trades;
    trades.push_back(make_trade(0, 1, 100.0f));
    trades.push_back(make_trade(2, 3, std::numeric_limits<float>::infinity()));

    BacktestResult result = compute_metrics(trades, bars.data(), 0, 5, es_cost());

    EXPECT_EQ(result.total_trades, 0)
        << "Inf trade PnL should cause early return with zero metrics";
}

TEST(MetricsEdge, ZeroBarRange) {
    auto               bars = make_daily_bars(5);
    std::vector<Trade> trades;
    trades.push_back(make_trade(0, 1, 100.0f));

    // bar_end == bar_start => zero range => should return default result
    BacktestResult result = compute_metrics(trades, bars.data(), 3, 3, es_cost());
    EXPECT_EQ(result.total_trades, 0);
}

// =============================================================================
// Checkpoint Round-Trip
// =============================================================================

TEST(CheckpointTest, SaveLoadRoundTrip) {
    Checkpoint cp;
    cp.run_id               = "test_run_001";
    cp.phase                = 1;
    cp.strategies_evaluated = 42;
    cp.total_strategies     = 100;

    // Add some completed results
    BacktestResult r1;
    r1.total_trades  = 50;
    r1.annual_roc    = 25.5f;
    r1.sharpe_ratio  = 1.8f;
    r1.profit_factor = 2.3f;
    r1.win_rate      = 0.65f;
    r1.viable        = true;
    r1.strategy_id   = 7;
    cp.completed_results.push_back(r1);

    BacktestResult r2;
    r2.total_trades = 30;
    r2.annual_roc   = -5.0f;
    r2.viable       = false;
    r2.strategy_id  = 12;
    cp.completed_results.push_back(r2);

    // Save
    std::string path = "test_checkpoint_roundtrip.wfo";
    ASSERT_TRUE(cp.save(path));

    // Load (returns std::optional)
    auto loaded_opt = Checkpoint::load(path);
    ASSERT_TRUE(loaded_opt.has_value());
    Checkpoint& loaded = *loaded_opt;

    // Verify all fields round-trip correctly
    EXPECT_EQ(loaded.run_id, "test_run_001");
    EXPECT_EQ(loaded.phase, 1);
    EXPECT_EQ(loaded.strategies_evaluated, 42);
    EXPECT_EQ(loaded.total_strategies, 100);
    ASSERT_EQ(loaded.completed_results.size(), 2u);

    EXPECT_EQ(loaded.completed_results[0].total_trades, 50);
    EXPECT_FLOAT_EQ(loaded.completed_results[0].annual_roc, 25.5f);
    EXPECT_FLOAT_EQ(loaded.completed_results[0].sharpe_ratio, 1.8f);
    EXPECT_FLOAT_EQ(loaded.completed_results[0].profit_factor, 2.3f);
    EXPECT_FLOAT_EQ(loaded.completed_results[0].win_rate, 0.65f);
    EXPECT_TRUE(loaded.completed_results[0].viable);
    EXPECT_EQ(loaded.completed_results[0].strategy_id, 7);

    EXPECT_EQ(loaded.completed_results[1].total_trades, 30);
    EXPECT_FLOAT_EQ(loaded.completed_results[1].annual_roc, -5.0f);
    EXPECT_FALSE(loaded.completed_results[1].viable);

    // Clean up
    std::remove(path.c_str());
}

TEST(CheckpointTest, IsValidRejectsGarbage) {
    std::string path = "test_checkpoint_garbage.wfo";
    {
        std::ofstream out(path, std::ios::binary);
        out << "this is not a valid checkpoint file";
    }

    EXPECT_FALSE(Checkpoint::is_valid(path));

    std::remove(path.c_str());
}

TEST(CheckpointTest, LoadRejectsCorrupt) {
    std::string path = "test_checkpoint_corrupt.wfo";
    {
        std::ofstream out(path, std::ios::binary);
        out.write("WFCK", 4);         // valid magic (WFCK, checkpoint.cpp CHECKPOINT_MAGIC)
        uint32_t bad_version = 9999;  // out-of-band version; load must reject
        out.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
    }

    // A corrupt or unsupported-version file loads as nullopt.
    auto loaded_opt = Checkpoint::load(path);
    EXPECT_FALSE(loaded_opt.has_value());

    std::remove(path.c_str());
}

// Verify that flipping a byte mid-file causes CRC-based rejection.
TEST(CheckpointTest, LoadRejectsBitFlip) {
    Checkpoint cp;
    cp.run_id               = "crc_test";
    cp.phase                = 1;
    cp.strategies_evaluated = 10;
    cp.total_strategies     = 100;

    BacktestResult r;
    r.total_trades = 25;
    r.annual_roc   = 12.0f;
    r.strategy_id  = 3;
    r.viable       = true;
    cp.completed_results.push_back(r);

    std::string path = "test_checkpoint_bitflip.wfo";
    ASSERT_TRUE(cp.save(path));

    // Verify the file loads successfully before corruption
    {
        auto ok = Checkpoint::load(path);
        ASSERT_TRUE(ok.has_value()) << "File must load before corruption";
    }

    // Corrupt one byte in the middle of the file
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(f.is_open());
        f.seekp(60);  // skip header, land in body
        char byte;
        f.read(&byte, 1);
        byte ^= 0xFF;  // flip all bits
        f.seekp(60);
        f.write(&byte, 1);
    }

    // The corrupted file must be rejected (CRC mismatch)
    auto loaded_opt = Checkpoint::load(path);
    EXPECT_FALSE(loaded_opt.has_value()) << "Bit-flipped checkpoint must be rejected by CRC check";

    std::remove(path.c_str());
}

// Verify that a config hash mismatch causes rejection.
TEST(CheckpointTest, LoadRejectsConfigHashMismatch) {
    Checkpoint cp;
    cp.run_id        = "hash_test";
    cp.phase         = 1;
    cp.config_hash   = 12345;
    cp.bar_data_hash = 67890;

    std::string path = "test_checkpoint_hashmismatch.wfo";
    ASSERT_TRUE(cp.save(path));

    // Load with matching hashes -> should succeed
    {
        auto ok = Checkpoint::load(path, 12345, 67890);
        EXPECT_TRUE(ok.has_value());
    }

    // Load with wrong config hash -> should fail
    {
        auto bad = Checkpoint::load(path, 99999, 67890);
        EXPECT_FALSE(bad.has_value()) << "Config hash mismatch must be rejected";
    }

    // Load with wrong bar data hash -> should fail
    {
        auto bad = Checkpoint::load(path, 12345, 99999);
        EXPECT_FALSE(bad.has_value()) << "Bar data hash mismatch must be rejected";
    }

    std::remove(path.c_str());
}

// =============================================================================
// Pareto Frontier Edge Cases
// =============================================================================

TEST(Pareto, EmptyInput) {
    std::vector<BacktestResult> results;
    auto                        frontier = extract_pareto_frontier(results);
    EXPECT_TRUE(frontier.empty());
}

TEST(Pareto, SingleEntry) {
    std::vector<BacktestResult> results(1);
    results[0].annual_roc           = 10.0f;
    results[0].green_weeks_pct      = 60.0f;
    results[0].max_red_weeks        = 2;
    results[0].trade_recovery_ratio = 1.5f;
    results[0].profit_factor        = 1.8f;

    auto frontier = extract_pareto_frontier(results);
    ASSERT_EQ(frontier.size(), 1u);
    EXPECT_EQ(frontier[0], 0);
}

TEST(Pareto, AllIdentical) {
    // When all entries have identical metrics, all are non-dominated
    std::vector<BacktestResult> results(5);
    for (auto& r : results) {
        r.annual_roc           = 10.0f;
        r.green_weeks_pct      = 60.0f;
        r.max_red_weeks        = 2;
        r.trade_recovery_ratio = 1.5f;
        r.profit_factor        = 1.8f;
    }

    auto frontier = extract_pareto_frontier(results);
    // All identical => none dominates another => all on frontier
    EXPECT_EQ(frontier.size(), 5u);
}

TEST(Pareto, ClearDominance) {
    // Entry 0 dominates entry 1 on all dimensions
    std::vector<BacktestResult> results(2);

    results[0].annual_roc           = 20.0f;
    results[0].green_weeks_pct      = 70.0f;
    results[0].max_red_weeks        = 1;
    results[0].trade_recovery_ratio = 2.0f;
    results[0].profit_factor        = 3.0f;

    results[1].annual_roc           = 10.0f;
    results[1].green_weeks_pct      = 50.0f;
    results[1].max_red_weeks        = 3;
    results[1].trade_recovery_ratio = 1.0f;
    results[1].profit_factor        = 1.5f;

    auto frontier = extract_pareto_frontier(results);
    ASSERT_EQ(frontier.size(), 1u);
    EXPECT_EQ(frontier[0], 0) << "Entry 0 should dominate entry 1";
}

TEST(Pareto, TradeOff) {
    // Entry 0: better ROC, worse consistency
    // Entry 1: worse ROC, better consistency
    // Neither dominates the other
    std::vector<BacktestResult> results(2);

    results[0].annual_roc           = 30.0f;
    results[0].green_weeks_pct      = 40.0f;
    results[0].max_red_weeks        = 5;
    results[0].trade_recovery_ratio = 1.0f;
    results[0].profit_factor        = 1.2f;

    results[1].annual_roc           = 10.0f;
    results[1].green_weeks_pct      = 80.0f;
    results[1].max_red_weeks        = 1;
    results[1].trade_recovery_ratio = 3.0f;
    results[1].profit_factor        = 4.0f;

    auto frontier = extract_pareto_frontier(results);
    EXPECT_EQ(frontier.size(), 2u) << "Neither dominates; both on frontier";
}

// =============================================================================
// Selection Edge Cases
// =============================================================================

TEST(Selection, EmptyFrontier) {
    std::vector<BacktestResult> results;
    std::vector<ParetoEntry>    frontier;

    EXPECT_EQ(select_winner(SelectionStrategy::BEST_ROC, results, frontier), -1);
    EXPECT_EQ(select_winner(SelectionStrategy::FRONTIER_CENTROID, results, frontier), -1);
}

TEST(Selection, SingleEntry) {
    std::vector<BacktestResult> results(1);
    results[0].annual_roc      = 15.0f;
    results[0].green_weeks_pct = 65.0f;

    std::vector<ParetoEntry> frontier(1);
    frontier[0].original_index = 0;
    frontier[0].archetype      = Archetype::BALANCED;

    // All selection strategies should return index 0
    EXPECT_EQ(select_winner(SelectionStrategy::BEST_ROC, results, frontier), 0);
    EXPECT_EQ(select_winner(SelectionStrategy::BEST_CONSISTENCY, results, frontier), 0);
    EXPECT_EQ(select_winner(SelectionStrategy::FRONTIER_CENTROID, results, frontier), 0);
}

// =============================================================================
// Robustness Edge Cases
// =============================================================================

TEST(Robustness, ZeroCenterMetric) {
    float                       center = 0.0f;
    std::vector<NeighborResult> neighbors;
    NeighborResult              nb;
    nb.param_axis       = 0;
    nb.metric_value     = 50.0f;
    nb.metrics_complete = true;
    neighbors.push_back(nb);

    float score = compute_robustness(center, neighbors, 1);
    EXPECT_FLOAT_EQ(score, 0.0f) << "Zero center metric should return 0 robustness";
}

TEST(Robustness, NoNeighborsScoresZero) {
    float                       center = 50.0f;
    std::vector<NeighborResult> neighbors;

    float score = compute_robustness(center, neighbors, 3);
    EXPECT_FLOAT_EQ(score, 0.0f)
        << "No neighbors should return 0 robustness (cannot assess stability)";
}

TEST(Robustness, IdenticalNeighbors) {
    float                       center = 60.0f;
    std::vector<NeighborResult> neighbors;

    for (int axis = 0; axis < 3; ++axis) {
        NeighborResult nb;
        nb.param_axis       = axis;
        nb.metric_value     = 60.0f;  // identical to center
        nb.metrics_complete = true;
        neighbors.push_back(nb);
    }

    float score = compute_robustness(center, neighbors, 3);
    EXPECT_FLOAT_EQ(score, 1.0f) << "Identical neighbors should yield perfect robustness";
}

TEST(Robustness, OneAxisDegraded) {
    float                       center = 60.0f;
    std::vector<NeighborResult> neighbors;

    // Axis 0: neighbor retains 50% of center
    NeighborResult nb0;
    nb0.param_axis       = 0;
    nb0.metric_value     = 30.0f;
    nb0.metrics_complete = true;
    neighbors.push_back(nb0);

    // Axis 1: neighbor retains 100% of center
    NeighborResult nb1;
    nb1.param_axis       = 1;
    nb1.metric_value     = 60.0f;
    nb1.metrics_complete = true;
    neighbors.push_back(nb1);

    float score = compute_robustness(center, neighbors, 2);
    // retention on axis 0 = 30/60 = 0.5, sensitivity = 0.5
    // retention on axis 1 = 60/60 = 1.0, sensitivity = 0.0
    // max_sensitivity = 0.5, robustness = 0.5
    EXPECT_NEAR(score, 0.5f, 1e-6f);
}

// =============================================================================
// PnL Decomposition Edge Cases
// =============================================================================

TEST(PnlDecomposition, EmptyTrades) {
    std::vector<Trade> trades;
    auto               bars   = make_daily_bars(5);
    PnLDecomposition   result = decompose_pnl(trades, bars.data(), es_cost());

    EXPECT_FLOAT_EQ(result.total_bar_body_pnl, 0.0f);
    EXPECT_FLOAT_EQ(result.total_gap_pnl, 0.0f);
    EXPECT_FLOAT_EQ(result.total_gross_pnl, 0.0f);
    EXPECT_TRUE(result.per_trade.empty());
}

TEST(PnlDecomposition, FillBarZero) {
    // Trade with entry_bar = 0 -> gap component undefined (no previous bar)
    auto               bars = make_daily_bars(5);
    std::vector<Trade> trades;
    Trade              t;
    t.entry_bar   = 0;
    t.exit_bar    = 1;
    t.entry_price = bars[0].open;
    t.exit_price  = bars[1].close;
    t.direction   = 1;
    t.gross_pnl   = es_cost().price_to_pnl(bars[1].close - bars[0].open);
    t.net_pnl     = t.gross_pnl - es_cost().round_trip_cost();
    trades.push_back(t);

    PnLDecomposition result = decompose_pnl(trades, bars.data(), es_cost());
    ASSERT_EQ(result.per_trade.size(), 1u);
    EXPECT_FLOAT_EQ(result.per_trade[0].gap_pnl, 0.0f)
        << "Gap PnL should be 0 when entry_bar is 0 (no prior bar)";
}

// =============================================================================
// Strategy Grid Boundary Tests
// =============================================================================

TEST(StrategyGrid, FirstAndLastEntry) {
    // With 5 indicators, test that index 0 and index total_count-1 both
    // produce valid entries without crashing.
    StrategyGrid grid(5);
    int64_t      total = grid.total_count();
    ASSERT_GT(total, 0);

    // First entry
    StrategyGridEntry first = grid.entry_at(0);
    EXPECT_GE(first.num_indicators, 1);
    EXPECT_LE(first.num_indicators, 4);

    // Last entry
    StrategyGridEntry last = grid.entry_at(total - 1);
    EXPECT_GE(last.num_indicators, 1);
    EXPECT_LE(last.num_indicators, 4);
}

TEST(StrategyGrid, EntryToConfig) {
    // Verify that every entry_at() produces a valid StrategyConfig
    StrategyGrid grid(3);
    int64_t      total = grid.total_count();

    // Check a sample of entries including boundaries
    std::vector<int64_t> indices = {0, 1, total / 4, total / 2, total - 2, total - 1};
    for (int64_t idx : indices) {
        StrategyGridEntry entry  = grid.entry_at(idx);
        StrategyConfig    config = entry.to_strategy_config();

        EXPECT_GE(config.num_indicators, 1) << "index=" << idx;
        EXPECT_LE(config.num_indicators, 4) << "index=" << idx;
        EXPECT_GE(config.entry_threshold, 0.0f) << "index=" << idx;
        EXPECT_LE(config.entry_threshold, 1.0f) << "index=" << idx;

        // All indicator indices should be valid
        for (int i = 0; i < config.num_indicators; ++i) {
            EXPECT_GE(config.indicator_indices[i], 0) << "index=" << idx;
            EXPECT_LT(config.indicator_indices[i], 3) << "index=" << idx;
        }
    }
}

TEST(StrategyGrid, RedundantSkip) {
    // Verify that is_redundant() correctly identifies and rejects duplicates
    StrategyGrid grid(3);

    // UNANIMOUS with 1 indicator should be redundant
    StrategyGridEntry e;
    e.num_indicators   = 1;
    e.indicator_ids[0] = 0;
    e.aggregation      = StrategyConfig::UNANIMOUS;
    e.weights[0]       = 1.0f;
    EXPECT_TRUE(e.is_redundant()) << "UNANIMOUS with 1 indicator should be redundant";

    // WEIGHTED_SUM with 1 indicator should NOT be redundant
    e.aggregation = StrategyConfig::WEIGHTED_SUM;
    EXPECT_FALSE(e.is_redundant());

    // Weighted sum with a single indicator: the weight cancels, so only
    // weight 1.0 is kept.
    e.weights[0] = 0.5f;
    EXPECT_TRUE(e.is_redundant());

    // MAJORITY_VOTE ignores weights: only the all-1.0 combination is kept.
    StrategyGridEntry e2;
    e2.num_indicators   = 2;
    e2.indicator_ids[0] = 0;
    e2.indicator_ids[1] = 1;
    e2.aggregation      = StrategyConfig::MAJORITY_VOTE;
    e2.weights[0]       = 0.5f;
    e2.weights[1]       = 0.5f;
    EXPECT_TRUE(e2.is_redundant()) << "MAJORITY_VOTE ignores weights";
    e2.weights[0] = 0.25f;
    e2.weights[1] = 0.75f;
    EXPECT_TRUE(e2.is_redundant()) << "MAJORITY_VOTE ignores weights";
    e2.weights[0] = 1.0f;
    e2.weights[1] = 1.0f;
    EXPECT_FALSE(e2.is_redundant());

    // Weighted sum: proportional weights are identical; the largest member
    // of each class is kept.
    StrategyGridEntry e3 = e2;
    e3.aggregation       = StrategyConfig::WEIGHTED_SUM;
    e3.weights[0]        = 0.25f;  // (1, 2) ~ (2, 4)
    e3.weights[1]        = 0.5f;
    EXPECT_TRUE(e3.is_redundant());
    e3.weights[0] = 0.5f;
    e3.weights[1] = 1.0f;
    EXPECT_FALSE(e3.is_redundant());
    e3.weights[0] = 0.75f;  // (3, 1) has no larger multiple in the grid
    e3.weights[1] = 0.25f;
    EXPECT_FALSE(e3.is_redundant());
}

// =============================================================================
// Archetype Thresholds Edge Case
// =============================================================================

TEST(Archetype, ThresholdsSingleEntry) {
    // Single-entry pool should not crash
    std::vector<BacktestResult> pool(1);
    pool[0].win_rate        = 0.7f;
    pool[0].green_weeks_pct = 60.0f;
    pool[0].annual_roc      = 15.0f;

    ArchetypeThresholds t = compute_archetype_thresholds(pool);

    // Grinder min is max(0.90, pool WR P75). With 1 entry, P75 = 0.7.
    // max(0.90, 0.7) = 0.90
    EXPECT_FLOAT_EQ(t.grinder_wr_min, 0.90f);
}

TEST(Archetype, ThresholdsEmptyPool) {
    std::vector<BacktestResult> pool;
    ArchetypeThresholds         t = compute_archetype_thresholds(pool);

    // Empty pool returns default-constructed ArchetypeThresholds
    // Default grinder_wr_min = 0.90f (from struct initializer)
    EXPECT_FLOAT_EQ(t.grinder_wr_min, 0.90f);
    EXPECT_FLOAT_EQ(t.grinder_gw_min, 75.0f);
}

// =============================================================================
// Metrics: avg_trade_dd and avg_trade_peak
// =============================================================================

TEST(MetricsEdge, AvgTradeDdPeakAndLiveDrawdown) {
    // Verify avg_trade_dd and avg_trade_peak are computed correctly
    auto               bars = make_daily_bars(10);
    std::vector<Trade> trades;

    Trade t1          = make_trade(0, 2, 200.0f);
    t1.min_unrealized = -50.0f;  // worst unrealized during trade
    t1.max_unrealized = 300.0f;  // best unrealized during trade
    trades.push_back(t1);

    Trade t2          = make_trade(4, 6, -100.0f);
    t2.min_unrealized = -200.0f;
    t2.max_unrealized = 50.0f;
    trades.push_back(t2);

    BacktestResult result = compute_metrics(trades, bars.data(), 0, 10, es_cost());

    // avg_trade_dd = -(sum_min_unrealized / total_trades) = -(-50 + -200) / 2 = 125
    EXPECT_NEAR(result.avg_trade_dd, 125.0f, 0.01f);

    // avg_trade_peak = sum_max_unrealized / total_trades = (300 + 50) / 2 = 175
    EXPECT_NEAR(result.avg_trade_peak, 175.0f, 0.01f);

    // max_live_drawdown = -worst_min_unrealized = 200
    EXPECT_NEAR(result.max_live_drawdown, 200.0f, 0.01f);
}

// =============================================================================
// Entry Lag Verification
// =============================================================================

TEST(BacktestEngine, EntryLagFillsAtCorrectBar) {
    // Signal on bar 2 with entry_lag=1 should fill at bar 3's open.
    // Signal on bar 2 with entry_lag=2 should fill at bar 4's open.
    std::vector<Bar> bars;
    for (int i = 0; i < 10; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 30LL) * NS_PER_SEC;
        // Prices: 100, 101, 102, ..., 109
        float p = 100.0f + static_cast<float>(i);
        bars.push_back(make_bar(ts, p, p + 0.5f, p - 0.5f, p));
    }

    // One indicator series: strong buy signal (+1) on bars 2-9, 0 elsewhere
    std::vector<float> signal(10, 0.0f);
    signal[2]            = 1.0f;  // Signal fires on bar 2
    const float* sig_ptr = signal.data();

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 3;

    CostModel cost = es_cost();

    // Lag-1: signal on bar 2 -> fill at bar 3 open (price 103)
    {
        BacktestEngine engine;
        engine.run(bars.data(), 0, 10, &sig_ptr, 1, config, cost, 1);
        const auto& trades = engine.last_trades();
        ASSERT_GE(trades.size(), 1u);
        EXPECT_EQ(trades[0].entry_bar, 3) << "Lag-1: signal at bar 2 should fill at bar 3";
        EXPECT_FLOAT_EQ(trades[0].entry_price, 103.0f) << "Entry price should be bar 3's open";
    }

    // Lag-2: signal on bar 2 -> fill at bar 4 open (price 104)
    {
        BacktestEngine engine;
        engine.run(bars.data(), 0, 10, &sig_ptr, 1, config, cost, 2);
        const auto& trades = engine.last_trades();
        ASSERT_GE(trades.size(), 1u);
        EXPECT_EQ(trades[0].entry_bar, 4) << "Lag-2: signal at bar 2 should fill at bar 4";
        EXPECT_FLOAT_EQ(trades[0].entry_price, 104.0f) << "Entry price should be bar 4's open";
    }
}

// =============================================================================
// Selection Bounds Safety
// =============================================================================

TEST(Selection, SkipsInvalidOriginalIndex) {
    // Verify selection functions don't crash on out-of-bounds original_index
    std::vector<BacktestResult> results(2);
    results[0].annual_roc      = 10.0f;
    results[0].green_weeks_pct = 60.0f;
    results[1].annual_roc      = 20.0f;
    results[1].green_weeks_pct = 70.0f;

    // Frontier with one valid index and one out-of-bounds index
    std::vector<ParetoEntry> frontier(2);
    frontier[0].original_index = 0;
    frontier[0].archetype      = Archetype::BALANCED;
    frontier[1].original_index = 999;  // out of bounds
    frontier[1].archetype      = Archetype::BALANCED;

    // Should skip the invalid entry and return the valid one
    int winner = select_winner(SelectionStrategy::BEST_ROC, results, frontier);
    EXPECT_EQ(winner, 0) << "Should select valid entry, skip out-of-bounds";

    // Centroid should also survive
    int centroid = select_winner(SelectionStrategy::FRONTIER_CENTROID, results, frontier);
    EXPECT_GE(centroid, 0) << "Centroid should produce a valid result";
}

// =============================================================================
// PnL Decomposition: zero gross PnL (division edge case)
// =============================================================================

TEST(PnlDecomposition, ZeroGrossPnl) {
    // Trades that net to exactly zero gross PnL
    auto               bars = make_daily_bars(5);
    std::vector<Trade> trades;
    Trade              t;
    t.entry_bar   = 1;
    t.exit_bar    = 2;
    t.entry_price = bars[1].open;
    t.exit_price  = bars[1].open;  // zero price change
    t.direction   = 1;
    t.gross_pnl   = 0.0f;
    t.net_pnl     = -es_cost().round_trip_cost();
    trades.push_back(t);

    PnLDecomposition result = decompose_pnl(trades, bars.data(), es_cost());

    // abs_total == 0 -> percentages stay at default 0
    EXPECT_FLOAT_EQ(result.bar_body_pct, 0.0f);
    EXPECT_FLOAT_EQ(result.gap_pct, 0.0f);
    EXPECT_FALSE(result.is_real_edge);
    EXPECT_FALSE(result.is_gap_artifact);
}

// =============================================================================
// ExitModePreset lookup table: verify all 24 presets produce valid configs
// =============================================================================

TEST(StrategyGrid, AllExitPresetsMapCorrectly) {
    StrategyGridEntry e;
    e.num_indicators   = 1;
    e.indicator_ids[0] = 0;
    e.weights[0]       = 1.0f;
    e.aggregation      = StrategyConfig::WEIGHTED_SUM;
    e.entry_threshold  = 0.5f;
    e.trading_hours    = StrategyConfig::ETH;

    // Iterate all 24 ExitModePreset values (0..23)
    for (int em = 0; em <= 23; ++em) {
        e.exit_mode       = static_cast<ExitModePreset>(em);
        StrategyConfig sc = e.to_strategy_config();

        // Every preset should produce a valid exit mode
        EXPECT_GE(static_cast<int>(sc.exit_mode), 0) << "preset=" << em;
        EXPECT_LE(static_cast<int>(sc.exit_mode), 6) << "preset=" << em;

        // Trailing stop presets should have positive mult
        if (sc.exit_mode == StrategyConfig::TRAILING_STOP) {
            EXPECT_GT(sc.trailing_atr_mult, 0.0f) << "preset=" << em;
        }
        // TP/SL presets should have positive multipliers
        if (sc.exit_mode == StrategyConfig::TP_SL_ATR) {
            EXPECT_GT(sc.tp_atr_mult, 0.0f) << "preset=" << em;
            EXPECT_GT(sc.sl_atr_mult, 0.0f) << "preset=" << em;
        }
    }
}
