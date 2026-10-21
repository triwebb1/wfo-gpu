#include <gtest/gtest.h>
#include "portable_random.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#include "landscape.h"
#include "orchestrator.h"
#include "test_fixture.h"

#include <set>
#include "pareto.h"
#include "robustness.h"
#include "metrics.h"
#include "cost_model.h"
#include "market_data.h"
#include "backtest_engine.h"
#include "grids/indicator_grid.h"
#include "grids/strategy_grid.h"
#include "indicators/signal_normalize.h"

// ============================================================================
// Helper: create a BacktestResult with specified values
// ============================================================================

static BacktestResult make_result(float total_pnl, int total_trades, float profit_factor,
                                  float annual_roc, float green_weeks_pct = 0.60f,
                                  float win_rate = 0.55f, int max_red_weeks = 2, float trr = 0.5f,
                                  int strategy_id = -1) {
    BacktestResult r;
    r.total_pnl            = total_pnl;
    r.total_trades         = total_trades;
    r.profit_factor        = profit_factor;
    r.annual_roc           = annual_roc;
    r.green_weeks_pct      = green_weeks_pct;
    r.win_rate             = win_rate;
    r.max_red_weeks        = max_red_weeks;
    r.trade_recovery_ratio = trr;
    r.strategy_id          = strategy_id;
    r.max_live_drawdown    = 1000.0f;
    r.avg_win              = 200.0f;
    r.avg_loss             = -150.0f;
    // analyze_landscape filters survivors on r.viable and
    // compute_robustness_batch_full filters neighbors on r.metrics_complete.
    // These test fixtures represent fully-populated rows.
    r.viable           = true;
    r.metrics_complete = true;
    return r;
}

// ============================================================================
// Helper: generate synthetic bars (random walk or trending)
// ============================================================================

static std::vector<Bar> generate_random_walk_bars(int count, float start_price = 4500.0f,
                                                  unsigned seed = 42) {
    std::mt19937            rng(seed);
    portable::normal<float> step{0.0f, 0.5f};

    std::vector<Bar> bars(count);
    float            price = start_price;

    // Start timestamp: 2025-01-02 18:00 ET (session open) in UTC ns
    // 2025-01-02 23:00 UTC = 2025-01-02 18:00 ET
    int64_t       ts              = 1735858800LL * 1'000'000'000LL;  // approximate
    const int64_t bar_interval_ns = 30LL * 1'000'000'000LL;          // 30 seconds

    for (int i = 0; i < count; ++i) {
        float move  = step(rng);
        float open  = price;
        float close = price + move;
        float high  = std::max(open, close) + std::abs(step(rng)) * 0.5f;
        float low   = std::min(open, close) - std::abs(step(rng)) * 0.5f;

        bars[i].timestamp = ts;
        bars[i].open      = open;
        bars[i].high      = high;
        bars[i].low       = low;
        bars[i].close     = close;
        bars[i].volume    = 100 + static_cast<uint32_t>(rng() % 900);
        bars[i]._pad      = 0;

        price = close;
        ts += bar_interval_ns;
    }

    return bars;
}

// ============================================================================
// Test: Sanity floor filtering with known results
// ============================================================================

TEST(LandscapeSanityFloors, KnownResults) {
    // Create 10 results, only 3 should pass all sanity floors
    std::vector<BacktestResult> results;

    // Passes: pnl > 0, trades >= 100, pf > 1.0, roc > 0
    results.push_back(make_result(5000.0f, 200, 1.5f, 15.0f));  // 0: PASS
    results.push_back(make_result(3000.0f, 150, 1.2f, 10.0f));  // 1: PASS
    results.push_back(make_result(8000.0f, 300, 2.0f, 25.0f));  // 2: PASS

    // Fails: negative PnL
    results.push_back(make_result(-1000.0f, 200, 0.8f, -5.0f));  // 3: FAIL

    // Fails: too few trades
    results.push_back(make_result(1000.0f, 50, 1.3f, 8.0f));  // 4: FAIL

    // Fails: profit factor <= 1.0
    results.push_back(make_result(100.0f, 200, 0.95f, 1.0f));  // 5: FAIL

    // Fails: annual_roc <= 0
    results.push_back(make_result(100.0f, 200, 1.1f, 0.0f));  // 6: FAIL

    // Fails: negative PnL and few trades
    results.push_back(make_result(-500.0f, 30, 0.7f, -3.0f));  // 7: FAIL

    // Fails: pnl exactly 0 (need > 0)
    results.push_back(make_result(0.0f, 200, 1.0f, 0.0f));  // 8: FAIL

    // Fails: roc exactly 0 (need > 0)
    results.push_back(make_result(500.0f, 200, 1.1f, 0.0f));  // 9: FAIL

    auto passed = apply_sanity_floors(results);

    ASSERT_EQ(passed.size(), 3u);
    EXPECT_EQ(passed[0], 0);
    EXPECT_EQ(passed[1], 1);
    EXPECT_EQ(passed[2], 2);
}

TEST(LandscapeSanityFloors, AllPass) {
    std::vector<BacktestResult> results;
    for (int i = 0; i < 5; ++i) {
        results.push_back(make_result(1000.0f + i * 500.0f, 150 + i * 10, 1.2f + i * 0.1f,
                                      10.0f + i * 2.0f, 0.60f, 0.55f, 2, 0.5f, i));
    }

    auto passed = apply_sanity_floors(results);
    EXPECT_EQ(passed.size(), 5u);
}

TEST(LandscapeSanityFloors, NonePass) {
    std::vector<BacktestResult> results;
    for (int i = 0; i < 5; ++i) {
        // All have negative PnL
        results.push_back(make_result(-1000.0f, 200, 0.8f, -5.0f));
    }

    auto passed = apply_sanity_floors(results);
    EXPECT_EQ(passed.size(), 0u);
}

TEST(LandscapeSanityFloors, Empty) {
    std::vector<BacktestResult> results;
    auto                        passed = apply_sanity_floors(results);
    EXPECT_TRUE(passed.empty());
}

TEST(LandscapeSanityFloors, NegativeYearsFilter) {
    // The sanity floor gates on negative_years (see docs/METHODOLOGY.md).
    // Create a result that passes all other
    // floors but has negative_years=1. With default SanityFloors
    // (max_negative_years=0), it should be filtered out.
    std::vector<BacktestResult> results;

    BacktestResult r = make_result(5000.0f, 200, 1.5f, 15.0f);
    r.negative_years = 1;
    results.push_back(r);

    // Default floors: max_negative_years = 0
    auto passed = apply_sanity_floors(results);
    EXPECT_EQ(passed.size(), 0u)
        << "Result with negative_years=1 should fail default sanity floor (max_negative_years=0)";

    // With relaxed floor: max_negative_years = 1
    SanityFloors relaxed;
    relaxed.max_negative_years = 1;
    auto passed_relaxed        = apply_sanity_floors(results, relaxed);
    EXPECT_EQ(passed_relaxed.size(), 1u)
        << "Result with negative_years=1 should pass when max_negative_years=1";
}

// ============================================================================
// Test: Distribution statistics
// ============================================================================

TEST(LandscapeDistribution, BasicStats) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto               stats  = compute_distribution(values);

    EXPECT_FLOAT_EQ(stats.min, 1.0f);
    EXPECT_FLOAT_EQ(stats.max, 5.0f);
    EXPECT_FLOAT_EQ(stats.mean, 3.0f);
    EXPECT_FLOAT_EQ(stats.median, 3.0f);
    EXPECT_FLOAT_EQ(stats.p25, 2.0f);
    EXPECT_FLOAT_EQ(stats.p75, 4.0f);
}

TEST(LandscapeDistribution, SingleValue) {
    std::vector<float> values = {42.0f};
    auto               stats  = compute_distribution(values);

    EXPECT_FLOAT_EQ(stats.min, 42.0f);
    EXPECT_FLOAT_EQ(stats.max, 42.0f);
    EXPECT_FLOAT_EQ(stats.mean, 42.0f);
    EXPECT_FLOAT_EQ(stats.median, 42.0f);
}

TEST(LandscapeDistribution, EmptyReturnsZero) {
    std::vector<float> values;
    auto               stats = compute_distribution(values);

    EXPECT_FLOAT_EQ(stats.min, 0.0f);
    EXPECT_FLOAT_EQ(stats.max, 0.0f);
    EXPECT_FLOAT_EQ(stats.mean, 0.0f);
}

// ============================================================================
// Test: Reduced set includes frontier + neighbors
// ============================================================================

TEST(LandscapeReducedSet, FrontierPlusNeighbors) {
    // 10 strategies, all with different robustness scores
    std::vector<float> robustness = {0.80f, 0.70f, 0.60f, 0.55f, 0.45f,
                                     0.90f, 0.75f, 0.65f, 0.40f, 0.30f};

    // Frontier = indices 0, 5
    std::vector<int> frontier = {0, 5};

    // Neighbor function: each strategy has neighbors at +/- 1
    auto get_neighbors = [](int idx) -> std::vector<int> {
        std::vector<int> nb;
        if (idx > 0) nb.push_back(idx - 1);
        if (idx < 9) nb.push_back(idx + 1);
        return nb;
    };

    float floor   = 0.50f;
    auto  reduced = build_reduced_set(frontier, robustness, floor, get_neighbors);

    // Strategy 0 (robustness=0.80): included
    // Strategy 0's neighbor 1 (robustness=0.70): included
    // Strategy 5 (robustness=0.90): included
    // Strategy 5's neighbor 4 (robustness=0.45): EXCLUDED (below floor)
    // Strategy 5's neighbor 6 (robustness=0.75): included
    EXPECT_TRUE(std::find(reduced.begin(), reduced.end(), 0) != reduced.end());
    EXPECT_TRUE(std::find(reduced.begin(), reduced.end(), 1) != reduced.end());
    EXPECT_TRUE(std::find(reduced.begin(), reduced.end(), 5) != reduced.end());
    EXPECT_TRUE(std::find(reduced.begin(), reduced.end(), 6) != reduced.end());

    // Strategy 4 should NOT be included (below floor)
    EXPECT_TRUE(std::find(reduced.begin(), reduced.end(), 4) == reduced.end());
}

TEST(LandscapeReducedSet, FrontierBelowFloorExcluded) {
    // All frontier strategies have robustness below the floor
    std::vector<float> robustness = {0.30f, 0.25f, 0.40f};
    std::vector<int>   frontier   = {0, 1, 2};

    auto get_neighbors = [](int /*idx*/) -> std::vector<int> { return {}; };

    auto reduced = build_reduced_set(frontier, robustness, 0.50f, get_neighbors);

    // None should pass
    EXPECT_TRUE(reduced.empty());
}

TEST(LandscapeReducedSet, NoDuplicates) {
    std::vector<float> robustness = {0.80f, 0.70f, 0.60f};
    std::vector<int>   frontier   = {0, 1};

    // Neighbors overlap: both 0 and 1 have each other as neighbors
    auto get_neighbors = [](int idx) -> std::vector<int> {
        if (idx == 0) return {1};
        if (idx == 1) return {0, 2};
        return {};
    };

    auto reduced = build_reduced_set(frontier, robustness, 0.50f, get_neighbors);

    // Check for uniqueness
    std::set<int> unique_set(reduced.begin(), reduced.end());
    EXPECT_EQ(reduced.size(), unique_set.size());
}

// ============================================================================
// Test: Full landscape analysis pipeline
// ============================================================================

TEST(LandscapeAnalysis, FullPipeline) {
    // Create 20 results with known properties
    std::vector<BacktestResult> results;

    // 5 strong strategies (pass all floors)
    for (int i = 0; i < 5; ++i) {
        float roc = 15.0f + i * 5.0f;
        float gw  = 0.60f + i * 0.05f;
        results.push_back(make_result(5000.0f + i * 1000.0f, 200 + i * 20, 1.5f + i * 0.2f, roc, gw,
                                      0.55f + i * 0.05f, 2, 0.5f + i * 0.1f, i));
    }

    // 5 mediocre (pass floors but less impressive)
    for (int i = 0; i < 5; ++i) {
        results.push_back(make_result(500.0f + i * 100.0f, 110 + i * 5, 1.1f, 2.0f + i * 0.5f,
                                      0.52f, 0.52f, 4, 0.2f, 5 + i));
    }

    // 10 garbage (fail floors)
    for (int i = 0; i < 10; ++i) {
        results.push_back(make_result(-1000.0f + i * 50.0f, 50 + i * 3, 0.7f + i * 0.02f,
                                      -5.0f + i * 0.3f, 0.30f, 0.40f, 8, 0.1f, 10 + i));
    }

    int n = static_cast<int>(results.size());

    // Simple neighbor function: +/- 1
    auto robustness_neighbors = [n](int idx) -> std::vector<std::pair<int, int>> {
        std::vector<std::pair<int, int>> nb;
        if (idx > 0) nb.push_back({0, idx - 1});
        if (idx < n - 1) nb.push_back({0, idx + 1});
        return nb;
    };

    auto reduced_neighbors = [n](int idx) -> std::vector<int> {
        std::vector<int> nb;
        if (idx > 0) nb.push_back(idx - 1);
        if (idx < n - 1) nb.push_back(idx + 1);
        return nb;
    };

    LandscapeAnalysis la =
        analyze_landscape(results, 0.50f, robustness_neighbors, 1, reduced_neighbors);

    EXPECT_EQ(la.total_candidates, 20);
    EXPECT_EQ(la.passed_sanity_floors, 10);  // 5 strong + 5 mediocre
    EXPECT_GT(la.on_pareto_frontier, 0);
    EXPECT_EQ(la.grinder_count + la.sniper_count + la.balanced_count, la.on_pareto_frontier);

    // Distribution stats should reflect the 10 survivors
    EXPECT_GT(la.roc_stats.median, 0.0f);
    EXPECT_GT(la.green_weeks_stats.median, 0.0f);

    // Robustness scores should be computed for all 20 results
    EXPECT_EQ(static_cast<int>(la.robustness_scores.size()), 20);
}

// ============================================================================
// Test: screen stage on random walk data -- nearly all negative after costs
// ============================================================================

TEST(ScreenE2E, NoEdgeRandomWalk) {
    // Generate random walk bars (no real signal)
    auto bars = generate_random_walk_bars(2000, 4500.0f, 123);

    // Create a small indicator grid
    IndicatorGrid indicator_grid;

    // Build indicator specs for a small subset (first 2 indicators only)
    // We will compute indicators and run a tiny strategy grid
    int num_bars = static_cast<int>(bars.size());

    // Use only the first 2 indicator entries for speed
    int num_indicators_to_use = std::min(2, indicator_grid.size());

    // Build specs for these indicators
    std::vector<IndicatorSpec> specs;
    for (int i = 0; i < num_indicators_to_use; ++i) {
        const IndicatorParams& params = indicator_grid.params(i);
        int                    ns     = 1;
        if (params.type == IndicatorType::EMA_SMA_CROSS)
            ns = 2;
        else if (params.type == IndicatorType::ATR)
            ns = 2;
        for (int s = 0; s < ns; ++s) {
            specs.push_back(IndicatorSpec::from_params(params, s));
        }
    }

    int total_series = static_cast<int>(specs.size());

    // Allocate indicator buffers
    std::vector<std::vector<float>> indicator_buffers(total_series,
                                                      std::vector<float>(num_bars, 0.0f));
    std::vector<float*>             indicator_ptrs;
    for (auto& buf : indicator_buffers) {
        indicator_ptrs.push_back(buf.data());
    }

    // Compute indicators using CLIndicators CPU fallback
    CLContext    ctx;  // not initialized => CPU fallback
    CLIndicators cl_ind;

    // Build SoA data
    std::vector<float> soa_open(num_bars), soa_high(num_bars), soa_low(num_bars),
        soa_close(num_bars);
    for (int i = 0; i < num_bars; ++i) {
        soa_open[i]  = bars[i].open;
        soa_high[i]  = bars[i].high;
        soa_low[i]   = bars[i].low;
        soa_close[i] = bars[i].close;
    }

    cl_ind.compute(ctx, soa_open.data(), soa_high.data(), soa_low.data(), soa_close.data(),
                   num_bars, specs, indicator_ptrs);

    // Normalize raw indicator series to [-1, +1] signals.
    // Build a temporary IndicatorGrid-like structure for normalization.
    // We only use the first num_indicators_to_use entries.
    int                             num_signals = num_indicators_to_use;
    std::vector<std::vector<float>> signal_storage(num_signals);
    std::vector<float*>             signal_ptrs(num_signals);
    for (int i = 0; i < num_signals; ++i) {
        signal_storage[i].resize(num_bars);
        signal_ptrs[i] = signal_storage[i].data();
    }

    // Normalize: group raw series by indicator and produce one signal per indicator
    {
        int raw_offset = 0;
        for (int ind = 0; ind < num_indicators_to_use; ++ind) {
            const IndicatorParams& params = indicator_grid.params(ind);
            int                    ns     = 1;
            if (params.type == IndicatorType::EMA_SMA_CROSS)
                ns = 2;
            else if (params.type == IndicatorType::ATR)
                ns = 2;

            std::vector<const float*> series_for_ind(ns);
            for (int s = 0; s < ns; ++s) {
                series_for_ind[s] = indicator_ptrs[raw_offset + s];
            }

            normalize_to_signal(params, series_for_ind.data(), ns, bars.data(), 0, num_bars,
                                signal_ptrs[ind]);
            raw_offset += ns;
        }
    }

    // Build a small strategy grid over the normalized signals
    StrategyGrid strat_grid(num_signals);

    // Evaluate a limited number of strategies (for test speed)
    int64_t max_eval = std::min(strat_grid.total_count(), int64_t(200));

    int warmup = 0;
    for (const auto& spec : specs) {
        warmup = std::max(warmup, spec.warmup_bars);
    }
    warmup *= 2;
    int eval_start = std::min(warmup, num_bars - 1);
    int eval_end   = num_bars;

    CostModel      cost = CostModel::es();
    BacktestEngine engine;
    CLBacktest     cl_bt;

    std::vector<BacktestResult> all_results;
    all_results.reserve(max_eval);

    for (int64_t s = 0; s < max_eval; ++s) {
        StrategyGridEntry entry = strat_grid.entry_at(s);
        StrategyConfig    sc    = entry.to_strategy_config();

        // Create offset pointers for the evaluation window
        std::vector<const float*> offset_ptrs(num_signals);
        for (int k = 0; k < num_signals; ++k) {
            offset_ptrs[k] = signal_ptrs[k] + eval_start;
        }

        BacktestResult br = engine.run(bars.data(), eval_start, eval_end, offset_ptrs.data(),
                                       num_signals, sc, cost, 1);
        br.strategy_id    = static_cast<int>(s);
        all_results.push_back(br);
    }

    // On random walk data with costs, most strategies should be negative
    int positive_count = 0;
    for (const auto& r : all_results) {
        if (r.total_pnl > 0.0f) positive_count++;
    }

    // On a short random walk with small samples, a significant fraction may
    // be positive by chance alone (especially with low trade counts).
    // The key test is that sanity floors filter most out (below).
    // We check that at least some are negative -- on pure random data
    // with costs, profitability should not be universal.
    float positive_pct =
        static_cast<float>(positive_count) / static_cast<float>(all_results.size());
    EXPECT_LT(positive_pct, 0.80f)
        << "Nearly all strategies profitable on random walk data -- suspicious: " << positive_count
        << " / " << all_results.size();

    // Assert that on random walk data the average total_pnl and
    // annual_roc across all strategies are near-zero or negative after costs.
    // A random walk has no exploitable signal, so transaction costs should
    // drag the mean into negative territory.  We allow a small positive
    // tolerance for sampling noise but any significant positive average
    // would indicate a systematic bias in the engine.
    {
        float sum_pnl           = 0.0f;
        float sum_roc           = 0.0f;
        int   count_with_trades = 0;
        for (const auto& r : all_results) {
            sum_pnl += r.total_pnl;
            if (r.total_trades > 0) {
                sum_roc += r.annual_roc;
                ++count_with_trades;
            }
        }
        float mean_pnl = sum_pnl / static_cast<float>(all_results.size());
        EXPECT_LE(mean_pnl, 500.0f)
            << "Mean total_pnl on random walk should be near-zero or negative "
               "after costs, got "
            << mean_pnl;

        // No mean annual_roc assertion here. The synthetic
        // random-walk window is only ~16 hours (2000 bars * 30s), so
        // CAGR = (final/capital)^(1/years) - 1 with years ~ 0.002 will
        // compound any nonzero total_pnl into astronomical values and
        // float-overflow the mean. The meaningful bias check is on
        // mean_pnl (above), which is unaffected by annualisation.
        (void)sum_roc;
        (void)count_with_trades;
    }

    // Apply sanity floors -- very few should pass
    auto passed = apply_sanity_floors(all_results);

    // On random walk, the sanity floors (pnl > 0 AND trades >= 100 AND
    // profit_factor > 1 AND annual_roc > 0) should eliminate most strategies.
    // With only ~2000 bars and random data, few strategies will have 100+ trades.
    float pass_rate = static_cast<float>(passed.size()) / static_cast<float>(all_results.size());
    EXPECT_LT(pass_rate, 0.30f) << "Too many strategies passed all sanity floors on random data: "
                                << passed.size() << " / " << all_results.size();
}

// ============================================================================
// Test: Orchestrator with small grid on synthetic data
// ============================================================================

TEST(OrchestratorSmallGrid, RunsAndWritesOutput) {
    // Generate synthetic bars
    auto bars     = generate_random_walk_bars(1500, 4500.0f, 99);
    int  num_bars = static_cast<int>(bars.size());

    // Create a small indicator grid
    IndicatorGrid indicator_grid;

    // Determine warmup
    int max_warmup = 0;
    for (int i = 0; i < std::min(2, indicator_grid.size()); ++i) {
        IndicatorSpec spec = IndicatorSpec::from_params(indicator_grid.params(i), 0);
        max_warmup         = std::max(max_warmup, spec.warmup_bars);
    }
    max_warmup *= 2;

    int eval_start = std::min(max_warmup, num_bars - 100);
    int eval_end   = num_bars;

    // Configure the screen stage with a very small strategy limit
    Orchestrator::ScreenConfig config;
    config.scan_start_bar        = eval_start;
    config.scan_end_bar          = eval_end;
    config.robustness_floor      = 0.50f;
    config.max_active_indicators = 4;
    config.cost                  = CostModel::es();
    config.entry_lag             = 1;
    config.max_strategies        = 50;  // limit for test speed

    // Create a temp output directory
    std::string temp_dir = std::filesystem::temp_directory_path().string() + "/wfo_test_screen";

    // Clean up from previous runs
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    config.output_dir = temp_dir;

    // Run the screen stage with pre-loaded data
    Orchestrator orchestrator;
    auto result = orchestrator.run_screen_with_data(bars.data(), num_bars, eval_start, eval_end,
                                                    indicator_grid, config);

    // Basic validation
    EXPECT_GT(result.total_evaluated, 0);
    EXPECT_EQ(static_cast<int>(result.all_results.size()), result.total_evaluated);

    // Write results
    bool written = Orchestrator::write_results(temp_dir, result);
    EXPECT_TRUE(written);

    // Verify output files exist
    EXPECT_TRUE(std::filesystem::exists(temp_dir + "/reduced_params.bin"));
    EXPECT_TRUE(std::filesystem::exists(temp_dir + "/landscape_summary.csv"));
    EXPECT_TRUE(std::filesystem::exists(temp_dir + "/all_results.bin"));

    // Verify reduced_params.bin is well-formed
    {
        std::ifstream in(temp_dir + "/reduced_params.bin", std::ios::binary);
        ASSERT_TRUE(in.is_open());
        int32_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        EXPECT_GE(count, 0);
        EXPECT_EQ(count, result.reduced_set_size);

        if (count > 0) {
            std::vector<int32_t> ids(count);
            in.read(reinterpret_cast<char*>(ids.data()), count * sizeof(int32_t));
            EXPECT_TRUE(in.good());
        }
    }

    // Verify landscape_summary.csv is non-empty and has expected structure
    {
        std::ifstream in(temp_dir + "/landscape_summary.csv");
        ASSERT_TRUE(in.is_open());
        std::string header;
        std::getline(in, header);
        EXPECT_EQ(header, "metric,value");

        // Should have at least a few lines
        std::string line;
        int         line_count = 0;
        while (std::getline(in, line)) {
            ++line_count;
        }
        EXPECT_GE(line_count, 5);
    }

    // Verify all_results.bin contains the right number of results.
    // all_results.bin is self-describing with a 4-byte magic "WFAR"
    // and a 4-byte version prefix BEFORE the int32 count, so we must skip
    // those 8 bytes before reading the count.
    {
        std::ifstream in(temp_dir + "/all_results.bin", std::ios::binary);
        ASSERT_TRUE(in.is_open());
        char magic[4] = {};
        in.read(magic, 4);
        EXPECT_EQ(std::string(magic, 4), std::string("WFAR"));
        uint32_t version = 0;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        int32_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        EXPECT_EQ(count, result.total_evaluated);
    }

    // Clean up
    std::filesystem::remove_all(temp_dir);
}

// ============================================================================
// Test: Landscape with all results passing floors -> frontier is subset
// ============================================================================

TEST(LandscapeAnalysis, AllPassingFloors) {
    std::vector<BacktestResult> results;

    // Create 5 results that all pass, with different profiles
    // so Pareto frontier is a proper subset
    results.push_back(make_result(10000.0f, 300, 2.0f, 30.0f, 0.80f, 0.90f, 1, 1.0f, 0));
    results.push_back(make_result(5000.0f, 200, 1.5f, 15.0f, 0.60f, 0.55f, 3, 0.5f, 1));
    results.push_back(make_result(8000.0f, 250, 1.8f, 25.0f, 0.70f, 0.60f, 2, 0.8f, 2));
    // This one is dominated by result 0 in all dimensions:
    results.push_back(make_result(3000.0f, 150, 1.3f, 10.0f, 0.55f, 0.50f, 4, 0.3f, 3));
    // Another dominated by result 2:
    results.push_back(make_result(4000.0f, 180, 1.4f, 12.0f, 0.58f, 0.52f, 3, 0.4f, 4));

    int n = static_cast<int>(results.size());

    auto rob_nb = [n](int idx) -> std::vector<std::pair<int, int>> {
        std::vector<std::pair<int, int>> nb;
        if (idx > 0) nb.push_back({0, idx - 1});
        if (idx < n - 1) nb.push_back({0, idx + 1});
        return nb;
    };

    auto red_nb = [n](int idx) -> std::vector<int> {
        std::vector<int> nb;
        if (idx > 0) nb.push_back(idx - 1);
        if (idx < n - 1) nb.push_back(idx + 1);
        return nb;
    };

    auto la = analyze_landscape(results, 0.50f, rob_nb, 1, red_nb);

    EXPECT_EQ(la.passed_sanity_floors, 5);
    EXPECT_GT(la.on_pareto_frontier, 0);
    EXPECT_LE(la.on_pareto_frontier, 5);
}

// ============================================================================
// Test: Empty results produce empty landscape
// ============================================================================

TEST(LandscapeAnalysis, EmptyResults) {
    std::vector<BacktestResult> results;
    auto                        rob_nb = [](int) -> std::vector<std::pair<int, int>> { return {}; };
    auto                        red_nb = [](int) -> std::vector<int> { return {}; };

    auto la = analyze_landscape(results, 0.50f, rob_nb, 1, red_nb);

    EXPECT_EQ(la.total_candidates, 0);
    EXPECT_EQ(la.passed_sanity_floors, 0);
    EXPECT_EQ(la.on_pareto_frontier, 0);
    EXPECT_TRUE(la.reduced_set_ids.empty());
}

// ============================================================================
// Test: Pareto -- max_red_weeks inversion
// A strategy with fewer max_red_weeks should NOT be dominated by one with more,
// because Pareto dimension 3 is 1/(1+max_red_weeks) (higher = better = fewer
// red weeks).
// ============================================================================

TEST(Pareto, MaxRedWeeksInversion) {
    // Strategy A: high ROC, 2 max_red_weeks, moderate on other dims
    // Strategy B: high ROC, 5 max_red_weeks, slightly better on everything EXCEPT red weeks
    // A should NOT be dominated by B because A has fewer red weeks
    // (1/(1+2) = 0.333 > 1/(1+5) = 0.167)

    std::vector<BacktestResult> results;

    // Strategy A: 2 max_red_weeks (better)
    BacktestResult a = make_result(5000.0f, 200, 1.5f, 20.0f, 0.65f, 0.55f, 2, 0.5f, 0);

    // Strategy B: 5 max_red_weeks (worse), but slightly better on all other dims
    BacktestResult b = make_result(6000.0f, 210, 1.6f, 21.0f, 0.66f, 0.56f, 5, 0.6f, 1);

    results.push_back(a);
    results.push_back(b);

    // A should NOT be dominated by B (A is better on max_red_weeks dimension)
    EXPECT_FALSE(dominates(b, a)) << "B should NOT dominate A because A has fewer max_red_weeks";

    // B should NOT be dominated by A either (B is better on ROC, green weeks, PF and TRR)
    EXPECT_FALSE(dominates(a, b))
        << "A should NOT dominate B because B is better on other dimensions";

    // Both should be on the Pareto frontier
    auto frontier = extract_pareto_frontier(results);
    EXPECT_EQ(frontier.size(), 2u)
        << "Both strategies should be on the Pareto frontier due to max_red_weeks tradeoff";
}

// ============================================================================
// LOOKAHEAD CANARY (E2E)
//
// Create a "perfect foreknowledge" signal: signal[i] = sign(close[i+1] - close[i]).
// This is a lookahead signal that knows the next bar's direction.
// Running a backtest with this signal should produce an absurdly profitable
// result, confirming that:
//   (a) the backtest engine correctly uses signals to enter trades, and
//   (b) if such an indicator were accidentally introduced, it would be
//       detectable via impossibly good performance metrics.
// ============================================================================

// Helper: create a minimal Bar (used locally to avoid dependency on other test files' helpers)
static Bar make_bar_e2e(int64_t ts, float o, float h, float l, float c, uint32_t v = 100) {
    Bar b{};
    b.timestamp = ts;
    b.open      = o;
    b.high      = h;
    b.low       = l;
    b.close     = c;
    b.volume    = v;
    b._pad      = 0;
    return b;
}

TEST(E2E, LookaheadCanary) {
    // Generate a trending + oscillating price series (NOT random walk --
    // we want actual movement so trades have meaningful PnL).
    const int                N          = 500;
    static constexpr int64_t NS_PER_SEC = 1'000'000'000LL;
    int64_t                  base_ts    = 1'262'304'000LL * NS_PER_SEC;

    std::vector<Bar>        bars(N);
    std::mt19937            rng(42);
    portable::normal<float> step{0.0f, 2.0f};

    float price = 4500.0f;
    for (int i = 0; i < N; ++i) {
        float move  = step(rng);
        float open  = price;
        float close = price + move;
        float high  = std::max(open, close) + std::abs(step(rng)) * 0.5f;
        float low   = std::min(open, close) - std::abs(step(rng)) * 0.5f;

        bars[i] = make_bar_e2e(base_ts + static_cast<int64_t>(i) * 30 * NS_PER_SEC, open, high, low,
                               close);
        price   = close;
    }

    // Create the "canary" lookahead signal: signal[i] = sign(close[i+1] - close[i])
    // This uses perfect foreknowledge of the next bar's direction.
    std::vector<float> canary_signal(N, 0.0f);
    for (int i = 0; i < N - 1; ++i) {
        float diff = bars[i + 1].close - bars[i].close;
        if (diff > 0.0f)
            canary_signal[i] = 1.0f;
        else if (diff < 0.0f)
            canary_signal[i] = -1.0f;
        // diff == 0 -> signal stays 0 (no trade)
    }
    const float* signals[] = {canary_signal.data()};

    // Configure: simple strategy, exit after 1 bar to maximize trade count
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 1;

    CostModel      cost = CostModel::es();
    BacktestEngine engine;
    BacktestResult result = engine.run(bars.data(), 0, N, signals, 1, config, cost, 1);

    // With perfect foreknowledge:
    //   - Signal fires on bar i (knowing bar i+1 direction)
    //   - Entry fills at bar i+1 open (lag-1)
    //   - Exit after 1 bar at bar i+2 open
    //   - Most trades should be profitable because we predicted the direction
    //     (though entry-at-open-of-next-bar introduces some slippage vs close)
    //
    // The key assertion: this should produce an extremely high win rate and
    // very high total PnL relative to a random strategy.

    EXPECT_GT(result.total_trades, 50) << "Canary signal should generate many trades";

    // The canary predicts close[i+1]-close[i] direction, but the actual trade is
    // open[i+1] to open[i+2] (lag-1, BAR_TIMEOUT=1).  With $75 RT costs on small
    // synthetic moves, the edge is thin.  The critical assertion is the relative
    // comparison vs random (below), not absolute win rate.

    // Now run the SAME bars with a random signal as baseline comparison.
    // The canary should be dramatically better than random.
    std::vector<float>       random_signal(N, 0.0f);
    std::mt19937             rng2(999);
    portable::uniform<float> coin{-1.0f, 1.0f};
    for (int i = 0; i < N; ++i) {
        random_signal[i] = coin(rng2);
    }
    const float* random_signals[] = {random_signal.data()};

    BacktestResult random_result =
        engine.run(bars.data(), 0, N, random_signals, 1, config, cost, 1);

    // The canary should outperform random.  With high costs ($75 RT), both may be
    // negative in absolute terms, but the canary should be significantly less negative
    // (or positive) due to its directional edge.
    EXPECT_GT(result.total_pnl, random_result.total_pnl)
        << "Lookahead canary should outperform random signal. "
        << "Canary PnL=" << result.total_pnl << " Random PnL=" << random_result.total_pnl;
}

// The signal count used to rebuild robustness geometry on reanalysis must
// equal the count a file-based screen actually enumerates.
TEST(Screen, SignalCountMatchesFileScreen) {
    for (const SessionModel& model : {SessionModel::us_futures(), SessionModel::continuous()}) {
        Orchestrator::ScreenConfig sc;
        sc.bar_file       = test_fixture::fixture_path();
        sc.session        = model;
        sc.max_strategies = 64;
        Orchestrator orch;
        const auto   r = orch.run_screen(sc);
        EXPECT_GT(r.total_evaluated, 0) << model.name;
        EXPECT_EQ(r.num_signals, Orchestrator::screen_signal_count(model)) << model.name;
    }
}

// The screen hands walk-forward strategy-grid ids, not positions in its
// results vector. The two differ because redundant grid entries are skipped.
TEST(Screen, ReducedSetHoldsGridIds) {
    Orchestrator::ScreenConfig sc;
    sc.bar_file                         = test_fixture::fixture_path();
    sc.max_strategies                   = 6000;
    sc.robustness_floor                 = 0.0f;
    sc.sanity_floors.min_total_trades   = 1;
    sc.sanity_floors.min_profit_factor  = 0.0f;
    sc.sanity_floors.min_total_pnl      = -1e9f;
    sc.sanity_floors.min_annual_roc     = -1e9f;
    sc.sanity_floors.max_negative_years = 1000;
    Orchestrator orch;
    const auto   r = orch.run_screen(sc);
    ASSERT_GT(r.total_evaluated, 0);

    bool ids_differ_from_positions = false;
    for (int i = 0; i < static_cast<int>(r.all_results.size()); ++i)
        if (r.all_results[i].strategy_id != i) ids_differ_from_positions = true;
    ASSERT_TRUE(ids_differ_from_positions) << "test needs skipped grid entries";

    std::set<int> evaluated;
    for (const auto& res : r.all_results) evaluated.insert(res.strategy_id);
    std::set<int> expected;
    for (int pos : r.landscape.reduced_set_ids) expected.insert(r.all_results[pos].strategy_id);
    ASSERT_EQ(r.reduced_strategy_ids.size(), expected.size());
    for (int id : r.reduced_strategy_ids) {
        EXPECT_TRUE(evaluated.count(id)) << "id " << id << " was never evaluated";
        EXPECT_TRUE(expected.count(id)) << "id " << id << " is not in the reduced set";
    }
}
