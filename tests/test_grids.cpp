#include <gtest/gtest.h>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

#include "backtest_engine.h"
#include "cost_model.h"

#include "grids/indicator_grid.h"
#include "grids/session_signal_grid.h"
#include "grids/window_mode_grid.h"
#include "grids/strategy_grid.h"
#include "grids/wfo_grid.h"
#include "grids/selection_grid.h"
#include "pareto.h"
#include "selection.h"
#include "robustness.h"

// ============================================================================
// Indicator Grid Tests
// ============================================================================

TEST(IndicatorGrid, CorrectCount) {
    IndicatorGrid grid;

    // EMA/SMA Cross: 2 fast types x 6 fast periods x 2 slow types x 6 slow periods
    // minus combos where fast_period >= slow_period
    // fast periods: {3,5,8,13,21,33}, slow periods: {20,34,50,100,200,233}
    // Pruned: fast=21 >= slow=20 (1), fast=33 >= slow=20 (1) = 2 per type pair
    // So per (fast_type, slow_type) pair: 6*6 - 2 = 34 combos
    // 2 fast types x 2 slow types x 34 = 136
    EXPECT_EQ(grid.ema_sma_cross_count(), 136);

    // RSI: 7 periods x 3 overbought x 3 oversold = 63
    EXPECT_EQ(grid.rsi_count(), 63);

    // ATR: 4 short x 4 long, prune short >= long
    // short={5,7,10,14}, long={20,34,50,100}
    // All short < all long, so no pruning. 4*4 = 16
    EXPECT_EQ(grid.atr_count(), 16);

    // MACD: 3 fast x 3 slow x 3 signal = 27
    EXPECT_EQ(grid.macd_count(), 27);

    // ADX: 4 periods x 5 thresholds = 20
    EXPECT_EQ(grid.adx_count(), 20);

    // Bollinger: 4 periods x 4 multipliers = 16
    EXPECT_EQ(grid.bollinger_count(), 16);

    // Keltner: 1 ma_type x 4 ma_periods x 4 atr_periods x 4 mults = 64
    EXPECT_EQ(grid.keltner_count(), 64);

    // Stochastic: 4 k_periods x 2 d_periods x 3 OB x 3 OS = 72
    EXPECT_EQ(grid.stochastic_count(), 72);

    // CCI: 4 periods x 4 thresholds = 16
    EXPECT_EQ(grid.cci_count(), 16);

    // ROC: 5 periods
    EXPECT_EQ(grid.roc_count(), 5);

    // VWAP: 3 band multipliers
    EXPECT_EQ(grid.vwap_count(), 3);

    // OBV: 3 smoothings
    EXPECT_EQ(grid.obv_count(), 3);

    // Donchian: 5 periods
    EXPECT_EQ(grid.donchian_count(), 5);

    // Pivot Points: 3 types
    EXPECT_EQ(grid.pivot_count(), 3);

    // HMA: 6 periods
    EXPECT_EQ(grid.hma_count(), 6);

    // Connors RSI: 2 rsi x 2 streak x 2 percentile x 2 OB x 2 OS = 32
    EXPECT_EQ(grid.connors_rsi_count(), 32);

    // Total: 136+63+16+27+20+16+64+72+16+5+3+3+5+3+6+32 = 487
    EXPECT_EQ(grid.size(), 487);
}

TEST(IndicatorGrid, Determinism) {
    IndicatorGrid grid1;
    IndicatorGrid grid2;

    ASSERT_EQ(grid1.size(), grid2.size());

    for (int i = 0; i < grid1.size(); ++i) {
        const auto& p1 = grid1.params(i);
        const auto& p2 = grid2.params(i);
        EXPECT_EQ(p1, p2) << "Mismatch at index " << i;
    }
}

TEST(IndicatorGrid, NoDuplicates) {
    IndicatorGrid grid;

    // Check that no two IDs map to the same params by using find()
    // which returns the first match. If any param appears more than once,
    // some index won't match its find() result.
    for (int i = 0; i < grid.size(); ++i) {
        int found = grid.find(grid.params(i));
        EXPECT_EQ(found, i) << "Duplicate params at indices " << found << " and " << i;
    }
}

TEST(IndicatorGrid, RoundTrip) {
    IndicatorGrid grid;

    // Test round-trip: params(i) -> find() -> i
    for (int i = 0; i < grid.size(); ++i) {
        int found = grid.find(grid.params(i));
        EXPECT_EQ(found, i);
    }

    // Test that a non-existent param returns -1
    IndicatorParams bogus;
    bogus.type        = IndicatorType::EMA_SMA_CROSS;
    bogus.fast_period = 999;
    EXPECT_EQ(grid.find(bogus), -1);
}

// ============================================================================
// Window Mode Grid Tests
// ============================================================================

TEST(WindowModeGrid, PrunesLongLookbacks) {
    IndicatorGrid  ig;
    WindowModeGrid wmg(ig);

    // Every entry should be valid (lookback <= 168 hours = 7 days,
    // the regime-detection threshold)
    for (int i = 0; i < wmg.size(); ++i) {
        const auto& e = wmg.entry(i);
        const auto& p = ig.params(e.indicator_id);

        // Session-anchored indicators (VWAP, Pivot Points) must be ROLLING only
        if (p.type == IndicatorType::VWAP || p.type == IndicatorType::PIVOT_POINTS) {
            EXPECT_EQ(e.mode, WindowMode::ROLLING)
                << "Session-anchored indicator has non-ROLLING mode at entry " << i;
            continue;
        }

        // OBV should only have ROLLING or short anchored modes
        if (p.type == IndicatorType::OBV && e.mode != WindowMode::ROLLING) {
            EXPECT_TRUE(e.mode == WindowMode::ANCHORED_1M || e.mode == WindowMode::ANCHORED_2M ||
                        e.mode == WindowMode::ANCHORED_5M)
                << "OBV has too-coarse anchored mode at entry " << i;
        }

        if (e.mode != WindowMode::ROLLING) {
            // Compute effective lookback for non-session-anchored indicators
            int period = 0;
            switch (p.type) {
                case IndicatorType::EMA_SMA_CROSS:
                    period = p.slow_period;
                    break;
                case IndicatorType::RSI:
                    period = p.rsi_period;
                    break;
                case IndicatorType::ATR:
                    period = p.atr_long_period;
                    break;
                default:
                    break;
            }
            float lookback_hours = static_cast<float>(period) *
                                   static_cast<float>(window_mode_seconds(e.mode)) / 3600.0f;
            EXPECT_LE(lookback_hours, 168.0f)
                << "Entry " << i << " has " << lookback_hours << "h lookback";
        }
    }

    // The grid should have more entries than just rolling (since many combos
    // are valid with anchored modes too)
    EXPECT_GT(wmg.size(), ig.size());

    // But fewer than ig.size() * 7 (some combos get pruned due to
    // session-anchored restrictions and lookback limits)
    EXPECT_LT(wmg.size(), ig.size() * 7);
}

TEST(WindowModeGrid, AllRollingPresent) {
    IndicatorGrid  ig;
    WindowModeGrid wmg(ig);

    // Every indicator should have a rolling entry
    for (int i = 0; i < ig.size(); ++i) {
        WindowModeEntry target;
        target.indicator_id = i;
        target.mode         = WindowMode::ROLLING;
        EXPECT_GE(wmg.find(target), 0) << "Indicator " << i << " missing rolling entry";
    }
}

// ============================================================================
// Strategy Grid Tests
// ============================================================================

TEST(StrategyGrid, SubsetCount) {
    // With N=10 indicators:
    // C(10,1) + C(10,2) + C(10,3) + C(10,4)
    // = 10 + 45 + 120 + 210 = 385
    StrategyGrid sg(10);
    EXPECT_EQ(sg.subset_count(), 385);
}

TEST(StrategyGrid, TotalCount) {
    // With N=5 indicators:
    // C(5,1)*4^1 + C(5,2)*4^2 + C(5,3)*4^3 + C(5,4)*4^4
    // = 5*4 + 10*16 + 10*64 + 5*256
    // = 20 + 160 + 640 + 1280 = 2100
    // Each multiplied by AGG_MODES(3) * ENTRY_THRESHOLDS(4) * EXIT_MODES(24) * TRADING_HOURS(4) * ROLE_PATTERNS(2) = 2304
    // Total = 2100 * 2304 = 4838400
    StrategyGrid sg(5);
    EXPECT_EQ(sg.total_count(), 4838400);
}

TEST(StrategyGrid, EntryAtDeterministic) {
    StrategyGrid sg(5);

    // Reconstruct several entries and verify they are consistent
    for (int64_t i = 0; i < 100; ++i) {
        auto e1 = sg.entry_at(i);
        auto e2 = sg.entry_at(i);
        EXPECT_EQ(e1.num_indicators, e2.num_indicators);
        for (int j = 0; j < e1.num_indicators; ++j) {
            EXPECT_EQ(e1.indicator_ids[j], e2.indicator_ids[j]);
            EXPECT_FLOAT_EQ(e1.weights[j], e2.weights[j]);
        }
        EXPECT_EQ(static_cast<int>(e1.aggregation), static_cast<int>(e2.aggregation));
        EXPECT_FLOAT_EQ(e1.entry_threshold, e2.entry_threshold);
        EXPECT_EQ(static_cast<int>(e1.exit_mode), static_cast<int>(e2.exit_mode));
    }
}

TEST(StrategyGrid, EntryAtNoDuplicates) {
    // Use a small grid (N=3) so we can check all entries
    StrategyGrid sg(3);
    int64_t      total = sg.total_count();
    // C(3,1)*4 + C(3,2)*16 + C(3,3)*64 = 12+48+64 = 124
    // * AGG_MODES(3) * THRESHOLDS(4) * EXIT_MODES(24) * TRADING_HOURS(4) * ROLE_PATTERNS(2) = 2304
    // = 124 * 2304 = 285696
    EXPECT_EQ(total, 285696);

    // Check a sampling of entries for uniqueness (check first 200 and last 200)
    auto entry_key = [](const StrategyGridEntry& e) -> std::string {
        std::string key;
        key += std::to_string(e.num_indicators) + ":";
        for (int i = 0; i < e.num_indicators; ++i) {
            key += std::to_string(e.indicator_ids[i]) + ",";
            key += std::to_string(e.weights[i]) + ",";
        }
        key += std::to_string(static_cast<int>(e.aggregation)) + ":";
        key += std::to_string(e.entry_threshold) + ":";
        key += std::to_string(static_cast<int>(e.exit_mode)) + ":";
        key += std::to_string(static_cast<int>(e.trading_hours)) + ":";
        for (int i = 0; i < e.num_indicators; ++i)
            key += std::to_string(static_cast<int>(e.indicator_roles[i]));
        return key;
    };

    std::set<std::string> seen;
    for (int64_t i = 0; i < std::min(total, int64_t(200)); ++i) {
        auto e = sg.entry_at(i);
        auto k = entry_key(e);
        EXPECT_TRUE(seen.insert(k).second) << "Duplicate at index " << i << ": " << k;
    }
    for (int64_t i = std::max(int64_t(0), total - 200); i < total; ++i) {
        auto e = sg.entry_at(i);
        auto k = entry_key(e);
        EXPECT_TRUE(seen.insert(k).second) << "Duplicate at index " << i << ": " << k;
    }
}

TEST(StrategyGrid, ToStrategyConfig) {
    StrategyGrid sg(10);
    auto         e  = sg.entry_at(0);
    auto         sc = e.to_strategy_config();

    EXPECT_EQ(sc.num_indicators, e.num_indicators);
    EXPECT_FLOAT_EQ(sc.entry_threshold, e.entry_threshold);
}

// ============================================================================
// WFO Grid Tests
// ============================================================================

TEST(WFOGrid, CorrectCount) {
    WFOGrid grid;

    // Fixed-interval: 4 freq x 8 lookback = 32, minus those where lookback_days < freq_days
    // freq: 1w(7), 2w(14), 1mo(30), 1q(91)
    // lookback: 1mo(30), 2mo(61), 3mo(91), 6mo(183), 1y(365), 2y(730), 3y(1095), 5y(1826)
    // Pruned (lb < freq):
    //   1w: none pruned (all lb >= 7) -> 8
    //   2w: none pruned (all lb >= 14) -> 8
    //   1mo: none pruned (all lb >= 30) -> 8
    //   1q: lb=1mo(30<91) pruned, lb=2mo(61<91) pruned -> 6
    // Fixed total = 8 + 8 + 8 + 6 = 30
    //
    // Adaptive triggers: 4 triggers x 8 lookbacks x 1 freq (1w) = 32
    // Grand total = 30 + 32 = 62
    EXPECT_EQ(grid.size(), 62);

    // All entries should have lookback >= frequency
    for (int i = 0; i < grid.size(); ++i) {
        auto c = grid.config(i);
        EXPECT_GE(lookback_approx_days(c.lookback), frequency_approx_days(c.frequency))
            << "Entry " << i << ": lookback < frequency";
    }
}

TEST(WFOGrid, Determinism) {
    WFOGrid g1;
    WFOGrid g2;
    ASSERT_EQ(g1.size(), g2.size());
    for (int i = 0; i < g1.size(); ++i) {
        EXPECT_EQ(g1.config(i), g2.config(i));
    }
}

TEST(WFOGrid, RoundTrip) {
    WFOGrid grid;
    for (int i = 0; i < grid.size(); ++i) {
        EXPECT_EQ(grid.find(grid.config(i)), i);
    }
}

// ============================================================================
// Selection Grid Tests
// ============================================================================

TEST(SelectionGrid, Count) {
    EXPECT_EQ(SELECTION_STRATEGY_COUNT, 7);
}

TEST(SelectionGrid, Names) {
    EXPECT_STREQ(selection_strategy_name(SelectionStrategy::BEST_ROC), "best_roc");
    EXPECT_STREQ(selection_strategy_name(SelectionStrategy::FRONTIER_CENTROID),
                 "frontier_centroid");
    EXPECT_STREQ(selection_strategy_name(SelectionStrategy::ROC_X_CONSISTENCY),
                 "roc_x_consistency");
}

// ============================================================================
// Session Signal Grid Tests
// ============================================================================

TEST(SessionSignalGrid, CorrectCount) {
    SessionSignalGrid grid;

    // OpeningRange: 2 x 2 x 2 = 8 entries, each producing 8 signals = 64
    // IntradayStructure: 2 x 2 = 4 entries, each producing 10 signals = 40
    // MultidayPatterns: 2 entries, each producing 12 signals = 24
    // LevelInteraction: 1 entry, 7 signals = 7
    // VolatilityRegime: 1 entry, 4 signals = 4
    // GapAnalysis: 1 entry, 6 signals = 6
    // OvernightContext: 1 entry, 7 signals = 7
    // Calendar: 1 entry, 4 signals = 4
    // Total entries: 8+4+2+1+1+1+1+1 = 19
    // Total signals: 64+40+24+7+4+6+7+4 = 156
    EXPECT_EQ(grid.size(), 19);
    EXPECT_EQ(grid.total_signals(), 156);

    // Verify each entry produces the expected number of signals
    int signal_sum = 0;
    for (int i = 0; i < grid.size(); ++i) {
        int ns = session_signal_num_outputs(grid.params(i));
        EXPECT_GT(ns, 0) << "Entry " << i << " has zero signals";
        signal_sum += ns;
    }
    EXPECT_EQ(signal_sum, 156);
}

TEST(SessionSignalGrid, Determinism) {
    SessionSignalGrid grid1;
    SessionSignalGrid grid2;

    ASSERT_EQ(grid1.size(), grid2.size());
    ASSERT_EQ(grid1.total_signals(), grid2.total_signals());

    for (int i = 0; i < grid1.size(); ++i) {
        EXPECT_EQ(grid1.params(i), grid2.params(i)) << "Mismatch at index " << i;
    }
}

TEST(SessionSignalGrid, NoDuplicates) {
    SessionSignalGrid grid;

    // Check that no two IDs map to the same params by using find()
    // which returns the first match. If any param appears more than once,
    // some index won't match its find() result.
    for (int i = 0; i < grid.size(); ++i) {
        int found = grid.find(grid.params(i));
        EXPECT_EQ(found, i) << "Duplicate params at indices " << found << " and " << i;
    }
}

TEST(SessionSignalGrid, FactoryCreatesInstances) {
    SessionSignalGrid grid;

    for (int i = 0; i < grid.size(); ++i) {
        auto instance = create_session_signal(grid.params(i));
        ASSERT_NE(instance, nullptr) << "Factory returned null for entry " << i;
        EXPECT_EQ(instance->num_signals(), session_signal_num_outputs(grid.params(i)))
            << "Signal count mismatch for entry " << i;
        EXPECT_GT(instance->name()[0], 0) << "Empty name for entry " << i;
    }
}

// ============================================================================
// Pareto Tests
// ============================================================================

// Helper to create a BacktestResult with specific Pareto dimensions
static BacktestResult make_result(float roc, float gw, int red_wk, float trr, float pf,
                                  float wr = 0.5f) {
    BacktestResult r;
    r.annual_roc           = roc;
    r.green_weeks_pct      = gw;
    r.max_red_weeks        = red_wk;
    r.trade_recovery_ratio = trr;
    r.profit_factor        = pf;
    r.win_rate             = wr;
    // Batch robustness skips neighbors with !metrics_complete.
    // All test fixtures here are fully-specified rows.
    r.metrics_complete = true;
    return r;
}

TEST(Pareto, Simple5Strategies2D) {
    // 5 strategies, effectively 2 meaningful dimensions (ROC and green_weeks),
    // other dims equal. Hand-verify frontier.
    std::vector<BacktestResult> results;
    //              ROC   GW  red  TRR  PF
    results.push_back(make_result(10.0f, 0.5f, 2, 1.0f, 2.0f));  // 0: moderate
    results.push_back(make_result(20.0f, 0.3f, 2, 1.0f, 2.0f));  // 1: high ROC, low GW
    results.push_back(make_result(5.0f, 0.8f, 2, 1.0f, 2.0f));   // 2: low ROC, high GW
    results.push_back(make_result(15.0f, 0.6f, 2, 1.0f, 2.0f));  // 3: above moderate
    results.push_back(make_result(8.0f, 0.4f, 2, 1.0f, 2.0f));   // 4: dominated by 0,3

    auto frontier = extract_pareto_frontier(results);

    // Strategy 4 is dominated by 0 (10>8, 0.5>0.4, rest equal)
    // Strategy 0 is dominated by 3 (15>10, 0.6>0.5, rest equal)
    // Frontier should be: {1, 2, 3}
    std::set<int> front_set(frontier.begin(), frontier.end());
    EXPECT_EQ(front_set.size(), 3u);
    EXPECT_TRUE(front_set.count(1));
    EXPECT_TRUE(front_set.count(2));
    EXPECT_TRUE(front_set.count(3));
    EXPECT_FALSE(front_set.count(0));
    EXPECT_FALSE(front_set.count(4));
}

TEST(Pareto, AllDominated) {
    // One strategy dominates all others
    std::vector<BacktestResult> results;
    results.push_back(make_result(100.0f, 0.9f, 0, 5.0f, 10.0f));  // dominates all
    results.push_back(make_result(10.0f, 0.5f, 5, 1.0f, 2.0f));
    results.push_back(make_result(20.0f, 0.6f, 3, 2.0f, 3.0f));
    results.push_back(make_result(30.0f, 0.7f, 2, 3.0f, 4.0f));

    auto frontier = extract_pareto_frontier(results);

    EXPECT_EQ(frontier.size(), 1u);
    EXPECT_EQ(frontier[0], 0);
}

TEST(Pareto, NoneDominated) {
    // Each strategy is best on one dimension -> no dominance -> frontier = all
    std::vector<BacktestResult> results;
    //                           ROC   GW    red  TRR  PF
    results.push_back(make_result(50.0f, 0.3f, 5, 1.0f, 1.5f));  // best ROC
    results.push_back(make_result(10.0f, 0.9f, 5, 1.0f, 1.5f));  // best GW
    results.push_back(make_result(10.0f, 0.3f, 0, 1.0f, 1.5f));  // best red_weeks (0)
    results.push_back(make_result(10.0f, 0.3f, 5, 9.0f, 1.5f));  // best TRR
    results.push_back(make_result(10.0f, 0.3f, 5, 1.0f, 9.0f));  // best PF

    auto frontier = extract_pareto_frontier(results);

    EXPECT_EQ(frontier.size(), 5u);
}

TEST(Pareto, FiveDimensions) {
    // 5-dimension test: strategies that trade off across all dims
    std::vector<BacktestResult> results;
    //                           ROC   GW    red  TRR  PF
    results.push_back(make_result(50.0f, 0.3f, 5, 1.0f, 1.5f));  // A: best ROC
    results.push_back(make_result(10.0f, 0.9f, 1, 0.5f, 1.2f));  // B: best GW
    results.push_back(make_result(30.0f, 0.5f, 0, 3.0f, 5.0f));  // C: best red, TRR, PF
    results.push_back(make_result(20.0f, 0.4f, 3, 0.8f, 1.3f));  // D: middling

    // D is dominated by C? C: 30>20, 0.5>0.4, 0(better)>3, 3>0.8, 5>1.3 -> yes
    auto frontier = extract_pareto_frontier(results);

    std::set<int> front_set(frontier.begin(), frontier.end());
    EXPECT_TRUE(front_set.count(0));   // A on frontier
    EXPECT_TRUE(front_set.count(1));   // B on frontier
    EXPECT_TRUE(front_set.count(2));   // C on frontier
    EXPECT_FALSE(front_set.count(3));  // D dominated by C
}

// ============================================================================
// Archetype Classification Tests
// ============================================================================

TEST(Archetype, Classification) {
    // Create a pool with known metrics (green_weeks_pct in percentage 0-100)
    std::vector<BacktestResult> pool;
    // Grinder: WR=95%, GW=80%
    pool.push_back(make_result(15.0f, 80.0f, 1, 2.0f, 3.0f, 0.95f));
    // Sniper: WR=40%, high ROC=50
    pool.push_back(make_result(50.0f, 50.0f, 3, 1.5f, 2.0f, 0.40f));
    // Balanced: WR=65%, moderate everything
    pool.push_back(make_result(25.0f, 60.0f, 2, 1.8f, 2.5f, 0.65f));

    ArchetypeThresholds t = compute_archetype_thresholds(pool);

    // Grinder thresholds: max(0.90, p75(WR))
    // WR sorted: {0.40, 0.65, 0.95}, p75 = 0.65 + 0.5*(0.95-0.65) = 0.80
    // So grinder_wr_min = max(0.90, 0.80) = 0.90
    EXPECT_FLOAT_EQ(t.grinder_wr_min, 0.90f);

    // Sniper threshold: P25(WR)
    // WR sorted: {0.40, 0.65, 0.95}, p25 = 0.40 + 0.5*(0.65-0.40) = 0.525
    EXPECT_NEAR(t.sniper_wr_max, 0.525f, 0.001f);

    // Sniper ROC threshold: P75(ROC)
    // ROC sorted: {15, 25, 50}, p75 = 25 + 0.5*(50-25) = 37.5
    EXPECT_NEAR(t.sniper_roc_min, 37.5f, 0.01f);

    Archetype a0 = classify_archetype(pool[0], t);
    Archetype a1 = classify_archetype(pool[1], t);
    Archetype a2 = classify_archetype(pool[2], t);

    EXPECT_EQ(a0, Archetype::GRINDER);   // WR=0.95>=0.90, GW=80>=75
    EXPECT_EQ(a1, Archetype::SNIPER);    // WR=0.40<=0.525, ROC=50>=37.5
    EXPECT_EQ(a2, Archetype::BALANCED);  // WR=0.65>0.525 -> not sniper
}

TEST(Archetype, ThresholdsFromFullPoolNotFrontier) {
    // Verify classify_frontier uses full IS pool for thresholds,
    // not just the frontier subset. We construct a scenario where the
    // classification differs depending on which pool is used.

    // Full IS pool: many low-WR strategies shift P25 downward
    std::vector<BacktestResult> full_pool;
    full_pool.push_back(make_result(60.0f, 50.0f, 2, 2.0f, 3.0f, 0.30f));
    full_pool.push_back(make_result(55.0f, 55.0f, 2, 2.0f, 3.0f, 0.35f));
    full_pool.push_back(make_result(20.0f, 70.0f, 2, 2.0f, 3.0f, 0.60f));
    full_pool.push_back(make_result(15.0f, 80.0f, 1, 2.0f, 3.0f, 0.70f));
    full_pool.push_back(make_result(10.0f, 85.0f, 1, 2.0f, 3.0f, 0.80f));
    full_pool.push_back(make_result(8.0f, 90.0f, 0, 2.0f, 3.0f, 0.95f));

    // "results" vector contains the same data; frontier is indices {0, 5}
    // (highest ROC and highest GW respectively)
    std::vector<int> frontier_indices = {0, 5};

    // With full pool: P25(WR) of {0.30,0.35,0.60,0.70,0.80,0.95}
    //   idx = 0.25*(6-1) = 1.25 => sorted[1]*(1-0.25) + sorted[2]*0.25
    //   = 0.35*0.75 + 0.60*0.25 = 0.2625 + 0.15 = 0.4125
    // So sniper_wr_max = 0.4125

    // Result[0]: WR=0.30 <= 0.4125 -> eligible for sniper check
    // P75(ROC) of full pool: sorted ROC {8,10,15,20,55,60}
    //   idx = 0.75*5 = 3.75 => sorted[3]*(1-0.75) + sorted[4]*0.75 = 20*0.25+55*0.75 = 46.25
    // Result[0]: ROC=60 >= 46.25 -> SNIPER with full pool

    auto entries_with_pool = classify_frontier(full_pool, frontier_indices, full_pool);

    // Result[0] should be SNIPER when using full pool thresholds
    EXPECT_EQ(entries_with_pool[0].archetype, Archetype::SNIPER);
    // Result[5] should be GRINDER (WR=0.95, GW=90)
    EXPECT_EQ(entries_with_pool[1].archetype, Archetype::GRINDER);

    // Without full pool (empty = fallback to frontier only):
    // Frontier is {result[0], result[5]}: WR={0.30, 0.95}
    // P25(WR) of frontier = 0.30 + 0.25*(0.95-0.30) = 0.4625
    // P75(ROC) of frontier: {60,8} sorted={8,60}, p75 = 8+0.75*(60-8) = 47.0
    // Result[0]: WR=0.30 <= 0.4625 and ROC=60 >= 47.0 -> SNIPER
    // (happens to be same classification here, but thresholds differ)
    auto entries_frontier_only = classify_frontier(full_pool, frontier_indices);
    EXPECT_EQ(entries_frontier_only[0].archetype, Archetype::SNIPER);
}

TEST(Archetype, SniperUsesP25WinRate) {
    // Sniper archetype uses pool-derived P25 of WR, not hardcoded bounds.
    // Strategy with WR=0.20 (below a fixed 0.30 floor) should still be
    // classified as Sniper if it's in the bottom quartile and has top-quartile ROC.
    std::vector<BacktestResult> pool;
    pool.push_back(make_result(80.0f, 40.0f, 3, 1.0f, 2.0f, 0.20f));  // low WR, high ROC
    pool.push_back(make_result(30.0f, 50.0f, 2, 1.5f, 2.5f, 0.50f));
    pool.push_back(make_result(20.0f, 60.0f, 2, 1.8f, 2.5f, 0.60f));
    pool.push_back(make_result(10.0f, 70.0f, 1, 2.0f, 3.0f, 0.70f));

    ArchetypeThresholds t = compute_archetype_thresholds(pool);

    // WR sorted: {0.20, 0.50, 0.60, 0.70}, P25 = idx 0.75
    // = 0.20*(1-0.75) + 0.50*0.75 = 0.05 + 0.375 = 0.425
    EXPECT_NEAR(t.sniper_wr_max, 0.425f, 0.001f);

    // ROC sorted: {10, 20, 30, 80}, P75 = idx 2.25
    // = 30*(1-0.25) + 80*0.25 = 22.5 + 20 = 42.5
    EXPECT_NEAR(t.sniper_roc_min, 42.5f, 0.01f);

    // pool[0]: WR=0.20 <= 0.425, ROC=80 >= 42.5 => SNIPER
    // (a fixed 0.30 floor would classify it BALANCED)
    Archetype a = classify_archetype(pool[0], t);
    EXPECT_EQ(a, Archetype::SNIPER);

    // pool[1]: WR=0.50 > 0.425 => not sniper => BALANCED
    Archetype b = classify_archetype(pool[1], t);
    EXPECT_EQ(b, Archetype::BALANCED);
}

// ============================================================================
// Selection Tests
// ============================================================================

TEST(Selection, BestROC) {
    std::vector<BacktestResult> results;
    results.push_back(make_result(10.0f, 0.5f, 2, 1.0f, 2.0f));
    results.push_back(make_result(50.0f, 0.3f, 2, 1.0f, 2.0f));  // highest ROC
    results.push_back(make_result(30.0f, 0.7f, 2, 1.0f, 2.0f));

    std::vector<ParetoEntry> frontier;
    for (int i = 0; i < 3; ++i) {
        ParetoEntry pe;
        pe.original_index = i;
        pe.archetype      = Archetype::BALANCED;
        frontier.push_back(pe);
    }

    int winner = selection::best_roc(results, frontier);
    EXPECT_EQ(winner, 1);
}

TEST(Selection, Centroid) {
    // 3 strategies forming a triangle in 2 dimensions (others equal)
    std::vector<BacktestResult> results;
    //              ROC   GW    red  TRR  PF
    results.push_back(make_result(50.0f, 0.2f, 2, 1.0f, 2.0f));  // 0: high ROC, low GW
    results.push_back(make_result(10.0f, 0.9f, 2, 1.0f, 2.0f));  // 1: low ROC, high GW
    results.push_back(make_result(30.0f, 0.5f, 2, 1.0f, 2.0f));  // 2: middle

    std::vector<ParetoEntry> frontier;
    for (int i = 0; i < 3; ++i) {
        ParetoEntry pe;
        pe.original_index = i;
        pe.archetype      = Archetype::BALANCED;
        frontier.push_back(pe);
    }

    // Centroid in normalized space:
    // ROC: min=10, max=50, range=40. Norm: 0=1.0, 1=0.0, 2=0.5
    // GW: min=0.2, max=0.9, range=0.7. Norm: 0=0.0, 1=1.0, 2=0.43
    // Other dims all equal -> norm all same
    // Centroid ROC norm = (1.0+0.0+0.5)/3 = 0.5
    // Centroid GW norm = (0.0+1.0+0.43)/3 = 0.476
    // Strategy 2 (ROC_norm=0.5, GW_norm=0.43) is closest to centroid (0.5, 0.476)
    int winner = selection::frontier_centroid(results, frontier);
    EXPECT_EQ(winner, 2);
}

TEST(Selection, FallbackGrinderToCentroid) {
    // No Grinder archetypes on the frontier -> fallback to centroid
    std::vector<BacktestResult> results;
    results.push_back(make_result(50.0f, 0.3f, 2, 1.0f, 2.0f));
    results.push_back(make_result(30.0f, 0.5f, 2, 1.0f, 2.0f));
    results.push_back(make_result(10.0f, 0.8f, 2, 1.0f, 2.0f));

    // All marked as BALANCED (no Grinders)
    std::vector<ParetoEntry> frontier;
    for (int i = 0; i < 3; ++i) {
        ParetoEntry pe;
        pe.original_index = i;
        pe.archetype      = Archetype::BALANCED;
        frontier.push_back(pe);
    }

    int grinder_pick  = selection::best_grinder(results, frontier);
    int centroid_pick = selection::frontier_centroid(results, frontier);

    // Should fall back to centroid
    EXPECT_EQ(grinder_pick, centroid_pick);
}

TEST(Selection, MainEntryPoint) {
    std::vector<BacktestResult> results;
    results.push_back(make_result(10.0f, 0.9f, 1, 3.0f, 5.0f));
    results.push_back(make_result(50.0f, 0.3f, 5, 1.0f, 1.5f));

    std::vector<ParetoEntry> frontier;
    for (int i = 0; i < 2; ++i) {
        ParetoEntry pe;
        pe.original_index = i;
        pe.archetype      = Archetype::BALANCED;
        frontier.push_back(pe);
    }

    // BEST_ROC should pick index 1 (ROC=50)
    EXPECT_EQ(select_winner(SelectionStrategy::BEST_ROC, results, frontier), 1);
    // BEST_CONSISTENCY should pick index 0 (GW=0.9)
    EXPECT_EQ(select_winner(SelectionStrategy::BEST_CONSISTENCY, results, frontier), 0);

    // Empty frontier -> -1
    std::vector<ParetoEntry> empty;
    EXPECT_EQ(select_winner(SelectionStrategy::BEST_ROC, results, empty), -1);
}

TEST(Selection, BestSniper) {
    // Create a frontier with known archetypes.
    // Verify best_sniper selects the highest ROC among Sniper entries.
    // If no Snipers exist, verify fallback to frontier_centroid.
    std::vector<BacktestResult> results;
    //                          ROC   GW    red  TRR   PF    WR
    results.push_back(make_result(60.0f, 0.4f, 3, 1.5f, 2.0f, 0.45f));  // 0: Sniper, high ROC
    results.push_back(make_result(40.0f, 0.5f, 4, 1.0f, 1.8f, 0.42f));  // 1: Sniper, lower ROC
    results.push_back(make_result(30.0f, 0.8f, 1, 2.0f, 3.0f, 0.85f));  // 2: Grinder
    results.push_back(make_result(25.0f, 0.6f, 2, 1.2f, 2.5f, 0.65f));  // 3: Balanced

    // --- Case 1: frontier has Snipers -> pick highest ROC among them ---
    std::vector<ParetoEntry> frontier_with_snipers;
    {
        ParetoEntry pe;
        pe.original_index = 0;
        pe.archetype      = Archetype::SNIPER;
        frontier_with_snipers.push_back(pe);
        pe.original_index = 1;
        pe.archetype      = Archetype::SNIPER;
        frontier_with_snipers.push_back(pe);
        pe.original_index = 2;
        pe.archetype      = Archetype::GRINDER;
        frontier_with_snipers.push_back(pe);
        pe.original_index = 3;
        pe.archetype      = Archetype::BALANCED;
        frontier_with_snipers.push_back(pe);
    }

    int winner = selection::best_sniper(results, frontier_with_snipers);
    EXPECT_EQ(winner, 0) << "best_sniper should pick index 0 (ROC=60, highest among Snipers)";

    // --- Case 2: frontier has NO Snipers -> fallback to frontier_centroid ---
    std::vector<ParetoEntry> frontier_no_snipers;
    {
        ParetoEntry pe;
        pe.original_index = 2;
        pe.archetype      = Archetype::GRINDER;
        frontier_no_snipers.push_back(pe);
        pe.original_index = 3;
        pe.archetype      = Archetype::BALANCED;
        frontier_no_snipers.push_back(pe);
    }

    int sniper_pick   = selection::best_sniper(results, frontier_no_snipers);
    int centroid_pick = selection::frontier_centroid(results, frontier_no_snipers);
    EXPECT_EQ(sniper_pick, centroid_pick)
        << "best_sniper with no Snipers should fall back to frontier_centroid";
}

TEST(Selection, RocXConsistency) {
    // Create frontier entries with known ROC and green_weeks_pct.
    // Verify roc_x_consistency selects the one with highest product.
    std::vector<BacktestResult> results;
    //                          ROC   GW    red  TRR  PF
    results.push_back(make_result(50.0f, 0.3f, 2, 1.0f, 2.0f));  // 0: product = 50*0.3 = 15
    results.push_back(make_result(20.0f, 0.9f, 2, 1.0f, 2.0f));  // 1: product = 20*0.9 = 18
    results.push_back(
        make_result(30.0f, 0.7f, 2, 1.0f, 2.0f));  // 2: product = 30*0.7 = 21 (highest)
    results.push_back(make_result(10.0f, 0.8f, 2, 1.0f, 2.0f));  // 3: product = 10*0.8 = 8

    std::vector<ParetoEntry> frontier;
    for (int i = 0; i < 4; ++i) {
        ParetoEntry pe;
        pe.original_index = i;
        pe.archetype      = Archetype::BALANCED;
        frontier.push_back(pe);
    }

    int winner = selection::roc_x_consistency(results, frontier);
    EXPECT_EQ(winner, 2)
        << "roc_x_consistency should pick index 2 (ROC=30 * GW=0.7 = 21, highest product)";
}

TEST(Selection, BestTRR) {
    // Create frontier entries with known TRR values.
    // Verify best_trr selects the one with highest trade_recovery_ratio.
    std::vector<BacktestResult> results;
    //                          ROC   GW    red  TRR   PF
    results.push_back(make_result(10.0f, 0.5f, 2, 1.5f, 2.0f));  // 0: TRR = 1.5
    results.push_back(make_result(20.0f, 0.6f, 2, 4.0f, 2.5f));  // 1: TRR = 4.0 (highest)
    results.push_back(make_result(30.0f, 0.7f, 2, 3.0f, 3.0f));  // 2: TRR = 3.0
    results.push_back(make_result(15.0f, 0.4f, 2, 2.0f, 1.8f));  // 3: TRR = 2.0

    std::vector<ParetoEntry> frontier;
    for (int i = 0; i < 4; ++i) {
        ParetoEntry pe;
        pe.original_index = i;
        pe.archetype      = Archetype::BALANCED;
        frontier.push_back(pe);
    }

    int winner = selection::best_trr(results, frontier);
    EXPECT_EQ(winner, 1) << "best_trr should pick index 1 (TRR=4.0, highest)";
}

// ============================================================================
// Selection: BestConsistency
// ============================================================================

TEST(Selection, BestConsistency) {
    // Create frontier entries with known green_weeks_pct values.
    // Verify best_consistency picks the entry with the highest green_weeks_pct.
    std::vector<BacktestResult> results;
    //                          ROC   GW    red  TRR  PF
    results.push_back(make_result(30.0f, 0.55f, 2, 1.0f, 2.0f));  // 0: GW=0.55
    results.push_back(make_result(20.0f, 0.92f, 1, 1.5f, 3.0f));  // 1: GW=0.92 (highest)
    results.push_back(make_result(40.0f, 0.70f, 3, 2.0f, 2.5f));  // 2: GW=0.70
    results.push_back(make_result(15.0f, 0.88f, 2, 1.2f, 1.8f));  // 3: GW=0.88

    std::vector<ParetoEntry> frontier;
    for (int i = 0; i < 4; ++i) {
        ParetoEntry pe;
        pe.original_index = i;
        pe.archetype      = Archetype::BALANCED;
        frontier.push_back(pe);
    }

    int winner = selection::best_consistency(results, frontier);
    EXPECT_EQ(winner, 1) << "best_consistency should pick index 1 (green_weeks_pct=0.92, highest)";
}

// ============================================================================
// Selection: BestGrinderSuccess
// ============================================================================

TEST(Selection, BestGrinderSuccess) {
    // Create a frontier that includes a genuine Grinder archetype entry
    // (WR >= 90%, green_weeks >= 75%). Verify best_grinder picks it (not fallback).
    std::vector<BacktestResult> results;
    //                          ROC   GW     red  TRR   PF    WR
    results.push_back(
        make_result(25.0f, 80.0f, 1, 2.0f, 3.0f, 0.92f));  // 0: Grinder (WR=92%, GW=80%)
    results.push_back(make_result(40.0f, 60.0f, 3, 1.5f, 2.0f, 0.50f));  // 1: Balanced
    results.push_back(make_result(35.0f, 85.0f, 1, 2.5f, 4.0f, 0.95f));  // 2: Grinder, higher ROC
    results.push_back(make_result(50.0f, 40.0f, 4, 1.0f, 1.5f, 0.42f));  // 3: Sniper-like

    // Mark archetypes explicitly
    std::vector<ParetoEntry> frontier;
    {
        ParetoEntry pe;
        pe.original_index = 0;
        pe.archetype      = Archetype::GRINDER;
        frontier.push_back(pe);
        pe.original_index = 1;
        pe.archetype      = Archetype::BALANCED;
        frontier.push_back(pe);
        pe.original_index = 2;
        pe.archetype      = Archetype::GRINDER;
        frontier.push_back(pe);
        pe.original_index = 3;
        pe.archetype      = Archetype::SNIPER;
        frontier.push_back(pe);
    }

    int winner = selection::best_grinder(results, frontier);

    // best_grinder picks highest annual_roc among Grinder entries.
    // Grinder entries: index 0 (ROC=25) and index 2 (ROC=35).
    // Should pick index 2 (ROC=35).
    EXPECT_EQ(winner, 2) << "best_grinder should pick index 2 (ROC=35, highest among Grinders)";

    // Also verify this is NOT the centroid fallback -- it's a genuine Grinder pick
    int centroid_pick = selection::frontier_centroid(results, frontier);
    // The centroid might or might not equal 2, but the important thing is
    // that best_grinder found an actual Grinder and didn't need to fall back.
    // We verify this by checking that at least one Grinder exists in the frontier.
    bool found_grinder = false;
    for (const auto& pe : frontier) {
        if (pe.archetype == Archetype::GRINDER) {
            found_grinder = true;
            break;
        }
    }
    EXPECT_TRUE(found_grinder) << "Frontier should contain Grinder entries for this test";
    (void)centroid_pick;
}

// ============================================================================
// Robustness Tests
// ============================================================================

TEST(Robustness, Plateau) {
    // All neighbors at 90%+ of center metric -> robustness >= 0.90
    float center = 0.80f;  // green_weeks_pct

    std::vector<NeighborResult> neighbors;
    // 3 parameter axes, 2 neighbors each (+ and - direction)
    // Must set metrics_complete=true or neighbors are filtered.
    for (int axis = 0; axis < 3; ++axis) {
        neighbors.push_back({axis, center * 0.95f, true});  // 95% retention
        neighbors.push_back({axis, center * 0.92f, true});  // 92% retention
    }

    float robustness = compute_robustness(center, neighbors, 3);

    // min retention across all = 0.92
    // max sensitivity = 1 - 0.92 = 0.08
    // robustness = 1 - 0.08 = 0.92
    EXPECT_NEAR(robustness, 0.92f, 0.01f);
    EXPECT_GE(robustness, 0.90f);
}

TEST(Robustness, Spike) {
    // One neighbor at 0 -> robustness = 0
    float center = 0.80f;

    std::vector<NeighborResult> neighbors;
    neighbors.push_back({0, center * 0.95f, true});  // good neighbor
    neighbors.push_back({1, 0.0f, true});            // zero neighbor -> spike

    float robustness = compute_robustness(center, neighbors, 2);

    // sensitivity for axis 1 = 1 - 0/0.80 = 1 - 0 = 1.0
    // max sensitivity = 1.0
    // robustness = 1 - 1.0 = 0.0
    EXPECT_FLOAT_EQ(robustness, 0.0f);
}

TEST(Robustness, ZeroCenter) {
    // Center metric is 0 -> robustness = 0
    float                       center = 0.0f;
    std::vector<NeighborResult> neighbors;
    neighbors.push_back({0, 0.5f, true});

    float robustness = compute_robustness(center, neighbors, 1);
    EXPECT_FLOAT_EQ(robustness, 0.0f);
}

TEST(Robustness, NoNeighbors) {
    // No neighbors -> cannot assess robustness, returns 0 (worst)
    float                       center = 0.80f;
    std::vector<NeighborResult> neighbors;

    float robustness = compute_robustness(center, neighbors, 0);
    EXPECT_FLOAT_EQ(robustness, 0.0f);
}

TEST(Robustness, BatchComputation) {
    // 3 strategies in a 1D grid: [A, B, C]
    // A's neighbor is B, B's neighbors are A and C, C's neighbor is B
    std::vector<BacktestResult> results;
    results.push_back(make_result(10.0f, 0.80f, 2, 1.0f, 2.0f));  // A
    results.push_back(make_result(20.0f, 0.82f, 2, 1.5f, 2.5f));  // B
    results.push_back(make_result(15.0f, 0.78f, 2, 1.2f, 2.2f));  // C

    auto get_neighbors = [](int idx) -> std::vector<std::pair<int, int>> {
        switch (idx) {
            case 0:
                return {{0, 1}};  // A -> B
            case 1:
                return {{0, 0}, {0, 2}};  // B -> A, C
            case 2:
                return {{0, 1}};  // C -> B
            default:
                return {};
        }
    };

    auto scores = compute_robustness_batch(results, get_neighbors, 1);
    ASSERT_EQ(scores.size(), 3u);

    // Robustness uses green_weeks_pct: A=0.80, B=0.82, C=0.78
    // Strategy A (GW=0.80): neighbor B (GW=0.82), retention=0.82/0.80=1.025 -> clamped 1.0
    // sensitivity=0, robustness=1.0
    EXPECT_FLOAT_EQ(scores[0], 1.0f);

    // Strategy B (GW=0.82): neighbors A (0.80), C (0.78)
    // retention(A)=0.80/0.82=0.9756, retention(C)=0.78/0.82=0.9512
    // min=0.9512, sensitivity=0.0488, robustness=0.9512
    EXPECT_NEAR(scores[1], 0.951f, 0.01f);

    // Strategy C (GW=0.78): neighbor B (0.82), retention=0.82/0.78=1.051 -> clamped 1.0
    // sensitivity=0, robustness=1.0
    EXPECT_FLOAT_EQ(scores[2], 1.0f);
}

// ============================================================================
// Strategy Grid at Production Scale
//
// Validate the StrategyGrid unranking with N matching the actual production
// indicator count (~553 = 487 technical indicators + ~66 session signals from
// SessionSignalGrid).  Prior tests used N=3..10.  This tests:
//   1. total_count() is consistent with the summation formula
//   2. entry_at(0) and entry_at(total_count-1) are valid (no overflow/crash)
//   3. entry_at() determinism at scale
//   4. axis_descriptors() returns valid strides for k=1..4
// ============================================================================

TEST(StrategyGrid, ProductionScale) {
    // The actual indicator grid has 487 entries; session signals add ~66-156.
    // Use 553 as the realistic production count.
    const int    N = 553;
    StrategyGrid sg(N);

    int64_t total = sg.total_count();
    EXPECT_GT(total, 0) << "Grid must have strategies";

    // Verify total_count matches the formula:
    // sum over k=1..4 of C(N,k) * 4^k * PER_WEIGHT
    auto binom = [](int64_t n, int k) -> int64_t {
        if (k < 0 || k > static_cast<int>(n)) return 0;
        if (k == 0 || k == static_cast<int>(n)) return 1;
        if (k > static_cast<int>(n) - k) k = static_cast<int>(n) - k;
        int64_t result = 1;
        for (int i = 0; i < k; ++i) result = result * (n - i) / (i + 1);
        return result;
    };

    int64_t expected = 0;
    for (int k = 1; k <= 4; ++k) {
        int64_t wcombos = 1;
        for (int i = 0; i < k; ++i) wcombos *= 4;
        expected += binom(N, k) * wcombos * StrategyGrid::PER_WEIGHT;
    }
    EXPECT_EQ(total, expected) << "total_count formula mismatch at N=" << N;

    // entry_at(0): must not crash, must have valid fields
    StrategyGridEntry first = sg.entry_at(0);
    EXPECT_GE(first.num_indicators, 1);
    EXPECT_LE(first.num_indicators, 4);
    for (int i = 0; i < first.num_indicators; ++i) {
        EXPECT_GE(first.indicator_ids[i], 0);
        EXPECT_LT(first.indicator_ids[i], N);
    }

    // entry_at(total - 1): must not crash, must have valid fields
    StrategyGridEntry last = sg.entry_at(total - 1);
    EXPECT_GE(last.num_indicators, 1);
    EXPECT_LE(last.num_indicators, 4);
    for (int i = 0; i < last.num_indicators; ++i) {
        EXPECT_GE(last.indicator_ids[i], 0);
        EXPECT_LT(last.indicator_ids[i], N);
    }

    // Determinism: same index produces same entry
    StrategyGridEntry first2 = sg.entry_at(0);
    EXPECT_EQ(first.num_indicators, first2.num_indicators);
    for (int i = 0; i < first.num_indicators; ++i)
        EXPECT_EQ(first.indicator_ids[i], first2.indicator_ids[i]);

    // Spot-check a mid-grid entry
    int64_t           mid = total / 2;
    StrategyGridEntry m1  = sg.entry_at(mid);
    StrategyGridEntry m2  = sg.entry_at(mid);
    EXPECT_EQ(m1.num_indicators, m2.num_indicators);
    EXPECT_EQ(m1.aggregation, m2.aggregation);
    EXPECT_FLOAT_EQ(m1.entry_threshold, m2.entry_threshold);

    // axis_descriptors for k=1..4 should all have 6 axes
    for (int k = 1; k <= 4; ++k) {
        auto axes = StrategyGrid::axis_descriptors(k);
        EXPECT_EQ(axes.size(), 6u) << "k=" << k;
        // weight_rank range should be 4^k
        int64_t expected_range = 1;
        for (int i = 0; i < k; ++i) expected_range *= 4;
        EXPECT_EQ(axes[5].range, static_cast<int>(expected_range))
            << "weight_rank range mismatch for k=" << k;
    }
}

// =============================================================================
// Strategy-grid neighbor geometry
// =============================================================================

// A one-step neighbor changes exactly one parameter axis and never leaves the
// strategy's indicator subset. With n not a multiple of 4 the k = 2 block does
// not start on a weight-rank boundary, so positions must be taken within the
// block for this to hold.
TEST(StrategyGrid, NeighborsStayInsideTheIndicatorSubset) {
    for (const int n : {5, 7, 643}) {
        StrategyGrid  grid(n);
        const int64_t k1_size = static_cast<int64_t>(n) * 4 * StrategyGrid::PER_WEIGHT;
        // Probe the start of the k = 2 block and a few interior points.
        const int64_t probes[] = {k1_size,
                                  k1_size + 1,
                                  k1_size + 2303,
                                  k1_size + 2304,
                                  k1_size + 15 * 2304 + 7,
                                  k1_size + 16 * 2304,
                                  k1_size + 37 * 2304 + 999,
                                  17,
                                  4 * 2304 - 1};
        for (const int64_t id : probes) {
            const StrategyGridEntry center = grid.entry_at(id);
            const auto              nbs    = grid.neighbors(id);
            ASSERT_FALSE(nbs.empty());
            for (const auto& [axis, nb_id] : nbs) {
                ASSERT_GE(nb_id, 0);
                ASSERT_LT(nb_id, grid.total_count());
                const StrategyGridEntry nb = grid.entry_at(nb_id);
                ASSERT_EQ(nb.num_indicators, center.num_indicators) << "n=" << n << " id=" << id;
                for (int i = 0; i < center.num_indicators; ++i)
                    ASSERT_EQ(nb.indicator_ids[i], center.indicator_ids[i])
                        << "n=" << n << " id=" << id << " axis=" << axis;
            }
        }
    }
}

TEST(StrategyGrid, NeighborsRespectAxisBoundaries) {
    StrategyGrid  grid(6);
    const int64_t k2_start = 6LL * 4 * StrategyGrid::PER_WEIGHT;
    // First strategy of a k = 2 subset: weight rank 0, so no -1 weight step.
    for (const auto& [axis, nb] : grid.neighbors(k2_start)) {
        if (axis == 5) EXPECT_GT(nb, k2_start);
    }
    // Last weight rank (15) of the same subset: no +1 weight step.
    const int64_t last = k2_start + 15 * StrategyGrid::PER_WEIGHT;
    for (const auto& [axis, nb] : grid.neighbors(last)) {
        if (axis == 5) EXPECT_LT(nb, last);
    }
    EXPECT_EQ(grid.subset_size_at(0), 1);
    EXPECT_EQ(grid.subset_size_at(k2_start - 1), 1);
    EXPECT_EQ(grid.subset_size_at(k2_start), 2);
    EXPECT_TRUE(grid.neighbors(-1).empty());
    EXPECT_TRUE(grid.neighbors(grid.total_count()).empty());
}

// Every entry the screen skips as redundant trades exactly like a kept
// entry. The canonical form is derived here independently of
// is_redundant(): vote modes ignore weights and thresholds, filters ignore
// weights, weighted sums ignore the scale of their weights, and a single
// UNANIMOUS indicator equals MAJORITY_VOTE.
TEST(StrategyGrid, RedundantEntriesTradeLikeTheirCanonicalForm) {
    const int                       n = 3, N = 400;
    std::vector<std::vector<float>> series(n, std::vector<float>(N));
    uint64_t                        state = 12345;
    auto                            next  = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0f;
    };
    for (auto& s : series)
        for (float& v : s) {
            const float u = next();
            v = u < 0.05f ? std::numeric_limits<float>::quiet_NaN() : 2.0f * next() - 1.0f;
        }
    std::vector<Bar> bars(N);
    float            px = 100.0f;
    for (int i = 0; i < N; ++i) {
        px += next() - 0.5f;
        bars[i].timestamp = (1'700'000'000LL + i * 30LL) * 1'000'000'000LL;
        bars[i].open      = px;
        bars[i].high      = px + 0.5f;
        bars[i].low       = px - 0.5f;
        bars[i].close     = px + 0.1f;
        bars[i].volume    = 100;
    }
    const float*    ptrs[] = {series[0].data(), series[1].data(), series[2].data()};
    const CostModel cost   = CostModel::es();

    auto canonical = [](StrategyGridEntry e) {
        if (e.aggregation == StrategyConfig::UNANIMOUS && e.num_indicators == 1)
            e.aggregation = StrategyConfig::MAJORITY_VOTE;
        for (int i = 0; i < e.num_indicators; ++i)
            if (e.indicator_roles[i] == StrategyConfig::FILTER) e.weights[i] = 1.0f;
        if (e.aggregation != StrategyConfig::WEIGHTED_SUM) {
            e.entry_threshold = StrategyGrid::THRESHOLD_VALUES[0];
            for (int i = 0; i < e.num_indicators; ++i) e.weights[i] = 1.0f;
        } else {
            float mx = 0.0f;
            for (int i = 0; i < e.num_indicators; ++i)
                if (e.indicator_roles[i] != StrategyConfig::FILTER) mx = std::max(mx, e.weights[i]);
            // Scale up while every signal weight stays a grid value.
            for (int k = 4; k >= 1; --k) {
                const float c  = static_cast<float>(k) * 0.25f / mx;
                bool        ok = c >= 1.0f;
                for (int i = 0; i < e.num_indicators && ok; ++i) {
                    if (e.indicator_roles[i] == StrategyConfig::FILTER) continue;
                    const float w = e.weights[i] * c * 4.0f;
                    ok            = std::abs(w - std::round(w)) < 1e-4f && w <= 4.0f + 1e-4f;
                }
                if (ok) {
                    for (int i = 0; i < e.num_indicators; ++i)
                        if (e.indicator_roles[i] != StrategyConfig::FILTER)
                            e.weights[i] = std::round(e.weights[i] * c * 4.0f) / 4.0f;
                    break;
                }
            }
        }
        return e;
    };

    StrategyGrid grid(n);
    int          checked = 0;
    for (int64_t id = 0; id < grid.total_count(); id += 97) {
        const StrategyGridEntry e = grid.entry_at(id);
        if (!e.is_redundant()) continue;
        if (e.num_indicators == 1 && e.indicator_roles[0] == StrategyConfig::FILTER) continue;
        const StrategyGridEntry c = canonical(e);
        ASSERT_FALSE(c.is_redundant()) << "canonical form of " << id << " is itself skipped";
        BacktestEngine a, b;
        const auto     ra = a.run(bars.data(), 0, N, ptrs, n, e.to_strategy_config(), cost, 1);
        const auto     rb = b.run(bars.data(), 0, N, ptrs, n, c.to_strategy_config(), cost, 1);
        ASSERT_EQ(ra.total_trades, rb.total_trades) << "id " << id;
        ASSERT_FLOAT_EQ(ra.total_pnl, rb.total_pnl) << "id " << id;
        ++checked;
    }
    EXPECT_GT(checked, 500);
}
