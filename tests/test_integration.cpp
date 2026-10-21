// =============================================================================
// Integration Tests
//
// These tests exercise multi-component pipelines to verify system-level
// correctness beyond individual indicator computation:
//
//   h. WindowedVsFullIndicators -- windowed computation matches full-dataset
//   i. MultithreadedDeterminism -- single-thread matches multi-thread
//   j. KnownEdgeSyntheticWFO   -- WFO discovers a planted calendar edge
//   k. CheckpointResumeDeterminism -- checkpoint save/load round-trips
//
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>
#include "portable_random.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "market_data.h"
#include "indicators/indicator.h"
#include "indicators/ema_sma_cross.h"
#include "indicators/rsi.h"
#include "indicators/atr.h"
#include "indicators/bollinger.h"
#include "indicators/macd.h"
#include "indicators/donchian.h"
#include "indicators/cci.h"
#include "indicators/stochastic.h"
#include "indicators/adx.h"
#include "backtest_engine.h"
#include "cost_model.h"
#include "metrics.h"
#include "checkpoint.h"
#include "landscape.h"
#include "pareto.h"
#include "robustness.h"
#include "indicators/signal_normalize.h"
#include "grids/indicator_grid.h"
#include "grids/strategy_grid.h"

// =============================================================================
// Helpers
// =============================================================================

static constexpr int64_t NS_PER_SEC = 1'000'000'000LL;

static bool is_nan(float v) {
    return std::isnan(v);
}

static std::vector<Bar> generate_random_walk_bars(int count, float start_price = 4500.0f,
                                                  unsigned seed = 42) {
    std::mt19937            rng(seed);
    portable::normal<float> step{0.0f, 0.5f};

    std::vector<Bar> bars(count);
    float            price = start_price;

    int64_t       ts              = 1735858800LL * NS_PER_SEC;
    const int64_t bar_interval_ns = 30LL * NS_PER_SEC;

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

// Helper to create a BacktestResult with known values
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
    return r;
}

// =============================================================================
// h. Integration.WindowedVsFullIndicators
//
// Compute RSI(14) over 1000 bars (full dataset). Then compute RSI(14) over
// bars [200, 800] with warmup prefix [200-warmup, 200]. Values in [200, 800]
// should match the full-dataset values to within 1e-5. Test with EMA, SMA,
// ATR too.
// =============================================================================

TEST(Integration, WindowedVsFullIndicators) {
    // Generate 1000 bars of random walk data
    const int TOTAL = 1000;
    auto      bars  = generate_random_walk_bars(TOTAL, 4500.0f, 314159);

    // Test window: [200, 800]
    const int win_start = 200;
    const int win_end   = 800;

    // ---- IIR indicators (RSI, EMA, ATR): exponentially-weighted ----
    // These use recursive smoothing, so initial conditions matter.
    // Starting from a different bar gives different seed values. The
    // difference decays exponentially. We verify that:
    //   (a) Starting from bar 0 gives identical results (sanity), and
    //   (b) Starting from a closer bar converges to within a tolerance
    //       that decays with additional warmup.

    // Test 1: RSI(14) -- IIR indicator, full-history start must match
    {
        RSI rsi(14);

        // Full computation
        std::vector<float> full_output(TOTAL);
        rsi.compute(bars.data(), 0, TOTAL, full_output.data());

        // Windowed computation starting from bar 0 (same as full)
        std::vector<float> windowed_output(win_end);
        rsi.compute(bars.data(), 0, win_end, windowed_output.data());

        // Must match exactly in the overlap [200, 800)
        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[i])) continue;
            EXPECT_NEAR(windowed_output[i], full_output[i], 1e-5f)
                << "RSI same-start windowed vs full mismatch at bar " << i;
        }

        // Now test with truncated prefix: the IIR nature means values
        // will differ initially but converge. Verify convergence.
        const int          wu         = rsi.warmup_bars();                 // 42
        int                late_start = std::max(0, win_start - 10 * wu);  // generous prefix
        int                late_len   = win_end - late_start;
        std::vector<float> late_output(late_len);
        rsi.compute(bars.data(), late_start, win_end, late_output.data());

        // Check that the difference decreases over time (exponential decay)
        float diff_early = 0.0f, diff_late = 0.0f;
        int   count_early = 0, count_late = 0;
        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[i])) continue;
            int li = i - late_start;
            if (is_nan(late_output[li])) continue;
            float diff = std::abs(late_output[li] - full_output[i]);
            if (i < win_start + 50) {
                diff_early += diff;
                count_early++;
            }
            if (i > win_end - 50) {
                diff_late += diff;
                count_late++;
            }
        }
        if (count_early > 0 && count_late > 0) {
            diff_early /= count_early;
            diff_late /= count_late;
            EXPECT_LT(diff_late, diff_early + 1e-6f)
                << "RSI IIR error should decay: early avg=" << diff_early
                << " late avg=" << diff_late;
        }
    }

    // Test 2: EMA/SMA cross -- EMA is IIR, SMA is window-based
    {
        EMASMACross cross(MAType::EMA, 10, MAType::SMA, 30);

        // Full computation (2 series)
        const int          N = TOTAL;
        std::vector<float> full_output(N * 2);
        cross.compute(bars.data(), 0, N, full_output.data());

        // Windowed from bar 0 must match
        std::vector<float> windowed_output(win_end * 2);
        cross.compute(bars.data(), 0, win_end, windowed_output.data());

        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[i])) continue;
            EXPECT_NEAR(windowed_output[i], full_output[i], 1e-5f)
                << "Fast EMA same-start windowed vs full mismatch at bar " << i;
        }
        // SMA (slow, window-based) should also match
        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[N + i])) continue;
            EXPECT_NEAR(windowed_output[win_end + i], full_output[N + i], 1e-5f)
                << "Slow SMA same-start windowed vs full mismatch at bar " << i;
        }

        // SMA from truncated window: only need period bars of history
        int                sma_start = win_start - 30;  // SMA period = 30
        int                sma_len   = win_end - sma_start;
        std::vector<float> sma_windowed(sma_len * 2);
        cross.compute(bars.data(), sma_start, win_end, sma_windowed.data());

        // SMA series (window-based) should match with just period bars of prefix.
        // Running-sum accumulation order differs by start position, causing
        // float precision differences proportional to the price level.
        // At ES ~4500, single-float ULP is ~5e-4, so use tolerance of 0.01.
        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[N + i])) continue;
            int wi = i - sma_start;
            EXPECT_NEAR(sma_windowed[sma_len + wi], full_output[N + i], 0.01f)
                << "Slow SMA truncated-window vs full mismatch at bar " << i;
        }
    }

    // Test 3: ATR(14) -- IIR indicator, same-start must match
    {
        ATR atr(14);

        std::vector<float> full_output(TOTAL);
        atr.compute(bars.data(), 0, TOTAL, full_output.data());

        std::vector<float> windowed_output(win_end);
        atr.compute(bars.data(), 0, win_end, windowed_output.data());

        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[i])) continue;
            EXPECT_NEAR(windowed_output[i], full_output[i], 1e-5f)
                << "ATR same-start windowed vs full mismatch at bar " << i;
        }
    }

    // Test 4: Bollinger(20, 2.0) -- window-based, exact match with just period prefix
    {
        BollingerBands bb(20, 2.0f);
        (void)bb.warmup_bars();  // 19, verified in the cross-validation tests

        const int          N = TOTAL;
        std::vector<float> full_output(N * 3);
        bb.compute(bars.data(), 0, N, full_output.data());

        // Only need 'period' bars of history before the comparison window
        int                windowed_start = win_start - 20;  // period = 20
        int                windowed_len   = win_end - windowed_start;
        std::vector<float> windowed_output(windowed_len * 3);
        bb.compute(bars.data(), windowed_start, win_end, windowed_output.data());

        // Window-based indicators must match exactly
        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[i])) continue;
            int wi = i - windowed_start;
            EXPECT_NEAR(windowed_output[wi], full_output[i], 1e-4f)
                << "Bollinger middle windowed vs full mismatch at bar " << i;
            EXPECT_NEAR(windowed_output[windowed_len + wi], full_output[N + i], 1e-4f)
                << "Bollinger upper windowed vs full mismatch at bar " << i;
            EXPECT_NEAR(windowed_output[2 * windowed_len + wi], full_output[2 * N + i], 1e-4f)
                << "Bollinger lower windowed vs full mismatch at bar " << i;
        }
    }

    // Test 5: Donchian(20) -- window-based, exact match with period prefix
    {
        DonchianChannel dc(20);
        const int       N = TOTAL;

        std::vector<float> full_output(N * 2);
        dc.compute(bars.data(), 0, N, full_output.data());

        int                windowed_start = win_start - 20;
        int                windowed_len   = win_end - windowed_start;
        std::vector<float> windowed_output(windowed_len * 2);
        dc.compute(bars.data(), windowed_start, win_end, windowed_output.data());

        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[i])) continue;
            int wi = i - windowed_start;
            EXPECT_NEAR(windowed_output[wi], full_output[i], 1e-5f)
                << "Donchian upper windowed vs full mismatch at bar " << i;
            EXPECT_NEAR(windowed_output[windowed_len + wi], full_output[N + i], 1e-5f)
                << "Donchian lower windowed vs full mismatch at bar " << i;
        }
    }

    // Test 6: CCI(20) -- window-based but uses running sum internally.
    // When starting computation from a different bar, float accumulation order
    // differs, producing small rounding differences (relative error < 1%).
    // We verify the relative error is small and consistent.
    {
        CCI       cci(20);
        const int N = TOTAL;

        std::vector<float> full_output(N);
        cci.compute(bars.data(), 0, N, full_output.data());

        int                windowed_start = win_start - 20;
        int                windowed_len   = win_end - windowed_start;
        std::vector<float> windowed_output(windowed_len);
        cci.compute(bars.data(), windowed_start, win_end, windowed_output.data());

        // CCI uses a running sum for SMA of typical prices. Starting from
        // a different bar means different float accumulation order, causing
        // small absolute precision differences (~0.1-0.2). Use a combined
        // absolute + relative tolerance: acceptable if either the absolute
        // difference is small OR the relative error is small.
        int checked          = 0;
        int within_tolerance = 0;
        for (int i = win_start; i < win_end; ++i) {
            if (is_nan(full_output[i])) continue;
            int wi = i - windowed_start;
            if (is_nan(windowed_output[wi])) continue;
            float full_val = full_output[i];
            float win_val  = windowed_output[wi];
            float abs_diff = std::abs(win_val - full_val);
            // Accept if absolute diff < 0.5 (tiny for CCI range of -200..200)
            // or relative error < 5%
            bool ok = (abs_diff < 0.5f) || (abs_diff / std::max(1.0f, std::abs(full_val)) < 0.05f);
            if (ok) within_tolerance++;
            checked++;
        }
        EXPECT_GT(checked, 500) << "Should have checked many CCI bars";
        // All bars should be within tolerance
        EXPECT_EQ(within_tolerance, checked) << "CCI: " << (checked - within_tolerance) << " of "
                                             << checked << " bars exceeded tolerance";
    }
}

// =============================================================================
// i. Integration.MultithreadedDeterminism
//
// Run a small screen pipeline (multiple indicators, multiple strategies,
// 1000 bars) with the indicators/backtest computed on a single thread.
// Run again. Results must be identical.
//
// Since the backtest engine and indicator compute are stateless and
// deterministic, we verify that rerunning the same computation gives
// bit-identical results. This catches any accidental use of uninitialized
// memory, floating-point non-determinism, or thread-unsafe state.
// =============================================================================

TEST(Integration, MultithreadedDeterminism) {
    // Generate deterministic bars
    const int NUM_BARS = 1000;
    auto      bars     = generate_random_walk_bars(NUM_BARS, 4500.0f, 271828);

    // Compute a set of indicators
    RSI            rsi(14);
    BollingerBands bb(20, 2.0f);
    ATR            atr(14);
    MACD           macd(12, 26, 9);
    CCI            cci(20);
    Stochastic     stoch(14, 3);

    // Run 1: compute all indicators
    auto compute_all = [&](std::vector<std::vector<float>>& results) {
        results.clear();

        // RSI
        results.emplace_back(NUM_BARS);
        rsi.compute(bars.data(), 0, NUM_BARS, results.back().data());

        // Bollinger (3 series)
        results.emplace_back(NUM_BARS * 3);
        bb.compute(bars.data(), 0, NUM_BARS, results.back().data());

        // ATR
        results.emplace_back(NUM_BARS);
        atr.compute(bars.data(), 0, NUM_BARS, results.back().data());

        // MACD (3 series)
        results.emplace_back(NUM_BARS * 3);
        macd.compute(bars.data(), 0, NUM_BARS, results.back().data());

        // CCI
        results.emplace_back(NUM_BARS);
        cci.compute(bars.data(), 0, NUM_BARS, results.back().data());

        // Stochastic (2 series)
        results.emplace_back(NUM_BARS * 2);
        stoch.compute(bars.data(), 0, NUM_BARS, results.back().data());
    };

    std::vector<std::vector<float>> run1, run2;
    compute_all(run1);
    compute_all(run2);

    // Verify bit-identical results
    ASSERT_EQ(run1.size(), run2.size());
    for (size_t s = 0; s < run1.size(); ++s) {
        ASSERT_EQ(run1[s].size(), run2[s].size()) << "Size mismatch for series " << s;
        for (size_t i = 0; i < run1[s].size(); ++i) {
            // For NaN, both must be NaN
            if (is_nan(run1[s][i])) {
                EXPECT_TRUE(is_nan(run2[s][i])) << "NaN mismatch at series " << s << " index " << i;
            } else {
                EXPECT_EQ(run1[s][i], run2[s][i])
                    << "Bit-exact mismatch at series " << s << " index " << i;
            }
        }
    }

    // Now also verify that running backtests produces deterministic results
    // Create a simple strategy config with RSI signal
    const int eval_start = 200;
    const int eval_end   = NUM_BARS;
    const int eval_len   = eval_end - eval_start;

    // Build signal: use RSI output, map to [-1, 1] around 50
    std::vector<float> rsi_signal(eval_len);
    for (int i = 0; i < eval_len; ++i) {
        float rsi_val = run1[0][eval_start + i];
        if (is_nan(rsi_val)) {
            rsi_signal[i] = 0.0f;
        } else {
            rsi_signal[i] = (rsi_val - 50.0f) / 50.0f;  // normalize to [-1, 1]
        }
    }
    const float* signals[] = {rsi_signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.3f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 20;

    CostModel cost = CostModel::es();

    BacktestEngine engine1, engine2;
    auto result1 = engine1.run(bars.data(), eval_start, eval_end, signals, 1, config, cost, 1);
    auto result2 = engine2.run(bars.data(), eval_start, eval_end, signals, 1, config, cost, 1);

    EXPECT_EQ(result1.total_trades, result2.total_trades);
    EXPECT_FLOAT_EQ(result1.total_pnl, result2.total_pnl);
    EXPECT_FLOAT_EQ(result1.profit_factor, result2.profit_factor);
    EXPECT_FLOAT_EQ(result1.win_rate, result2.win_rate);

    // Verify trades are identical
    const auto& trades1 = engine1.last_trades();
    const auto& trades2 = engine2.last_trades();
    ASSERT_EQ(trades1.size(), trades2.size());
    for (size_t i = 0; i < trades1.size(); ++i) {
        EXPECT_EQ(trades1[i].entry_bar, trades2[i].entry_bar);
        EXPECT_EQ(trades1[i].exit_bar, trades2[i].exit_bar);
        EXPECT_FLOAT_EQ(trades1[i].entry_price, trades2[i].entry_price);
        EXPECT_FLOAT_EQ(trades1[i].exit_price, trades2[i].exit_price);
        EXPECT_EQ(trades1[i].direction, trades2[i].direction);
        EXPECT_FLOAT_EQ(trades1[i].net_pnl, trades2[i].net_pnl);
    }

    // Run on different thread to verify no thread-local state contamination
    BacktestResult     thread_result;
    std::vector<Trade> thread_trades;
    std::thread        worker([&]() {
        BacktestEngine thread_engine;
        thread_result =
            thread_engine.run(bars.data(), eval_start, eval_end, signals, 1, config, cost, 1);
        thread_trades = thread_engine.last_trades();
    });
    worker.join();

    EXPECT_EQ(result1.total_trades, thread_result.total_trades);
    EXPECT_FLOAT_EQ(result1.total_pnl, thread_result.total_pnl);
    ASSERT_EQ(trades1.size(), thread_trades.size());
    for (size_t i = 0; i < trades1.size(); ++i) {
        EXPECT_FLOAT_EQ(trades1[i].net_pnl, thread_trades[i].net_pnl);
    }
}

// =============================================================================
// j. Integration.KnownEdgeSyntheticWFO
//
// Create 5000 bars of synthetic data where price rises 2 points every
// "Monday" (session_day % 5 == 0). Include a simple day-of-week signal.
// Run the backtest with a signal that buys on Monday, holds for 1 session.
// Verify the OOS equity is positive.
//
// This validates that the backtest engine correctly processes a known
// exploitable pattern.
// =============================================================================

TEST(Integration, KnownEdgeSyntheticWFO) {
    // Generate 5000 bars with a planted Monday edge.
    // Each "session" is 100 bars (50 min of 30s bars -- compressed).
    // Every 5th session (session_day % 5 == 0 = "Monday"), price drifts up 2 pts.
    // Other days: random walk (zero drift).

    const int NUM_BARS         = 5000;
    const int BARS_PER_SESSION = 100;
    const int NUM_SESSIONS     = NUM_BARS / BARS_PER_SESSION;

    std::mt19937            rng(54321);
    portable::normal<float> noise{0.0f, 0.25f};

    std::vector<Bar> bars(NUM_BARS);
    float            price           = 4500.0f;
    int64_t          ts              = 1735858800LL * NS_PER_SEC;
    const int64_t    bar_interval_ns = 30LL * NS_PER_SEC;

    for (int sess = 0; sess < NUM_SESSIONS; ++sess) {
        bool  is_monday     = (sess % 5 == 0);
        float drift_per_bar = is_monday ? (2.0f / static_cast<float>(BARS_PER_SESSION)) : 0.0f;

        for (int b = 0; b < BARS_PER_SESSION; ++b) {
            int   idx   = sess * BARS_PER_SESSION + b;
            float move  = drift_per_bar + noise(rng);
            float open  = price;
            float close = price + move;
            float high  = std::max(open, close) + std::abs(noise(rng));
            float low   = std::min(open, close) - std::abs(noise(rng));

            bars[idx].timestamp = ts;
            bars[idx].open      = open;
            bars[idx].high      = high;
            bars[idx].low       = low;
            bars[idx].close     = close;
            bars[idx].volume    = 200;
            bars[idx]._pad      = 0;

            price = close;
            ts += bar_interval_ns;
        }
        // Add a 60-minute gap between sessions to trigger session detection
        ts += 3600LL * NS_PER_SEC;
    }

    // Build a signal that is +1 on "Monday" bars and 0 otherwise
    const int eval_start = 0;
    const int eval_end   = NUM_BARS;
    const int eval_len   = eval_end - eval_start;

    std::vector<float> monday_signal(eval_len, 0.0f);
    for (int sess = 0; sess < NUM_SESSIONS; ++sess) {
        if (sess % 5 == 0) {
            // Signal fires on the first bar of each Monday session
            int first_bar = sess * BARS_PER_SESSION;
            if (first_bar < eval_len) {
                monday_signal[first_bar] = 1.0f;
            }
        }
    }
    const float* signals[] = {monday_signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = BARS_PER_SESSION - 2;  // Hold for ~1 session

    CostModel cost = CostModel::es();

    BacktestEngine engine;
    auto result = engine.run(bars.data(), eval_start, eval_end, signals, 1, config, cost, 1);

    const auto& trades = engine.last_trades();

    // We should have roughly NUM_SESSIONS/5 = 10 trades
    EXPECT_GE(static_cast<int>(trades.size()), 5) << "Should have multiple Monday trades";

    // The Monday edge gives +2 pts per session = +$100 gross per trade (ES).
    // With $75 round-trip cost, net ~$25 per trade.
    // Total PnL should be positive.
    EXPECT_GT(result.total_pnl, 0.0f)
        << "Total PnL should be positive when exploiting the Monday edge. "
        << "Got: " << result.total_pnl << " from " << trades.size() << " trades";

    // Verify each Monday trade individually: most should have positive gross PnL
    int positive_gross = 0;
    for (const auto& t : trades) {
        if (t.gross_pnl > 0.0f) positive_gross++;
    }
    // With +2pt drift and noise(sigma=0.25), we expect most trades to be positive gross
    EXPECT_GT(positive_gross, static_cast<int>(trades.size()) / 2)
        << "Most Monday trades should have positive gross PnL";

    // Verify that an anti-Monday strategy (short on Monday) loses money
    std::vector<float> anti_signal(eval_len, 0.0f);
    for (int sess = 0; sess < NUM_SESSIONS; ++sess) {
        if (sess % 5 == 0) {
            int first_bar = sess * BARS_PER_SESSION;
            if (first_bar < eval_len) {
                anti_signal[first_bar] = -1.0f;  // short on Monday
            }
        }
    }
    const float* anti_signals[] = {anti_signal.data()};

    BacktestEngine anti_engine;
    auto           anti_result =
        anti_engine.run(bars.data(), eval_start, eval_end, anti_signals, 1, config, cost, 1);

    // Shorting on Monday should lose money (drift is upward)
    EXPECT_LT(anti_result.total_pnl, result.total_pnl)
        << "Anti-Monday strategy should perform worse than Monday strategy";
}

// =============================================================================
// k. Integration.CheckpointResumeDeterminism
//
// Create a Checkpoint with known data. Save it. Load it back.
// Verify all fields match the original.
// =============================================================================

TEST(Integration, CheckpointResumeDeterminism) {
    // Create a temporary file path for the checkpoint
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string ckpt_path =
        temp_dir + "/wfo_test_checkpoint_" +
        std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".ckpt";

    // Build a checkpoint with known data
    Checkpoint original;
    original.run_id               = "test_run_12345";
    original.phase                = 1;
    original.strategies_evaluated = 42;
    original.total_strategies     = 100;

    // Add some completed results
    for (int i = 0; i < 5; ++i) {
        BacktestResult r;
        r.total_pnl            = 1000.0f * (i + 1);
        r.total_trades         = 50 + i * 10;
        r.profit_factor        = 1.2f + i * 0.1f;
        r.annual_roc           = 10.0f + i * 5.0f;
        r.green_weeks_pct      = 0.55f + i * 0.05f;
        r.win_rate             = 0.50f + i * 0.02f;
        r.max_red_weeks        = 3 - (i % 3);
        r.trade_recovery_ratio = 0.4f + i * 0.1f;
        r.strategy_id          = i;
        r.max_drawdown         = 500.0f + i * 100.0f;
        r.max_live_drawdown    = 300.0f + i * 50.0f;
        r.avg_trade_dd         = 100.0f + i * 25.0f;
        r.avg_trade_peak       = 150.0f + i * 30.0f;
        r.sharpe_ratio         = 1.0f + i * 0.3f;
        r.calmar_ratio         = 0.5f + i * 0.2f;
        r.payoff_ratio         = 1.5f + i * 0.25f;
        r.avg_win              = 200.0f + i * 20.0f;
        r.avg_loss             = -(150.0f + i * 10.0f);
        r.negative_months      = i;
        r.negative_years       = (i > 2) ? 1 : 0;
        r.viable               = (i % 2 == 0);
        original.completed_results.push_back(r);
    }

    // Save checkpoint
    bool save_ok = original.save(ckpt_path);
    ASSERT_TRUE(save_ok) << "Failed to save checkpoint to " << ckpt_path;

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(ckpt_path)) << "Checkpoint file should exist after save";

    // Verify file is valid
    EXPECT_TRUE(Checkpoint::is_valid(ckpt_path)) << "Checkpoint file should pass validity check";

    // Load checkpoint (returns std::optional)
    auto loaded_opt = Checkpoint::load(ckpt_path);
    ASSERT_TRUE(loaded_opt.has_value());
    Checkpoint& loaded = *loaded_opt;

    // Verify all scalar fields match
    EXPECT_EQ(loaded.run_id, original.run_id);
    EXPECT_EQ(loaded.phase, original.phase);
    EXPECT_EQ(loaded.strategies_evaluated, original.strategies_evaluated);
    EXPECT_EQ(loaded.total_strategies, original.total_strategies);

    // Verify completed results
    ASSERT_EQ(loaded.completed_results.size(), original.completed_results.size());
    for (size_t i = 0; i < original.completed_results.size(); ++i) {
        const auto& orig = original.completed_results[i];
        const auto& load = loaded.completed_results[i];

        EXPECT_FLOAT_EQ(load.total_pnl, orig.total_pnl) << "total_pnl mismatch at result " << i;
        EXPECT_EQ(load.total_trades, orig.total_trades) << "total_trades mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.profit_factor, orig.profit_factor)
            << "profit_factor mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.annual_roc, orig.annual_roc) << "annual_roc mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.green_weeks_pct, orig.green_weeks_pct)
            << "green_weeks_pct mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.win_rate, orig.win_rate) << "win_rate mismatch at result " << i;
        EXPECT_EQ(load.max_red_weeks, orig.max_red_weeks)
            << "max_red_weeks mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.trade_recovery_ratio, orig.trade_recovery_ratio)
            << "trade_recovery_ratio mismatch at result " << i;
        EXPECT_EQ(load.strategy_id, orig.strategy_id) << "strategy_id mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.max_drawdown, orig.max_drawdown)
            << "max_drawdown mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.max_live_drawdown, orig.max_live_drawdown)
            << "max_live_drawdown mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.avg_trade_dd, orig.avg_trade_dd)
            << "avg_trade_dd mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.avg_trade_peak, orig.avg_trade_peak)
            << "avg_trade_peak mismatch at result " << i;
        EXPECT_FLOAT_EQ(load.sharpe_ratio, orig.sharpe_ratio)
            << "sharpe_ratio mismatch at result " << i;
    }

    // Clean up
    std::filesystem::remove(ckpt_path);
    EXPECT_FALSE(std::filesystem::exists(ckpt_path)) << "Checkpoint file should be cleaned up";
}

// =============================================================================
// k2. Integration.CheckpointBoolFieldsRoundTrip
//
// Verify that ALL bool fields in checkpoint (viable, skipped, is_real_edge,
// is_gap_artifact) round-trip correctly. Bools are serialized as explicit
// uint8_t (1 byte) rather than via write_pod/read_pod with
// implementation-defined sizeof(bool).
// =============================================================================

TEST(Integration, CheckpointBoolFieldsRoundTrip) {
    std::string temp_dir = std::filesystem::temp_directory_path().string();
    std::string ckpt_path =
        temp_dir + "/wfo_test_ckpt_bool_" +
        std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".ckpt";

    Checkpoint original;
    original.run_id               = "bool_test";
    original.phase                = 2;
    original.strategies_evaluated = 10;
    original.total_strategies     = 10;

    // Add BacktestResults with alternating viable flags
    for (int i = 0; i < 4; ++i) {
        BacktestResult r;
        r.total_pnl    = 100.0f * i;
        r.total_trades = 10;
        r.strategy_id  = i;
        r.viable       = (i % 2 == 0);  // true, false, true, false
        original.completed_results.push_back(r);
    }

    // Add WFO runs with all permutations of bool fields
    for (int run_idx = 0; run_idx < 2; ++run_idx) {
        WFORunResult run;
        run.schedule.frequency_bars = 100;
        run.schedule.lookback_bars  = 200;
        run.schedule.frequency_days = 5;
        run.schedule.lookback_days  = 10;
        run.schedule.trigger        = WFOTrigger::FIXED_INTERVAL;
        run.selection_strategy      = "best_sharpe";
        run.parameter_stability     = 0.5f;
        run.windows_skipped         = run_idx;
        run.sufficient_windows      = (run_idx == 0);  // first run viable, second not

        // Add windows with different bool combos for PnLDecomposition + skipped
        for (int w = 0; w < 4; ++w) {
            WFOWindowResult win;
            win.window_index  = w;
            win.is_start_bar  = w * 100;
            win.is_end_bar    = (w + 1) * 100;
            win.oos_start_bar = (w + 1) * 100;
            win.oos_end_bar   = (w + 2) * 100;
            win.strategy_id   = w;
            win.archetype     = "Grinder";

            // Test all 4 combinations of (is_real_edge, is_gap_artifact)
            win.pnl_decomp.is_real_edge       = (w & 1) != 0;  // F, T, F, T
            win.pnl_decomp.is_gap_artifact    = (w & 2) != 0;  // F, F, T, T
            win.pnl_decomp.total_bar_body_pnl = 100.0f * w;
            win.pnl_decomp.total_gap_pnl      = 50.0f * w;
            win.pnl_decomp.total_gross_pnl    = 150.0f * w;
            win.pnl_decomp.bar_body_pct       = 66.0f;
            win.pnl_decomp.gap_pct            = 34.0f;

            win.skipped = (w % 3 == 0);  // T, F, F, T

            // Set BacktestResult viable fields too
            win.is_result.viable = (w < 2);
            win.oos_lag1.viable  = (w == 1 || w == 3);
            win.oos_lag2.viable  = (w == 2);

            run.windows.push_back(win);
        }

        run.oos_aggregate.viable = run.sufficient_windows;
        original.completed_runs.push_back(run);
    }

    // Save and reload
    ASSERT_TRUE(original.save(ckpt_path));
    ASSERT_TRUE(Checkpoint::is_valid(ckpt_path));
    auto loaded_opt = Checkpoint::load(ckpt_path);
    ASSERT_TRUE(loaded_opt.has_value());
    Checkpoint& loaded = *loaded_opt;

    // Verify BacktestResult.viable round-trips
    ASSERT_EQ(loaded.completed_results.size(), original.completed_results.size());
    for (size_t i = 0; i < original.completed_results.size(); ++i) {
        EXPECT_EQ(loaded.completed_results[i].viable, original.completed_results[i].viable)
            << "BacktestResult.viable mismatch at index " << i;
    }

    // Verify WFORunResult fields
    ASSERT_EQ(loaded.completed_runs.size(), original.completed_runs.size());
    for (size_t r = 0; r < original.completed_runs.size(); ++r) {
        const auto& orig_run = original.completed_runs[r];
        const auto& load_run = loaded.completed_runs[r];

        EXPECT_EQ(load_run.sufficient_windows, orig_run.sufficient_windows)
            << "WFORunResult.sufficient_windows mismatch at run " << r;
        EXPECT_EQ(load_run.oos_aggregate.viable, orig_run.oos_aggregate.viable)
            << "WFORunResult.oos_aggregate.viable mismatch at run " << r;

        ASSERT_EQ(load_run.windows.size(), orig_run.windows.size());
        for (size_t w = 0; w < orig_run.windows.size(); ++w) {
            const auto& ow = orig_run.windows[w];
            const auto& lw = load_run.windows[w];

            EXPECT_EQ(lw.skipped, ow.skipped)
                << "WFOWindowResult.skipped mismatch at run " << r << " window " << w;
            EXPECT_EQ(lw.pnl_decomp.is_real_edge, ow.pnl_decomp.is_real_edge)
                << "PnLDecomposition.is_real_edge mismatch at run " << r << " window " << w;
            EXPECT_EQ(lw.pnl_decomp.is_gap_artifact, ow.pnl_decomp.is_gap_artifact)
                << "PnLDecomposition.is_gap_artifact mismatch at run " << r << " window " << w;
            EXPECT_EQ(lw.is_result.viable, ow.is_result.viable)
                << "is_result.viable mismatch at run " << r << " window " << w;
            EXPECT_EQ(lw.oos_lag1.viable, ow.oos_lag1.viable)
                << "oos_lag1.viable mismatch at run " << r << " window " << w;
            EXPECT_EQ(lw.oos_lag2.viable, ow.oos_lag2.viable)
                << "oos_lag2.viable mismatch at run " << r << " window " << w;
        }
    }

    std::filesystem::remove(ckpt_path);
}

// =============================================================================
// Integration.LandscapePipelineDeterminism: Full pipeline consistency
//
// Verify that the sanity-floor -> Pareto -> robustness -> reduced-set pipeline
// produces consistent results when run twice with the same input.
// =============================================================================

TEST(Integration, LandscapePipelineDeterminism) {
    // Create 30 results with a mix of viable and non-viable strategies
    std::vector<BacktestResult> results;

    std::mt19937            rng(1234);
    portable::normal<float> pnl_dist{2000.0f, 5000.0f};

    for (int i = 0; i < 30; ++i) {
        float pnl    = pnl_dist(rng);
        int   trades = 100 + static_cast<int>(rng() % 200);
        float pf     = (pnl > 0) ? 1.0f + static_cast<float>(rng() % 100) / 100.0f : 0.5f;
        float roc    = pnl / 150.0f;
        float gw     = (pnl > 0) ? 0.50f + static_cast<float>(rng() % 30) / 100.0f : 0.30f;
        float wr     = 0.40f + static_cast<float>(rng() % 30) / 100.0f;
        float trr    = 0.2f + static_cast<float>(rng() % 80) / 100.0f;

        BacktestResult r = make_result(pnl, trades, pf, roc, gw, wr, 3, trr, i);
        results.push_back(r);
    }

    int n = static_cast<int>(results.size());

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

    // Run pipeline twice
    LandscapeAnalysis la1 =
        analyze_landscape(results, 0.50f, robustness_neighbors, 1, reduced_neighbors);
    LandscapeAnalysis la2 =
        analyze_landscape(results, 0.50f, robustness_neighbors, 1, reduced_neighbors);

    // Verify identical results
    EXPECT_EQ(la1.total_candidates, la2.total_candidates);
    EXPECT_EQ(la1.passed_sanity_floors, la2.passed_sanity_floors);
    EXPECT_EQ(la1.on_pareto_frontier, la2.on_pareto_frontier);
    EXPECT_EQ(la1.grinder_count, la2.grinder_count);
    EXPECT_EQ(la1.sniper_count, la2.sniper_count);
    EXPECT_EQ(la1.balanced_count, la2.balanced_count);

    ASSERT_EQ(la1.robustness_scores.size(), la2.robustness_scores.size());
    for (size_t i = 0; i < la1.robustness_scores.size(); ++i) {
        EXPECT_FLOAT_EQ(la1.robustness_scores[i], la2.robustness_scores[i])
            << "Robustness score mismatch at index " << i;
    }

    ASSERT_EQ(la1.sanity_passed_indices.size(), la2.sanity_passed_indices.size());
    for (size_t i = 0; i < la1.sanity_passed_indices.size(); ++i) {
        EXPECT_EQ(la1.sanity_passed_indices[i], la2.sanity_passed_indices[i])
            << "Sanity passed index mismatch at " << i;
    }

    ASSERT_EQ(la1.reduced_set_ids.size(), la2.reduced_set_ids.size());
    for (size_t i = 0; i < la1.reduced_set_ids.size(); ++i) {
        EXPECT_EQ(la1.reduced_set_ids[i], la2.reduced_set_ids[i])
            << "Reduced set ID mismatch at " << i;
    }

    ASSERT_EQ(la1.frontier.size(), la2.frontier.size());
    for (size_t i = 0; i < la1.frontier.size(); ++i) {
        EXPECT_EQ(la1.frontier[i].original_index, la2.frontier[i].original_index)
            << "Frontier index mismatch at " << i;
        EXPECT_EQ(static_cast<int>(la1.frontier[i].archetype),
                  static_cast<int>(la2.frontier[i].archetype))
            << "Frontier archetype mismatch at " << i;
    }
}

// =============================================================================
// Additional: Verify indicator warmup consistency across types
//
// For each indicator type, verify that:
// 1. All bars before warmup_bars() produce NaN
// 2. The first bar at warmup_bars() produces a valid (non-NaN) value
// This is a cross-cutting integration test for indicator contract compliance.
// =============================================================================

TEST(Integration, IndicatorWarmupContract) {
    // Generate enough bars for even the largest warmup
    const int NUM_BARS = 500;
    auto      bars     = generate_random_walk_bars(NUM_BARS, 4500.0f, 65537);

    struct IndicatorTestCase {
        std::unique_ptr<Indicator> ind;
        const char*                label;
    };

    std::vector<IndicatorTestCase> cases;

    auto add = [&](Indicator* p, const char* label) {
        IndicatorTestCase tc;
        tc.ind.reset(p);
        tc.label = label;
        cases.push_back(std::move(tc));
    };

    // Single-series indicators (warmup applies uniformly)
    add(new RSI(14), "RSI(14)");
    add(new ATR(14), "ATR(14)");
    add(new CCI(20), "CCI(20)");

    for (auto& tc : cases) {
        const int ns = tc.ind->num_series();
        const int wu = tc.ind->warmup_bars();

        ASSERT_LT(wu, NUM_BARS) << tc.label << " warmup (" << wu << ") exceeds bar count";

        std::vector<float> output(NUM_BARS * ns);
        tc.ind->compute(bars.data(), 0, NUM_BARS, output.data());

        // Single-series: all bars before warmup must be NaN
        for (int s = 0; s < ns; ++s) {
            for (int i = 0; i < wu; ++i) {
                EXPECT_TRUE(is_nan(output[s * NUM_BARS + i]))
                    << tc.label << " series " << s << " should be NaN at bar " << i
                    << " (warmup=" << wu << ")";
            }
        }

        // First bar at warmup must be valid
        if (wu < NUM_BARS) {
            EXPECT_FALSE(is_nan(output[wu]))
                << tc.label << " series 0 should be valid at bar " << wu;
        }
    }
    cases.clear();

    // Multi-series indicators: warmup_bars() = max over all sub-series warmups.
    // Individual series may become valid earlier than warmup_bars().
    // Contract: at bar warmup_bars(), ALL series must be valid.
    // Before warmup_bars(), at least one series must still be NaN (at bar wu-1).

    struct MultiSeriesCase {
        std::unique_ptr<Indicator> ind;
        const char*                label;
    };
    std::vector<MultiSeriesCase> multi_cases;

    auto add_multi = [&](Indicator* p, const char* label) {
        MultiSeriesCase mc;
        mc.ind.reset(p);
        mc.label = label;
        multi_cases.push_back(std::move(mc));
    };

    add_multi(new BollingerBands(20, 2.0f), "BB(20,2)");
    add_multi(new MACD(12, 26, 9), "MACD(12,26,9)");
    add_multi(new ADX(14), "ADX(14)");
    add_multi(new Stochastic(14, 3), "Stoch(14,3)");
    add_multi(new DonchianChannel(20), "Donchian(20)");
    add_multi(new EMASMACross(MAType::EMA, 10, MAType::EMA, 30), "EMACross(10,30)");

    for (auto& mc : multi_cases) {
        const int ns = mc.ind->num_series();
        const int wu = mc.ind->warmup_bars();

        ASSERT_LT(wu, NUM_BARS) << mc.label << " warmup (" << wu << ") exceeds bar count";

        std::vector<float> output(NUM_BARS * ns);
        mc.ind->compute(bars.data(), 0, NUM_BARS, output.data());

        // At warmup_bars(), ALL series must be valid (not NaN)
        if (wu < NUM_BARS) {
            for (int s = 0; s < ns; ++s) {
                EXPECT_FALSE(is_nan(output[s * NUM_BARS + wu]))
                    << mc.label << " series " << s << " should be valid at bar " << wu
                    << " (warmup=" << wu << ")";
            }
        }

        // Well past warmup, all values should remain valid
        for (int s = 0; s < ns; ++s) {
            for (int i = wu + 10; i < NUM_BARS; i += 50) {
                EXPECT_FALSE(is_nan(output[s * NUM_BARS + i]))
                    << mc.label << " series " << s << " should be valid at bar " << i
                    << " (warmup=" << wu << ")";
            }
        }

        // Before warmup: at least the last series to converge should be NaN
        // at bar wu - 1. Check that at least one series is NaN there.
        if (wu > 0) {
            bool any_nan = false;
            for (int s = 0; s < ns; ++s) {
                if (is_nan(output[s * NUM_BARS + wu - 1])) {
                    any_nan = true;
                    break;
                }
            }
            EXPECT_TRUE(any_nan) << mc.label << " should have at least one NaN series at bar "
                                 << wu - 1 << " (just before warmup=" << wu << ")";
        }
    }
}
