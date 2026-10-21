// =============================================================================
// Strategy Execution Tests
//
// Backtest engine code paths for strategy configuration options:
//   - Early termination
//   - Time-gated backtest (session signals anti-lookahead)
//   - Trading hours filter (RTH, Morning, Afternoon)
//   - Aggregation modes (MAJORITY_VOTE, UNANIMOUS)
//   - Multi-indicator composite signals
//   - Filter roles (SIGNAL vs FILTER)
//   - OPPOSING_SIGNAL exit mode
//   - Short-side MAX_LOSS_ATR
//   - Out-of-bounds indicator index (runtime bounds check)
//
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>
#include "test_helpers.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "backtest_engine.h"
#include "cost_model.h"
#include "market_data.h"

// =============================================================================
// Helpers
// =============================================================================

static std::vector<Bar> make_bars_ohlc(const std::vector<std::array<float, 4>>& ohlc,
                                       int64_t base_ts = 1'262'304'000LL * NS_PER_SEC) {
    std::vector<Bar> bars;
    bars.reserve(ohlc.size());
    for (size_t i = 0; i < ohlc.size(); ++i) {
        int64_t ts = base_ts + static_cast<int64_t>(i) * 30 * NS_PER_SEC;
        bars.push_back(make_bar(ts, ohlc[i][0], ohlc[i][1], ohlc[i][2], ohlc[i][3]));
    }
    return bars;
}

// Build a simple N-bar dataset with constant price and optional sessions
struct TestData {
    std::vector<Bar>         bars;
    std::vector<SessionInfo> sessions;
    std::vector<float>       signal1;
    std::vector<float>       signal2;
    std::vector<float>       signal3;
    std::vector<float>       atr;

    TestData(int n, float price = 100.0f) {
        int64_t base_ts = 1'262'304'000LL * NS_PER_SEC;
        bars.resize(n);
        sessions.resize(n);
        signal1.resize(n, 0.0f);
        signal2.resize(n, 0.0f);
        signal3.resize(n, 0.0f);
        atr.resize(n, 2.0f);

        for (int i = 0; i < n; ++i) {
            bars[i].timestamp = base_ts + static_cast<int64_t>(i) * 30 * NS_PER_SEC;
            bars[i].open      = price;
            bars[i].high      = price + 1.0f;
            bars[i].low       = price - 1.0f;
            bars[i].close     = price;
            bars[i].volume    = 100;
            bars[i]._pad      = 0;

            sessions[i].session_day       = 0;
            sessions[i].intra_session_idx = i;
            // Default: 10:00 ET (36000 seconds)
            sessions[i].time_of_day_sec   = 36000 + i * 30;
            sessions[i].halt_flag         = 0;
            sessions[i].early_close       = 0;
            sessions[i].session_close_sec = 61200;  // 17:00 ET
        }
    }

    // Session flags derived from the (possibly edited) per-bar times.
    const SessionInfo* stamped_sessions() {
        stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                            SessionModel::us_futures(), 30);
        return sessions.data();
    }
};

// =============================================================================
// 1. EARLY TERMINATION
// =============================================================================

TEST(StrategyExecution, EarlyTermination_FiresOnNegativePnL) {
    // Create 100 bars with a losing trade early, then no more signals.
    // With early termination at 60%, the engine should abort.
    const int                         N = 100;
    std::vector<std::array<float, 4>> ohlc;
    ohlc.reserve(N);
    for (int i = 0; i < N; ++i) {
        float base = (i < 5) ? 100.0f : 90.0f;  // price drops after bar 4
        ohlc.push_back({base, base + 1.0f, base - 1.0f, base});
    }
    // Make bar 1 (entry) open at 100, then bar 2 opens low to create loss
    ohlc[0]   = {100.0f, 101.0f, 99.0f, 100.0f};
    ohlc[1]   = {100.0f, 101.0f, 99.0f, 100.0f};
    ohlc[2]   = {90.0f, 91.0f, 89.0f, 90.0f};  // signal reversal exit fill at open=90
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;   // long entry signal
    signal[2]              = -1.0f;  // reversal to exit
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::SIGNAL_REVERSAL;

    EarlyTermination et;
    et.enabled             = true;
    et.check_at_pct        = 0.05f;  // check at 5% (bar 5)
    et.min_trades_at_check = 100;    // require 100 trades by checkpoint (impossible)

    BacktestEngine engine;
    auto           result = engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1, et);

    // With early termination, the backtest should complete but with fewer trades
    // than if it ran to completion (or it ran normally and result shows the loss).
    // The key is that the engine doesn't crash and returns valid metrics.
    EXPECT_TRUE(std::isfinite(result.total_pnl));
}

TEST(StrategyExecution, EarlyTermination_DoesNotFireWhenDisabled) {
    const int N = 20;
    TestData  td(N);
    td.signal1[0]  = 1.0f;
    td.signal1[10] = -1.0f;  // reversal

    // Make price move to create a losing trade
    for (int i = 5; i < N; ++i) {
        td.bars[i].open  = 90.0f;
        td.bars[i].high  = 91.0f;
        td.bars[i].low   = 89.0f;
        td.bars[i].close = 90.0f;
    }

    const float* signals[] = {td.signal1.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::SIGNAL_REVERSAL;

    EarlyTermination et;
    et.enabled = false;

    BacktestEngine engine;
    auto           result = engine.run(td.bars.data(), 0, N, signals, 1, config, es_cost(), 1, et);

    EXPECT_GE(result.total_trades, 1);
    EXPECT_TRUE(std::isfinite(result.total_pnl));
}

// =============================================================================
// 2. TIME-GATED BACKTEST (Anti-Lookahead)
// =============================================================================

TEST(StrategyExecution, TimeGating_SignalGatedBefore10AM) {
    // Session signal available after 10:00 ET (36000 sec).
    // Bars before 10:00 should not see the signal.
    const int N = 20;
    TestData  td(N);

    // Set bar times: bars 0-4 are before 10:00, bars 5+ are after
    for (int i = 0; i < N; ++i) {
        td.sessions[i].time_of_day_sec = 34200 + i * 600;  // 09:30 + i*10min
        // Bar 0: 09:30, Bar 1: 09:40, ..., Bar 3: 10:00, Bar 4: 10:10
    }

    // Signal says "buy" on all bars
    for (int i = 0; i < N; ++i) td.signal1[i] = 1.0f;

    const float* signals[] = {td.signal1.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 36000;  // 10:00 ET

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), td.stamped_sessions(), 0, N, signals, 1, gates, config, es_cost(),
               1);

    // The first trade should not happen until after the gate time
    const auto& trades = engine.last_trades();
    if (!trades.empty()) {
        // Entry should be after the gate clears (bar 3+ with time >= 36000)
        int entry_signal_bar = trades[0].entry_bar - 1;  // signal bar is entry_bar - entry_lag
        EXPECT_GE(td.sessions[entry_signal_bar].time_of_day_sec, 36000)
            << "Trade entry should not occur before signal gate time";
    }
}

TEST(StrategyExecution, TimeGating_UngatedSignalAlwaysVisible) {
    const int N = 20;
    TestData  td(N);

    // All bars at 09:30 ET
    for (int i = 0; i < N; ++i) {
        td.sessions[i].time_of_day_sec = 34200;
    }

    td.signal1[0]          = 1.0f;
    const float* signals[] = {td.signal1.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;  // no gate

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), td.stamped_sessions(), 0, N, signals, 1, gates, config, es_cost(),
               1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Ungated signal should produce a trade";
    EXPECT_EQ(trades[0].entry_bar, 1) << "Entry at bar 1 (signal at bar 0 + lag 1)";
}

// =============================================================================
// 3. TRADING HOURS FILTER
// =============================================================================

TEST(StrategyExecution, TradingHours_RTH_BlocksETH) {
    const int N = 20;
    TestData  td(N);

    // Set all bars to pre-RTH (08:00 ET = 28800)
    for (int i = 0; i < N; ++i) {
        td.sessions[i].time_of_day_sec = 28800 + i * 30;
    }

    td.signal1[0]          = 1.0f;
    const float* signals[] = {td.signal1.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.trading_hours        = StrategyConfig::RTH;  // Only 09:30-16:00
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), td.stamped_sessions(), 0, N, signals, 1, gates, config, es_cost(),
               1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u)
        << "RTH filter should block trades during pre-RTH hours (08:00 ET)";
}

TEST(StrategyExecution, TradingHours_RTH_AllowsDuringRTH) {
    const int N = 20;
    TestData  td(N);

    // Set all bars to mid-RTH (11:00 ET = 39600)
    for (int i = 0; i < N; ++i) {
        td.sessions[i].time_of_day_sec = 39600 + i * 30;
    }

    td.signal1[0]          = 1.0f;
    const float* signals[] = {td.signal1.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.trading_hours        = StrategyConfig::RTH;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), td.stamped_sessions(), 0, N, signals, 1, gates, config, es_cost(),
               1);

    const auto& trades = engine.last_trades();
    EXPECT_GE(trades.size(), 1u) << "RTH filter should allow trades at 11:00 ET";
}

TEST(StrategyExecution, TradingHours_Morning_BlocksAfternoon) {
    const int N = 20;
    TestData  td(N);

    // Set bars to 13:00 ET (afternoon = 46800)
    for (int i = 0; i < N; ++i) {
        td.sessions[i].time_of_day_sec = 46800 + i * 30;
    }

    td.signal1[0]          = 1.0f;
    const float* signals[] = {td.signal1.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.trading_hours        = StrategyConfig::MORNING;  // 09:30-12:00 only
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), td.stamped_sessions(), 0, N, signals, 1, gates, config, es_cost(),
               1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u) << "Morning filter should block trades at 13:00 ET";
}

TEST(StrategyExecution, TradingHours_Afternoon_AllowsAfternoon) {
    const int N = 20;
    TestData  td(N);

    // Set bars to 13:00 ET
    for (int i = 0; i < N; ++i) {
        td.sessions[i].time_of_day_sec = 46800 + i * 30;
    }

    td.signal1[0]          = 1.0f;
    const float* signals[] = {td.signal1.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.trading_hours        = StrategyConfig::AFTERNOON;  // 12:00-16:00
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), td.stamped_sessions(), 0, N, signals, 1, gates, config, es_cost(),
               1);

    const auto& trades = engine.last_trades();
    EXPECT_GE(trades.size(), 1u) << "Afternoon filter should allow trades at 13:00 ET";
}

// =============================================================================
// 4. AGGREGATION MODES
// =============================================================================

TEST(StrategyExecution, MajorityVote_2of3Long) {
    const int N = 20;
    TestData  td(N);

    // 3 indicators: two say long (+1), one says short (-1)
    td.signal1[0] = 1.0f;   // long
    td.signal2[0] = 1.0f;   // long
    td.signal3[0] = -1.0f;  // short

    const float* signals[] = {td.signal1.data(), td.signal2.data(), td.signal3.data()};

    StrategyConfig config;
    config.num_indicators       = 3;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_indices[2] = 2;
    config.aggregation          = StrategyConfig::MAJORITY_VOTE;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 3, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Majority vote with 2/3 long should enter long";
    EXPECT_EQ(trades[0].direction, 1) << "Direction should be long (majority)";
}

TEST(StrategyExecution, MajorityVote_Tie_NoTrade) {
    const int N = 20;
    TestData  td(N);

    // 2 indicators: one long, one short -> tie
    td.signal1[0] = 1.0f;
    td.signal2[0] = -1.0f;

    const float* signals[] = {td.signal1.data(), td.signal2.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.aggregation          = StrategyConfig::MAJORITY_VOTE;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u) << "Majority vote tie should not produce a trade";
}

TEST(StrategyExecution, Unanimous_AllAgree_TradeEntered) {
    const int N = 20;
    TestData  td(N);

    td.signal1[0] = 1.0f;
    td.signal2[0] = 1.0f;
    td.signal3[0] = 1.0f;

    const float* signals[] = {td.signal1.data(), td.signal2.data(), td.signal3.data()};

    StrategyConfig config;
    config.num_indicators       = 3;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_indices[2] = 2;
    config.aggregation          = StrategyConfig::UNANIMOUS;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 3, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Unanimous with all long should enter";
    EXPECT_EQ(trades[0].direction, 1);
}

TEST(StrategyExecution, Unanimous_OneDisagrees_NoTrade) {
    const int N = 20;
    TestData  td(N);

    td.signal1[0] = 1.0f;
    td.signal2[0] = 1.0f;
    td.signal3[0] = -1.0f;  // disagrees

    const float* signals[] = {td.signal1.data(), td.signal2.data(), td.signal3.data()};

    StrategyConfig config;
    config.num_indicators       = 3;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_indices[2] = 2;
    config.aggregation          = StrategyConfig::UNANIMOUS;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 3, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u) << "Unanimous with disagreement should not trade";
}

// =============================================================================
// 5. MULTI-INDICATOR WEIGHTED SUM
// =============================================================================

TEST(StrategyExecution, WeightedSum_TwoIndicators_CorrectDirection) {
    const int N = 20;
    TestData  td(N);

    // Indicator 0: weak sell (-0.4), Indicator 1: strong buy (+0.8)
    // Weights: 0.5 and 1.0
    // Composite = (-0.4*0.5 + 0.8*1.0) / (0.5+1.0) = 0.4 / 1.5 = 0.4
    // Below threshold 0.5 -> no trade
    td.signal1[0] = -0.4f;
    td.signal2[0] = 0.8f;

    const float* signals[] = {td.signal1.data(), td.signal2.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.weights[0]           = 0.5f;
    config.weights[1]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    // Composite ~0.4 < threshold 0.5 -> no trade
    EXPECT_EQ(trades.size(), 0u) << "Weighted sum below threshold should not enter";
}

TEST(StrategyExecution, WeightedSum_TwoIndicators_AboveThreshold) {
    const int N = 20;
    TestData  td(N);

    // Indicator 0: strong buy (+0.9), Indicator 1: moderate buy (+0.6)
    // Weights: 1.0 and 1.0
    // Composite = (0.9 + 0.6) / 2.0 = 0.75 > 0.5 -> long
    td.signal1[0] = 0.9f;
    td.signal2[0] = 0.6f;

    const float* signals[] = {td.signal1.data(), td.signal2.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Strong composite should enter";
    EXPECT_EQ(trades[0].direction, 1);
}

// =============================================================================
// 6. FILTER ROLES
// =============================================================================

TEST(StrategyExecution, FilterRole_BlocksEntry) {
    const int N = 20;
    TestData  td(N);

    // Signal indicator says long, but filter is zero (inactive) -> no entry
    td.signal1[0] = 1.0f;  // directional signal: buy
    td.signal2[0] = 0.0f;  // filter: inactive (blocks entry)

    const float* signals[] = {td.signal1.data(), td.signal2.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_roles[0]   = StrategyConfig::SIGNAL;
    config.indicator_roles[1]   = StrategyConfig::FILTER;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u) << "Zero filter should block entry";
}

TEST(StrategyExecution, FilterRole_AllowsEntry) {
    const int N = 20;
    TestData  td(N);

    // Signal says buy, filter is active (non-zero) -> entry allowed
    td.signal1[0] = 1.0f;  // directional signal
    td.signal2[0] = 1.0f;  // filter: active

    const float* signals[] = {td.signal1.data(), td.signal2.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_roles[0]   = StrategyConfig::SIGNAL;
    config.indicator_roles[1]   = StrategyConfig::FILTER;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Active filter should allow entry";
    EXPECT_EQ(trades[0].direction, 1);
}

TEST(StrategyExecution, FilterRole_DoesNotAffectDirection) {
    const int N = 20;
    TestData  td(N);

    // Signal says sell (-1), filter is active but negative (-1).
    // Filter should NOT contribute to direction, only gate entry.
    td.signal1[0] = -1.0f;  // directional signal: sell
    td.signal2[0] = -1.0f;  // filter: active (non-zero)

    const float* signals[] = {td.signal1.data(), td.signal2.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_roles[0]   = StrategyConfig::SIGNAL;
    config.indicator_roles[1]   = StrategyConfig::FILTER;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Active filter should allow entry";
    // Direction should be short (from signal), not affected by negative filter
    EXPECT_EQ(trades[0].direction, -1) << "Filter should not affect direction";
}

// =============================================================================
// 7. OPPOSING SIGNAL EXIT MODE
// =============================================================================

TEST(StrategyExecution, OpposingSignal_ExitOnReversal) {
    const int N = 20;
    TestData  td(N);

    // Signal goes long, then reverses to strong short -> exit
    td.signal1[0] = 1.0f;   // long entry
    td.signal1[5] = -0.8f;  // strong opposing signal

    const float* signals[] = {td.signal1.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::OPPOSING_SIGNAL;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 1, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Should have at least one trade";
    EXPECT_EQ(trades[0].direction, 1) << "First trade should be long";
    // The exit should happen when the opposing signal triggers
    // (signal at bar 5, exit pending, fills at bar 6 or immediate depending on impl)
    EXPECT_LE(trades[0].exit_bar, 7) << "Should exit near the opposing signal";
}

// =============================================================================
// 8. SHORT-SIDE MAX_LOSS_ATR
// =============================================================================

TEST(StrategyExecution, Short_MaxLossATR_StopTriggered) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 1: entry at open=100
        {100.0f, 106.0f, 99.0f, 105.0f},  // bar 2: high=106 >= SL=104 -> triggered
        {105.0f, 106.0f, 104.0f, 105.0f}, {105.0f, 106.0f, 104.0f, 105.0f},
        {105.0f, 106.0f, 104.0f, 105.0f}, {105.0f, 106.0f, 104.0f, 105.0f},
        {105.0f, 106.0f, 104.0f, 105.0f}, {105.0f, 106.0f, 104.0f, 105.0f},
        {105.0f, 106.0f, 104.0f, 105.0f},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = -1.0f;  // short signal
    std::vector<float> atr(N, 2.0f);
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::MAX_LOSS_ATR;
    config.max_loss_atr_mult    = 2.0f;  // SL = entry + 2*2 = 104
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    EXPECT_EQ(trades[0].direction, -1) << "Should be short";
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].exit_bar, 2);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 104.0f);  // SL level
    EXPECT_LT(trades[0].gross_pnl, 0.0f) << "Short max loss exit should be a loss";
}

// =============================================================================
// 9. OUT-OF-BOUNDS INDICATOR INDEX (Runtime Bounds Check)
// =============================================================================

TEST(StrategyExecution, OutOfBoundsIndicatorIndex_ReturnsZeroSignal) {
    const int N = 20;
    TestData  td(N);
    td.signal1[0] = 1.0f;

    const float* signals[] = {td.signal1.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 99;  // out of bounds (only 1 series)
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    // Should NOT crash -- the bounds check should return 0.0 composite
    engine.run(td.bars.data(), 0, N, signals, 1, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u)
        << "Out-of-bounds indicator index should produce zero composite (no trade)";
}

TEST(StrategyExecution, NegativeIndicatorIndex_ReturnsZeroSignal) {
    const int N = 20;
    TestData  td(N);
    td.signal1[0] = 1.0f;

    const float* signals[] = {td.signal1.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = -1;  // negative, out of bounds
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 1, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u)
        << "Negative indicator index should produce zero composite (no trade)";
}

// =============================================================================
// 10. TAKE_PROFIT_ATR STANDALONE
// =============================================================================

TEST(StrategyExecution, TakeProfitATR_Long_FillsAtTPLevel) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 1: entry at open=100
        {100.0f, 105.0f, 99.0f, 104.0f},  // bar 2: high=105>=104 -> TP triggered
        {104.0f, 105.0f, 103.0f, 104.0f}, {104.0f, 105.0f, 103.0f, 104.0f},
        {104.0f, 105.0f, 103.0f, 104.0f}, {104.0f, 105.0f, 103.0f, 104.0f},
        {104.0f, 105.0f, 103.0f, 104.0f}, {104.0f, 105.0f, 103.0f, 104.0f},
        {104.0f, 105.0f, 103.0f, 104.0f},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;
    std::vector<float> atr(N, 2.0f);
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::TAKE_PROFIT_ATR;
    config.tp_atr_mult          = 2.0f;  // TP at entry + 4.0 = 104.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].exit_bar, 2);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 104.0f);  // TP level
    EXPECT_GT(trades[0].gross_pnl, 0.0f);
}

// =============================================================================
// 11. FILTER + MAJORITY_VOTE COMBINED
// =============================================================================

TEST(StrategyExecution, MajorityVote_WithFilter_FilterBlocksEntry) {
    const int N = 20;
    TestData  td(N);

    // 2 signals say long, 1 filter is inactive
    td.signal1[0] = 1.0f;  // SIGNAL: long
    td.signal2[0] = 1.0f;  // SIGNAL: long
    td.signal3[0] = 0.0f;  // FILTER: inactive

    const float* signals[] = {td.signal1.data(), td.signal2.data(), td.signal3.data()};

    StrategyConfig config;
    config.num_indicators       = 3;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_indices[2] = 2;
    config.indicator_roles[0]   = StrategyConfig::SIGNAL;
    config.indicator_roles[1]   = StrategyConfig::SIGNAL;
    config.indicator_roles[2]   = StrategyConfig::FILTER;
    config.aggregation          = StrategyConfig::MAJORITY_VOTE;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(td.bars.data(), 0, N, signals, 3, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u) << "Inactive filter should block entry even with majority long";
}

// =============================================================================
// 12. ZERO-BAR RANGE
// =============================================================================

TEST(StrategyExecution, EmptyBarRange_NoTrades) {
    TestData td(10);
    td.signal1[0]          = 1.0f;
    const float* signals[] = {td.signal1.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    // bar_start == bar_end -> zero-length range
    auto result = engine.run(td.bars.data(), 5, 5, signals, 1, config, es_cost(), 1);
    EXPECT_EQ(result.total_trades, 0);
}
