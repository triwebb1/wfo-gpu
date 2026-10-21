// =============================================================================
// Backtest Edge-Case Tests
//
// Tests for TP/SL same-bar ambiguity and related edge cases.
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>
#include "test_helpers.h"

#include <array>
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

// =============================================================================
// TP/SL ATR Edge Cases
// =============================================================================

// ---------------------------------------------------------------------------
// Both TP and SL trigger on the same bar (long position).
// Entry at 100.0, ATR=2.0, tp_mult=2.0 (TP at 104), sl_mult=2.0 (SL at 96).
// Bar has high=105 (>=104, TP triggers) and low=95 (<=96, SL triggers).
// The trade should exit (SL takes precedence per convention).
// ---------------------------------------------------------------------------
TEST(BacktestEdge, TpSlAtr_BothTriggerSameBar) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 1: entry at open=100.0
        {100.0f, 105.0f, 95.0f, 100.0f},  // bar 2: high=105>=104 (TP), low=95<=96 (SL)
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 3: exit fills at open=100.0
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 4-9: padding
        {100.0f, 101.0f, 99.0f, 100.0f}, {100.0f, 101.0f, 99.0f, 100.0f},
        {100.0f, 101.0f, 99.0f, 100.0f}, {100.0f, 101.0f, 99.0f, 100.0f},
        {100.0f, 101.0f, 99.0f, 100.0f},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;  // long signal at bar 0
    std::vector<float> atr(N, 2.0f);
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::TP_SL_ATR;
    config.tp_atr_mult          = 2.0f;  // TP at entry + 4.0 = 104.0
    config.sl_atr_mult          = 2.0f;  // SL at entry - 4.0 =  96.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Should have at least one trade when both TP and SL trigger";

    // Level-triggered: SL takes precedence, fills at SL level (96.0) on bar 2
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].exit_bar, 2);              // immediate fill on detection bar
    EXPECT_FLOAT_EQ(trades[0].exit_price, 96.0f);  // SL level, not next-bar open
    EXPECT_EQ(trades[0].direction, 1);
    EXPECT_LT(trades[0].gross_pnl, 0.0f) << "SL-precedence exit should be a loss";
}

// ---------------------------------------------------------------------------
// Only TP triggers (long position).
// Entry at 100.0, ATR=2.0, tp_mult=1.5 (TP at 103), sl_mult=2.0 (SL at 96).
// Bar has high=104 (>=103, TP triggers) and low=97 (>96, SL does NOT trigger).
// ---------------------------------------------------------------------------
TEST(BacktestEdge, TpSlAtr_OnlyTpTriggers) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},   // bar 0: signal fires
        {100.0f, 101.0f, 99.0f, 100.5f},   // bar 1: entry at open=100.0
        {100.5f, 104.0f, 97.0f, 102.0f},   // bar 2: high=104>=103 (TP), low=97>96 (no SL)
        {102.0f, 103.0f, 101.0f, 102.5f},  // bar 3: exit fills at open=102.0
        {102.5f, 103.0f, 102.0f, 102.5f},  // bar 4-9: padding
        {102.5f, 103.0f, 102.0f, 102.5f}, {102.5f, 103.0f, 102.0f, 102.5f},
        {102.5f, 103.0f, 102.0f, 102.5f}, {102.5f, 103.0f, 102.0f, 102.5f},
        {102.5f, 103.0f, 102.0f, 102.5f},
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
    config.exit_mode            = StrategyConfig::TP_SL_ATR;
    config.tp_atr_mult          = 1.5f;  // TP at entry + 3.0 = 103.0
    config.sl_atr_mult          = 2.0f;  // SL at entry - 4.0 =  96.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Should have at least one trade when TP triggers";

    // Level-triggered: TP fills at TP level (103.0) on bar 2
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].exit_bar, 2);               // immediate fill on detection bar
    EXPECT_FLOAT_EQ(trades[0].exit_price, 103.0f);  // TP level, not next-bar open
    EXPECT_EQ(trades[0].direction, 1);
    EXPECT_GT(trades[0].gross_pnl, 0.0f) << "TP-only exit should be profitable";
}

// ---------------------------------------------------------------------------
// Only SL triggers (long position).
// Entry at 100.0, ATR=2.0, tp_mult=2.0 (TP at 104), sl_mult=1.0 (SL at 98).
// Bar has high=103 (<104, TP does NOT trigger) and low=97 (<=98, SL triggers).
// ---------------------------------------------------------------------------
TEST(BacktestEdge, TpSlAtr_OnlySlTriggers) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 1: entry at open=100.0
        {100.0f, 103.0f, 97.0f, 98.0f},   // bar 2: high=103<104 (no TP), low=97<=98 (SL)
        {98.0f, 99.0f, 97.0f, 98.0f},     // bar 3: exit fills at open=98.0
        {98.0f, 99.0f, 97.0f, 98.0f},     // bar 4-9: padding
        {98.0f, 99.0f, 97.0f, 98.0f},    {98.0f, 99.0f, 97.0f, 98.0f}, {98.0f, 99.0f, 97.0f, 98.0f},
        {98.0f, 99.0f, 97.0f, 98.0f},    {98.0f, 99.0f, 97.0f, 98.0f},
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
    config.exit_mode            = StrategyConfig::TP_SL_ATR;
    config.tp_atr_mult          = 2.0f;  // TP at entry + 4.0 = 104.0
    config.sl_atr_mult          = 1.0f;  // SL at entry - 2.0 =  98.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Should have at least one trade when SL triggers";

    // Level-triggered: SL fills at SL level (98.0) on bar 2
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].exit_bar, 2);              // immediate fill on detection bar
    EXPECT_FLOAT_EQ(trades[0].exit_price, 98.0f);  // SL level, not next-bar open
    EXPECT_EQ(trades[0].direction, 1);
    EXPECT_LT(trades[0].gross_pnl, 0.0f) << "SL-only exit should be a losing trade";
}

// =============================================================================
// SHORT POSITION Level-Triggered Exits
//
// Critical: short positions use inverted formulas:
//   SL = entry + dist (triggers on high >= SL)
//   TP = entry - dist (triggers on low <= TP)
// These must be tested independently from long positions.
// =============================================================================

// ---------------------------------------------------------------------------
// Short position, TP_SL_ATR: only TP triggers.
// Short entry at 100.0, ATR=2.0, tp_mult=2.0 (TP at 96), sl_mult=2.0 (SL at 104).
// Bar has low=95 (<=96, TP triggers) and high=103 (<104, SL does NOT trigger).
// ---------------------------------------------------------------------------
TEST(BacktestEdge, Short_TpSlAtr_OnlyTpTriggers) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires (-1 = short)
        {100.0f, 101.0f, 99.0f, 99.5f},   // bar 1: entry at open=100.0
        {99.5f, 103.0f, 95.0f, 97.0f},    // bar 2: low=95<=96 (TP), high=103<104 (no SL)
        {97.0f, 98.0f, 96.0f, 97.0f},     // bar 3: padding
        {97.0f, 98.0f, 96.0f, 97.0f},    {97.0f, 98.0f, 96.0f, 97.0f}, {97.0f, 98.0f, 96.0f, 97.0f},
        {97.0f, 98.0f, 96.0f, 97.0f},    {97.0f, 98.0f, 96.0f, 97.0f}, {97.0f, 98.0f, 96.0f, 97.0f},
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
    config.exit_mode            = StrategyConfig::TP_SL_ATR;
    config.tp_atr_mult          = 2.0f;  // Short TP at entry - 4.0 = 96.0
    config.sl_atr_mult          = 2.0f;  // Short SL at entry + 4.0 = 104.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].exit_bar, 2);              // immediate fill on detection bar
    EXPECT_FLOAT_EQ(trades[0].exit_price, 96.0f);  // TP level for short
    EXPECT_EQ(trades[0].direction, -1);
    // Short PnL: direction * (exit - entry) * point_value = -1 * (96 - 100) * 50 = +200
    EXPECT_GT(trades[0].gross_pnl, 0.0f) << "Short TP exit should be profitable";
    CostModel cost           = es_cost();
    float     expected_gross = cost.price_to_pnl(-1.0f * (96.0f - 100.0f));
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
}

// ---------------------------------------------------------------------------
// Short position, TP_SL_ATR: only SL triggers.
// Short entry at 100.0, ATR=2.0, tp_mult=2.0 (TP at 96), sl_mult=1.5 (SL at 103).
// Bar has high=104 (>=103, SL triggers) and low=97 (>96, TP does NOT trigger).
// ---------------------------------------------------------------------------
TEST(BacktestEdge, Short_TpSlAtr_OnlySlTriggers) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires
        {100.0f, 101.0f, 99.0f, 100.5f},  // bar 1: entry at open=100.0
        {100.5f, 104.0f, 97.0f, 101.0f},  // bar 2: high=104>=103 (SL), low=97>96 (no TP)
        {101.0f, 102.0f, 100.0f, 101.0f}, {101.0f, 102.0f, 100.0f, 101.0f},
        {101.0f, 102.0f, 100.0f, 101.0f}, {101.0f, 102.0f, 100.0f, 101.0f},
        {101.0f, 102.0f, 100.0f, 101.0f}, {101.0f, 102.0f, 100.0f, 101.0f},
        {101.0f, 102.0f, 100.0f, 101.0f},
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
    config.exit_mode            = StrategyConfig::TP_SL_ATR;
    config.tp_atr_mult          = 2.0f;  // Short TP at 96.0
    config.sl_atr_mult          = 1.5f;  // Short SL at entry + 3.0 = 103.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].exit_bar, 2);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 103.0f);  // SL level for short
    EXPECT_EQ(trades[0].direction, -1);
    // Short PnL: -1 * (103 - 100) * 50 = -150
    EXPECT_LT(trades[0].gross_pnl, 0.0f) << "Short SL exit should be a loss";
    CostModel cost           = es_cost();
    float     expected_gross = cost.price_to_pnl(-1.0f * (103.0f - 100.0f));
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
}

// ---------------------------------------------------------------------------
// Short position, TRAILING_STOP.
// Short entry at 100.0, price drops to 95 (best). ATR=2.0, trail_mult=2.0.
// Trail stop = best + dist = 95 + 4 = 99.  Bar has high=100 (>=99, triggered).
// ---------------------------------------------------------------------------
TEST(BacktestEdge, Short_TrailingStop) {
    const int N = 10;
    // Short entry at 100. Best price tracks the lowest low.
    // Bar 2: price drops, best=95, stop=95+4=99, high=97 < 99 -> no trigger
    // Bar 3: high=100 >= 99 -> triggered, fill at 99
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires
        {100.0f, 100.5f, 99.0f, 99.5f},   // bar 1: entry at open=100
        {99.5f, 97.0f, 95.0f, 96.0f},     // bar 2: best=95, stop=99, high=97<99 -> safe
        {96.0f, 100.0f, 95.5f, 98.0f},    // bar 3: high=100>=99 -> triggered
        {98.0f, 99.0f, 97.0f, 98.0f},    {98.0f, 99.0f, 97.0f, 98.0f}, {98.0f, 99.0f, 97.0f, 98.0f},
        {98.0f, 99.0f, 97.0f, 98.0f},    {98.0f, 99.0f, 97.0f, 98.0f}, {98.0f, 99.0f, 97.0f, 98.0f},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = -1.0f;
    std::vector<float> atr(N, 2.0f);
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::TRAILING_STOP;
    config.trailing_atr_mult    = 2.0f;  // trail dist = 2.0 * 2.0 = 4.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].direction, -1);
    // Bar 2: best_price = min(99.0, 95.0) = 95.0, stop = 95+4 = 99, high=97<99 -> safe
    // Bar 3: best stays 95 (low=95.5>95), stop=99, high=100>=99 -> triggered
    EXPECT_EQ(trades[0].exit_bar, 3);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 99.0f);
    // Short PnL: -1 * (99 - 100) * 50 = +50
    EXPECT_GT(trades[0].gross_pnl, 0.0f);
    CostModel cost           = es_cost();
    float     expected_gross = cost.price_to_pnl(-1.0f * (99.0f - 100.0f));
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
}

// =============================================================================
// Same-Bar Entry + Exit
// =============================================================================

// ---------------------------------------------------------------------------
// Entry fills at bar 1 open, and on the same bar the SL level is breached.
// This tests the edge case where Step 1 (entry fill) and Step 2 (exit check)
// both fire on the same loop iteration.
// ---------------------------------------------------------------------------
TEST(BacktestEdge, SameBarEntryAndLevelExit) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires
        {100.0f, 101.0f, 90.0f, 92.0f},   // bar 1: entry at open=100, low=90 breaches SL=96
        {92.0f, 93.0f, 91.0f, 92.0f},     // bar 2: padding
        {92.0f, 93.0f, 91.0f, 92.0f},    {92.0f, 93.0f, 91.0f, 92.0f}, {92.0f, 93.0f, 91.0f, 92.0f},
        {92.0f, 93.0f, 91.0f, 92.0f},    {92.0f, 93.0f, 91.0f, 92.0f}, {92.0f, 93.0f, 91.0f, 92.0f},
        {92.0f, 93.0f, 91.0f, 92.0f},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;  // long signal
    std::vector<float> atr(N, 2.0f);
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::MAX_LOSS_ATR;
    config.max_loss_atr_mult    = 2.0f;  // SL at entry - 4.0 = 96.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Should have a trade even if entry+exit on same bar";

    // Same-bar level exits are suppressed. Entry on bar 1,
    // level-exit check starts on bar 2. Bar 2 opens at 92, already through
    // SL=96, so the stop fills at the open, not at the stop level.
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_EQ(trades[0].exit_bar, 2);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 92.0f);  // gapped through: open
    EXPECT_EQ(trades[0].direction, 1);
    EXPECT_LT(trades[0].gross_pnl, 0.0f) << "Stop-out on bar 2 should be a loss";
}

// =============================================================================
// Session-Close Force-Close Boundary
//
// When sessions are provided, the last 30s bar before the close (time of day
// in [session_close_sec - 30, session_close_sec)) force-closes open positions
// (pending exit at next bar open). This tests the exact boundary: a bar AT
// session_close - 30 triggers, a bar AT session_close - 31 does not. The exit
// fills at the close bar's own close.
// =============================================================================

TEST(BacktestEdge, SessionClose_ExactBoundary) {
    // 10 bars.  Session close = 61200 (17:00 ET).
    // Bar 0: signal fires. Bar 1: entry fills.
    // Bar 5: time_of_day = 61170 (= 61200 - 30) -> force-close should fire
    // Bar 6: exit fills at open.
    const int                         N = 10;
    std::vector<std::array<float, 4>> ohlc;
    for (int i = 0; i < N; ++i) ohlc.push_back({100.0f, 101.0f, 99.0f, 100.0f});
    auto bars = make_bars_ohlc(ohlc);

    // Build SessionInfo with session_close_sec = 61200 (17:00).
    // Bars 0-4: time_of_day well before close.
    // Bar 5: time_of_day = 61170 = 61200 - 30 (exact boundary).
    // Bar 6+: time_of_day = 61200+ (past close).
    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i]                   = {};
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].session_close_sec = 61200;             // 17:00 ET
        sessions[i].time_of_day_sec   = 34200 + i * 5400;  // spread across the day
    }
    // Override bar 5 to be at exact boundary
    sessions[5].time_of_day_sec = 61170;  // 61200 - 30 = exact trigger point
    // Bar 4 is just below the boundary
    sessions[4].time_of_day_sec = 61169;  // 61200 - 31 = should NOT trigger
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;  // long signal

    // Use SIGNAL_REVERSAL exit (signal stays positive, so only session close exits)
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 100;  // large timeout so only session close fires

    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, nullptr, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    // The force-close fires on bar 5 and fills at bar 5's close (a
    // market-on-close exit), not after the break.
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_EQ(trades[0].exit_bar, 5);
    EXPECT_FLOAT_EQ(trades[0].exit_price, bars[5].close);
    EXPECT_EQ(trades[0].fill_type, FillType::SESSION_CLOSE);
}

TEST(BacktestEdge, SessionClose_JustBelowBoundary_NoForceClose) {
    // Same setup but ALL bars have time_of_day < session_close - 30.
    // The position should be force-closed at range end, not by session logic.
    const int                         N = 8;
    std::vector<std::array<float, 4>> ohlc;
    for (int i = 0; i < N; ++i) ohlc.push_back({100.0f, 101.0f, 99.0f, 100.0f});
    auto bars = make_bars_ohlc(ohlc);

    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i]                   = {};
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].session_close_sec = 61200;
        // All bars well before the close boundary
        sessions[i].time_of_day_sec = 34200 + i * 1800;
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 100;

    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, nullptr, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    // No session close fired; position force-closed at range end
    EXPECT_EQ(trades[0].exit_bar, N - 1);
    EXPECT_EQ(trades[0].fill_type, FillType::FORCED_CLOSE);
}

TEST(BacktestEdge, HaltFlag_SuppressesNewEntries) {
    // Verify that the halt flag prevents new entries but does not affect
    // existing positions or pending exits.
    const int                         N = 10;
    std::vector<std::array<float, 4>> ohlc;
    for (int i = 0; i < N; ++i) ohlc.push_back({100.0f, 101.0f, 99.0f, 100.0f});
    auto bars = make_bars_ohlc(ohlc);

    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i]                   = {};
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].session_close_sec = 61200;
        sessions[i].time_of_day_sec   = 34200 + i * 1800;
        sessions[i].halt_flag         = 0;
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);
    // Set halt on bars 0 and 1 (signal bar and entry bar)
    sessions[0].halt_flag = 1;
    sessions[1].halt_flag = 1;

    // Strong buy signal on every bar
    std::vector<float> signal(N, 1.0f);
    const float*       signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 3;

    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, nullptr, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    // First entry should be delayed: signal on bar 0 suppressed by halt,
    // signal on bar 1 also suppressed, first valid signal on bar 2,
    // entry fills on bar 3.
    ASSERT_GE(trades.size(), 1u);
    EXPECT_GE(trades[0].entry_bar, 3) << "Entry must be delayed past halted bars";
}

// =============================================================================
// Flat-at-close applies only to the bar that spans the close
//
// A US futures session runs 18:00 -> 17:00 ET, so evening bars have a larger
// time of day than the 17:00 close. A position opened in the evening must be
// held through the night and closed only on the 16:59:30 bar.
// =============================================================================

TEST(BacktestEdge, SessionClose_EveningBarsDoNotForceExit) {
    // Bars at 19:00, 19:00:30, ... then the 16:59:30 close bar.
    const int                         N = 12;
    std::vector<std::array<float, 4>> ohlc(N, {100.0f, 101.0f, 99.0f, 100.0f});
    auto                              bars = make_bars_ohlc(ohlc);

    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i]                   = {};
        sessions[i].session_close_sec = 61200;
        sessions[i].time_of_day_sec   = 68400 + i * 30;  // 19:00 ET onward
        sessions[i].intra_session_idx = i;
    }
    sessions[9].time_of_day_sec  = 61170;  // 16:59:30, last bar before the close
    sessions[10].time_of_day_sec = 64800;  // 18:00, next session
    sessions[10].session_day     = 1;
    sessions[11].time_of_day_sec = 64830;
    sessions[11].session_day     = 1;
    stamp_session_flags(sessions.data(), N, SessionModel::us_futures(), 30);

    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 100;

    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, nullptr, config, es_cost(), 1);
    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_EQ(trades[0].exit_bar, 9) << "exit fills at the close of the close bar";
    EXPECT_FLOAT_EQ(trades[0].exit_price, bars[9].close);
    EXPECT_EQ(trades[0].fill_type, FillType::SESSION_CLOSE);
}

// Continuous (24/7) markets have no daily close, so nothing is force-closed.
TEST(BacktestEdge, ContinuousModelNeverForcesFlat) {
    const int                         N = 50;
    std::vector<std::array<float, 4>> ohlc(N, {100.0f, 101.0f, 99.0f, 100.0f});
    auto                              bars = make_bars_ohlc(ohlc);
    std::vector<SessionInfo>          sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i]                   = {};
        sessions[i].session_close_sec = -1;
        sessions[i].time_of_day_sec   = 86400 - 25 * 60 + i * 60;  // crosses UTC midnight
        if (sessions[i].time_of_day_sec >= 86400) sessions[i].time_of_day_sec -= 86400;
    }
    stamp_session_flags(sessions.data(), N, SessionModel::continuous(), 60);
    for (const auto& s : sessions) EXPECT_EQ(s.flags & SESSION_CLOSE_BAR, 0);

    std::vector<float> signal(N, 0.0f);
    signal[0]                = 1.0f;
    const float*   signals[] = {signal.data()};
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 1000;
    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, nullptr, config, es_cost(), 1);
    ASSERT_EQ(engine.last_trades().size(), 1u);
    EXPECT_EQ(engine.last_trades()[0].exit_bar, N - 1) << "closed only at the end of data";
}

// =============================================================================
// Engine-computed exit ATR
//
// Pipeline strategies leave atr_series_index at -1, so ATR-based exits rely on
// the engine's own Wilder ATR(EXIT_ATR_PERIOD) from completed bars. Without it
// every ATR exit preset would silently degrade to "hold until the close".
// =============================================================================

TEST(BacktestEdge, AtrExitsUseEngineAtrByDefault) {
    // 30 quiet bars (range 2.0 -> ATR 2.0), then a sharp drop.
    const int                         N = 40;
    std::vector<std::array<float, 4>> ohlc(N, {100.0f, 101.0f, 99.0f, 100.0f});
    for (int i = 30; i < N; ++i) ohlc[i] = {95.0f, 96.0f, 90.0f, 91.0f};
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[20]             = 1.0f;  // long; fills at bar 21, after the ATR has warmed up
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::MAX_LOSS_ATR;
    config.max_loss_atr_mult    = 2.0f;  // stop = 100 - 2 * 2.0 = 96
    ASSERT_EQ(config.atr_series_index, -1);

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);
    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].entry_bar, 21);
    EXPECT_EQ(trades[0].exit_bar, 30) << "the stop must fire on the drop, not at the end";
    EXPECT_FLOAT_EQ(trades[0].exit_price, 95.0f) << "gapped below 96: fills at the open";
    EXPECT_EQ(trades[0].fill_type, FillType::LEVEL_EXIT);
}

TEST(BacktestEdge, AtrExitsWaitForAtrWarmup) {
    // Entry on bar 2: the engine ATR is not warm yet, so a stop cannot fire
    // even though the bar range is huge.
    const int                         N = 8;
    std::vector<std::array<float, 4>> ohlc(N, {100.0f, 101.0f, 99.0f, 100.0f});
    ohlc[3]                 = {100.0f, 100.0f, 80.0f, 85.0f};
    auto               bars = make_bars_ohlc(ohlc);
    std::vector<float> signal(N, 0.0f);
    signal[1]                = 1.0f;
    const float*   signals[] = {signal.data()};
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.exit_mode            = StrategyConfig::MAX_LOSS_ATR;
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);
    ASSERT_EQ(engine.last_trades().size(), 1u);
    EXPECT_EQ(engine.last_trades()[0].fill_type, FillType::FORCED_CLOSE);
}
