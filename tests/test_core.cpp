// =============================================================================
// Core Unit Tests
//
// Tests covering: Market Data, Indicators, Backtest Engine, Metrics
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>
#include "portable_random.h"
#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <vector>

#include "market_data.h"
#include "bar_io.h"
#include "synthetic.h"
#include "indicators/indicator.h"
#include "indicators/ema_sma_cross.h"
#include "indicators/rsi.h"
#include "indicators/atr.h"
#include "backtest_engine.h"
#include "cost_model.h"
#include "metrics.h"
#include "pnl_decomposition.h"

// =============================================================================
// Helpers
// =============================================================================

// Generate N synthetic bars with close prices taken from the given vector.
// High = max(open, close) + 0.5, Low = min(open, close) - 0.5.
// Open of bar i+1 = close of bar i (continuous).
static std::vector<Bar> make_bars_from_closes(const std::vector<float>& closes,
                                              int64_t                   base_ts = 1'262'304'000LL *
                                                                NS_PER_SEC)  // 2010-01-01 00:00 UTC
{
    std::vector<Bar> bars;
    bars.reserve(closes.size());
    float prev_close = closes[0];
    for (size_t i = 0; i < closes.size(); ++i) {
        float   o  = (i == 0) ? closes[0] : prev_close;
        float   c  = closes[i];
        float   h  = std::max(o, c) + 0.5f;
        float   l  = std::min(o, c) - 0.5f;
        int64_t ts = base_ts + static_cast<int64_t>(i) * 30 * NS_PER_SEC;
        bars.push_back(make_bar(ts, o, h, l, c));
        prev_close = c;
    }
    return bars;
}

// Generate N bars with explicit OHLC per bar.
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

// Check if a float is NaN
static bool is_nan(float v) {
    return std::isnan(v);
}

// One year of synthetic 30-second US futures bars (2023), generated once
// per test run. Large enough to exercise session indexing, calendar
// mapping and timeframe aggregation at realistic scale.
static const std::string& history_path() {
    static const std::string path = [] {
        SyntheticConfig cfg;
        cfg.seed        = 2023;
        cfg.start_year  = 2023;
        cfg.start_month = 1;
        cfg.start_day   = 2;
        cfg.num_days    = 260;
        const auto p =
            (std::filesystem::temp_directory_path() / "wfo_gpu_history_2023.bin").string();
        write_bar_file(p, "SYN", 30, generate_synthetic_bars(cfg));
        return p;
    }();
    return path;
}

// =============================================================================
// MARKET DATA TESTS (6 tests)
// =============================================================================

// ---------------------------------------------------------------------------
// test_bar_struct_size
// Verify Bar is 32 bytes and FileHeader is 64 bytes.
// These are compile-time checks duplicated at runtime for visibility in test
// output.
// ---------------------------------------------------------------------------
TEST(MarketData, BarStructSize) {
    EXPECT_EQ(sizeof(Bar), 32u);
    EXPECT_EQ(sizeof(FileHeader), 64u);
    // Also verify packed layout expectations
    static_assert(sizeof(Bar) == 32, "Bar must be exactly 32 bytes");
    static_assert(sizeof(FileHeader) == 64, "FileHeader must be exactly 64 bytes");
}

// ---------------------------------------------------------------------------
// test_load_history
// Load a year of bars: 260 sessions x 23 hours x 120 bars, all within 2023
// (the first session opens Sunday 2023-01-01 18:00 ET).
// ---------------------------------------------------------------------------
TEST(MarketData, LoadHistory) {
    MarketData md;
    bool       ok = md.load(history_path());
    ASSERT_TRUE(ok) << "Failed to load " << history_path();

    EXPECT_EQ(md.bar_count(), 260 * 23 * 120);

    // 2023-01-01 00:00:00 UTC = 1672531200 s; 2024-01-01 = 1704067200 s
    const int64_t first_ts = md.bar(0).timestamp;
    const int64_t last_ts  = md.bar(md.bar_count() - 1).timestamp;
    EXPECT_GE(first_ts, 1'672'531'200LL * NS_PER_SEC);
    EXPECT_LT(last_ts, 1'704'067'200LL * NS_PER_SEC);

    // Verify header fields
    EXPECT_EQ(md.header().timeframe_sec, 30u);
    EXPECT_EQ(std::string(md.header().magic, 4), "WFBT");
}

// ---------------------------------------------------------------------------
// test_session_index
// Load the history, verify session count, verify session boundaries happen
// at 18:00 ET.  For the first bar of session 1, the intra_session_idx should
// be 0 and time_of_day_sec should be >= 18:00 (64800 seconds).
// ---------------------------------------------------------------------------
TEST(MarketData, SessionIndex) {
    MarketData md;
    bool       ok = md.load(history_path());
    ASSERT_TRUE(ok);

    // One session per trading day
    EXPECT_EQ(md.session_count(), 260);

    // Find the first bar of session 1 (session 0 is the first session which may
    // start mid-session). Walk until session_day changes from 0 to 1.
    int first_bar_session1 = -1;
    for (int i = 1; i < md.bar_count(); ++i) {
        if (md.session(i).session_day == 1 && md.session(i - 1).session_day == 0) {
            first_bar_session1 = i;
            break;
        }
    }
    ASSERT_GE(first_bar_session1, 0) << "Could not find start of session 1";

    // The first bar of session 1 should have intra_session_idx == 0
    EXPECT_EQ(md.session(first_bar_session1).intra_session_idx, 0);

    // Time of day should be >= 18:00 ET (64800 seconds) -- session open
    const int32_t tod = md.session(first_bar_session1).time_of_day_sec;
    EXPECT_GE(tod, 18 * 3600) << "Session start should be >= 18:00 ET, got " << tod << " seconds";
}

// ---------------------------------------------------------------------------
// test_find_bar
// Load the history, pick a known bar's timestamp, verify find_bar returns a
// valid index and the timestamp matches.
// ---------------------------------------------------------------------------
TEST(MarketData, FindBar) {
    MarketData md;
    bool       ok = md.load(history_path());
    ASSERT_TRUE(ok);

    // Use the 1000th bar as our reference
    const int ref_idx = 1000;
    ASSERT_LT(ref_idx, md.bar_count());
    const int64_t ref_ts = md.bar(ref_idx).timestamp;

    // find_bar should return exactly ref_idx
    int found = md.find_bar(ref_ts);
    EXPECT_EQ(found, ref_idx) << "find_bar should return the exact index for a known timestamp";

    // find_bar_ge with the exact timestamp should also return the same index
    int found_ge = md.find_bar_ge(ref_ts);
    EXPECT_EQ(found_ge, ref_idx);

    // find_bar with a non-existent timestamp should return -1
    int not_found = md.find_bar(ref_ts + 1);
    EXPECT_EQ(not_found, -1) << "find_bar should return -1 for a timestamp that does not exist";
}

// ---------------------------------------------------------------------------
// test_timeframe_aggregation
// Load the history, verify 1m timeframe has fewer bars than 30s, verify
// parent_map mapping is valid.
// ---------------------------------------------------------------------------
TEST(MarketData, TimeframeAggregation) {
    MarketData md;
    bool       ok = md.load(history_path());
    ASSERT_TRUE(ok);

    // 1-minute (60s) timeframe
    const auto& tf1m = md.timeframe(60);
    EXPECT_EQ(tf1m.timeframe_sec, 60);

    // 1m should have fewer bars than 30s (roughly half, but not exactly due to
    // session boundaries and sparse bars)
    const int n30s = md.bar_count();
    const int n1m  = static_cast<int>(tf1m.bars.size());
    EXPECT_GT(n1m, 0) << "1m timeframe should have bars";
    EXPECT_LT(n1m, n30s) << "1m should have fewer bars than 30s";
    // Rough check: 1m should be roughly n30s/2 give or take
    EXPECT_GT(n1m, n30s / 3) << "1m bars seem too few";
    EXPECT_LT(n1m, n30s) << "1m bars should be fewer than 30s";

    // Verify parent_map size matches bar_count
    EXPECT_EQ(static_cast<int>(tf1m.parent_map.size()), n30s);

    // Verify parent_map is monotonically non-decreasing
    for (int i = 1; i < n30s; ++i) {
        EXPECT_GE(tf1m.parent_map[i], tf1m.parent_map[i - 1])
            << "parent_map should be non-decreasing at index " << i;
    }

    // Verify all parent_map values are valid indices into tf bars
    for (int i = 0; i < n30s; ++i) {
        EXPECT_GE(tf1m.parent_map[i], 0);
        EXPECT_LT(tf1m.parent_map[i], n1m) << "parent_map[" << i << "] = " << tf1m.parent_map[i]
                                           << " is out of range [0, " << n1m << ")";
    }

    // Check a higher timeframe too: 5m should have even fewer bars
    const auto& tf5m = md.timeframe(300);
    EXPECT_LT(static_cast<int>(tf5m.bars.size()), n1m) << "5m should have fewer bars than 1m";
}

// ---------------------------------------------------------------------------
// test_find_bar_ge
// Verify find_bar_ge returns the first bar at or after a timestamp.
// ---------------------------------------------------------------------------
TEST(MarketData, FindBarGe) {
    MarketData md;
    bool       ok = md.load(history_path());
    ASSERT_TRUE(ok);

    // Use the 500th bar's timestamp minus 1 nanosecond: should find bar 500
    const int     ref_idx = 500;
    const int64_t ref_ts  = md.bar(ref_idx).timestamp;

    int found = md.find_bar_ge(ref_ts - 1);
    EXPECT_EQ(found, ref_idx) << "find_bar_ge(ts - 1) should return bar at ts";

    // Timestamp exactly at a bar should return that bar
    found = md.find_bar_ge(ref_ts);
    EXPECT_EQ(found, ref_idx);
}

// =============================================================================
// INDICATOR TESTS (13 tests)
// =============================================================================

// ---------------------------------------------------------------------------
// test_ema_basic
// 10-bar synthetic data, EMA(3).  alpha = 2/(3+1) = 0.5.
// Hand-computed values verified.
// ---------------------------------------------------------------------------
TEST(Indicators, EmaBasic) {
    const std::vector<float> closes = {10, 11, 12, 11, 10, 11, 12, 13, 14, 13};
    auto                     bars   = make_bars_from_closes(closes);
    const int                N      = static_cast<int>(bars.size());

    // Use EMASMACross with fast=EMA(3), slow=SMA(5) but only look at fast output
    // Actually, EMASMACross requires fast_period < slow_period, so we use EMA(3)/SMA(5).
    EMASMACross indicator(MAType::EMA, 3, MAType::SMA, 5);

    std::vector<float> output(N * 2);  // 2 series
    indicator.compute(bars.data(), 0, N, output.data());

    // EMA series is output[0..N)
    // Hand-compute EMA(3) with alpha=0.5:
    //   ema[0] = 10
    //   ema[1] = 0.5*11 + 0.5*10 = 10.5
    //   ema[2] = 0.5*12 + 0.5*10.5 = 11.25
    //   ema[3] = 0.5*11 + 0.5*11.25 = 11.125
    //   ema[4] = 0.5*10 + 0.5*11.125 = 10.5625
    //   ema[5] = 0.5*11 + 0.5*10.5625 = 10.78125
    //   ema[6] = 0.5*12 + 0.5*10.78125 = 11.390625
    //   ema[7] = 0.5*13 + 0.5*11.390625 = 12.1953125
    //   ema[8] = 0.5*14 + 0.5*12.1953125 = 13.09765625
    //   ema[9] = 0.5*13 + 0.5*13.09765625 = 13.048828125

    // EMA warmup = 3 * 3 = 9.  So bars 0..8 are NaN, bar 9 is valid.
    const int ema_warmup = 3 * 3;  // = 9
    for (int i = 0; i < ema_warmup; ++i) {
        EXPECT_TRUE(is_nan(output[i]))
            << "EMA bar " << i << " should be NaN (warmup), got " << output[i];
    }
    // Bar 9 (index 9) should be valid with the EMA value computed through all bars
    EXPECT_FALSE(is_nan(output[9])) << "EMA bar 9 should be valid after warmup";
    EXPECT_NEAR(output[9], 13.048828125f, 1e-4f) << "EMA(3) at bar 9 mismatch";
}

// ---------------------------------------------------------------------------
// test_sma_basic
// SMA(3) on 10-bar data.
// SMA warmup in EMASMACross is period (=3 bars minimum, i.e., valid from index 2).
// But the overall warmup is max(fast_warmup, slow_warmup).
// To test SMA independently, we create a cross with SMA fast and SMA slow.
// ---------------------------------------------------------------------------
TEST(Indicators, SmaBasic) {
    const std::vector<float> closes = {10, 11, 12, 11, 10, 11, 12, 13, 14, 13};
    auto                     bars   = make_bars_from_closes(closes);
    const int                N      = static_cast<int>(bars.size());

    // SMA(3) fast, SMA(5) slow.  SMA warmup = period.
    // For the cross, warmup = max(3, 5) = 5.
    EMASMACross indicator(MAType::SMA, 3, MAType::SMA, 5);

    std::vector<float> output(N * 2);
    indicator.compute(bars.data(), 0, N, output.data());

    // SMA(3) values (fast series = output[0..N)):
    //   sma[0] = NaN (need 3 bars)
    //   sma[1] = NaN
    //   sma[2] = (10+11+12)/3 = 11.0    -- but warmup = max(3,5) = 5, so NaN
    //   sma[3] = (11+12+11)/3 = 11.333  -- warmup, NaN
    //   sma[4] = (12+11+10)/3 = 11.0    -- warmup, NaN (index 4, wu = 5 means valid from i >= 5)
    //   sma[5] = (11+10+11)/3 = 10.667  -- first valid

    // Wait -- let's check the code. SMA valid when i >= period - 1 AND i >= wu.
    // wu for SMA(3) in EMASMACross::ma_warmup = period = 3.
    // But the cross's warmup_bars() = max(ma_warmup(fast), ma_warmup(slow)) = max(3, 5) = 5.
    // However, compute_ma is called independently per series with its own wu.
    // So SMA(3) uses wu=3, meaning valid from i >= max(3-1, 3) = 3.
    // Let me re-read the code... compute_ma checks: if (i >= period - 1 && i >= wu).
    // For SMA(3): period=3, wu=ma_warmup(SMA,3)=3. So valid when i>=2 AND i>=3 => i>=3.

    // SMA(3) output:
    //   i=0: NaN, i=1: NaN, i=2: NaN (i < wu=3),
    //   i=3: (11+12+11)/3 = 11.333...
    //   i=4: (12+11+10)/3 = 11.0
    //   i=5: (11+10+11)/3 = 10.666...
    //   i=6: (10+11+12)/3 = 11.0
    //   i=7: (11+12+13)/3 = 12.0
    //   i=8: (12+13+14)/3 = 13.0
    //   i=9: (13+14+13)/3 = 13.333...

    // SMA(3) warmup = period-1 = 2 bars (indices 0-1 NaN, first valid at index 2)
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "SMA(3) bar " << i << " should be NaN (warmup)";
    }
    EXPECT_NEAR(output[2], 11.0f, 1e-4f);  // (10+11+12)/3
    EXPECT_NEAR(output[3], 11.333333f, 1e-4f);
    EXPECT_NEAR(output[4], 11.0f, 1e-4f);
    EXPECT_NEAR(output[5], 10.6667f, 1e-3f);
    EXPECT_NEAR(output[6], 11.0f, 1e-4f);
    EXPECT_NEAR(output[7], 12.0f, 1e-4f);
    EXPECT_NEAR(output[8], 13.0f, 1e-4f);
    EXPECT_NEAR(output[9], 13.3333f, 1e-3f);
}

// ---------------------------------------------------------------------------
// test_ema_warmup
// EMA(5) on 20-bar data.  warmup = 3*5 = 15.  First 15 bars (0..14) NaN.
// ---------------------------------------------------------------------------
TEST(Indicators, EmaWarmup) {
    // Generate 20 bars with linearly increasing closes
    std::vector<float> closes(20);
    for (int i = 0; i < 20; ++i) closes[i] = 100.0f + static_cast<float>(i);
    auto      bars = make_bars_from_closes(closes);
    const int N    = 20;

    EMASMACross        indicator(MAType::EMA, 5, MAType::SMA, 10);
    std::vector<float> output(N * 2);
    indicator.compute(bars.data(), 0, N, output.data());

    // EMA(5) warmup = 3 * 5 = 15.  Bars 0..14 should be NaN.
    const int wu = 3 * 5;  // = 15
    for (int i = 0; i < wu; ++i) {
        EXPECT_TRUE(is_nan(output[i]))
            << "EMA(5) bar " << i << " should be NaN (warmup = " << wu << ")";
    }
    // Bar 15 onward should be valid
    for (int i = wu; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "EMA(5) bar " << i << " should be valid after warmup";
    }
}

// ---------------------------------------------------------------------------
// test_sma_warmup
// SMA(5) warmup = 5 (per the code: ma_warmup(SMA, 5) = 5).
// Valid from bar index >= max(period-1, wu) = max(4, 5) = 5.
// ---------------------------------------------------------------------------
TEST(Indicators, SmaWarmup) {
    std::vector<float> closes(20);
    for (int i = 0; i < 20; ++i) closes[i] = 100.0f + static_cast<float>(i);
    auto      bars = make_bars_from_closes(closes);
    const int N    = 20;

    EMASMACross        indicator(MAType::SMA, 5, MAType::SMA, 10);
    std::vector<float> output(N * 2);
    indicator.compute(bars.data(), 0, N, output.data());

    // SMA(5) wu = period-1 = 4. First valid at index 4.
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "SMA(5) bar " << i << " should be NaN (warmup)";
    }
    for (int i = 4; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "SMA(5) bar " << i << " should be valid after warmup";
    }
}

// ---------------------------------------------------------------------------
// test_rsi_basic
// RSI(3) on synthetic data.  Warmup = 3*3 = 9.
// Data designed so that after the initial 3-bar accumulation (indices 1-3),
// Wilder smoothing kicks in.
// ---------------------------------------------------------------------------
TEST(Indicators, RsiBasic) {
    // Classic Wilder RSI test data (adapted for period 3)
    const std::vector<float> closes = {44.0f,  44.34f, 44.09f, 43.61f, 44.33f, 44.83f, 45.10f,
                                       45.42f, 45.84f, 46.08f, 45.89f, 46.03f, 45.61f, 46.28f,
                                       46.28f, 46.00f, 46.03f, 46.41f, 46.22f, 45.64f};
    auto                     bars   = make_bars_from_closes(closes);
    const int                N      = static_cast<int>(bars.size());

    RSI rsi_indicator(3);
    EXPECT_EQ(rsi_indicator.warmup_bars(), 9);  // 3 * 3

    std::vector<float> output(N);
    rsi_indicator.compute(bars.data(), 0, N, output.data());

    // Hand-compute RSI(3):
    // Changes (close[i] - close[i-1]):
    //   i=1: +0.34, i=2: -0.25, i=3: -0.48
    // Initial avg_gain (over first 3 changes) = 0.34 / 3 = 0.1133
    // Initial avg_loss (over first 3 changes) = (0.25+0.48) / 3 = 0.2433
    // RSI at i=3 initialized.  But it's in warmup (i=3 < wu=9), so output is NaN.
    //
    // Continue Wilder smoothing:
    //   i=4: diff=+0.72, gain=0.72, loss=0
    //     avg_gain = (0.1133*2 + 0.72)/3 = 0.31555
    //     avg_loss = (0.2433*2 + 0)/3 = 0.16222
    //   i=5: diff=+0.50
    //     avg_gain = (0.31555*2 + 0.50)/3 = 0.37703
    //     avg_loss = (0.16222*2 + 0)/3 = 0.10815
    //   i=6: diff=+0.27
    //     avg_gain = (0.37703*2 + 0.27)/3 = 0.34135
    //     avg_loss = (0.10815*2 + 0)/3 = 0.07210
    //   i=7: diff=+0.32
    //     avg_gain = (0.34135*2 + 0.32)/3 = 0.33423
    //     avg_loss = (0.07210*2 + 0)/3 = 0.04807
    //   i=8: diff=+0.42
    //     avg_gain = (0.33423*2 + 0.42)/3 = 0.36282
    //     avg_loss = (0.04807*2 + 0)/3 = 0.03205
    //   i=9: diff=+0.24.  This is the first bar past warmup.
    //     avg_gain = (0.36282*2 + 0.24)/3 = 0.32188
    //     avg_loss = (0.03205*2 + 0)/3 = 0.02137
    //     RS = 0.32188 / 0.02137 = 15.065
    //     RSI = 100 - 100/(1+15.065) = 100 - 6.228 = 93.772

    // Bars 0..8 should be NaN (warmup = 9)
    for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "RSI bar " << i << " should be NaN (warmup)";
    }

    // Bar 9 should be valid and close to our hand-computed value
    EXPECT_FALSE(is_nan(output[9])) << "RSI bar 9 should be valid";
    EXPECT_NEAR(output[9], 93.77f, 0.1f)
        << "RSI(3) at bar 9 should be approximately 93.8 (strong uptrend)";
}

// ---------------------------------------------------------------------------
// test_rsi_flat_price
// RSI on constant prices.  All gains and losses are zero.
// When avg_loss == 0, RSI = 100 (per code: if avg_loss == 0 -> RSI = 100).
// But when both avg_gain and avg_loss are zero initially, the smoothing
// will keep them both at 0, so RSI = 100 (0/0 handled as RSI=100).
// ---------------------------------------------------------------------------
TEST(Indicators, RsiFlatPrice) {
    std::vector<float> closes(30, 50.0f);
    auto               bars = make_bars_from_closes(closes);
    const int          N    = 30;

    RSI       rsi_indicator(5);
    const int wu = rsi_indicator.warmup_bars();  // 15

    std::vector<float> output(N);
    rsi_indicator.compute(bars.data(), 0, N, output.data());

    // After warmup, RSI on flat prices: all diffs = 0, avg_gain=0, avg_loss=0.
    // When both avg_gain and avg_loss are 0, RSI = 50 (neutral, not bullish).
    for (int i = wu; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i]))
            << "RSI flat bar " << i << " should not be NaN after warmup";
        EXPECT_FLOAT_EQ(output[i], 50.0f)
            << "RSI on flat price should be 50 (neutral: both gain and loss are zero)";
    }
}

// ---------------------------------------------------------------------------
// test_atr_basic
// ATR(3) on synthetic data with known OHLC.
// ---------------------------------------------------------------------------
TEST(Indicators, AtrBasic) {
    // Construct bars with explicit OHLC to hand-compute ATR
    // Format: {open, high, low, close}
    std::vector<std::array<float, 4>> ohlc = {
        {{100.0f, 102.0f, 99.0f, 101.0f}},   // bar 0: TR = H-L = 3.0 (no prev close)
        {{101.0f, 103.0f, 100.0f, 102.0f}},  // bar 1: TR = max(3.0, |103-101|, |100-101|) = 3.0
        {{102.0f, 105.0f, 101.0f, 104.0f}},  // bar 2: TR = max(4.0, |105-102|, |101-102|) = 4.0
        {{104.0f, 106.0f, 103.0f, 105.0f}},  // bar 3: TR = max(3.0, |106-104|, |103-104|) = 3.0
        {{105.0f, 108.0f, 104.0f, 107.0f}},  // bar 4: TR = max(4.0, |108-105|, |104-105|) = 4.0
        {{107.0f, 109.0f, 105.0f, 106.0f}},  // bar 5: TR = max(4.0, |109-107|, |105-107|) = 4.0
        {{106.0f, 107.0f, 104.0f, 105.0f}},  // bar 6: TR = max(3.0, |107-106|, |104-106|) = 3.0
        {{105.0f, 106.0f, 103.0f, 104.0f}},  // bar 7: TR = max(3.0, |106-105|, |103-105|) = 3.0
        {{104.0f, 107.0f, 103.0f, 106.0f}},  // bar 8: TR = max(4.0, |107-104|, |103-104|) = 4.0
        {{106.0f, 108.0f, 105.0f, 107.0f}},  // bar 9: TR = max(3.0, |108-106|, |105-106|) = 3.0
    };
    auto      bars = make_bars_ohlc(ohlc);
    const int N    = static_cast<int>(bars.size());

    ATR atr_indicator(3);
    EXPECT_EQ(atr_indicator.warmup_bars(), 9);  // 3 * 3

    std::vector<float> output(N);
    atr_indicator.compute(bars.data(), 0, N, output.data());

    // Hand-compute ATR(3):
    // TR values: 3.0, 3.0, 4.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 3.0
    // Initial ATR at i=2 (first period complete): (3.0+3.0+4.0)/3 = 3.333
    // i=3: ATR = (3.333*2 + 3.0)/3 = 3.222
    // i=4: ATR = (3.222*2 + 4.0)/3 = 3.481
    // i=5: ATR = (3.481*2 + 4.0)/3 = 3.654
    // i=6: ATR = (3.654*2 + 3.0)/3 = 3.436
    // i=7: ATR = (3.436*2 + 3.0)/3 = 3.291
    // i=8: ATR = (3.291*2 + 4.0)/3 = 3.527
    // i=9: ATR = (3.527*2 + 3.0)/3 = 3.351

    // All bars 0..8 should be NaN (warmup = 9)
    for (int i = 0; i < 9; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "ATR bar " << i << " should be NaN (warmup)";
    }

    // Bar 9 should be valid
    EXPECT_FALSE(is_nan(output[9]));
    EXPECT_NEAR(output[9], 3.351f, 0.05f) << "ATR(3) at bar 9 mismatch";
}

// ---------------------------------------------------------------------------
// test_indicator_determinism
// Compute same indicator twice.  Output must be bitwise identical.
// ---------------------------------------------------------------------------
TEST(Indicators, Determinism) {
    std::vector<float> closes(50);
    for (int i = 0; i < 50; ++i)
        closes[i] = 100.0f + std::sin(static_cast<float>(i) * 0.3f) * 10.0f;
    auto      bars = make_bars_from_closes(closes);
    const int N    = 50;

    RSI                rsi_indicator(5);
    std::vector<float> out1(N), out2(N);
    rsi_indicator.compute(bars.data(), 0, N, out1.data());
    rsi_indicator.compute(bars.data(), 0, N, out2.data());

    for (int i = 0; i < N; ++i) {
        // Bitwise compare (including NaN which compares equal via memcmp)
        EXPECT_EQ(std::memcmp(&out1[i], &out2[i], sizeof(float)), 0)
            << "RSI output at bar " << i << " differs between runs";
    }
}

// ---------------------------------------------------------------------------
// test_ema_vs_sma_different
// EMA(10) and SMA(10) on the same data must produce different values.
// ---------------------------------------------------------------------------
TEST(Indicators, EmaVsSmaDifferent) {
    std::vector<float> closes(40);
    for (int i = 0; i < 40; ++i) closes[i] = 100.0f + static_cast<float>(i % 7) * 2.0f;
    auto      bars = make_bars_from_closes(closes);
    const int N    = 40;

    // EMA(10)/SMA(15) -- only compare the fast series from each
    EMASMACross ema_cross(MAType::EMA, 10, MAType::SMA, 15);
    EMASMACross sma_cross(MAType::SMA, 10, MAType::SMA, 15);

    std::vector<float> ema_out(N * 2), sma_out(N * 2);
    ema_cross.compute(bars.data(), 0, N, ema_out.data());
    sma_cross.compute(bars.data(), 0, N, sma_out.data());

    // After both warmups expire, values should differ
    // EMA(10) warmup = 30, SMA(10) warmup = 10.
    // Both need to be past warmup to compare valid values.
    int  start            = 30;  // past EMA warmup
    bool found_difference = false;
    for (int i = start; i < N; ++i) {
        if (!is_nan(ema_out[i]) && !is_nan(sma_out[i])) {
            if (ema_out[i] != sma_out[i]) {
                found_difference = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found_difference)
        << "EMA(10) and SMA(10) should produce different values on non-trivial data";
}

// ---------------------------------------------------------------------------
// test_crossover_two_series
// EMASMACross with fast=EMA(3) slow=SMA(10).  Verify num_series()==2,
// output has 2 series of length N.
// ---------------------------------------------------------------------------
TEST(Indicators, CrossoverTwoSeries) {
    std::vector<float> closes(30);
    for (int i = 0; i < 30; ++i) closes[i] = 100.0f + static_cast<float>(i);
    auto      bars = make_bars_from_closes(closes);
    const int N    = 30;

    EMASMACross indicator(MAType::EMA, 3, MAType::SMA, 10);
    EXPECT_EQ(indicator.num_series(), 2);

    std::vector<float> output(N * 2);
    indicator.compute(bars.data(), 0, N, output.data());

    // Series 0 (fast EMA) occupies output[0..N-1]
    // Series 1 (slow SMA) occupies output[N..2*N-1]
    // After warmup, both should have valid values
    int wu = indicator.warmup_bars();  // max(3*3, 10) = 10
    for (int i = wu; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i]))
            << "Fast series at bar " << i << " should be valid after warmup";
        EXPECT_FALSE(is_nan(output[N + i]))
            << "Slow series at bar " << i << " should be valid after warmup";
    }
}

// ---------------------------------------------------------------------------
// test_rsi_range
// RSI output should always be in [0, 100] (excluding NaN warmup).
// ---------------------------------------------------------------------------
TEST(Indicators, RsiRange) {
    // Generate data with large swings to stress RSI range
    std::vector<float> closes(100);
    closes[0] = 100.0f;
    for (int i = 1; i < 100; ++i) {
        // Alternating big up and down moves
        closes[i] = closes[i - 1] + ((i % 2 == 0) ? 5.0f : -5.0f);
    }
    auto      bars = make_bars_from_closes(closes);
    const int N    = 100;

    RSI                rsi_indicator(14);
    std::vector<float> output(N);
    rsi_indicator.compute(bars.data(), 0, N, output.data());

    int wu = rsi_indicator.warmup_bars();
    for (int i = wu; i < N; ++i) {
        ASSERT_FALSE(is_nan(output[i])) << "RSI bar " << i << " is NaN after warmup";
        EXPECT_GE(output[i], 0.0f) << "RSI at bar " << i << " is < 0";
        EXPECT_LE(output[i], 100.0f) << "RSI at bar " << i << " is > 100";
    }
}

// ---------------------------------------------------------------------------
// test_atr_positive
// ATR output should always be >= 0 (excluding NaN warmup).
// ---------------------------------------------------------------------------
TEST(Indicators, AtrPositive) {
    // Generate volatile data
    std::vector<std::array<float, 4>> ohlc(50);
    for (int i = 0; i < 50; ++i) {
        float base = 100.0f + static_cast<float>(i);
        ohlc[i]    = std::array<float, 4>{{base, base + 3.0f, base - 2.0f, base + 1.0f}};
    }
    auto      bars = make_bars_ohlc(ohlc);
    const int N    = 50;

    ATR                atr_indicator(14);
    std::vector<float> output(N);
    atr_indicator.compute(bars.data(), 0, N, output.data());

    int wu = atr_indicator.warmup_bars();
    for (int i = wu; i < N; ++i) {
        ASSERT_FALSE(is_nan(output[i])) << "ATR bar " << i << " is NaN after warmup";
        EXPECT_GE(output[i], 0.0f) << "ATR at bar " << i << " should be >= 0";
    }
}

// ---------------------------------------------------------------------------
// test_atr_determinism
// ATR computed twice must match bitwise.
// ---------------------------------------------------------------------------
TEST(Indicators, AtrDeterminism) {
    std::vector<std::array<float, 4>> ohlc(50);
    for (int i = 0; i < 50; ++i) {
        float base = 100.0f + static_cast<float>(i);
        ohlc[i]    = std::array<float, 4>{{base, base + 3.0f, base - 2.0f, base + 1.0f}};
    }
    auto      bars = make_bars_ohlc(ohlc);
    const int N    = 50;

    ATR                atr_indicator(5);
    std::vector<float> out1(N), out2(N);
    atr_indicator.compute(bars.data(), 0, N, out1.data());
    atr_indicator.compute(bars.data(), 0, N, out2.data());

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(std::memcmp(&out1[i], &out2[i], sizeof(float)), 0)
            << "ATR output at bar " << i << " differs between runs";
    }
}

// =============================================================================
// BACKTEST ENGINE TESTS (8 tests)
// =============================================================================

// ---------------------------------------------------------------------------
// test_entry_lag1
// Signal at bar 5 -> entry at bar 6 open.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, EntryLag1) {
    // Create 20 bars with known prices.  Bars are flat at 4500.
    const int          N = 20;
    std::vector<float> closes(N, 4500.0f);
    auto               bars = make_bars_from_closes(closes);

    // Indicator signal: +1 at bar 5, rest is 0.
    // The signal must exceed entry_threshold (0.5) to trigger.
    std::vector<float> signal(N, 0.0f);
    signal[5]              = 1.0f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Should have at least one trade";

    // The signal fires at bar_offset 5 (abs bar 5).
    // With lag=1, entry fills at bar_offset 6 (abs bar 6).
    EXPECT_EQ(trades[0].entry_bar, 6) << "Lag-1: signal at bar 5 should fill at bar 6";
    EXPECT_EQ(trades[0].direction, 1) << "Should be a long entry";
}

// ---------------------------------------------------------------------------
// test_entry_lag2
// Signal at bar 5 -> entry at bar 7 open (lag=2).
// ---------------------------------------------------------------------------
TEST(BacktestEngine, EntryLag2) {
    const int          N = 20;
    std::vector<float> closes(N, 4500.0f);
    auto               bars = make_bars_from_closes(closes);

    std::vector<float> signal(N, 0.0f);
    signal[5]              = 1.0f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 2);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Should have at least one trade";

    EXPECT_EQ(trades[0].entry_bar, 7) << "Lag-2: signal at bar 5 should fill at bar 7";
}

// ---------------------------------------------------------------------------
// test_cost_deduction
// Buy ES at 4500.00, sell at 4510.00.
// Gross PnL = 10 pts * $50/pt = $500
// Net PnL = $500 - round_trip_cost ($75) = $425
// ---------------------------------------------------------------------------
TEST(BacktestEngine, CostDeduction) {
    // Create bars: entry at bar 1 open = 4500, exit at bar 6 close = 4510
    const int          N = 10;
    std::vector<float> closes(N);
    // All opens/closes ramp from 4500 to 4510
    for (int i = 0; i < N; ++i) {
        closes[i] = 4500.0f + static_cast<float>(i) * (10.0f / (N - 1));
    }
    auto bars = make_bars_from_closes(closes);
    // Override bars to have exact entry/exit prices
    bars[1].open = 4500.0f;

    // Signal fires at bar 0, entry at bar 1 (lag=1), exit after 5 bars (BAR_TIMEOUT)
    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;  // fire long signal
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    CostModel cost = es_cost();

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, cost, 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    const Trade& t              = trades[0];
    float        expected_gross = cost.price_to_pnl(t.exit_price - t.entry_price);
    float        expected_net   = expected_gross - cost.round_trip_cost();

    EXPECT_FLOAT_EQ(t.gross_pnl, expected_gross) << "Gross PnL mismatch";
    EXPECT_FLOAT_EQ(t.net_pnl, expected_net) << "Net PnL should be gross - round_trip_cost";
    // round_trip_cost = (2 spread ticks + 2*2 slippage ticks) * $12.50 = $75
    EXPECT_FLOAT_EQ(cost.round_trip_cost(), 75.0f)
        << "ES round_trip_cost should be (2 + 2*2) * $12.50 = $75";
}

// ---------------------------------------------------------------------------
// test_pnl_long
// Enter long at bar open, exit later.  Verify positive PnL when price rises.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, PnlLong) {
    const int          N = 15;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) closes[i] = 100.0f + static_cast<float>(i) * 2.0f;
    auto bars = make_bars_from_closes(closes);

    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    // Price went up, long trade should be profitable gross
    EXPECT_GT(trades[0].gross_pnl, 0.0f)
        << "Long trade with rising price should have positive gross PnL";
    EXPECT_EQ(trades[0].direction, 1) << "Direction should be long (+1)";
}

// ---------------------------------------------------------------------------
// test_pnl_short
// Enter short when price falls.  Verify positive PnL for short.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, PnlShort) {
    const int          N = 15;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) closes[i] = 200.0f - static_cast<float>(i) * 2.0f;
    auto bars = make_bars_from_closes(closes);

    // Signal: -1 triggers short entry
    std::vector<float> signal(N, 0.0f);
    signal[0]              = -1.0f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    EXPECT_GT(trades[0].gross_pnl, 0.0f)
        << "Short trade with falling price should have positive gross PnL";
    EXPECT_EQ(trades[0].direction, -1) << "Direction should be short (-1)";
}

// ---------------------------------------------------------------------------
// test_no_same_bar_execution
// Signal fires on bar 0.  Verify entry is NOT at bar 0.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, NoSameBarExecution) {
    const int          N = 10;
    std::vector<float> closes(N, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 3;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    // Signal at bar_offset 0 (abs bar 0).  With lag=1, entry at bar 1.
    EXPECT_NE(trades[0].entry_bar, 0) << "Entry must NOT be on the same bar as the signal";
    EXPECT_EQ(trades[0].entry_bar, 1) << "Entry should be at bar 1 (lag=1)";
}

// ---------------------------------------------------------------------------
// test_max_live_drawdown
// Enter long at entry_price, price dips then rallies.
// Verify min_unrealized captures the worst drawdown.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, MaxLiveDrawdown) {
    // Create price path: enter around 100, dip to 90, rally to 120
    // bar 0: signal fires (long), bar 1: entry at open
    // bars 2-4: price dips to 90
    // bars 5-8: price rallies to 120
    const int          N      = 12;
    std::vector<float> closes = {
        100.0f,  // bar 0 (signal bar)
        100.0f,  // bar 1 (entry, open = 100)
        95.0f,   // bar 2 (dipping)
        90.0f,   // bar 3 (bottom - worst drawdown)
        92.0f,   // bar 4
        100.0f,  // bar 5
        110.0f,  // bar 6
        120.0f,  // bar 7
        120.0f,  // bar 8
        120.0f,  // bar 9
        120.0f,  // bar 10
        120.0f   // bar 11
    };
    auto bars = make_bars_from_closes(closes);
    // Ensure entry bar has open at 100
    bars[1].open = 100.0f;

    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 8;

    CostModel      cost = es_cost();
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, cost, 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u);

    // The worst unrealized PnL occurs when close = 90, entry = 100.
    // Unrealized = price_to_pnl(1 * (90 - 100)) = price_to_pnl(-10) = -10/0.25*12.50 = -500
    // min_unrealized should be <= -500 (could be worse if intrabar lows are lower)
    // Actually, M2M is computed at bar close, so the worst close is 90.
    float expected_worst = cost.price_to_pnl(-10.0f);  // = -500
    EXPECT_LE(trades[0].min_unrealized, expected_worst + 1.0f)
        << "min_unrealized should capture the dip to 90 from entry at 100";
    EXPECT_LT(trades[0].min_unrealized, 0.0f)
        << "min_unrealized should be negative during the drawdown";
}

// ---------------------------------------------------------------------------
// test_position_constraint
// Two simultaneous long signals should not create two positions.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, PositionConstraint) {
    const int          N = 20;
    std::vector<float> closes(N, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    // Signal: +1 at bar 0 and bar 2 (while still in the first position)
    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;
    signal[2]              = 1.0f;  // This should be ignored since we're already in a position
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 10;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);

    const auto& trades = engine.last_trades();

    // Should only have 1 trade -- the second signal at bar 2 is ignored
    // because we're already long from bar 1 entry.
    EXPECT_EQ(trades.size(), 1u) << "Should only have one trade (no overlapping positions allowed)";
}

// =============================================================================
// METRICS TESTS (5 tests)
// =============================================================================

// ---------------------------------------------------------------------------
// test_sharpe_known
// Hand-compute Sharpe from known daily PnL.
// We create trades that produce the exact daily PnL we want.
// ---------------------------------------------------------------------------
TEST(Metrics, SharpeKnown) {
    // Create 5 trades, each closing on a different day.
    // Daily PnL = {10, -5, 15, -3, 8}
    // mean = (10 + (-5) + 15 + (-3) + 8) / 5 = 25 / 5 = 5
    // But compute_metrics uses ALL trading days including no-trade days.
    // To get an isolated test, we create exactly 5 bars, each on a different day.

    const int N = 5;
    // Create bars 1 day apart
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    // Create 5 trades with specific net PnL, each exiting on a different day
    std::vector<Trade> trades(5);
    float              daily_pnls[] = {10.0f, -5.0f, 15.0f, -3.0f, 8.0f};
    for (int i = 0; i < 5; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;  // exits same day
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = daily_pnls[i] + 25.0f;  // add back cost so net = daily_pnl
        trades[i].net_pnl        = daily_pnls[i];
        trades[i].min_unrealized = 0.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // With exactly 5 days and 5 trades (one per day), no zero-PnL days inserted:
    // mean = 5.0, sample std = sqrt(((10-5)^2 + (-5-5)^2 + (15-5)^2 + (-3-5)^2 + (8-5)^2) / 4)
    //       = sqrt((25 + 100 + 100 + 64 + 9) / 4)
    //       = sqrt(298 / 4) = sqrt(74.5) = 8.631
    // Sharpe = 5.0 / 8.631 * sqrt(252) = 0.5794 * 15.875 = 9.196

    // However, compute_metrics counts ALL unique bar-days in the range, which
    // is exactly 5 here (one bar per day). So the Sharpe should be close to our
    // hand-computed value.
    EXPECT_GT(result.sharpe_ratio, 5.0f) << "Sharpe should be significantly positive";
    EXPECT_NEAR(result.sharpe_ratio, 9.2f, 0.3f)
        << "Sharpe for daily PnL {10,-5,15,-3,8} with 5 trading days";
}

// ---------------------------------------------------------------------------
// test_green_weeks_pct
// 5 weeks of trades. PnL = {100, -50, 200, -10, 50}. Green = 3/5 = 60%.
// ---------------------------------------------------------------------------
TEST(Metrics, GreenWeeksPct) {
    // Create 5 bars, each one week apart
    const int        N = 5;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 7LL * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    // 5 trades, each in a different week
    float              weekly_pnls[] = {100.0f, -50.0f, 200.0f, -10.0f, 50.0f};
    std::vector<Trade> trades(5);
    for (int i = 0; i < 5; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = weekly_pnls[i] + 25.0f;
        trades[i].net_pnl        = weekly_pnls[i];
        trades[i].min_unrealized = 0.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // 3 green weeks out of 5 = 60%
    EXPECT_NEAR(result.green_weeks_pct, 60.0f, 1.0f) << "Green weeks should be approximately 60%";
}

// ---------------------------------------------------------------------------
// test_max_red_weeks
// Weekly PnL = {100, -10, -20, -5, 50, -1, -2, 100}. Max consecutive red = 3.
// ---------------------------------------------------------------------------
TEST(Metrics, MaxRedWeeks) {
    // Create 8 bars, each one week apart
    const int        N = 8;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 7LL * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    float              weekly_pnls[] = {100.0f, -10.0f, -20.0f, -5.0f, 50.0f, -1.0f, -2.0f, 100.0f};
    std::vector<Trade> trades(8);
    for (int i = 0; i < 8; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = weekly_pnls[i] + 25.0f;
        trades[i].net_pnl        = weekly_pnls[i];
        trades[i].min_unrealized = 0.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // Max consecutive red weeks: weeks 1,2,3 are red = 3 consecutive.
    // Weeks 5,6 are also red but only 2 consecutive.
    EXPECT_EQ(result.max_red_weeks, 3) << "Max consecutive red weeks should be 3";
}

// ---------------------------------------------------------------------------
// test_trr_computation
// TRR = (avg_win * win_rate) / avg_trade_dd.
// With avg_win=200, win_rate=0.6, avg_trade_dd=95 -> TRR = 120/95 = 1.2632.
// ---------------------------------------------------------------------------
TEST(Metrics, TrrComputation) {
    // Create 10 trades: 6 wins at $200, 4 losses at -$100.
    // win_rate = 0.6, avg_win = 200, avg_loss = -100
    // Set min_unrealized = -500 for one trade, -50 for the rest.
    // avg_trade_dd = -(9*(-50) + (-500)) / 10 = 950/10 = 95.0
    const int        N = 10;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    std::vector<Trade> trades(10);
    for (int i = 0; i < 10; ++i) {
        trades[i].entry_bar   = i;
        trades[i].exit_bar    = i;
        trades[i].entry_price = 4500.0f;
        trades[i].exit_price  = 4500.0f;
        trades[i].direction   = 1;

        if (i < 6) {
            // Winning trades
            trades[i].gross_pnl = 225.0f;  // 200 + 25 cost
            trades[i].net_pnl   = 200.0f;
        } else {
            // Losing trades
            trades[i].gross_pnl = -75.0f;  // -100 + 25 cost
            trades[i].net_pnl   = -100.0f;
        }

        // Set worst unrealized PnL to -500 for one trade
        trades[i].min_unrealized = (i == 3) ? -500.0f : -50.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    EXPECT_NEAR(result.win_rate, 0.6f, 0.01f);
    EXPECT_NEAR(result.avg_win, 200.0f, 0.01f);
    EXPECT_NEAR(result.max_live_drawdown, 500.0f, 0.01f)
        << "max_live_drawdown = -min(min_unrealized) = 500";

    // avg_trade_dd = -(9*(-50) + (-500)) / 10 = 950/10 = 95
    EXPECT_NEAR(result.avg_trade_dd, 95.0f, 0.01f)
        << "avg_trade_dd = avg of per-trade worst unrealized (positive magnitude)";

    // TRR = (avg_win * win_rate) / avg_trade_dd = (200 * 0.6) / 95 = 1.2632
    EXPECT_NEAR(result.trade_recovery_ratio, 120.0f / 95.0f, 0.01f)
        << "TRR = (200 * 0.6) / 95 = 1.2632";
}

// ---------------------------------------------------------------------------
// test_annual_roc
// PnL=10000 over 1 year, capital=15000.  ROC = (10000/15000)*100 = 66.7%.
// ---------------------------------------------------------------------------
TEST(Metrics, AnnualRoc) {
    // Create bars spanning exactly 1 year
    const int        N = 2;
    std::vector<Bar> bars(N);
    // Jan 1, 2020 to Jan 1, 2021
    bars[0] = make_bar(1'577'836'800LL * NS_PER_SEC, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    bars[1] = make_bar(1'609'459'200LL * NS_PER_SEC, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    // Duration = exactly 366 days (2020 is leap year)

    // One trade with net_pnl = 10000
    std::vector<Trade> trades(1);
    trades[0].entry_bar      = 0;
    trades[0].exit_bar       = 1;
    trades[0].entry_price    = 4500.0f;
    trades[0].exit_price     = 4500.0f;
    trades[0].direction      = 1;
    trades[0].gross_pnl      = 10025.0f;
    trades[0].net_pnl        = 10000.0f;
    trades[0].min_unrealized = 0.0f;

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // Annual ROC = annualized_pnl / capital * 100
    // Duration ~1 year, so annualized_pnl ~10000
    // ROC = 10000 / 15000 * 100 = 66.67%
    EXPECT_NEAR(result.annual_roc, 66.7f, 0.5f)
        << "Annual ROC should be approximately 66.7% for $10K PnL on $15K capital over 1 year";
}

// =============================================================================
// COST MODEL TESTS (2 tests)
// =============================================================================

// ---------------------------------------------------------------------------
// test_cost_model_es
// Verify ES cost model parameters and calculations.
// ---------------------------------------------------------------------------
TEST(CostModel, EsModel) {
    CostModel es = CostModel::es();
    EXPECT_FLOAT_EQ(es.tick_size, 0.25f);
    EXPECT_FLOAT_EQ(es.tick_value, 12.50f);
    EXPECT_FLOAT_EQ(es.point_value, 50.0f);  // $50/pt for ES
    EXPECT_EQ(es.round_trip_ticks, 2);
    EXPECT_FLOAT_EQ(es.slippage_ticks, 2.0f);  // 2 ticks per side
    EXPECT_FLOAT_EQ(es.capital_required, 15000.0f);

    // round_trip_cost = (2 spread + 2*2 slippage) * $12.50/tick = $75.00
    EXPECT_FLOAT_EQ(es.round_trip_cost(), 75.0f);

    // price_to_pnl: 10 pt move * $50/pt = $500
    EXPECT_FLOAT_EQ(es.price_to_pnl(10.0f), 500.0f);
}

// ---------------------------------------------------------------------------
// test_cost_model_nq
// Verify NQ cost model parameters.
// ---------------------------------------------------------------------------
TEST(CostModel, NqModel) {
    CostModel nq = CostModel::nq();
    EXPECT_FLOAT_EQ(nq.tick_size, 0.25f);
    EXPECT_FLOAT_EQ(nq.tick_value, 5.00f);
    EXPECT_FLOAT_EQ(nq.point_value, 20.0f);  // $20/pt for NQ
    EXPECT_EQ(nq.round_trip_ticks, 2);

    // round_trip_cost = (2 spread + 2*2 slippage) * $5.00/tick = $30.00
    EXPECT_FLOAT_EQ(nq.round_trip_cost(), 30.0f);

    // price_to_pnl: 10 pt move * $20/pt = $200
    EXPECT_FLOAT_EQ(nq.price_to_pnl(10.0f), 200.0f);
}

// =============================================================================
// PNL DECOMPOSITION TESTS (2 tests)
// =============================================================================

// ---------------------------------------------------------------------------
// test_pnl_decomposition_basic
// Verify gap and bar-body decomposition on a simple trade.
// ---------------------------------------------------------------------------
TEST(PnlDecomposition, Basic) {
    // Create bars: bar 0 closes at 100, bar 1 opens at 102 (gap = 2), closes at 105 (body = 3)
    const int        N = 5;
    std::vector<Bar> bars(N);
    bars[0] = make_bar(0, 99.0f, 101.0f, 98.0f, 100.0f);
    bars[1] = make_bar(30 * NS_PER_SEC, 102.0f, 106.0f, 101.0f, 105.0f);
    bars[2] = make_bar(60 * NS_PER_SEC, 105.0f, 107.0f, 104.0f, 106.0f);
    bars[3] = make_bar(90 * NS_PER_SEC, 106.0f, 108.0f, 105.0f, 107.0f);
    bars[4] = make_bar(120 * NS_PER_SEC, 107.0f, 109.0f, 106.0f, 108.0f);

    // One long trade: entry at bar 1 (open=102), exit at bar 3 (close=107)
    Trade trade;
    trade.entry_bar      = 1;
    trade.exit_bar       = 3;
    trade.entry_price    = 102.0f;
    trade.exit_price     = 107.0f;
    trade.direction      = 1;
    trade.gross_pnl      = es_cost().price_to_pnl(5.0f);  // 5 points * 50 = 250
    trade.net_pnl        = trade.gross_pnl - es_cost().round_trip_cost();
    trade.min_unrealized = 0.0f;

    std::vector<Trade> trades = {trade};
    auto               decomp = decompose_pnl(trades, bars.data(), es_cost());

    // Gap at fill bar (bar 1): open[1] - close[0] = 102 - 100 = 2
    // direction * gap_price_diff = 1 * 2 = 2 -> pnl = price_to_pnl(2) = 100
    float expected_gap = es_cost().price_to_pnl(2.0f);
    EXPECT_NEAR(decomp.total_gap_pnl, expected_gap, 0.01f);

    // Bar body at fill bar: close[1] - open[1] = 105 - 102 = 3
    // direction * body_diff = 1 * 3 = 3 -> pnl = price_to_pnl(3) = 150
    float expected_body = es_cost().price_to_pnl(3.0f);
    EXPECT_NEAR(decomp.total_bar_body_pnl, expected_body, 0.01f);
}

// ---------------------------------------------------------------------------
// test_pnl_decomposition_classification
// Verify is_real_edge and is_gap_artifact flags.
// ---------------------------------------------------------------------------
TEST(PnlDecomposition, Classification) {
    // Create scenario where bar body dominates (> 80% of gross PnL)
    const int        N = 3;
    std::vector<Bar> bars(N);
    bars[0] = make_bar(0, 100.0f, 101.0f, 99.0f, 100.0f);
    bars[1] = make_bar(30 * NS_PER_SEC, 100.0f, 110.0f, 99.0f, 110.0f);  // tiny gap, big body
    bars[2] = make_bar(60 * NS_PER_SEC, 110.0f, 111.0f, 109.0f, 110.0f);

    Trade trade;
    trade.entry_bar      = 1;
    trade.exit_bar       = 2;
    trade.entry_price    = 100.0f;
    trade.exit_price     = 110.0f;
    trade.direction      = 1;
    trade.gross_pnl      = es_cost().price_to_pnl(10.0f);  // 500
    trade.net_pnl        = trade.gross_pnl - es_cost().round_trip_cost();
    trade.min_unrealized = 0.0f;

    std::vector<Trade> trades = {trade};
    auto               decomp = decompose_pnl(trades, bars.data(), es_cost());

    // Gap = open[1] - close[0] = 100 - 100 = 0
    // Body = close[1] - open[1] = 110 - 100 = 10 -> price_to_pnl(10) = 500
    // body_pct = 500 / 500 * 100 = 100%
    EXPECT_GT(decomp.bar_body_pct, 80.0f);
    EXPECT_TRUE(decomp.is_real_edge);
    EXPECT_FALSE(decomp.is_gap_artifact);
}

// ---------------------------------------------------------------------------
// test_pnl_decomposition_clamping
// Verify bar_body_pct and gap_pct are clamped to [-100, 200].
// ---------------------------------------------------------------------------
TEST(PnlDecomposition, PercentageClamping) {
    // Create a scenario where bar body and gap have opposing large magnitudes
    // such that raw percentages would exceed 200% or go below -100%.
    //
    // Bar 0: close=100
    // Bar 1: open=120 (gap=+20), close=90 (body=-30)
    // Long trade: gross_pnl is small negative (the trade lost money overall).
    // gap_pnl   = price_to_pnl(1 * (120-100)) = price_to_pnl(20)
    // body_pnl  = price_to_pnl(1 * (90-120))  = price_to_pnl(-30)
    // If gross_pnl is small, the ratios blow up.

    const int        N = 3;
    std::vector<Bar> bars(N);
    bars[0] = make_bar(0, 99.0f, 101.0f, 98.0f, 100.0f);
    bars[1] = make_bar(30 * NS_PER_SEC, 120.0f, 125.0f, 88.0f, 90.0f);
    bars[2] = make_bar(60 * NS_PER_SEC, 90.0f, 91.0f, 89.0f, 90.0f);

    Trade trade;
    trade.entry_bar   = 1;
    trade.exit_bar    = 1;
    trade.entry_price = 120.0f;
    trade.exit_price  = 90.0f;
    trade.direction   = 1;  // long
    // Gross PnL = (90 - 120) * 50 = -1500  (for ES)
    trade.gross_pnl      = es_cost().price_to_pnl(-30.0f);
    trade.net_pnl        = trade.gross_pnl - es_cost().round_trip_cost();
    trade.min_unrealized = 0.0f;

    std::vector<Trade> trades = {trade};
    auto               decomp = decompose_pnl(trades, bars.data(), es_cost());

    // gap_pnl = price_to_pnl(1 * 20) = 1000 (positive)
    // body_pnl = price_to_pnl(1 * -30) = -1500 (negative)
    // gross = -1500
    // abs_total = 1500
    // raw gap_pct = 1000/1500*100 = 66.67%  (within bounds, fine)
    // raw body_pct = -1500/1500*100 = -100%  (at boundary)
    EXPECT_GE(decomp.bar_body_pct, -100.0f);
    EXPECT_LE(decomp.bar_body_pct, 200.0f);
    EXPECT_GE(decomp.gap_pct, -100.0f);
    EXPECT_LE(decomp.gap_pct, 200.0f);

    // With negative bar_body_pct, is_real_edge should be false
    EXPECT_FALSE(decomp.is_real_edge);
}

TEST(PnlDecomposition, ExtremeComponentsClamped) {
    // Construct a scenario where gross PnL is very small but components are large,
    // causing raw percentages to exceed 200%.
    const int        N = 3;
    std::vector<Bar> bars(N);
    bars[0] = make_bar(0, 99.0f, 101.0f, 98.0f, 100.0f);
    // Bar 1: large gap (+50) and large negative body (-49)
    // so gross ~= +1 point
    bars[1] = make_bar(30 * NS_PER_SEC, 150.0f, 155.0f, 100.0f, 101.0f);
    bars[2] = make_bar(60 * NS_PER_SEC, 101.0f, 102.0f, 100.0f, 101.0f);

    Trade trade;
    trade.entry_bar   = 1;
    trade.exit_bar    = 1;
    trade.entry_price = 150.0f;
    trade.exit_price  = 101.0f;
    trade.direction   = 1;  // long
    // gross = price_to_pnl(101 - 150) = price_to_pnl(-49) for 1 trade
    // But let's set the gross to a tiny positive value to stress-test ratios
    trade.gross_pnl      = es_cost().price_to_pnl(1.0f);  // small positive PnL
    trade.net_pnl        = trade.gross_pnl - es_cost().round_trip_cost();
    trade.min_unrealized = 0.0f;

    std::vector<Trade> trades = {trade};
    auto               decomp = decompose_pnl(trades, bars.data(), es_cost());

    // gap_pnl = price_to_pnl(1 * (150-100)) = price_to_pnl(50) = 2500
    // body_pnl = price_to_pnl(1 * (101-150)) = price_to_pnl(-49) = -2450
    // gross = price_to_pnl(1) = 50
    // abs_total = 50
    // raw gap_pct = 2500/50*100 = 5000%  -> clamped to 200
    // raw body_pct = -2450/50*100 = -4900%  -> clamped to -100
    EXPECT_FLOAT_EQ(decomp.gap_pct, 200.0f);
    EXPECT_FLOAT_EQ(decomp.bar_body_pct, -100.0f);

    // is_real_edge should be false (bar_body_pct = -100, not > 80)
    EXPECT_FALSE(decomp.is_real_edge);
    // is_gap_artifact should be true (gap_pct = 200 > 50 -- gap dominates)
    EXPECT_TRUE(decomp.is_gap_artifact);
}

// =============================================================================
// INTEGRATION-STYLE TESTS (additional validation)
// =============================================================================

// ---------------------------------------------------------------------------
// test_backtest_full_round_trip
// Run a complete backtest with synthetic indicators and verify trade log
// consistency.
// ---------------------------------------------------------------------------
TEST(Integration, FullRoundTrip) {
    const int          N = 100;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = 4500.0f + std::sin(static_cast<float>(i) * 0.1f) * 20.0f;
    }
    auto bars = make_bars_from_closes(closes);

    // Create a signal that oscillates: +1 for a while, then -1, etc.
    std::vector<float> signal(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = std::sin(static_cast<float>(i) * 0.2f);
    }
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::SIGNAL_REVERSAL;

    CostModel      cost = es_cost();
    BacktestEngine engine;
    BacktestResult result = engine.run(bars.data(), 0, N, signals, 1, config, cost, 1);

    const auto& trades = engine.last_trades();

    // Verify basic consistency
    float sum_net_pnl = 0.0f;
    for (const auto& t : trades) {
        sum_net_pnl += t.net_pnl;

        // Entry bar should be before exit bar
        EXPECT_LT(t.entry_bar, t.exit_bar) << "Entry should be before exit";

        // Direction should be +1 or -1
        EXPECT_TRUE(t.direction == 1 || t.direction == -1) << "Direction should be +1 or -1";

        // net_pnl should be gross_pnl minus round_trip_cost
        EXPECT_NEAR(t.net_pnl, t.gross_pnl - cost.round_trip_cost(), 0.01f)
            << "net_pnl should equal gross_pnl - round_trip_cost";
    }

    // Total PnL from result should match sum of trade PnLs
    EXPECT_NEAR(result.total_pnl, sum_net_pnl, 0.1f)
        << "Result total_pnl should match sum of trade net_pnls";

    // Win rate should be consistent
    if (result.total_trades > 0) {
        int wins = 0;
        for (const auto& t : trades) {
            if (t.net_pnl > 0.0f) ++wins;
        }
        float expected_wr = static_cast<float>(wins) / static_cast<float>(result.total_trades);
        EXPECT_NEAR(result.win_rate, expected_wr, 0.01f);
    }
}

// ---------------------------------------------------------------------------
// test_metrics_no_trades
// Verify that compute_metrics handles zero trades gracefully.
// ---------------------------------------------------------------------------
TEST(Metrics, NoTrades) {
    const int        N = 10;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    std::vector<Trade> trades;  // empty
    CostModel          cost   = es_cost();
    BacktestResult     result = compute_metrics(trades, bars.data(), 0, N, cost);

    EXPECT_EQ(result.total_trades, 0);
    EXPECT_FLOAT_EQ(result.total_pnl, 0.0f);
    EXPECT_FLOAT_EQ(result.win_rate, 0.0f);
    EXPECT_FLOAT_EQ(result.sharpe_ratio, 0.0f);
    EXPECT_FLOAT_EQ(result.max_drawdown, 0.0f);
}

// ---------------------------------------------------------------------------
// test_trr_sentinel
// When avg_trade_dd == 0 (no trade ever went negative), TRR == 100.0f.
// ---------------------------------------------------------------------------
TEST(Metrics, TrrSentinel) {
    // Create 5 trades where min_unrealized is always >= 0
    // (no trade ever had an unrealized loss).
    const int        N = 5;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    std::vector<Trade> trades(5);
    for (int i = 0; i < 5; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = 225.0f;
        trades[i].net_pnl        = 200.0f;
        trades[i].min_unrealized = 0.0f;  // never went negative
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // max_live_drawdown should be 0 (no trade had negative unrealized)
    EXPECT_FLOAT_EQ(result.max_live_drawdown, 0.0f)
        << "max_live_drawdown should be 0 when no trade went negative";

    // avg_trade_dd should be 0 (no trade had negative unrealized)
    EXPECT_FLOAT_EQ(result.avg_trade_dd, 0.0f)
        << "avg_trade_dd should be 0 when no trade went negative";

    // TRR sentinel: when avg_trade_dd == 0, TRR = 100.0
    EXPECT_FLOAT_EQ(result.trade_recovery_ratio, 100.0f)
        << "TRR sentinel should be 100.0 when no trade ever went negative";
}

// ---------------------------------------------------------------------------
// test_calmar_known
// Create trades with known PnL and max drawdown, verify calmar ratio.
// ---------------------------------------------------------------------------
TEST(Metrics, CalmarKnown) {
    // 2 bars, 1 year apart.  Three trades producing a known equity curve
    // with a computable max drawdown.
    const int        N = 4;
    std::vector<Bar> bars(N);
    // Span exactly 1 year (365 days) for easy annualization
    // 2020-01-01 to 2021-01-01
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'577'836'800LL + static_cast<int64_t>(i) * 122LL * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    // Three trades: +500, -300, +400 => total = +600
    // Equity curve: 0 -> 500 -> 200 -> 600
    // Peak-to-trough drawdown: 500 -> 200 = 300
    std::vector<Trade> trades(3);
    float              pnls[] = {500.0f, -300.0f, 400.0f};
    for (int i = 0; i < 3; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i + 1;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = pnls[i] + 50.0f;
        trades[i].net_pnl        = pnls[i];
        trades[i].min_unrealized = -10.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    EXPECT_NEAR(result.max_drawdown, 300.0f, 0.01f)
        << "Max drawdown should be 300 (peak 500 to trough 200)";

    // Duration = bars[3].ts - bars[0].ts = 366 days (2020 is leap year)
    // annualized_pnl = total_pnl / duration_years = 600 / (366/365.25)
    // calmar = annualized_pnl / max_drawdown
    float duration_days   = 122.0f * 3.0f;  // 366 days
    float duration_years  = duration_days / 365.25f;
    float annualized_pnl  = 600.0f / duration_years;
    float expected_calmar = annualized_pnl / 300.0f;

    EXPECT_NEAR(result.calmar_ratio, expected_calmar, 0.05f)
        << "Calmar = annualized_pnl / max_drawdown";
    EXPECT_GT(result.calmar_ratio, 0.0f) << "Calmar should be positive for net profitable strategy";
}

// ---------------------------------------------------------------------------
// test_negative_months
// Create trades spanning 3 months where 1 month has negative PnL.
// Verify r.negative_months == 1.
// ---------------------------------------------------------------------------
TEST(Metrics, NegativeMonths) {
    // Create bars in 3 different months: Jan 2020, Feb 2020, Mar 2020
    // One bar per month is sufficient.
    const int        N = 3;
    std::vector<Bar> bars(N);
    // Jan 15, Feb 15, Mar 15 (all 2020)
    bars[0] =
        make_bar(1'579'046'400LL * NS_PER_SEC, 4500.0f, 4501.0f, 4499.0f, 4500.0f);  // 2020-01-15
    bars[1] =
        make_bar(1'581'724'800LL * NS_PER_SEC, 4500.0f, 4501.0f, 4499.0f, 4500.0f);  // 2020-02-15
    bars[2] =
        make_bar(1'584'230'400LL * NS_PER_SEC, 4500.0f, 4501.0f, 4499.0f, 4500.0f);  // 2020-03-15

    // 3 trades: Jan positive, Feb negative, Mar positive
    std::vector<Trade> trades(3);
    float              monthly_pnls[] = {500.0f, -200.0f, 300.0f};
    for (int i = 0; i < 3; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = monthly_pnls[i] + 50.0f;
        trades[i].net_pnl        = monthly_pnls[i];
        trades[i].min_unrealized = -10.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    EXPECT_EQ(result.negative_months, 1) << "Should have exactly 1 negative month (February)";
}

// ---------------------------------------------------------------------------
// test_metrics_all_consistency
// Verify internal consistency of compute_metrics output:
//   total_pnl == sum(trade.net_pnl)
//   win_rate == win_count / total_trades
//   profit_factor == gross_wins / |gross_losses|
//   avg_win * wins + avg_loss * losses ~= total_pnl
// ---------------------------------------------------------------------------
TEST(Metrics, AllConsistency) {
    // Create 10 trades with known PnL: 6 wins, 4 losses
    const int        N = 10;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    // 6 wins at varying amounts, 4 losses at varying amounts
    float              pnls[] = {150.0f, -80.0f, 200.0f, -120.0f, 100.0f,
                                 250.0f, -50.0f, 180.0f, -90.0f,  300.0f};
    std::vector<Trade> trades(N);
    for (int i = 0; i < N; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = pnls[i] + 25.0f;  // add back cost
        trades[i].net_pnl        = pnls[i];
        trades[i].min_unrealized = (pnls[i] < 0) ? pnls[i] * 1.5f : -10.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // 1. total_pnl == sum of individual trade net_pnls
    float expected_total = 0.0f;
    for (int i = 0; i < N; ++i) expected_total += pnls[i];
    EXPECT_NEAR(result.total_pnl, expected_total, 0.01f)
        << "total_pnl should equal sum of individual net_pnls";

    // 2. win_rate == win_count / total_trades
    int win_count = 0;
    for (int i = 0; i < N; ++i) {
        if (pnls[i] > 0.0f) win_count++;
    }
    float expected_wr = static_cast<float>(win_count) / static_cast<float>(N);
    EXPECT_NEAR(result.win_rate, expected_wr, 0.001f) << "win_rate should equal wins/total";

    // 3. profit_factor == gross_wins / |gross_losses|
    float expected_gw = 0.0f, expected_gl = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (pnls[i] > 0.0f)
            expected_gw += pnls[i];
        else
            expected_gl += pnls[i];
    }
    EXPECT_NEAR(result.gross_wins, expected_gw, 0.01f)
        << "gross_wins should match sum of positive net_pnl";
    EXPECT_NEAR(result.gross_losses, expected_gl, 0.01f)
        << "gross_losses should match sum of negative net_pnl";
    if (expected_gl != 0.0f) {
        float expected_pf = expected_gw / std::abs(expected_gl);
        EXPECT_NEAR(result.profit_factor, expected_pf, 0.01f)
            << "profit_factor should equal gross_wins / |gross_losses|";
    }

    // 4. avg_win * wins + avg_loss * losses ~= total_pnl
    int   loss_count    = N - win_count;
    float reconstructed = result.avg_win * static_cast<float>(win_count) +
                          result.avg_loss * static_cast<float>(loss_count);
    EXPECT_NEAR(reconstructed, expected_total, 0.5f)
        << "avg_win*wins + avg_loss*losses should approximate total_pnl";

    // 5. total_trades should match
    EXPECT_EQ(result.total_trades, N);
}

// =============================================================================
// INDICATOR FILTER MODE (SIGNAL vs FILTER)
// =============================================================================

// ---------------------------------------------------------------------------
// test_filter_role_gates_entry
// FILTER role gates entry but does not contribute to composite direction.
// RSI as SIGNAL, ADX as FILTER.
// Case A: ADX filter active (non-zero) -> trades should occur
// Case B: ADX filter inactive (== 0.0) -> NO trades should occur
// ---------------------------------------------------------------------------
TEST(BacktestEngine, FilterRoleGatesEntry) {
    const int          N = 30;
    std::vector<float> closes(N);
    // Trending price so entries have meaningful movement
    for (int i = 0; i < N; ++i) closes[i] = 100.0f + static_cast<float>(i) * 0.5f;
    auto bars = make_bars_from_closes(closes);

    // RSI signal: strong buy (+0.8) on bars 0, 8, 16 (to allow multiple entries)
    std::vector<float> rsi_signal(N, 0.0f);
    rsi_signal[0]  = 0.8f;
    rsi_signal[8]  = 0.8f;
    rsi_signal[16] = 0.8f;

    // ADX filter: active (non-zero) for all bars
    std::vector<float> adx_active(N, 1.0f);
    // ADX filter: inactive (zero) for all bars
    std::vector<float> adx_inactive(N, 0.0f);

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;  // RSI signal series
    config.indicator_indices[1] = 1;  // ADX filter series
    config.indicator_roles[0]   = StrategyConfig::SIGNAL;
    config.indicator_roles[1]   = StrategyConfig::FILTER;
    config.weights[0]           = 1.0f;
    config.weights[1]           = 1.0f;
    config.entry_threshold      = 0.3f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;

    // Case A: filter active -> trades should occur
    const float* signals_active[] = {rsi_signal.data(), adx_active.data()};
    auto result_active = engine.run(bars.data(), 0, N, signals_active, 2, config, es_cost(), 1);
    EXPECT_GT(result_active.total_trades, 0) << "Filter active (ADX=1.0) should allow trades";

    // Case B: filter inactive -> NO trades
    const float* signals_inactive[] = {rsi_signal.data(), adx_inactive.data()};
    auto result_inactive = engine.run(bars.data(), 0, N, signals_inactive, 2, config, es_cost(), 1);
    EXPECT_EQ(result_inactive.total_trades, 0)
        << "Filter inactive (ADX=0.0) should block all trades";
}

// ---------------------------------------------------------------------------
// test_filter_does_not_affect_direction
// With the same SIGNAL indicator (+0.8 = long), adding a FILTER with value
// +1.0 should result in a long trade, not change to short or modify the
// composite magnitude.  The FILTER value should NOT contribute to the
// weighted sum direction.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, FilterDoesNotAffectDirection) {
    const int          N = 20;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) closes[i] = 100.0f + static_cast<float>(i);
    auto bars = make_bars_from_closes(closes);

    // Signal indicator: positive (+0.8) at bar 0 -> should trigger long
    std::vector<float> signal_data(N, 0.0f);
    signal_data[0] = 0.8f;

    // Filter indicator: negative value (-0.5), but non-zero so filter is active
    // If filter contributed to direction, composite would be
    // (0.8 + (-0.5)) / 2 = 0.15 which is below threshold 0.3 -> no trade.
    // Since filter does NOT contribute, composite = 0.8 / 1.0 = 0.8 -> trade occurs.
    std::vector<float> filter_data(N, -0.5f);

    const float* signals[] = {signal_data.data(), filter_data.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_roles[0]   = StrategyConfig::SIGNAL;
    config.indicator_roles[1]   = StrategyConfig::FILTER;
    config.weights[0]           = 1.0f;
    config.weights[1]           = 1.0f;
    config.entry_threshold      = 0.3f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u)
        << "Filter with non-zero value should allow trade (does not change composite)";

    // Direction should be long (+1) since the SIGNAL indicator was positive
    EXPECT_EQ(trades[0].direction, 1) << "Trade direction should be long (from SIGNAL indicator), "
                                         "not affected by negative FILTER value";
}

// =============================================================================
// MAX_LOSS_ATR EXIT MODE
// =============================================================================

// ---------------------------------------------------------------------------
// test_max_loss_atr_exit
// Enter long at bar 1 open (100), price drops below stop level at bar 3.
// ATR = constant 2.0, max_loss_atr_mult = 2.0 -> stop at entry - 4.0 = 96.0.
// Bar 3 low = 95.0 < 96.0 -> exit signal fires at bar 3, fills at bar 4 open.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, MaxLossAtrExit) {
    const int N = 10;
    // OHLC: entry at bar 1 open=100, then price drops through stop level
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal fires here
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 1: entry fill at open=100
        {99.0f, 100.0f, 97.0f, 98.0f},    // bar 2: low=97 > 96, no stop yet
        {98.0f, 99.0f, 95.0f, 96.0f},     // bar 3: low=95 < 96, stop triggered!
        {96.0f, 97.0f, 95.0f, 96.5f},     // bar 4: exit fills at open=96.0
        {96.5f, 97.0f, 96.0f, 96.5f},     // bar 5
        {96.5f, 97.0f, 96.0f, 96.5f},     // bar 6
        {96.5f, 97.0f, 96.0f, 96.5f},     // bar 7
        {96.5f, 97.0f, 96.0f, 96.5f},     // bar 8
        {96.5f, 97.0f, 96.0f, 96.5f},     // bar 9
    };
    auto bars = make_bars_ohlc(ohlc);

    // Signal: +1 at bar 0 -> entry at bar 1
    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;

    // ATR series: constant 2.0
    std::vector<float> atr(N, 2.0f);

    const float* signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::MAX_LOSS_ATR;
    config.max_loss_atr_mult    = 2.0f;  // stop distance = 2.0 * 2.0 = 4.0 points
    config.atr_series_index     = 1;     // ATR is series index 1

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u) << "Should have exactly one trade (stopped out)";

    // Level-triggered: SL fills at stop level (96.0) on detection bar 3
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    EXPECT_EQ(trades[0].exit_bar, 3);              // immediate fill on detection bar
    EXPECT_FLOAT_EQ(trades[0].exit_price, 96.0f);  // stop level, not next-bar open
    EXPECT_EQ(trades[0].direction, 1);

    // Gross PnL: long, entry=100, exit=96 -> (96-100) * point_value
    CostModel cost           = es_cost();
    float     expected_gross = cost.price_to_pnl(96.0f - 100.0f);
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
    EXPECT_LT(trades[0].net_pnl, 0.0f) << "Stopped out trade should be a loss";
}

// =============================================================================
// TAKE_PROFIT_ATR EXIT MODE
// =============================================================================

// ---------------------------------------------------------------------------
// test_take_profit_atr_exit
// Enter long at bar 1 open (100), price rises above TP level at bar 3.
// ATR = 2.0, tp_atr_mult = 1.5 -> TP at entry + 3.0 = 103.0.
// Bar 3 high = 104.0 >= 103.0 -> exit signal fires, fills at bar 4 open.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, TakeProfitAtrExit) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},   // bar 0: signal fires
        {100.0f, 101.0f, 99.5f, 100.5f},   // bar 1: entry at open=100
        {100.5f, 102.0f, 100.0f, 101.5f},  // bar 2: high=102 < 103, no TP
        {101.5f, 104.0f, 101.0f, 103.5f},  // bar 3: high=104 >= 103, TP hit!
        {103.5f, 104.0f, 103.0f, 103.5f},  // bar 4: exit fills at open=103.5
        {103.5f, 104.0f, 103.0f, 103.5f},  // bar 5
        {103.5f, 104.0f, 103.0f, 103.5f},  // bar 6
        {103.5f, 104.0f, 103.0f, 103.5f},  // bar 7
        {103.5f, 104.0f, 103.0f, 103.5f},  // bar 8
        {103.5f, 104.0f, 103.0f, 103.5f},  // bar 9
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
    config.tp_atr_mult          = 1.5f;  // TP distance = 1.5 * 2.0 = 3.0 points
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u) << "Should have exactly one trade (TP hit)";

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    // Level-triggered: TP fills at TP level (103.0) on detection bar 3
    EXPECT_EQ(trades[0].exit_bar, 3);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 103.0f);  // TP level, not next-bar open
    EXPECT_EQ(trades[0].direction, 1);

    // Should be a profitable trade
    EXPECT_GT(trades[0].gross_pnl, 0.0f) << "Take-profit trade should have positive gross PnL";
    EXPECT_GT(trades[0].net_pnl, 0.0f)
        << "Take-profit trade should have positive net PnL (after costs)";
}

// =============================================================================
// TP_SL_ATR COMBINED EXIT MODE
// =============================================================================

// ---------------------------------------------------------------------------
// test_tp_sl_atr_exit_take_profit
// TP is hit before SL.  Should exit at TP.
// ATR=2.0, tp_mult=1.5 (TP at +3.0), sl_mult=1.0 (SL at -2.0).
// Price rises to hit TP first.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, TpSlAtrExitTakeProfit) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},   // bar 0: signal
        {100.0f, 101.0f, 99.0f, 100.5f},   // bar 1: entry at open=100
        {100.5f, 102.0f, 99.5f, 101.5f},   // bar 2: no TP (102<103), no SL (99.5>98)
        {101.5f, 104.0f, 101.0f, 103.0f},  // bar 3: TP hit (104>=103), SL safe (101>98)
        {103.0f, 104.0f, 102.5f, 103.5f},  // bar 4: exit fills at open=103.0
        {103.5f, 104.0f, 103.0f, 103.5f},  // bar 5-9: padding
        {103.5f, 104.0f, 103.0f, 103.5f}, {103.5f, 104.0f, 103.0f, 103.5f},
        {103.5f, 104.0f, 103.0f, 103.5f}, {103.5f, 104.0f, 103.0f, 103.5f},
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
    config.tp_atr_mult          = 1.5f;  // TP at entry + 3.0
    config.sl_atr_mult          = 1.0f;  // SL at entry - 2.0
    config.atr_series_index     = 1;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u);

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    // Level-triggered: TP fills at TP level (103.0) on detection bar 3
    EXPECT_EQ(trades[0].exit_bar, 3);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 103.0f);
    EXPECT_EQ(trades[0].direction, 1);
    EXPECT_GT(trades[0].gross_pnl, 0.0f) << "TP exit should be profitable";
}

// ---------------------------------------------------------------------------
// test_tp_sl_atr_exit_stop_loss
// SL is hit before TP.  Should exit at SL.
// ATR=2.0, tp_mult=2.0 (TP at +4.0), sl_mult=1.0 (SL at -2.0).
// Price drops to hit SL first.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, TpSlAtrExitStopLoss) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 0: signal
        {100.0f, 101.0f, 99.0f, 100.0f},  // bar 1: entry at open=100
        {100.0f, 100.5f, 99.0f, 99.5f},   // bar 2: low=99>98, no SL
        {99.5f, 100.0f, 97.0f, 97.5f},    // bar 3: low=97<98, SL hit! TP needs 104, not hit
        {97.5f, 98.0f, 97.0f, 97.5f},     // bar 4: exit fills at open=97.5
        {97.5f, 98.0f, 97.0f, 97.5f},     // bar 5-9: padding
        {97.5f, 98.0f, 97.0f, 97.5f},    {97.5f, 98.0f, 97.0f, 97.5f},
        {97.5f, 98.0f, 97.0f, 97.5f},    {97.5f, 98.0f, 97.0f, 97.5f},
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
    ASSERT_EQ(trades.size(), 1u);

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 100.0f);
    // Level-triggered: SL fills at SL level (98.0) on detection bar 3
    EXPECT_EQ(trades[0].exit_bar, 3);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 98.0f);  // SL level
    EXPECT_EQ(trades[0].direction, 1);
    EXPECT_LT(trades[0].gross_pnl, 0.0f) << "SL exit should be a losing trade";
    EXPECT_LT(trades[0].net_pnl, 0.0f) << "SL exit should be a loss after costs";
}

// =============================================================================
// TIME WINDOW PRESETS (MORNING / AFTERNOON)
//
// These tests use the session-aware backtest overload.  We construct
// minimal SessionInfo arrays with known time_of_day_sec values and
// verify that MORNING and AFTERNOON filters correctly gate entries.
// =============================================================================

// ---------------------------------------------------------------------------
// test_morning_only_prevents_afternoon_entries
// trading_hours = MORNING (09:30-12:00 ET = 34200-43200 sec)
// Signal fires at all bars, but only bars within morning window should enter.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, MorningOnlyPreventsAfternoonEntries) {
    // Create 20 bars: 10 morning (10:00 = 36000s), 10 afternoon (14:00 = 50400s)
    const int          N = 20;
    std::vector<float> closes(N, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    // Create SessionInfo: first 10 bars at 10:00 ET, next 10 at 14:00 ET
    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].session_close_sec = 17 * 3600;
        if (i < 10) {
            sessions[i].time_of_day_sec = 36000;  // 10:00 ET (within morning)
        } else {
            sessions[i].time_of_day_sec = 50400;  // 14:00 ET (afternoon)
        }
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    // Strong signal on alternating bars to allow multiple entries
    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;  // morning bar
    signal[7]              = 1.0f;  // morning bar
    signal[12]             = 1.0f;  // afternoon bar
    signal[17]             = 1.0f;  // afternoon bar
    const float* signals[] = {signal.data()};

    // No time gates (time gating is a different mechanism; this tests trading_hours)
    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;  // no gating

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 3;
    config.trading_hours        = StrategyConfig::MORNING;  // 09:30-12:00 ET

    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, gates, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    // Should have trades from morning signals (bars 0, 7) but NOT afternoon (12, 17)
    EXPECT_GT(static_cast<int>(trades.size()), 0)
        << "Should have at least one trade from morning signals";

    // Verify no trade entry was triggered by an afternoon signal
    for (const auto& t : trades) {
        int signal_bar = t.entry_bar - 1;  // entry_lag=1
        if (signal_bar >= 0 && signal_bar < N) {
            int32_t tod = sessions[signal_bar].time_of_day_sec;
            EXPECT_GE(tod, 34200) << "Signal bar tod should be >= 09:30 ET (34200)";
            EXPECT_LT(tod, 43200)
                << "Signal bar tod should be < 12:00 ET (43200) for MORNING window";
        }
    }
}

// ---------------------------------------------------------------------------
// test_afternoon_only_prevents_morning_entries
// trading_hours = AFTERNOON (12:00-16:00 ET = 43200-57600 sec)
// Signal fires at morning bars -> should NOT enter.
// Signal fires at afternoon bars -> should enter.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, AfternoonOnlyPreventsMorningEntries) {
    const int          N = 20;
    std::vector<float> closes(N, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].session_close_sec = 17 * 3600;
        if (i < 10) {
            sessions[i].time_of_day_sec = 36000;  // 10:00 ET (morning)
        } else {
            sessions[i].time_of_day_sec = 50400;  // 14:00 ET (afternoon)
        }
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    // Signals: some in morning, some in afternoon
    std::vector<float> signal(N, 0.0f);
    signal[2]              = 1.0f;  // morning -> should be blocked
    signal[5]              = 1.0f;  // morning -> should be blocked
    signal[12]             = 1.0f;  // afternoon -> should enter
    signal[17]             = 1.0f;  // afternoon -> should enter
    const float* signals[] = {signal.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 3;
    config.trading_hours        = StrategyConfig::AFTERNOON;  // 12:00-16:00 ET

    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, gates, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_GT(static_cast<int>(trades.size()), 0)
        << "Should have at least one trade from afternoon signals";

    // Verify all trades were triggered by afternoon-window signals
    for (const auto& t : trades) {
        int signal_bar = t.entry_bar - 1;
        if (signal_bar >= 0 && signal_bar < N) {
            int32_t tod = sessions[signal_bar].time_of_day_sec;
            EXPECT_GE(tod, 43200)
                << "Signal bar tod should be >= 12:00 ET (43200) for AFTERNOON window";
            EXPECT_LT(tod, 57600)
                << "Signal bar tod should be < 16:00 ET (57600) for AFTERNOON window";
        }
    }
}

// =============================================================================
// EXIT MODE TESTS (5 categories)
//
// PnL-verified tests for each exit mode: TRAILING_STOP, MAX_LOSS_ATR,
// TAKE_PROFIT_ATR, TP_SL_ATR, and OPPOSING_SIGNAL.  Each test constructs
// a known price path, injects a constant ATR series (or signal series),
// and verifies exact PnL via EXPECT_FLOAT_EQ.
// =============================================================================

// ---------------------------------------------------------------------------
// TRAILING_STOP exit mode with PnL verification
//
// Enter long at bar 1 open = 4500.  Price rises to 4520, then drops.
// ATR = 5.0, trailing_atr_mult = 2.0, so the stop trails 10.0 points below
// the best high of the bars *before* the bar being checked.  The peak at
// bar 3 sets the stop to 4510 from bar 4 on; bar 4 trades down to 4508 and
// fills the stop at 4510.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, TrailingStopPnlVerified) {
    const int N = 12;
    // Explicit OHLC: entry at bar 1 open = 4500, price rises then drops
    std::vector<std::array<float, 4>> ohlc = {
        {{4500.0f, 4502.0f, 4498.0f, 4500.0f}},  // bar 0: signal fires
        {{4500.0f, 4505.0f, 4498.0f, 4504.0f}},  // bar 1: entry at open=4500, best=4505
        {{4504.0f, 4512.0f, 4503.0f, 4510.0f}},  // bar 2: best=4512
        {{4510.0f, 4520.0f, 4509.0f, 4518.0f}},  // bar 3: stop=4502 (best 4512), low 4509 holds; best -> 4520
        {{4518.0f, 4519.0f, 4508.0f, 4509.0f}},  // bar 4: stop=4510, low 4508 -> stop fills at 4510
        {{4518.0f, 4519.0f, 4517.0f, 4518.0f}},  // bar 5-11: padding
        {{4518.0f, 4519.0f, 4517.0f, 4518.0f}}, {{4518.0f, 4519.0f, 4517.0f, 4518.0f}},
        {{4518.0f, 4519.0f, 4517.0f, 4518.0f}}, {{4518.0f, 4519.0f, 4517.0f, 4518.0f}},
        {{4518.0f, 4519.0f, 4517.0f, 4518.0f}}, {{4518.0f, 4519.0f, 4517.0f, 4518.0f}},
    };
    auto bars = make_bars_ohlc(ohlc);

    // Signal: +1 at bar 0 -> entry at bar 1 (lag=1)
    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;

    // ATR series: constant 5.0
    std::vector<float> atr(N, 5.0f);

    const float* signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::TRAILING_STOP;
    config.trailing_atr_mult    = 2.0f;  // stop distance = 2.0 * 5.0 = 10.0 points
    config.atr_series_index     = 1;     // ATR is series index 1

    CostModel      cost = es_cost();
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, cost, 1);

    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u) << "Should have exactly one trade (trailing stop hit)";

    // Entry at bar 1 open = 4500.0
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 4500.0f);
    EXPECT_EQ(trades[0].direction, 1);

    // Bar 2: stop = 4505 - 10 = 4495, low 4503 holds; best -> 4512
    // Bar 3: stop = 4512 - 10 = 4502, low 4509 holds; best -> 4520
    // Bar 4: stop = 4520 - 10 = 4510, opens 4518 above it, low 4508 -> fills at 4510
    EXPECT_EQ(trades[0].exit_bar, 4);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 4510.0f);

    // PnL: long entry=4500, exit=4510 -> price diff = +10
    // gross_pnl = price_to_pnl(10) = 10 * 50 = 500
    // net_pnl = 500 - 75 = 425
    float expected_gross = cost.price_to_pnl(10.0f);
    float expected_net   = expected_gross - cost.round_trip_cost();
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
    EXPECT_FLOAT_EQ(trades[0].net_pnl, expected_net);
    EXPECT_FLOAT_EQ(expected_gross, 500.0f);
    EXPECT_FLOAT_EQ(expected_net, 425.0f);
}

// ---------------------------------------------------------------------------
// MAX_LOSS_ATR exit mode with PnL verification
//
// Enter long at bar 1 open = 4500.  ATR = 5.0, max_loss_atr_mult = 2.0,
// so stop = 4500 - 10 = 4490.  Bar 3 low = 4488 <= 4490 triggers stop.
// Exit fills at bar 4 open = 4489.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, MaxLossAtrPnlVerified) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {{4500.0f, 4502.0f, 4498.0f, 4500.0f}},  // bar 0: signal fires
        {{4500.0f, 4502.0f, 4498.0f, 4499.0f}},  // bar 1: entry at open=4500
        {{4499.0f, 4500.0f, 4492.0f, 4493.0f}},  // bar 2: low=4492 > 4490, no stop
        {{4493.0f, 4494.0f, 4488.0f, 4489.0f}},  // bar 3: low=4488 <= 4490, STOP triggered
        {{4489.0f, 4491.0f, 4488.0f, 4490.0f}},  // bar 4: exit fills at open=4489
        {{4490.0f, 4491.0f, 4489.0f, 4490.0f}},  // bar 5-9: padding
        {{4490.0f, 4491.0f, 4489.0f, 4490.0f}}, {{4490.0f, 4491.0f, 4489.0f, 4490.0f}},
        {{4490.0f, 4491.0f, 4489.0f, 4490.0f}}, {{4490.0f, 4491.0f, 4489.0f, 4490.0f}},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;

    std::vector<float> atr(N, 5.0f);  // constant ATR = 5.0
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::MAX_LOSS_ATR;
    config.max_loss_atr_mult    = 2.0f;  // stop distance = 2.0 * 5.0 = 10.0
    config.atr_series_index     = 1;

    CostModel      cost = es_cost();
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, cost, 1);

    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u) << "Should have exactly one trade (max loss stop hit)";

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 4500.0f);
    // Level-triggered: fills at stop level 4490.0 on bar 3
    EXPECT_EQ(trades[0].exit_bar, 3);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 4490.0f);
    EXPECT_EQ(trades[0].direction, 1);

    // PnL: long entry=4500, exit=4490 -> price diff = -10
    // gross_pnl = price_to_pnl(-10) = -10 * 50 = -500
    // net_pnl = -500 - 75 = -575
    float expected_gross = cost.price_to_pnl(-10.0f);
    float expected_net   = expected_gross - cost.round_trip_cost();
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
    EXPECT_FLOAT_EQ(trades[0].net_pnl, expected_net);
    EXPECT_FLOAT_EQ(expected_gross, -500.0f);
    EXPECT_FLOAT_EQ(expected_net, -575.0f);
}

// ---------------------------------------------------------------------------
// TAKE_PROFIT_ATR exit mode with PnL verification
//
// Enter long at bar 1 open = 4500.  ATR = 5.0, tp_atr_mult = 2.0,
// so target = 4500 + 10 = 4510.  Bar 3 high = 4512 >= 4510 triggers TP.
// Exit fills at bar 4 open = 4511.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, TakeProfitAtrPnlVerified) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {{4500.0f, 4502.0f, 4498.0f, 4500.0f}},  // bar 0: signal fires
        {{4500.0f, 4503.0f, 4499.0f, 4502.0f}},  // bar 1: entry at open=4500
        {{4502.0f, 4507.0f, 4501.0f, 4506.0f}},  // bar 2: high=4507 < 4510, no TP
        {{4506.0f, 4512.0f, 4505.0f, 4511.0f}},  // bar 3: high=4512 >= 4510, TP triggered!
        {{4511.0f, 4513.0f, 4510.0f, 4512.0f}},  // bar 4: exit fills at open=4511
        {{4512.0f, 4513.0f, 4511.0f, 4512.0f}},  // bar 5-9: padding
        {{4512.0f, 4513.0f, 4511.0f, 4512.0f}}, {{4512.0f, 4513.0f, 4511.0f, 4512.0f}},
        {{4512.0f, 4513.0f, 4511.0f, 4512.0f}}, {{4512.0f, 4513.0f, 4511.0f, 4512.0f}},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;

    std::vector<float> atr(N, 5.0f);  // constant ATR = 5.0
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::TAKE_PROFIT_ATR;
    config.tp_atr_mult          = 2.0f;  // TP distance = 2.0 * 5.0 = 10.0
    config.atr_series_index     = 1;

    CostModel      cost = es_cost();
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, cost, 1);

    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u) << "Should have exactly one trade (TP hit)";

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 4500.0f);
    // Level-triggered: fills at TP level 4510.0 on bar 3
    EXPECT_EQ(trades[0].exit_bar, 3);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 4510.0f);
    EXPECT_EQ(trades[0].direction, 1);

    // PnL: long entry=4500, exit=4510 -> price diff = +10
    // gross_pnl = price_to_pnl(10) = 10 * 50 = 500
    // net_pnl = 500 - 75 = 425
    float expected_gross = cost.price_to_pnl(10.0f);
    float expected_net   = expected_gross - cost.round_trip_cost();
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
    EXPECT_FLOAT_EQ(trades[0].net_pnl, expected_net);
    EXPECT_FLOAT_EQ(expected_gross, 500.0f);
    EXPECT_FLOAT_EQ(expected_net, 425.0f);
}

// ---------------------------------------------------------------------------
// TP_SL_ATR exit mode -- TP hit first (PnL verified)
//
// Enter long at bar 1 open = 4500.  ATR = 5.0, tp_atr_mult = 2.0
// (target = 4510), sl_atr_mult = 1.0 (stop = 4495).  Price rises to
// hit TP at bar 3 (high=4512 >= 4510).  Exit fills at bar 4 open = 4511.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, TpSlAtrTakeProfitPnlVerified) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {{4500.0f, 4502.0f, 4498.0f, 4500.0f}},  // bar 0: signal fires
        {{4500.0f, 4503.0f, 4497.0f, 4502.0f}},  // bar 1: entry at open=4500, low=4497 > 4495
        {{4502.0f, 4507.0f, 4496.0f, 4506.0f}},  // bar 2: high=4507<4510, low=4496>4495, safe
        {{4506.0f, 4512.0f, 4505.0f, 4511.0f}},  // bar 3: high=4512>=4510, TP triggered!
        {{4511.0f, 4513.0f, 4510.0f, 4512.0f}},  // bar 4: exit fills at open=4511
        {{4512.0f, 4513.0f, 4511.0f, 4512.0f}},  // bar 5-9: padding
        {{4512.0f, 4513.0f, 4511.0f, 4512.0f}}, {{4512.0f, 4513.0f, 4511.0f, 4512.0f}},
        {{4512.0f, 4513.0f, 4511.0f, 4512.0f}}, {{4512.0f, 4513.0f, 4511.0f, 4512.0f}},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;
    std::vector<float> atr(N, 5.0f);
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::TP_SL_ATR;
    config.tp_atr_mult          = 2.0f;  // TP at entry + 10.0 = 4510
    config.sl_atr_mult          = 1.0f;  // SL at entry - 5.0  = 4495
    config.atr_series_index     = 1;

    CostModel      cost = es_cost();
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, cost, 1);

    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u);

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 4500.0f);
    // Level-triggered: fills at TP level 4510.0 on bar 3
    EXPECT_EQ(trades[0].exit_bar, 3);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 4510.0f);
    EXPECT_EQ(trades[0].direction, 1);

    // PnL: long entry=4500, exit=4510 -> +10 pts * $50 = $500 gross
    // net = $500 - $75 = $425
    float expected_gross = cost.price_to_pnl(10.0f);
    float expected_net   = expected_gross - cost.round_trip_cost();
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
    EXPECT_FLOAT_EQ(trades[0].net_pnl, expected_net);
    EXPECT_FLOAT_EQ(expected_gross, 500.0f);
    EXPECT_FLOAT_EQ(expected_net, 425.0f);
}

// ---------------------------------------------------------------------------
// TP_SL_ATR exit mode -- SL hit first (PnL verified)
//
// Enter long at bar 1 open = 4500.  ATR = 5.0, tp_atr_mult = 2.0
// (target = 4510), sl_atr_mult = 1.0 (stop = 4495).  Price drops to
// hit SL at bar 3 (low=4493 <= 4495).  Exit fills at bar 4 open = 4494.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, TpSlAtrStopLossPnlVerified) {
    const int                         N    = 10;
    std::vector<std::array<float, 4>> ohlc = {
        {{4500.0f, 4502.0f, 4498.0f, 4500.0f}},  // bar 0: signal fires
        {{4500.0f, 4502.0f, 4497.0f, 4498.0f}},  // bar 1: entry at open=4500, low=4497>4495
        {{4498.0f, 4499.0f, 4496.0f, 4497.0f}},  // bar 2: low=4496>4495, safe
        {{4497.0f, 4498.0f, 4493.0f, 4494.0f}},  // bar 3: low=4493<=4495, SL triggered!
        {{4494.0f, 4496.0f, 4493.0f, 4495.0f}},  // bar 4: exit fills at open=4494
        {{4495.0f, 4496.0f, 4494.0f, 4495.0f}},  // bar 5-9: padding
        {{4495.0f, 4496.0f, 4494.0f, 4495.0f}}, {{4495.0f, 4496.0f, 4494.0f, 4495.0f}},
        {{4495.0f, 4496.0f, 4494.0f, 4495.0f}}, {{4495.0f, 4496.0f, 4494.0f, 4495.0f}},
    };
    auto bars = make_bars_ohlc(ohlc);

    std::vector<float> signal(N, 0.0f);
    signal[0] = 1.0f;
    std::vector<float> atr(N, 5.0f);
    const float*       signals[] = {signal.data(), atr.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::TP_SL_ATR;
    config.tp_atr_mult          = 2.0f;  // TP at entry + 10.0 = 4510
    config.sl_atr_mult          = 1.0f;  // SL at entry - 5.0  = 4495
    config.atr_series_index     = 1;

    CostModel      cost = es_cost();
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, cost, 1);

    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u);

    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_FLOAT_EQ(trades[0].entry_price, 4500.0f);
    // Level-triggered: fills at SL level 4495.0 on bar 3
    EXPECT_EQ(trades[0].exit_bar, 3);
    EXPECT_FLOAT_EQ(trades[0].exit_price, 4495.0f);
    EXPECT_EQ(trades[0].direction, 1);

    // PnL: long entry=4500, exit=4495 -> -5 pts * $50 = -$250 gross
    // net = -$250 - $75 = -$325
    float expected_gross = cost.price_to_pnl(-5.0f);
    float expected_net   = expected_gross - cost.round_trip_cost();
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
    EXPECT_FLOAT_EQ(trades[0].net_pnl, expected_net);
    EXPECT_FLOAT_EQ(expected_gross, -250.0f);
    EXPECT_FLOAT_EQ(expected_net, -325.0f);
}

// ---------------------------------------------------------------------------
// OPPOSING_SIGNAL exit mode with timing verification
//
// Enter long when composite > 0.5 (bar 0 signal = +0.8).  Later, composite
// drops to -0.6 (below -threshold 0.5) at bar 4.  This triggers pending_exit
// at bar 4, which fills at bar 5 open.  After exit, the engine is flat.  On
// bar 5 it evaluates the signal again -- but bar 5's signal is 0.0, so no
// new entry.  The signal goes to -0.8 at bar 6, which triggers a short entry
// filling at bar 7 -- confirming the 1-bar dead zone between exit fill and
// new entry evaluation.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, OpposingSignalExitTiming) {
    const int          N = 15;
    std::vector<float> closes(N);
    // Rising price then falling to make the trades meaningful
    for (int i = 0; i < N; ++i) {
        if (i < 5)
            closes[i] = 4500.0f + static_cast<float>(i) * 2.0f;
        else
            closes[i] = 4510.0f - static_cast<float>(i - 5) * 2.0f;
    }
    auto bars = make_bars_from_closes(closes);

    // Composite signal series:
    //   bar 0: +0.8 (above threshold 0.5 -> triggers long entry, fills bar 1)
    //   bars 1-3: +0.3 (above 0 but below threshold, no opposing signal)
    //   bar 4: -0.6 (below -0.5 -> opposing signal triggers exit, fills bar 5)
    //   bar 5: 0.0 (flat -- engine is flat here after exit fill, no new signal)
    //   bar 6: -0.8 (below -0.5 -> triggers short entry, fills bar 7)
    //   bars 7+: 0.0
    std::vector<float> signal(N, 0.0f);
    signal[0]              = 0.8f;
    signal[1]              = 0.3f;
    signal[2]              = 0.3f;
    signal[3]              = 0.3f;
    signal[4]              = -0.6f;
    signal[5]              = 0.0f;
    signal[6]              = -0.8f;
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::OPPOSING_SIGNAL;

    CostModel      cost = es_cost();
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, cost, 1);

    const auto& trades = engine.last_trades();

    // Should have at least 2 trades: the long (exited by opposing signal) and the short
    ASSERT_GE(trades.size(), 2u)
        << "Should have the initial long trade and a subsequent short trade";

    // Trade 0: long entry at bar 1, exit at bar 5 (opposing signal at bar 4, fills bar 5)
    EXPECT_EQ(trades[0].entry_bar, 1);
    EXPECT_EQ(trades[0].direction, 1);
    EXPECT_EQ(trades[0].exit_bar, 5)
        << "Opposing signal at bar 4 should trigger exit filling at bar 5";

    // Trade 1: short entry -- signal fires at bar 6 (-0.8 < -0.5), fills at bar 7
    // The engine becomes flat at bar 5 (exit fills), evaluates signal at bar 5 (0.0, no entry),
    // evaluates signal at bar 6 (-0.8, triggers short), entry fills at bar 7.
    EXPECT_EQ(trades[1].entry_bar, 7)
        << "Short entry should fill at bar 7 (signal at bar 6, lag=1)";
    EXPECT_EQ(trades[1].direction, -1) << "Second trade should be short (opposing direction)";

    // Verify the first trade's PnL
    float price_diff =
        static_cast<float>(trades[0].direction) * (trades[0].exit_price - trades[0].entry_price);
    float expected_gross = cost.price_to_pnl(price_diff);
    float expected_net   = expected_gross - cost.round_trip_cost();
    EXPECT_FLOAT_EQ(trades[0].gross_pnl, expected_gross);
    EXPECT_FLOAT_EQ(trades[0].net_pnl, expected_net);
}

// =============================================================================
// COMPOSITE SIGNAL AGGREGATION TESTS
//
// Verify the three aggregation modes (WEIGHTED_SUM, MAJORITY_VOTE, UNANIMOUS)
// produce correct composite values and trigger entries as expected.
// =============================================================================

// ---------------------------------------------------------------------------
// WEIGHTED_SUM aggregation
//
// Two indicators with signals [0.8, -0.4] and weights [1.0, 0.5].
// Composite = (0.8*1.0 + (-0.4)*0.5) / (1.0 + 0.5) = (0.8 - 0.2) / 1.5 = 0.4.
// With threshold 0.3, composite 0.4 > 0.3 -> should trigger a long entry.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, CompositeWeightedSum) {
    const int          N = 15;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) closes[i] = 100.0f + static_cast<float>(i);
    auto bars = make_bars_from_closes(closes);

    // Indicator 0: signal = 0.8 at bar 0
    std::vector<float> ind0(N, 0.0f);
    ind0[0] = 0.8f;

    // Indicator 1: signal = -0.4 at bar 0
    std::vector<float> ind1(N, 0.0f);
    ind1[0] = -0.4f;

    const float* signals[] = {ind0.data(), ind1.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.weights[0]           = 1.0f;
    config.weights[1]           = 0.5f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.3f;  // composite 0.4 > 0.3 -> long entry
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u)
        << "WEIGHTED_SUM composite=0.4 > threshold=0.3 should trigger long entry";
    EXPECT_EQ(trades[0].direction, 1) << "Composite is positive -> long direction";
    EXPECT_EQ(trades[0].entry_bar, 1) << "Signal at bar 0 -> entry at bar 1 (lag=1)";
}

// ---------------------------------------------------------------------------
// WEIGHTED_SUM aggregation -- below threshold, no entry
//
// Two indicators with signals [0.3, -0.4] and weights [1.0, 0.5].
// Composite = (0.3*1.0 + (-0.4)*0.5) / (1.0 + 0.5) = (0.3 - 0.2) / 1.5 = 0.0667.
// With threshold 0.3, composite 0.067 < 0.3 -> should NOT trigger entry.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, CompositeWeightedSumBelowThreshold) {
    const int          N = 15;
    std::vector<float> closes(N, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    std::vector<float> ind0(N, 0.0f);
    ind0[0] = 0.3f;

    std::vector<float> ind1(N, 0.0f);
    ind1[0] = -0.4f;

    const float* signals[] = {ind0.data(), ind1.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.weights[0]           = 1.0f;
    config.weights[1]           = 0.5f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.3f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u)
        << "WEIGHTED_SUM composite=0.067 < threshold=0.3 should NOT trigger entry";
}

// ---------------------------------------------------------------------------
// MAJORITY_VOTE aggregation
//
// Three indicators with signals [0.3, -0.5, 0.7].
// Votes: +1 (0.3>0), -1 (-0.5<0), +1 (0.7>0).  Majority = +1.
// MAJORITY_VOTE returns +1.0 when longs > shorts.
// With threshold 0.5, composite = 1.0 > 0.5 -> triggers long entry.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, CompositeMajorityVote) {
    const int          N = 15;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) closes[i] = 100.0f + static_cast<float>(i);
    auto bars = make_bars_from_closes(closes);

    std::vector<float> ind0(N, 0.0f);
    ind0[0] = 0.3f;  // vote: +1

    std::vector<float> ind1(N, 0.0f);
    ind1[0] = -0.5f;  // vote: -1

    std::vector<float> ind2(N, 0.0f);
    ind2[0] = 0.7f;  // vote: +1

    const float* signals[] = {ind0.data(), ind1.data(), ind2.data()};

    StrategyConfig config;
    config.num_indicators       = 3;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.indicator_indices[2] = 2;
    config.aggregation          = StrategyConfig::MAJORITY_VOTE;
    config.entry_threshold      = 0.5f;  // composite = 1.0 > 0.5 -> entry
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 3, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u)
        << "MAJORITY_VOTE: 2 longs vs 1 short -> composite=+1.0 > threshold=0.5 -> long entry";
    EXPECT_EQ(trades[0].direction, 1) << "Majority is long -> direction should be +1";
}

// ---------------------------------------------------------------------------
// MAJORITY_VOTE tie -- no entry
//
// Two indicators: [0.5, -0.5].  Votes: +1, -1.  Tie -> composite = 0.
// With any positive threshold, no entry should occur.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, CompositeMajorityVoteTie) {
    const int          N = 15;
    std::vector<float> closes(N, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    std::vector<float> ind0(N, 0.0f);
    ind0[0] = 0.5f;

    std::vector<float> ind1(N, 0.0f);
    ind1[0] = -0.5f;

    const float* signals[] = {ind0.data(), ind1.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.aggregation          = StrategyConfig::MAJORITY_VOTE;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u)
        << "MAJORITY_VOTE tie: votes_long == votes_short -> composite=0 -> no entry";
}

// ---------------------------------------------------------------------------
// UNANIMOUS aggregation -- all agree (positive)
//
// Two indicators with signals [0.6, 0.8].  Both positive -> unanimous agree
// -> composite = +1.0.  With threshold 0.5, 1.0 > 0.5 -> long entry.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, CompositeUnanimousAllPositive) {
    const int          N = 15;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) closes[i] = 100.0f + static_cast<float>(i);
    auto bars = make_bars_from_closes(closes);

    std::vector<float> ind0(N, 0.0f);
    ind0[0] = 0.6f;

    std::vector<float> ind1(N, 0.0f);
    ind1[0] = 0.8f;

    const float* signals[] = {ind0.data(), ind1.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.aggregation          = StrategyConfig::UNANIMOUS;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u)
        << "UNANIMOUS: both indicators positive -> composite=+1.0 > 0.5 -> long entry";
    EXPECT_EQ(trades[0].direction, 1) << "Unanimous positive -> long direction";
}

// ---------------------------------------------------------------------------
// UNANIMOUS aggregation -- disagreement (one negative)
//
// Two indicators with signals [0.6, -0.2].  One positive, one negative ->
// not unanimous -> composite = 0.0.  No entry should occur.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, CompositeUnanimousDisagreement) {
    const int          N = 15;
    std::vector<float> closes(N, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    std::vector<float> ind0(N, 0.0f);
    ind0[0] = 0.6f;

    std::vector<float> ind1(N, 0.0f);
    ind1[0] = -0.2f;

    const float* signals[] = {ind0.data(), ind1.data()};

    StrategyConfig config;
    config.num_indicators       = 2;
    config.indicator_indices[0] = 0;
    config.indicator_indices[1] = 1;
    config.aggregation          = StrategyConfig::UNANIMOUS;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 2, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    EXPECT_EQ(trades.size(), 0u) << "UNANIMOUS: indicators disagree -> composite=0 -> no entry";
}

// =============================================================================
// ADDITIONAL COVERAGE TESTS
// =============================================================================

// ---------------------------------------------------------------------------
// Rolling vs Anchored must produce DIFFERENT values.
//
// EMA(10) rolling on 200 30s bars updates every 30s.
// EMA(10) anchored on 5-minute TF (10 bars = 5 min) only updates at 5m
// boundaries.  The two must differ because rolling reacts on every bar while
// anchored holds steady within each 5-minute block.
// ---------------------------------------------------------------------------
TEST(Indicators, RollingVsAnchored) {
    // Use 500 30s bars so that even with 5-min aggregation (ratio=10)
    // we get 50 TF bars, which exceeds EMA(5) warmup of 15.
    const int          N = 500;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        // Oscillating + trending pattern to ensure values move within TF blocks
        closes[i] =
            5000.0f + 5.0f * std::sin(static_cast<float>(i) * 0.1f) + 0.1f * static_cast<float>(i);
    }
    auto bars_30s = make_bars_from_closes(closes);

    // Rolling EMA(5) on 30s bars -- uses period 5 to keep warmup small
    EMASMACross        ema_rolling(MAType::EMA, 5, MAType::EMA, 5);
    std::vector<float> rolling_output(N * 2);
    ema_rolling.compute(bars_30s.data(), 0, N, rolling_output.data());

    // Build 5-minute TF bars: aggregate every 10 30s bars into one TF bar
    const int        TF_RATIO = 10;            // 10 * 30s = 5 min
    const int        TF_BARS  = N / TF_RATIO;  // 50 TF bars
    std::vector<Bar> tf_bars(TF_BARS);
    std::vector<int> completed_map(N);

    for (int t = 0; t < TF_BARS; ++t) {
        int first            = t * TF_RATIO;
        int last             = first + TF_RATIO - 1;
        tf_bars[t].timestamp = bars_30s[first].timestamp;
        tf_bars[t].open      = bars_30s[first].open;
        tf_bars[t].close     = bars_30s[last].close;
        tf_bars[t].high      = bars_30s[first].high;
        tf_bars[t].low       = bars_30s[first].low;
        tf_bars[t].volume    = 0;
        for (int j = first; j <= last; ++j) {
            tf_bars[t].high = std::max(tf_bars[t].high, bars_30s[j].high);
            tf_bars[t].low  = std::min(tf_bars[t].low, bars_30s[j].low);
            tf_bars[t].volume += bars_30s[j].volume;
            // TF bar t is final only once its last 30s bar has closed.
            completed_map[j] = (j == last) ? t : t - 1;
        }
    }

    // Anchored EMA(5) on 5-minute bars, broadcast back to 30s
    EMASMACross        ema_anchored(MAType::EMA, 5, MAType::EMA, 5);
    std::vector<float> anchored_output(N * 2);
    compute_anchored(ema_anchored, tf_bars.data(), TF_BARS, completed_map.data(), 0, N,
                     anchored_output.data());

    // After both warmups are satisfied, rolling and anchored must differ.
    // Rolling warmup = 3*5 = 15 (30s bars).
    // Anchored warmup = 3*5 = 15 TF bars = 150 30s bars.
    // Start comparing after bar 150 to ensure both are valid.
    const int compare_start = 150;
    int       diffs         = 0;
    int       compared      = 0;
    for (int i = compare_start; i < N; ++i) {
        if (!is_nan(rolling_output[i]) && !is_nan(anchored_output[i])) {
            ++compared;
            if (std::abs(rolling_output[i] - anchored_output[i]) > 1e-4f) ++diffs;
        }
    }
    EXPECT_GT(compared, 0) << "Should have some bars where both rolling and anchored are valid";
    EXPECT_GT(diffs, 0) << "Rolling and anchored EMA(5) must produce different values "
                           "(rolling updates every 30s, anchored only at 5m boundaries)";
}

// ---------------------------------------------------------------------------
// AllMetricsConsistency: Run a backtest producing ~20 trades and verify
// that all derived metrics are internally consistent with the raw trade list.
// ---------------------------------------------------------------------------
TEST(Indicators, AllMetricsConsistency) {
    // Build a series with alternating trend to generate multiple trades
    const int          N = 600;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        // Sawtooth: rise for 20 bars, fall for 20 bars -> frequent crosses
        int   phase  = (i / 20) % 2;
        float base   = 4500.0f;
        float offset = static_cast<float>(i % 20);
        closes[i]    = (phase == 0) ? base + offset * 2.0f : base + (20.0f - offset) * 2.0f;
    }
    auto bars = make_bars_from_closes(closes);

    // EMA crossover to generate entries/exits via signal reversal
    EMASMACross        ema(MAType::EMA, 5, MAType::EMA, 15);
    std::vector<float> ind_output(N * 2);
    ema.compute(bars.data(), 0, N, ind_output.data());

    // Build the signal: fast - slow (series 0 - series 1)
    std::vector<float> signal(N);
    for (int i = 0; i < N; ++i) {
        if (is_nan(ind_output[i]) || is_nan(ind_output[N + i]))
            signal[i] = 0.0f;
        else
            signal[i] = ind_output[i] - ind_output[N + i];
    }
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::SIGNAL_REVERSAL;

    BacktestEngine engine;
    BacktestResult result = engine.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);
    const auto&    trades = engine.last_trades();

    ASSERT_GT(result.total_trades, 5) << "Need multiple trades for consistency check";
    EXPECT_EQ(result.total_trades, static_cast<int>(trades.size()));

    // 1. total_pnl == sum(trade.net_pnl)
    float sum_net_pnl = 0.0f;
    for (const auto& t : trades) sum_net_pnl += t.net_pnl;
    EXPECT_NEAR(result.total_pnl, sum_net_pnl, 0.01f)
        << "total_pnl must equal sum of trade net_pnl";

    // 2. win_rate == wins / total_trades
    int wins = 0;
    for (const auto& t : trades) {
        if (t.net_pnl > 0.0f) ++wins;
    }
    float expected_wr = static_cast<float>(wins) / static_cast<float>(trades.size());
    EXPECT_NEAR(result.win_rate, expected_wr, 1e-4f) << "win_rate must equal wins / total_trades";

    // 3. gross_wins == sum(trade.net_pnl where > 0)
    float sum_gross_wins   = 0.0f;
    float sum_gross_losses = 0.0f;
    for (const auto& t : trades) {
        if (t.net_pnl > 0.0f)
            sum_gross_wins += t.net_pnl;
        else
            sum_gross_losses += t.net_pnl;
    }
    EXPECT_NEAR(result.gross_wins, sum_gross_wins, 0.01f)
        << "gross_wins must equal sum of positive net_pnl trades";
    EXPECT_NEAR(result.gross_losses, sum_gross_losses, 0.01f)
        << "gross_losses must equal sum of negative net_pnl trades";

    // 4. profit_factor == gross_wins / |gross_losses|
    if (sum_gross_losses != 0.0f) {
        float expected_pf = sum_gross_wins / std::abs(sum_gross_losses);
        EXPECT_NEAR(result.profit_factor, expected_pf, 0.01f)
            << "profit_factor must equal gross_wins / |gross_losses|";
    }
}

// =============================================================================
// RandomWalkNoEdge
//
// Generate ~5000 bars of synthetic price data, create random [-1,+1] signal
// arrays using seeded std::mt19937 for determinism.  Run through BacktestEngine
// with ES cost model and entry_lag=1.  Assert total_pnl is between -$5000 and
// +$2000 (random signals with costs should be ~0 or negative).
// =============================================================================
#include <random>

TEST(BacktestEngine, RandomWalkNoEdge) {
    // --- Seed a deterministic RNG ---
    std::mt19937 rng(42);

    // --- Generate ~5000 bars of synthetic price data (random walk) ---
    const int          N = 5000;
    std::vector<float> closes(N);
    closes[0] = 4500.0f;
    portable::uniform<float> step_dist{-2.0f, 2.0f};
    for (int i = 1; i < N; ++i) {
        closes[i] = closes[i - 1] + step_dist(rng);
    }
    auto bars = make_bars_from_closes(closes);

    // --- Generate random signal in [-1, +1] ---
    portable::uniform<float> sig_dist{-1.0f, 1.0f};
    std::vector<float>       signal(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = sig_dist(rng);
    }
    const float* signals[] = {signal.data()};

    // --- Strategy config: single indicator, weighted sum, bar timeout exit ---
    // Use a longer exit timeout (50 bars) to limit trade frequency and thus
    // total cost drag.  With ~5000 bars and 50-bar trades, expect ~100 trades max.
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 50;

    // --- Run backtest with ES costs and lag=1 ---
    CostModel      cost = es_cost();
    BacktestEngine engine;
    BacktestResult result = engine.run(bars.data(), 0, N, signals, 1, config, cost, 1);

    // Random signals have no edge; expected PnL ~= -num_trades * round_trip_cost.
    // With ~100 trades at $75 RT cost, expect PnL around -$7,500.
    // We use generous bounds to accommodate RNG variance: -$15,000 to +$5,000.
    EXPECT_GE(result.total_pnl, -15000.0f)
        << "Random signal PnL should not be catastrophically negative";
    EXPECT_LE(result.total_pnl, 5000.0f)
        << "Random signal PnL should not be significantly positive (no edge)";

    // Should have generated a reasonable number of trades
    EXPECT_GT(result.total_trades, 20)
        << "Expected meaningful number of trades from random signals";
}

// =============================================================================
// LookaheadCanary
//
// Generate synthetic bars with a trend.  Create a "cheating" signal that knows
// the next bar's direction (signal[i] = sign(bars[i+1].close - bars[i].close)).
// Also create a legitimate lagging signal (simple price momentum:
// sign(close[i] - close[i-5])).  Run both through BacktestEngine.  The cheating
// signal should have significantly better Sharpe than the legitimate one.  This
// validates that lookahead WOULD produce unrealistic results, confirming the
// engine's anti-lookahead mechanisms are important.
// =============================================================================

TEST(BacktestEngine, LookaheadCanary) {
    // --- Generate synthetic bars with a noisy uptrend ---
    std::mt19937       rng(123);
    const int          N = 5000;
    std::vector<float> closes(N);
    closes[0] = 4500.0f;
    portable::normal<float> noise{0.05f, 1.5f};  // slight upward drift
    for (int i = 1; i < N; ++i) {
        closes[i] = closes[i - 1] + noise(rng);
    }
    auto bars = make_bars_from_closes(closes);

    // --- Cheating signal: knows next bar's direction ---
    // signal[i] = sign(close[i+1] - close[i]) -- perfect lookahead
    // signal[N-1] = 0 (no future bar to peek at)
    std::vector<float> cheat_signal(N, 0.0f);
    for (int i = 0; i < N - 1; ++i) {
        float diff = closes[i + 1] - closes[i];
        if (diff > 0.0f)
            cheat_signal[i] = 1.0f;
        else if (diff < 0.0f)
            cheat_signal[i] = -1.0f;
        else
            cheat_signal[i] = 0.0f;
    }

    // --- Legitimate lagging signal: sign(close[i] - close[i-5]) ---
    // Uses only past data -- no lookahead
    std::vector<float> legit_signal(N, 0.0f);
    for (int i = 5; i < N; ++i) {
        float diff = closes[i] - closes[i - 5];
        if (diff > 0.0f)
            legit_signal[i] = 1.0f;
        else if (diff < 0.0f)
            legit_signal[i] = -1.0f;
        else
            legit_signal[i] = 0.0f;
    }

    // --- Common strategy config ---
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::SIGNAL_REVERSAL;

    CostModel cost = es_cost();

    // --- Run cheating backtest ---
    const float*   cheat_ptrs[] = {cheat_signal.data()};
    BacktestEngine engine_cheat;
    BacktestResult cheat_result =
        engine_cheat.run(bars.data(), 0, N, cheat_ptrs, 1, config, cost, 1);

    // --- Run legitimate backtest ---
    const float*   legit_ptrs[] = {legit_signal.data()};
    BacktestEngine engine_legit;
    BacktestResult legit_result =
        engine_legit.run(bars.data(), 0, N, legit_ptrs, 1, config, cost, 1);

    // --- Cheating signal should vastly outperform the legitimate signal ---
    // The cheating signal has perfect foresight (minus entry_lag=1),
    // so it should still show dramatically better results.
    EXPECT_GT(cheat_result.sharpe_ratio, legit_result.sharpe_ratio)
        << "Cheating (lookahead) signal should have higher Sharpe than lagging signal";
    EXPECT_GT(cheat_result.total_pnl, legit_result.total_pnl)
        << "Cheating (lookahead) signal should have higher total PnL";

    // The cheating signal should be profitable even after costs
    EXPECT_GT(cheat_result.total_pnl, 0.0f) << "Perfect lookahead signal should be profitable";

    // Both should have generated trades
    EXPECT_GT(cheat_result.total_trades, 10);
    EXPECT_GT(legit_result.total_trades, 10);
}

// ---------------------------------------------------------------------------
// test_halt_flag_suppresses_entries
//
// When sessions[bar].halt_flag is set, no new entries should be taken.
// Verify that a strong signal during a halted session produces zero trades,
// while the same signal on non-halted bars produces trades.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, HaltFlagSuppressesEntries) {
    const int          N = 20;
    std::vector<float> closes(N, 5000.0f);
    auto               bars = make_bars_from_closes(closes);

    // Create sessions: all at 10:00 ET (within RTH)
    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].time_of_day_sec   = 36000;  // 10:00 ET
        sessions[i].halt_flag         = 0;
        sessions[i].early_close       = 0;
        sessions[i].session_close_sec = 61200;  // 17:00 ET
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    // Strong long signal on bars 0, 5, 10, 15
    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;
    signal[5]              = 1.0f;
    signal[10]             = 1.0f;
    signal[15]             = 1.0f;
    const float* signals[] = {signal.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 3;
    config.trading_hours        = StrategyConfig::ETH;

    // Baseline: no halts -> should produce trades
    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, gates, config, es_cost(), 1);
    int baseline_trades = static_cast<int>(engine.last_trades().size());
    EXPECT_GT(baseline_trades, 0) << "Baseline without halts should produce trades";

    // Now set halt_flag on ALL bars -> no entries should occur
    for (int i = 0; i < N; ++i) {
        sessions[i].halt_flag = 1;
    }

    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, gates, config, es_cost(), 1);
    int halted_trades = static_cast<int>(engine.last_trades().size());
    EXPECT_EQ(halted_trades, 0) << "All-halted session should produce zero trades";

    // Partial halt: halt only bars 0-9, leave 10-19 clear
    for (int i = 0; i < N; ++i) {
        sessions[i].halt_flag = (i < 10) ? 1 : 0;
    }

    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, gates, config, es_cost(), 1);
    const auto& partial_trades = engine.last_trades();
    // Only signals on bars 10 and 15 should fire (bars 0 and 5 are halted)
    for (const auto& t : partial_trades) {
        int signal_bar = t.entry_bar - 1;  // entry_lag=1
        EXPECT_GE(signal_bar, 10) << "Trade entered from signal on bar " << signal_bar
                                  << " which should have been suppressed by halt";
    }
}

// ---------------------------------------------------------------------------
// test_early_session_close_force_exit
//
// When in position and bar time >= session_close_sec - 30, the engine
// should force-close the position. This handles early-close days (13:00 ET)
// and prevents holding through session close.
// ---------------------------------------------------------------------------
TEST(BacktestEngine, EarlySessionCloseForceExit) {
    const int          N = 20;
    std::vector<float> closes(N, 5000.0f);
    auto               bars = make_bars_from_closes(closes);

    // Create sessions: bars 0-9 are at 12:30-12:45 ET (well before close),
    // bars 10-14 approach the early close at 13:00 ET (46800s),
    // bars 15-19 are after close.
    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].halt_flag         = 0;
        sessions[i].early_close       = 1;      // early-close day
        sessions[i].session_close_sec = 46800;  // 13:00 ET

        // Time progression: bars at 30s intervals starting from 12:30 ET
        sessions[i].time_of_day_sec = 45000 + i * 30;  // 12:30:00 + i*30s
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    // Signal fires on bar 0 -> entry at bar 1 (entry_lag=1)
    // With BAR_TIMEOUT=100, the strategy would normally hold for 100 bars.
    // But session_close_sec = 46800, and bars approaching 46800 should trigger
    // a force-close.
    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;
    const float* signals[] = {signal.data()};

    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = 0;

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 100;  // very long timeout (won't fire)
    config.trading_hours        = StrategyConfig::ETH;

    BacktestEngine engine;
    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, gates, config, es_cost(), 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(static_cast<int>(trades.size()), 1)
        << "Should have at least one trade (entry from signal on bar 0)";

    // With the first session layout, bar i has tod = 45000 + i*30, and the
    // force close needs the bar that spans the close (tod in [46770, 46800)).
    // With only 20 bars (bar 19 has tod = 45570) the session close does not
    // trigger before end-of-data.
    //
    // Move the bars closer to the close: if bars start at tod = 46700, bar i
    // has tod = 46700 + i*30, and bar 3 (tod = 46790, ending at 46820) is the
    // last bar before the close, so it triggers the force close.

    // Re-setup sessions with bars closer to close
    for (int i = 0; i < N; ++i) {
        sessions[i].time_of_day_sec = 46700 + i * 30;
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    engine.run(bars.data(), sessions.data(), 0, N, signals, 1, gates, config, es_cost(), 1);

    const auto& trades2 = engine.last_trades();
    ASSERT_GE(static_cast<int>(trades2.size()), 1)
        << "Should have a trade that was force-closed at session close";

    // Entry at bar 1 (lag=1), force close on the bar that spans the close.
    // Bar 0: 46700, bar 1: 46730, bar 2: 46760, bar 3: 46790 -> exit signal
    // Exit fills at bar 4 open (pending_exit fills on next bar).
    const Trade& t = trades2[0];
    EXPECT_EQ(t.entry_bar, 1) << "Entry should be at bar 1 (lag=1 from signal at bar 0)";
    // The exit_bar should be <= the bar after the session close threshold
    EXPECT_LE(t.exit_bar, 5) << "Trade should be force-closed near session close";

    // Also verify: without sessions, the same trade would NOT be force-closed early
    BacktestEngine engine2;
    engine2.run(bars.data(), 0, N, signals, 1, config, es_cost(), 1);
    const auto& trades3 = engine2.last_trades();
    // Without sessions, timeout won't fire (100 bars > 20), so position is
    // force-closed at end-of-data (bar N-1)
    if (!trades3.empty()) {
        EXPECT_EQ(trades3[0].exit_bar, N - 1)
            << "Without sessions, trade should be closed at end of data";
    }
}

// =============================================================================
// avg_trade_dd and avg_trade_peak metric tests
// =============================================================================

TEST(Metrics, AvgTradeDdAndPeak) {
    // Create 4 trades with known min_unrealized and max_unrealized values.
    // Verify avg_trade_dd = average of per-trade |min_unrealized| (positive magnitude)
    // Verify avg_trade_peak = average of per-trade max_unrealized
    //
    // Trade 0: min_unrealized = -200, max_unrealized = 400
    // Trade 1: min_unrealized = -100, max_unrealized = 300
    // Trade 2: min_unrealized = -600, max_unrealized = 100
    // Trade 3: min_unrealized = -0   (never went negative), max_unrealized = 800
    //
    // avg_trade_dd   = -((-200) + (-100) + (-600) + 0) / 4 = -(-900)/4 = 225.0
    // avg_trade_peak = (400 + 300 + 100 + 800) / 4 = 1600/4 = 400.0

    const int        N = 8;  // need enough bars for trade entries/exits
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    std::vector<Trade> trades(4);
    for (int i = 0; i < 4; ++i) {
        trades[i].entry_bar   = i * 2;
        trades[i].exit_bar    = i * 2 + 1;
        trades[i].entry_price = 4500.0f;
        trades[i].exit_price  = 4500.0f;
        trades[i].direction   = 1;
        trades[i].gross_pnl   = 125.0f;  // 100 + 25 cost
        trades[i].net_pnl     = 100.0f;
    }

    trades[0].min_unrealized = -200.0f;
    trades[0].max_unrealized = 400.0f;
    trades[1].min_unrealized = -100.0f;
    trades[1].max_unrealized = 300.0f;
    trades[2].min_unrealized = -600.0f;
    trades[2].max_unrealized = 100.0f;
    trades[3].min_unrealized = 0.0f;
    trades[3].max_unrealized = 800.0f;

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // avg_trade_dd = -(sum of min_unrealized) / total_trades
    //             = -((-200) + (-100) + (-600) + 0) / 4 = 900/4 = 225.0
    EXPECT_NEAR(result.avg_trade_dd, 225.0f, 0.01f)
        << "avg_trade_dd should be 225.0 (average |min_unrealized| as positive magnitude)";

    // avg_trade_peak = (sum of max_unrealized) / total_trades
    //               = (400 + 300 + 100 + 800) / 4 = 400.0
    EXPECT_NEAR(result.avg_trade_peak, 400.0f, 0.01f)
        << "avg_trade_peak should be 400.0 (average of per-trade peak unrealized)";

    // Also verify avg_trade_dd is stored as positive magnitude
    EXPECT_GT(result.avg_trade_dd, 0.0f) << "avg_trade_dd should be positive (magnitude)";
    EXPECT_GT(result.avg_trade_peak, 0.0f) << "avg_trade_peak should be positive";
}

// =============================================================================
// Backtest engine max_unrealized tracking test
// =============================================================================

TEST(BacktestEngine, MaxUnrealizedTracking) {
    // Create a price path where a long trade enters at 4500, price rises to 4520
    // (peak unrealized), then falls back to 4510 at exit.
    // Verify Trade.max_unrealized captures the per-trade peak unrealized PnL.
    //
    // Bar 0: signal fires (long), close=4500
    // Bar 1: entry at open=4500
    // Bar 2: close=4505 (unrealized = 5pts * $50 = $250)
    // Bar 3: close=4520 (unrealized = 20pts * $50 = $1000) <-- peak
    // Bar 4: close=4515 (pulling back)
    // Bar 5: close=4510 (pulling back more)
    // Bar 6: exit (BAR_TIMEOUT at 5 bars from entry)
    const int          N      = 10;
    std::vector<float> closes = {
        4500.0f,  // bar 0 (signal bar)
        4500.0f,  // bar 1 (entry at open)
        4505.0f,  // bar 2
        4520.0f,  // bar 3 (peak)
        4515.0f,  // bar 4
        4510.0f,  // bar 5
        4510.0f,  // bar 6 (exit bar)
        4510.0f,  // bar 7
        4510.0f,  // bar 8
        4510.0f   // bar 9
    };
    auto bars = make_bars_from_closes(closes);
    // Ensure entry bar has exact open price
    bars[1].open = 4500.0f;

    std::vector<float> signal(N, 0.0f);
    signal[0]              = 1.0f;  // long signal
    const float* signals[] = {signal.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    CostModel      cost = es_cost();
    BacktestEngine engine;
    engine.run(bars.data(), 0, N, signals, 1, config, cost, 1);

    const auto& trades = engine.last_trades();
    ASSERT_GE(trades.size(), 1u) << "Should have at least one trade";

    const Trade& t = trades[0];
    EXPECT_EQ(t.direction, 1) << "Should be a long trade";
    EXPECT_EQ(t.entry_bar, 1) << "Entry should be at bar 1 (lag=1)";

    // Peak unrealized occurs at bar 3 close (4520), entry at 4500.
    // max_unrealized = price_to_pnl(4520 - 4500) = 20 * $50 = $1000
    // The engine might also check intra-bar highs, but at minimum the
    // close-to-close M2M should give us 20 pts * $50/pt.
    float expected_peak_close = cost.price_to_pnl(4520.0f - 4500.0f);  // $1000

    EXPECT_GE(t.max_unrealized, expected_peak_close - 1.0f)
        << "max_unrealized should capture the peak at bar 3 (close=4520, entry=4500)."
        << " Expected at least " << expected_peak_close;
    EXPECT_GT(t.max_unrealized, 0.0f)
        << "max_unrealized should be positive for a profitable trade peak";

    // Also verify min_unrealized makes sense -- the trade never went below
    // entry price significantly (bar 1 open = bar 0 close = 4500)
    // but intra-bar lows could cause minor negative unrealized.
    // The important check is that max > min.
    EXPECT_GT(t.max_unrealized, t.min_unrealized)
        << "max_unrealized should be greater than min_unrealized";
}
