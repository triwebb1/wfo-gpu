// =============================================================================
// Full Indicator Catalog Tests
//
// Tests for the 11 extended-catalog indicators. Each indicator has at least one test for:
// - Basic computation with hand-verified expected values
// - Warmup period (NaN before warmup, valid after)
// - Range/validity constraints where applicable
//
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>
#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "market_data.h"
#include "indicators/indicator.h"
#include "indicators/atr.h"
#include "indicators/macd.h"
#include "indicators/adx.h"
#include "indicators/bollinger.h"
#include "indicators/keltner.h"
#include "indicators/stochastic.h"
#include "indicators/cci.h"
#include "indicators/roc.h"
#include "indicators/vwap.h"
#include "indicators/obv.h"
#include "indicators/donchian.h"
#include "indicators/ema_sma_cross.h"
#include "indicators/rsi.h"
#include "indicators/hma.h"
#include "indicators/connors_rsi.h"
#include "indicators/pivot_points.h"

// =============================================================================
// Helpers
// =============================================================================

// Generate N synthetic bars with close prices from the given vector.
// High = max(open, close) + 0.5, Low = min(open, close) - 0.5.
// Open of bar i+1 = close of bar i (continuous).
static std::vector<Bar> make_bars_from_closes(const std::vector<float>& closes,
                                              int64_t base_ts = 1'262'304'000LL * NS_PER_SEC) {
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

// Generate N bars with explicit OHLCV per bar.
// Kept for future use in expanded indicator tests.
#if 0
static std::vector<Bar> make_bars_ohlcv(
    const std::vector<std::array<float, 5>>& ohlcv,
    int64_t base_ts = 1'262'304'000LL * NS_PER_SEC)
{
    std::vector<Bar> bars;
    bars.reserve(ohlcv.size());
    for (size_t i = 0; i < ohlcv.size(); ++i) {
        int64_t ts = base_ts + static_cast<int64_t>(i) * 30 * NS_PER_SEC;
        bars.push_back(make_bar(ts, ohlcv[i][0], ohlcv[i][1], ohlcv[i][2],
                                ohlcv[i][3], static_cast<uint32_t>(ohlcv[i][4])));
    }
    return bars;
}
#endif

static bool is_nan(float v) {
    return std::isnan(v);
}

// =============================================================================
// 1. MACD TESTS
// =============================================================================

TEST(MACD_Test, BasicComputation) {
    // Generate enough bars to get past warmup for MACD(3,5,3)
    // warmup = 3 * slow_period + 3 * signal_period = 3*5 + 3*3 = 24
    std::vector<float> closes;
    for (int i = 0; i < 40; ++i) {
        closes.push_back(100.0f + static_cast<float>(i) * 0.5f);  // trending up
    }
    auto bars = make_bars_from_closes(closes);

    MACD macd(3, 5, 3);
    EXPECT_EQ(macd.num_series(), 3);
    EXPECT_EQ(macd.warmup_bars(), 24);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    macd.compute(bars.data(), 0, N, output.data());

    // Check warmup: first 24 values should be NaN for all 3 series
    for (int i = 0; i < 24; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "MACD line should be NaN at i=" << i;
        EXPECT_TRUE(is_nan(output[N + i])) << "Signal should be NaN at i=" << i;
        EXPECT_TRUE(is_nan(output[2 * N + i])) << "Histogram should be NaN at i=" << i;
    }

    // After warmup: values should be valid (not NaN)
    for (int i = 24; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "MACD line should be valid at i=" << i;
        EXPECT_FALSE(is_nan(output[N + i])) << "Signal should be valid at i=" << i;
        EXPECT_FALSE(is_nan(output[2 * N + i])) << "Histogram should be valid at i=" << i;
    }

    // In an uptrend, MACD (fast - slow) should be positive
    // because fast EMA reacts more quickly to the uptrend
    for (int i = 30; i < N; ++i) {
        EXPECT_GT(output[i], 0.0f) << "MACD should be positive in uptrend at i=" << i;
    }

    // Histogram = MACD - signal
    for (int i = 24; i < N; ++i) {
        float expected_hist = output[i] - output[N + i];
        EXPECT_NEAR(output[2 * N + i], expected_hist, 1e-5f)
            << "Histogram should equal MACD - signal at i=" << i;
    }
}

TEST(MACD_Test, WarmupPeriod) {
    std::vector<float> closes(120, 50.0f);  // flat data (enough for warmup 105)
    auto               bars = make_bars_from_closes(closes);

    MACD macd(12, 26, 9);
    EXPECT_EQ(macd.warmup_bars(), 105);  // 3 * 26 + 3 * 9

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    macd.compute(bars.data(), 0, N, output.data());

    // All values during warmup should be NaN
    for (int i = 0; i < 105; ++i) {
        EXPECT_TRUE(is_nan(output[i]));
    }
    // Values after warmup should be valid
    for (int i = 105; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i]));
    }
}

// =============================================================================
// 2. ADX TESTS
// =============================================================================

TEST(ADX_Test, BasicComputation) {
    // Generate data with clear trend to get positive ADX
    std::vector<float> closes;
    for (int i = 0; i < 80; ++i) {
        closes.push_back(100.0f + static_cast<float>(i) * 1.0f);  // strong uptrend
    }
    auto bars = make_bars_from_closes(closes);

    ADX adx(7);
    EXPECT_EQ(adx.num_series(), 3);
    EXPECT_EQ(adx.warmup_bars(), 28);  // 4 * 7

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    adx.compute(bars.data(), 0, N, output.data());

    // Check warmup: first 28 values should be NaN
    for (int i = 0; i < 28; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "ADX NaN at i=" << i;
        EXPECT_TRUE(is_nan(output[N + i])) << "+DI NaN at i=" << i;
        EXPECT_TRUE(is_nan(output[2 * N + i])) << "-DI NaN at i=" << i;
    }

    // After warmup: ADX should be valid and in [0, 100]
    bool found_valid_adx = false;
    for (int i = 28; i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_GE(output[i], 0.0f) << "ADX >= 0 at i=" << i;
            EXPECT_LE(output[i], 100.0f) << "ADX <= 100 at i=" << i;
            found_valid_adx = true;
        }
        // +DI and -DI should be valid and >= 0
        if (!is_nan(output[N + i])) {
            EXPECT_GE(output[N + i], 0.0f) << "+DI >= 0 at i=" << i;
        }
        if (!is_nan(output[2 * N + i])) {
            EXPECT_GE(output[2 * N + i], 0.0f) << "-DI >= 0 at i=" << i;
        }
    }
    EXPECT_TRUE(found_valid_adx) << "Should have at least some valid ADX values";
}

// =============================================================================
// 3. BOLLINGER BANDS TESTS
// =============================================================================

TEST(Bollinger_Test, BasicComputation) {
    // Use constant close prices -- SMA = price, stddev = 0
    // So upper = lower = middle = price
    const int          period = 5;
    std::vector<float> closes(20, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    BollingerBands bb(period, 2.0f);
    EXPECT_EQ(bb.num_series(), 3);
    EXPECT_EQ(bb.warmup_bars(), period - 1);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    bb.compute(bars.data(), 0, N, output.data());

    // After warmup, middle should be 100 and upper == lower == 100
    // because stddev of constant series is 0
    for (int i = period - 1; i < N; ++i) {
        EXPECT_NEAR(output[i], 100.0f, 1e-3f) << "Middle at i=" << i;
        EXPECT_NEAR(output[N + i], 100.0f, 1e-3f) << "Upper at i=" << i;
        EXPECT_NEAR(output[2 * N + i], 100.0f, 1e-3f) << "Lower at i=" << i;
    }
}

TEST(Bollinger_Test, BandOrdering) {
    // With varying prices, upper >= middle >= lower should always hold
    std::vector<float> closes;
    for (int i = 0; i < 30; ++i) {
        closes.push_back(100.0f + std::sin(static_cast<float>(i) * 0.5f) * 5.0f);
    }
    auto bars = make_bars_from_closes(closes);

    BollingerBands     bb(10, 2.0f);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    bb.compute(bars.data(), 0, N, output.data());

    for (int i = bb.warmup_bars(); i < N; ++i) {
        EXPECT_GE(output[N + i], output[i]) << "Upper >= middle at i=" << i;
        EXPECT_LE(output[2 * N + i], output[i]) << "Lower <= middle at i=" << i;
    }
}

// =============================================================================
// 4. KELTNER CHANNELS TESTS
// =============================================================================

TEST(Keltner_Test, BasicComputation) {
    // Keltner with EMA center
    std::vector<float> closes;
    for (int i = 0; i < 60; ++i) {
        closes.push_back(100.0f + static_cast<float>(i % 10) * 0.5f);
    }
    auto bars = make_bars_from_closes(closes);

    KeltnerChannels kc(MAType::EMA, 10, 7, 2.0f);
    EXPECT_EQ(kc.num_series(), 3);
    // warmup = max(3*10, 3*7) = max(30, 21) = 30
    EXPECT_EQ(kc.warmup_bars(), 30);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    kc.compute(bars.data(), 0, N, output.data());

    // Warmup check
    for (int i = 0; i < 30; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "Center NaN at i=" << i;
    }

    // After warmup: upper > center > lower (ATR > 0 implies bands spread)
    for (int i = 30; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Center valid at i=" << i;
        EXPECT_GT(output[N + i], output[i]) << "Upper > center at i=" << i;
        EXPECT_LT(output[2 * N + i], output[i]) << "Lower < center at i=" << i;
    }
}

TEST(Keltner_Test, SMAWarmup) {
    KeltnerChannels kc_sma(MAType::SMA, 10, 7, 1.5f);
    // SMA warmup = max(10, 3*7) = max(10, 21) = 21
    EXPECT_EQ(kc_sma.warmup_bars(), 21);

    KeltnerChannels kc_ema(MAType::EMA, 10, 7, 1.5f);
    // EMA warmup = max(3*10, 3*7) = max(30, 21) = 30
    EXPECT_EQ(kc_ema.warmup_bars(), 30);
}

// =============================================================================
// 5. STOCHASTIC TESTS
// =============================================================================

TEST(Stochastic_Test, BasicComputation) {
    // Create bars where we know the highest high and lowest low
    // Use period 5. Bars 0-4: prices going from 100 to 104.
    // At bar 4: highest_high = 104.5, lowest_low = 99.5,
    // close = 104. %K = (104-99.5)/(104.5-99.5)*100 = 90.0
    std::vector<float> closes;
    for (int i = 0; i < 20; ++i) {
        closes.push_back(100.0f + static_cast<float>(i));
    }
    auto bars = make_bars_from_closes(closes);

    Stochastic stoch(5, 3);
    EXPECT_EQ(stoch.num_series(), 2);
    EXPECT_EQ(stoch.warmup_bars(), 8);  // 5 + 3

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 2);
    stoch.compute(bars.data(), 0, N, output.data());

    // Warmup check
    for (int i = 0; i < 8; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "%K NaN at i=" << i;
        EXPECT_TRUE(is_nan(output[N + i])) << "%D NaN at i=" << i;
    }

    // After warmup: %K and %D should be in [0, 100]
    for (int i = 8; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "%K valid at i=" << i;
        EXPECT_FALSE(is_nan(output[N + i])) << "%D valid at i=" << i;
        EXPECT_GE(output[i], 0.0f) << "%K >= 0 at i=" << i;
        EXPECT_LE(output[i], 100.0f) << "%K <= 100 at i=" << i;
        EXPECT_GE(output[N + i], 0.0f) << "%D >= 0 at i=" << i;
        EXPECT_LE(output[N + i], 100.0f) << "%D <= 100 at i=" << i;
    }
}

TEST(Stochastic_Test, RangeValidity) {
    // Random-ish data, verify %K always in [0,100]
    std::vector<float> closes = {100, 102, 98,  105, 97,  103, 99,  101, 104, 96,
                                 106, 95,  108, 93,  110, 91,  112, 90,  114, 88};
    auto               bars   = make_bars_from_closes(closes);

    Stochastic         stoch(5, 3);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 2);
    stoch.compute(bars.data(), 0, N, output.data());

    for (int i = stoch.warmup_bars(); i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_GE(output[i], 0.0f);
            EXPECT_LE(output[i], 100.0f);
        }
    }
}

// =============================================================================
// 6. CCI TESTS
// =============================================================================

TEST(CCI_Test, BasicComputation) {
    // With constant prices, CCI should be 0 (no deviation from mean)
    std::vector<float> closes(20, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    CCI cci(5);
    EXPECT_EQ(cci.num_series(), 1);
    EXPECT_EQ(cci.warmup_bars(), 4);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    cci.compute(bars.data(), 0, N, output.data());

    // Warmup check
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "NaN at i=" << i;
    }

    // With constant prices, all typical prices are the same,
    // so mean_deviation = 0 and CCI = 0
    for (int i = 4; i < N; ++i) {
        EXPECT_NEAR(output[i], 0.0f, 1e-3f) << "CCI should be 0 for constant prices at i=" << i;
    }
}

TEST(CCI_Test, TrendDirection) {
    // Rising prices should produce positive CCI
    std::vector<float> closes;
    for (int i = 0; i < 30; ++i) {
        closes.push_back(100.0f + static_cast<float>(i) * 2.0f);
    }
    auto bars = make_bars_from_closes(closes);

    CCI                cci(10);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    cci.compute(bars.data(), 0, N, output.data());

    // After sufficient bars, CCI should be positive in an uptrend
    for (int i = 15; i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_GT(output[i], 0.0f) << "CCI > 0 in uptrend at i=" << i;
        }
    }
}

// =============================================================================
// 7. ROC TESTS (renumbered from 8)
// =============================================================================

TEST(ROC_Test, BasicComputation) {
    // ROC = (close - close[period_ago]) / close[period_ago] * 100
    // With close going 100, 110, 120... and period=1:
    // ROC[1] = (110 - 100) / 100 * 100 = 10.0
    std::vector<float> closes = {100.0f, 110.0f, 121.0f, 133.1f, 146.41f};
    auto               bars   = make_bars_from_closes(closes);

    ROC roc(1);
    EXPECT_EQ(roc.num_series(), 1);
    EXPECT_EQ(roc.warmup_bars(), 1);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    roc.compute(bars.data(), 0, N, output.data());

    // i=0: NaN (warmup)
    EXPECT_TRUE(is_nan(output[0]));

    // i=1: (110-100)/100*100 = 10.0
    EXPECT_NEAR(output[1], 10.0f, 0.1f);

    // i=2: (121-110)/110*100 = 10.0
    EXPECT_NEAR(output[2], 10.0f, 0.1f);
}

TEST(ROC_Test, SmoothedOutput) {
    // Create ~50 bars with alternating up/down moves so raw ROC oscillates
    std::vector<float> closes;
    float              price = 100.0f;
    for (int i = 0; i < 50; ++i) {
        // Alternating +2 / -1 pattern creates oscillating ROC
        if (i % 2 == 0)
            price += 2.0f;
        else
            price -= 1.0f;
        closes.push_back(price);
    }
    auto bars = make_bars_from_closes(closes);

    const int period    = 5;
    const int smoothing = 3;

    // Compute raw ROC (no smoothing)
    ROC roc_raw(period, 0);
    EXPECT_EQ(roc_raw.num_series(), 1);

    // Compute smoothed ROC
    ROC roc_smooth(period, smoothing);
    EXPECT_EQ(roc_smooth.num_series(), 2);  // raw + smoothed
    EXPECT_EQ(roc_smooth.warmup_bars(), period + 3 * smoothing);

    const int N = static_cast<int>(bars.size());

    std::vector<float> raw_output(N);
    roc_raw.compute(bars.data(), 0, N, raw_output.data());

    std::vector<float> smooth_output(N * 2);
    roc_smooth.compute(bars.data(), 0, N, smooth_output.data());

    const int wu = roc_smooth.warmup_bars();

    // Verify warmup: both raw and smoothed series should be NaN during warmup
    for (int i = 0; i < wu; ++i) {
        EXPECT_TRUE(is_nan(smooth_output[i])) << "Raw ROC NaN at i=" << i;
        EXPECT_TRUE(is_nan(smooth_output[N + i])) << "Smoothed ROC NaN at i=" << i;
    }

    // After warmup: smoothed should be valid
    for (int i = wu; i < N; ++i) {
        EXPECT_FALSE(is_nan(smooth_output[N + i])) << "Smoothed valid at i=" << i;
    }

    // Collect post-warmup raw ROC values (from the unsmoothed indicator)
    // and post-warmup smoothed values for comparison
    std::vector<float> raw_vals, smooth_vals;
    int                raw_wu = roc_raw.warmup_bars();
    int                start  = std::max(wu, raw_wu);
    for (int i = start; i < N; ++i) {
        if (!is_nan(raw_output[i]) && !is_nan(smooth_output[N + i])) {
            raw_vals.push_back(raw_output[i]);
            smooth_vals.push_back(smooth_output[N + i]);
        }
    }
    ASSERT_GT(static_cast<int>(raw_vals.size()), 3) << "Need enough post-warmup bars";

    // Verify at least one bar where smoothed != raw (EMA changes values)
    bool found_diff = false;
    for (size_t i = 0; i < raw_vals.size(); ++i) {
        if (std::abs(raw_vals[i] - smooth_vals[i]) > 1e-6f) {
            found_diff = true;
            break;
        }
    }
    EXPECT_TRUE(found_diff) << "Smoothed values should differ from raw ROC";

    // Verify smoothing reduces variance (EMA is a low-pass filter)
    auto variance = [](const std::vector<float>& v) {
        float sum = 0.0f, sum2 = 0.0f;
        for (float x : v) {
            sum += x;
            sum2 += x * x;
        }
        float mean = sum / static_cast<float>(v.size());
        return sum2 / static_cast<float>(v.size()) - mean * mean;
    };
    float raw_var    = variance(raw_vals);
    float smooth_var = variance(smooth_vals);
    EXPECT_LT(smooth_var, raw_var) << "Smoothed variance should be lower than raw";
}

// =============================================================================
// 9. VWAP TESTS
// =============================================================================

TEST(VWAP_Test, BasicComputation) {
    // Single session of 10 bars with known prices and volumes.
    // tp = (H+L+C)/3. With our make_bar, we control all fields.
    // Bar 0: tp=100, vol=100 -> VWAP = 100
    // Bar 1: tp=110, vol=200 -> VWAP = (100*100 + 110*200) / (100+200) = 32000/300 = 106.667
    std::vector<Bar> bars;
    int64_t          base_ts = 1'262'304'000LL * NS_PER_SEC;
    // Bar 0: O=100, H=100, L=100, C=100, V=100 -> tp = 100
    bars.push_back(make_bar(base_ts, 100, 100, 100, 100, 100));
    // Bar 1: O=110, H=110, L=110, C=110, V=200 -> tp = 110
    bars.push_back(make_bar(base_ts + 30 * NS_PER_SEC, 110, 110, 110, 110, 200));
    // Bar 2: O=105, H=105, L=105, C=105, V=150 -> tp = 105
    bars.push_back(make_bar(base_ts + 60 * NS_PER_SEC, 105, 105, 105, 105, 150));

    VWAP vwap(2.0f);
    EXPECT_EQ(vwap.num_series(), 3);
    EXPECT_EQ(vwap.warmup_bars(), 0);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    vwap.compute(bars.data(), 0, N, output.data());

    // Bar 0: VWAP = 100
    EXPECT_NEAR(output[0], 100.0f, 0.01f);

    // Bar 1: VWAP = (100*100 + 110*200) / 300 = 32000/300 = 106.667
    EXPECT_NEAR(output[1], 106.667f, 0.01f);

    // Bar 2: VWAP = (100*100 + 110*200 + 105*150) / 450 = 47750/450 = 106.111
    EXPECT_NEAR(output[2], 106.111f, 0.01f);

    // VWAP values should never be NaN (warmup = 0)
    for (int i = 0; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i]));
    }
}

TEST(VWAP_Test, SessionReset) {
    // Two sessions separated by > 55 min gap
    int64_t base_ts = 1'262'304'000LL * NS_PER_SEC;
    int64_t gap     = 60LL * 60 * NS_PER_SEC;  // 60 min gap

    std::vector<Bar> bars;
    // Session 1: bars at base, base+30s
    bars.push_back(make_bar(base_ts, 100, 100, 100, 100, 100));
    bars.push_back(make_bar(base_ts + 30 * NS_PER_SEC, 110, 110, 110, 110, 200));
    // Session 2: bars after 60 min gap
    bars.push_back(make_bar(base_ts + 30 * NS_PER_SEC + gap, 50, 50, 50, 50, 300));
    bars.push_back(make_bar(base_ts + 60 * NS_PER_SEC + gap, 60, 60, 60, 60, 100));

    VWAP               vwap(1.0f);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    vwap.compute(bars.data(), 0, N, output.data());

    // Session 1, bar 0: VWAP = 100
    EXPECT_NEAR(output[0], 100.0f, 0.01f);

    // Session 2 resets at bar 2: VWAP = 50 (only bar 2's data)
    EXPECT_NEAR(output[2], 50.0f, 0.01f);

    // Session 2, bar 3: VWAP = (50*300 + 60*100) / 400 = 21000/400 = 52.5
    EXPECT_NEAR(output[3], 52.5f, 0.01f);
}

// =============================================================================
// 10. OBV TESTS
// =============================================================================

TEST(OBV_Test, BasicRawOBV) {
    // Bars: close goes up, down, up, up, down
    std::vector<float> closes = {100, 102, 99, 105, 107, 103};
    auto               bars   = make_bars_from_closes(closes);

    OBV obv(0);  // raw, no smoothing
    EXPECT_EQ(obv.num_series(), 1);
    EXPECT_EQ(obv.warmup_bars(), 0);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    obv.compute(bars.data(), 0, N, output.data());

    // OBV starts at 0
    EXPECT_NEAR(output[0], 0.0f, 1e-3f);

    // Bar 1: close up (102>100) -> OBV += 100 = 100
    EXPECT_NEAR(output[1], 100.0f, 1e-3f);

    // Bar 2: close down (99<102) -> OBV -= 100 = 0
    EXPECT_NEAR(output[2], 0.0f, 1e-3f);

    // Bar 3: close up (105>99) -> OBV += 100 = 100
    EXPECT_NEAR(output[3], 100.0f, 1e-3f);

    // Bar 4: close up (107>105) -> OBV += 100 = 200
    EXPECT_NEAR(output[4], 200.0f, 1e-3f);

    // Bar 5: close down (103<107) -> OBV -= 100 = 100
    EXPECT_NEAR(output[5], 100.0f, 1e-3f);
}

TEST(OBV_Test, SmoothedOBV) {
    std::vector<float> closes = {100, 102, 99, 105, 107, 103, 108, 106, 110, 112};
    auto               bars   = make_bars_from_closes(closes);

    OBV obv(3);  // SMA-3 smoothing
    EXPECT_EQ(obv.warmup_bars(), 2);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    obv.compute(bars.data(), 0, N, output.data());

    // First 2 bars should be NaN (warmup = 2)
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "NaN at i=" << i;
    }

    // After warmup: should be valid
    for (int i = 2; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Valid at i=" << i;
    }
}

TEST(OBV_Test, WarmupBoundaryPeriod1) {
    // SMA-1 smoothing: warmup_bars should be 0 (first bar valid)
    std::vector<float> closes = {100, 102, 99, 105, 107};
    auto               bars   = make_bars_from_closes(closes);

    OBV obv(1);
    EXPECT_EQ(obv.warmup_bars(), 0);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    obv.compute(bars.data(), 0, N, output.data());

    // First bar should be valid (no warmup needed for SMA-1)
    for (int i = 0; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Valid at i=" << i;
    }
}

TEST(OBV_Test, WarmupBoundaryPeriod5) {
    // SMA-5 smoothing: warmup_bars should be 4
    std::vector<float> closes = {100, 102, 99, 105, 107, 103, 108, 106, 110, 112};
    auto               bars   = make_bars_from_closes(closes);

    OBV obv(5);
    EXPECT_EQ(obv.warmup_bars(), 4);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    obv.compute(bars.data(), 0, N, output.data());

    // First 4 bars should be NaN (warmup = 4)
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "NaN at i=" << i;
    }

    // Bar 4 onward should be valid
    for (int i = 4; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Valid at i=" << i;
    }
}

TEST(OBV_Test, RawOBVWarmup) {
    // Raw OBV (smoothing_period=0): warmup_bars should be 0
    std::vector<float> closes = {100, 102, 99, 105, 107};
    auto               bars   = make_bars_from_closes(closes);

    OBV obv(0);
    EXPECT_EQ(obv.warmup_bars(), 0);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    obv.compute(bars.data(), 0, N, output.data());

    // Bar 0 should be valid (no warmup for raw OBV)
    for (int i = 0; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Valid at i=" << i;
    }
}

// =============================================================================
// 11. DONCHIAN CHANNEL TESTS
// =============================================================================

TEST(Donchian_Test, BasicComputation) {
    // Known OHLC data to verify highest high / lowest low
    // Period = 3
    std::vector<float> closes = {100, 102, 98, 105, 97, 103, 101, 106};
    auto               bars   = make_bars_from_closes(closes);

    DonchianChannel dc(3);
    EXPECT_EQ(dc.num_series(), 2);
    EXPECT_EQ(dc.warmup_bars(), 2);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 2);
    dc.compute(bars.data(), 0, N, output.data());

    // Warmup check
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "Upper NaN at i=" << i;
        EXPECT_TRUE(is_nan(output[N + i])) << "Lower NaN at i=" << i;
    }

    // After warmup: upper >= lower always
    for (int i = 2; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Upper valid at i=" << i;
        EXPECT_FALSE(is_nan(output[N + i])) << "Lower valid at i=" << i;
        EXPECT_GE(output[i], output[N + i]) << "Upper >= lower at i=" << i;
    }
}

TEST(Donchian_Test, ConstantPrices) {
    // With constant prices, upper should equal lower (approx, accounting for H/L spread)
    std::vector<float> closes(15, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    DonchianChannel    dc(5);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 2);
    dc.compute(bars.data(), 0, N, output.data());

    // After warmup, upper and lower should be equal since all H and L are identical
    // (make_bars_from_closes with constant close gives H=100.5, L=99.5)
    for (int i = dc.warmup_bars(); i < N; ++i) {
        EXPECT_NEAR(output[i], 100.5f, 0.01f) << "Upper at i=" << i;
        EXPECT_NEAR(output[N + i], 99.5f, 0.01f) << "Lower at i=" << i;
    }
}

// =============================================================================
// CROSS-INDICATOR: Determinism test
// =============================================================================

TEST(Indicators_Full, Determinism) {
    // Run every indicator twice on the same data and verify identical output
    std::vector<float> closes;
    for (int i = 0; i < 100; ++i) {
        closes.push_back(100.0f + std::sin(static_cast<float>(i) * 0.3f) * 10.0f);
    }
    auto      bars = make_bars_from_closes(closes);
    const int N    = static_cast<int>(bars.size());

    // Test a representative subset
    std::vector<std::unique_ptr<Indicator>> indicators;
    indicators.push_back(std::make_unique<MACD>(3, 5, 3));
    indicators.push_back(std::make_unique<BollingerBands>(10, 2.0f));
    indicators.push_back(std::make_unique<Stochastic>(5, 3));
    indicators.push_back(std::make_unique<CCI>(10));
    indicators.push_back(std::make_unique<ROC>(5));
    indicators.push_back(std::make_unique<HMA>(9));
    indicators.push_back(std::make_unique<ConnorsRSI>(3, 2, 50));
    indicators.push_back(std::make_unique<DonchianChannel>(10));

    for (auto& ind : indicators) {
        const int          total = N * ind->num_series();
        std::vector<float> out1(total);
        std::vector<float> out2(total);

        ind->compute(bars.data(), 0, N, out1.data());
        ind->compute(bars.data(), 0, N, out2.data());

        for (int i = 0; i < total; ++i) {
            if (is_nan(out1[i])) {
                EXPECT_TRUE(is_nan(out2[i])) << ind->name() << " determinism fail at i=" << i;
            } else {
                EXPECT_EQ(out1[i], out2[i]) << ind->name() << " determinism fail at i=" << i;
            }
        }
    }
}

// =============================================================================
// Name and series count verification
// =============================================================================

TEST(Indicators_Full, NameAndSeriesCounts) {
    MACD macd(8, 21, 9);
    EXPECT_STREQ(macd.name(), "MACD");
    EXPECT_EQ(macd.num_series(), 3);

    ADX adx(14);
    EXPECT_STREQ(adx.name(), "ADX");
    EXPECT_EQ(adx.num_series(), 3);

    BollingerBands bb(20, 2.0f);
    EXPECT_STREQ(bb.name(), "BollingerBands");
    EXPECT_EQ(bb.num_series(), 3);

    KeltnerChannels kc(MAType::EMA, 20, 10, 2.0f);
    EXPECT_STREQ(kc.name(), "KeltnerChannels");
    EXPECT_EQ(kc.num_series(), 3);

    Stochastic stoch(14, 3);
    EXPECT_STREQ(stoch.name(), "Stochastic");
    EXPECT_EQ(stoch.num_series(), 2);

    CCI cci(20);
    EXPECT_STREQ(cci.name(), "CCI");
    EXPECT_EQ(cci.num_series(), 1);

    ROC roc_raw(10);
    EXPECT_STREQ(roc_raw.name(), "ROC");
    EXPECT_EQ(roc_raw.num_series(), 1);

    ROC roc_smooth(10, 3);
    EXPECT_EQ(roc_smooth.num_series(), 2);

    VWAP vwap(2.0f);
    EXPECT_STREQ(vwap.name(), "VWAP");
    EXPECT_EQ(vwap.num_series(), 3);

    OBV obv(0);
    EXPECT_STREQ(obv.name(), "OBV");
    EXPECT_EQ(obv.num_series(), 1);

    DonchianChannel dc(20);
    EXPECT_STREQ(dc.name(), "DonchianChannel");
    EXPECT_EQ(dc.num_series(), 2);

    HMA hma(20);
    EXPECT_STREQ(hma.name(), "HMA");
    EXPECT_EQ(hma.num_series(), 1);

    ConnorsRSI crsi(3, 2, 100);
    EXPECT_STREQ(crsi.name(), "ConnorsRSI");
    EXPECT_EQ(crsi.num_series(), 1);
}

// =============================================================================
// HMA TESTS
// =============================================================================

TEST(HMA_Test, BasicComputation) {
    // Generate trending data to verify HMA produces valid output
    std::vector<float> closes;
    for (int i = 0; i < 60; ++i) {
        closes.push_back(100.0f + static_cast<float>(i) * 0.5f);  // uptrend
    }
    auto bars = make_bars_from_closes(closes);

    HMA hma(9);
    EXPECT_EQ(hma.num_series(), 1);
    // warmup = 9 + floor(sqrt(9)) = 9 + 3 = 12
    EXPECT_EQ(hma.warmup_bars(), 12);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    hma.compute(bars.data(), 0, N, output.data());

    // Warmup check
    for (int i = 0; i < 12; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "HMA NaN at i=" << i;
    }

    // After warmup: values should be valid
    for (int i = 12; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "HMA valid at i=" << i;
    }

    // In an uptrend, HMA should generally be rising
    // Check that most consecutive pairs are increasing
    int rising_count = 0;
    for (int i = 13; i < N; ++i) {
        if (output[i] > output[i - 1]) rising_count++;
    }
    EXPECT_GT(rising_count, (N - 14) / 2) << "HMA should be mostly rising in uptrend";
}

TEST(HMA_Test, WarmupPeriod) {
    std::vector<float> closes(100, 50.0f);  // flat data
    auto               bars = make_bars_from_closes(closes);

    HMA hma(21);
    // warmup = 21 + floor(sqrt(21)) = 21 + 4 = 25
    EXPECT_EQ(hma.warmup_bars(), 25);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    hma.compute(bars.data(), 0, N, output.data());

    // All values during warmup should be NaN
    for (int i = 0; i < 25; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "NaN at i=" << i;
    }
    // Values after warmup should be valid
    for (int i = 25; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Valid at i=" << i;
    }
}

TEST(HMA_Test, ConstantPrice) {
    // With constant prices, HMA should converge to that price
    std::vector<float> closes(80, 100.0f);
    auto               bars = make_bars_from_closes(closes);

    HMA                hma(14);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    hma.compute(bars.data(), 0, N, output.data());

    int wu = hma.warmup_bars();
    // After sufficient warmup, HMA of constant series should equal the constant
    for (int i = wu; i < N; ++i) {
        EXPECT_NEAR(output[i], 100.0f, 0.1f) << "HMA should be ~100 for flat prices at i=" << i;
    }
}

TEST(HMA_Test, Determinism) {
    std::vector<float> closes;
    for (int i = 0; i < 60; ++i) {
        closes.push_back(100.0f + std::sin(static_cast<float>(i) * 0.3f) * 5.0f);
    }
    auto      bars = make_bars_from_closes(closes);
    const int N    = static_cast<int>(bars.size());

    HMA                hma(9);
    std::vector<float> out1(N), out2(N);
    hma.compute(bars.data(), 0, N, out1.data());
    hma.compute(bars.data(), 0, N, out2.data());

    for (int i = 0; i < N; ++i) {
        if (is_nan(out1[i])) {
            EXPECT_TRUE(is_nan(out2[i])) << "HMA determinism at i=" << i;
        } else {
            EXPECT_EQ(out1[i], out2[i]) << "HMA determinism at i=" << i;
        }
    }
}

// =============================================================================
// CONNORS RSI TESTS
// =============================================================================

TEST(ConnorsRSI_Test, BasicComputation) {
    // Generate enough bars to get past warmup
    std::vector<float> closes;
    for (int i = 0; i < 150; ++i) {
        // Oscillating pattern to generate meaningful ConnorsRSI values
        closes.push_back(100.0f + std::sin(static_cast<float>(i) * 0.5f) * 5.0f);
    }
    auto bars = make_bars_from_closes(closes);

    ConnorsRSI crsi(3, 2, 100);
    EXPECT_EQ(crsi.num_series(), 1);
    // warmup = max(3*3, 3*2, 100) = max(9, 6, 100) = 100
    EXPECT_EQ(crsi.warmup_bars(), 100);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    crsi.compute(bars.data(), 0, N, output.data());

    // Warmup check
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "ConnorsRSI NaN at i=" << i;
    }

    // After warmup: values should be valid and in [0, 100]
    for (int i = 100; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "ConnorsRSI valid at i=" << i;
        EXPECT_GE(output[i], 0.0f) << "ConnorsRSI >= 0 at i=" << i;
        EXPECT_LE(output[i], 100.0f) << "ConnorsRSI <= 100 at i=" << i;
    }
}

TEST(ConnorsRSI_Test, WarmupPeriod) {
    std::vector<float> closes;
    for (int i = 0; i < 80; ++i) {
        closes.push_back(100.0f + static_cast<float>(i) * 0.1f);
    }
    auto bars = make_bars_from_closes(closes);

    ConnorsRSI crsi(3, 2, 50);
    // warmup = max(3*3, 3*2, 50) = max(9, 6, 50) = 50
    EXPECT_EQ(crsi.warmup_bars(), 50);

    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    crsi.compute(bars.data(), 0, N, output.data());

    // All values during warmup should be NaN
    for (int i = 0; i < 50; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "NaN at i=" << i;
    }
    // Values after warmup should be valid
    for (int i = 50; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Valid at i=" << i;
    }
}

TEST(ConnorsRSI_Test, Determinism) {
    std::vector<float> closes;
    for (int i = 0; i < 150; ++i) {
        closes.push_back(100.0f + std::sin(static_cast<float>(i) * 0.3f) * 10.0f);
    }
    auto      bars = make_bars_from_closes(closes);
    const int N    = static_cast<int>(bars.size());

    ConnorsRSI         crsi(3, 2, 50);
    std::vector<float> out1(N), out2(N);
    crsi.compute(bars.data(), 0, N, out1.data());
    crsi.compute(bars.data(), 0, N, out2.data());

    for (int i = 0; i < N; ++i) {
        if (is_nan(out1[i])) {
            EXPECT_TRUE(is_nan(out2[i])) << "ConnorsRSI determinism at i=" << i;
        } else {
            EXPECT_EQ(out1[i], out2[i]) << "ConnorsRSI determinism at i=" << i;
        }
    }
}

TEST(ConnorsRSI_Test, RangeAlwaysValid) {
    // Random-ish data, verify ConnorsRSI always in [0, 100]
    std::vector<float> closes = {
        100, 102, 98,  105, 97,  103, 99,  101, 104, 96,  106, 95,  108, 93,  110, 91, 112, 90,
        114, 88,  100, 102, 98,  105, 97,  103, 99,  101, 104, 96,  106, 95,  108, 93, 110, 91,
        112, 90,  114, 88,  100, 102, 98,  105, 97,  103, 99,  101, 104, 96,  106, 95, 108, 93,
        110, 91,  112, 90,  114, 88,  100, 105, 95,  110, 90,  115, 85,  120, 80,  125};
    auto bars = make_bars_from_closes(closes);

    ConnorsRSI         crsi(3, 2, 50);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    crsi.compute(bars.data(), 0, N, output.data());

    for (int i = crsi.warmup_bars(); i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_GE(output[i], 0.0f) << "ConnorsRSI >= 0 at i=" << i;
            EXPECT_LE(output[i], 100.0f) << "ConnorsRSI <= 100 at i=" << i;
        }
    }
}

TEST(ConnorsRSI_Test, PercentRankNoSyntheticZero) {
    // changes[0] has no prior close, so it is NaN and skipped by PercentRank.
    // A synthetic 0.0 there would pollute the PercentRank lookback window.
    //
    // Strategy: compute ConnorsRSI with a small percentile_period so the first
    // valid PercentRank window includes index 0. Verify the output is still
    // valid and in [0, 100].
    std::vector<float> closes;
    // Strong upward trend -- all changes are positive, so a synthetic zero
    // at index 0 would be the only value below every other change,
    // distorting PercentRank toward higher values.
    for (int i = 0; i < 30; ++i) {
        closes.push_back(100.0f + static_cast<float>(i) * 2.0f);
    }
    auto bars = make_bars_from_closes(closes);

    // Use percentile_period=5 so the first PercentRank window covers changes[0..4]
    ConnorsRSI         crsi(3, 2, 5);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N);
    crsi.compute(bars.data(), 0, N, output.data());

    // First valid PercentRank bar is index 5 (needs 5 prior changes).
    // A changes[0]=0 would count as "less than" every positive change,
    // inflating PercentRank. Because changes[0]=NaN is skipped, the
    // effective window is 4 bars (indices 1-4), all equal changes of 2.0.
    // Verify all post-warmup values are valid and in range.
    int warmup = crsi.warmup_bars();
    for (int i = warmup; i < N; ++i) {
        EXPECT_FALSE(is_nan(output[i])) << "Valid at i=" << i;
        EXPECT_GE(output[i], 0.0f) << "ConnorsRSI >= 0 at i=" << i;
        EXPECT_LE(output[i], 100.0f) << "ConnorsRSI <= 100 at i=" << i;
    }

    // With a constant step, all changes (indices 1..N-1) are identical (2.0).
    // PercentRank of an equal value: count_less=0 out of count_valid changes.
    // So PercentRank component = 0 for all bars.
    // Verify the first post-warmup bar has the same value as later bars
    // (if old synthetic zero were present, the first window would differ).
    if (warmup + 1 < N) {
        EXPECT_FLOAT_EQ(output[warmup], output[warmup + 1])
            << "First post-warmup bar should match next bar for constant-step data";
    }
}

// =============================================================================
// EDGE CASE TESTS: Flat price, monotonic increase, single spike
// =============================================================================

// Helper: generate N bars with all closes = flat_price.
// open = close, high = close + 0.5, low = close - 0.5.
static std::vector<Bar> make_flat_bars(int count, float flat_price = 100.0f) {
    std::vector<float> closes(count, flat_price);
    return make_bars_from_closes(closes);
}

// Helper: generate N bars with monotonically increasing closes.
static std::vector<Bar> make_trending_bars(int count, float start = 100.0f, float step = 0.5f) {
    std::vector<float> closes(count);
    for (int i = 0; i < count; ++i) {
        closes[i] = start + static_cast<float>(i) * step;
    }
    return make_bars_from_closes(closes);
}

// Helper: generate N bars with a single spike at spike_bar.
static std::vector<Bar> make_spike_bars(int count, int spike_bar, float base = 100.0f,
                                        float spike = 200.0f) {
    std::vector<float> closes(count, base);
    closes[spike_bar] = spike;
    return make_bars_from_closes(closes);
}

// ---------------------------------------------------------------------------
// EMA edge cases
// ---------------------------------------------------------------------------
TEST(Indicators_EdgeCase, EMA_FlatPrice) {
    // On flat closes, EMA should converge to the flat value
    const int N    = 200;
    auto      bars = make_flat_bars(N, 100.0f);

    EMASMACross        ema(MAType::EMA, 10, MAType::EMA, 50);
    int                wu = ema.warmup_bars();
    std::vector<float> output(N * 2);
    ema.compute(bars.data(), 0, N, output.data());

    // After warmup, fast EMA and slow EMA should both equal 100.0
    for (int i = wu; i < N; ++i) {
        EXPECT_NEAR(output[i], 100.0f, 0.01f)
            << "Fast EMA should converge to flat price at i=" << i;
        EXPECT_NEAR(output[N + i], 100.0f, 0.01f)
            << "Slow EMA should converge to flat price at i=" << i;
    }
}

TEST(Indicators_EdgeCase, EMA_MonotonicIncrease) {
    // On monotonically increasing closes, fast EMA > slow EMA (bullish cross)
    const int N    = 200;
    auto      bars = make_trending_bars(N, 100.0f, 0.5f);

    EMASMACross        ema(MAType::EMA, 10, MAType::EMA, 50);
    int                wu = ema.warmup_bars();
    std::vector<float> output(N * 2);
    ema.compute(bars.data(), 0, N, output.data());

    // Well past warmup, fast EMA should be above slow EMA in an uptrend
    for (int i = wu + 20; i < N; ++i) {
        EXPECT_GT(output[i], output[N + i])
            << "Fast EMA should be above slow EMA in uptrend at i=" << i;
    }
}

TEST(Indicators_EdgeCase, EMA_SingleSpike) {
    // A single spike should cause EMA to rise then decay back toward base
    const int N    = 200;
    auto      bars = make_spike_bars(N, 50, 100.0f, 200.0f);

    EMASMACross        ema(MAType::EMA, 10, MAType::EMA, 50);
    int                wu = ema.warmup_bars();
    std::vector<float> output(N * 2);
    ema.compute(bars.data(), 0, N, output.data());

    // Well after the spike (bar 50), EMA should decay back toward 100
    // Check at end of series: fast EMA (shorter period) decays faster
    if (N - 1 >= wu) {
        EXPECT_NEAR(output[N - 1], 100.0f, 5.0f)
            << "Fast EMA should mostly decay back to base after spike";
        // Slow EMA retains more memory but should also be close
        EXPECT_NEAR(output[2 * N - 1], 100.0f, 10.0f)
            << "Slow EMA should mostly decay back to base after spike";
    }
}

// ---------------------------------------------------------------------------
// RSI edge cases
// ---------------------------------------------------------------------------
TEST(Indicators_EdgeCase, RSI_FlatPrice) {
    // On flat closes (zero change), RSI should be ~50 (no net gain or loss)
    const int N    = 200;
    auto      bars = make_flat_bars(N, 100.0f);

    RSI                rsi(14);
    int                wu = rsi.warmup_bars();
    std::vector<float> output(N);
    rsi.compute(bars.data(), 0, N, output.data());

    // After warmup, RSI should be 50 (or NaN if implementation handles zero-change
    // as a special case; accept either)
    for (int i = wu; i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_NEAR(output[i], 50.0f, 1.0f) << "RSI on flat price should be ~50 at i=" << i;
        }
    }
}

TEST(Indicators_EdgeCase, RSI_MonotonicIncrease) {
    // On monotonically increasing closes, RSI should be near 100
    const int N    = 200;
    auto      bars = make_trending_bars(N, 100.0f, 0.5f);

    RSI                rsi(14);
    int                wu = rsi.warmup_bars();
    std::vector<float> output(N);
    rsi.compute(bars.data(), 0, N, output.data());

    for (int i = wu; i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_GT(output[i], 85.0f)
                << "RSI should be very high in sustained uptrend at i=" << i;
        }
    }
}

TEST(Indicators_EdgeCase, RSI_SingleSpike) {
    // After a single spike up and back down, RSI should eventually return to ~50
    const int N    = 200;
    auto      bars = make_spike_bars(N, 50, 100.0f, 200.0f);

    RSI                rsi(14);
    std::vector<float> output(N);
    rsi.compute(bars.data(), 0, N, output.data());

    // Well after the spike, RSI should settle near 50 as gains/losses average out
    for (int i = N - 20; i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_NEAR(output[i], 50.0f, 15.0f)
                << "RSI should return toward 50 well after spike at i=" << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Bollinger Bands edge cases
// ---------------------------------------------------------------------------
TEST(Indicators_EdgeCase, Bollinger_FlatPrice) {
    // On flat closes, stddev = 0 so upper == lower == middle == price
    const int N    = 200;
    auto      bars = make_flat_bars(N, 100.0f);

    BollingerBands     bb(20, 2.0f);
    int                wu = bb.warmup_bars();
    std::vector<float> output(N * 3);
    bb.compute(bars.data(), 0, N, output.data());

    for (int i = wu; i < N; ++i) {
        EXPECT_NEAR(output[i], 100.0f, 0.01f) << "BB middle should equal flat price at i=" << i;
        EXPECT_NEAR(output[N + i], 100.0f, 0.01f)
            << "BB upper should converge to middle (zero stddev) at i=" << i;
        EXPECT_NEAR(output[2 * N + i], 100.0f, 0.01f)
            << "BB lower should converge to middle (zero stddev) at i=" << i;
    }
}

TEST(Indicators_EdgeCase, Bollinger_MonotonicIncrease) {
    // On trending data, bands should widen and upper > middle > lower
    const int N    = 200;
    auto      bars = make_trending_bars(N, 100.0f, 0.5f);

    BollingerBands     bb(20, 2.0f);
    int                wu = bb.warmup_bars();
    std::vector<float> output(N * 3);
    bb.compute(bars.data(), 0, N, output.data());

    for (int i = wu + 5; i < N; ++i) {
        EXPECT_GT(output[N + i], output[i]) << "BB upper > middle at i=" << i;
        EXPECT_GT(output[i], output[2 * N + i]) << "BB middle > lower at i=" << i;
    }
}

TEST(Indicators_EdgeCase, Bollinger_SingleSpike) {
    // After a spike, bands should widen during the spike window then narrow
    const int N    = 200;
    auto      bars = make_spike_bars(N, 50, 100.0f, 200.0f);

    BollingerBands     bb(20, 2.0f);
    int                wu = bb.warmup_bars();
    std::vector<float> output(N * 3);
    bb.compute(bars.data(), 0, N, output.data());

    // While spike is in the 20-bar window (bars 50..69), bandwidth should be large
    if (60 >= wu) {
        float bandwidth_during = output[N + 60] - output[2 * N + 60];
        EXPECT_GT(bandwidth_during, 10.0f) << "BB bandwidth should be wide during spike window";
    }

    // Well after spike exits the window (bar 71+), bandwidth should be near zero
    if (N - 1 >= wu) {
        float bandwidth_after = output[N + (N - 1)] - output[2 * N + (N - 1)];
        EXPECT_NEAR(bandwidth_after, 0.0f, 1.0f)
            << "BB bandwidth should narrow after spike exits the lookback window";
    }
}

// ---------------------------------------------------------------------------
// Stochastic edge cases
// ---------------------------------------------------------------------------
TEST(Indicators_EdgeCase, Stochastic_FlatPrice) {
    // On flat closes, %K depends on H-L range.
    // make_bars_from_closes creates high = max(O,C)+0.5, low = min(O,C)-0.5.
    // For flat data: open=close=100, high=100.5, low=99.5.
    // %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    //    = (100 - 99.5) / (100.5 - 99.5) * 100 = 50
    const int N    = 100;
    auto      bars = make_flat_bars(N, 100.0f);

    Stochastic         stoch(14, 3);
    int                wu = stoch.warmup_bars();
    std::vector<float> output(N * 2);
    stoch.compute(bars.data(), 0, N, output.data());

    for (int i = wu; i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_NEAR(output[i], 50.0f, 1.0f) << "%K on flat price should be ~50 at i=" << i;
        }
    }
}

TEST(Indicators_EdgeCase, Stochastic_MonotonicIncrease) {
    // Monotonically increasing: close is always near highest high
    // %K should be near 100
    const int N    = 100;
    auto      bars = make_trending_bars(N, 100.0f, 1.0f);

    Stochastic         stoch(14, 3);
    int                wu = stoch.warmup_bars();
    std::vector<float> output(N * 2);
    stoch.compute(bars.data(), 0, N, output.data());

    for (int i = wu + 5; i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_GT(output[i], 80.0f) << "%K should be high in sustained uptrend at i=" << i;
        }
    }
}

TEST(Indicators_EdgeCase, Stochastic_SingleSpike) {
    // After a spike exits the lookback window, %K should return to ~50
    const int N    = 100;
    auto      bars = make_spike_bars(N, 30, 100.0f, 200.0f);

    Stochastic         stoch(14, 3);
    std::vector<float> output(N * 2);
    stoch.compute(bars.data(), 0, N, output.data());

    // After spike exits the 14-bar window (bar 30+14=44), %K should settle
    for (int i = 50; i < N; ++i) {
        if (!is_nan(output[i])) {
            EXPECT_GE(output[i], 0.0f) << "%K in range at i=" << i;
            EXPECT_LE(output[i], 100.0f) << "%K in range at i=" << i;
            // Should be near 50 on flat data after spike exits window
            EXPECT_NEAR(output[i], 50.0f, 5.0f)
                << "%K should settle near 50 after spike exits window at i=" << i;
        }
    }
}

// =============================================================================
// ADDITIONAL COVERAGE TESTS
// =============================================================================

// ---------------------------------------------------------------------------
// ZeroVolumeVWAP: VWAP should handle bars with volume=0 gracefully.
// It must not produce NaN on zero-volume bars when preceding bars had
// non-zero volume (the cumulative VWAP just holds steady).
// ---------------------------------------------------------------------------
TEST(Indicators, ZeroVolumeVWAP) {
    int64_t          base_ts = 1'262'304'000LL * NS_PER_SEC;
    std::vector<Bar> bars;

    // Bar 0: normal volume
    bars.push_back(make_bar(base_ts, 100, 100, 100, 100, 500));
    // Bar 1: normal volume
    bars.push_back(make_bar(base_ts + 30 * NS_PER_SEC, 105, 105, 105, 105, 300));
    // Bars 2-5: zero volume
    for (int i = 2; i <= 5; ++i) {
        bars.push_back(
            make_bar(base_ts + static_cast<int64_t>(i) * 30 * NS_PER_SEC, 110, 110, 110, 110, 0));
    }
    // Bar 6: resume normal volume
    bars.push_back(make_bar(base_ts + 6 * 30 * NS_PER_SEC, 108, 108, 108, 108, 200));

    VWAP               vwap(2.0f);
    const int          N = static_cast<int>(bars.size());
    std::vector<float> output(N * 3);
    vwap.compute(bars.data(), 0, N, output.data());

    // VWAP at bar 0 should be exactly 100 (only bar with volume so far)
    EXPECT_NEAR(output[0], 100.0f, 0.01f);

    // VWAP at bar 1 should incorporate bar 1: (100*500 + 105*300) / 800 = 81500/800 = 101.875
    EXPECT_NEAR(output[1], 101.875f, 0.01f);

    // During zero-volume bars, VWAP should not become NaN.
    // It should either hold at the previous value or gracefully handle zero volume.
    for (int i = 2; i <= 5; ++i) {
        // VWAP should be valid (not NaN) even with zero volume
        bool vwap_valid = !is_nan(output[i]);
        EXPECT_TRUE(vwap_valid) << "VWAP must not produce NaN on zero-volume bar " << i;
    }

    // After resuming volume, VWAP should again be valid
    EXPECT_FALSE(is_nan(output[6])) << "VWAP should be valid after volume resumes at bar 6";
}

// ---------------------------------------------------------------------------
// FlatPriceAllIndicators: Constant OHLC = 5000, volume = 1000 for 100 bars.
// Compute each of the 16 indicators. Verify none produces NaN after warmup.
// This catches division-by-zero and degenerate-case bugs across the full
// indicator catalog.
// ---------------------------------------------------------------------------
TEST(Indicators, FlatPriceAllIndicators) {
    const int      N = 100;
    const float    P = 5000.0f;
    const uint32_t V = 1000;

    int64_t          base_ts = 1'262'304'000LL * NS_PER_SEC;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        bars[i] = make_bar(base_ts + static_cast<int64_t>(i) * 30 * NS_PER_SEC, P, P, P, P, V);
    }

    // Lambda to check: after warmup, no NaN in any series
    auto check_no_nan = [&](Indicator& ind, const char* label) {
        int                wu = ind.warmup_bars();
        int                ns = ind.num_series();
        std::vector<float> out(static_cast<size_t>(N) * ns);
        ind.compute(bars.data(), 0, N, out.data());

        for (int s = 0; s < ns; ++s) {
            for (int i = wu; i < N; ++i) {
                EXPECT_FALSE(is_nan(out[s * N + i]))
                    << label << " series " << s << " produced NaN at bar " << i << " (warmup=" << wu
                    << ")";
            }
        }
    };

    // 1. EMASMACross (EMA/EMA)
    EMASMACross ema_cross(MAType::EMA, 10, MAType::EMA, 20);
    check_no_nan(ema_cross, "EMASMACross(EMA10/EMA20)");

    // 2. EMASMACross (SMA/SMA)
    EMASMACross sma_cross(MAType::SMA, 10, MAType::SMA, 20);
    check_no_nan(sma_cross, "EMASMACross(SMA10/SMA20)");

    // 3. RSI
    RSI rsi(14);
    check_no_nan(rsi, "RSI(14)");

    // 4. ATR
    ATR atr(14);
    check_no_nan(atr, "ATR(14)");

    // 5. MACD
    MACD macd(12, 26, 9);
    check_no_nan(macd, "MACD(12,26,9)");

    // 6. ADX
    ADX adx(14);
    check_no_nan(adx, "ADX(14)");

    // 7. Bollinger Bands
    BollingerBands bb(20, 2.0f);
    check_no_nan(bb, "Bollinger(20,2)");

    // 8. Keltner Channels
    KeltnerChannels kc(MAType::EMA, 20, 10, 1.5f);
    check_no_nan(kc, "Keltner(EMA,20,10,1.5)");

    // 9. Stochastic
    Stochastic stoch(14, 3);
    check_no_nan(stoch, "Stochastic(14,3)");

    // 10. CCI
    CCI cci(20);
    check_no_nan(cci, "CCI(20)");

    // 11. ROC
    ROC roc(10, 0);
    check_no_nan(roc, "ROC(10)");

    // 12. OBV
    OBV obv(0);
    check_no_nan(obv, "OBV(0)");

    // 13. Donchian Channel
    DonchianChannel dc(20);
    check_no_nan(dc, "Donchian(20)");

    // 14. VWAP
    VWAP vwap(2.0f);
    check_no_nan(vwap, "VWAP(2.0)");

    // 15. HMA
    HMA hma(9);
    check_no_nan(hma, "HMA(9)");

    // 16. ConnorsRSI
    ConnorsRSI crsi(3, 2, 10);
    check_no_nan(crsi, "ConnorsRSI(3,2,10)");
}

// =============================================================================
// PIVOT POINTS TESTS
// =============================================================================

TEST(PivotPoints_Test, BasicComputation) {
    // Create synthetic multi-session data with a >55 minute gap between sessions.
    // Session 1: 5 bars with known OHLC, then a gap, then session 2: 5 bars.
    // After session 1 completes, pivot levels should be computed from session 1's
    // aggregate H, L, C and applied to all bars in session 2.

    const int BARS_PER_SESSION = 5;
    const int N                = BARS_PER_SESSION * 2;  // 2 sessions, 5 bars each

    // Session 1: bars at 30s intervals starting at some base timestamp
    // Session 2: starts after a 60-minute gap (>55 min threshold)
    int64_t base_ts = 1'262'304'000LL * NS_PER_SEC;  // 2010-01-01 00:00 UTC
    int64_t gap_ns  = 60LL * 60 * NS_PER_SEC;        // 60 minutes gap

    std::vector<Bar> bars(N);

    // Session 1: prices range H=110, L=90, Close of last bar = 105
    // Bar 0: O=100 H=102 L=98  C=101
    // Bar 1: O=101 H=106 L=100 C=104
    // Bar 2: O=104 H=110 L=103 C=108  <-- session high = 110
    // Bar 3: O=108 H=109 L=90  C=95   <-- session low = 90
    // Bar 4: O=95  H=107 L=94  C=105  <-- session close = 105
    bars[0] = make_bar(base_ts + 0 * 30 * NS_PER_SEC, 100.0f, 102.0f, 98.0f, 101.0f);
    bars[1] = make_bar(base_ts + 1 * 30 * NS_PER_SEC, 101.0f, 106.0f, 100.0f, 104.0f);
    bars[2] = make_bar(base_ts + 2 * 30 * NS_PER_SEC, 104.0f, 110.0f, 103.0f, 108.0f);
    bars[3] = make_bar(base_ts + 3 * 30 * NS_PER_SEC, 108.0f, 109.0f, 90.0f, 95.0f);
    bars[4] = make_bar(base_ts + 4 * 30 * NS_PER_SEC, 95.0f, 107.0f, 94.0f, 105.0f);

    // Session 2: starts after the gap; flat bars at 200 (values don't matter for pivot check)
    int64_t session2_start = base_ts + 4 * 30 * NS_PER_SEC + gap_ns;
    for (int i = 0; i < BARS_PER_SESSION; ++i) {
        bars[BARS_PER_SESSION + i] =
            make_bar(session2_start + i * 30 * NS_PER_SEC, 200.0f, 201.0f, 199.0f, 200.0f);
    }

    // Compute Classic pivot points
    PivotPoints pivot(PivotType::CLASSIC);
    EXPECT_EQ(pivot.num_series(), 7);

    std::vector<float> output(N * 7);
    pivot.compute(bars.data(), 0, N, output.data());

    // Series layout: PP[0..N), S1[N..2N), S2[2N..3N), S3[3N..4N),
    //                R1[4N..5N), R2[5N..6N), R3[6N..7N)
    float* pp_out = output.data();
    float* s1_out = output.data() + N;
    float* r1_out = output.data() + 4 * N;

    // Session 1 H=110, L=90, C=105
    // Classic PP = (110 + 90 + 105) / 3 = 305 / 3 = 101.6667
    // Classic S1 = 2*PP - H = 2*101.6667 - 110 = 93.3333
    // Classic R1 = 2*PP - L = 2*101.6667 - 90  = 113.3333
    const float expected_pp = (110.0f + 90.0f + 105.0f) / 3.0f;
    const float expected_s1 = 2.0f * expected_pp - 110.0f;
    const float expected_r1 = 2.0f * expected_pp - 90.0f;

    // Session 2 bars (indices 5-9) should have valid pivot levels
    for (int i = BARS_PER_SESSION; i < N; ++i) {
        EXPECT_FALSE(is_nan(pp_out[i])) << "PP should be valid in session 2 at bar " << i;
        EXPECT_NEAR(pp_out[i], expected_pp, 0.01f) << "PP mismatch at bar " << i;
        EXPECT_NEAR(s1_out[i], expected_s1, 0.01f) << "S1 mismatch at bar " << i;
        EXPECT_NEAR(r1_out[i], expected_r1, 0.01f) << "R1 mismatch at bar " << i;
    }

    // Verify ordering: S1 < PP < R1
    for (int i = BARS_PER_SESSION; i < N; ++i) {
        EXPECT_LT(s1_out[i], pp_out[i]) << "S1 should be below PP";
        EXPECT_LT(pp_out[i], r1_out[i]) << "PP should be below R1";
    }
}

TEST(PivotPoints_Test, WarmupBehavior) {
    // The first session has no prior session data, so all pivot levels
    // should be NaN throughout the first session.

    const int BARS_PER_SESSION = 10;
    int64_t   base_ts          = 1'262'304'000LL * NS_PER_SEC;

    // Single session: 10 bars at 30s intervals, no gap -> all in one session
    std::vector<Bar> bars(BARS_PER_SESSION);
    for (int i = 0; i < BARS_PER_SESSION; ++i) {
        float price = 100.0f + static_cast<float>(i);
        bars[i] = make_bar(base_ts + i * 30 * NS_PER_SEC, price, price + 1.0f, price - 1.0f, price);
    }

    PivotPoints        pivot(PivotType::CLASSIC);
    std::vector<float> output(BARS_PER_SESSION * 7);
    pivot.compute(bars.data(), 0, BARS_PER_SESSION, output.data());

    // All 7 series should be NaN for every bar in the first session
    // (no prior session to compute levels from)
    for (int s = 0; s < 7; ++s) {
        for (int i = 0; i < BARS_PER_SESSION; ++i) {
            EXPECT_TRUE(is_nan(output[s * BARS_PER_SESSION + i]))
                << "Series " << s << " at bar " << i << " should be NaN (no prior session)";
        }
    }
}
