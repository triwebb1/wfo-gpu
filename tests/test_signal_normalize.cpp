// =============================================================================
// Signal Normalization Tests
//
// Validates that normalize_to_signal() converts raw indicator output to the
// [-1, +1] signal range correctly for all 15 indicator types.
//
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "indicators/signal_normalize.h"
#include "market_data.h"

// =============================================================================
// Helpers
// =============================================================================

static constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

static Bar make_bar(float close, float high = 0.0f, float low = 0.0f) {
    Bar b{};
    b.close     = close;
    b.high      = (high == 0.0f) ? close + 1.0f : high;
    b.low       = (low == 0.0f) ? close - 1.0f : low;
    b.open      = close;
    b._pad      = 0;
    b.volume    = 100;
    b.timestamp = 0;
    return b;
}

static void expect_signal_range(const float* signal, int count) {
    for (int i = 0; i < count; ++i) {
        if (std::isnan(signal[i])) continue;
        EXPECT_GE(signal[i], -1.0f) << "signal[" << i << "] below -1";
        EXPECT_LE(signal[i], 1.0f) << "signal[" << i << "] above +1";
    }
}

// =============================================================================
// RSI normalization
// =============================================================================

TEST(SignalNormalize, RSI_Overbought) {
    IndicatorParams params;
    params.type           = IndicatorType::RSI;
    params.rsi_overbought = 70.0f;
    params.rsi_oversold   = 30.0f;

    // RSI at 85 (overbought) -> sell signal (negative)
    float        raw[]    = {85.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_LT(out[0], 0.0f);
    EXPECT_GE(out[0], -1.0f);
    // (85 - 70) / (100 - 70) = 0.5 -> signal = -0.5
    EXPECT_NEAR(out[0], -0.5f, 0.01f);
}

TEST(SignalNormalize, RSI_Oversold) {
    IndicatorParams params;
    params.type           = IndicatorType::RSI;
    params.rsi_overbought = 70.0f;
    params.rsi_oversold   = 30.0f;

    // RSI at 15 (oversold) -> buy signal (positive)
    float        raw[]    = {15.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_GT(out[0], 0.0f);
    EXPECT_LE(out[0], 1.0f);
    // (30 - 15) / 30 = 0.5
    EXPECT_NEAR(out[0], 0.5f, 0.01f);
}

TEST(SignalNormalize, RSI_Neutral) {
    IndicatorParams params;
    params.type           = IndicatorType::RSI;
    params.rsi_overbought = 70.0f;
    params.rsi_oversold   = 30.0f;

    // RSI at 50 (neutral zone) -> 0
    float        raw[]    = {50.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(SignalNormalize, RSI_NaN_Preserved) {
    IndicatorParams params;
    params.type           = IndicatorType::RSI;
    params.rsi_overbought = 70.0f;
    params.rsi_oversold   = 30.0f;

    float        raw[]    = {NaN};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_TRUE(std::isnan(out[0]));
}

// =============================================================================
// EMA/SMA Cross normalization
// =============================================================================

TEST(SignalNormalize, EMA_Cross_BullishCross) {
    IndicatorParams params;
    params.type = IndicatorType::EMA_SMA_CROSS;

    // fast > slow -> bullish
    float        fast[]   = {100.5f};
    float        slow[]   = {100.0f};
    const float* series[] = {fast, slow};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 2, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

TEST(SignalNormalize, EMA_Cross_BearishCross) {
    IndicatorParams params;
    params.type = IndicatorType::EMA_SMA_CROSS;

    // fast < slow -> bearish
    float        fast[]   = {99.5f};
    float        slow[]   = {100.0f};
    const float* series[] = {fast, slow};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 2, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], -1.0f);
}

// =============================================================================
// MACD normalization
// =============================================================================

TEST(SignalNormalize, MACD_Histogram_Positive) {
    IndicatorParams params;
    params.type = IndicatorType::MACD;

    // Histogram > 0 -> bullish
    float        macd_line[]   = {2.0f};
    float        signal_line[] = {1.0f};
    float        histogram[]   = {1.0f};
    const float* series[]      = {macd_line, signal_line, histogram};
    Bar          bars[]        = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

TEST(SignalNormalize, MACD_Histogram_Negative) {
    IndicatorParams params;
    params.type = IndicatorType::MACD;

    float        macd_line[]   = {-1.0f};
    float        signal_line[] = {0.5f};
    float        histogram[]   = {-1.5f};
    const float* series[]      = {macd_line, signal_line, histogram};
    Bar          bars[]        = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], -1.0f);
}

// =============================================================================
// ADX normalization
// =============================================================================

TEST(SignalNormalize, ADX_StrongTrend_BullishDI) {
    IndicatorParams params;
    params.type          = IndicatorType::ADX;
    params.adx_threshold = 25.0f;

    // ADX > 25, +DI > -DI -> bullish
    float        adx_val[]  = {35.0f};
    float        plus_di[]  = {30.0f};
    float        minus_di[] = {15.0f};
    const float* series[]   = {adx_val, plus_di, minus_di};
    Bar          bars[]     = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

TEST(SignalNormalize, ADX_WeakTrend_Neutral) {
    IndicatorParams params;
    params.type          = IndicatorType::ADX;
    params.adx_threshold = 25.0f;

    // ADX <= 25 -> neutral regardless of DI
    float        adx_val[]  = {20.0f};
    float        plus_di[]  = {30.0f};
    float        minus_di[] = {10.0f};
    const float* series[]   = {adx_val, plus_di, minus_di};
    Bar          bars[]     = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(SignalNormalize, ADX_CustomThreshold) {
    // Verify normalization uses params.adx_threshold, not hardcoded 25
    IndicatorParams params;
    params.type          = IndicatorType::ADX;
    params.adx_threshold = 30.0f;

    // ADX = 28: above default 25, but below custom threshold 30 -> should be neutral
    float        adx_val[]  = {28.0f};
    float        plus_di[]  = {30.0f};
    float        minus_di[] = {10.0f};
    const float* series[]   = {adx_val, plus_di, minus_di};
    Bar          bars[]     = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f) << "With adx_threshold=30, ADX=28 should be neutral";

    // ADX = 35: above custom threshold 30 -> should be bullish (since +DI > -DI)
    float        adx_val2[] = {35.0f};
    const float* series2[]  = {adx_val2, plus_di, minus_di};

    normalize_to_signal(params, series2, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f) << "With adx_threshold=30, ADX=35 should be bullish";
}

// =============================================================================
// Bollinger Bands normalization
// =============================================================================

TEST(SignalNormalize, Bollinger_BelowLower_Buy) {
    IndicatorParams params;
    params.type = IndicatorType::BOLLINGER;

    float        middle[] = {100.0f};
    float        upper[]  = {104.0f};
    float        lower[]  = {96.0f};
    const float* series[] = {middle, upper, lower};
    Bar          bars[]   = {make_bar(95.0f)};  // price below lower band
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

TEST(SignalNormalize, Bollinger_AboveUpper_Sell) {
    IndicatorParams params;
    params.type = IndicatorType::BOLLINGER;

    float        middle[] = {100.0f};
    float        upper[]  = {104.0f};
    float        lower[]  = {96.0f};
    const float* series[] = {middle, upper, lower};
    Bar          bars[]   = {make_bar(105.0f)};  // price above upper band
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], -1.0f);
}

TEST(SignalNormalize, Bollinger_InsideBands_Neutral) {
    IndicatorParams params;
    params.type = IndicatorType::BOLLINGER;

    float        middle[] = {100.0f};
    float        upper[]  = {104.0f};
    float        lower[]  = {96.0f};
    const float* series[] = {middle, upper, lower};
    Bar          bars[]   = {make_bar(100.0f)};  // price inside bands
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
}

// =============================================================================
// Stochastic normalization
// =============================================================================

TEST(SignalNormalize, Stochastic_Oversold) {
    IndicatorParams params;
    params.type             = IndicatorType::STOCHASTIC;
    params.stoch_overbought = 80.0f;
    params.stoch_oversold   = 20.0f;

    // %K = 10 (< 20) -> buy signal
    float        pct_k[]  = {10.0f};
    float        pct_d[]  = {15.0f};
    const float* series[] = {pct_k, pct_d};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 2, bars, 0, 1, out);
    EXPECT_GT(out[0], 0.0f);
    EXPECT_NEAR(out[0], 0.5f, 0.01f);  // (20-10)/20 = 0.5
}

TEST(SignalNormalize, Stochastic_Overbought) {
    IndicatorParams params;
    params.type             = IndicatorType::STOCHASTIC;
    params.stoch_overbought = 80.0f;
    params.stoch_oversold   = 20.0f;

    // %K = 90 (> 80) -> sell signal
    float        pct_k[]  = {90.0f};
    float        pct_d[]  = {85.0f};
    const float* series[] = {pct_k, pct_d};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 2, bars, 0, 1, out);
    EXPECT_LT(out[0], 0.0f);
    EXPECT_NEAR(out[0], -0.5f, 0.01f);  // -(90-80)/20 = -0.5
}

TEST(SignalNormalize, Stochastic_CustomThresholds) {
    // Verify normalization uses params, not hardcoded 80/20
    IndicatorParams params;
    params.type             = IndicatorType::STOCHASTIC;
    params.stoch_overbought = 75.0f;
    params.stoch_oversold   = 25.0f;

    // %K = 22 is between default OS(20) and custom OS(25) -> should be oversold with custom
    float        pct_k_os[]  = {22.0f};
    float        pct_d_os[]  = {20.0f};
    const float* series_os[] = {pct_k_os, pct_d_os};
    Bar          bars[]      = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series_os, 2, bars, 0, 1, out);
    EXPECT_GT(out[0], 0.0f) << "With stoch_oversold=25, K=22 should be buy signal";
    // (25-22)/25 = 0.12
    EXPECT_NEAR(out[0], 0.12f, 0.01f);

    // %K = 78 is between custom OB(75) and default OB(80) -> should be overbought with custom
    float        pct_k_ob[]  = {78.0f};
    float        pct_d_ob[]  = {76.0f};
    const float* series_ob[] = {pct_k_ob, pct_d_ob};

    normalize_to_signal(params, series_ob, 2, bars, 0, 1, out);
    EXPECT_LT(out[0], 0.0f) << "With stoch_overbought=75, K=78 should be sell signal";
    // -(78-75)/(100-75) = -3/25 = -0.12
    EXPECT_NEAR(out[0], -0.12f, 0.01f);
}

// =============================================================================
// CCI normalization
// =============================================================================

TEST(SignalNormalize, CCI_Oversold_Buy) {
    IndicatorParams params;
    params.type          = IndicatorType::CCI;
    params.cci_threshold = 100.0f;

    // CCI = -150 -> buy signal (mean-reversion)
    float        raw[]    = {-150.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_GT(out[0], 0.0f);
    EXPECT_NEAR(out[0], 0.5f, 0.01f);  // (-100 - (-150))/100 = 0.5
}

TEST(SignalNormalize, CCI_Overbought_Sell) {
    IndicatorParams params;
    params.type          = IndicatorType::CCI;
    params.cci_threshold = 100.0f;

    // CCI = +200 -> sell signal (mean-reversion)
    float        raw[]    = {200.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], -1.0f);  // -(200-100)/100 = -1.0
}

TEST(SignalNormalize, CCI_CustomThreshold) {
    // Verify normalization uses params.cci_threshold, not hardcoded 100
    IndicatorParams params;
    params.type          = IndicatorType::CCI;
    params.cci_threshold = 150.0f;

    // CCI = +120: above default 100, but below custom 150 -> should be neutral
    float        raw_neutral[] = {120.0f};
    const float* series_n[]    = {raw_neutral};
    Bar          bars[]        = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series_n, 1, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f) << "With cci_threshold=150, CCI=120 should be neutral";

    // CCI = -200: below -150 threshold -> buy signal
    // (-150 - (-200))/150 = 50/150 = 0.333
    float        raw_buy[]  = {-200.0f};
    const float* series_b[] = {raw_buy};

    normalize_to_signal(params, series_b, 1, bars, 0, 1, out);
    EXPECT_GT(out[0], 0.0f) << "With cci_threshold=150, CCI=-200 should be buy signal";
    EXPECT_NEAR(out[0], 0.333f, 0.01f);

    // CCI = +300: above +150 threshold -> sell signal (clamped)
    // -(300-150)/150 = -1.0
    float        raw_sell[] = {300.0f};
    const float* series_s[] = {raw_sell};

    normalize_to_signal(params, series_s, 1, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], -1.0f) << "With cci_threshold=150, CCI=300 should be max sell";
}

// =============================================================================
// ROC normalization
// =============================================================================

TEST(SignalNormalize, ROC_Positive) {
    IndicatorParams params;
    params.type       = IndicatorType::ROC;
    params.roc_period = 5;  // reference period: SCALE = 0.5 * sqrt(5/5) = 0.5

    // ROC = 0.25% -> signal = 0.25 / 0.5 = 0.5
    float        raw[]    = {0.25f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_NEAR(out[0], 0.5f, 0.01f);
}

TEST(SignalNormalize, ROC_Clamped) {
    IndicatorParams params;
    params.type       = IndicatorType::ROC;
    params.roc_period = 5;  // SCALE = 0.5

    // ROC = 1.0% -> 1.0 / 0.5 = 2.0 -> clamp to 1.0
    float        raw[]    = {1.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

TEST(SignalNormalize, ROC_ScalesWithPeriod) {
    // Longer ROC lookback produces larger SCALE, preventing premature saturation.
    // period=5:  SCALE = 0.5 * sqrt(5/5) = 0.5
    // period=20: SCALE = 0.5 * sqrt(20/5) = 0.5 * 2.0 = 1.0
    Bar          bars[]   = {make_bar(100.0f)};
    float        raw[]    = {0.5f};  // 0.5% ROC
    const float* series[] = {raw};
    float        out[1];

    // With period=5, signal = 0.5/0.5 = 1.0 (saturated)
    IndicatorParams p5;
    p5.type       = IndicatorType::ROC;
    p5.roc_period = 5;
    normalize_to_signal(p5, series, 1, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);

    // With period=20, signal = 0.5/1.0 = 0.5 (proportional)
    IndicatorParams p20;
    p20.type       = IndicatorType::ROC;
    p20.roc_period = 20;
    normalize_to_signal(p20, series, 1, bars, 0, 1, out);
    EXPECT_NEAR(out[0], 0.5f, 0.01f);
}

// =============================================================================
// VWAP normalization
// =============================================================================

TEST(SignalNormalize, VWAP_BelowLower_Buy) {
    IndicatorParams params;
    params.type = IndicatorType::VWAP;

    float        vwap[]   = {100.0f};
    float        upper[]  = {102.0f};
    float        lower[]  = {98.0f};
    const float* series[] = {vwap, upper, lower};
    Bar          bars[]   = {make_bar(97.0f)};  // below lower
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

// =============================================================================
// OBV normalization
// =============================================================================

TEST(SignalNormalize, OBV_Rising) {
    IndicatorParams params;
    params.type          = IndicatorType::OBV;
    params.obv_smoothing = 20;  // lookback = obv_smoothing = 20

    // With 3 bars and lookback=20, all i<lookback -> fallback to earliest valid bar
    float        raw[]    = {100.0f, 200.0f, 300.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f), make_bar(101.0f), make_bar(102.0f)};
    float        out[3];

    normalize_to_signal(params, series, 1, bars, 0, 3, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);  // first bar: no prev reference
    EXPECT_FLOAT_EQ(out[1], 1.0f);  // rising vs earliest valid (bar 0)
    EXPECT_FLOAT_EQ(out[2], 1.0f);  // rising vs earliest valid (bar 0)
}

TEST(SignalNormalize, OBV_SlopeOverN_SmoothsNoise) {
    // With bar-to-bar direction, alternating OBV would flip signal each bar.
    // With slope-over-N, the overall upward trend dominates.
    IndicatorParams params;
    params.type          = IndicatorType::OBV;
    params.obv_smoothing = 20;  // lookback = obv_smoothing = 20

    // OBV with upward trend but noisy bar-to-bar: up, down, up, down...
    // Net trend is +400 over 12 bars. With lookback=20 > N=12, all i<lookback
    // fall into the earliest-valid-bar fallback branch (ref = raw[0]=100).
    const int    N        = 12;
    float        raw[N]   = {100, 200, 180, 280, 260, 360, 340, 440, 420, 520, 500, 500};
    const float* series[] = {raw};
    Bar          bars[N];
    for (int i = 0; i < N; ++i) bars[i] = make_bar(100.0f + i);
    float out[N];

    normalize_to_signal(params, series, 1, bars, 0, N, out);

    // With lookback=20 and N=12, both bars use fallback ref=raw[0]=100.
    // Bar 10: raw[10]=500 > 100 -> rising -> +1.
    EXPECT_FLOAT_EQ(out[10], 1.0f);
    // Bar 11: raw[11]=500 > 100 -> rising -> +1.
    EXPECT_FLOAT_EQ(out[11], 1.0f);
}

TEST(SignalNormalize, OBV_SlopeOverN_Falling) {
    // Falling OBV over the lookback window
    IndicatorParams params;
    params.type          = IndicatorType::OBV;
    params.obv_smoothing = 20;  // lookback = obv_smoothing = 20

    const int N = 12;
    float     raw[N];
    for (int i = 0; i < N; ++i) raw[i] = 1000.0f - i * 50.0f;
    const float* series[] = {raw};
    Bar          bars[N];
    for (int i = 0; i < N; ++i) bars[i] = make_bar(100.0f);
    float out[N];

    normalize_to_signal(params, series, 1, bars, 0, N, out);

    // Bar 10: compares raw[10]=500 vs raw[0]=1000 -> falling -> -1
    EXPECT_FLOAT_EQ(out[10], -1.0f);
    EXPECT_FLOAT_EQ(out[11], -1.0f);
}

// =============================================================================
// Donchian normalization
// =============================================================================

TEST(SignalNormalize, Donchian_BreakoutUp) {
    IndicatorParams params;
    params.type = IndicatorType::DONCHIAN;

    float        upper[]  = {105.0f};
    float        lower[]  = {95.0f};
    const float* series[] = {upper, lower};
    Bar          bars[]   = {make_bar(105.0f)};  // at upper
    float        out[1];

    normalize_to_signal(params, series, 2, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

TEST(SignalNormalize, Donchian_BreakoutDown) {
    IndicatorParams params;
    params.type = IndicatorType::DONCHIAN;

    float        upper[]  = {105.0f};
    float        lower[]  = {95.0f};
    const float* series[] = {upper, lower};
    Bar          bars[]   = {make_bar(95.0f)};  // at lower
    float        out[1];

    normalize_to_signal(params, series, 2, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], -1.0f);
}

// =============================================================================
// Pivot Points normalization
// =============================================================================

TEST(SignalNormalize, PivotPoints_AboveR1_Bullish) {
    IndicatorParams params;
    params.type = IndicatorType::PIVOT_POINTS;

    float        pp[]     = {100.0f};
    float        s1[]     = {98.0f};
    float        s2[]     = {96.0f};
    float        s3[]     = {94.0f};
    float        r1[]     = {102.0f};
    float        r2[]     = {104.0f};
    float        r3[]     = {106.0f};
    const float* series[] = {pp, s1, s2, s3, r1, r2, r3};
    Bar          bars[]   = {make_bar(103.0f)};  // above R1
    float        out[1];

    normalize_to_signal(params, series, 7, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

TEST(SignalNormalize, PivotPoints_BelowS1_Bearish) {
    IndicatorParams params;
    params.type = IndicatorType::PIVOT_POINTS;

    float        pp[]     = {100.0f};
    float        s1[]     = {98.0f};
    float        s2[]     = {96.0f};
    float        s3[]     = {94.0f};
    float        r1[]     = {102.0f};
    float        r2[]     = {104.0f};
    float        r3[]     = {106.0f};
    const float* series[] = {pp, s1, s2, s3, r1, r2, r3};
    Bar          bars[]   = {make_bar(97.0f)};  // below S1
    float        out[1];

    normalize_to_signal(params, series, 7, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], -1.0f);
}

// =============================================================================
// ATR normalization (always neutral)
// =============================================================================

TEST(SignalNormalize, ATR_AlwaysZero) {
    IndicatorParams params;
    params.type = IndicatorType::ATR;

    float        short_atr[] = {5.0f, 6.0f};
    float        long_atr[]  = {10.0f, 12.0f};
    const float* series[]    = {short_atr, long_atr};
    Bar          bars[]      = {make_bar(100.0f), make_bar(101.0f)};
    float        out[2];

    normalize_to_signal(params, series, 2, bars, 0, 2, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
}

// =============================================================================
// Keltner normalization
// =============================================================================

TEST(SignalNormalize, Keltner_BelowLower_Buy) {
    IndicatorParams params;
    params.type = IndicatorType::KELTNER;

    float        center[] = {100.0f};
    float        upper[]  = {104.0f};
    float        lower[]  = {96.0f};
    const float* series[] = {center, upper, lower};
    Bar          bars[]   = {make_bar(95.0f)};  // below lower
    float        out[1];

    normalize_to_signal(params, series, 3, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
}

// =============================================================================
// HMA normalization
// =============================================================================

TEST(SignalNormalize, HMA_Rising_Bullish) {
    IndicatorParams params;
    params.type = IndicatorType::HMA;

    // HMA rising -> +1
    float        raw[]    = {100.0f, 101.0f, 102.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f), make_bar(101.0f), make_bar(102.0f)};
    float        out[3];

    normalize_to_signal(params, series, 1, bars, 0, 3, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);  // first bar: no prev
    EXPECT_FLOAT_EQ(out[1], 1.0f);  // rising
    EXPECT_FLOAT_EQ(out[2], 1.0f);  // rising
}

TEST(SignalNormalize, HMA_Falling_Bearish) {
    IndicatorParams params;
    params.type = IndicatorType::HMA;

    // HMA falling -> -1
    float        raw[]    = {102.0f, 101.0f, 100.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(102.0f), make_bar(101.0f), make_bar(100.0f)};
    float        out[3];

    normalize_to_signal(params, series, 1, bars, 0, 3, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);   // first bar: no prev
    EXPECT_FLOAT_EQ(out[1], -1.0f);  // falling
    EXPECT_FLOAT_EQ(out[2], -1.0f);  // falling
}

TEST(SignalNormalize, HMA_NaN_Preserved) {
    IndicatorParams params;
    params.type = IndicatorType::HMA;

    float        raw[]    = {NaN};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_TRUE(std::isnan(out[0]));
}

// =============================================================================
// Connors RSI normalization
// =============================================================================

TEST(SignalNormalize, ConnorsRSI_Overbought) {
    IndicatorParams params;
    params.type               = IndicatorType::CONNORS_RSI;
    params.connors_overbought = 90.0f;
    params.connors_oversold   = 10.0f;

    // ConnorsRSI at 95 (overbought) -> sell signal
    float        raw[]    = {95.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_LT(out[0], 0.0f);
    EXPECT_GE(out[0], -1.0f);
    // (95 - 90) / (100 - 90) = 0.5 -> signal = -0.5
    EXPECT_NEAR(out[0], -0.5f, 0.01f);
}

TEST(SignalNormalize, ConnorsRSI_Oversold) {
    IndicatorParams params;
    params.type               = IndicatorType::CONNORS_RSI;
    params.connors_overbought = 90.0f;
    params.connors_oversold   = 10.0f;

    // ConnorsRSI at 5 (oversold) -> buy signal
    float        raw[]    = {5.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_GT(out[0], 0.0f);
    EXPECT_LE(out[0], 1.0f);
    // (10 - 5) / 10 = 0.5
    EXPECT_NEAR(out[0], 0.5f, 0.01f);
}

TEST(SignalNormalize, ConnorsRSI_Neutral) {
    IndicatorParams params;
    params.type               = IndicatorType::CONNORS_RSI;
    params.connors_overbought = 90.0f;
    params.connors_oversold   = 10.0f;

    // ConnorsRSI at 50 (neutral zone) -> 0
    float        raw[]    = {50.0f};
    const float* series[] = {raw};
    Bar          bars[]   = {make_bar(100.0f)};
    float        out[1];

    normalize_to_signal(params, series, 1, bars, 0, 1, out);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(SignalNormalize, ConnorsRSI_CustomThresholds) {
    // Verify that different OB/OS thresholds produce different signals
    // for the same raw ConnorsRSI value.

    Bar bars[] = {make_bar(100.0f)};

    // Default thresholds: OB=90, OS=10.  Raw value 92 is above OB -> sell.
    {
        IndicatorParams params;
        params.type               = IndicatorType::CONNORS_RSI;
        params.connors_overbought = 90.0f;
        params.connors_oversold   = 10.0f;

        float        raw[]    = {92.0f};
        const float* series[] = {raw};
        float        out[1];

        normalize_to_signal(params, series, 1, bars, 0, 1, out);
        EXPECT_LT(out[0], 0.0f) << "92 above OB=90 should be sell (negative)";
    }

    // Tight thresholds: OB=95, OS=5.  Same raw value 92 is below OB -> neutral.
    {
        IndicatorParams params;
        params.type               = IndicatorType::CONNORS_RSI;
        params.connors_overbought = 95.0f;
        params.connors_oversold   = 5.0f;

        float        raw[]    = {92.0f};
        const float* series[] = {raw};
        float        out[1];

        normalize_to_signal(params, series, 1, bars, 0, 1, out);
        EXPECT_FLOAT_EQ(out[0], 0.0f) << "92 below OB=95 should be neutral";
    }
}

TEST(SignalNormalize, ConnorsRSI_TightThresholds) {
    // With very tight thresholds (OB=95, OS=5), most values should be neutral.
    // With looser thresholds (OB=60, OS=40), more values should generate signals.

    // Spread of raw ConnorsRSI values across the 0-100 range
    const float test_values[] = {8.0f, 15.0f, 35.0f, 50.0f, 65.0f, 85.0f, 93.0f};
    const int   count         = sizeof(test_values) / sizeof(test_values[0]);

    Bar bars[1] = {make_bar(100.0f)};

    // Count signals with tight thresholds (OB=95, OS=5)
    int tight_signals = 0;
    {
        IndicatorParams params;
        params.type               = IndicatorType::CONNORS_RSI;
        params.connors_overbought = 95.0f;
        params.connors_oversold   = 5.0f;

        for (int i = 0; i < count; ++i) {
            float        raw[]    = {test_values[i]};
            const float* series[] = {raw};
            float        out[1];
            normalize_to_signal(params, series, 1, bars, 0, 1, out);
            if (out[0] != 0.0f) ++tight_signals;
        }
    }

    // Count signals with loose thresholds (OB=60, OS=40)
    int loose_signals = 0;
    {
        IndicatorParams params;
        params.type               = IndicatorType::CONNORS_RSI;
        params.connors_overbought = 60.0f;
        params.connors_oversold   = 40.0f;

        for (int i = 0; i < count; ++i) {
            float        raw[]    = {test_values[i]};
            const float* series[] = {raw};
            float        out[1];
            normalize_to_signal(params, series, 1, bars, 0, 1, out);
            if (out[0] != 0.0f) ++loose_signals;
        }
    }

    EXPECT_GT(loose_signals, tight_signals)
        << "Looser thresholds should produce more signals than tighter ones";
    EXPECT_GT(loose_signals, 0) << "Loose thresholds should produce at least some signals";
}

// =============================================================================
// clamp_signal tests
// =============================================================================

TEST(SignalNormalize, ClampSignal) {
    EXPECT_FLOAT_EQ(clamp_signal(0.5f), 0.5f);
    EXPECT_FLOAT_EQ(clamp_signal(1.5f), 1.0f);
    EXPECT_FLOAT_EQ(clamp_signal(-1.5f), -1.0f);
    EXPECT_FLOAT_EQ(clamp_signal(0.0f), 0.0f);
    EXPECT_TRUE(std::isnan(clamp_signal(NaN)));
}

// =============================================================================
// Batch normalization test: normalize_all_indicators
// =============================================================================

TEST(SignalNormalize, NormalizeAllIndicators_Basic) {
    // Create a minimal indicator grid and verify batch normalization
    IndicatorGrid grid;
    int           num_indicators = grid.size();
    ASSERT_GT(num_indicators, 0);

    // Create synthetic bar data
    const int        num_bars = 200;
    std::vector<Bar> bars(num_bars);
    for (int i = 0; i < num_bars; ++i) {
        bars[i].open      = 100.0f + static_cast<float>(i) * 0.1f;
        bars[i].high      = bars[i].open + 1.0f;
        bars[i].low       = bars[i].open - 1.0f;
        bars[i].close     = bars[i].open + 0.05f;
        bars[i]._pad      = 0;
        bars[i].volume    = 100;
        bars[i].timestamp = static_cast<int64_t>(i) * 30'000'000'000LL;
    }

    // Build raw indicator series (filled with simple test patterns)
    int total_raw = 0;
    for (int i = 0; i < num_indicators; ++i) {
        total_raw += indicator_num_series(grid.params(i));
    }

    std::vector<std::vector<float>> raw_storage(total_raw);
    std::vector<float*>             raw_ptrs(total_raw);

    // Fill raw series with values appropriate for each indicator type
    int offset = 0;
    for (int ind = 0; ind < num_indicators; ++ind) {
        const IndicatorParams& params = grid.params(ind);
        int                    ns     = indicator_num_series(params);

        for (int s = 0; s < ns; ++s) {
            raw_storage[offset + s].resize(num_bars);
            raw_ptrs[offset + s] = raw_storage[offset + s].data();

            // Fill with simple values based on indicator type
            for (int i = 0; i < num_bars; ++i) {
                switch (params.type) {
                    case IndicatorType::RSI:
                        raw_ptrs[offset + s][i] = 50.0f;  // neutral RSI
                        break;
                    case IndicatorType::EMA_SMA_CROSS:
                        // fast > slow for bullish
                        raw_ptrs[offset + s][i] = bars[i].close + (s == 0 ? 0.5f : 0.0f);
                        break;
                    case IndicatorType::MACD:
                        if (s == 2)
                            raw_ptrs[offset + s][i] = 0.1f;  // positive histogram
                        else
                            raw_ptrs[offset + s][i] = 1.0f;
                        break;
                    case IndicatorType::ADX:
                        if (s == 0)
                            raw_ptrs[offset + s][i] = 30.0f;  // ADX
                        else if (s == 1)
                            raw_ptrs[offset + s][i] = 25.0f;  // +DI
                        else
                            raw_ptrs[offset + s][i] = 15.0f;  // -DI
                        break;
                    default:
                        raw_ptrs[offset + s][i] = bars[i].close;
                        break;
                }
            }
        }
        offset += ns;
    }

    // Allocate signal output
    std::vector<std::vector<float>> signal_storage(num_indicators);
    std::vector<float*>             signal_ptrs(num_indicators);
    for (int i = 0; i < num_indicators; ++i) {
        signal_storage[i].resize(num_bars);
        signal_ptrs[i] = signal_storage[i].data();
    }

    // Run batch normalization
    normalize_all_indicators(grid, raw_ptrs, bars.data(), num_bars, signal_ptrs);

    // Verify all outputs are in [-1, +1]
    for (int ind = 0; ind < num_indicators; ++ind) {
        expect_signal_range(signal_ptrs[ind], num_bars);
    }
}
