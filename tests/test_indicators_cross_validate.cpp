// =============================================================================
// Indicator Cross-Validation Tests
//
// Cross-validate indicators against independent hand-computed
// reference values. Each test builds a deterministic synthetic price series,
// hand-computes the expected indicator output step-by-step using first
// principles (not the codebase's own functions), and verifies the codebase
// output matches to tight tolerance.
//
// These provide reference checks that do not depend on values exported
// from external charting software.
//
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>
#include "portable_random.h"
#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "market_data.h"
#include "indicators/indicator.h"
#include "indicators/bollinger.h"
#include "indicators/macd.h"
#include "indicators/stochastic.h"
#include "indicators/adx.h"
#include "indicators/donchian.h"
#include "indicators/cci.h"
#include "indicators/pivot_points.h"

// =============================================================================
// Helpers
// =============================================================================

// Generate bars from explicit OHLC arrays.
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

// Generate N bars from close prices. Open = prev close, H = max(O,C)+0.5, L = min(O,C)-0.5.
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

static bool is_nan(float v) {
    return std::isnan(v);
}

// =============================================================================
// a. Bollinger Bands Manual Cross-Validation
//
// Bollinger(20, 2.0) on a 100-bar synthetic series.
// Hand-compute SMA and stddev for bars 19-25.
// Verify middle = SMA, upper/lower = SMA +/- 2*stddev.
// =============================================================================

TEST(Indicators, BollingerBandsManual) {
    // Generate 100 bars with a deterministic seeded RNG for close prices
    std::mt19937            rng(12345);
    portable::normal<float> dist{100.0f, 2.0f};

    std::vector<float> closes(100);
    for (int i = 0; i < 100; ++i) {
        closes[i] = dist(rng);
    }

    auto        bars   = make_bars_from_closes(closes);
    const int   N      = 100;
    const int   period = 20;
    const float mult   = 2.0f;

    // Compute via codebase
    BollingerBands bb(period, mult);
    ASSERT_EQ(bb.warmup_bars(), period - 1);  // 19
    std::vector<float> output(N * 3);
    bb.compute(bars.data(), 0, N, output.data());

    const float* middle = output.data();
    const float* upper  = output.data() + N;
    const float* lower  = output.data() + 2 * N;

    // Hand-compute for bars 19 through 25 (first 7 valid bars)
    for (int i = 19; i <= 25; ++i) {
        // SMA over closes[i-19 .. i]
        double sum = 0.0;
        for (int j = i - period + 1; j <= i; ++j) {
            sum += static_cast<double>(closes[j]);
        }
        double sma = sum / static_cast<double>(period);

        // Population stddev: sqrt(mean((x - sma)^2))
        double var_sum = 0.0;
        for (int j = i - period + 1; j <= i; ++j) {
            double diff = static_cast<double>(closes[j]) - sma;
            var_sum += diff * diff;
        }
        double stddev = std::sqrt(var_sum / static_cast<double>(period));

        float expected_middle = static_cast<float>(sma);
        float expected_upper  = static_cast<float>(sma + mult * stddev);
        float expected_lower  = static_cast<float>(sma - mult * stddev);

        EXPECT_NEAR(middle[i], expected_middle, 1e-4f) << "Bollinger middle mismatch at bar " << i;
        EXPECT_NEAR(upper[i], expected_upper, 1e-4f) << "Bollinger upper mismatch at bar " << i;
        EXPECT_NEAR(lower[i], expected_lower, 1e-4f) << "Bollinger lower mismatch at bar " << i;
    }

    // Also verify at the last bar (bar 99)
    {
        double sum = 0.0;
        for (int j = 80; j <= 99; ++j) {
            sum += static_cast<double>(closes[j]);
        }
        double sma     = sum / 20.0;
        double var_sum = 0.0;
        for (int j = 80; j <= 99; ++j) {
            double diff = static_cast<double>(closes[j]) - sma;
            var_sum += diff * diff;
        }
        double stddev = std::sqrt(var_sum / 20.0);

        EXPECT_NEAR(middle[99], static_cast<float>(sma), 1e-4f)
            << "Bollinger middle mismatch at last bar";
        EXPECT_NEAR(upper[99], static_cast<float>(sma + 2.0 * stddev), 1e-4f)
            << "Bollinger upper mismatch at last bar";
        EXPECT_NEAR(lower[99], static_cast<float>(sma - 2.0 * stddev), 1e-4f)
            << "Bollinger lower mismatch at last bar";
    }
}

// =============================================================================
// b. MACD Manual Cross-Validation
//
// MACD(12, 26, 9) on a 50-bar synthetic series.
// Hand-compute EMA(12) and EMA(26) for the final 5 bars.
// Verify MACD line = fast - slow. Verify signal = EMA(9) of MACD line.
// =============================================================================

TEST(Indicators, MACDManual) {
    // Create 200 bars (need warmup = 3*26 + 3*9 = 105 bars)
    // Use a deterministic trending + oscillating series
    std::mt19937            rng(42);
    portable::normal<float> noise{0.0f, 0.3f};
    std::vector<float>      closes(200);
    closes[0] = 100.0f;
    for (int i = 1; i < 200; ++i) {
        closes[i] = closes[i - 1] + 0.05f + noise(rng);
    }

    auto      bars = make_bars_from_closes(closes);
    const int N    = 200;

    // Compute via codebase
    MACD      macd(12, 26, 9);
    const int wu = macd.warmup_bars();  // 105
    ASSERT_EQ(wu, 105);
    std::vector<float> output(N * 3);
    macd.compute(bars.data(), 0, N, output.data());

    const float* macd_line   = output.data();
    const float* signal_line = output.data() + N;
    const float* histogram   = output.data() + 2 * N;

    // Hand-compute EMA(12) and EMA(26) from scratch
    const float fast_alpha = 2.0f / (12.0f + 1.0f);
    const float slow_alpha = 2.0f / (26.0f + 1.0f);
    const float sig_alpha  = 2.0f / (9.0f + 1.0f);

    float fast_ema = closes[0];
    float slow_ema = closes[0];

    std::vector<float> macd_vals(N);
    macd_vals[0] = fast_ema - slow_ema;

    for (int i = 1; i < N; ++i) {
        fast_ema     = fast_alpha * closes[i] + (1.0f - fast_alpha) * fast_ema;
        slow_ema     = slow_alpha * closes[i] + (1.0f - slow_alpha) * slow_ema;
        macd_vals[i] = fast_ema - slow_ema;
    }

    // Hand-compute signal EMA of MACD values
    float              sig_ema = macd_vals[0];
    std::vector<float> signal_vals(N);
    signal_vals[0] = sig_ema;
    for (int i = 1; i < N; ++i) {
        sig_ema        = sig_alpha * macd_vals[i] + (1.0f - sig_alpha) * sig_ema;
        signal_vals[i] = sig_ema;
    }

    // Verify final 5 bars (195-199) -- well past warmup
    for (int i = 195; i < 200; ++i) {
        EXPECT_NEAR(macd_line[i], macd_vals[i], 1e-3f) << "MACD line mismatch at bar " << i;
        EXPECT_NEAR(signal_line[i], signal_vals[i], 1e-3f) << "Signal line mismatch at bar " << i;

        float expected_hist = macd_vals[i] - signal_vals[i];
        EXPECT_NEAR(histogram[i], expected_hist, 1e-3f) << "Histogram mismatch at bar " << i;
    }

    // Also verify histogram identity for all post-warmup bars
    for (int i = wu; i < N; ++i) {
        EXPECT_NEAR(histogram[i], macd_line[i] - signal_line[i], 1e-5f)
            << "Histogram != MACD - signal at bar " << i;
    }
}

// =============================================================================
// c. Stochastic Manual Cross-Validation
//
// 30-bar series with known high/low.
// Hand-compute %K(14) for specific bars.
// =============================================================================

TEST(Indicators, StochasticManual) {
    // Build 30 bars with explicit OHLC so we know exact highs and lows
    std::vector<std::array<float, 4>> ohlc(30);
    for (int i = 0; i < 30; ++i) {
        float base  = 100.0f + static_cast<float>(i) * 0.5f;
        float swing = 2.0f * std::sin(static_cast<float>(i) * 0.4f);
        float c     = base + swing;
        float o     = base;
        float h     = std::max(o, c) + 1.0f;
        float l     = std::min(o, c) - 1.0f;
        ohlc[i]     = {o, h, l, c};
    }
    auto      bars     = make_bars_ohlc(ohlc);
    const int N        = 30;
    const int k_period = 14;
    const int d_period = 3;

    // Compute via codebase
    Stochastic stoch(k_period, d_period);
    ASSERT_EQ(stoch.warmup_bars(), k_period + d_period);  // 17
    std::vector<float> output(N * 2);
    stoch.compute(bars.data(), 0, N, output.data());

    const float* pct_k = output.data();
    const float* pct_d = output.data() + N;

    // Hand-compute %K for bars 13..29 (first k_period-1 bars have no %K)
    std::vector<float> raw_k(N, std::numeric_limits<float>::quiet_NaN());
    for (int i = k_period - 1; i < N; ++i) {
        // Find highest high and lowest low in [i - k_period + 1, i]
        float hh = -1e30f;
        float ll = 1e30f;
        for (int j = i - k_period + 1; j <= i; ++j) {
            hh = std::max(hh, ohlc[j][1]);  // high
            ll = std::min(ll, ohlc[j][2]);  // low
        }
        float range = hh - ll;
        if (range == 0.0f) {
            raw_k[i] = 50.0f;
        } else {
            raw_k[i] = (ohlc[i][3] - ll) / range * 100.0f;  // close
        }
    }

    // Verify %K for bars after warmup (bar 17 onward)
    for (int i = stoch.warmup_bars(); i < N; ++i) {
        ASSERT_FALSE(is_nan(pct_k[i])) << "%K should be valid at bar " << i;
        EXPECT_NEAR(pct_k[i], raw_k[i], 1e-4f) << "%K mismatch at bar " << i;
    }

    // Hand-compute %D = SMA(%K, 3) starting from first valid %K
    // The first valid %K is at index k_period - 1 = 13.
    // %D needs d_period=3 valid %K values, so first %D is at index 15.
    // But warmup masks output before index 17, so verify bars 17-29.
    for (int i = stoch.warmup_bars(); i < N; ++i) {
        // SMA of raw_k[i-2], raw_k[i-1], raw_k[i]
        float d_sum = 0.0f;
        int   count = 0;
        for (int j = i - d_period + 1; j <= i; ++j) {
            if (j >= k_period - 1 && !std::isnan(raw_k[j])) {
                d_sum += raw_k[j];
                count++;
            }
        }
        if (count == d_period) {
            float expected_d = d_sum / static_cast<float>(d_period);
            ASSERT_FALSE(is_nan(pct_d[i])) << "%D should be valid at bar " << i;
            EXPECT_NEAR(pct_d[i], expected_d, 1e-3f) << "%D mismatch at bar " << i;
        }
    }

    // Verify range: all valid %K and %D should be in [0, 100]
    for (int i = stoch.warmup_bars(); i < N; ++i) {
        if (!is_nan(pct_k[i])) {
            EXPECT_GE(pct_k[i], 0.0f);
            EXPECT_LE(pct_k[i], 100.0f);
        }
        if (!is_nan(pct_d[i])) {
            EXPECT_GE(pct_d[i], 0.0f);
            EXPECT_LE(pct_d[i], 100.0f);
        }
    }
}

// =============================================================================
// d. ADX Manual Cross-Validation
//
// 50-bar series. Hand-compute +DM, -DM, TR, Wilder-smoothed values,
// DX, and ADX for specific bars after warmup.
// =============================================================================

TEST(Indicators, ADXManual) {
    // Build 80 bars with explicit OHLC for a strong uptrend
    // so we can hand-compute the Wilder smoothing precisely
    const int                         N      = 80;
    const int                         period = 7;
    std::vector<std::array<float, 4>> ohlc(N);

    // Create a steady uptrend: each bar's high is above the previous
    std::mt19937             rng(99);
    portable::uniform<float> jitter{0.0f, 0.3f};
    for (int i = 0; i < N; ++i) {
        float base = 100.0f + static_cast<float>(i) * 1.5f;
        float o    = base;
        float c    = base + 0.8f + jitter(rng);
        float h    = std::max(o, c) + 0.5f + jitter(rng);
        float l    = std::min(o, c) - 0.3f - jitter(rng);
        ohlc[i]    = {o, h, l, c};
    }
    auto bars = make_bars_ohlc(ohlc);

    // Compute via codebase
    ADX adx(period);
    ASSERT_EQ(adx.warmup_bars(), 4 * period);  // 28
    std::vector<float> output(N * 3);
    adx.compute(bars.data(), 0, N, output.data());

    const float* adx_vals = output.data();
    const float* plus_di  = output.data() + N;
    const float* minus_di = output.data() + 2 * N;

    // Hand-compute ADX step by step
    // Step 1: compute +DM, -DM, TR for bars 1..N-1
    std::vector<float> plus_dm(N, 0.0f);
    std::vector<float> minus_dm(N, 0.0f);
    std::vector<float> tr(N, 0.0f);

    for (int i = 1; i < N; ++i) {
        float up_move   = ohlc[i][1] - ohlc[i - 1][1];  // high[i] - high[i-1]
        float down_move = ohlc[i - 1][2] - ohlc[i][2];  // low[i-1] - low[i]

        if (up_move > down_move && up_move > 0.0f) plus_dm[i] = up_move;
        if (down_move > up_move && down_move > 0.0f) minus_dm[i] = down_move;

        float prev_close = ohlc[i - 1][3];
        float hl         = ohlc[i][1] - ohlc[i][2];
        float hpc        = std::abs(ohlc[i][1] - prev_close);
        float lpc        = std::abs(ohlc[i][2] - prev_close);
        tr[i]            = std::max({hl, hpc, lpc});
    }

    // Step 2: First smoothed values at bar = period (sum of first 'period' changes)
    float smooth_pdm = 0.0f, smooth_mdm = 0.0f, smooth_tr = 0.0f;
    for (int i = 1; i <= period; ++i) {
        smooth_pdm += plus_dm[i];
        smooth_mdm += minus_dm[i];
        smooth_tr += tr[i];
    }

    // Compute first DI values
    float pdi_val = (smooth_tr > 0.0f) ? (smooth_pdm / smooth_tr) * 100.0f : 0.0f;
    float mdi_val = (smooth_tr > 0.0f) ? (smooth_mdm / smooth_tr) * 100.0f : 0.0f;
    float di_sum  = pdi_val + mdi_val;
    float dx_val  = (di_sum > 0.0f) ? (std::abs(pdi_val - mdi_val) / di_sum) * 100.0f : 0.0f;

    float sum_dx          = dx_val;
    int   dx_count        = 1;
    float adx_val         = 0.0f;
    bool  adx_initialized = false;

    const float period_f = static_cast<float>(period);

    // Step 3+: Wilder-smooth through the rest
    for (int i = period + 1; i < N; ++i) {
        smooth_pdm = smooth_pdm - (smooth_pdm / period_f) + plus_dm[i];
        smooth_mdm = smooth_mdm - (smooth_mdm / period_f) + minus_dm[i];
        smooth_tr  = smooth_tr - (smooth_tr / period_f) + tr[i];

        pdi_val = (smooth_tr > 0.0f) ? (smooth_pdm / smooth_tr) * 100.0f : 0.0f;
        mdi_val = (smooth_tr > 0.0f) ? (smooth_mdm / smooth_tr) * 100.0f : 0.0f;
        di_sum  = pdi_val + mdi_val;
        dx_val  = (di_sum > 0.0f) ? (std::abs(pdi_val - mdi_val) / di_sum) * 100.0f : 0.0f;

        if (!adx_initialized) {
            sum_dx += dx_val;
            dx_count++;
            if (dx_count == period) {
                adx_val         = sum_dx / period_f;
                adx_initialized = true;
            }
        } else {
            adx_val = (adx_val * (period_f - 1.0f) + dx_val) / period_f;
        }

        // Only verify bars past warmup (bar >= 28)
        if (i >= adx.warmup_bars()) {
            ASSERT_FALSE(is_nan(adx_vals[i])) << "ADX should be valid at bar " << i;
            EXPECT_NEAR(adx_vals[i], adx_val, 1e-2f) << "ADX mismatch at bar " << i;
            EXPECT_NEAR(plus_di[i], pdi_val, 1e-2f) << "+DI mismatch at bar " << i;
            EXPECT_NEAR(minus_di[i], mdi_val, 1e-2f) << "-DI mismatch at bar " << i;
        }
    }

    // Verify ADX is in [0, 100] for all valid bars
    for (int i = adx.warmup_bars(); i < N; ++i) {
        if (!is_nan(adx_vals[i])) {
            EXPECT_GE(adx_vals[i], 0.0f);
            EXPECT_LE(adx_vals[i], 100.0f);
        }
    }

    // In a strong uptrend, +DI should be > -DI for most bars
    int plus_dominant = 0;
    for (int i = adx.warmup_bars(); i < N; ++i) {
        if (!is_nan(plus_di[i]) && !is_nan(minus_di[i]) && plus_di[i] > minus_di[i])
            plus_dominant++;
    }
    int valid_bars = N - adx.warmup_bars();
    EXPECT_GT(plus_dominant, valid_bars / 2)
        << "+DI should be > -DI for most bars in a strong uptrend";
}

// =============================================================================
// e. Donchian Channel Manual Cross-Validation
//
// 30-bar series. Verify upper = highest high of last N bars,
// lower = lowest low of last N bars. Easy to hand-verify.
// =============================================================================

TEST(Indicators, DonchianManual) {
    // Build 30 bars with explicit OHLC
    const int                         N      = 30;
    const int                         period = 10;
    std::vector<std::array<float, 4>> ohlc(N);

    // Create a recognizable price pattern
    for (int i = 0; i < N; ++i) {
        float base = 100.0f;
        // Oscillating pattern: rises then falls
        float offset = 5.0f * std::sin(static_cast<float>(i) * 0.3f);
        float o      = base + offset;
        float c      = base + offset + 0.5f;
        float h      = std::max(o, c) + 1.0f + static_cast<float>(i % 3) * 0.5f;
        float l      = std::min(o, c) - 1.0f - static_cast<float>(i % 4) * 0.3f;
        ohlc[i]      = {o, h, l, c};
    }
    auto bars = make_bars_ohlc(ohlc);

    // Compute via codebase
    DonchianChannel dc(period);
    ASSERT_EQ(dc.warmup_bars(), period - 1);  // 9
    std::vector<float> output(N * 2);
    dc.compute(bars.data(), 0, N, output.data());

    const float* upper = output.data();
    const float* lower = output.data() + N;

    // Verify: before warmup should be NaN
    for (int i = 0; i < period - 1; ++i) {
        EXPECT_TRUE(is_nan(upper[i])) << "Upper should be NaN at bar " << i;
        EXPECT_TRUE(is_nan(lower[i])) << "Lower should be NaN at bar " << i;
    }

    // Hand-verify for each bar from warmup onward
    for (int i = period - 1; i < N; ++i) {
        float expected_high = -1e30f;
        float expected_low  = 1e30f;
        for (int j = i - period + 1; j <= i; ++j) {
            expected_high = std::max(expected_high, ohlc[j][1]);  // high
            expected_low  = std::min(expected_low, ohlc[j][2]);   // low
        }

        EXPECT_NEAR(upper[i], expected_high, 1e-5f) << "Donchian upper mismatch at bar " << i;
        EXPECT_NEAR(lower[i], expected_low, 1e-5f) << "Donchian lower mismatch at bar " << i;

        // Structural check: upper >= lower always
        EXPECT_GE(upper[i], lower[i]) << "Upper must be >= lower at bar " << i;
    }
}

// =============================================================================
// f. CCI Manual Cross-Validation
//
// 30-bar series. Hand-compute typical price, SMA of typical, mean absolute
// deviation. Verify CCI = (TP - SMA_TP) / (0.015 * MAD).
// =============================================================================

TEST(Indicators, CCIManual) {
    const int N      = 30;
    const int period = 10;

    // Build bars with explicit OHLC
    std::vector<std::array<float, 4>> ohlc(N);
    std::mt19937                      rng(777);
    portable::normal<float>           noise{0.0f, 1.0f};
    for (int i = 0; i < N; ++i) {
        float base = 100.0f + static_cast<float>(i) * 0.5f + noise(rng);
        float h    = base + 2.0f;
        float l    = base - 2.0f;
        float o    = base;
        float c    = base + 0.3f;
        ohlc[i]    = {o, h, l, c};
    }
    auto bars = make_bars_ohlc(ohlc);

    // Compute via codebase
    CCI cci(period);
    ASSERT_EQ(cci.warmup_bars(), period - 1);  // 9
    std::vector<float> output(N);
    cci.compute(bars.data(), 0, N, output.data());

    // Hand-compute typical prices
    std::vector<float> tp(N);
    for (int i = 0; i < N; ++i) {
        tp[i] = (ohlc[i][1] + ohlc[i][2] + ohlc[i][3]) / 3.0f;  // (H+L+C)/3
    }

    // Verify CCI for all bars past warmup
    for (int i = period - 1; i < N; ++i) {
        // SMA of TP over [i-period+1, i]
        float tp_sum = 0.0f;
        for (int j = i - period + 1; j <= i; ++j) {
            tp_sum += tp[j];
        }
        float sma_tp = tp_sum / static_cast<float>(period);

        // Mean Absolute Deviation
        float mad_sum = 0.0f;
        for (int j = i - period + 1; j <= i; ++j) {
            mad_sum += std::abs(tp[j] - sma_tp);
        }
        float mad = mad_sum / static_cast<float>(period);

        float expected_cci;
        if (mad == 0.0f) {
            expected_cci = 0.0f;
        } else {
            expected_cci = (tp[i] - sma_tp) / (0.015f * mad);
        }

        ASSERT_FALSE(is_nan(output[i])) << "CCI should be valid at bar " << i;
        EXPECT_NEAR(output[i], expected_cci, 1e-2f) << "CCI mismatch at bar " << i;
    }

    // Verify warmup bars are NaN
    for (int i = 0; i < period - 1; ++i) {
        EXPECT_TRUE(is_nan(output[i])) << "CCI should be NaN during warmup at bar " << i;
    }
}

// =============================================================================
// g. Pivot Points Manual Cross-Validation
//
// Create 2-session synthetic data with known OHLC.
// Verify Classic pivot = (H+L+C)/3, S1 = 2*PP - H, R1 = 2*PP - L.
// =============================================================================

TEST(Indicators, PivotPointsManual) {
    // Session 1: 10 bars at 30s intervals
    // Session 2: starts after a 60-minute gap (triggers session boundary)
    // Session 3: starts after another 60-minute gap
    // Pivot levels in session 2 should be derived from session 1 OHLC.
    // Pivot levels in session 3 should be derived from session 2 OHLC.

    const int64_t base_ts      = 1'262'304'000LL * NS_PER_SEC;
    const int64_t bar_interval = 30LL * NS_PER_SEC;
    const int64_t session_gap  = 3600LL * NS_PER_SEC;  // 60 min gap

    std::vector<Bar> bars;

    // Session 1: 10 bars. H=110, L=90, C(last)=105
    for (int i = 0; i < 10; ++i) {
        float c = 100.0f + static_cast<float>(i) * 0.5f;
        float o = 100.0f;
        float h, l;
        if (i == 2) {
            h = 110.0f;
            l = 98.0f;
        }  // session high
        else if (i == 5) {
            h = 101.0f;
            l = 90.0f;
        }  // session low
        else {
            h = 102.0f;
            l = 99.0f;
        }
        if (i == 9) c = 105.0f;  // last close = 105
        bars.push_back(make_bar(base_ts + i * bar_interval, o, h, l, c));
    }
    // Record session 1 extremes
    float s1_high  = 110.0f;
    float s1_low   = 90.0f;
    float s1_close = 105.0f;

    // Session 2: 10 bars after 60-min gap. H=130, L=115, C(last)=125
    int64_t s2_start = base_ts + 10 * bar_interval + session_gap;
    for (int i = 0; i < 10; ++i) {
        float c = 120.0f + static_cast<float>(i) * 0.5f;
        float o = 120.0f;
        float h, l;
        if (i == 3) {
            h = 130.0f;
            l = 118.0f;
        }  // session high
        else if (i == 7) {
            h = 121.0f;
            l = 115.0f;
        }  // session low
        else {
            h = 122.0f;
            l = 119.0f;
        }
        if (i == 9) c = 125.0f;  // last close = 125
        bars.push_back(make_bar(s2_start + i * bar_interval, o, h, l, c));
    }
    float s2_high  = 130.0f;
    float s2_low   = 115.0f;
    float s2_close = 125.0f;

    // Session 3: 5 bars after another 60-min gap
    int64_t s3_start = s2_start + 10 * bar_interval + session_gap;
    for (int i = 0; i < 5; ++i) {
        float c = 126.0f + static_cast<float>(i) * 0.3f;
        bars.push_back(make_bar(s3_start + i * bar_interval, 126.0f, 127.0f, 125.0f, c));
    }

    const int N = static_cast<int>(bars.size());

    // Compute via codebase (Classic pivots)
    PivotPoints        pp(PivotType::CLASSIC);
    std::vector<float> output(N * 7);
    pp.compute(bars.data(), 0, N, output.data());

    // Series layout: PP, S1, S2, S3, R1, R2, R3
    const float* pp_out = output.data();
    const float* s1_out = output.data() + N;
    const float* s2_out = output.data() + 2 * N;
    const float* s3_out = output.data() + 3 * N;
    const float* r1_out = output.data() + 4 * N;
    const float* r2_out = output.data() + 5 * N;
    const float* r3_out = output.data() + 6 * N;

    // Session 1 bars (0-9): no prior session => NaN
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(is_nan(pp_out[i])) << "PP should be NaN in first session at bar " << i;
    }

    // Session 2 bars (10-19): pivots from session 1
    {
        float expected_pp = (s1_high + s1_low + s1_close) / 3.0f;
        float expected_s1 = 2.0f * expected_pp - s1_high;
        float expected_r1 = 2.0f * expected_pp - s1_low;
        float expected_s2 = expected_pp - (s1_high - s1_low);
        float expected_r2 = expected_pp + (s1_high - s1_low);
        float expected_s3 = s1_low - 2.0f * (s1_high - expected_pp);
        float expected_r3 = s1_high + 2.0f * (expected_pp - s1_low);

        for (int i = 10; i < 20; ++i) {
            ASSERT_FALSE(is_nan(pp_out[i])) << "PP should be valid in session 2 at bar " << i;
            EXPECT_NEAR(pp_out[i], expected_pp, 1e-4f) << "PP mismatch at bar " << i;
            EXPECT_NEAR(s1_out[i], expected_s1, 1e-4f) << "S1 mismatch at bar " << i;
            EXPECT_NEAR(r1_out[i], expected_r1, 1e-4f) << "R1 mismatch at bar " << i;
            EXPECT_NEAR(s2_out[i], expected_s2, 1e-4f) << "S2 mismatch at bar " << i;
            EXPECT_NEAR(r2_out[i], expected_r2, 1e-4f) << "R2 mismatch at bar " << i;
            EXPECT_NEAR(s3_out[i], expected_s3, 1e-4f) << "S3 mismatch at bar " << i;
            EXPECT_NEAR(r3_out[i], expected_r3, 1e-4f) << "R3 mismatch at bar " << i;
        }
    }

    // Session 3 bars (20-24): pivots from session 2
    {
        float expected_pp = (s2_high + s2_low + s2_close) / 3.0f;
        float expected_s1 = 2.0f * expected_pp - s2_high;
        float expected_r1 = 2.0f * expected_pp - s2_low;

        for (int i = 20; i < 25; ++i) {
            ASSERT_FALSE(is_nan(pp_out[i])) << "PP should be valid in session 3 at bar " << i;
            EXPECT_NEAR(pp_out[i], expected_pp, 1e-4f) << "PP mismatch at bar " << i;
            EXPECT_NEAR(s1_out[i], expected_s1, 1e-4f) << "S1 mismatch at bar " << i;
            EXPECT_NEAR(r1_out[i], expected_r1, 1e-4f) << "R1 mismatch at bar " << i;
        }
    }

    // Also test Fibonacci pivot type
    {
        PivotPoints        pp_fib(PivotType::FIBONACCI);
        std::vector<float> fib_output(N * 7);
        pp_fib.compute(bars.data(), 0, N, fib_output.data());

        const float* fib_pp = fib_output.data();
        const float* fib_s1 = fib_output.data() + N;
        const float* fib_r1 = fib_output.data() + 4 * N;

        // Session 2 bars: Fibonacci pivots from session 1
        float expected_pp     = (s1_high + s1_low + s1_close) / 3.0f;
        float range           = s1_high - s1_low;
        float expected_fib_s1 = expected_pp - 0.382f * range;
        float expected_fib_r1 = expected_pp + 0.382f * range;

        EXPECT_NEAR(fib_pp[10], expected_pp, 1e-4f);
        EXPECT_NEAR(fib_s1[10], expected_fib_s1, 1e-4f);
        EXPECT_NEAR(fib_r1[10], expected_fib_r1, 1e-4f);
    }
}

// =============================================================================
// Fixture-Data Cross-Validation Tests
//
// Load the test fixture bar file (tests/data/synthetic_30s.bin),
// compute indicators with the C++ implementation, load reference values
// from tests/reference/synthetic_indicators_reference.csv (generated by an
// independent Python re-implementation), and compare with tight tolerance.
//
// These tests validate the C++ indicators against a completely independent
// codebase, catching any algorithmic bugs that synthetic tests might miss
// (e.g., edge cases with session gaps, price levels, etc.).
// =============================================================================

#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

#include "indicators/ema_sma_cross.h"
#include "indicators/rsi.h"
#include "indicators/atr.h"
#include "test_fixture.h"

// ---------------------------------------------------------------------------
// Helpers for fixture-data cross-validation
// ---------------------------------------------------------------------------

// Derive test file paths from __FILE__ (the source file is in tests/)
static const std::string TEST_BIN_PATH      = test_fixture::fixture_path();
static const std::string REFERENCE_CSV_PATH = test_fixture::reference_csv_path();

// Parse the reference CSV into parallel vectors.
// Returns the number of rows parsed.
struct ReferenceRow {
    int   bar_index;
    float ema20;
    float rsi14;
    float atr14;
    float sma50;
    float macd_line;
    float macd_signal;
    float macd_histogram;
    float bb_upper;
    float bb_middle;
    float bb_lower;
};

static float parse_ref_float(const std::string& s) {
    if (s == "NaN" || s == "nan" || s == "NAN" || s.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::stof(s);
}

static std::vector<ReferenceRow> load_reference_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return {};
    }

    std::vector<ReferenceRow> rows;
    std::string               line;

    // Skip header
    std::getline(f, line);

    while (std::getline(f, line)) {
        if (line.empty()) continue;

        std::istringstream       ss(line);
        std::string              token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 11) continue;

        ReferenceRow row{};
        row.bar_index      = std::stoi(tokens[0]);
        row.ema20          = parse_ref_float(tokens[1]);
        row.rsi14          = parse_ref_float(tokens[2]);
        row.atr14          = parse_ref_float(tokens[3]);
        row.sma50          = parse_ref_float(tokens[4]);
        row.macd_line      = parse_ref_float(tokens[5]);
        row.macd_signal    = parse_ref_float(tokens[6]);
        row.macd_histogram = parse_ref_float(tokens[7]);
        row.bb_upper       = parse_ref_float(tokens[8]);
        row.bb_middle      = parse_ref_float(tokens[9]);
        row.bb_lower       = parse_ref_float(tokens[10]);

        rows.push_back(row);
    }

    return rows;
}

// Test fixture that loads data once for all fixture-data tests
class FixtureCrossValidation : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Load binary test data
        s_data_loaded = s_market_data.load(TEST_BIN_PATH);
        if (!s_data_loaded) {
            // File not present -- tests will be skipped
            return;
        }
        s_bar_count = s_market_data.bar_count();

        // Load reference CSV
        s_ref        = load_reference_csv(REFERENCE_CSV_PATH);
        s_ref_loaded = !s_ref.empty();
    }

    void SetUp() override {
        if (!s_data_loaded) {
            GTEST_SKIP() << "Test data not found: " << TEST_BIN_PATH;
        }
        if (!s_ref_loaded) {
            GTEST_SKIP() << "Reference CSV not found: " << REFERENCE_CSV_PATH;
        }
    }

    static MarketData                s_market_data;
    static bool                      s_data_loaded;
    static int                       s_bar_count;
    static std::vector<ReferenceRow> s_ref;
    static bool                      s_ref_loaded;
};

MarketData                FixtureCrossValidation::s_market_data;
bool                      FixtureCrossValidation::s_data_loaded = false;
int                       FixtureCrossValidation::s_bar_count   = 0;
std::vector<ReferenceRow> FixtureCrossValidation::s_ref;
bool                      FixtureCrossValidation::s_ref_loaded = false;

// ---------------------------------------------------------------------------
// h. EMA-20 Cross-Validation Against Python Reference
// ---------------------------------------------------------------------------

TEST_F(FixtureCrossValidation, EMA20) {
    const int N      = s_bar_count;
    const int period = 20;

    // Compute EMA-20 using EMASMACross with both fast and slow set to EMA-20
    // (we only need the fast series)
    EMASMACross        ema_cross(MAType::EMA, period, MAType::EMA, period + 1);
    std::vector<float> output(N * 2);
    ema_cross.compute(s_market_data.bars(), 0, N, output.data());

    // Series 0 = fast EMA (period 20)
    const float* ema_out = output.data();

    // Cross-validate against reference
    const int warmup     = 3 * period;  // 60
    int       compared   = 0;
    int       mismatches = 0;

    for (int i = warmup; i < N && i < static_cast<int>(s_ref.size()); ++i) {
        if (std::isnan(s_ref[i].ema20)) continue;
        if (std::isnan(ema_out[i])) {
            ADD_FAILURE() << "C++ EMA-20 is NaN at bar " << i << " but reference is "
                          << s_ref[i].ema20;
            mismatches++;
            if (mismatches > 10) break;
            continue;
        }

        // Relative tolerance: 0.01% of the reference value
        float ref_val   = s_ref[i].ema20;
        float tolerance = std::abs(ref_val) * 0.0001f;
        if (tolerance < 1e-6f) tolerance = 1e-6f;

        EXPECT_NEAR(ema_out[i], ref_val, tolerance) << "EMA-20 mismatch at bar " << i;
        compared++;
    }

    EXPECT_GT(compared, 1000) << "Expected at least 1000 bars compared for EMA-20";
    std::cout << "  EMA-20: compared " << compared << " bars" << std::endl;
}

// ---------------------------------------------------------------------------
// i. RSI-14 Cross-Validation Against Python Reference
// ---------------------------------------------------------------------------

TEST_F(FixtureCrossValidation, RSI14) {
    const int N = s_bar_count;

    RSI                rsi(14);
    std::vector<float> output(N);
    rsi.compute(s_market_data.bars(), 0, N, output.data());

    const int warmup   = rsi.warmup_bars();  // 42
    int       compared = 0;

    for (int i = warmup; i < N && i < static_cast<int>(s_ref.size()); ++i) {
        if (std::isnan(s_ref[i].rsi14)) continue;
        if (std::isnan(output[i])) {
            ADD_FAILURE() << "C++ RSI-14 is NaN at bar " << i << " but reference is "
                          << s_ref[i].rsi14;
            continue;
        }

        // RSI is in [0, 100]. Use absolute tolerance of 0.01 (0.01%)
        float tolerance = 0.01f;

        EXPECT_NEAR(output[i], s_ref[i].rsi14, tolerance) << "RSI-14 mismatch at bar " << i;
        compared++;
    }

    EXPECT_GT(compared, 1000) << "Expected at least 1000 bars compared for RSI-14";
    std::cout << "  RSI-14: compared " << compared << " bars" << std::endl;
}

// ---------------------------------------------------------------------------
// j. ATR-14 Cross-Validation Against Python Reference
// ---------------------------------------------------------------------------

TEST_F(FixtureCrossValidation, ATR14) {
    const int N = s_bar_count;

    ATR                atr(14);
    std::vector<float> output(N);
    atr.compute(s_market_data.bars(), 0, N, output.data());

    const int warmup   = atr.warmup_bars();  // 42
    int       compared = 0;

    for (int i = warmup; i < N && i < static_cast<int>(s_ref.size()); ++i) {
        if (std::isnan(s_ref[i].atr14)) continue;
        if (std::isnan(output[i])) {
            ADD_FAILURE() << "C++ ATR-14 is NaN at bar " << i << " but reference is "
                          << s_ref[i].atr14;
            continue;
        }

        // Relative tolerance: 0.01% of reference value
        float ref_val   = s_ref[i].atr14;
        float tolerance = std::abs(ref_val) * 0.0001f;
        if (tolerance < 1e-6f) tolerance = 1e-6f;

        EXPECT_NEAR(output[i], ref_val, tolerance) << "ATR-14 mismatch at bar " << i;
        compared++;
    }

    EXPECT_GT(compared, 1000) << "Expected at least 1000 bars compared for ATR-14";
    std::cout << "  ATR-14: compared " << compared << " bars" << std::endl;
}

// ---------------------------------------------------------------------------
// k. SMA-50 Cross-Validation Against Python Reference
// ---------------------------------------------------------------------------

TEST_F(FixtureCrossValidation, SMA50) {
    const int N      = s_bar_count;
    const int period = 50;

    // Use EMASMACross with SMA type
    EMASMACross        sma_cross(MAType::SMA, period - 1, MAType::SMA, period);
    std::vector<float> output(N * 2);
    sma_cross.compute(s_market_data.bars(), 0, N, output.data());

    // Series 1 = slow SMA (period 50)
    const float* sma_out = output.data() + N;

    const int warmup   = period - 1;  // 49
    int       compared = 0;

    for (int i = warmup; i < N && i < static_cast<int>(s_ref.size()); ++i) {
        if (std::isnan(s_ref[i].sma50)) continue;
        if (std::isnan(sma_out[i])) {
            ADD_FAILURE() << "C++ SMA-50 is NaN at bar " << i << " but reference is "
                          << s_ref[i].sma50;
            continue;
        }

        // Relative tolerance: 0.01% of reference value
        float ref_val   = s_ref[i].sma50;
        float tolerance = std::abs(ref_val) * 0.0001f;
        if (tolerance < 1e-6f) tolerance = 1e-6f;

        EXPECT_NEAR(sma_out[i], ref_val, tolerance) << "SMA-50 mismatch at bar " << i;
        compared++;
    }

    EXPECT_GT(compared, 1000) << "Expected at least 1000 bars compared for SMA-50";
    std::cout << "  SMA-50: compared " << compared << " bars" << std::endl;
}

// ---------------------------------------------------------------------------
// l. MACD(12,26,9) Cross-Validation Against Python Reference
// ---------------------------------------------------------------------------

TEST_F(FixtureCrossValidation, MACD_12_26_9) {
    const int N = s_bar_count;

    MACD               macd(12, 26, 9);
    std::vector<float> output(N * 3);
    macd.compute(s_market_data.bars(), 0, N, output.data());

    const float* macd_line_out = output.data();
    const float* signal_out    = output.data() + N;
    const float* hist_out      = output.data() + 2 * N;

    const int warmup   = macd.warmup_bars();  // 105
    int       compared = 0;

    for (int i = warmup; i < N && i < static_cast<int>(s_ref.size()); ++i) {
        if (std::isnan(s_ref[i].macd_line)) continue;

        // MACD line
        if (!std::isnan(macd_line_out[i])) {
            float ref_val = s_ref[i].macd_line;
            // MACD can be near zero, use max of relative and absolute tolerance
            float tolerance = std::max(std::abs(ref_val) * 0.0001f, 0.001f);

            EXPECT_NEAR(macd_line_out[i], ref_val, tolerance) << "MACD line mismatch at bar " << i;
        }

        // Signal line
        if (!std::isnan(signal_out[i]) && !std::isnan(s_ref[i].macd_signal)) {
            float ref_val   = s_ref[i].macd_signal;
            float tolerance = std::max(std::abs(ref_val) * 0.0001f, 0.001f);

            EXPECT_NEAR(signal_out[i], ref_val, tolerance) << "MACD signal mismatch at bar " << i;
        }

        // Histogram
        if (!std::isnan(hist_out[i]) && !std::isnan(s_ref[i].macd_histogram)) {
            float ref_val   = s_ref[i].macd_histogram;
            float tolerance = std::max(std::abs(ref_val) * 0.0001f, 0.001f);

            EXPECT_NEAR(hist_out[i], ref_val, tolerance) << "MACD histogram mismatch at bar " << i;
        }

        compared++;
    }

    EXPECT_GT(compared, 1000) << "Expected at least 1000 bars compared for MACD";
    std::cout << "  MACD(12,26,9): compared " << compared << " bars" << std::endl;
}

// ---------------------------------------------------------------------------
// m. Bollinger(20,2) Cross-Validation Against Python Reference
// ---------------------------------------------------------------------------

TEST_F(FixtureCrossValidation, Bollinger_20_2) {
    const int N = s_bar_count;

    BollingerBands     bb(20, 2.0f);
    std::vector<float> output(N * 3);
    bb.compute(s_market_data.bars(), 0, N, output.data());

    // C++ BollingerBands layout: middle, upper, lower
    const float* middle_out = output.data();
    const float* upper_out  = output.data() + N;
    const float* lower_out  = output.data() + 2 * N;

    const int warmup   = bb.warmup_bars();  // 19
    int       compared = 0;

    for (int i = warmup; i < N && i < static_cast<int>(s_ref.size()); ++i) {
        if (std::isnan(s_ref[i].bb_middle)) continue;

        // Middle band (SMA-20)
        if (!std::isnan(middle_out[i])) {
            float ref_val   = s_ref[i].bb_middle;
            float tolerance = std::abs(ref_val) * 0.0001f;
            if (tolerance < 1e-6f) tolerance = 1e-6f;

            EXPECT_NEAR(middle_out[i], ref_val, tolerance)
                << "Bollinger middle mismatch at bar " << i;
        }

        // Upper band
        if (!std::isnan(upper_out[i]) && !std::isnan(s_ref[i].bb_upper)) {
            float ref_val   = s_ref[i].bb_upper;
            float tolerance = std::abs(ref_val) * 0.0001f;
            if (tolerance < 1e-6f) tolerance = 1e-6f;

            EXPECT_NEAR(upper_out[i], ref_val, tolerance)
                << "Bollinger upper mismatch at bar " << i;
        }

        // Lower band
        if (!std::isnan(lower_out[i]) && !std::isnan(s_ref[i].bb_lower)) {
            float ref_val   = s_ref[i].bb_lower;
            float tolerance = std::abs(ref_val) * 0.0001f;
            if (tolerance < 1e-6f) tolerance = 1e-6f;

            EXPECT_NEAR(lower_out[i], ref_val, tolerance)
                << "Bollinger lower mismatch at bar " << i;
        }

        compared++;
    }

    EXPECT_GT(compared, 1000) << "Expected at least 1000 bars compared for Bollinger";
    std::cout << "  Bollinger(20,2): compared " << compared << " bars" << std::endl;
}
