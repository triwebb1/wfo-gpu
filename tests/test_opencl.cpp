// =============================================================================
// OpenCL Compute Unit Tests
//
// All tests work WITHOUT an OpenCL SDK installed.  They exercise:
//   - CLContext initialization (must report unavailable when no GPU/SDK)
//   - CPU fallback path for indicator computation
//   - CPU fallback path for backtest evaluation
//   - SoA data layout conversion
//   - Warmup computation
//   - Window splitting with overlap
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
#include "cost_model.h"
#include "backtest_engine.h"
#include "metrics.h"
#include "indicators/indicator.h"
#include "indicators/ema_sma_cross.h"
#include "indicators/rsi.h"
#include "indicators/atr.h"
#include "indicators/macd.h"
#include "indicators/bollinger.h"
#include "grids/indicator_grid.h"

#include <filesystem>

#include "compute/cl_context.h"
#include "compute/cl_memory.h"
#include "compute/cl_indicators.h"
#include "compute/cl_backtest.h"
#include "compute/indicator_warmup.h"
#include "test_fixture.h"

// =============================================================================
// Helpers
// =============================================================================

// Generate N synthetic bars with close prices from the given vector.
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

// Generate bars with explicit OHLC per bar
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
// TEST: CLContext -- no GPU available
// =============================================================================

TEST(OpenCL_Context, NoGPU) {
    CLContext ctx;

    // Without an OpenCL SDK or GPU, initialize() should return false
    bool result = ctx.initialize(DeviceType::GPU);

    // In a CI/dev environment without GPU, this will be false.
    // If a GPU is present, it will be true -- both are valid.
    // What we CAN assert: is_available() matches the return value.
    EXPECT_EQ(ctx.is_available(), result);

    if (!result) {
        EXPECT_EQ(ctx.device_name(), "");
        EXPECT_EQ(ctx.device_memory_bytes(), 0u);
        EXPECT_EQ(ctx.max_work_group_size(), 0u);
    }
}

TEST(OpenCL_Context, AutoFallback) {
    CLContext ctx;
    bool      result = ctx.initialize(DeviceType::AUTO);

    // Same as above: the test is that it doesn't crash and state is consistent
    EXPECT_EQ(ctx.is_available(), result);
}

TEST(OpenCL_Context, Release) {
    CLContext ctx;
    ctx.initialize(DeviceType::AUTO);
    ctx.release();

    EXPECT_FALSE(ctx.is_available());
    EXPECT_EQ(ctx.device_name(), "");
    EXPECT_EQ(ctx.device_memory_bytes(), 0u);
}

TEST(OpenCL_Context, MoveSemantics) {
    CLContext ctx1;
    ctx1.initialize(DeviceType::AUTO);
    bool was_available = ctx1.is_available();

    CLContext ctx2 = std::move(ctx1);
    EXPECT_EQ(ctx2.is_available(), was_available);
    EXPECT_FALSE(ctx1.is_available());  // moved-from is empty
}

// =============================================================================
// TEST: SoA layout conversion
// =============================================================================

TEST(OpenCL_Memory, SoAConversion) {
    // Create 5 bars with known values
    std::vector<Bar> bars = {
        make_bar(0, 100.0f, 102.0f, 99.0f, 101.0f, 500),
        make_bar(1, 101.0f, 103.0f, 100.0f, 102.0f, 600),
        make_bar(2, 102.0f, 104.0f, 101.0f, 103.0f, 700),
        make_bar(3, 103.0f, 105.0f, 102.0f, 104.0f, 800),
        make_bar(4, 104.0f, 106.0f, 103.0f, 105.0f, 900),
    };

    SoABarData soa;
    soa.from_bars(bars.data(), 0, 5);

    EXPECT_EQ(soa.num_bars, 5);
    EXPECT_EQ(soa.open.size(), 5u);
    EXPECT_EQ(soa.high.size(), 5u);
    EXPECT_EQ(soa.low.size(), 5u);
    EXPECT_EQ(soa.close.size(), 5u);
    EXPECT_EQ(soa.volume.size(), 5u);

    // Verify each field is correctly separated
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(soa.open[i], bars[i].open);
        EXPECT_FLOAT_EQ(soa.high[i], bars[i].high);
        EXPECT_FLOAT_EQ(soa.low[i], bars[i].low);
        EXPECT_FLOAT_EQ(soa.close[i], bars[i].close);
        EXPECT_EQ(soa.volume[i], bars[i].volume);
    }
}

TEST(OpenCL_Memory, SoAConversionSubrange) {
    std::vector<Bar> bars = {
        make_bar(0, 10.0f, 12.0f, 9.0f, 11.0f, 100),
        make_bar(1, 20.0f, 22.0f, 19.0f, 21.0f, 200),
        make_bar(2, 30.0f, 32.0f, 29.0f, 31.0f, 300),
        make_bar(3, 40.0f, 42.0f, 39.0f, 41.0f, 400),
    };

    // Convert only bars[1..3) (indices 1 and 2)
    SoABarData soa;
    soa.from_bars(bars.data(), 1, 2);

    EXPECT_EQ(soa.num_bars, 2);
    EXPECT_FLOAT_EQ(soa.open[0], 20.0f);
    EXPECT_FLOAT_EQ(soa.open[1], 30.0f);
    EXPECT_FLOAT_EQ(soa.close[0], 21.0f);
    EXPECT_FLOAT_EQ(soa.close[1], 31.0f);
}

TEST(OpenCL_Memory, UploadBarsNoGPU) {
    CLContext ctx;  // not initialized -- no OpenCL
    CLMemory  mem;

    std::vector<Bar> bars = {
        make_bar(0, 100.0f, 102.0f, 99.0f, 101.0f, 500),
        make_bar(1, 101.0f, 103.0f, 100.0f, 102.0f, 600),
    };

    bool ok = mem.upload_bars(ctx, bars.data(), 0, 2);
    EXPECT_TRUE(ok);  // Host-side SoA still works

    const auto& soa = mem.soa_bars();
    EXPECT_EQ(soa.num_bars, 2);
    EXPECT_FLOAT_EQ(soa.close[0], 101.0f);
    EXPECT_FLOAT_EQ(soa.close[1], 102.0f);
}

TEST(OpenCL_Memory, AllocateIndicatorBuffers) {
    CLContext ctx;  // not initialized
    CLMemory  mem;

    auto ptrs = mem.allocate_indicator_buffers(ctx, 3, 100);
    EXPECT_EQ(ptrs.size(), 3u);

    // Each pointer should be non-null and distinct
    for (auto* p : ptrs) {
        EXPECT_NE(p, nullptr);
    }
    EXPECT_NE(ptrs[0], ptrs[1]);
    EXPECT_NE(ptrs[1], ptrs[2]);

    // Write and read back
    for (int i = 0; i < 100; ++i) {
        ptrs[0][i] = static_cast<float>(i);
    }
    EXPECT_FLOAT_EQ(ptrs[0][50], 50.0f);
}

// =============================================================================
// TEST: CPU fallback EMA -- compare CLIndicators vs standalone EMASMACross
// =============================================================================

TEST(OpenCL_Indicators, CPUFallbackEMA) {
    // Generate 200 bars with a known price pattern
    std::vector<float> closes(200);
    for (int i = 0; i < 200; ++i) {
        closes[i] =
            100.0f + static_cast<float>(i) * 0.5f + 5.0f * std::sin(static_cast<float>(i) * 0.1f);
    }
    auto bars = make_bars_from_closes(closes);

    const int N      = static_cast<int>(bars.size());
    const int period = 10;

    // -- Reference: EMASMACross (fast=EMA(10), slow=EMA(20)) --
    EMASMACross        cross(MAType::EMA, period, MAType::EMA, 20);
    std::vector<float> ref_output(static_cast<size_t>(N) * 2);
    cross.compute(bars.data(), 0, N, ref_output.data());
    // Fast MA values are in ref_output[0..N), slow in ref_output[N..2N)

    // -- CL fallback: compute EMA(10) --
    CLContext    ctx;  // not initialized -- CPU fallback
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    IndicatorSpec spec;
    spec.type         = IndicatorType::EMA_SMA_CROSS;
    spec.fast_ma_type = MAType::EMA;
    spec.fast_period  = period;
    spec.slow_ma_type = MAType::EMA;
    spec.slow_period  = 20;
    spec.ma_series    = 0;  // fast MA
    // The fast EMA(10) has its own warmup of 3*10 = 30, matching the reference
    // EMASMACross::compute_ma uses per-MA warmup, not the combined cross warmup.
    spec.warmup_bars = 3 * period;  // EMA warmup = 3 * period

    std::vector<float>         cl_output(N, 0.0f);
    std::vector<float*>        output_ptrs = {cl_output.data()};
    std::vector<IndicatorSpec> specs       = {spec};

    // Extract SoA close prices
    SoABarData soa;
    soa.from_bars(bars.data(), 0, N);

    cl_ind.compute(ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, output_ptrs);

    // Compare: both should be NaN during warmup, then match
    for (int i = 0; i < N; ++i) {
        if (std::isnan(ref_output[i])) {
            EXPECT_TRUE(std::isnan(cl_output[i])) << "Expected NaN at bar " << i;
        } else {
            EXPECT_NEAR(cl_output[i], ref_output[i], 1e-4f) << "Mismatch at bar " << i;
        }
    }
}

TEST(OpenCL_Indicators, CPUFallbackSMA) {
    std::vector<float> closes(100);
    for (int i = 0; i < 100; ++i) {
        closes[i] = 50.0f + static_cast<float>(i) * 0.3f;
    }
    auto bars = make_bars_from_closes(closes);

    const int N      = static_cast<int>(bars.size());
    const int period = 10;

    // Reference: SMA via EMASMACross
    EMASMACross        cross(MAType::SMA, 5, MAType::SMA, period);
    std::vector<float> ref_output(static_cast<size_t>(N) * 2);
    cross.compute(bars.data(), 0, N, ref_output.data());
    // Slow MA (SMA(10)) values in ref_output[N..2N)

    // CL fallback
    CLContext    ctx;
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    IndicatorSpec spec;
    spec.type         = IndicatorType::EMA_SMA_CROSS;
    spec.fast_ma_type = MAType::SMA;
    spec.fast_period  = 5;
    spec.slow_ma_type = MAType::SMA;
    spec.slow_period  = period;
    spec.ma_series    = 1;  // slow MA
    spec.warmup_bars  = cross.warmup_bars();

    std::vector<float>         cl_output(N, 0.0f);
    std::vector<float*>        output_ptrs = {cl_output.data()};
    std::vector<IndicatorSpec> specs       = {spec};

    SoABarData soa;
    soa.from_bars(bars.data(), 0, N);

    cl_ind.compute(ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, output_ptrs);

    for (int i = 0; i < N; ++i) {
        if (std::isnan(ref_output[N + i])) {
            EXPECT_TRUE(std::isnan(cl_output[i])) << "Expected NaN at bar " << i;
        } else {
            EXPECT_NEAR(cl_output[i], ref_output[N + i], 1e-4f) << "Mismatch at bar " << i;
        }
    }
}

// =============================================================================
// TEST: CPU fallback RSI
// =============================================================================

TEST(OpenCL_Indicators, CPUFallbackRSI) {
    // Create a price series with ups and downs
    std::vector<float> closes(150);
    closes[0] = 100.0f;
    for (int i = 1; i < 150; ++i) {
        float delta = (i % 3 == 0) ? -1.5f : 1.0f;
        closes[i]   = closes[i - 1] + delta;
    }
    auto bars = make_bars_from_closes(closes);

    const int N      = static_cast<int>(bars.size());
    const int period = 14;

    // Reference RSI
    RSI                rsi(period);
    std::vector<float> ref_output(N);
    rsi.compute(bars.data(), 0, N, ref_output.data());

    // CL fallback RSI
    CLContext    ctx;
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    IndicatorSpec spec;
    spec.type        = IndicatorType::RSI;
    spec.rsi_period  = period;
    spec.warmup_bars = rsi.warmup_bars();

    std::vector<float>         cl_output(N, 0.0f);
    std::vector<float*>        output_ptrs = {cl_output.data()};
    std::vector<IndicatorSpec> specs       = {spec};

    SoABarData soa;
    soa.from_bars(bars.data(), 0, N);

    cl_ind.compute(ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, output_ptrs);

    for (int i = 0; i < N; ++i) {
        if (std::isnan(ref_output[i])) {
            EXPECT_TRUE(std::isnan(cl_output[i])) << "Expected NaN at bar " << i;
        } else {
            EXPECT_NEAR(cl_output[i], ref_output[i], 1e-3f) << "RSI mismatch at bar " << i;
        }
    }
}

// =============================================================================
// TEST: CPU fallback ATR
// =============================================================================

TEST(OpenCL_Indicators, CPUFallbackATR) {
    // Create bars with varying volatility
    std::vector<std::array<float, 4>> ohlc;
    float                             base = 100.0f;
    for (int i = 0; i < 100; ++i) {
        float range = 1.0f + 0.5f * static_cast<float>(i % 10);
        float o     = base;
        float h     = base + range;
        float l     = base - range * 0.5f;
        float c     = base + range * 0.3f;
        ohlc.push_back({o, h, l, c});
        base = c;
    }
    auto bars = make_bars_ohlc(ohlc);

    const int N      = static_cast<int>(bars.size());
    const int period = 14;

    // Reference ATR
    ATR                atr(period);
    std::vector<float> ref_output(N);
    atr.compute(bars.data(), 0, N, ref_output.data());

    // CL fallback ATR
    CLContext    ctx;
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    IndicatorSpec spec;
    spec.type        = IndicatorType::ATR;
    spec.atr_period  = period;
    spec.warmup_bars = atr.warmup_bars();

    std::vector<float>         cl_output(N, 0.0f);
    std::vector<float*>        output_ptrs = {cl_output.data()};
    std::vector<IndicatorSpec> specs       = {spec};

    SoABarData soa;
    soa.from_bars(bars.data(), 0, N);

    cl_ind.compute(ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, output_ptrs);

    for (int i = 0; i < N; ++i) {
        if (std::isnan(ref_output[i])) {
            EXPECT_TRUE(std::isnan(cl_output[i])) << "Expected NaN at bar " << i;
        } else {
            EXPECT_NEAR(cl_output[i], ref_output[i], 1e-3f) << "ATR mismatch at bar " << i;
        }
    }
}

// =============================================================================
// TEST: CPU fallback Backtest -- compare CLBacktest vs standalone BacktestEngine
// =============================================================================

TEST(OpenCL_Backtest, CPUFallbackMatchesEngine) {
    // Generate bars with a trend + reversal for tradeable signals
    std::vector<float> closes(300);
    closes[0] = 100.0f;
    for (int i = 1; i < 300; ++i) {
        // Create a cycling pattern to generate trades
        float phase = static_cast<float>(i) * 0.05f;
        closes[i]   = 100.0f + 10.0f * std::sin(phase) + static_cast<float>(i) * 0.02f;
    }
    auto      bars = make_bars_from_closes(closes);
    const int N    = static_cast<int>(bars.size());

    // Compute EMA indicators using the original C++ class
    const int          fast_period = 5;
    const int          slow_period = 20;
    EMASMACross        cross(MAType::EMA, fast_period, MAType::EMA, slow_period);
    std::vector<float> ind_output(static_cast<size_t>(N) * 2);
    cross.compute(bars.data(), 0, N, ind_output.data());

    // Create a crossover signal: fast - slow (positive = bullish)
    std::vector<float> signal(N);
    for (int i = 0; i < N; ++i) {
        float fast = ind_output[i];
        float slow = ind_output[N + i];
        if (std::isnan(fast) || std::isnan(slow)) {
            signal[i] = std::numeric_limits<float>::quiet_NaN();
        } else {
            signal[i] = fast - slow;
        }
    }

    // Strategy config: 1 indicator (the crossover signal)
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.3f;
    config.exit_mode            = StrategyConfig::SIGNAL_REVERSAL;

    CostModel cost      = CostModel::es();
    const int entry_lag = 1;

    // Warmup: skip first slow_period * 3 bars
    const int warmup     = cross.warmup_bars();
    const int eval_start = warmup;
    const int eval_end   = N;

    // -- Reference: BacktestEngine directly --
    BacktestEngine     engine;
    const float*       signal_ptr       = signal.data() + eval_start;
    const float* const indicator_ptrs[] = {signal_ptr};

    BacktestResult ref =
        engine.run(bars.data(), eval_start, eval_end, indicator_ptrs, 1, config, cost, entry_lag);

    // -- CL fallback: CLBacktest --
    CLContext  ctx;  // not initialized
    CLBacktest cl_bt;

    const float*       full_signal_ptr = signal.data();
    const float* const full_ptrs[]     = {full_signal_ptr};

    std::vector<StrategyConfig> strategies = {config};
    std::vector<BacktestResult> results;

    cl_bt.evaluate(ctx, bars.data(),
                   0,  // bar_offset (SoA starts at bar 0)
                   full_ptrs,
                   1,  // num_indicator_series
                   N,  // num_bars in SoA
                   eval_start, eval_end, strategies, cost, entry_lag, results);

    ASSERT_EQ(results.size(), 1u);

    // Compare key metrics
    EXPECT_EQ(results[0].total_trades, ref.total_trades) << "Trade count mismatch";
    EXPECT_NEAR(results[0].total_pnl, ref.total_pnl, 0.01f) << "Total PnL mismatch";
    EXPECT_NEAR(results[0].win_rate, ref.win_rate, 1e-4f) << "Win rate mismatch";
    EXPECT_NEAR(results[0].max_drawdown, ref.max_drawdown, 0.01f) << "Max drawdown mismatch";
    EXPECT_NEAR(results[0].gross_wins, ref.gross_wins, 0.01f) << "Gross wins mismatch";
    EXPECT_NEAR(results[0].gross_losses, ref.gross_losses, 0.01f) << "Gross losses mismatch";
}

// =============================================================================
// TEST: Warmup computation
// =============================================================================

TEST(OpenCL_WindowSplit, WarmupCalcEMA) {
    // EMA(50): warmup = 3 * 50 = 150, with 2x safety = 300
    IndicatorParams params;
    params.type         = IndicatorType::EMA_SMA_CROSS;
    params.fast_ma_type = MAType::EMA;
    params.fast_period  = 10;
    params.slow_ma_type = MAType::EMA;
    params.slow_period  = 50;

    std::vector<IndicatorParams> list = {params};
    int                          wu   = max_indicator_warmup(list);

    // max(3*10, 3*50) = 150, *2 = 300
    EXPECT_EQ(wu, 300);
}

TEST(OpenCL_WindowSplit, WarmupCalcSMA) {
    IndicatorParams params;
    params.type         = IndicatorType::EMA_SMA_CROSS;
    params.fast_ma_type = MAType::SMA;
    params.fast_period  = 10;
    params.slow_ma_type = MAType::SMA;
    params.slow_period  = 50;

    std::vector<IndicatorParams> list = {params};
    int                          wu   = max_indicator_warmup(list);

    // max(10-1, 50-1) = 49, *2 = 98
    EXPECT_EQ(wu, 98);
}

TEST(OpenCL_WindowSplit, WarmupCalcRSI) {
    IndicatorParams params;
    params.type       = IndicatorType::RSI;
    params.rsi_period = 14;

    std::vector<IndicatorParams> list = {params};
    int                          wu   = max_indicator_warmup(list);

    // 3 * 14 = 42, *2 = 84
    EXPECT_EQ(wu, 84);
}

TEST(OpenCL_WindowSplit, WarmupCalcATR) {
    IndicatorParams params;
    params.type             = IndicatorType::ATR;
    params.atr_short_period = 5;
    params.atr_long_period  = 20;

    std::vector<IndicatorParams> list = {params};
    int                          wu   = max_indicator_warmup(list);

    // max(3*5, 3*20) = 60, *2 = 120
    EXPECT_EQ(wu, 120);
}

TEST(OpenCL_WindowSplit, WarmupMaxAcrossCatalog) {
    // Multiple indicators -- max should win
    IndicatorParams ema_params;
    ema_params.type         = IndicatorType::EMA_SMA_CROSS;
    ema_params.fast_ma_type = MAType::EMA;
    ema_params.fast_period  = 10;
    ema_params.slow_ma_type = MAType::EMA;
    ema_params.slow_period  = 200;  // large period

    IndicatorParams rsi_params;
    rsi_params.type       = IndicatorType::RSI;
    rsi_params.rsi_period = 14;

    IndicatorParams atr_params;
    atr_params.type             = IndicatorType::ATR;
    atr_params.atr_short_period = 5;
    atr_params.atr_long_period  = 20;

    std::vector<IndicatorParams> list = {ema_params, rsi_params, atr_params};
    int                          wu   = max_indicator_warmup(list);

    // EMA(200): 3*200 = 600, *2 = 1200  (largest)
    // RSI(14):  3*14  = 42,  *2 = 84
    // ATR(20):  3*20  = 60,  *2 = 120
    EXPECT_EQ(wu, 1200);
}

TEST(OpenCL_WindowSplit, WarmupFromGrid) {
    // Test compute_max_warmup with an IndicatorGrid
    IndicatorGrid grid;
    int           wu = max_indicator_warmup(grid);

    // The grid contains all parameter combos; the max warmup should be > 0
    EXPECT_GT(wu, 0);

    // It should equal the max across all entries computed individually
    int max_individual = 0;
    for (int i = 0; i < grid.size(); ++i) {
        std::vector<IndicatorParams> single = {grid.params(i)};
        int                          w      = max_indicator_warmup(single);
        max_individual                      = std::max(max_individual, w);
    }
    EXPECT_EQ(wu, max_individual);
}

// =============================================================================
// TEST: Anchored warmup scaling -- warmup scaled by TF ratio for anchored modes
// =============================================================================

TEST(OpenCL_WindowSplit, AnchoredWarmupScaling) {
    // EMA(200) on ANCHORED_5M: native warmup = 3*200*2 = 1200 (with safety)
    // TF ratio for 5m = 300/30 = 10
    // Scaled warmup should be 1200 * 10 = 12000
    IndicatorSpec spec;
    spec.type         = IndicatorType::EMA_SMA_CROSS;
    spec.fast_ma_type = MAType::EMA;
    spec.fast_period  = 50;
    spec.slow_ma_type = MAType::EMA;
    spec.slow_period  = 200;
    spec.window_mode  = WindowMode::ANCHORED_5M;

    std::vector<IndicatorSpec> specs  = {spec};
    int                        warmup = max_indicator_warmup(specs);
    // Native: max(3*50, 3*200) = 600, with 2x safety = 1200
    // Scaled by 5m ratio (10): 12000
    EXPECT_EQ(warmup, 12000);

    // Same indicator in ROLLING mode should be 1200
    spec.window_mode = WindowMode::ROLLING;
    specs            = {spec};
    warmup           = max_indicator_warmup(specs);
    EXPECT_EQ(warmup, 1200);

    // RSI(14) on ANCHORED_1H: native = 3*14*2 = 84, ratio = 3600/30 = 120
    // Scaled = 84 * 120 = 10080
    IndicatorSpec rsi_spec;
    rsi_spec.type        = IndicatorType::RSI;
    rsi_spec.rsi_period  = 14;
    rsi_spec.window_mode = WindowMode::ANCHORED_1H;
    specs                = {rsi_spec};
    warmup               = max_indicator_warmup(specs);
    EXPECT_EQ(warmup, 10080);
}

// =============================================================================
// TEST: IndicatorSpec::from_params
// =============================================================================

TEST(OpenCL_Indicators, SpecFromParams_EMA) {
    IndicatorParams params;
    params.type         = IndicatorType::EMA_SMA_CROSS;
    params.fast_ma_type = MAType::EMA;
    params.fast_period  = 10;
    params.slow_ma_type = MAType::EMA;
    params.slow_period  = 50;

    auto spec0 = IndicatorSpec::from_params(params, 0);
    EXPECT_EQ(spec0.type, IndicatorType::EMA_SMA_CROSS);
    EXPECT_EQ(spec0.ma_series, 0);
    EXPECT_EQ(spec0.fast_period, 10);
    EXPECT_EQ(spec0.warmup_bars, 150);  // max(3*10, 3*50) = 150

    auto spec1 = IndicatorSpec::from_params(params, 1);
    EXPECT_EQ(spec1.ma_series, 1);
    EXPECT_EQ(spec1.slow_period, 50);
}

TEST(OpenCL_Indicators, SpecFromParams_RSI) {
    IndicatorParams params;
    params.type       = IndicatorType::RSI;
    params.rsi_period = 14;

    auto spec = IndicatorSpec::from_params(params);
    EXPECT_EQ(spec.type, IndicatorType::RSI);
    EXPECT_EQ(spec.rsi_period, 14);
    EXPECT_EQ(spec.warmup_bars, 42);  // 3 * 14
}

TEST(OpenCL_Indicators, SpecFromParams_ATR) {
    IndicatorParams params;
    params.type             = IndicatorType::ATR;
    params.atr_short_period = 5;
    params.atr_long_period  = 20;

    auto spec0 = IndicatorSpec::from_params(params, 0);
    EXPECT_EQ(spec0.atr_period, 5);
    EXPECT_EQ(spec0.warmup_bars, 15);  // 3 * 5

    auto spec1 = IndicatorSpec::from_params(params, 1);
    EXPECT_EQ(spec1.atr_period, 20);
    EXPECT_EQ(spec1.warmup_bars, 60);  // 3 * 20
}

// =============================================================================
// TEST: Multiple indicators computed in one batch
// =============================================================================

TEST(OpenCL_Indicators, MultiIndicatorBatch) {
    // Compute EMA, RSI, and ATR in a single compute() call
    std::vector<float> closes(200);
    closes[0] = 100.0f;
    for (int i = 1; i < 200; ++i) {
        closes[i] = closes[i - 1] + (i % 3 == 0 ? -1.0f : 0.8f);
    }
    auto      bars = make_bars_from_closes(closes);
    const int N    = static_cast<int>(bars.size());

    SoABarData soa;
    soa.from_bars(bars.data(), 0, N);

    CLContext    ctx;
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    // Three different indicators
    IndicatorSpec ema_spec;
    ema_spec.type         = IndicatorType::EMA_SMA_CROSS;
    ema_spec.fast_ma_type = MAType::EMA;
    ema_spec.fast_period  = 10;
    ema_spec.slow_ma_type = MAType::EMA;
    ema_spec.slow_period  = 20;
    ema_spec.ma_series    = 0;
    ema_spec.warmup_bars  = 60;

    IndicatorSpec rsi_spec;
    rsi_spec.type        = IndicatorType::RSI;
    rsi_spec.rsi_period  = 14;
    rsi_spec.warmup_bars = 42;

    IndicatorSpec atr_spec;
    atr_spec.type        = IndicatorType::ATR;
    atr_spec.atr_period  = 14;
    atr_spec.warmup_bars = 42;

    std::vector<IndicatorSpec> specs = {ema_spec, rsi_spec, atr_spec};

    std::vector<float>  out_ema(N), out_rsi(N), out_atr(N);
    std::vector<float*> outputs = {out_ema.data(), out_rsi.data(), out_atr.data()};

    cl_ind.compute(ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, outputs);

    // Verify EMA output has valid values after warmup
    int ema_valid = 0;
    for (int i = 0; i < N; ++i) {
        if (!std::isnan(out_ema[i])) ema_valid++;
    }
    EXPECT_GT(ema_valid, 0);

    // Verify RSI output is in [0, 100] range where valid
    for (int i = 0; i < N; ++i) {
        if (!std::isnan(out_rsi[i])) {
            EXPECT_GE(out_rsi[i], 0.0f);
            EXPECT_LE(out_rsi[i], 100.0f);
        }
    }

    // Verify ATR output is positive where valid
    for (int i = 0; i < N; ++i) {
        if (!std::isnan(out_atr[i])) {
            EXPECT_GT(out_atr[i], 0.0f);
        }
    }
}

// =============================================================================
// GPU vs CPU Backtest Kernel Correctness Tests
//
// These tests compare GPU kernel results against CPU BacktestEngine results.
// When OpenCL is not available, GPU tests are skipped via GTEST_SKIP().
// The CPU fallback test always runs.
// =============================================================================

// Helper: create a CLContext with OpenCL initialized (AUTO = tries GPU, then CPU).
// Returns true if OpenCL is available.
static bool try_init_opencl(CLContext& ctx) {
    return ctx.initialize(DeviceType::AUTO);
}

// Guard: when the caller expects the GPU path (initialized context), require
// the backtest kernel to build. Without this, a build failure (e.g., embedded
// kernel source corrupt, driver regression) would cause CLBacktest::evaluate()
// to silently fall back to CPU, and every "GPU matches CPU" assertion would
// trivially hold because both sides ran on CPU. Callers that pass an
// uninitialized context are intentionally exercising the CPU fallback path
// (the `cpu_ctx` half of each parity pair) and are left alone.
static void assert_gpu_kernel_ready(CLContext& ctx, CLBacktest& cl_bt) {
#ifdef HAS_OPENCL
    if (!ctx.is_available()) return;  // intentional CPU fallback.
    if (!cl_bt.build_kernel(ctx)) {
        ADD_FAILURE() << "CLBacktest::build_kernel() failed -- GPU parity test "
                         "would degenerate to CPU-vs-CPU (silent fallback).";
    }
#else
    (void)ctx;
    (void)cl_bt;
#endif
}

// Helper: run a single strategy through CLBacktest::evaluate() with the given context.
// Returns the BacktestResult.
static BacktestResult run_clbacktest(CLContext& ctx, CLBacktest& cl_bt,
                                     const std::vector<Bar>&          bars,
                                     const std::vector<const float*>& indicator_ptrs,
                                     int num_indicator_series, int eval_start, int eval_end,
                                     const std::vector<StrategyConfig>& strategies,
                                     const CostModel& cost, int entry_lag) {
    assert_gpu_kernel_ready(ctx, cl_bt);
    std::vector<BacktestResult> results;
    cl_bt.evaluate(ctx, bars.data(),
                   0,  // bar_offset
                   indicator_ptrs.data(), num_indicator_series, static_cast<int>(bars.size()),
                   eval_start, eval_end, strategies, cost, entry_lag, results);
    return results.empty() ? BacktestResult{} : results[0];
}

// Helper: run multiple strategies through CLBacktest::evaluate() with the given context.
static std::vector<BacktestResult> run_clbacktest_multi(
    CLContext& ctx, CLBacktest& cl_bt, const std::vector<Bar>& bars,
    const std::vector<const float*>& indicator_ptrs, int num_indicator_series, int eval_start,
    int eval_end, const std::vector<StrategyConfig>& strategies, const CostModel& cost,
    int entry_lag) {
    assert_gpu_kernel_ready(ctx, cl_bt);
    std::vector<BacktestResult> results;
    cl_bt.evaluate(ctx, bars.data(), 0, indicator_ptrs.data(), num_indicator_series,
                   static_cast<int>(bars.size()), eval_start, eval_end, strategies, cost, entry_lag,
                   results);
    return results;
}

// Helper: compare two BacktestResults for approximate equality
static void expect_results_match(const BacktestResult& gpu, const BacktestResult& cpu,
                                 const std::string& label) {
    EXPECT_EQ(gpu.total_trades, cpu.total_trades) << label << ": trade count mismatch";
    EXPECT_NEAR(gpu.total_pnl, cpu.total_pnl, 0.01f) << label << ": total PnL mismatch";
    EXPECT_NEAR(gpu.win_rate, cpu.win_rate, 1e-4f) << label << ": win rate mismatch";
    EXPECT_NEAR(gpu.gross_wins, cpu.gross_wins, 0.01f) << label << ": gross wins mismatch";
    EXPECT_NEAR(gpu.gross_losses, cpu.gross_losses, 0.01f) << label << ": gross losses mismatch";
    EXPECT_NEAR(gpu.max_drawdown, cpu.max_drawdown, 0.01f) << label << ": max drawdown mismatch";
    EXPECT_NEAR(gpu.max_live_drawdown, cpu.max_live_drawdown, 0.01f)
        << label << ": max live drawdown mismatch";
}

// =============================================================================
// TEST: GPU matches CPU -- basic long trade with trending price
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_BasicLong) {
    // Initialize OpenCL
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create ~50 bars with a clear uptrend
    // Bars rise from 100 to ~125, creating a strong positive signal
    const int          N = 50;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = 100.0f + static_cast<float>(i) * 0.5f;
    }
    auto bars = make_bars_from_closes(closes);

    // One constant indicator series: value = 1.0 everywhere (always long signal)
    std::vector<float>        signal(N, 1.0f);
    const float*              signal_ptr     = signal.data();
    std::vector<const float*> indicator_ptrs = {signal_ptr};

    // Strategy: 1 indicator, WEIGHTED_SUM, threshold 0.5 -> signal(1.0) > 0.5 = long
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 10;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies = {config};

    // GPU result
    CLBacktest     gpu_bt;
    BacktestResult gpu_result = run_clbacktest(gpu_ctx, gpu_bt, bars, indicator_ptrs, 1, eval_start,
                                               eval_end, strategies, cost, entry_lag);

    // CPU result (uninitialized context forces CPU fallback)
    CLContext      cpu_ctx;  // not initialized
    CLBacktest     cpu_bt;
    BacktestResult cpu_result = run_clbacktest(cpu_ctx, cpu_bt, bars, indicator_ptrs, 1, eval_start,
                                               eval_end, strategies, cost, entry_lag);

    // Verify we actually got some trades
    EXPECT_GT(cpu_result.total_trades, 0) << "CPU should produce trades";

    // THE critical comparison: GPU must match CPU exactly
    expect_results_match(gpu_result, cpu_result, "BasicLong");
}

// =============================================================================
// TEST: GPU matches CPU -- short trade with negative signal
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_ShortTrade) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 60 bars with a downtrend
    const int          N = 60;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = 200.0f - static_cast<float>(i) * 0.4f;
    }
    auto bars = make_bars_from_closes(closes);

    // Constant negative signal: -1.0 everywhere -> always short
    std::vector<float>        signal(N, -1.0f);
    const float*              signal_ptr     = signal.data();
    std::vector<const float*> indicator_ptrs = {signal_ptr};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 8;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies = {config};

    // GPU
    CLBacktest     gpu_bt;
    BacktestResult gpu_result = run_clbacktest(gpu_ctx, gpu_bt, bars, indicator_ptrs, 1, eval_start,
                                               eval_end, strategies, cost, entry_lag);

    // CPU
    CLContext      cpu_ctx;
    CLBacktest     cpu_bt;
    BacktestResult cpu_result = run_clbacktest(cpu_ctx, cpu_bt, bars, indicator_ptrs, 1, eval_start,
                                               eval_end, strategies, cost, entry_lag);

    EXPECT_GT(cpu_result.total_trades, 0) << "CPU should produce short trades";

    // Short trades on a downtrend should be profitable (before costs)
    // PnL sign verification: gross_wins should be > 0 for shorts on downtrend
    // (net PnL may be negative due to high ES costs on small moves)

    expect_results_match(gpu_result, cpu_result, "ShortTrade");
}

// =============================================================================
// TEST: GPU matches CPU -- ATR-based exit modes
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_ExitModes) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 80 bars with a known price path and ATR series.
    // Price rises then falls, so both longs and shorts get tested.
    const int                         N = 80;
    std::vector<std::array<float, 4>> ohlc(N);
    for (int i = 0; i < N; ++i) {
        float base;
        if (i < 40) {
            base = 100.0f + static_cast<float>(i) * 0.5f;  // up-trend
        } else {
            base = 120.0f - static_cast<float>(i - 40) * 0.5f;  // down-trend
        }
        float range = 2.0f;
        ohlc[i]     = {base, base + range, base - range, base + 0.25f};
    }
    auto bars = make_bars_ohlc(ohlc);

    // Two indicator series:
    //   [0] = signal: +1.0 for first 40 bars (long), -1.0 for last 40 (short)
    //   [1] = ATR: constant 2.0
    std::vector<float> signal_series(N);
    std::vector<float> atr_series(N, 2.0f);
    for (int i = 0; i < N; ++i) {
        signal_series[i] = (i < 40) ? 1.0f : -1.0f;
    }
    const float*              ptrs[2] = {signal_series.data(), atr_series.data()};
    std::vector<const float*> indicator_ptrs(ptrs, ptrs + 2);

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    // Test each ATR-based exit mode
    StrategyConfig::ExitMode modes[] = {StrategyConfig::TRAILING_STOP, StrategyConfig::MAX_LOSS_ATR,
                                        StrategyConfig::TAKE_PROFIT_ATR, StrategyConfig::TP_SL_ATR};

    const char* mode_names[] = {"TRAILING_STOP", "MAX_LOSS_ATR", "TAKE_PROFIT_ATR", "TP_SL_ATR"};

    for (int m = 0; m < 4; ++m) {
        StrategyConfig config;
        config.num_indicators       = 1;
        config.indicator_indices[0] = 0;
        config.weights[0]           = 1.0f;
        config.aggregation          = StrategyConfig::WEIGHTED_SUM;
        config.entry_threshold      = 0.5f;
        config.exit_mode            = modes[m];
        config.trailing_atr_mult    = 1.5f;
        config.max_loss_atr_mult    = 2.0f;
        config.tp_atr_mult          = 1.5f;
        config.sl_atr_mult          = 1.0f;
        config.atr_series_index     = 1;  // index into indicator_ptrs

        std::vector<StrategyConfig> strategies = {config};

        // GPU
        CLBacktest     gpu_bt;
        BacktestResult gpu_result =
            run_clbacktest(gpu_ctx, gpu_bt, bars, indicator_ptrs, 2, eval_start, eval_end,
                           strategies, cost, entry_lag);

        // CPU
        CLContext      cpu_ctx;
        CLBacktest     cpu_bt;
        BacktestResult cpu_result =
            run_clbacktest(cpu_ctx, cpu_bt, bars, indicator_ptrs, 2, eval_start, eval_end,
                           strategies, cost, entry_lag);

        std::string label = std::string("ExitMode_") + mode_names[m];
        expect_results_match(gpu_result, cpu_result, label);
    }
}

// =============================================================================
// TEST: GPU matches CPU -- 100 strategies dispatched simultaneously
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_MultiStrategy) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 200 bars with a cyclic price pattern to generate varied trades
    const int          N = 200;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        float phase = static_cast<float>(i) * 0.08f;
        closes[i]   = 100.0f + 8.0f * std::sin(phase) + static_cast<float>(i) * 0.02f;
    }
    auto bars = make_bars_from_closes(closes);

    // Two indicator series:
    //   [0] = smoothed signal (sine-based, varies between -1 and +1)
    //   [1] = ATR (constant 1.5)
    std::vector<float> signal_series(N);
    std::vector<float> atr_series(N, 1.5f);
    for (int i = 0; i < N; ++i) {
        signal_series[i] = std::sin(static_cast<float>(i) * 0.08f);
    }
    const float*              ptrs[2] = {signal_series.data(), atr_series.data()};
    std::vector<const float*> indicator_ptrs(ptrs, ptrs + 2);

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    // Create 100 strategies with varying thresholds and exit modes
    const int                   NUM_STRATEGIES = 100;
    std::vector<StrategyConfig> strategies(NUM_STRATEGIES);
    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        StrategyConfig& cfg      = strategies[s];
        cfg.num_indicators       = 1;
        cfg.indicator_indices[0] = 0;
        cfg.weights[0]           = 1.0f;
        cfg.aggregation          = StrategyConfig::WEIGHTED_SUM;

        // Vary threshold from 0.1 to 0.9
        cfg.entry_threshold = 0.1f + 0.8f * (static_cast<float>(s) / 99.0f);

        // Rotate through the exit modes
        int mode_idx = s % 5;
        if (mode_idx == 0) {
            cfg.exit_mode         = StrategyConfig::BAR_TIMEOUT;
            cfg.exit_timeout_bars = 5 + (s % 20);
        } else if (mode_idx == 1) {
            cfg.exit_mode = StrategyConfig::SIGNAL_REVERSAL;
        } else if (mode_idx == 2) {
            cfg.exit_mode         = StrategyConfig::TRAILING_STOP;
            cfg.trailing_atr_mult = 1.0f + static_cast<float>(s % 10) * 0.3f;
            cfg.atr_series_index  = 1;
        } else if (mode_idx == 3) {
            cfg.exit_mode        = StrategyConfig::TAKE_PROFIT_ATR;
            cfg.tp_atr_mult      = 1.0f + static_cast<float>(s % 8) * 0.5f;
            cfg.atr_series_index = 1;
        } else {
            cfg.exit_mode        = StrategyConfig::TP_SL_ATR;
            cfg.tp_atr_mult      = 2.0f;
            cfg.sl_atr_mult      = 1.0f + static_cast<float>(s % 5) * 0.2f;
            cfg.atr_series_index = 1;
        }
    }

    // GPU: all 100 strategies at once
    CLBacktest gpu_bt;
    auto gpu_results = run_clbacktest_multi(gpu_ctx, gpu_bt, bars, indicator_ptrs, 2, eval_start,
                                            eval_end, strategies, cost, entry_lag);

    // CPU: all 100 strategies
    CLContext  cpu_ctx;
    CLBacktest cpu_bt;
    auto cpu_results = run_clbacktest_multi(cpu_ctx, cpu_bt, bars, indicator_ptrs, 2, eval_start,
                                            eval_end, strategies, cost, entry_lag);

    ASSERT_EQ(gpu_results.size(), static_cast<size_t>(NUM_STRATEGIES));
    ASSERT_EQ(cpu_results.size(), static_cast<size_t>(NUM_STRATEGIES));

    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        std::string label = "Strategy_" + std::to_string(s);
        expect_results_match(gpu_results[s], cpu_results[s], label);
    }
}

// =============================================================================
// TEST: GPU matches CPU -- trading hours filtering (RTH only)
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_TradingHours) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 100 bars. Assign time-of-day values such that:
    //   bars 0-29:  08:00 ET (28800s) -- pre-market, RTH should suppress
    //   bars 30-69: 10:00 ET (36000s) -- within RTH 09:30-16:00, entries allowed
    //   bars 70-99: 17:00 ET (61200s) -- post-market, RTH should suppress
    const int          N = 100;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = 100.0f + static_cast<float>(i) * 0.3f;
    }
    auto bars = make_bars_from_closes(closes);

    // Build time-of-day array
    std::vector<int> bar_tod(N);
    for (int i = 0; i < N; ++i) {
        if (i < 30)
            bar_tod[i] = 28800;  // 08:00 ET
        else if (i < 70)
            bar_tod[i] = 36000;  // 10:00 ET (RTH)
        else
            bar_tod[i] = 61200;  // 17:00 ET
    }
    const SessionModel model = SessionModel::us_futures();
    std::vector<int>   bar_flags(N);
    for (int i = 0; i < N; ++i) bar_flags[i] = model.hours_flags(bar_tod[i]);

    // Constant positive signal -> wants to go long always
    std::vector<float>        signal(N, 1.0f);
    const float*              signal_ptr     = signal.data();
    std::vector<const float*> indicator_ptrs = {signal_ptr};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 10;
    config.trading_hours        = StrategyConfig::RTH;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies = {config};

    // GPU: set per-bar session flags
    CLBacktest gpu_bt;
    gpu_bt.set_bar_flags(bar_flags);
    BacktestResult gpu_result = run_clbacktest(gpu_ctx, gpu_bt, bars, indicator_ptrs, 1, eval_start,
                                               eval_end, strategies, cost, entry_lag);

    // CPU: for the CPU path, trading hours filtering is done via SessionInfo
    // in BacktestEngine::run_impl(). The CLBacktest CPU fallback does NOT
    // use set_bar_flags() -- it uses BacktestEngine which needs SessionInfo.
    // However, CLBacktest::evaluate_cpu() calls BacktestEngine::run() without
    // sessions, so trading hours filtering on the CPU side depends on SessionInfo
    // being provided.
    //
    // Since CLBacktest::evaluate_cpu() does NOT pass SessionInfo to the engine,
    // the CPU fallback will NOT filter by trading hours. The GPU kernel, however,
    // DOES filter using the flags when has_bar_flags=1.
    //
    // To get a valid comparison, we need both paths to use the same filtering.
    // The simplest approach: also run GPU without flags (ETH mode) as baseline,
    // then verify the RTH-filtered result has fewer trades.
    CLBacktest gpu_bt_eth;
    // No set_bar_flags -> ETH mode (no filtering)
    BacktestResult gpu_eth_result =
        run_clbacktest(gpu_ctx, gpu_bt_eth, bars, indicator_ptrs, 1, eval_start, eval_end,
                       strategies, cost, entry_lag);

    // The RTH-filtered GPU should have fewer or equal trades than ETH GPU
    EXPECT_LE(gpu_result.total_trades, gpu_eth_result.total_trades)
        << "RTH filtering should suppress some entries";

    // Also run CPU without filtering (ETH) and compare with GPU ETH
    CLContext      cpu_ctx;
    CLBacktest     cpu_bt;
    BacktestResult cpu_result = run_clbacktest(cpu_ctx, cpu_bt, bars, indicator_ptrs, 1, eval_start,
                                               eval_end, strategies, cost, entry_lag);

    // GPU ETH vs CPU ETH should match (both unfiltered)
    expect_results_match(gpu_eth_result, cpu_result, "TradingHours_ETH");

    // Verify RTH mode actually produced trades (entries happen in bars 30-69)
    EXPECT_GT(gpu_result.total_trades, 0)
        << "RTH mode should still produce trades during 10:00 ET window";
}

// =============================================================================
// TEST: Fallback when no OpenCL -- evaluate() still works via CPU
// =============================================================================

TEST(CLBacktest, FallbackWhenNoOpenCL) {
    // Create a CLContext that is NOT initialized -- simulates no OpenCL available
    CLContext ctx;  // default-constructed, NOT initialized
    ASSERT_FALSE(ctx.is_available()) << "Context should not be available";

    // Create bars with a clear pattern
    const int          N = 40;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = 100.0f + static_cast<float>(i) * 0.3f;
    }
    auto bars = make_bars_from_closes(closes);

    // Constant positive signal
    std::vector<float>        signal(N, 1.0f);
    const float*              signal_ptr     = signal.data();
    std::vector<const float*> indicator_ptrs = {signal_ptr};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 8;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies = {config};

    // Call evaluate() with unavailable context -- must not crash, must produce results
    CLBacktest     cl_bt;
    BacktestResult result = run_clbacktest(ctx, cl_bt, bars, indicator_ptrs, 1, eval_start,
                                           eval_end, strategies, cost, entry_lag);

    // Verify we got valid results from CPU fallback
    EXPECT_GT(result.total_trades, 0) << "CPU fallback should produce trades";
    EXPECT_NE(result.total_pnl, 0.0f) << "Total PnL should not be zero with trending data";

    // Cross-validate against BacktestEngine directly
    BacktestEngine     engine;
    const float*       offset_ptr    = signal.data() + eval_start;
    const float* const engine_ptrs[] = {offset_ptr};
    BacktestResult     ref =
        engine.run(bars.data(), eval_start, eval_end, engine_ptrs, 1, config, cost, entry_lag);

    EXPECT_EQ(result.total_trades, ref.total_trades)
        << "Fallback trade count should match BacktestEngine";
    EXPECT_NEAR(result.total_pnl, ref.total_pnl, 0.01f)
        << "Fallback PnL should match BacktestEngine";
    EXPECT_NEAR(result.win_rate, ref.win_rate, 1e-4f)
        << "Fallback win rate should match BacktestEngine";
}

// =============================================================================
// TEST: GPU matches CPU -- entry_lag = 2
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_Lag2) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 80 bars with a cycling pattern
    const int          N = 80;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        float phase = static_cast<float>(i) * 0.1f;
        closes[i]   = 100.0f + 5.0f * std::sin(phase) + static_cast<float>(i) * 0.03f;
    }
    auto bars = make_bars_from_closes(closes);

    // Signal that oscillates: positive in first half, negative in second
    std::vector<float> signal(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = std::sin(static_cast<float>(i) * 0.1f);
    }
    const float*              signal_ptr     = signal.data();
    std::vector<const float*> indicator_ptrs = {signal_ptr};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.3f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 10;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 2;  // KEY: lag-2
    const int eval_start = 0;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies = {config};

    // GPU with lag=2
    CLBacktest     gpu_bt;
    BacktestResult gpu_result = run_clbacktest(gpu_ctx, gpu_bt, bars, indicator_ptrs, 1, eval_start,
                                               eval_end, strategies, cost, entry_lag);

    // CPU with lag=2
    CLContext      cpu_ctx;
    CLBacktest     cpu_bt;
    BacktestResult cpu_result = run_clbacktest(cpu_ctx, cpu_bt, bars, indicator_ptrs, 1, eval_start,
                                               eval_end, strategies, cost, entry_lag);

    EXPECT_GT(cpu_result.total_trades, 0) << "Lag-2 CPU should produce trades";

    expect_results_match(gpu_result, cpu_result, "Lag2");

    // Also verify that lag-2 produces fewer or equal trades than lag-1
    // (lag-2 has stricter timing, may miss some entries near end of range)
    CLBacktest     gpu_bt_lag1;
    BacktestResult gpu_lag1 =
        run_clbacktest(gpu_ctx, gpu_bt_lag1, bars, indicator_ptrs, 1, eval_start, eval_end,
                       strategies, cost, /*entry_lag=*/1);

    EXPECT_LE(gpu_result.total_trades, gpu_lag1.total_trades)
        << "Lag-2 should have fewer or equal trades than lag-1";
}

// =============================================================================
// Helper: run CLBacktest::evaluate() with SessionInfo
// =============================================================================
static BacktestResult run_clbacktest_with_sessions(
    CLContext& ctx, CLBacktest& cl_bt, const std::vector<Bar>& bars,
    const std::vector<const float*>& indicator_ptrs, int num_indicator_series, int eval_start,
    int eval_end, const std::vector<StrategyConfig>& strategies, const CostModel& cost,
    int entry_lag, const SessionInfo* sessions) {
    assert_gpu_kernel_ready(ctx, cl_bt);
    std::vector<BacktestResult> results;
    cl_bt.evaluate(ctx, bars.data(),
                   0,  // bar_offset
                   indicator_ptrs.data(), num_indicator_series, static_cast<int>(bars.size()),
                   eval_start, eval_end, strategies, cost, entry_lag, results, EarlyTermination{},
                   sessions);
    return results.empty() ? BacktestResult{} : results[0];
}

// =============================================================================
// TEST: GPU matches CPU -- session-aware backtesting (halt, trading hours,
//       early close)
//
// Verifies that the GPU kernel produces identical results to the CPU
// BacktestEngine when SessionInfo is provided with halt flags, trading
// hours filtering, and session close force-exit.
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_Sessions) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 120 bars with a clear uptrend
    const int          N = 120;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = 100.0f + static_cast<float>(i) * 0.4f;
    }
    auto bars = make_bars_from_closes(closes);

    // Build SessionInfo array with realistic session data:
    //   bars  0-29:  pre-market  08:00 ET (28800s), normal close 17:00 (61200s)
    //   bars 30-39:  halted      10:00 ET (36000s), halt_flag=1
    //   bars 40-89:  RTH         10:30 ET (37800s), normal close
    //   bars 90-99:  near close  16:59 ET (61140s) -- within 30s of session close
    //   bars 100-119: post-close 17:05 ET (61500s)
    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].halt_flag         = 0;
        sessions[i].early_close       = 0;
        sessions[i].session_close_sec = 61200;  // 17:00 ET

        if (i < 30) {
            sessions[i].time_of_day_sec = 28800;  // 08:00 ET
        } else if (i < 40) {
            sessions[i].time_of_day_sec = 36000;  // 10:00 ET
            sessions[i].halt_flag       = 1;      // halted
        } else if (i < 90) {
            sessions[i].time_of_day_sec = 37800;  // 10:30 ET (RTH)
        } else if (i < 100) {
            sessions[i].time_of_day_sec = 61140 + (i - 90) * 3;  // 16:59:00+ ET
            // These are within 30s of session close (61200), should force exit
        } else {
            sessions[i].time_of_day_sec = 61500;  // 17:05 ET (post-close)
        }
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    // Constant positive signal -> always wants to go long
    std::vector<float>        signal(N, 1.0f);
    const float*              signal_ptr     = signal.data();
    std::vector<const float*> indicator_ptrs = {signal_ptr};

    // RTH strategy: entries only allowed 09:30-16:00 ET
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 15;
    config.trading_hours        = StrategyConfig::RTH;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies = {config};

    // GPU result with sessions
    CLBacktest     gpu_bt;
    BacktestResult gpu_result =
        run_clbacktest_with_sessions(gpu_ctx, gpu_bt, bars, indicator_ptrs, 1, eval_start, eval_end,
                                     strategies, cost, entry_lag, sessions.data());

    // CPU result with sessions (uninitialized context forces CPU fallback)
    CLContext      cpu_ctx;  // not initialized
    CLBacktest     cpu_bt;
    BacktestResult cpu_result =
        run_clbacktest_with_sessions(cpu_ctx, cpu_bt, bars, indicator_ptrs, 1, eval_start, eval_end,
                                     strategies, cost, entry_lag, sessions.data());

    EXPECT_GT(cpu_result.total_trades, 0) << "CPU should produce trades";

    // THE critical comparison: GPU with sessions must match CPU with sessions
    expect_results_match(gpu_result, cpu_result, "Sessions");
}

// =============================================================================
// TEST: GPU matches CPU -- session early close force exit
//
// Specifically tests that both GPU and CPU force-exit positions when
// time_of_day >= session_close - 30.
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_SessionEarlyClose) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 60 bars, all within RTH, with an early close at 13:00 ET (46800s)
    const int          N = 60;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = 100.0f + static_cast<float>(i) * 0.5f;
    }
    auto bars = make_bars_from_closes(closes);

    // Session: starts at 10:00 ET, early close at 13:00 ET
    // Bars advance 30 seconds each, so bar 0 = 36000s, bar 59 = 37770s
    // Put bars 50-59 near the early close boundary
    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].halt_flag         = 0;
        sessions[i].early_close       = 1;
        sessions[i].session_close_sec = 46800;  // 13:00 ET early close

        if (i < 50) {
            // Normal RTH: 10:00 - 10:25 ET range
            sessions[i].time_of_day_sec = 36000 + i * 30;
        } else {
            // Near session close: 12:59:30+ ET (within 30s of 13:00)
            sessions[i].time_of_day_sec = 46770 + (i - 50) * 3;
        }
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    // Constant positive signal
    std::vector<float>        signal(N, 1.0f);
    const float*              signal_ptr     = signal.data();
    std::vector<const float*> indicator_ptrs = {signal_ptr};

    // ETH strategy (no trading hours filter) to isolate the session close effect
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 100;  // very long timeout -- only session close forces exit
    config.trading_hours        = StrategyConfig::ETH;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies = {config};

    // GPU
    CLBacktest     gpu_bt;
    BacktestResult gpu_result =
        run_clbacktest_with_sessions(gpu_ctx, gpu_bt, bars, indicator_ptrs, 1, eval_start, eval_end,
                                     strategies, cost, entry_lag, sessions.data());

    // CPU
    CLContext      cpu_ctx;
    CLBacktest     cpu_bt;
    BacktestResult cpu_result =
        run_clbacktest_with_sessions(cpu_ctx, cpu_bt, bars, indicator_ptrs, 1, eval_start, eval_end,
                                     strategies, cost, entry_lag, sessions.data());

    EXPECT_GT(cpu_result.total_trades, 0) << "CPU should produce trades with early close";

    expect_results_match(gpu_result, cpu_result, "SessionEarlyClose");
}

// =============================================================================
// TEST: GPU matches CPU -- halt flag suppresses entries
// =============================================================================

TEST(CLBacktest, GPUMatchesCPU_HaltFlag) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 80 bars
    const int          N = 80;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = 100.0f + static_cast<float>(i) * 0.3f;
    }
    auto bars = make_bars_from_closes(closes);

    // All bars in RTH, but bars 0-39 are halted, bars 40-79 are not
    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].halt_flag         = (i < 40) ? 1 : 0;
        sessions[i].early_close       = 0;
        sessions[i].session_close_sec = 61200;
        sessions[i].time_of_day_sec   = 36000;  // 10:00 ET (RTH)
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    std::vector<float>        signal(N, 1.0f);
    const float*              signal_ptr     = signal.data();
    std::vector<const float*> indicator_ptrs = {signal_ptr};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 8;
    config.trading_hours        = StrategyConfig::ETH;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies = {config};

    // GPU with halt flags
    CLBacktest     gpu_bt;
    BacktestResult gpu_result =
        run_clbacktest_with_sessions(gpu_ctx, gpu_bt, bars, indicator_ptrs, 1, eval_start, eval_end,
                                     strategies, cost, entry_lag, sessions.data());

    // CPU with halt flags
    CLContext      cpu_ctx;
    CLBacktest     cpu_bt;
    BacktestResult cpu_result =
        run_clbacktest_with_sessions(cpu_ctx, cpu_bt, bars, indicator_ptrs, 1, eval_start, eval_end,
                                     strategies, cost, entry_lag, sessions.data());

    EXPECT_GT(cpu_result.total_trades, 0) << "CPU should produce trades in non-halted region";

    expect_results_match(gpu_result, cpu_result, "HaltFlag");

    // Also verify: without sessions (no halt), we get MORE trades
    CLBacktest     gpu_bt_nohalt;
    BacktestResult nohalt_result =
        run_clbacktest(gpu_ctx, gpu_bt_nohalt, bars, indicator_ptrs, 1, eval_start, eval_end,
                       strategies, cost, entry_lag);

    EXPECT_GE(nohalt_result.total_trades, gpu_result.total_trades)
        << "Without halt flags, should have at least as many trades";
}

// =============================================================================
// OpenCL Kernel Fixture-Data Tests
//
// These tests initialize CLContext with DeviceType::AUTO to use the actual
// OpenCL kernel code path (not the C++ fallback). They load the test fixture
// bar file (tests/data/synthetic_30s.bin), compute indicators/backtests via
// the GPU kernel path, and compare against C++ reference implementations.
//
// When no OpenCL device is available, all tests are gracefully skipped.
// =============================================================================

// ---------------------------------------------------------------------------
// Helpers for fixture-data kernel tests
// ---------------------------------------------------------------------------

static const std::string KERNEL_TEST_BIN_PATH = test_fixture::fixture_path();

// ---------------------------------------------------------------------------
// Test fixture: loads the fixture bar file once, initializes OpenCL once
// ---------------------------------------------------------------------------

class OpenCL_KernelFixtureData : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Load binary test data
        s_data_loaded = s_market_data.load(KERNEL_TEST_BIN_PATH);
        if (!s_data_loaded) return;
        s_bar_count = s_market_data.bar_count();

        // Initialize OpenCL (AUTO = tries GPU, then CPU runtime)
        s_opencl_available = s_ctx.initialize(DeviceType::AUTO);
    }

    static void TearDownTestSuite() { s_ctx.release(); }

    void SetUp() override {
        if (!s_data_loaded) {
            GTEST_SKIP() << "Test data not found: " << KERNEL_TEST_BIN_PATH;
        }
        if (!s_opencl_available) {
            SKIP_UNLESS_OPENCL(
                "no OpenCL device is available (install an OpenCL CPU runtime "
                "such as POCL or Intel's for kernel testing)");
        }
    }

    static MarketData s_market_data;
    static bool       s_data_loaded;
    static int        s_bar_count;
    static CLContext  s_ctx;
    static bool       s_opencl_available;
};

MarketData OpenCL_KernelFixtureData::s_market_data;
bool       OpenCL_KernelFixtureData::s_data_loaded = false;
int        OpenCL_KernelFixtureData::s_bar_count   = 0;
CLContext  OpenCL_KernelFixtureData::s_ctx;
bool       OpenCL_KernelFixtureData::s_opencl_available = false;

// =============================================================================
// a) EMA-20: OpenCL kernel vs C++ EMASMACross reference
// =============================================================================

TEST_F(OpenCL_KernelFixtureData, EMA20_KernelMatchesCPP) {
    const int N      = s_bar_count;
    const int period = 20;

    // -- C++ reference: EMASMACross --
    EMASMACross        cross(MAType::EMA, period, MAType::EMA, period + 1);
    std::vector<float> ref_output(static_cast<size_t>(N) * 2);
    cross.compute(s_market_data.bars(), 0, N, ref_output.data());
    // Fast MA (EMA-20) is series 0: ref_output[0..N)

    // -- OpenCL kernel path --
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    IndicatorSpec spec;
    spec.type         = IndicatorType::EMA_SMA_CROSS;
    spec.fast_ma_type = MAType::EMA;
    spec.fast_period  = period;
    spec.slow_ma_type = MAType::EMA;
    spec.slow_period  = period + 1;
    spec.ma_series    = 0;  // fast MA
    spec.warmup_bars  = 3 * period;

    std::vector<float>         cl_output(N, 0.0f);
    std::vector<float*>        output_ptrs = {cl_output.data()};
    std::vector<IndicatorSpec> specs       = {spec};

    SoABarData soa;
    soa.from_bars(s_market_data.bars(), 0, N);

    cl_ind.compute(s_ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, output_ptrs);

    // Compare after warmup
    const int warmup   = 3 * period;
    int       compared = 0;
    for (int i = warmup; i < N; ++i) {
        if (std::isnan(ref_output[i])) {
            EXPECT_TRUE(std::isnan(cl_output[i])) << "Expected NaN at bar " << i;
        } else {
            // Use relative tolerance for price-level indicators (EMA ~5500).
            // Float32 has ~7 significant digits, so 0.01% relative error is expected.
            float tol = std::abs(ref_output[i]) * 1e-4f + 1e-6f;
            EXPECT_NEAR(cl_output[i], ref_output[i], tol) << "EMA-20 kernel mismatch at bar " << i;
            compared++;
        }
    }
    EXPECT_GT(compared, 0) << "No valid EMA-20 values to compare";
}

// =============================================================================
// b) RSI-14: OpenCL kernel vs C++ RSI reference
// =============================================================================

TEST_F(OpenCL_KernelFixtureData, RSI14_KernelMatchesCPP) {
    const int N      = s_bar_count;
    const int period = 14;

    // -- C++ reference --
    RSI                rsi(period);
    std::vector<float> ref_output(N);
    rsi.compute(s_market_data.bars(), 0, N, ref_output.data());

    // -- OpenCL kernel path --
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    IndicatorSpec spec;
    spec.type        = IndicatorType::RSI;
    spec.rsi_period  = period;
    spec.warmup_bars = rsi.warmup_bars();

    std::vector<float>         cl_output(N, 0.0f);
    std::vector<float*>        output_ptrs = {cl_output.data()};
    std::vector<IndicatorSpec> specs       = {spec};

    SoABarData soa;
    soa.from_bars(s_market_data.bars(), 0, N);

    cl_ind.compute(s_ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, output_ptrs);

    // Compare after warmup
    const int warmup   = rsi.warmup_bars();
    int       compared = 0;
    for (int i = warmup; i < N; ++i) {
        if (std::isnan(ref_output[i])) {
            EXPECT_TRUE(std::isnan(cl_output[i])) << "Expected NaN at bar " << i;
        } else {
            EXPECT_NEAR(cl_output[i], ref_output[i], 1e-4f)
                << "RSI-14 kernel mismatch at bar " << i;
            compared++;
        }
    }
    EXPECT_GT(compared, 0) << "No valid RSI-14 values to compare";
}

// =============================================================================
// c) ATR-14: OpenCL kernel vs C++ ATR reference
// =============================================================================

TEST_F(OpenCL_KernelFixtureData, ATR14_KernelMatchesCPP) {
    const int N      = s_bar_count;
    const int period = 14;

    // -- C++ reference --
    ATR                atr(period);
    std::vector<float> ref_output(N);
    atr.compute(s_market_data.bars(), 0, N, ref_output.data());

    // -- OpenCL kernel path --
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    IndicatorSpec spec;
    spec.type        = IndicatorType::ATR;
    spec.atr_period  = period;
    spec.warmup_bars = atr.warmup_bars();

    std::vector<float>         cl_output(N, 0.0f);
    std::vector<float*>        output_ptrs = {cl_output.data()};
    std::vector<IndicatorSpec> specs       = {spec};

    SoABarData soa;
    soa.from_bars(s_market_data.bars(), 0, N);

    cl_ind.compute(s_ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, output_ptrs);

    // Compare after warmup
    const int warmup   = atr.warmup_bars();
    int       compared = 0;
    for (int i = warmup; i < N; ++i) {
        if (std::isnan(ref_output[i])) {
            EXPECT_TRUE(std::isnan(cl_output[i])) << "Expected NaN at bar " << i;
        } else {
            EXPECT_NEAR(cl_output[i], ref_output[i], 1e-4f)
                << "ATR-14 kernel mismatch at bar " << i;
            compared++;
        }
    }
    EXPECT_GT(compared, 0) << "No valid ATR-14 values to compare";
}

// =============================================================================
// d) Bollinger(20,2): OpenCL kernel vs C++ BollingerBands reference
//    Tests all three series (middle, upper, lower) to exercise both
//    fp64 and NO_FP64 Welford code paths.
// =============================================================================

TEST_F(OpenCL_KernelFixtureData, Bollinger20_KernelMatchesCPP) {
    const int   N          = s_bar_count;
    const int   period     = 20;
    const float multiplier = 2.0f;

    // -- C++ reference: 3 series (middle, upper, lower) --
    BollingerBands     bb(period, multiplier);
    std::vector<float> ref_output(static_cast<size_t>(N) * 3);
    bb.compute(s_market_data.bars(), 0, N, ref_output.data());
    // Series layout: [middle(N), upper(N), lower(N)]

    SoABarData soa;
    soa.from_bars(s_market_data.bars(), 0, N);

    const char* series_names[] = {"middle", "upper", "lower"};

    for (int series = 0; series < 3; ++series) {
        CLIndicators cl_ind;
        cl_ind.set_on_device(true);

        IndicatorSpec spec;
        spec.type                 = IndicatorType::BOLLINGER;
        spec.bollinger_period     = period;
        spec.bollinger_multiplier = multiplier;
        spec.bollinger_series     = series;
        spec.warmup_bars          = bb.warmup_bars();

        std::vector<float>         cl_output(N, 0.0f);
        std::vector<float*>        output_ptrs = {cl_output.data()};
        std::vector<IndicatorSpec> specs       = {spec};

        cl_ind.compute(s_ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                       specs, output_ptrs);

        const float* ref_series = ref_output.data() + static_cast<size_t>(series) * N;
        const int    warmup     = bb.warmup_bars();
        int          compared   = 0;
        for (int i = warmup; i < N; ++i) {
            if (std::isnan(ref_series[i])) {
                EXPECT_TRUE(std::isnan(cl_output[i]))
                    << "Bollinger " << series_names[series] << ": expected NaN at bar " << i;
            } else {
                EXPECT_NEAR(cl_output[i], ref_series[i], 1e-4f)
                    << "Bollinger " << series_names[series] << " kernel mismatch at bar " << i;
                compared++;
            }
        }
        EXPECT_GT(compared, 0) << "No valid Bollinger " << series_names[series]
                               << " values to compare";
    }
}

// =============================================================================
// e) MACD(12,26,9): OpenCL kernel vs C++ MACD reference
//    Tests all three series: MACD line, signal line, histogram
// =============================================================================

TEST_F(OpenCL_KernelFixtureData, MACD_KernelMatchesCPP) {
    const int N             = s_bar_count;
    const int fast_period   = 12;
    const int slow_period   = 26;
    const int signal_period = 9;

    // -- C++ reference: 3 series (MACD, signal, histogram) --
    MACD               macd(fast_period, slow_period, signal_period);
    std::vector<float> ref_output(static_cast<size_t>(N) * 3);
    macd.compute(s_market_data.bars(), 0, N, ref_output.data());
    // Series layout: [MACD_line(N), signal(N), histogram(N)]

    SoABarData soa;
    soa.from_bars(s_market_data.bars(), 0, N);

    const char* series_names[] = {"MACD_line", "signal", "histogram"};

    for (int series = 0; series < 3; ++series) {
        CLIndicators cl_ind;
        cl_ind.set_on_device(true);

        IndicatorSpec spec;
        spec.type               = IndicatorType::MACD;
        spec.macd_fast_period   = fast_period;
        spec.macd_slow_period   = slow_period;
        spec.macd_signal_period = signal_period;
        spec.macd_series        = series;
        spec.warmup_bars        = macd.warmup_bars();

        std::vector<float>         cl_output(N, 0.0f);
        std::vector<float*>        output_ptrs = {cl_output.data()};
        std::vector<IndicatorSpec> specs       = {spec};

        cl_ind.compute(s_ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                       specs, output_ptrs);

        const float* ref_series = ref_output.data() + static_cast<size_t>(series) * N;
        const int    warmup     = macd.warmup_bars();
        int          compared   = 0;
        for (int i = warmup; i < N; ++i) {
            if (std::isnan(ref_series[i])) {
                EXPECT_TRUE(std::isnan(cl_output[i]))
                    << "MACD " << series_names[series] << ": expected NaN at bar " << i;
            } else {
                // MACD values are differences of two EMAs (small, near zero).
                // Float32 EMA over 13K bars accumulates ~0.002 absolute error.
                // Use relative 0.01% + absolute 0.01 (covers the compounding).
                float tol = std::abs(ref_series[i]) * 1e-3f + 0.01f;
                EXPECT_NEAR(cl_output[i], ref_series[i], tol)
                    << "MACD " << series_names[series] << " kernel mismatch at bar " << i;
                compared++;
            }
        }
        EXPECT_GT(compared, 0) << "No valid MACD " << series_names[series] << " values to compare";
    }
}

// =============================================================================
// f) Backtest kernel: EMA crossover strategy through full pipeline
//    OpenCL indicator kernel -> signal -> OpenCL backtest kernel
//    vs C++ indicator compute + BacktestEngine::run()
// =============================================================================

TEST_F(OpenCL_KernelFixtureData, BacktestKernelMatchesCPP) {
    const int N           = s_bar_count;
    const int fast_period = 10;
    const int slow_period = 30;

    // -- C++ reference: compute EMA cross indicators --
    EMASMACross        cross(MAType::EMA, fast_period, MAType::EMA, slow_period);
    std::vector<float> ind_output(static_cast<size_t>(N) * 2);
    cross.compute(s_market_data.bars(), 0, N, ind_output.data());

    // Build crossover signal: fast - slow
    std::vector<float> ref_signal(N);
    for (int i = 0; i < N; ++i) {
        float fast = ind_output[i];
        float slow = ind_output[N + i];
        if (std::isnan(fast) || std::isnan(slow)) {
            ref_signal[i] = std::numeric_limits<float>::quiet_NaN();
        } else {
            ref_signal[i] = fast - slow;
        }
    }

    // -- OpenCL kernel path: compute EMA indicators via GPU --
    CLIndicators cl_ind;
    cl_ind.set_on_device(true);

    IndicatorSpec fast_spec;
    fast_spec.type         = IndicatorType::EMA_SMA_CROSS;
    fast_spec.fast_ma_type = MAType::EMA;
    fast_spec.fast_period  = fast_period;
    fast_spec.slow_ma_type = MAType::EMA;
    fast_spec.slow_period  = slow_period;
    fast_spec.ma_series    = 0;  // fast MA
    fast_spec.warmup_bars  = std::max(3 * fast_period, 3 * slow_period);

    IndicatorSpec slow_spec = fast_spec;
    slow_spec.ma_series     = 1;  // slow MA

    SoABarData soa;
    soa.from_bars(s_market_data.bars(), 0, N);

    std::vector<float>         cl_fast(N, 0.0f), cl_slow(N, 0.0f);
    std::vector<float*>        output_ptrs = {cl_fast.data(), cl_slow.data()};
    std::vector<IndicatorSpec> specs       = {fast_spec, slow_spec};

    cl_ind.compute(s_ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                   specs, output_ptrs);

    // Build GPU crossover signal
    std::vector<float> gpu_signal(N);
    for (int i = 0; i < N; ++i) {
        if (std::isnan(cl_fast[i]) || std::isnan(cl_slow[i])) {
            gpu_signal[i] = std::numeric_limits<float>::quiet_NaN();
        } else {
            gpu_signal[i] = cl_fast[i] - cl_slow[i];
        }
    }

    // Strategy config: EMA crossover with trailing stop exit
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.3f;
    config.exit_mode            = StrategyConfig::TRAILING_STOP;
    config.trailing_atr_mult    = 2.0f;
    config.atr_series_index     = 1;  // ATR series

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int warmup     = cross.warmup_bars();
    const int eval_start = warmup;
    const int eval_end   = N;

    // Compute ATR for trailing stop (C++ reference)
    ATR                atr(14);
    std::vector<float> ref_atr(N);
    atr.compute(s_market_data.bars(), 0, N, ref_atr.data());

    // Compute ATR via GPU
    CLIndicators cl_ind_atr;
    cl_ind_atr.set_on_device(true);
    IndicatorSpec atr_spec;
    atr_spec.type        = IndicatorType::ATR;
    atr_spec.atr_period  = 14;
    atr_spec.warmup_bars = atr.warmup_bars();

    std::vector<float>         cl_atr(N, 0.0f);
    std::vector<float*>        atr_ptrs  = {cl_atr.data()};
    std::vector<IndicatorSpec> atr_specs = {atr_spec};

    cl_ind_atr.compute(s_ctx, soa.open.data(), soa.high.data(), soa.low.data(), soa.close.data(), N,
                       atr_specs, atr_ptrs);

    // -- C++ BacktestEngine reference --
    BacktestEngine engine;
    const float*   ref_ptrs_arr[] = {ref_signal.data() + eval_start, ref_atr.data() + eval_start};
    BacktestResult ref_result = engine.run(s_market_data.bars(), eval_start, eval_end, ref_ptrs_arr,
                                           2, config, cost, entry_lag);

    // -- OpenCL backtest kernel --
    CLBacktest cl_bt;
    // Require the GPU kernel to actually build; otherwise evaluate() would
    // silently fall back to CPU and this parity test would be CPU-vs-CPU.
#ifdef HAS_OPENCL
    ASSERT_TRUE(cl_bt.build_kernel(s_ctx))
        << "CLBacktest::build_kernel failed -- would degenerate to CPU-vs-CPU.";
#endif

    const float*                gpu_ptrs_arr[] = {gpu_signal.data(), cl_atr.data()};
    std::vector<const float*>   gpu_ptrs(gpu_ptrs_arr, gpu_ptrs_arr + 2);
    std::vector<StrategyConfig> strategies = {config};
    std::vector<BacktestResult> gpu_results;

    cl_bt.evaluate(s_ctx, s_market_data.bars(),
                   0,  // bar_offset
                   gpu_ptrs.data(),
                   2,  // num_indicator_series
                   N,  // num_bars
                   eval_start, eval_end, strategies, cost, entry_lag, gpu_results);

    ASSERT_EQ(gpu_results.size(), 1u);

    // Compare key metrics.
    // GPU float FMA rounding can shift trailing-stop fill prices by a
    // fraction of a tick, accumulating to ~$0.20 over many trades on
    // $150k+ total PnL.  Use a tolerance proportional to the magnitude.
    const float pnl_tol = std::max(1.0f,
                                   std::abs(ref_result.total_pnl) * 1e-5f);  // 0.001% or $1 min

    EXPECT_EQ(gpu_results[0].total_trades, ref_result.total_trades)
        << "Backtest kernel: trade count mismatch";
    EXPECT_NEAR(gpu_results[0].total_pnl, ref_result.total_pnl, pnl_tol)
        << "Backtest kernel: total PnL mismatch";
    EXPECT_NEAR(gpu_results[0].win_rate, ref_result.win_rate, 1e-4f)
        << "Backtest kernel: win rate mismatch";
    EXPECT_NEAR(gpu_results[0].gross_wins, ref_result.gross_wins, pnl_tol)
        << "Backtest kernel: gross wins mismatch";
    EXPECT_NEAR(gpu_results[0].gross_losses, ref_result.gross_losses, pnl_tol)
        << "Backtest kernel: gross losses mismatch";
    EXPECT_NEAR(gpu_results[0].max_drawdown, ref_result.max_drawdown, pnl_tol)
        << "Backtest kernel: max drawdown mismatch";
}

// =============================================================================
// g) Backtest with sessions: session-aware GPU kernel vs CPU
//    Verifies halt flags, early close, and trading hours filtering
// =============================================================================

TEST_F(OpenCL_KernelFixtureData, BacktestWithSessions_KernelMatchesCPP) {
    const int N = s_bar_count;

    // Build session info from the fixture bar data
    // The MarketData has session info pre-built for each bar
    std::vector<SessionInfo> sessions(N);
    for (int i = 0; i < N; ++i) {
        sessions[i] = s_market_data.session(i);
    }
    stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                        SessionModel::us_futures(), 30);

    // Compute EMA crossover signal on the fixture data
    const int          fast_period = 10;
    const int          slow_period = 30;
    EMASMACross        cross(MAType::EMA, fast_period, MAType::EMA, slow_period);
    std::vector<float> ind_output(static_cast<size_t>(N) * 2);
    cross.compute(s_market_data.bars(), 0, N, ind_output.data());

    std::vector<float> signal(N);
    for (int i = 0; i < N; ++i) {
        float fast = ind_output[i];
        float slow = ind_output[N + i];
        if (std::isnan(fast) || std::isnan(slow)) {
            signal[i] = std::numeric_limits<float>::quiet_NaN();
        } else {
            signal[i] = fast - slow;
        }
    }

    // Strategy: RTH only, bar timeout exit
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.3f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 20;
    config.trading_hours        = StrategyConfig::RTH;

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int warmup     = cross.warmup_bars();
    const int eval_start = warmup;
    const int eval_end   = N;

    std::vector<StrategyConfig> strategies     = {config};
    const float*                signal_ptr     = signal.data();
    std::vector<const float*>   indicator_ptrs = {signal_ptr};

    // GPU result with sessions (actual kernel path)
    CLBacktest     gpu_bt;
    BacktestResult gpu_result = run_clbacktest_with_sessions(
        s_ctx, gpu_bt, std::vector<Bar>(s_market_data.bars(), s_market_data.bars() + N),
        indicator_ptrs, 1, eval_start, eval_end, strategies, cost, entry_lag, sessions.data());

    // CPU result with sessions (uninitialized context forces CPU fallback)
    CLContext        cpu_ctx;  // not initialized
    CLBacktest       cpu_bt;
    std::vector<Bar> bars_copy(s_market_data.bars(), s_market_data.bars() + N);
    BacktestResult   cpu_result =
        run_clbacktest_with_sessions(cpu_ctx, cpu_bt, bars_copy, indicator_ptrs, 1, eval_start,
                                     eval_end, strategies, cost, entry_lag, sessions.data());

    EXPECT_GT(cpu_result.total_trades, 0) << "CPU should produce trades on the fixture data";

    // THE critical comparison: GPU kernel with sessions must match CPU with sessions
    expect_results_match(gpu_result, cpu_result, "BacktestWithSessions_FixtureData");
}

// =============================================================================
// TEST: Indicator-Set Grouping -- grouped dispatch matches CPU
//
// Creates strategies with 3 different indicator sets:
//   Set A: indicators 0, 1
//   Set B: indicators 0, 2
//   Set C: indicators 1, 2, 3
// Runs through both CPU and GPU paths.  Verifies results match.
// =============================================================================

TEST(CLBacktest_Grouping, GroupedMatchesCPU) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 200 bars with a cyclic price pattern
    const int          N = 200;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        float phase = static_cast<float>(i) * 0.08f;
        closes[i]   = 100.0f + 8.0f * std::sin(phase) + static_cast<float>(i) * 0.02f;
    }
    auto bars = make_bars_from_closes(closes);

    // Four indicator series with distinct patterns
    std::vector<float> ind0(N), ind1(N), ind2(N), ind3(N);
    for (int i = 0; i < N; ++i) {
        float t = static_cast<float>(i);
        ind0[i] = std::sin(t * 0.08f);  // slow sine
        ind1[i] = std::sin(t * 0.15f);  // faster sine
        ind2[i] = std::cos(t * 0.1f);   // cosine
        ind3[i] = 1.5f;                 // constant ATR
    }
    std::vector<const float*> indicator_ptrs = {ind0.data(), ind1.data(), ind2.data(), ind3.data()};

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    // Build 120 strategies (>= 64 to trigger grouping) across 3 indicator sets
    const int                   NUM_STRATEGIES = 120;
    std::vector<StrategyConfig> strategies(NUM_STRATEGIES);
    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        StrategyConfig& cfg = strategies[s];

        // Assign to one of 3 indicator sets based on s % 3
        int set_id = s % 3;
        if (set_id == 0) {
            // Set A: indicators 0, 1
            cfg.num_indicators       = 2;
            cfg.indicator_indices[0] = 0;
            cfg.indicator_indices[1] = 1;
            cfg.weights[0]           = 0.6f;
            cfg.weights[1]           = 0.4f;
        } else if (set_id == 1) {
            // Set B: indicators 0, 2
            cfg.num_indicators       = 2;
            cfg.indicator_indices[0] = 0;
            cfg.indicator_indices[1] = 2;
            cfg.weights[0]           = 0.5f;
            cfg.weights[1]           = 0.5f;
        } else {
            // Set C: indicators 1, 2, 3 (3 indicators)
            cfg.num_indicators       = 3;
            cfg.indicator_indices[0] = 1;
            cfg.indicator_indices[1] = 2;
            cfg.indicator_indices[2] = 3;
            cfg.weights[0]           = 0.4f;
            cfg.weights[1]           = 0.3f;
            cfg.weights[2]           = 0.3f;
        }

        cfg.aggregation = StrategyConfig::WEIGHTED_SUM;
        cfg.entry_threshold =
            0.1f + 0.7f * (static_cast<float>(s) / static_cast<float>(NUM_STRATEGIES - 1));

        // Vary exit modes across strategies
        int mode_idx = s % 5;
        if (mode_idx == 0) {
            cfg.exit_mode         = StrategyConfig::BAR_TIMEOUT;
            cfg.exit_timeout_bars = 5 + (s % 15);
        } else if (mode_idx == 1) {
            cfg.exit_mode = StrategyConfig::SIGNAL_REVERSAL;
        } else if (mode_idx == 2) {
            cfg.exit_mode         = StrategyConfig::TRAILING_STOP;
            cfg.trailing_atr_mult = 1.0f + static_cast<float>(s % 10) * 0.3f;
            cfg.atr_series_index  = 3;
        } else if (mode_idx == 3) {
            cfg.exit_mode        = StrategyConfig::TAKE_PROFIT_ATR;
            cfg.tp_atr_mult      = 1.0f + static_cast<float>(s % 8) * 0.5f;
            cfg.atr_series_index = 3;
        } else {
            cfg.exit_mode        = StrategyConfig::TP_SL_ATR;
            cfg.tp_atr_mult      = 2.0f;
            cfg.sl_atr_mult      = 1.0f + static_cast<float>(s % 5) * 0.2f;
            cfg.atr_series_index = 3;
        }
    }

    // GPU: grouped dispatch (120 strategies > 64 threshold, 3 groups)
    CLBacktest gpu_bt;
    auto gpu_results = run_clbacktest_multi(gpu_ctx, gpu_bt, bars, indicator_ptrs, 4, eval_start,
                                            eval_end, strategies, cost, entry_lag);

    // CPU: all strategies via CPU fallback
    CLContext  cpu_ctx;  // not initialized
    CLBacktest cpu_bt;
    auto cpu_results = run_clbacktest_multi(cpu_ctx, cpu_bt, bars, indicator_ptrs, 4, eval_start,
                                            eval_end, strategies, cost, entry_lag);

    ASSERT_EQ(gpu_results.size(), static_cast<size_t>(NUM_STRATEGIES));
    ASSERT_EQ(cpu_results.size(), static_cast<size_t>(NUM_STRATEGIES));

    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        std::string label = "Grouped_Strategy_" + std::to_string(s);
        expect_results_match(gpu_results[s], cpu_results[s], label);
    }
}

// =============================================================================
// TEST: Indicator-Set Grouping -- single group (no overhead)
//
// All strategies use the same indicator set.  Verifies that grouping
// produces 1 group and results match CPU.
// =============================================================================

TEST(CLBacktest_Grouping, SingleGroupNoOverhead) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 100 bars with an uptrend
    const int          N = 100;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] =
            100.0f + static_cast<float>(i) * 0.3f + 3.0f * std::sin(static_cast<float>(i) * 0.12f);
    }
    auto bars = make_bars_from_closes(closes);

    // Two indicator series
    std::vector<float> signal(N), atr(N, 1.5f);
    for (int i = 0; i < N; ++i) {
        signal[i] = std::sin(static_cast<float>(i) * 0.1f);
    }
    std::vector<const float*> indicator_ptrs = {signal.data(), atr.data()};

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    // All 80 strategies use the same indicator set (indices 0, 1)
    // 80 > 64 threshold but only 1 group -- should skip grouping
    const int                   NUM_STRATEGIES = 80;
    std::vector<StrategyConfig> strategies(NUM_STRATEGIES);
    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        StrategyConfig& cfg      = strategies[s];
        cfg.num_indicators       = 2;
        cfg.indicator_indices[0] = 0;
        cfg.indicator_indices[1] = 1;
        cfg.weights[0]           = 0.7f;
        cfg.weights[1]           = 0.3f;
        cfg.aggregation          = StrategyConfig::WEIGHTED_SUM;
        cfg.entry_threshold =
            0.1f + 0.6f * (static_cast<float>(s) / static_cast<float>(NUM_STRATEGIES - 1));

        // Vary exit modes
        int mode_idx = s % 3;
        if (mode_idx == 0) {
            cfg.exit_mode         = StrategyConfig::BAR_TIMEOUT;
            cfg.exit_timeout_bars = 5 + (s % 20);
        } else if (mode_idx == 1) {
            cfg.exit_mode         = StrategyConfig::TRAILING_STOP;
            cfg.trailing_atr_mult = 1.5f;
            cfg.atr_series_index  = 1;
        } else {
            cfg.exit_mode = StrategyConfig::SIGNAL_REVERSAL;
        }
    }

    // GPU
    CLBacktest gpu_bt;
    auto gpu_results = run_clbacktest_multi(gpu_ctx, gpu_bt, bars, indicator_ptrs, 2, eval_start,
                                            eval_end, strategies, cost, entry_lag);

    // CPU
    CLContext  cpu_ctx;
    CLBacktest cpu_bt;
    auto cpu_results = run_clbacktest_multi(cpu_ctx, cpu_bt, bars, indicator_ptrs, 2, eval_start,
                                            eval_end, strategies, cost, entry_lag);

    ASSERT_EQ(gpu_results.size(), static_cast<size_t>(NUM_STRATEGIES));
    ASSERT_EQ(cpu_results.size(), static_cast<size_t>(NUM_STRATEGIES));

    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        std::string label = "SingleGroup_Strategy_" + std::to_string(s);
        expect_results_match(gpu_results[s], cpu_results[s], label);
    }
}

// =============================================================================
// TEST: Indicator-Set Grouping -- small batch skips grouping
//
// Creates < 64 strategies with multiple indicator sets.  Verifies that
// the flat dispatch path is used (threshold not met) and results still
// match CPU.
// =============================================================================

TEST(CLBacktest_Grouping, SmallBatchSkipsGrouping) {
    CLContext gpu_ctx;
    if (!try_init_opencl(gpu_ctx)) {
        SKIP_UNLESS_OPENCL("no OpenCL device is available");
    }

    // Create 150 bars
    const int          N = 150;
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] =
            100.0f + 5.0f * std::sin(static_cast<float>(i) * 0.06f) + static_cast<float>(i) * 0.01f;
    }
    auto bars = make_bars_from_closes(closes);

    // Three indicator series
    std::vector<float> ind0(N), ind1(N), ind2(N);
    for (int i = 0; i < N; ++i) {
        ind0[i] = std::sin(static_cast<float>(i) * 0.1f);
        ind1[i] = std::cos(static_cast<float>(i) * 0.08f);
        ind2[i] = 1.2f;  // constant ATR
    }
    std::vector<const float*> indicator_ptrs = {ind0.data(), ind1.data(), ind2.data()};

    CostModel cost       = CostModel::es();
    const int entry_lag  = 1;
    const int eval_start = 0;
    const int eval_end   = N;

    // 30 strategies (< 64) with 3 different indicator sets
    // This should use the flat dispatch path (below threshold)
    const int                   NUM_STRATEGIES = 30;
    std::vector<StrategyConfig> strategies(NUM_STRATEGIES);
    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        StrategyConfig& cfg = strategies[s];

        int set_id = s % 3;
        if (set_id == 0) {
            cfg.num_indicators       = 1;
            cfg.indicator_indices[0] = 0;
            cfg.weights[0]           = 1.0f;
        } else if (set_id == 1) {
            cfg.num_indicators       = 2;
            cfg.indicator_indices[0] = 0;
            cfg.indicator_indices[1] = 1;
            cfg.weights[0]           = 0.5f;
            cfg.weights[1]           = 0.5f;
        } else {
            cfg.num_indicators       = 2;
            cfg.indicator_indices[0] = 1;
            cfg.indicator_indices[1] = 2;
            cfg.weights[0]           = 0.6f;
            cfg.weights[1]           = 0.4f;
        }

        cfg.aggregation = StrategyConfig::WEIGHTED_SUM;
        cfg.entry_threshold =
            0.2f + 0.5f * (static_cast<float>(s) / static_cast<float>(NUM_STRATEGIES - 1));

        int mode_idx = s % 4;
        if (mode_idx == 0) {
            cfg.exit_mode         = StrategyConfig::BAR_TIMEOUT;
            cfg.exit_timeout_bars = 8 + (s % 10);
        } else if (mode_idx == 1) {
            cfg.exit_mode = StrategyConfig::SIGNAL_REVERSAL;
        } else if (mode_idx == 2) {
            cfg.exit_mode         = StrategyConfig::TRAILING_STOP;
            cfg.trailing_atr_mult = 2.0f;
            cfg.atr_series_index  = 2;
        } else {
            cfg.exit_mode        = StrategyConfig::TP_SL_ATR;
            cfg.tp_atr_mult      = 2.0f;
            cfg.sl_atr_mult      = 1.0f;
            cfg.atr_series_index = 2;
        }
    }

    // GPU (flat dispatch -- below 64-strategy threshold)
    CLBacktest gpu_bt;
    auto gpu_results = run_clbacktest_multi(gpu_ctx, gpu_bt, bars, indicator_ptrs, 3, eval_start,
                                            eval_end, strategies, cost, entry_lag);

    // CPU
    CLContext  cpu_ctx;
    CLBacktest cpu_bt;
    auto cpu_results = run_clbacktest_multi(cpu_ctx, cpu_bt, bars, indicator_ptrs, 3, eval_start,
                                            eval_end, strategies, cost, entry_lag);

    ASSERT_EQ(gpu_results.size(), static_cast<size_t>(NUM_STRATEGIES));
    ASSERT_EQ(cpu_results.size(), static_cast<size_t>(NUM_STRATEGIES));

    for (int s = 0; s < NUM_STRATEGIES; ++s) {
        std::string label = "SmallBatch_Strategy_" + std::to_string(s);
        expect_results_match(gpu_results[s], cpu_results[s], label);
    }
}
