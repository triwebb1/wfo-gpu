#pragma once
// ---------------------------------------------------------------------------
// CLIndicators -- Host-side indicator compute dispatcher
//
// When OpenCL is available: builds kernels from .cl source, sets args,
// enqueues ND-range kernels on the device.
//
// When OpenCL is not available: falls back to CPU computation using the
// existing C++ indicator classes.
//
// Supports the full indicator grid (14+ types): EMA/SMA cross, RSI, ATR,
// MACD, ADX, Bollinger Bands, Keltner Channels, Stochastic, CCI, ROC,
// VWAP, OBV, Donchian Channels, Pivot Points, HMA, Connors RSI. See
// IndicatorType in indicators/indicator.h for the authoritative list.
//
// Interface is agnostic to compute backend -- callers pass SoA bar data
// and an indicator specification, and receive computed output arrays.
// ---------------------------------------------------------------------------

#include <vector>
#include <string>
#include <cstdint>

#include "cl_context.h"
#include "cl_memory.h"
#include "../market_data.h"
#include "../indicators/indicator.h"
#include "../grids/indicator_grid.h"
#include "../grids/window_mode_grid.h"

#ifdef HAS_OPENCL
#include <CL/cl.h>
#endif

// ---------------------------------------------------------------------------
// IndicatorSpec -- describes one indicator series to compute
// ---------------------------------------------------------------------------
struct IndicatorSpec {
    IndicatorType type        = IndicatorType::EMA_SMA_CROSS;
    WindowMode    window_mode = WindowMode::ROLLING;  // ROLLING = 30s bars, ANCHORED_* = TF bars

    // MA cross parameters
    MAType fast_ma_type = MAType::EMA;
    int    fast_period  = 3;
    MAType slow_ma_type = MAType::EMA;
    int    slow_period  = 20;
    int    ma_series    = 0;  // 0 = fast MA output, 1 = slow MA output

    // RSI parameters
    int rsi_period = 14;

    // ATR parameters
    int atr_period = 14;

    // MACD parameters
    int macd_fast_period   = 12;
    int macd_slow_period   = 26;
    int macd_signal_period = 9;
    int macd_series        = 0;  // 0=MACD line, 1=signal, 2=histogram

    // ADX parameters
    int adx_period = 14;
    int adx_series = 0;  // 0=ADX, 1=+DI, 2=-DI

    // Bollinger Bands parameters
    int   bollinger_period     = 20;
    float bollinger_multiplier = 2.0f;
    int   bollinger_series     = 0;  // 0=middle, 1=upper, 2=lower

    // Keltner Channels parameters
    MAType keltner_ma_type    = MAType::EMA;
    int    keltner_ma_period  = 20;
    int    keltner_atr_period = 10;
    float  keltner_multiplier = 2.0f;
    int    keltner_series     = 0;  // 0=center, 1=upper, 2=lower

    // Stochastic parameters
    int stoch_k_period = 14;
    int stoch_d_period = 3;
    int stoch_series   = 0;  // 0=%K, 1=%D

    // CCI parameters
    int cci_period = 20;

    // ROC parameters
    int roc_period = 10;

    // VWAP parameters
    float vwap_band_mult = 2.0f;
    int   vwap_series    = 0;  // 0=VWAP, 1=upper, 2=lower

    // OBV parameters
    int obv_smoothing = 20;

    // Donchian parameters
    int donchian_period = 20;
    int donchian_series = 0;  // 0=upper, 1=lower

    // Pivot Points parameters
    PivotType pivot_type   = PivotType::CLASSIC;
    int       pivot_series = 0;  // 0=PP, 1=S1, 2=S2, 3=S3, 4=R1, 5=R2, 6=R3

    // HMA parameters
    int hma_period = 20;

    // Connors RSI parameters
    int connors_rsi_period        = 3;
    int connors_streak_period     = 2;
    int connors_percentile_period = 100;

    // Warmup bars for this indicator
    int warmup_bars = 0;

    // Construct from IndicatorParams + series index within the indicator
    static IndicatorSpec from_params(const IndicatorParams& params, int series_idx = 0);
};

// ---------------------------------------------------------------------------
// CLIndicators -- compute dispatcher
// ---------------------------------------------------------------------------
class CLIndicators {
public:
    CLIndicators() = default;
    ~CLIndicators();

    // Whether indicators run as OpenCL kernels when a device is available.
    // Off by default: computed on the host, every device sees bit-identical
    // signals, so the device cannot change a backtest's outcome. Indicator
    // work is small next to the backtests; turn this on for very large
    // datasets and accept floating-point differences between devices. The
    // process default comes from [compute] indicators_on_device.
    static bool default_on_device();
    static void set_default_on_device(bool on);
    void        set_on_device(bool on) { m_on_device = on; }

    // Compute indicator series.
    //
    // Parameters:
    //   ctx          - OpenCL context (may be unavailable => CPU fallback)
    //   bar_open     - open price array (SoA), length = num_bars
    //   bar_high     - high price array (SoA), length = num_bars
    //   bar_low      - low price array (SoA), length = num_bars
    //   bar_close    - close price array (SoA), length = num_bars
    //   num_bars     - number of bars
    //   indicators   - list of indicator specifications to compute
    //   output_series - pre-allocated output arrays, one per indicator spec
    //                   Each float* has room for num_bars floats.
    //
    // When OpenCL is available, launches kernels on the device.
    // When not available, computes on CPU using the C++ indicator classes.
    void compute(const CLContext& ctx, const float* bar_open, const float* bar_high,
                 const float* bar_low, const float* bar_close, int num_bars,
                 const std::vector<IndicatorSpec>& indicators, std::vector<float*>& output_series);

    // Extended compute that also provides volume and timestamp data.
    // Required for VWAP, OBV, and PivotPoints which need volume/timestamps.
    // If bars is nullptr, falls back to the SoA-only path (volume/timestamp
    // dependent indicators will output NaN).
    void compute_with_bars(const CLContext& ctx, const Bar* bars, int bar_offset,
                           const float* bar_open, const float* bar_high, const float* bar_low,
                           const float* bar_close, int num_bars,
                           const std::vector<IndicatorSpec>& indicators,
                           std::vector<float*>&              output_series);

    // Anchored-aware compute: like compute_with_bars, but specs may carry
    // window_mode != ROLLING.  For those specs, indicator values are computed
    // on the higher-timeframe bars from MarketData::timeframe() and broadcast
    // back to 30s alignment.  bar_start_30s is the absolute 30s bar index
    // that corresponds to bars[0] (needed for completed_map lookup).
    //
    // Specs with window_mode == ROLLING are computed on the 30s bars as usual.
    void compute_with_bars_anchored(const CLContext& ctx, const Bar* bars_30s,
                                    int bar_start_30s,  // absolute 30s index of bars_30s[0]
                                    const float* bar_open, const float* bar_high,
                                    const float* bar_low, const float* bar_close, int num_bars_30s,
                                    const MarketData&                 market_data,
                                    const std::vector<IndicatorSpec>& indicators,
                                    std::vector<float*>&              output_series);

private:
    bool m_on_device = default_on_device();

    // ---- CPU fallback implementations ----
    void compute_ema_cpu(const float* close, float* output, int num_bars, int period,
                         int warmup) const;
    void compute_sma_cpu(const float* close, float* output, int num_bars, int period,
                         int warmup) const;
    void compute_rsi_cpu(const float* close, float* output, int num_bars, int period,
                         int warmup) const;
    void compute_atr_cpu(const float* high, const float* low, const float* close, float* output,
                         int num_bars, int period, int warmup) const;

    // CPU fallback: delegate to the C++ indicator classes.
    // These create a temporary Bar array from SoA data, instantiate the
    // indicator, call compute(), and extract the requested series.
    void compute_indicator_cpu(const IndicatorSpec& spec, const Bar* bars, int num_bars,
                               const float* bar_open, const float* bar_high, const float* bar_low,
                               const float* bar_close, const uint32_t* bar_volume,
                               const int64_t* bar_timestamp, float* output) const;

    // Anchored CPU fallback: compute on TF bars and broadcast to 30s.
    // Uses compute_anchored() from indicator.h.
    void compute_indicator_cpu_anchored(const IndicatorSpec& spec, const Bar* tf_bars,
                                        int tf_bar_count, const int* completed_map,
                                        int bar_start_30s, int num_bars_30s, float* output) const;

    // Helper: build a temporary Bar array from SoA arrays
    static void build_temp_bars(std::vector<Bar>& temp_bars, const float* bar_open,
                                const float* bar_high, const float* bar_low, const float* bar_close,
                                int num_bars);

#ifdef HAS_OPENCL
    // ---- OpenCL kernel management ----
    bool build_kernels(const CLContext& ctx);

    cl_program m_program    = nullptr;
    cl_kernel  m_ema_kernel = nullptr;
    cl_kernel  m_sma_kernel = nullptr;
    cl_kernel  m_rsi_kernel = nullptr;
    cl_kernel  m_atr_kernel = nullptr;

    // Additional GPU kernels (easy-to-port indicators)
    cl_kernel m_macd_kernel      = nullptr;
    cl_kernel m_bollinger_kernel = nullptr;
    cl_kernel m_keltner_kernel   = nullptr;
    cl_kernel m_cci_kernel       = nullptr;
    cl_kernel m_roc_kernel       = nullptr;
    cl_kernel m_obv_kernel       = nullptr;
    cl_kernel m_hma_kernel       = nullptr;

    bool m_kernels_built = false;

    void release_kernels();

    // ---- Pooled GPU buffer management ----
    // Persistent cl_mem objects reused across compute_with_bars() calls.
    // Only reallocated when a larger size is needed.
    struct PooledBuffer {
        cl_mem buf      = nullptr;
        size_t capacity = 0;  // current allocation size in bytes
    };

    // Ensure a pooled buffer has at least `needed` bytes.
    // Returns the cl_mem handle (may be newly allocated or reused).
    // On failure sets err and returns nullptr.
    cl_mem ensure_buffer(PooledBuffer& pb, size_t needed, cl_mem_flags flags, cl_context context,
                         cl_int& err);

    static void release_pooled(PooledBuffer& pb);
    void        release_pooled_buffers();

    // Shared input buffers (close, high, low)
    PooledBuffer m_pool_close;
    PooledBuffer m_pool_high;
    PooledBuffer m_pool_low;

    // Per-batch output and params buffers (reused across indicator types).
    //
    // Invariant: m_pool_params holds the params struct for the most-
    // recently-dispatched indicator batch. Every kernel path that sets
    // kernel arg(N, m_pool_params.buf) MUST first call ensure_buffer()
    // followed by clEnqueueWriteBuffer() to populate it this batch. The
    // buffer's contents are NOT persisted across batches / indicator types,
    // so relying on stale params across kernel invocations is a bug.
    PooledBuffer m_pool_output;
    PooledBuffer m_pool_params;

    // HMA temporary buffers
    PooledBuffer m_pool_hma_temp1;
    PooledBuffer m_pool_hma_temp2;

    // OBV volume buffer
    PooledBuffer m_pool_volume;
#endif
};
