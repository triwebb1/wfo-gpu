#include "cl_indicators.h"

#include <atomic>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Include all indicator headers for CPU fallback
#include "../indicators/ema_sma_cross.h"
#include "../indicators/rsi.h"
#include "../indicators/atr.h"
#include "../indicators/macd.h"
#include "../indicators/adx.h"
#include "../indicators/bollinger.h"
#include "../indicators/keltner.h"
#include "../indicators/stochastic.h"
#include "../indicators/cci.h"
#include "../indicators/roc.h"
#include "../indicators/vwap.h"
#include "../indicators/obv.h"
#include "../indicators/donchian.h"
#include "../indicators/pivot_points.h"
#include "../indicators/hma.h"
#include "../indicators/connors_rsi.h"

// ---------------------------------------------------------------------------
// Host-side period validation. Zero/negative periods would produce
// silent NaN/inf on the GPU (RSI div-by-zero, EMA alpha=2, ATR no init, ...).
// Currently masked by IndicatorSpec defaults + grid validation; this guard
// fails fast if a regression ever bypasses those.
// ---------------------------------------------------------------------------
static inline void validate_period(int period, const char* context) {
    (void)context;
    (void)period;  // only read by the assert
    assert(period >= 1 && "indicator period must be >= 1");
}

static inline void validate_spec_periods(const IndicatorSpec& spec) {
    switch (spec.type) {
        case IndicatorType::EMA_SMA_CROSS:
            validate_period(spec.fast_period, "EMA_SMA_CROSS.fast_period");
            validate_period(spec.slow_period, "EMA_SMA_CROSS.slow_period");
            break;
        case IndicatorType::RSI:
            validate_period(spec.rsi_period, "RSI.rsi_period");
            break;
        case IndicatorType::ATR:
            validate_period(spec.atr_period, "ATR.atr_period");
            break;
        case IndicatorType::MACD:
            validate_period(spec.macd_fast_period, "MACD.fast_period");
            validate_period(spec.macd_slow_period, "MACD.slow_period");
            validate_period(spec.macd_signal_period, "MACD.signal_period");
            break;
        case IndicatorType::ADX:
            validate_period(spec.adx_period, "ADX.adx_period");
            break;
        case IndicatorType::BOLLINGER:
            validate_period(spec.bollinger_period, "BOLLINGER.bollinger_period");
            break;
        case IndicatorType::KELTNER:
            validate_period(spec.keltner_ma_period, "KELTNER.ma_period");
            validate_period(spec.keltner_atr_period, "KELTNER.atr_period");
            break;
        case IndicatorType::STOCHASTIC:
            validate_period(spec.stoch_k_period, "STOCHASTIC.k_period");
            validate_period(spec.stoch_d_period, "STOCHASTIC.d_period");
            break;
        case IndicatorType::CCI:
            validate_period(spec.cci_period, "CCI.cci_period");
            break;
        case IndicatorType::ROC:
            validate_period(spec.roc_period, "ROC.roc_period");
            break;
        case IndicatorType::DONCHIAN:
            validate_period(spec.donchian_period, "DONCHIAN.donchian_period");
            break;
        case IndicatorType::HMA:
            validate_period(spec.hma_period, "HMA.hma_period");
            break;
        case IndicatorType::CONNORS_RSI:
            validate_period(spec.connors_rsi_period, "CONNORS_RSI.rsi_period");
            validate_period(spec.connors_streak_period, "CONNORS_RSI.streak_period");
            validate_period(spec.connors_percentile_period, "CONNORS_RSI.percentile_period");
            break;
        case IndicatorType::VWAP:
        case IndicatorType::OBV:
        case IndicatorType::PIVOT_POINTS:
        default:
            // No period-like int parameters to validate.
            break;
    }
}

// ---------------------------------------------------------------------------
// IndicatorSpec::from_params
// ---------------------------------------------------------------------------

IndicatorSpec IndicatorSpec::from_params(const IndicatorParams& params, int series_idx) {
    IndicatorSpec spec;
    spec.type = params.type;

    switch (params.type) {
        case IndicatorType::EMA_SMA_CROSS:
            spec.fast_ma_type = params.fast_ma_type;
            spec.fast_period  = params.fast_period;
            spec.slow_ma_type = params.slow_ma_type;
            spec.slow_period  = params.slow_period;
            spec.ma_series    = series_idx;  // 0 = fast, 1 = slow
            // Warmup = max of both MAs
            {
                int fast_wu      = (params.fast_ma_type == MAType::EMA) ? 3 * params.fast_period
                                                                        : params.fast_period - 1;
                int slow_wu      = (params.slow_ma_type == MAType::EMA) ? 3 * params.slow_period
                                                                        : params.slow_period - 1;
                spec.warmup_bars = std::max(fast_wu, slow_wu);
            }
            break;

        case IndicatorType::RSI:
            spec.rsi_period  = params.rsi_period;
            spec.warmup_bars = 3 * params.rsi_period;
            break;

        case IndicatorType::ATR:
            // ATR grid produces two series: short and long period ATR
            if (series_idx == 0) {
                spec.atr_period  = params.atr_short_period;
                spec.warmup_bars = 3 * params.atr_short_period;
            } else {
                spec.atr_period  = params.atr_long_period;
                spec.warmup_bars = 3 * params.atr_long_period;
            }
            break;

        case IndicatorType::MACD:
            spec.macd_fast_period   = params.macd_fast_period;
            spec.macd_slow_period   = params.macd_slow_period;
            spec.macd_signal_period = params.macd_signal_period;
            spec.macd_series        = series_idx;  // 0=MACD, 1=signal, 2=histogram
            spec.warmup_bars        = 3 * params.macd_slow_period + 3 * params.macd_signal_period;
            break;

        case IndicatorType::ADX:
            spec.adx_period  = params.adx_period;
            spec.adx_series  = series_idx;  // 0=ADX, 1=+DI, 2=-DI
            spec.warmup_bars = 4 * params.adx_period;
            break;

        case IndicatorType::BOLLINGER:
            spec.bollinger_period     = params.bollinger_period;
            spec.bollinger_multiplier = params.bollinger_multiplier;
            spec.bollinger_series     = series_idx;  // 0=middle, 1=upper, 2=lower
            spec.warmup_bars          = params.bollinger_period - 1;
            break;

        case IndicatorType::KELTNER:
            spec.keltner_ma_type    = params.keltner_ma_type;
            spec.keltner_ma_period  = params.keltner_ma_period;
            spec.keltner_atr_period = params.keltner_atr_period;
            spec.keltner_multiplier = params.keltner_multiplier;
            spec.keltner_series     = series_idx;  // 0=center, 1=upper, 2=lower
            {
                int atr_wu = 3 * params.keltner_atr_period;
                if (params.keltner_ma_type == MAType::EMA) {
                    spec.warmup_bars = std::max(3 * params.keltner_ma_period, atr_wu);
                } else {
                    spec.warmup_bars = std::max(params.keltner_ma_period, atr_wu);
                }
            }
            break;

        case IndicatorType::STOCHASTIC:
            spec.stoch_k_period = params.stoch_k_period;
            spec.stoch_d_period = params.stoch_d_period;
            spec.stoch_series   = series_idx;  // 0=%K, 1=%D
            spec.warmup_bars    = params.stoch_k_period + params.stoch_d_period;
            break;

        case IndicatorType::CCI:
            spec.cci_period  = params.cci_period;
            spec.warmup_bars = params.cci_period - 1;
            break;

        case IndicatorType::ROC:
            spec.roc_period  = params.roc_period;
            spec.warmup_bars = params.roc_period;
            break;

        case IndicatorType::VWAP:
            spec.vwap_band_mult = params.vwap_band_mult;
            spec.vwap_series    = series_idx;  // 0=VWAP, 1=upper, 2=lower
            spec.warmup_bars    = 0;           // VWAP resets each session
            break;

        case IndicatorType::OBV:
            spec.obv_smoothing = params.obv_smoothing;
            spec.warmup_bars   = (params.obv_smoothing > 0) ? (params.obv_smoothing - 1) : 0;
            break;

        case IndicatorType::DONCHIAN:
            spec.donchian_period = params.donchian_period;
            spec.donchian_series = series_idx;  // 0=upper, 1=lower
            spec.warmup_bars     = params.donchian_period - 1;
            break;

        case IndicatorType::PIVOT_POINTS:
            spec.pivot_type   = params.pivot_type;
            spec.pivot_series = series_idx;  // 0=PP,1=S1,2=S2,3=S3,4=R1,5=R2,6=R3
            spec.warmup_bars  = 2640;        // ~1 session at 30s bars
            break;

        case IndicatorType::HMA:
            spec.hma_period = params.hma_period;
            {
                int sqrt_p =
                    static_cast<int>(std::floor(std::sqrt(static_cast<float>(params.hma_period))));
                spec.warmup_bars = params.hma_period + sqrt_p;
            }
            break;

        case IndicatorType::CONNORS_RSI:
            spec.connors_rsi_period        = params.connors_rsi_period;
            spec.connors_streak_period     = params.connors_streak_period;
            spec.connors_percentile_period = params.connors_percentile_period;
            spec.warmup_bars =
                std::max({3 * params.connors_rsi_period, 3 * params.connors_streak_period,
                          params.connors_percentile_period});
            break;
    }

    return spec;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

CLIndicators::~CLIndicators() {
#ifdef HAS_OPENCL
    release_pooled_buffers();
    release_kernels();
#endif
}

// ---------------------------------------------------------------------------
// build_temp_bars -- create temporary AoS Bar array from SoA data
// Used for the indicator CPU fallback when full Bar data is not available.
// ---------------------------------------------------------------------------

void CLIndicators::build_temp_bars(std::vector<Bar>& temp_bars, const float* bar_open,
                                   const float* bar_high, const float* bar_low,
                                   const float* bar_close, int num_bars) {
    temp_bars.resize(num_bars);
    for (int i = 0; i < num_bars; ++i) {
        temp_bars[i].open      = bar_open[i];
        temp_bars[i].high      = bar_high[i];
        temp_bars[i].low       = bar_low[i];
        temp_bars[i].close     = bar_close[i];
        temp_bars[i]._pad      = 0;
        temp_bars[i].volume    = 0;
        temp_bars[i].timestamp = 0;
    }
}

// ---------------------------------------------------------------------------
// compute_indicator_cpu -- CPU fallback via C++ indicator classes
//
// Creates the appropriate indicator object, calls compute() on full Bar data,
// and extracts the requested series into the output buffer.
//
// When bars is non-null (from compute_with_bars), uses them directly.
// Otherwise builds temporary Bar data from SoA arrays.
// ---------------------------------------------------------------------------

void CLIndicators::compute_indicator_cpu(const IndicatorSpec& spec, const Bar* bars, int num_bars,
                                         const float* bar_open, const float* bar_high,
                                         const float* bar_low, const float* bar_close,
                                         const uint32_t* bar_volume, const int64_t* bar_timestamp,
                                         float* output) const {
    validate_spec_periods(spec);

    // If we have full Bar data, use it directly; otherwise build temp bars
    std::vector<Bar> temp_bars;
    const Bar*       compute_bars = bars;

    // Detect the silent-zero-volume / silent-zero-timestamp failure
    // mode. When called via the SoA entrypoint (bars == nullptr), indicators
    // that consume volume (VWAP, OBV) or session timestamps (VWAP session
    // resets, PIVOT_POINTS daily boundaries) would receive all-zeros and
    // produce undefined output. Fail loudly with NaN + stderr instead.
    const bool needs_volume =
        (spec.type == IndicatorType::VWAP) || (spec.type == IndicatorType::OBV);
    const bool needs_timestamp =
        (spec.type == IndicatorType::VWAP) || (spec.type == IndicatorType::PIVOT_POINTS);
    if (!bars) {
        if ((needs_volume && !bar_volume) || (needs_timestamp && !bar_timestamp)) {
            std::cerr << "[WARN] indicator requires " << (needs_volume ? "volume" : "")
                      << ((needs_volume && needs_timestamp) ? "+" : "")
                      << (needs_timestamp ? "timestamp" : "")
                      << " but SoA path provided none; emitting NaN" << std::endl;
            for (int i = 0; i < num_bars; ++i) {
                output[i] = std::numeric_limits<float>::quiet_NaN();
            }
            return;
        }
    }

    if (!compute_bars) {
        build_temp_bars(temp_bars, bar_open, bar_high, bar_low, bar_close, num_bars);
        // Fill volume and timestamp if available
        if (bar_volume) {
            for (int i = 0; i < num_bars; ++i) {
                temp_bars[i].volume = bar_volume[i];
            }
        }
        if (bar_timestamp) {
            for (int i = 0; i < num_bars; ++i) {
                temp_bars[i].timestamp = bar_timestamp[i];
            }
        }
        compute_bars = temp_bars.data();
    }

    // Allocate multi-series output buffer
    std::unique_ptr<Indicator> indicator;
    int                        total_series     = 0;
    int                        requested_series = 0;

    switch (spec.type) {
        case IndicatorType::MACD:
            indicator        = std::make_unique<MACD>(spec.macd_fast_period, spec.macd_slow_period,
                                                      spec.macd_signal_period);
            total_series     = 3;
            requested_series = spec.macd_series;
            break;

        case IndicatorType::ADX:
            indicator        = std::make_unique<ADX>(spec.adx_period);
            total_series     = 3;
            requested_series = spec.adx_series;
            break;

        case IndicatorType::BOLLINGER:
            indicator =
                std::make_unique<BollingerBands>(spec.bollinger_period, spec.bollinger_multiplier);
            total_series     = 3;
            requested_series = spec.bollinger_series;
            break;

        case IndicatorType::KELTNER:
            indicator =
                std::make_unique<KeltnerChannels>(spec.keltner_ma_type, spec.keltner_ma_period,
                                                  spec.keltner_atr_period, spec.keltner_multiplier);
            total_series     = 3;
            requested_series = spec.keltner_series;
            break;

        case IndicatorType::STOCHASTIC:
            indicator    = std::make_unique<Stochastic>(spec.stoch_k_period, spec.stoch_d_period);
            total_series = 2;
            requested_series = spec.stoch_series;
            break;

        case IndicatorType::CCI:
            indicator        = std::make_unique<CCI>(spec.cci_period);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::ROC:
            // ROC in the grid has no smoothing (raw ROC only)
            indicator        = std::make_unique<ROC>(spec.roc_period, 0);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::VWAP:
            indicator        = std::make_unique<VWAP>(spec.vwap_band_mult);
            total_series     = 3;
            requested_series = spec.vwap_series;
            break;

        case IndicatorType::OBV:
            indicator        = std::make_unique<OBV>(spec.obv_smoothing);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::DONCHIAN:
            indicator        = std::make_unique<DonchianChannel>(spec.donchian_period);
            total_series     = 2;
            requested_series = spec.donchian_series;
            break;

        case IndicatorType::PIVOT_POINTS:
            indicator        = std::make_unique<PivotPoints>(spec.pivot_type);
            total_series     = 7;
            requested_series = spec.pivot_series;
            break;

        case IndicatorType::HMA:
            indicator        = std::make_unique<HMA>(spec.hma_period);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::CONNORS_RSI:
            indicator =
                std::make_unique<ConnorsRSI>(spec.connors_rsi_period, spec.connors_streak_period,
                                             spec.connors_percentile_period);
            total_series     = 1;
            requested_series = 0;
            break;

        default:
            // EMA_SMA_CROSS, RSI, ATR handled by dedicated fast paths
            // Fill with NaN as safety
            for (int i = 0; i < num_bars; ++i) {
                output[i] = std::numeric_limits<float>::quiet_NaN();
            }
            return;
    }

    // Allocate the multi-series buffer, compute, and extract
    std::vector<float> multi_output(static_cast<size_t>(total_series) * num_bars);
    indicator->compute(compute_bars, 0, num_bars, multi_output.data());

    // Copy the requested series to output
    const float* series_start =
        multi_output.data() + static_cast<size_t>(requested_series) * num_bars;
    std::memcpy(output, series_start, sizeof(float) * num_bars);
}

// ---------------------------------------------------------------------------
// compute_indicator_cpu_anchored -- compute on TF bars, broadcast to 30s
//
// Creates the appropriate indicator object (same switch as compute_indicator_cpu),
// then uses compute_anchored() to compute on TF bars and broadcast back to
// 30s granularity via the completed-bar map.
// ---------------------------------------------------------------------------

void CLIndicators::compute_indicator_cpu_anchored(const IndicatorSpec& spec, const Bar* tf_bars,
                                                  int tf_bar_count, const int* completed_map,
                                                  int bar_start_30s, int num_bars_30s,
                                                  float* output) const {
    validate_spec_periods(spec);

    // Build the indicator object
    std::unique_ptr<Indicator> indicator;
    int                        total_series     = 0;
    int                        requested_series = 0;

    switch (spec.type) {
        case IndicatorType::EMA_SMA_CROSS:
            indicator        = std::make_unique<EMASMACross>(spec.fast_ma_type, spec.fast_period,
                                                             spec.slow_ma_type, spec.slow_period);
            total_series     = 2;
            requested_series = spec.ma_series;
            break;

        case IndicatorType::RSI:
            indicator        = std::make_unique<RSI>(spec.rsi_period);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::ATR:
            indicator        = std::make_unique<ATR>(spec.atr_period);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::MACD:
            indicator        = std::make_unique<MACD>(spec.macd_fast_period, spec.macd_slow_period,
                                                      spec.macd_signal_period);
            total_series     = 3;
            requested_series = spec.macd_series;
            break;

        case IndicatorType::ADX:
            indicator        = std::make_unique<ADX>(spec.adx_period);
            total_series     = 3;
            requested_series = spec.adx_series;
            break;

        case IndicatorType::BOLLINGER:
            indicator =
                std::make_unique<BollingerBands>(spec.bollinger_period, spec.bollinger_multiplier);
            total_series     = 3;
            requested_series = spec.bollinger_series;
            break;

        case IndicatorType::KELTNER:
            indicator =
                std::make_unique<KeltnerChannels>(spec.keltner_ma_type, spec.keltner_ma_period,
                                                  spec.keltner_atr_period, spec.keltner_multiplier);
            total_series     = 3;
            requested_series = spec.keltner_series;
            break;

        case IndicatorType::STOCHASTIC:
            indicator    = std::make_unique<Stochastic>(spec.stoch_k_period, spec.stoch_d_period);
            total_series = 2;
            requested_series = spec.stoch_series;
            break;

        case IndicatorType::CCI:
            indicator        = std::make_unique<CCI>(spec.cci_period);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::ROC:
            indicator        = std::make_unique<ROC>(spec.roc_period, 0);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::VWAP:
            indicator        = std::make_unique<VWAP>(spec.vwap_band_mult);
            total_series     = 3;
            requested_series = spec.vwap_series;
            break;

        case IndicatorType::OBV:
            indicator        = std::make_unique<OBV>(spec.obv_smoothing);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::DONCHIAN:
            indicator        = std::make_unique<DonchianChannel>(spec.donchian_period);
            total_series     = 2;
            requested_series = spec.donchian_series;
            break;

        case IndicatorType::PIVOT_POINTS:
            indicator        = std::make_unique<PivotPoints>(spec.pivot_type);
            total_series     = 7;
            requested_series = spec.pivot_series;
            break;

        case IndicatorType::HMA:
            indicator        = std::make_unique<HMA>(spec.hma_period);
            total_series     = 1;
            requested_series = 0;
            break;

        case IndicatorType::CONNORS_RSI:
            indicator =
                std::make_unique<ConnorsRSI>(spec.connors_rsi_period, spec.connors_streak_period,
                                             spec.connors_percentile_period);
            total_series     = 1;
            requested_series = 0;
            break;

        default:
            for (int i = 0; i < num_bars_30s; ++i) {
                output[i] = std::numeric_limits<float>::quiet_NaN();
            }
            return;
    }

    // Use compute_anchored: compute on TF bars, then broadcast to 30s
    int                bar_end_30s = bar_start_30s + num_bars_30s;
    std::vector<float> anchored_output(static_cast<size_t>(num_bars_30s) * total_series);
    compute_anchored(*indicator, tf_bars, tf_bar_count, completed_map, bar_start_30s, bar_end_30s,
                     anchored_output.data());

    // Extract the requested series
    const float* series_start =
        anchored_output.data() + static_cast<size_t>(requested_series) * num_bars_30s;
    std::memcpy(output, series_start, sizeof(float) * num_bars_30s);
}

// ---------------------------------------------------------------------------
// Pooled buffer management
// ---------------------------------------------------------------------------

#ifdef HAS_OPENCL

void CLIndicators::release_pooled(PooledBuffer& pb) {
    if (pb.buf) {
        clReleaseMemObject(pb.buf);
        pb.buf = nullptr;
    }
    pb.capacity = 0;
}

void CLIndicators::release_pooled_buffers() {
    release_pooled(m_pool_close);
    release_pooled(m_pool_high);
    release_pooled(m_pool_low);
    release_pooled(m_pool_output);
    release_pooled(m_pool_params);
    release_pooled(m_pool_hma_temp1);
    release_pooled(m_pool_hma_temp2);
    release_pooled(m_pool_volume);
}

cl_mem CLIndicators::ensure_buffer(PooledBuffer& pb, size_t needed, cl_mem_flags flags,
                                   cl_context context, cl_int& err) {
    err = CL_SUCCESS;
    if (pb.buf && pb.capacity >= needed) {
        return pb.buf;  // reuse existing buffer
    }
    // Need a new (larger) buffer
    if (pb.buf) {
        clReleaseMemObject(pb.buf);
        pb.buf      = nullptr;
        pb.capacity = 0;
    }
    pb.buf = clCreateBuffer(context, flags, needed, nullptr, &err);
    if (err == CL_SUCCESS) {
        pb.capacity = needed;
    }
    return pb.buf;
}

#endif  // HAS_OPENCL

// ---------------------------------------------------------------------------
// compute() -- main dispatcher (SoA only path, original interface)
// ---------------------------------------------------------------------------

void CLIndicators::compute(const CLContext& ctx, const float* bar_open, const float* bar_high,
                           const float* bar_low, const float* bar_close, int num_bars,
                           const std::vector<IndicatorSpec>& indicators,
                           std::vector<float*>&              output_series) {
    // Delegate to compute_with_bars with null Bar pointer
    compute_with_bars(ctx, nullptr, 0, bar_open, bar_high, bar_low, bar_close, num_bars, indicators,
                      output_series);
}

// ---------------------------------------------------------------------------
// compute_with_bars() -- extended dispatcher with full Bar data
// ---------------------------------------------------------------------------

namespace {
std::atomic<bool> g_indicators_on_device{false};
}

bool CLIndicators::default_on_device() {
    return g_indicators_on_device.load();
}

void CLIndicators::set_default_on_device(bool on) {
    g_indicators_on_device.store(on);
}

void CLIndicators::compute_with_bars(const CLContext& ctx, const Bar* bars, int bar_offset,
                                     const float* bar_open, const float* bar_high,
                                     const float* bar_low, const float* bar_close, int num_bars,
                                     const std::vector<IndicatorSpec>& indicators,
                                     std::vector<float*>&              output_series) {
    // bar_offset is reserved for future per-window anchoring and must be
    // zero under the current caller contract. All SoA pointers and `bars` are
    // expected to be aligned to the same start index (no independent offset).
    assert(bar_offset == 0 && "bar_offset must be 0 per current caller contract");
    (void)bar_offset;

    assert(output_series.size() == indicators.size());
    assert(num_bars > 0);

    for (const auto& spec : indicators) {
        validate_spec_periods(spec);
    }

#ifdef HAS_OPENCL
    // ---- Batched GPU dispatch: group indicators by type, one kernel launch per type ----
    bool use_gpu = m_on_device && build_kernels(ctx);
    if (use_gpu && ctx.is_available()) {
        cl_int           err;
        const size_t     buf_bytes = static_cast<size_t>(num_bars) * sizeof(float);
        cl_context       cl_ctx    = ctx.context();
        cl_command_queue queue     = ctx.queue();

        // Shared pooled input buffer for close prices
        ensure_buffer(m_pool_close, buf_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
        bool close_ok = (err == CL_SUCCESS && m_pool_close.buf != nullptr);
        if (close_ok) {
            err = clEnqueueWriteBuffer(queue, m_pool_close.buf, CL_FALSE, 0, buf_bytes, bar_close,
                                       0, nullptr, nullptr);
            close_ok = (err == CL_SUCCESS);
        }

        // Shared pooled input buffers for high, low (created on demand)
        bool hl_ok = false;

        bool need_hl = false;
        for (size_t i = 0; i < indicators.size(); ++i) {
            const auto t = indicators[i].type;
            if (t == IndicatorType::ATR || t == IndicatorType::KELTNER || t == IndicatorType::CCI) {
                need_hl = true;
                break;
            }
        }
        if (need_hl) {
            ensure_buffer(m_pool_high, buf_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
            if (err == CL_SUCCESS && m_pool_high.buf != nullptr) {
                err = clEnqueueWriteBuffer(queue, m_pool_high.buf, CL_FALSE, 0, buf_bytes, bar_high,
                                           0, nullptr, nullptr);
                if (err == CL_SUCCESS) {
                    ensure_buffer(m_pool_low, buf_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                    if (err == CL_SUCCESS && m_pool_low.buf != nullptr) {
                        err = clEnqueueWriteBuffer(queue, m_pool_low.buf, CL_FALSE, 0, buf_bytes,
                                                   bar_low, 0, nullptr, nullptr);
                        if (err == CL_SUCCESS) {
                            hl_ok = true;
                        }
                    }
                }
            }
        }

        // ---- Group indicators by kernel type ----
        // Each entry: original index in indicators[]
        // EMA_SMA_CROSS splits into EMA and SMA batches
        std::vector<size_t> ema_indices, sma_indices, rsi_indices, atr_indices;
        std::vector<size_t> macd_indices, bollinger_indices, keltner_indices;
        std::vector<size_t> cci_indices, roc_indices, obv_indices, hma_indices;
        std::vector<size_t> cpu_only_indices;

        for (size_t i = 0; i < indicators.size(); ++i) {
            const IndicatorSpec& spec = indicators[i];
            switch (spec.type) {
                case IndicatorType::EMA_SMA_CROSS: {
                    bool is_ema;
                    if (spec.ma_series == 0)
                        is_ema = (spec.fast_ma_type == MAType::EMA);
                    else
                        is_ema = (spec.slow_ma_type == MAType::EMA);

                    if (is_ema)
                        ema_indices.push_back(i);
                    else
                        sma_indices.push_back(i);
                    break;
                }
                case IndicatorType::RSI:
                    rsi_indices.push_back(i);
                    break;
                case IndicatorType::ATR:
                    atr_indices.push_back(i);
                    break;
                case IndicatorType::MACD:
                    macd_indices.push_back(i);
                    break;
                case IndicatorType::BOLLINGER:
                    bollinger_indices.push_back(i);
                    break;
                case IndicatorType::KELTNER:
                    keltner_indices.push_back(i);
                    break;
                case IndicatorType::CCI:
                    cci_indices.push_back(i);
                    break;
                case IndicatorType::ROC:
                    roc_indices.push_back(i);
                    break;
                case IndicatorType::OBV:
                    obv_indices.push_back(i);
                    break;
                case IndicatorType::HMA:
                    hma_indices.push_back(i);
                    break;
                default:
                    cpu_only_indices.push_back(i);
                    break;
            }
        }

        // Helper lambda: batch dispatch for close-only kernels with 2-int params
        // Used for EMA, SMA, RSI, ROC
        // Uses pooled m_pool_params and m_pool_output buffers.
        auto batch_close_2int = [&](cl_kernel kernel, const std::vector<size_t>& indices,
                                    auto pack_params_fn, auto cpu_fallback_fn) {
            if (indices.empty()) return;
            if (!close_ok) {
                for (size_t idx : indices) cpu_fallback_fn(idx);
                return;
            }
            size_t count = indices.size();

            // Pack params: {period, warmup_bars} per indicator
            struct Params2i {
                int period;
                int warmup_bars;
            };
            std::vector<Params2i> host_params(count);
            for (size_t k = 0; k < count; ++k) {
                pack_params_fn(indices[k], host_params[k]);
            }

            size_t params_bytes = count * sizeof(Params2i);
            ensure_buffer(m_pool_params, params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
            if (err != CL_SUCCESS || m_pool_params.buf == nullptr) {
                for (size_t idx : indices) cpu_fallback_fn(idx);
                return;
            }
            err = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_FALSE, 0, params_bytes,
                                       host_params.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                for (size_t idx : indices) cpu_fallback_fn(idx);
                return;
            }

            size_t out_bytes = count * buf_bytes;
            ensure_buffer(m_pool_output, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
            if (err != CL_SUCCESS || m_pool_output.buf == nullptr) {
                for (size_t idx : indices) cpu_fallback_fn(idx);
                return;
            }

            cl_int arg_err = CL_SUCCESS;
            arg_err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_pool_close.buf);
            arg_err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &m_pool_output.buf);
            arg_err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &m_pool_params.buf);
            arg_err |= clSetKernelArg(kernel, 3, sizeof(int), &num_bars);

            size_t global_size = count;
            if (arg_err != CL_SUCCESS) {
                err = arg_err;
            } else {
                err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0,
                                             nullptr, nullptr);
            }

            if (err == CL_SUCCESS) {
                // Read back batched output and scatter to caller's pointers
                std::vector<float> host_output(count * num_bars);
                cl_int             read_err =
                    clEnqueueReadBuffer(queue, m_pool_output.buf, CL_TRUE, 0, out_bytes,
                                        host_output.data(), 0, nullptr, nullptr);
                if (read_err == CL_SUCCESS) {
                    for (size_t k = 0; k < count; ++k) {
                        std::memcpy(output_series[indices[k]], host_output.data() + k * num_bars,
                                    buf_bytes);
                    }
                } else {
                    for (size_t idx : indices) cpu_fallback_fn(idx);
                }
            } else {
                for (size_t idx : indices) cpu_fallback_fn(idx);
            }
        };

        // ---- Batch EMA ----
        {
            struct P2i {
                int period;
                int warmup_bars;
            };
            batch_close_2int(
                m_ema_kernel, ema_indices,
                [&](size_t idx, auto& p) {
                    const auto& s = indicators[idx];
                    p.period      = (s.ma_series == 0) ? s.fast_period : s.slow_period;
                    p.warmup_bars = s.warmup_bars;
                },
                [&](size_t idx) {
                    const auto& s      = indicators[idx];
                    int         period = (s.ma_series == 0) ? s.fast_period : s.slow_period;
                    compute_ema_cpu(bar_close, output_series[idx], num_bars, period, s.warmup_bars);
                });
        }

        // ---- Batch SMA ----
        {
            batch_close_2int(
                m_sma_kernel, sma_indices,
                [&](size_t idx, auto& p) {
                    const auto& s = indicators[idx];
                    p.period      = (s.ma_series == 0) ? s.fast_period : s.slow_period;
                    p.warmup_bars = s.warmup_bars;
                },
                [&](size_t idx) {
                    const auto& s      = indicators[idx];
                    int         period = (s.ma_series == 0) ? s.fast_period : s.slow_period;
                    compute_sma_cpu(bar_close, output_series[idx], num_bars, period, s.warmup_bars);
                });
        }

        // ---- Batch RSI ----
        {
            batch_close_2int(
                m_rsi_kernel, rsi_indices,
                [&](size_t idx, auto& p) {
                    p.period      = indicators[idx].rsi_period;
                    p.warmup_bars = indicators[idx].warmup_bars;
                },
                [&](size_t idx) {
                    const auto& s = indicators[idx];
                    compute_rsi_cpu(bar_close, output_series[idx], num_bars, s.rsi_period,
                                    s.warmup_bars);
                });
        }

        // ---- Batch ROC ----
        {
            batch_close_2int(
                m_roc_kernel, roc_indices,
                [&](size_t idx, auto& p) {
                    p.period      = indicators[idx].roc_period;
                    p.warmup_bars = indicators[idx].warmup_bars;
                },
                [&](size_t idx) {
                    compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                          bar_low, bar_close, nullptr, nullptr, output_series[idx]);
                });
        }

        // ---- Batch ATR (needs high, low, close + 2-int params) ----
        if (!atr_indices.empty()) {
            if (!close_ok || !hl_ok) {
                for (size_t idx : atr_indices) {
                    const auto& s = indicators[idx];
                    compute_atr_cpu(bar_high, bar_low, bar_close, output_series[idx], num_bars,
                                    s.atr_period, s.warmup_bars);
                }
            } else {
                struct ATRParams {
                    int period;
                    int warmup_bars;
                };
                size_t                 count = atr_indices.size();
                std::vector<ATRParams> host_params(count);
                for (size_t k = 0; k < count; ++k) {
                    host_params[k].period      = indicators[atr_indices[k]].atr_period;
                    host_params[k].warmup_bars = indicators[atr_indices[k]].warmup_bars;
                }

                size_t params_bytes = count * sizeof(ATRParams);
                ensure_buffer(m_pool_params, params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                bool params_ok = (err == CL_SUCCESS && m_pool_params.buf != nullptr);
                if (params_ok) {
                    err = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_FALSE, 0, params_bytes,
                                               host_params.data(), 0, nullptr, nullptr);
                    params_ok = (err == CL_SUCCESS);
                }

                size_t out_bytes = count * buf_bytes;
                bool   output_ok = false;
                if (params_ok) {
                    ensure_buffer(m_pool_output, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    output_ok = (err == CL_SUCCESS && m_pool_output.buf != nullptr);
                }

                if (params_ok && output_ok) {
                    cl_int arg_err = CL_SUCCESS;
                    arg_err |= clSetKernelArg(m_atr_kernel, 0, sizeof(cl_mem), &m_pool_high.buf);
                    arg_err |= clSetKernelArg(m_atr_kernel, 1, sizeof(cl_mem), &m_pool_low.buf);
                    arg_err |= clSetKernelArg(m_atr_kernel, 2, sizeof(cl_mem), &m_pool_close.buf);
                    arg_err |= clSetKernelArg(m_atr_kernel, 3, sizeof(cl_mem), &m_pool_output.buf);
                    arg_err |= clSetKernelArg(m_atr_kernel, 4, sizeof(cl_mem), &m_pool_params.buf);
                    arg_err |= clSetKernelArg(m_atr_kernel, 5, sizeof(int), &num_bars);

                    size_t global_size = count;
                    bool   launch_ok   = false;
                    if (arg_err == CL_SUCCESS) {
                        err = clEnqueueNDRangeKernel(queue, m_atr_kernel, 1, nullptr, &global_size,
                                                     nullptr, 0, nullptr, nullptr);
                        launch_ok = (err == CL_SUCCESS);
                    }

                    if (launch_ok) {
                        std::vector<float> host_output(count * num_bars);
                        cl_int             read_err =
                            clEnqueueReadBuffer(queue, m_pool_output.buf, CL_TRUE, 0, out_bytes,
                                                host_output.data(), 0, nullptr, nullptr);
                        if (read_err == CL_SUCCESS) {
                            for (size_t k = 0; k < count; ++k) {
                                std::memcpy(output_series[atr_indices[k]],
                                            host_output.data() + k * num_bars, buf_bytes);
                            }
                        } else {
                            for (size_t idx : atr_indices) {
                                const auto& s = indicators[idx];
                                compute_atr_cpu(bar_high, bar_low, bar_close, output_series[idx],
                                                num_bars, s.atr_period, s.warmup_bars);
                            }
                        }
                    } else {
                        for (size_t idx : atr_indices) {
                            const auto& s = indicators[idx];
                            compute_atr_cpu(bar_high, bar_low, bar_close, output_series[idx],
                                            num_bars, s.atr_period, s.warmup_bars);
                        }
                    }
                } else {
                    for (size_t idx : atr_indices) {
                        const auto& s = indicators[idx];
                        compute_atr_cpu(bar_high, bar_low, bar_close, output_series[idx], num_bars,
                                        s.atr_period, s.warmup_bars);
                    }
                }
            }
        }

        // ---- Batch MACD ----
        if (!macd_indices.empty()) {
            if (!close_ok) {
                for (size_t idx : macd_indices)
                    compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                          bar_low, bar_close, nullptr, nullptr, output_series[idx]);
            } else {
                struct MACDParams {
                    int fast_period;
                    int slow_period;
                    int signal_period;
                    int series_idx;
                    int warmup_bars;
                    int _pad0;
                };
                size_t                  count = macd_indices.size();
                std::vector<MACDParams> host_params(count);
                for (size_t k = 0; k < count; ++k) {
                    const auto& s  = indicators[macd_indices[k]];
                    host_params[k] = {s.macd_fast_period, s.macd_slow_period, s.macd_signal_period,
                                      s.macd_series,      s.warmup_bars,      0};
                }

                size_t params_bytes = count * sizeof(MACDParams);
                ensure_buffer(m_pool_params, params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                size_t out_bytes = count * buf_bytes;
                bool   ok        = (err == CL_SUCCESS && m_pool_params.buf != nullptr);
                if (ok) {
                    err = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_FALSE, 0, params_bytes,
                                               host_params.data(), 0, nullptr, nullptr);
                    ok  = (err == CL_SUCCESS);
                }
                if (ok) {
                    ensure_buffer(m_pool_output, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    ok = (err == CL_SUCCESS && m_pool_output.buf != nullptr);
                }

                if (ok) {
                    cl_int arg_err = CL_SUCCESS;
                    arg_err |= clSetKernelArg(m_macd_kernel, 0, sizeof(cl_mem), &m_pool_close.buf);
                    arg_err |= clSetKernelArg(m_macd_kernel, 1, sizeof(cl_mem), &m_pool_output.buf);
                    arg_err |= clSetKernelArg(m_macd_kernel, 2, sizeof(cl_mem), &m_pool_params.buf);
                    arg_err |= clSetKernelArg(m_macd_kernel, 3, sizeof(int), &num_bars);

                    size_t global_size = count;
                    bool   launch_ok   = false;
                    if (arg_err == CL_SUCCESS) {
                        err = clEnqueueNDRangeKernel(queue, m_macd_kernel, 1, nullptr, &global_size,
                                                     nullptr, 0, nullptr, nullptr);
                        launch_ok = (err == CL_SUCCESS);
                    }

                    if (launch_ok) {
                        std::vector<float> host_output(count * num_bars);
                        cl_int             read_err =
                            clEnqueueReadBuffer(queue, m_pool_output.buf, CL_TRUE, 0, out_bytes,
                                                host_output.data(), 0, nullptr, nullptr);
                        if (read_err == CL_SUCCESS) {
                            for (size_t k = 0; k < count; ++k)
                                std::memcpy(output_series[macd_indices[k]],
                                            host_output.data() + k * num_bars, buf_bytes);
                        } else {
                            for (size_t idx : macd_indices)
                                compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                      bar_high, bar_low, bar_close, nullptr,
                                                      nullptr, output_series[idx]);
                        }
                    } else {
                        for (size_t idx : macd_indices)
                            compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                  bar_high, bar_low, bar_close, nullptr, nullptr,
                                                  output_series[idx]);
                    }
                } else {
                    for (size_t idx : macd_indices)
                        compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                              bar_low, bar_close, nullptr, nullptr,
                                              output_series[idx]);
                }
            }
        }

        // ---- Batch Bollinger ----
        if (!bollinger_indices.empty()) {
            if (!close_ok) {
                for (size_t idx : bollinger_indices)
                    compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                          bar_low, bar_close, nullptr, nullptr, output_series[idx]);
            } else {
                struct BollingerParams {
                    int   period;
                    float multiplier;
                    int   series_idx;
                    int   warmup_bars;
                };
                size_t                       count = bollinger_indices.size();
                std::vector<BollingerParams> host_params(count);
                for (size_t k = 0; k < count; ++k) {
                    const auto& s  = indicators[bollinger_indices[k]];
                    host_params[k] = {s.bollinger_period, s.bollinger_multiplier,
                                      s.bollinger_series, s.warmup_bars};
                }

                size_t params_bytes = count * sizeof(BollingerParams);
                ensure_buffer(m_pool_params, params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                size_t out_bytes = count * buf_bytes;
                bool   ok        = (err == CL_SUCCESS && m_pool_params.buf != nullptr);
                if (ok) {
                    err = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_FALSE, 0, params_bytes,
                                               host_params.data(), 0, nullptr, nullptr);
                    ok  = (err == CL_SUCCESS);
                }
                if (ok) {
                    ensure_buffer(m_pool_output, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    ok = (err == CL_SUCCESS && m_pool_output.buf != nullptr);
                }

                if (ok) {
                    cl_int arg_err = CL_SUCCESS;
                    arg_err |=
                        clSetKernelArg(m_bollinger_kernel, 0, sizeof(cl_mem), &m_pool_close.buf);
                    arg_err |=
                        clSetKernelArg(m_bollinger_kernel, 1, sizeof(cl_mem), &m_pool_output.buf);
                    arg_err |=
                        clSetKernelArg(m_bollinger_kernel, 2, sizeof(cl_mem), &m_pool_params.buf);
                    arg_err |= clSetKernelArg(m_bollinger_kernel, 3, sizeof(int), &num_bars);

                    size_t global_size = count;
                    bool   launch_ok   = false;
                    if (arg_err == CL_SUCCESS) {
                        err       = clEnqueueNDRangeKernel(queue, m_bollinger_kernel, 1, nullptr,
                                                           &global_size, nullptr, 0, nullptr, nullptr);
                        launch_ok = (err == CL_SUCCESS);
                    }

                    if (launch_ok) {
                        std::vector<float> host_output(count * num_bars);
                        cl_int             read_err =
                            clEnqueueReadBuffer(queue, m_pool_output.buf, CL_TRUE, 0, out_bytes,
                                                host_output.data(), 0, nullptr, nullptr);
                        if (read_err == CL_SUCCESS) {
                            for (size_t k = 0; k < count; ++k)
                                std::memcpy(output_series[bollinger_indices[k]],
                                            host_output.data() + k * num_bars, buf_bytes);
                        } else {
                            for (size_t idx : bollinger_indices)
                                compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                      bar_high, bar_low, bar_close, nullptr,
                                                      nullptr, output_series[idx]);
                        }
                    } else {
                        for (size_t idx : bollinger_indices)
                            compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                  bar_high, bar_low, bar_close, nullptr, nullptr,
                                                  output_series[idx]);
                    }
                } else {
                    for (size_t idx : bollinger_indices)
                        compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                              bar_low, bar_close, nullptr, nullptr,
                                              output_series[idx]);
                }
            }
        }

        // ---- Batch Keltner (needs high, low, close) ----
        if (!keltner_indices.empty()) {
            // Need high/low/close; ensure pooled hl buffers on demand if not already
            bool kelt_hl_ok = hl_ok;
            if (!kelt_hl_ok && close_ok) {
                ensure_buffer(m_pool_high, buf_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                if (err == CL_SUCCESS && m_pool_high.buf != nullptr) {
                    err = clEnqueueWriteBuffer(queue, m_pool_high.buf, CL_FALSE, 0, buf_bytes,
                                               bar_high, 0, nullptr, nullptr);
                    if (err == CL_SUCCESS) {
                        ensure_buffer(m_pool_low, buf_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                        if (err == CL_SUCCESS && m_pool_low.buf != nullptr) {
                            err = clEnqueueWriteBuffer(queue, m_pool_low.buf, CL_FALSE, 0,
                                                       buf_bytes, bar_low, 0, nullptr, nullptr);
                            if (err == CL_SUCCESS) {
                                kelt_hl_ok = true;
                                hl_ok      = true;  // mark as available for subsequent batches
                            }
                        }
                    }
                }
            }

            if (!close_ok || !kelt_hl_ok) {
                for (size_t idx : keltner_indices)
                    compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                          bar_low, bar_close, nullptr, nullptr, output_series[idx]);
            } else {
                struct KeltnerParams {
                    int   ma_type_int;
                    int   ma_period;
                    int   atr_period;
                    float atr_multiplier;
                    int   series_idx;
                    int   warmup_bars;
                    int   _pad0;
                    int   _pad1;
                };
                size_t                     count = keltner_indices.size();
                std::vector<KeltnerParams> host_params(count);
                for (size_t k = 0; k < count; ++k) {
                    const auto& s  = indicators[keltner_indices[k]];
                    host_params[k] = {(s.keltner_ma_type == MAType::EMA) ? 0 : 1,
                                      s.keltner_ma_period,
                                      s.keltner_atr_period,
                                      s.keltner_multiplier,
                                      s.keltner_series,
                                      s.warmup_bars,
                                      0,
                                      0};
                }

                size_t params_bytes = count * sizeof(KeltnerParams);
                ensure_buffer(m_pool_params, params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                size_t out_bytes = count * buf_bytes;
                bool   ok        = (err == CL_SUCCESS && m_pool_params.buf != nullptr);
                if (ok) {
                    err = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_FALSE, 0, params_bytes,
                                               host_params.data(), 0, nullptr, nullptr);
                    ok  = (err == CL_SUCCESS);
                }
                if (ok) {
                    ensure_buffer(m_pool_output, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    ok = (err == CL_SUCCESS && m_pool_output.buf != nullptr);
                }

                if (ok) {
                    cl_int arg_err = CL_SUCCESS;
                    arg_err |=
                        clSetKernelArg(m_keltner_kernel, 0, sizeof(cl_mem), &m_pool_high.buf);
                    arg_err |= clSetKernelArg(m_keltner_kernel, 1, sizeof(cl_mem), &m_pool_low.buf);
                    arg_err |=
                        clSetKernelArg(m_keltner_kernel, 2, sizeof(cl_mem), &m_pool_close.buf);
                    arg_err |=
                        clSetKernelArg(m_keltner_kernel, 3, sizeof(cl_mem), &m_pool_output.buf);
                    arg_err |=
                        clSetKernelArg(m_keltner_kernel, 4, sizeof(cl_mem), &m_pool_params.buf);
                    arg_err |= clSetKernelArg(m_keltner_kernel, 5, sizeof(int), &num_bars);

                    size_t global_size = count;
                    bool   launch_ok   = false;
                    if (arg_err == CL_SUCCESS) {
                        err       = clEnqueueNDRangeKernel(queue, m_keltner_kernel, 1, nullptr,
                                                           &global_size, nullptr, 0, nullptr, nullptr);
                        launch_ok = (err == CL_SUCCESS);
                    }

                    if (launch_ok) {
                        std::vector<float> host_output(count * num_bars);
                        cl_int             read_err =
                            clEnqueueReadBuffer(queue, m_pool_output.buf, CL_TRUE, 0, out_bytes,
                                                host_output.data(), 0, nullptr, nullptr);
                        if (read_err == CL_SUCCESS) {
                            for (size_t k = 0; k < count; ++k)
                                std::memcpy(output_series[keltner_indices[k]],
                                            host_output.data() + k * num_bars, buf_bytes);
                        } else {
                            for (size_t idx : keltner_indices)
                                compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                      bar_high, bar_low, bar_close, nullptr,
                                                      nullptr, output_series[idx]);
                        }
                    } else {
                        for (size_t idx : keltner_indices)
                            compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                  bar_high, bar_low, bar_close, nullptr, nullptr,
                                                  output_series[idx]);
                    }
                } else {
                    for (size_t idx : keltner_indices)
                        compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                              bar_low, bar_close, nullptr, nullptr,
                                              output_series[idx]);
                }
            }
        }

        // ---- Batch CCI (needs high, low, close) ----
        if (!cci_indices.empty()) {
            bool cci_hl_ok = hl_ok;
            if (!cci_hl_ok && close_ok) {
                ensure_buffer(m_pool_high, buf_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                if (err == CL_SUCCESS && m_pool_high.buf != nullptr) {
                    err = clEnqueueWriteBuffer(queue, m_pool_high.buf, CL_FALSE, 0, buf_bytes,
                                               bar_high, 0, nullptr, nullptr);
                    if (err == CL_SUCCESS) {
                        ensure_buffer(m_pool_low, buf_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                        if (err == CL_SUCCESS && m_pool_low.buf != nullptr) {
                            err = clEnqueueWriteBuffer(queue, m_pool_low.buf, CL_FALSE, 0,
                                                       buf_bytes, bar_low, 0, nullptr, nullptr);
                            if (err == CL_SUCCESS) {
                                cci_hl_ok = true;
                                hl_ok     = true;
                            }
                        }
                    }
                }
            }

            if (!close_ok || !cci_hl_ok) {
                for (size_t idx : cci_indices)
                    compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                          bar_low, bar_close, nullptr, nullptr, output_series[idx]);
            } else {
                struct CCIParams {
                    int period;
                    int warmup_bars;
                };
                size_t                 count = cci_indices.size();
                std::vector<CCIParams> host_params(count);
                for (size_t k = 0; k < count; ++k) {
                    host_params[k] = {indicators[cci_indices[k]].cci_period,
                                      indicators[cci_indices[k]].warmup_bars};
                }

                size_t params_bytes = count * sizeof(CCIParams);
                ensure_buffer(m_pool_params, params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                size_t out_bytes = count * buf_bytes;
                bool   ok        = (err == CL_SUCCESS && m_pool_params.buf != nullptr);
                if (ok) {
                    err = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_FALSE, 0, params_bytes,
                                               host_params.data(), 0, nullptr, nullptr);
                    ok  = (err == CL_SUCCESS);
                }
                if (ok) {
                    ensure_buffer(m_pool_output, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    ok = (err == CL_SUCCESS && m_pool_output.buf != nullptr);
                }

                if (ok) {
                    cl_int arg_err = CL_SUCCESS;
                    arg_err |= clSetKernelArg(m_cci_kernel, 0, sizeof(cl_mem), &m_pool_high.buf);
                    arg_err |= clSetKernelArg(m_cci_kernel, 1, sizeof(cl_mem), &m_pool_low.buf);
                    arg_err |= clSetKernelArg(m_cci_kernel, 2, sizeof(cl_mem), &m_pool_close.buf);
                    arg_err |= clSetKernelArg(m_cci_kernel, 3, sizeof(cl_mem), &m_pool_output.buf);
                    arg_err |= clSetKernelArg(m_cci_kernel, 4, sizeof(cl_mem), &m_pool_params.buf);
                    arg_err |= clSetKernelArg(m_cci_kernel, 5, sizeof(int), &num_bars);

                    size_t global_size = count;
                    bool   launch_ok   = false;
                    if (arg_err == CL_SUCCESS) {
                        err = clEnqueueNDRangeKernel(queue, m_cci_kernel, 1, nullptr, &global_size,
                                                     nullptr, 0, nullptr, nullptr);
                        launch_ok = (err == CL_SUCCESS);
                    }

                    if (launch_ok) {
                        std::vector<float> host_output(count * num_bars);
                        cl_int             read_err =
                            clEnqueueReadBuffer(queue, m_pool_output.buf, CL_TRUE, 0, out_bytes,
                                                host_output.data(), 0, nullptr, nullptr);
                        if (read_err == CL_SUCCESS) {
                            for (size_t k = 0; k < count; ++k)
                                std::memcpy(output_series[cci_indices[k]],
                                            host_output.data() + k * num_bars, buf_bytes);
                        } else {
                            for (size_t idx : cci_indices)
                                compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                      bar_high, bar_low, bar_close, nullptr,
                                                      nullptr, output_series[idx]);
                        }
                    } else {
                        for (size_t idx : cci_indices)
                            compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                  bar_high, bar_low, bar_close, nullptr, nullptr,
                                                  output_series[idx]);
                    }
                } else {
                    for (size_t idx : cci_indices)
                        compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                              bar_low, bar_close, nullptr, nullptr,
                                              output_series[idx]);
                }
            }
        }

        // ---- Batch OBV (needs close + volume) ----
        if (!obv_indices.empty()) {
            if (!close_ok || !bars) {
                for (size_t idx : obv_indices)
                    compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                          bar_low, bar_close, nullptr, nullptr, output_series[idx]);
            } else {
                // Extract volume once for all OBV indicators
                std::vector<uint32_t> vol_data(num_bars);
                for (int vi = 0; vi < num_bars; ++vi) vol_data[vi] = bars[vi].volume;

                size_t vol_bytes = static_cast<size_t>(num_bars) * sizeof(uint32_t);
                ensure_buffer(m_pool_volume, vol_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                bool vol_ok = (err == CL_SUCCESS && m_pool_volume.buf != nullptr);
                if (vol_ok) {
                    err    = clEnqueueWriteBuffer(queue, m_pool_volume.buf, CL_FALSE, 0, vol_bytes,
                                                  vol_data.data(), 0, nullptr, nullptr);
                    vol_ok = (err == CL_SUCCESS);
                }

                struct OBVParams {
                    int smoothing_period;
                    int warmup_bars;
                };
                size_t                 count = obv_indices.size();
                std::vector<OBVParams> host_params(count);
                for (size_t k = 0; k < count; ++k) {
                    host_params[k] = {indicators[obv_indices[k]].obv_smoothing,
                                      indicators[obv_indices[k]].warmup_bars};
                }

                bool params_ok = false;
                if (vol_ok) {
                    size_t params_bytes = count * sizeof(OBVParams);
                    ensure_buffer(m_pool_params, params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                    params_ok = (err == CL_SUCCESS && m_pool_params.buf != nullptr);
                    if (params_ok) {
                        err       = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_FALSE, 0,
                                                         params_bytes, host_params.data(), 0, nullptr,
                                                         nullptr);
                        params_ok = (err == CL_SUCCESS);
                    }
                }

                size_t out_bytes = count * buf_bytes;
                bool   output_ok = false;
                if (params_ok) {
                    ensure_buffer(m_pool_output, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    output_ok = (err == CL_SUCCESS && m_pool_output.buf != nullptr);
                }

                if (vol_ok && params_ok && output_ok) {
                    cl_int arg_err = CL_SUCCESS;
                    arg_err |= clSetKernelArg(m_obv_kernel, 0, sizeof(cl_mem), &m_pool_close.buf);
                    arg_err |= clSetKernelArg(m_obv_kernel, 1, sizeof(cl_mem), &m_pool_volume.buf);
                    arg_err |= clSetKernelArg(m_obv_kernel, 2, sizeof(cl_mem), &m_pool_output.buf);
                    arg_err |= clSetKernelArg(m_obv_kernel, 3, sizeof(cl_mem), &m_pool_params.buf);
                    arg_err |= clSetKernelArg(m_obv_kernel, 4, sizeof(int), &num_bars);

                    size_t global_size = count;
                    bool   launch_ok   = false;
                    if (arg_err == CL_SUCCESS) {
                        err = clEnqueueNDRangeKernel(queue, m_obv_kernel, 1, nullptr, &global_size,
                                                     nullptr, 0, nullptr, nullptr);
                        launch_ok = (err == CL_SUCCESS);
                    }

                    if (launch_ok) {
                        std::vector<float> host_output(count * num_bars);
                        cl_int             read_err =
                            clEnqueueReadBuffer(queue, m_pool_output.buf, CL_TRUE, 0, out_bytes,
                                                host_output.data(), 0, nullptr, nullptr);
                        if (read_err == CL_SUCCESS) {
                            for (size_t k = 0; k < count; ++k)
                                std::memcpy(output_series[obv_indices[k]],
                                            host_output.data() + k * num_bars, buf_bytes);
                        } else {
                            for (size_t idx : obv_indices)
                                compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                      bar_high, bar_low, bar_close, nullptr,
                                                      nullptr, output_series[idx]);
                        }
                    } else {
                        for (size_t idx : obv_indices)
                            compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                  bar_high, bar_low, bar_close, nullptr, nullptr,
                                                  output_series[idx]);
                    }
                } else {
                    for (size_t idx : obv_indices)
                        compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                              bar_low, bar_close, nullptr, nullptr,
                                              output_series[idx]);
                }
            }
        }

        // ---- Batch HMA (needs close + 2 temp buffers) ----
        if (!hma_indices.empty()) {
            if (!close_ok) {
                for (size_t idx : hma_indices)
                    compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                          bar_low, bar_close, nullptr, nullptr, output_series[idx]);
            } else {
                struct HMAParams {
                    int period;
                    int warmup_bars;
                };
                size_t                 count = hma_indices.size();
                std::vector<HMAParams> host_params(count);
                for (size_t k = 0; k < count; ++k) {
                    host_params[k] = {indicators[hma_indices[k]].hma_period,
                                      indicators[hma_indices[k]].warmup_bars};
                }

                size_t params_bytes = count * sizeof(HMAParams);
                ensure_buffer(m_pool_params, params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
                size_t out_bytes = count * buf_bytes;
                bool   ok        = (err == CL_SUCCESS && m_pool_params.buf != nullptr);
                if (ok) {
                    err = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_FALSE, 0, params_bytes,
                                               host_params.data(), 0, nullptr, nullptr);
                    ok  = (err == CL_SUCCESS);
                }
                if (ok) {
                    ensure_buffer(m_pool_output, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    ok = (err == CL_SUCCESS && m_pool_output.buf != nullptr);
                }
                if (ok) {
                    ensure_buffer(m_pool_hma_temp1, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    ok = (err == CL_SUCCESS && m_pool_hma_temp1.buf != nullptr);
                }
                if (ok) {
                    ensure_buffer(m_pool_hma_temp2, out_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
                    ok = (err == CL_SUCCESS && m_pool_hma_temp2.buf != nullptr);
                }

                if (ok) {
                    cl_int arg_err = CL_SUCCESS;
                    arg_err |= clSetKernelArg(m_hma_kernel, 0, sizeof(cl_mem), &m_pool_close.buf);
                    arg_err |= clSetKernelArg(m_hma_kernel, 1, sizeof(cl_mem), &m_pool_output.buf);
                    arg_err |=
                        clSetKernelArg(m_hma_kernel, 2, sizeof(cl_mem), &m_pool_hma_temp1.buf);
                    arg_err |=
                        clSetKernelArg(m_hma_kernel, 3, sizeof(cl_mem), &m_pool_hma_temp2.buf);
                    arg_err |= clSetKernelArg(m_hma_kernel, 4, sizeof(cl_mem), &m_pool_params.buf);
                    arg_err |= clSetKernelArg(m_hma_kernel, 5, sizeof(int), &num_bars);

                    size_t global_size = count;
                    bool   launch_ok   = false;
                    if (arg_err == CL_SUCCESS) {
                        err = clEnqueueNDRangeKernel(queue, m_hma_kernel, 1, nullptr, &global_size,
                                                     nullptr, 0, nullptr, nullptr);
                        launch_ok = (err == CL_SUCCESS);
                    }

                    if (launch_ok) {
                        std::vector<float> host_output(count * num_bars);
                        cl_int             read_err =
                            clEnqueueReadBuffer(queue, m_pool_output.buf, CL_TRUE, 0, out_bytes,
                                                host_output.data(), 0, nullptr, nullptr);
                        if (read_err == CL_SUCCESS) {
                            for (size_t k = 0; k < count; ++k)
                                std::memcpy(output_series[hma_indices[k]],
                                            host_output.data() + k * num_bars, buf_bytes);
                        } else {
                            for (size_t idx : hma_indices)
                                compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                      bar_high, bar_low, bar_close, nullptr,
                                                      nullptr, output_series[idx]);
                        }
                    } else {
                        for (size_t idx : hma_indices)
                            compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open,
                                                  bar_high, bar_low, bar_close, nullptr, nullptr,
                                                  output_series[idx]);
                    }
                } else {
                    for (size_t idx : hma_indices)
                        compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high,
                                              bar_low, bar_close, nullptr, nullptr,
                                              output_series[idx]);
                }
            }
        }

        // ---- CPU-only indicators ----
        for (size_t idx : cpu_only_indices) {
            compute_indicator_cpu(indicators[idx], bars, num_bars, bar_open, bar_high, bar_low,
                                  bar_close, nullptr, nullptr, output_series[idx]);
        }

        // Shared input buffers are managed by pooled buffer lifecycle
        // (released in destructor or reused across calls).

        return;
    }
#else
    (void)ctx;
#endif  // HAS_OPENCL

    // ---- CPU fallback: compute each indicator sequentially ----
    for (size_t i = 0; i < indicators.size(); ++i) {
        const IndicatorSpec& spec = indicators[i];
        float*               out  = output_series[i];

        switch (spec.type) {
            case IndicatorType::EMA_SMA_CROSS: {
                // Compute the requested MA series (fast=0 or slow=1)
                if (spec.ma_series == 0) {
                    // Fast MA
                    if (spec.fast_ma_type == MAType::EMA) {
                        compute_ema_cpu(bar_close, out, num_bars, spec.fast_period,
                                        spec.warmup_bars);
                    } else {
                        compute_sma_cpu(bar_close, out, num_bars, spec.fast_period,
                                        spec.warmup_bars);
                    }
                } else {
                    // Slow MA
                    if (spec.slow_ma_type == MAType::EMA) {
                        compute_ema_cpu(bar_close, out, num_bars, spec.slow_period,
                                        spec.warmup_bars);
                    } else {
                        compute_sma_cpu(bar_close, out, num_bars, spec.slow_period,
                                        spec.warmup_bars);
                    }
                }
                break;
            }

            case IndicatorType::RSI:
                compute_rsi_cpu(bar_close, out, num_bars, spec.rsi_period, spec.warmup_bars);
                break;

            case IndicatorType::ATR:
                compute_atr_cpu(bar_high, bar_low, bar_close, out, num_bars, spec.atr_period,
                                spec.warmup_bars);
                break;

            // Remaining indicators: delegate to the C++ indicator classes
            case IndicatorType::MACD:
            case IndicatorType::ADX:
            case IndicatorType::BOLLINGER:
            case IndicatorType::KELTNER:
            case IndicatorType::STOCHASTIC:
            case IndicatorType::CCI:
            case IndicatorType::ROC:
            case IndicatorType::VWAP:
            case IndicatorType::OBV:
            case IndicatorType::DONCHIAN:
            case IndicatorType::PIVOT_POINTS:
            case IndicatorType::HMA:
            case IndicatorType::CONNORS_RSI:
                compute_indicator_cpu(spec, bars, num_bars, bar_open, bar_high, bar_low, bar_close,
                                      nullptr, nullptr, out);
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// compute_with_bars_anchored() -- anchored-aware dispatcher
//
// For each spec, if window_mode != ROLLING, computes the indicator on the
// higher-timeframe bars from MarketData and broadcasts back to 30s via
// completed_map, so each base bar sees only finished TF bars.  Otherwise
// falls through to the standard 30s compute path.
// ---------------------------------------------------------------------------

void CLIndicators::compute_with_bars_anchored(const CLContext& ctx, const Bar* bars_30s,
                                              int bar_start_30s, const float* bar_open,
                                              const float* bar_high, const float* bar_low,
                                              const float* bar_close, int num_bars_30s,
                                              const MarketData&                 market_data,
                                              const std::vector<IndicatorSpec>& indicators,
                                              std::vector<float*>&              output_series) {
    assert(output_series.size() == indicators.size());
    assert(num_bars_30s > 0);

    for (const auto& spec : indicators) {
        validate_spec_periods(spec);
    }

    // Partition specs into ROLLING and ANCHORED groups.
    //
    // ROLLING specs are batched and forwarded to compute_with_bars(), which
    // internally routes GPU-supported types (EMA_SMA_CROSS, RSI, ATR, MACD,
    // BOLLINGER, KELTNER, CCI, ROC, OBV, HMA) to the OpenCL kernel path and
    // CPU-only types (ADX, STOCHASTIC, DONCHIAN, VWAP, PIVOT_POINTS,
    // CONNORS_RSI) to the CPU path.  The gpu/cpu partition logic
    // lives entirely inside compute_with_bars() to avoid duplication.
    //
    // ANCHORED specs remain on the CPU-anchored path (compute on TF bars,
    // broadcast back to 30s).  No GPU anchored path exists.

    std::vector<IndicatorSpec> rolling_specs;
    std::vector<float*>        rolling_outputs;
    rolling_specs.reserve(indicators.size());
    rolling_outputs.reserve(indicators.size());

    for (size_t i = 0; i < indicators.size(); ++i) {
        const IndicatorSpec& spec = indicators[i];
        float*               out  = output_series[i];

        if (spec.window_mode == WindowMode::ROLLING) {
            rolling_specs.push_back(spec);
            rolling_outputs.push_back(out);
        } else {
            // Anchored path: compute on TF bars, broadcast to 30s.
            // Use try_timeframe() so an unknown timeframe produces a
            // graceful NaN fill instead of an exception unwinding the
            // dispatcher. Spec validation still catches real misconfigs
            // earlier; this is defense-in-depth.
            int                              tf_sec = window_mode_seconds(spec.window_mode);
            const MarketData::TimeframeData* tfd    = market_data.try_timeframe(tf_sec);
            if (tfd == nullptr) {
                // Unknown timeframe: fill output with NaN and continue.
                std::fill(out, out + num_bars_30s, std::numeric_limits<float>::quiet_NaN());
                continue;
            }

            compute_indicator_cpu_anchored(
                spec, tfd->bars.data(), static_cast<int>(tfd->bars.size()),
                tfd->completed_map.data(), bar_start_30s, num_bars_30s, out);
        }
    }

    // Dispatch all ROLLING specs through the GPU-capable path.
    // compute_with_bars() handles OpenCL availability, type partitioning,
    // and CPU fallback internally -- no duplication required here.
    if (!rolling_specs.empty()) {
        compute_with_bars(ctx, bars_30s, bar_start_30s, bar_open, bar_high, bar_low, bar_close,
                          num_bars_30s, rolling_specs, rolling_outputs);
    }
}

// ---------------------------------------------------------------------------
// CPU fallback: EMA
// Matches ema_sma_cross.cpp logic exactly:
//   ema[0] = close[0]
//   ema[i] = alpha * close[i] + (1-alpha) * ema[i-1]
//   output NaN for i < warmup
// ---------------------------------------------------------------------------

void CLIndicators::compute_ema_cpu(const float* close, float* output, int num_bars, int period,
                                   int warmup) const {
    validate_period(period, "compute_ema_cpu.period");
    const float alpha           = 2.0f / (static_cast<float>(period) + 1.0f);
    const float one_minus_alpha = 1.0f - alpha;

    float ema = 0.0f;

    for (int i = 0; i < num_bars; ++i) {
        if (i == 0) {
            ema = close[i];
        } else {
            ema = alpha * close[i] + one_minus_alpha * ema;
        }
        output[i] = (i < warmup) ? std::numeric_limits<float>::quiet_NaN() : ema;
    }
}

// ---------------------------------------------------------------------------
// CPU fallback: SMA (running sum, O(1) per bar)
// Matches ema_sma_cross.cpp logic exactly.
// ---------------------------------------------------------------------------

void CLIndicators::compute_sma_cpu(const float* close, float* output, int num_bars, int period,
                                   int warmup) const {
    validate_period(period, "compute_sma_cpu.period");
    double       running_sum = 0.0;
    const double period_d    = static_cast<double>(period);

    for (int i = 0; i < num_bars; ++i) {
        running_sum += static_cast<double>(close[i]);

        if (i >= period) {
            running_sum -= static_cast<double>(close[i - period]);
        }

        if (i >= period - 1 && i >= warmup) {
            output[i] = static_cast<float>(running_sum / period_d);
        } else {
            output[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }
}

// ---------------------------------------------------------------------------
// CPU fallback: RSI (Wilder smoothing)
// Matches rsi.cpp logic exactly.
// ---------------------------------------------------------------------------

void CLIndicators::compute_rsi_cpu(const float* close, float* output, int num_bars, int period,
                                   int warmup) const {
    validate_period(period, "compute_rsi_cpu.period");
    const float period_f       = static_cast<float>(period);
    const float period_minus_1 = period_f - 1.0f;

    float avg_gain        = 0.0f;
    float avg_loss        = 0.0f;
    bool  initialized     = false;
    int   changes_counted = 0;

    for (int i = 0; i < num_bars; ++i) {
        if (i == 0) {
            output[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        const float diff = close[i] - close[i - 1];
        const float gain = (diff > 0.0f) ? diff : 0.0f;
        const float loss = (diff < 0.0f) ? -diff : 0.0f;

        if (!initialized) {
            avg_gain += gain;
            avg_loss += loss;
            changes_counted++;

            if (changes_counted == period) {
                avg_gain /= period_f;
                avg_loss /= period_f;
                initialized = true;

                if (i < warmup) {
                    output[i] = std::numeric_limits<float>::quiet_NaN();
                } else {
                    if (avg_gain == 0.0f && avg_loss == 0.0f) {
                        output[i] = 50.0f;
                    } else if (avg_loss == 0.0f) {
                        output[i] = 100.0f;
                    } else {
                        const float rs = avg_gain / avg_loss;
                        output[i]      = 100.0f - 100.0f / (1.0f + rs);
                    }
                }
            } else {
                output[i] = std::numeric_limits<float>::quiet_NaN();
            }
        } else {
            avg_gain = (avg_gain * period_minus_1 + gain) / period_f;
            avg_loss = (avg_loss * period_minus_1 + loss) / period_f;

            if (i < warmup) {
                output[i] = std::numeric_limits<float>::quiet_NaN();
            } else {
                if (avg_gain == 0.0f && avg_loss == 0.0f) {
                    output[i] = 50.0f;
                } else if (avg_loss == 0.0f) {
                    output[i] = 100.0f;
                } else {
                    const float rs = avg_gain / avg_loss;
                    output[i]      = 100.0f - 100.0f / (1.0f + rs);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU fallback: ATR (Wilder smoothing)
// Matches atr.cpp logic exactly, operating on SoA arrays instead of Bar*.
// ---------------------------------------------------------------------------

void CLIndicators::compute_atr_cpu(const float* high, const float* low, const float* close,
                                   float* output, int num_bars, int period, int warmup) const {
    validate_period(period, "compute_atr_cpu.period");
    const float period_f       = static_cast<float>(period);
    const float period_minus_1 = period_f - 1.0f;

    float atr_val     = 0.0f;
    float tr_sum      = 0.0f;
    bool  initialized = false;

    for (int i = 0; i < num_bars; ++i) {
        // Compute True Range
        float tr;
        if (i == 0) {
            tr = high[i] - low[i];
        } else {
            const float prev_close = close[i - 1];
            const float hl         = high[i] - low[i];
            const float hpc        = std::abs(high[i] - prev_close);
            const float lpc        = std::abs(low[i] - prev_close);
            tr                     = std::max({hl, hpc, lpc});
        }

        if (!initialized) {
            tr_sum += tr;

            if (i == period - 1) {
                atr_val     = tr_sum / period_f;
                initialized = true;
                output[i]   = (i < warmup) ? std::numeric_limits<float>::quiet_NaN() : atr_val;
            } else {
                output[i] = std::numeric_limits<float>::quiet_NaN();
            }
        } else {
            atr_val   = (atr_val * period_minus_1 + tr) / period_f;
            output[i] = (i < warmup) ? std::numeric_limits<float>::quiet_NaN() : atr_val;
        }
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel management
// ---------------------------------------------------------------------------

#ifdef HAS_OPENCL

#include "indicators_cl_source.h"

// ---------------------------------------------------------------------------
// Helper: check if device supports cl_khr_fp64
// ---------------------------------------------------------------------------

static bool device_supports_fp64(cl_device_id dev) {
    size_t ext_size = 0;
    cl_int err      = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    if (err != CL_SUCCESS || ext_size == 0) return false;
    std::string extensions(ext_size, '\0');
    err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
    if (err != CL_SUCCESS) return false;
    return extensions.find("cl_khr_fp64") != std::string::npos;
}

// indicators_cl_source.h is generated at
// build time from src/compute/kernels/indicators.cl by cmake/EmbedKernel.cmake.

bool CLIndicators::build_kernels(const CLContext& ctx) {
    if (m_kernels_built) return true;
    if (!ctx.is_available()) return false;

    const char* source     = reinterpret_cast<const char*>(wfo::INDICATORS_CL_SOURCE);
    size_t      source_len = wfo::INDICATORS_CL_SOURCE_LEN;

    cl_int err;
    m_program = clCreateProgramWithSource(ctx.context(), 1, &source, &source_len, &err);
    if (err != CL_SUCCESS) return false;

    cl_device_id dev = ctx.device();

    // Check fp64 support; if absent, define NO_FP64 so Bollinger uses float fallback
    std::string build_options;
    if (!device_supports_fp64(dev)) {
        build_options = "-DNO_FP64";
    }

    err = clBuildProgram(m_program, 1, &dev, build_options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(m_program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(m_program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "[CLIndicators] Kernel build failed:\n" << log << std::endl;
        clReleaseProgram(m_program);
        m_program = nullptr;
        return false;
    }

    m_ema_kernel = clCreateKernel(m_program, "compute_ema", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_sma_kernel = clCreateKernel(m_program, "compute_sma", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_rsi_kernel = clCreateKernel(m_program, "compute_rsi", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_atr_kernel = clCreateKernel(m_program, "compute_atr", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    // Additional GPU kernels
    m_macd_kernel = clCreateKernel(m_program, "compute_macd", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_bollinger_kernel = clCreateKernel(m_program, "compute_bollinger", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_keltner_kernel = clCreateKernel(m_program, "compute_keltner", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_cci_kernel = clCreateKernel(m_program, "compute_cci", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_roc_kernel = clCreateKernel(m_program, "compute_roc", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_obv_kernel = clCreateKernel(m_program, "compute_obv", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_hma_kernel = clCreateKernel(m_program, "compute_hma", &err);
    if (err != CL_SUCCESS) {
        release_kernels();
        return false;
    }

    m_kernels_built = true;
    return true;
}

void CLIndicators::release_kernels() {
    auto safe_release_kernel = [](cl_kernel& k) {
        if (k) {
            clReleaseKernel(k);
            k = nullptr;
        }
    };
    safe_release_kernel(m_ema_kernel);
    safe_release_kernel(m_sma_kernel);
    safe_release_kernel(m_rsi_kernel);
    safe_release_kernel(m_atr_kernel);

    // Additional GPU kernels
    safe_release_kernel(m_macd_kernel);
    safe_release_kernel(m_bollinger_kernel);
    safe_release_kernel(m_keltner_kernel);
    safe_release_kernel(m_cci_kernel);
    safe_release_kernel(m_roc_kernel);
    safe_release_kernel(m_obv_kernel);
    safe_release_kernel(m_hma_kernel);

    if (m_program) {
        clReleaseProgram(m_program);
        m_program = nullptr;
    }
    m_kernels_built = false;
}

#endif  // HAS_OPENCL
