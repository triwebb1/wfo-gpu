#include "cl_backtest.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "../common.h"
#include "../resource_monitor.h"
#include "backtest_cl_source.h"  // wfo::BACKTEST_CL_SOURCE, generated at build time.

// ---------------------------------------------------------------------------
// GPU-side struct definitions -- must exactly match the typedefs in
// kernels/backtest.cl (field ordering, sizes, and packing).
// All fields are int/float (4 bytes), so there are no alignment surprises.
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct StrategyParamsGPU {
    int   indicator_indices[4];
    int   num_indicators;
    float weights[4];
    int   indicator_roles[4];  // 0=SIGNAL, 1=FILTER
    int   aggregation;         // 0=WEIGHTED_SUM, 1=MAJORITY_VOTE, 2=UNANIMOUS
    float entry_threshold;
    int   exit_mode;
    int   exit_timeout_bars;
    float trailing_atr_mult;
    float max_loss_atr_mult;
    float tp_atr_mult;
    float sl_atr_mult;
    int   atr_series_index;
    int   trading_hours;  // 0=ETH, 1=RTH, 2=MORNING, 3=AFTERNOON
};

struct BacktestResultGPU {
    float total_pnl;
    int   total_trades;
    int   win_count;
    float gross_wins;
    float gross_losses;
    float max_drawdown;
    float max_live_drawdown;
    float peak_equity;
    int   total_bars_in_position;
    float sum_min_unreal;  // sum of per-trade worst unrealized (negative)
    float sum_max_unreal;  // sum of per-trade peak unrealized (positive)
};
#pragma pack(pop)

static_assert(sizeof(StrategyParamsGPU) == 92, "GPU struct layout must match kernel");
static_assert(sizeof(BacktestResultGPU) == 44, "BacktestResultGPU size mismatch");

// ---------------------------------------------------------------------------
// IndicatorSetKey -- identifies a unique set of indicator indices
//
// Built from each StrategyConfig's indicator_indices[0..num_indicators-1],
// sorted ascending.  Two strategies with the same sorted indicator indices
// share an indicator set; the dispatcher orders work-items by key so
// neighbouring work-items read the same indicator columns.
// ---------------------------------------------------------------------------

struct IndicatorSetKey {
    int indices[4] = {-1, -1, -1, -1};
    int count      = 0;

    bool operator<(const IndicatorSetKey& o) const {
        if (count != o.count) return count < o.count;
        for (int i = 0; i < count; ++i) {
            if (indices[i] != o.indices[i]) return indices[i] < o.indices[i];
        }
        return false;
    }

    bool operator==(const IndicatorSetKey& o) const {
        if (count != o.count) return false;
        for (int i = 0; i < count; ++i) {
            if (indices[i] != o.indices[i]) return false;
        }
        return true;
    }
};

// Build an IndicatorSetKey from a StrategyConfig
[[maybe_unused]] static IndicatorSetKey make_indicator_set_key(const StrategyConfig& sc) {
    IndicatorSetKey key;
    key.count = sc.num_indicators;
    for (int i = 0; i < sc.num_indicators; ++i) {
        key.indices[i] = sc.indicator_indices[i];
    }
    // Sort the active indices ascending for canonical representation
    std::sort(key.indices, key.indices + key.count);
    return key;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

CLBacktest::~CLBacktest() {
#ifdef HAS_OPENCL
    release_pooled_buffers();
    release_kernel();
#endif
}

// ---------------------------------------------------------------------------
// evaluate() -- main dispatcher
//
// Attempts OpenCL GPU path first.  If the kernel cannot be built or any
// OpenCL call fails, falls back to evaluate_cpu() which produces identical
// results using the C++ BacktestEngine.
// ---------------------------------------------------------------------------

void CLBacktest::evaluate(const CLContext& ctx, const Bar* bars, int bar_offset,
                          const float* const* indicator_series, int num_indicator_series,
                          int num_bars, int eval_start, int eval_end,
                          const std::vector<StrategyConfig>& strategies, const CostModel& cost,
                          int entry_lag, std::vector<BacktestResult>& results,
                          const EarlyTermination& early_term, const SessionInfo* sessions) {
#ifdef HAS_OPENCL
    if (ctx.is_available() && !strategies.empty() && !early_term.enabled) {
        bool ok =
            evaluate_gpu(ctx, bars, bar_offset, indicator_series, num_indicator_series, num_bars,
                         eval_start, eval_end, strategies, cost, entry_lag, results, sessions);
        if (ok) return;
        // GPU path failed -- fall through to CPU
        // Surface the GPU fallback so OOM/driver faults
        // do not masquerade as CPU-only slow runs.
        std::cerr << "[WARN] GPU backtest path failed (likely OOM or driver error); "
                     "falling back to CPU for this batch of "
                  << strategies.size() << " strategies\n";
    }
#else
    (void)ctx;
#endif

    evaluate_cpu(bars, bar_offset, indicator_series, num_indicator_series, num_bars, eval_start,
                 eval_end, strategies, cost, entry_lag, results, early_term, sessions);
}

// ---------------------------------------------------------------------------
// CPU execution path: run BacktestEngine for each strategy.
// Uses the same logic as kernels/backtest.cl -- results must be identical.
// ---------------------------------------------------------------------------

void CLBacktest::evaluate_cpu(const Bar* bars, int bar_offset, const float* const* indicator_series,
                              int num_indicator_series, int num_bars, int eval_start, int eval_end,
                              const std::vector<StrategyConfig>& strategies, const CostModel& cost,
                              int entry_lag, std::vector<BacktestResult>& results,
                              const EarlyTermination& early_term, const SessionInfo* sessions) {
    (void)num_bars;  // eval range is [eval_start, eval_end)

    const int num_strategies = static_cast<int>(strategies.size());
    results.resize(static_cast<size_t>(num_strategies));

    if (num_strategies == 0) return;

    // Precompute offset pointers once (shared read-only across threads).
    std::vector<const float*> offset_ptrs(static_cast<size_t>(num_indicator_series));
    for (int k = 0; k < num_indicator_series; ++k) {
        offset_ptrs[k] = indicator_series[k] + eval_start;
    }

    const int abs_bar_start = bar_offset + eval_start;
    const int abs_bar_end   = bar_offset + eval_end;

    // Lambda to evaluate a range of strategies [range_start, range_end).
    // Each invocation creates its own BacktestEngine (thread-safe by design).
    auto evaluate_range = [&](int range_start, int range_end) {
        BacktestEngine engine;
        for (int s = range_start; s < range_end; ++s) {
            if (sessions) {
                results[s] = engine.run(bars, sessions, abs_bar_start, abs_bar_end,
                                        offset_ptrs.data(), num_indicator_series,
                                        nullptr,  // no per-series time gates in the screen stage
                                        strategies[s], cost, entry_lag, early_term);
            } else {
                results[s] =
                    engine.run(bars, abs_bar_start, abs_bar_end, offset_ptrs.data(),
                               num_indicator_series, strategies[s], cost, entry_lag, early_term);
            }
            results[s].strategy_id = s;
        }
    };

    // Determine thread count: honour the configured CPU limit.
    // For small batches, threading overhead exceeds benefit.
    ResourceMonitor rm;
    const int       max_threads               = rm.max_threads();
    const int       min_strategies_per_thread = 4;

    if (max_threads <= 1 || num_strategies < max_threads * min_strategies_per_thread) {
        // Single-threaded: not enough work to justify threading overhead
        evaluate_range(0, num_strategies);
    } else {
        // Multi-threaded: partition strategies into chunks, one per thread.
        // Each thread writes to its own index range in results[] (no races).
        const int                  chunk_size = (num_strategies + max_threads - 1) / max_threads;
        std::vector<JoiningThread> threads;
        threads.reserve(max_threads);

        for (int t = 0; t < max_threads; ++t) {
            int start = t * chunk_size;
            int end   = std::min(start + chunk_size, num_strategies);
            if (start >= end) break;
            threads.emplace_back([&, start, end]() { evaluate_range(start, end); });
        }
        // JoiningThread joins on destruction
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel management
// ---------------------------------------------------------------------------

#ifdef HAS_OPENCL

// ---------------------------------------------------------------------------
// build_kernel -- compile the embedded backtest.cl source.
//
// The kernel source is embedded into the binary at build time by
// cmake/EmbedKernel.cmake (see BACKTEST_CL_SOURCE). This avoids runtime
// CWD dependence and keeps CLBacktest symmetric with CLIndicators.
// ---------------------------------------------------------------------------

bool CLBacktest::build_kernel(const CLContext& ctx) {
    if (m_kernel_built) return true;
    if (!ctx.is_available()) return false;

    const char* src_ptr = reinterpret_cast<const char*>(wfo::BACKTEST_CL_SOURCE);
    size_t      src_len = wfo::BACKTEST_CL_SOURCE_LEN;

    cl_int err;
    m_program = clCreateProgramWithSource(ctx.context(), 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[CLBacktest] clCreateProgramWithSource failed: " << err << "\n";
        return false;
    }

    // Correctly rounded division where the device offers it, so composite
    // signals that land exactly on an entry threshold compare as they do on
    // the CPU. FP_CONTRACT is switched off in the kernel source.
    cl_device_id        dev       = ctx.device();
    cl_device_fp_config fp_config = 0;
    clGetDeviceInfo(dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr);
    const char* options = (fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
                              ? "-cl-fp32-correctly-rounded-divide-sqrt"
                              : "";
    err                 = clBuildProgram(m_program, 1, &dev, options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(m_program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(m_program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "[CLBacktest] Kernel build failed:\n" << log << std::endl;
        clReleaseProgram(m_program);
        m_program = nullptr;
        return false;
    }

    m_kernel = clCreateKernel(m_program, "evaluate_backtest", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[CLBacktest] clCreateKernel failed: " << err << "\n";
        release_kernel();
        return false;
    }

    m_kernel_built = true;
    return true;
}

// ---------------------------------------------------------------------------
// release_kernel -- clean up OpenCL objects
// ---------------------------------------------------------------------------

void CLBacktest::release_kernel() {
    if (m_kernel) {
        clReleaseKernel(m_kernel);
        m_kernel = nullptr;
    }
    if (m_program) {
        clReleaseProgram(m_program);
        m_program = nullptr;
    }
    m_kernel_built = false;
}

// ---------------------------------------------------------------------------
// Pooled buffer management
// ---------------------------------------------------------------------------

void CLBacktest::release_pooled(PooledBuffer& pb) {
    if (pb.buf) {
        clReleaseMemObject(pb.buf);
        pb.buf = nullptr;
    }
    pb.capacity = 0;
}

void CLBacktest::release_pooled_buffers() {
    release_pooled(m_pool_open);
    release_pooled(m_pool_high);
    release_pooled(m_pool_low);
    release_pooled(m_pool_close);
    release_pooled(m_pool_flags);
    release_pooled(m_pool_ind);
    release_pooled(m_pool_params);
    release_pooled(m_pool_results);
}

cl_mem CLBacktest::ensure_buffer(PooledBuffer& pb, size_t needed, cl_mem_flags flags,
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

// ---------------------------------------------------------------------------
// convert_strategy -- StrategyConfig -> StrategyParamsGPU
// ---------------------------------------------------------------------------

static StrategyParamsGPU convert_strategy(const StrategyConfig& sc) {
    StrategyParamsGPU p{};
    for (int i = 0; i < 4; ++i) {
        p.indicator_indices[i] = sc.indicator_indices[i];
        p.weights[i]           = sc.weights[i];
        p.indicator_roles[i]   = static_cast<int>(sc.indicator_roles[i]);
    }
    p.num_indicators    = sc.num_indicators;
    p.aggregation       = static_cast<int>(sc.aggregation);
    p.entry_threshold   = sc.entry_threshold;
    p.exit_mode         = static_cast<int>(sc.exit_mode);
    p.exit_timeout_bars = sc.exit_timeout_bars;
    p.trailing_atr_mult = sc.trailing_atr_mult;
    p.max_loss_atr_mult = sc.max_loss_atr_mult;
    p.tp_atr_mult       = sc.tp_atr_mult;
    p.sl_atr_mult       = sc.sl_atr_mult;
    p.atr_series_index  = sc.atr_series_index;
    p.trading_hours     = static_cast<int>(sc.trading_hours);
    return p;
}

// ---------------------------------------------------------------------------
// convert_gpu_result -- BacktestResultGPU -> BacktestResult
//
// The GPU kernel only produces raw accumulators.  Derived metrics (win_rate,
// Sharpe, Calmar, etc.) are computed here on the host, matching the same
// formulas used by compute_metrics().
// ---------------------------------------------------------------------------

static BacktestResult convert_gpu_result(const BacktestResultGPU& gpu, const CostModel& cost,
                                         int64_t start_ts, int64_t end_ts, int strategy_id) {
    BacktestResult r{};
    r.strategy_id       = strategy_id;
    r.start_ts          = start_ts;
    r.end_ts            = end_ts;
    r.total_trades      = gpu.total_trades;
    r.total_pnl         = gpu.total_pnl;
    r.gross_wins        = gpu.gross_wins;
    r.gross_losses      = gpu.gross_losses;
    r.max_drawdown      = gpu.max_drawdown;
    r.max_live_drawdown = gpu.max_live_drawdown;

    if (gpu.total_trades > 0) {
        const float trades = static_cast<float>(gpu.total_trades);
        r.win_rate         = static_cast<float>(gpu.win_count) / trades;

        const float total_wins   = gpu.gross_wins;
        const float total_losses = -gpu.gross_losses;  // gross_losses is negative
        const int   win_count    = gpu.win_count;
        const int   loss_count   = gpu.total_trades - gpu.win_count;

        r.avg_win  = (win_count > 0) ? (total_wins / win_count) : 0.0f;
        r.avg_loss = (loss_count > 0) ? (-total_losses / loss_count) : 0.0f;

        r.profit_factor = (total_losses > 0.0f) ? (total_wins / total_losses)
                                                : ((total_wins > 0.0f) ? 1e6f : 0.0f);
        r.payoff_ratio  = compute_payoff_ratio(r.avg_win, r.avg_loss);

        if (gpu.total_bars_in_position > 0)
            r.avg_hold_bars = static_cast<float>(gpu.total_bars_in_position) / trades;

        // Viability: avg per-trade PnL >= 2x spread cost
        r.viable = (gpu.total_pnl / trades >= 2.0f * cost.round_trip_cost());

        // The kernel accumulates per-trade worst (negative) and best
        // unrealized PnL; averages are stored as magnitudes, as on the CPU.
        r.avg_trade_dd   = -gpu.sum_min_unreal / trades;
        r.avg_trade_peak = gpu.sum_max_unreal / trades;

        // Trade recovery ratio, with the CPU's sentinel when no trade ever
        // went negative.
        r.trade_recovery_ratio =
            (r.avg_trade_dd > 0.0f) ? (r.avg_win * r.win_rate) / r.avg_trade_dd : 100.0f;

        // Calendar-time metrics from the first and last evaluated bar, with
        // the same formulas as compute_metrics().
        const double duration_days  = static_cast<double>(end_ts - start_ts) / 86400e9;
        const double duration_weeks = duration_days / 7.0;
        const double duration_years = duration_days / 365.25;

        if (duration_weeks > 0.0)
            r.trades_per_week = static_cast<float>(gpu.total_trades / duration_weeks);
        if (duration_years > 0.0 && gpu.max_drawdown > 0.0f) {
            const float annualized_pnl =
                static_cast<float>(static_cast<double>(gpu.total_pnl) / duration_years);
            r.calmar_ratio = annualized_pnl / gpu.max_drawdown;
        }

        const double capital = static_cast<double>(cost.capital_required);
        if (duration_years > 0.0 && capital > 0.0) {
            const double final_equity = capital + static_cast<double>(gpu.total_pnl);
            if (final_equity <= 0.0) {
                // Total wipeout: sentinel values.
                r.annual_roc     = -100.0f;
                r.time_to_double = -1.0f;
            } else {
                const double cagr =
                    (std::pow(final_equity / capital, 1.0 / duration_years) - 1.0) * 100.0;
                r.annual_roc = static_cast<float>(cagr);
                r.time_to_double =
                    (cagr > 0.0) ? static_cast<float>(
                                       (std::log(2.0) / std::log(1.0 + cagr / 100.0)) * 365.25)
                                 : -1.0f;
            }
        }

        // The kernel does not produce the consistency metrics (green weeks
        // and months, red-week streaks, negative years) or Sharpe, so the
        // result is not complete until the CPU engine fills them in
        // (Orchestrator's refinement pass). Robustness skips incomplete
        // neighbors.
        r.metrics_complete = false;
    }

    return r;
}

// ---------------------------------------------------------------------------
// evaluate_gpu -- full OpenCL dispatch
//
// Returns true if GPU evaluation succeeded.  On any failure, returns false
// and the caller falls back to evaluate_cpu().
//
// Performance:
//   - Pooled GPU buffers and host staging vectors are reused across calls
//        and only reallocated when a larger size is needed.
//   - One kernel launch per batch, with work-items ordered by indicator set
//        and exit mode so neighbours share cache lines and branch less.
//        Results are written back in input order.
//
// NOTE: early_term is not supported on the GPU path.  The caller should
// check early_term.enabled and skip the GPU path if true.
// ---------------------------------------------------------------------------

bool CLBacktest::evaluate_gpu(const CLContext& ctx, const Bar* bars, int bar_offset,
                              const float* const* indicator_series, int num_indicator_series,
                              int num_bars, int eval_start, int eval_end,
                              const std::vector<StrategyConfig>& strategies, const CostModel& cost,
                              int entry_lag, std::vector<BacktestResult>& results,
                              const SessionInfo* sessions) {
    // Step 1: Build / cache the kernel
    if (!build_kernel(ctx)) return false;

    // The window starts bar_offset bars into `bars` (and `sessions`), as in
    // evaluate_cpu(); from here on index 0 is the window's first bar.
    bars += bar_offset;
    if (sessions != nullptr) sessions += bar_offset;

    const int num_strategies = static_cast<int>(strategies.size());
    if (num_strategies == 0) {
        results.clear();
        return true;
    }

    cl_int           err;
    cl_command_queue queue  = ctx.queue();
    cl_context       cl_ctx = ctx.context();

    // ------------------------------------------------------------------
    // Step 2: Prepare SoA bar data (OHLC) from the AoS Bar array. The host
    // staging vectors are reused across calls; their contents are always
    // rebuilt, which costs far less than the upload that follows.
    // ------------------------------------------------------------------
    m_h_open.resize(static_cast<size_t>(num_bars));
    m_h_high.resize(static_cast<size_t>(num_bars));
    m_h_low.resize(static_cast<size_t>(num_bars));
    m_h_close.resize(static_cast<size_t>(num_bars));
    for (int i = 0; i < num_bars; ++i) {
        m_h_open[i]  = bars[i].open;
        m_h_high[i]  = bars[i].high;
        m_h_low[i]   = bars[i].low;
        m_h_close[i] = bars[i].close;
    }

    // ------------------------------------------------------------------
    // Step 3: Build the per-bar session flag array for the kernel
    //
    // One int per bar holding SessionFlag bits (coalesced reads on GPU).
    // From SessionInfo when provided (hours, close and halt bits); from
    // set_bar_flags() otherwise (hours bits only); else a single zero entry
    // and the kernel treats every bar as ETH.
    // ------------------------------------------------------------------
    int              has_bar_flags    = 0;
    int              has_session_data = 0;
    std::vector<int> h_bar_flags;

    if (sessions != nullptr) {
        h_bar_flags.resize(static_cast<size_t>(num_bars));
        for (int i = 0; i < num_bars; ++i) {
            h_bar_flags[i] = static_cast<int>(sessions[i].flags);
        }
        has_bar_flags    = 1;
        has_session_data = 1;
    } else if (!m_bar_flags.empty() && static_cast<int>(m_bar_flags.size()) >= num_bars) {
        h_bar_flags   = m_bar_flags;
        has_bar_flags = 1;
    } else {
        h_bar_flags.resize(1, 0);
    }

    // ------------------------------------------------------------------
    // Step 4: Flatten indicator data into one contiguous buffer
    //
    // Layout: indicator_data[ind_idx * stride + bar_idx], stride = num_bars.
    // Rebuilt on every call: callers reuse and recycle signal buffers, so
    // pointer identity says nothing about their contents.
    // ------------------------------------------------------------------
    const int indicator_stride = num_bars;
    m_h_indicator_data.assign(
        static_cast<size_t>(num_indicator_series) * static_cast<size_t>(indicator_stride), 0.0f);
    for (int k = 0; k < num_indicator_series; ++k) {
        if (indicator_series[k]) {
            std::memcpy(&m_h_indicator_data[static_cast<size_t>(k) * indicator_stride],
                        indicator_series[k], static_cast<size_t>(indicator_stride) * sizeof(float));
        }
    }

    // ------------------------------------------------------------------
    // Step 5: Convert StrategyConfig -> StrategyParamsGPU
    // ------------------------------------------------------------------
    std::vector<StrategyParamsGPU> h_params(num_strategies);
    for (int s = 0; s < num_strategies; ++s) {
        h_params[s] = convert_strategy(strategies[s]);
    }

    // ------------------------------------------------------------------
    // Step 6: Order work-items by (indicator set, exit mode)
    //
    // One kernel launch covers the whole batch. Ordering puts strategies
    // that read the same indicator columns and take the same exit branch
    // next to each other, so a wavefront shares cache lines and diverges
    // less. Launching per indicator set instead would cost one launch and
    // one blocking read per set, which dominates when sets are scattered
    // (walk-forward reduced sets, sampled benchmarks).
    // ------------------------------------------------------------------
    std::vector<IndicatorSetKey> keys(static_cast<size_t>(num_strategies));
    for (int s = 0; s < num_strategies; ++s) keys[s] = make_indicator_set_key(strategies[s]);
    std::vector<int> launch_order(static_cast<size_t>(num_strategies));
    std::iota(launch_order.begin(), launch_order.end(), 0);
    std::stable_sort(launch_order.begin(), launch_order.end(), [&](int a, int b) {
        if (keys[a] == keys[b]) return h_params[a].exit_mode < h_params[b].exit_mode;
        return keys[a] < keys[b];
    });
    const size_t max_group_size = static_cast<size_t>(num_strategies);
    // ------------------------------------------------------------------
    // Step 7: Upload shared data (bars, indicators, sessions)
    //
    // These are shared by every work-item and uploaded once.
    // ------------------------------------------------------------------
    const size_t bar_bytes        = static_cast<size_t>(num_bars) * sizeof(float);
    const size_t flags_bytes      = h_bar_flags.size() * sizeof(int);
    const size_t ind_bytes        = m_h_indicator_data.size() * sizeof(float);
    const size_t actual_ind_bytes = (ind_bytes > 0) ? ind_bytes : sizeof(float);

    // Params and results buffers hold the whole batch.
    const size_t max_params_bytes  = max_group_size * sizeof(StrategyParamsGPU);
    const size_t max_results_bytes = max_group_size * sizeof(BacktestResultGPU);

    // Ensure all pooled buffers are large enough
    ensure_buffer(m_pool_open, bar_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
    if (err != CL_SUCCESS) return false;
    ensure_buffer(m_pool_high, bar_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
    if (err != CL_SUCCESS) return false;
    ensure_buffer(m_pool_low, bar_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
    if (err != CL_SUCCESS) return false;
    ensure_buffer(m_pool_close, bar_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
    if (err != CL_SUCCESS) return false;
    ensure_buffer(m_pool_flags, flags_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
    if (err != CL_SUCCESS) return false;
    ensure_buffer(m_pool_ind, actual_ind_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
    if (err != CL_SUCCESS) return false;
    ensure_buffer(m_pool_params, max_params_bytes, CL_MEM_READ_ONLY, cl_ctx, err);
    if (err != CL_SUCCESS) return false;
    ensure_buffer(m_pool_results, max_results_bytes, CL_MEM_READ_WRITE, cl_ctx, err);
    if (err != CL_SUCCESS) return false;

    // Upload shared bar data
    err = clEnqueueWriteBuffer(queue, m_pool_open.buf, CL_FALSE, 0, bar_bytes, m_h_open.data(), 0,
                               nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    err = clEnqueueWriteBuffer(queue, m_pool_high.buf, CL_FALSE, 0, bar_bytes, m_h_high.data(), 0,
                               nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    err = clEnqueueWriteBuffer(queue, m_pool_low.buf, CL_FALSE, 0, bar_bytes, m_h_low.data(), 0,
                               nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    err = clEnqueueWriteBuffer(queue, m_pool_close.buf, CL_FALSE, 0, bar_bytes, m_h_close.data(), 0,
                               nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    // Upload session data. Blocking: h_bar_flags is a local, and an early
    // return must not leave a transfer reading it.
    err = clEnqueueWriteBuffer(queue, m_pool_flags.buf, CL_TRUE, 0, flags_bytes, h_bar_flags.data(),
                               0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;

    // Upload indicator data
    if (ind_bytes > 0) {
        err = clEnqueueWriteBuffer(queue, m_pool_ind.buf, CL_FALSE, 0, ind_bytes,
                                   m_h_indicator_data.data(), 0, nullptr, nullptr);
    } else {
        float dummy = 0.0f;
        err = clEnqueueWriteBuffer(queue, m_pool_ind.buf, CL_TRUE, 0, sizeof(float), &dummy, 0,
                                   nullptr, nullptr);
    }
    if (err != CL_SUCCESS) return false;

    // Common kernel arguments and result conversion state
    float         point_value = cost.point_value;
    float         rt_cost     = cost.round_trip_cost();
    const int64_t start_ts    = bars[eval_start].timestamp;
    const int64_t end_ts      = bars[eval_end - 1].timestamp;

    results.resize(static_cast<size_t>(num_strategies));

    // ------------------------------------------------------------------
    // Step 8: Set kernel arguments (once -- shared across all dispatches)
    //
    // All cl_mem handles and scalar args are identical for every group
    // dispatch.  Only the buffer *contents* and global_size change.
    // ------------------------------------------------------------------
    {
        int  arg     = 0;
        bool args_ok = true;
        auto set_arg = [&](size_t size, const void* ptr) {
            cl_int se = clSetKernelArg(m_kernel, arg++, size, ptr);
            if (se != CL_SUCCESS) args_ok = false;
        };

        set_arg(sizeof(cl_mem), &m_pool_open.buf);     // 0
        set_arg(sizeof(cl_mem), &m_pool_high.buf);     // 1
        set_arg(sizeof(cl_mem), &m_pool_low.buf);      // 2
        set_arg(sizeof(cl_mem), &m_pool_close.buf);    // 3
        set_arg(sizeof(cl_mem), &m_pool_flags.buf);    // 4
        set_arg(sizeof(cl_mem), &m_pool_ind.buf);      // 5
        set_arg(sizeof(int), &indicator_stride);       // 6
        set_arg(sizeof(int), &num_indicator_series);   // 7
        set_arg(sizeof(cl_mem), &m_pool_params.buf);   // 8
        set_arg(sizeof(cl_mem), &m_pool_results.buf);  // 9
        set_arg(sizeof(int), &num_bars);               // 10
        set_arg(sizeof(int), &eval_start);             // 11
        set_arg(sizeof(int), &eval_end);               // 12
        set_arg(sizeof(float), &point_value);          // 13
        set_arg(sizeof(float), &rt_cost);              // 14
        set_arg(sizeof(int), &entry_lag);              // 15
        set_arg(sizeof(int), &has_bar_flags);          // 16
        set_arg(sizeof(int), &has_session_data);       // 17

        if (!args_ok) return false;
    }

    // ------------------------------------------------------------------
    // Helper lambda: dispatch a batch of strategies via the kernel.
    //
    // Takes original indices in launch order, uploads their params,
    // dispatches, reads results, and writes converted BacktestResults to
    // the correct output slots.
    // ------------------------------------------------------------------
    auto dispatch_batch = [&](const std::vector<int>& original_indices) -> bool {
        const int batch_size = static_cast<int>(original_indices.size());
        if (batch_size == 0) return true;

        std::vector<StrategyParamsGPU> sorted_params(batch_size);
        for (int i = 0; i < batch_size; ++i) sorted_params[i] = h_params[original_indices[i]];

        // Zero-init results buffer
        std::vector<BacktestResultGPU> h_results_batch(batch_size);
        std::memset(h_results_batch.data(), 0,
                    static_cast<size_t>(batch_size) * sizeof(BacktestResultGPU));

        const size_t params_bytes_batch =
            static_cast<size_t>(batch_size) * sizeof(StrategyParamsGPU);
        const size_t results_bytes_batch =
            static_cast<size_t>(batch_size) * sizeof(BacktestResultGPU);

        // Upload params and zero-init results
        // Blocking writes: both sources are locals of this lambda.
        cl_int e = clEnqueueWriteBuffer(queue, m_pool_params.buf, CL_TRUE, 0, params_bytes_batch,
                                        sorted_params.data(), 0, nullptr, nullptr);
        if (e != CL_SUCCESS) return false;

        e = clEnqueueWriteBuffer(queue, m_pool_results.buf, CL_TRUE, 0, results_bytes_batch,
                                 h_results_batch.data(), 0, nullptr, nullptr);
        if (e != CL_SUCCESS) return false;

        // Dispatch kernel
        size_t global_size = static_cast<size_t>(batch_size);
        e = clEnqueueNDRangeKernel(queue, m_kernel, 1, nullptr, &global_size, nullptr, 0, nullptr,
                                   nullptr);
        if (e != CL_SUCCESS) {
            std::cerr << "[CLBacktest] clEnqueueNDRangeKernel failed: " << e << "\n";
            return false;
        }

        // Read back results (blocking)
        e = clEnqueueReadBuffer(queue, m_pool_results.buf, CL_TRUE, 0, results_bytes_batch,
                                h_results_batch.data(), 0, nullptr, nullptr);
        if (e != CL_SUCCESS) {
            std::cerr << "[CLBacktest] clEnqueueReadBuffer failed: " << e << "\n";
            return false;
        }

        // Place each result at its original index
        for (int i = 0; i < batch_size; ++i) {
            const int orig_idx = original_indices[i];
            results[orig_idx] =
                convert_gpu_result(h_results_batch[i], cost, start_ts, end_ts, orig_idx);
        }

        return true;
    };

    // ------------------------------------------------------------------
    // Step 9: Dispatch the whole batch in launch order
    // ------------------------------------------------------------------
    if (!dispatch_batch(launch_order)) return false;

    return true;
}

#endif  // HAS_OPENCL
