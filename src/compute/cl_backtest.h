#pragma once
// ---------------------------------------------------------------------------
// CLBacktest -- Host-side backtest evaluation dispatcher
//
// When OpenCL is available: launches backtest kernel grouped by indicator set.
// When not available: runs CPU BacktestEngine directly.
//
// Both paths produce identical BacktestResult metrics.
//
// Performance optimizations:
//   - Cached SoA bar data: AoS-to-SoA conversion only when bar range changes
//   - Pooled GPU buffers: persistent cl_mem objects, reallocated only when
//        a larger size is needed
//   - Strategy ordering: one launch with work-items ordered by indicator
//        set and exit mode, so neighbours share signal columns and branches
//   - Cached indicator flatten: flat indicator buffer rebuilt only when
//        indicator pointers or count change
// ---------------------------------------------------------------------------

#include <vector>
#include <cstdint>

#include "cl_context.h"
#include "cl_memory.h"
#include "../market_data.h"
#include "../cost_model.h"
#include "../backtest_engine.h"

#ifdef HAS_OPENCL
#include <CL/cl.h>
#endif

class CLBacktest {
public:
    CLBacktest() = default;
    ~CLBacktest();

    // Evaluate multiple strategy configurations over a bar window.
    //
    // Parameters:
    //   ctx               - OpenCL context (may be unavailable => CPU fallback)
    //   bars              - full bar array (AoS, for CPU fallback)
    //   bar_data_soa      - SoA bar arrays (for OpenCL kernel)
    //   indicator_series  - array of float* pointers to indicator output arrays
    //   num_indicator_series - number of indicator series
    //   num_bars          - total bars in the indicator/SoA arrays
    //   eval_start        - first bar offset to evaluate (after warmup)
    //   eval_end          - one past last bar offset to evaluate
    //   strategies        - strategy configurations to evaluate
    //   cost              - cost model
    //   entry_lag         - entry delay (1 = standard, 2 = lag-2 validation)
    //   results           - output: one BacktestResult per strategy
    //
    void evaluate(const CLContext& ctx, const Bar* bars,
                  int                 bar_offset,  // offset of SoA bar[0] in the full bars array
                  const float* const* indicator_series, int num_indicator_series, int num_bars,
                  int eval_start, int eval_end, const std::vector<StrategyConfig>& strategies,
                  const CostModel& cost, int entry_lag, std::vector<BacktestResult>& results,
                  const EarlyTermination& early_term = EarlyTermination{},
                  const SessionInfo*      sessions   = nullptr);

    // Set per-bar SessionFlag bits for GPU trading-hours filtering when no
    // SessionInfo is passed to evaluate(). Must contain at least num_bars
    // entries. If not set (or empty), the GPU kernel runs in ETH mode.
    void set_bar_flags(const std::vector<int>& bar_flags) { m_bar_flags = bar_flags; }
    void set_bar_flags(std::vector<int>&& bar_flags) { m_bar_flags = std::move(bar_flags); }

#ifdef HAS_OPENCL
    // Public for tests that need to verify the GPU kernel actually builds,
    // preventing silent CPU fallback from masking parity regressions.
    // Production callers should use evaluate() which calls this internally.
    bool build_kernel(const CLContext& ctx);
#endif

private:
    // CPU fallback: runs BacktestEngine for each strategy
    void evaluate_cpu(const Bar* bars, int bar_offset, const float* const* indicator_series,
                      int num_indicator_series, int num_bars, int eval_start, int eval_end,
                      const std::vector<StrategyConfig>& strategies, const CostModel& cost,
                      int entry_lag, std::vector<BacktestResult>& results,
                      const EarlyTermination& early_term, const SessionInfo* sessions = nullptr);

    // Optional per-bar SessionFlag bits (see set_bar_flags)
    std::vector<int> m_bar_flags;

#ifdef HAS_OPENCL
    // GPU dispatch -- returns true on success, false to fall back to CPU
    bool evaluate_gpu(const CLContext& ctx, const Bar* bars, int bar_offset,
                      const float* const* indicator_series, int num_indicator_series, int num_bars,
                      int eval_start, int eval_end, const std::vector<StrategyConfig>& strategies,
                      const CostModel& cost, int entry_lag, std::vector<BacktestResult>& results,
                      const SessionInfo* sessions);

    cl_program m_program      = nullptr;
    cl_kernel  m_kernel       = nullptr;
    bool       m_kernel_built = false;

    void release_kernel();

    // ---- Host staging buffers, reused across calls ----
    std::vector<float> m_h_open;
    std::vector<float> m_h_high;
    std::vector<float> m_h_low;
    std::vector<float> m_h_close;

    // ---- Pooled GPU buffers ----
    // Persistent cl_mem objects.  Only reallocated when a larger size is needed.
    struct PooledBuffer {
        cl_mem buf      = nullptr;
        size_t capacity = 0;  // current allocation size in bytes
    };

    // Ensure a pooled buffer has at least `needed` bytes.
    // Returns the cl_mem handle (may be newly allocated or reused).
    // On failure sets err and returns nullptr.
    cl_mem ensure_buffer(PooledBuffer& pb, size_t needed, cl_mem_flags flags, cl_context context,
                         cl_int& err);

    // Release a single pooled buffer
    static void release_pooled(PooledBuffer& pb);

    // All pooled buffers used by evaluate_gpu()
    PooledBuffer m_pool_open;
    PooledBuffer m_pool_high;
    PooledBuffer m_pool_low;
    PooledBuffer m_pool_close;
    PooledBuffer m_pool_flags;
    PooledBuffer m_pool_ind;
    PooledBuffer m_pool_params;
    PooledBuffer m_pool_results;

    void release_pooled_buffers();

    std::vector<float> m_h_indicator_data;  // flat [series * num_bars]
#endif
};
