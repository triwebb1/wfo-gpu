#pragma once
// ---------------------------------------------------------------------------
// Orchestrator -- top-level management of the screen and walk-forward stages
//
// Screen pipeline:
//   1. Load bar data for the scan period
//   2. Compute session signals (CPU)
//   3. Upload bar data + TF bars + session signals to device
//   4. Compute all indicator series in parallel
//   5. Evaluate backtests grouped by indicator set
//   6. Apply sanity floors (total_pnl > 0, trades >= 100, etc.)
//   7. Compute robustness scores (lookup in results)
//   8. Extract Pareto frontier, classify archetypes
//   9. Build reduced parameter set (frontier + neighborhoods, robustness >= 0.50)
//  10. Write results to output directory
//
// The orchestrator delegates to:
//   - CLIndicators (compute, CPU fallback)
//   - CLBacktest (evaluate, CPU fallback)
//   - LandscapeAnalysis (sanity floors, Pareto, archetypes, reduced set)
//   - Robustness (neighborhood stability scoring)
// ---------------------------------------------------------------------------

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "compute/cl_backtest.h"
#include "compute/cl_context.h"
#include "compute/cl_indicators.h"
#include "compute/cl_memory.h"
#include "cost_model.h"
#include "grids/indicator_grid.h"
#include "grids/selection_grid.h"
#include "grids/session_signal_grid.h"
#include "grids/strategy_grid.h"
#include "grids/wfo_grid.h"
#include "grids/window_mode_grid.h"
#include "landscape.h"
#include "market_data.h"
#include "metrics.h"
#include "wfo_engine.h"

class Orchestrator {
public:
    // ---- Screen Configuration ----
    struct ScreenConfig {
        std::string  bar_file;                     // path to binary bar file
        int          scan_start_bar   = 0;         // first bar of the scan period
        int          scan_end_bar     = 0;         // one past the last bar of the scan period
        float        robustness_floor = 0.50f;     // minimum robustness for walk-forward inclusion
        int          max_active_indicators = 4;    // max indicators per strategy
        std::string  output_dir;                   // directory for result files
        CostModel    cost      = CostModel::es();  // instrument cost model
        SessionModel session   = SessionModel::us_futures();  // session layout
        int          entry_lag = 1;                           // entry delay (1 = standard)
        // Abort clearly losing strategies part-way through the scan (CPU
        // path only). Off by default so results do not depend on the device.
        bool early_termination = false;

        // If > 0, the number of distinct strategies to evaluate, in grid order.
        int64_t max_strategies = 0;

        // Results from a partial screen checkpoint. Evaluation continues
        // after the last grid entry they hold.
        std::vector<BacktestResult> resumed_results;

        // Sanity floors applied in the screen stage. Threaded through
        // analyze_landscape and the pre-reeval survivor gate so all three
        // sites share one source of truth.
        SanityFloors sanity_floors;

        // Progress callback invoked at screen batch boundaries. Arguments:
        // (strategies_done, strategies_total, results_so_far). Typical use:
        // write a mid-run checkpoint.
        //
        // CONTRACT: the callback is invoked SYNCHRONOUSLY on the
        // orchestrator thread. Implementations MUST NOT defer work to
        // other threads or assume pointers/references outlive the call.
        // Throttling is the caller's responsibility.
        std::function<void(int done, int total, const std::vector<BacktestResult>& partial)>
            progress_cb;
    };

    // ---- Screen Result ----
    struct ScreenResult {
        int num_signals      = 0;  // signals the strategy grid was built over
        int total_evaluated  = 0;
        int passed_floors    = 0;
        int on_frontier      = 0;
        int reduced_set_size = 0;

        // Strategy IDs that pass into the walk-forward stage
        std::vector<int> reduced_strategy_ids;

        // Full landscape analysis
        LandscapeAnalysis landscape;

        // All backtest results (for inspection/debugging)
        std::vector<BacktestResult> all_results;

        // Archetype counts
        int grinder_count  = 0;
        int sniper_count   = 0;
        int balanced_count = 0;

        // True if results were written to disk
        bool output_written = false;
    };

    Orchestrator() = default;

    // ---- Walk-forward Configuration ----
    struct WalkForwardConfig {
        std::string      bar_file;                              // path to binary bar file
        std::vector<int> reduced_strategy_ids;                  // from the screen stage
        std::string      output_dir;                            // directory for result files
        CostModel        cost    = CostModel::es();             // instrument cost model
        SessionModel     session = SessionModel::us_futures();  // session layout
        // Bars per calendar day, used to turn window lengths in days into bar
        // counts. 0 measures it from the bar file.
        float bars_per_day = 0.0f;

        // Earliest bar an out-of-sample window may start at; set it to the
        // end of the screen range so OOS results never reuse the data the
        // reduced set was selected on.
        int oos_start_bar = 0;

        // Reduced indicator set: only compute these indicator IDs in walk-forward.
        // Populated automatically from reduced_strategy_ids in run_walk_forward().
        // Empty means compute all indicators.
        std::vector<int> reduced_indicator_ids;

        // Resume support: skip the first N schedule indices (already completed).
        // Set from checkpoint's current_schedule_idx when resuming.
        int start_schedule_idx = 0;

        // Completed runs from a previous checkpoint. They start the result,
        // and every progress callback and the final outputs include them.
        std::vector<WFORunResult> resumed_runs;

        // Progress callback invoked at walk-forward schedule
        // boundaries (one call per completed schedule). Arguments:
        // (schedules_done, total_schedules, runs_so_far), where runs_so_far
        // includes resumed_runs.
        //
        // CONTRACT: invoked SYNCHRONOUSLY on the orchestrator thread;
        // no cross-thread dispatch. See ScreenConfig::progress_cb for details.
        std::function<void(int done, int total, const std::vector<WFORunResult>& partial)>
            progress_cb;
    };

    // ---- Walk-forward Result ----
    struct WalkForwardResult {
        std::vector<WFORunResult> all_runs;  // one run per schedule x selection strategy

        // Meta-analysis: which selection strategy wins?
        std::string best_selection_strategy;
        float       best_selection_avg_oos_roc = 0.0f;

        // True if results were written to disk
        bool output_written = false;
    };

    // ---- Run screen stage ----
    // Loads data, enumerates grids, evaluates all strategy combos,
    // applies landscape analysis, and writes results.
    ScreenResult run_screen(const ScreenConfig& config);

    // ---- Run walk-forward stage ----
    // Walk-forward optimization over reduced parameter set.
    // Iterates every WFO config (each schedule x each selection strategy).
    // Shares backtest evaluations across selection strategies for each schedule.
    WalkForwardResult run_walk_forward(const WalkForwardConfig& config);

    // ---- Run walk-forward with pre-loaded data ----
    // For testing: skip file I/O, use provided bars directly.
    // sessions: optional per-bar SessionInfo for DST-correct Sharpe bucketing.
    // start_schedule_idx: skip first N schedule configs (resume support).
    // md: optional MarketData for session signal computation; when null, session signals are skipped.
    WalkForwardResult run_walk_forward_with_data(
        const Bar* bars, int num_bars, const std::vector<int>& reduced_strategy_ids,
        const CostModel& cost, float bars_per_day = 2880.0f,
        const std::vector<int>& reduced_indicator_ids = {}, const SessionInfo* sessions = nullptr,
        int start_schedule_idx = 0, const MarketData* md = nullptr);

    // ---- Run screen with pre-loaded data ----
    // For testing: skip file I/O, use provided bars and indicator data directly.
    // The caller provides:
    //   - bars: the bar array
    //   - num_bars: total bars in the array
    //   - eval_start/eval_end: evaluation range within bars
    //   - indicator_grid: the indicator grid to use
    //   - config: screen configuration (bar_file is ignored)
    //   - sessions: optional per-bar SessionInfo for time-gating
    //   - md: optional MarketData for session signal computation; when null, session signals are skipped
    ScreenResult run_screen_with_data(const Bar* bars, int num_bars, int eval_start, int eval_end,
                                      const IndicatorGrid& indicator_grid,
                                      const ScreenConfig&  config,
                                      const SessionInfo*   sessions = nullptr,
                                      const MarketData*    md       = nullptr);

    // ---- Write screen results to disk ----
    static bool write_results(const std::string& output_dir, const ScreenResult& result);

    // Re-derive landscape analysis from existing results (for checkpoint resume).
    // Populates reduced_strategy_ids, archetype counts, frontier, etc.
    // num_signals is the signal count the screen's strategy grid was built
    // over (screen_signal_count()); 0 infers it from the evaluated ids.
    ScreenResult reanalyze_results(std::vector<BacktestResult> results, float robustness_floor,
                                   const SanityFloors& floors      = SanityFloors{},
                                   int                 num_signals = 0) const;

    // Number of signals (technical indicators plus session signals when the
    // session model enables them) that a file-based screen builds its
    // strategy grid over.
    static int screen_signal_count(const SessionModel& session);

private:
    // ---- Internal pipeline steps ----

    // Step: compute all indicator series for the scan window
    void compute_indicators(const CLContext& ctx, const Bar* bars, int bar_offset, int num_bars,
                            const IndicatorGrid& indicator_grid, CLMemory& memory,
                            CLIndicators& cl_ind, std::vector<float*>& indicator_ptrs,
                            std::vector<IndicatorSpec>& indicator_specs);

    // Step: evaluate all strategy combos via CLBacktest
    void evaluate_strategies(const CLContext& ctx, const Bar* bars, int bar_offset, int num_bars,
                             int eval_start, int eval_end, const float* const* indicator_series,
                             int num_indicator_series, const IndicatorGrid& indicator_grid,
                             const ScreenConfig& config, std::vector<BacktestResult>& results,
                             const EarlyTermination& early_term = EarlyTermination{},
                             const SessionInfo*      sessions   = nullptr);

    // Step: build neighbor lookup functions for robustness + reduced set
    // The neighbor functions operate on the flat strategy index space.
    struct NeighborFunctions {
        // For robustness: returns (param_axis, neighbor_index) pairs
        std::function<std::vector<std::pair<int, int>>(int)> for_robustness;
        // For reduced set: returns flat neighbor indices
        std::function<std::vector<int>(int)> for_reduced_set;
        // Number of parameter axes
        int num_param_axes = 0;
    };

    // num_signals: signal count of the enumerated strategy grid (0 = infer).
    NeighborFunctions build_neighbor_functions(const std::vector<BacktestResult>& results,
                                               int num_signals) const;

    // Threads the WalkForwardConfig::progress_cb through to the
    // inner schedule loop in run_walk_forward_with_data without expanding its
    // test-facing signature. run_walk_forward copies the callback here before
    // calling run_walk_forward_with_data; the loop body invokes it per schedule.
    std::function<void(int done, int total, const std::vector<WFORunResult>& partial)>
        m_walk_forward_progress_cb;

    // WalkForwardConfig::oos_start_bar and resumed_runs, stashed for
    // run_walk_forward_with_data.
    int                              m_oos_start_bar = 0;
    const std::vector<WFORunResult>* m_resumed_runs  = nullptr;

    // Similarly, the screen stage's progress callback. Set by
    // run_screen before run_screen_with_data, invoked inside the
    // strategy-evaluation loop.
    std::function<void(int done, int total, const std::vector<BacktestResult>& partial)>
        m_screen_progress_cb;
};
