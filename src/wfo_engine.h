#pragma once
// ---------------------------------------------------------------------------
// WFOEngine -- Walk-Forward Optimization engine (walk-forward stage)
//
// Implements the sliding-window WFO pipeline (see docs/ARCHITECTURE.md):
//   1. Slide forward by frequency_bars at each step
//   2. Per window: compute indicators, evaluate all reduced-grid combos,
//      apply quality floors, extract Pareto frontier, apply selection strategy
//   3. If no strategies pass floors: skip window (flat equity)
//   4. If >50% windows skipped: mark run as non-viable
//   5. Lag-2 validation: re-run winner at lag=2 on OOS period
//   6. If lag-2 Sharpe < lag-1 Sharpe * 0.5: compute PnL decomposition
//   7. Stitch OOS equity curves: concatenate PnL from each OOS window
//   8. Compute aggregate OOS metrics from the stitched equity curve
// ---------------------------------------------------------------------------

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backtest_engine.h"
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
#include "landscape.h"
#include "market_data.h"
#include "metrics.h"
#include "pareto.h"
#include "pnl_decomposition.h"
#include "selection.h"

// ---------------------------------------------------------------------------
// WFOConfig -- one walk-forward schedule configuration
// ---------------------------------------------------------------------------
struct WFOConfig {
    int frequency_bars = 0;  // re-optimization frequency in bars
    int lookback_bars  = 0;  // in-sample lookback in bars
    int frequency_days = 0;  // calendar equivalent for reporting
    int lookback_days  = 0;

    // Adaptive re-optimization trigger
    WFOTrigger trigger          = WFOTrigger::FIXED_INTERVAL;
    float      trigger_dd_frac  = 0.0f;  // for DD triggers (fraction of capital)
    int        trigger_losers_n = 0;     // for LOSERS triggers

    // Construct from a WFOScheduleConfig + approximate bars-per-day
    static WFOConfig from_schedule(const WFOScheduleConfig& sc, float bars_per_day);
};

// ---------------------------------------------------------------------------
// WFOWindowResult -- result for a single walk-forward window
// ---------------------------------------------------------------------------
struct WFOWindowResult {
    int              window_index  = 0;
    int              is_start_bar  = 0;
    int              is_end_bar    = 0;
    int              oos_start_bar = 0;
    int              oos_end_bar   = 0;
    int              strategy_id   = -1;  // winning strategy from IS
    std::string      archetype;           // Grinder/Sniper/Balanced
    BacktestResult   is_result;           // IS metrics of winner
    BacktestResult   oos_lag1;            // OOS at lag-1
    BacktestResult   oos_lag2;            // OOS at lag-2
    PnLDecomposition pnl_decomp;          // when lag-2 degrades
    bool             skipped = false;     // true if no strategies passed quality floors
};

// ---------------------------------------------------------------------------
// WindowEvaluation -- intermediate per-window evaluation shared across
// all selection strategies (the expensive part computed once per window)
// ---------------------------------------------------------------------------
struct WindowEvaluation {
    int is_start_bar  = 0;
    int is_end_bar    = 0;
    int oos_start_bar = 0;
    int oos_end_bar   = 0;

    // All backtest results for the IS period (reduced grid)
    std::vector<BacktestResult> is_results;

    // Indices that passed quality floors
    std::vector<int> passed_indices;

    // Pareto frontier from passed results
    std::vector<int> frontier_indices;

    // Classified frontier entries
    std::vector<ParetoEntry> frontier;

    // True if no strategies passed floors (empty frontier)
    bool empty_frontier = false;
};

// ---------------------------------------------------------------------------
// PrecomputedOOSWindow -- precomputed OOS indicator signals for one window.
// Shared across all 7 selection strategies to avoid redundant computation.
// ---------------------------------------------------------------------------
struct PrecomputedOOSWindow {
    // Signal arrays indexed by original indicator ID, NaN-filled for unused.
    std::vector<std::vector<float>> signal_storage;

    // Coordinate mappings (load-local space)
    int load_start  = 0;  // absolute bar index where load begins
    int load_count  = 0;  // number of bars loaded (including warmup)
    int eval_start  = 0;  // OOS eval start in load-local coords
    int eval_end    = 0;  // OOS eval end in load-local coords
    int num_signals = 0;  // total number of signal slots (indicator_grid.size())

    bool valid = false;  // true if indicators were successfully computed

    // Full signal pointer array (num_tech + num_session entries).
    // In the precomputed fast path, tech pointers reference the engine's
    // full-dataset arrays while session pointers reference signal_storage.
    // Both outlive OOS evaluation. Populated by precompute_oos_signals.
    std::vector<float*> cached_signal_ptrs;
};

// ---------------------------------------------------------------------------
// PrecomputedOOSData -- precomputed OOS indicators for all windows in a schedule.
// One entry per WindowEvaluation. Computed once, shared across all 7 selections.
// ---------------------------------------------------------------------------
struct PrecomputedOOSData {
    std::vector<PrecomputedOOSWindow> windows;
};

// ---------------------------------------------------------------------------
// WFORunResult -- result for one complete WFO run (one schedule + one selection)
// ---------------------------------------------------------------------------
struct WFORunResult {
    WFOConfig                    schedule;
    std::string                  selection_strategy;
    std::vector<WFOWindowResult> windows;

    // Stitched OOS equity curve
    std::vector<float>   equity_curve;
    std::vector<int64_t> equity_timestamps;
    std::vector<int32_t> equity_session_days;  // DST-correct session day per equity point

    // Full-run OOS metrics
    BacktestResult oos_aggregate;

    // Stability
    float parameter_stability =
        0.0f;  // share of consecutive traded windows keeping the same winner
    int  windows_skipped    = 0;
    bool sufficient_windows = true;  // <50% windows skipped
};

// ---------------------------------------------------------------------------
// WFOEngine -- walk-forward optimization engine
// ---------------------------------------------------------------------------
class WFOEngine {
public:
    WFOEngine() = default;

    // Set reduced indicator IDs for walk-forward optimization.
    // When non-empty, only these indicator grid indices are computed and
    // normalized in each WFO window. Unused signal arrays are filled with
    // NaN so the backtest engine's composite signal computation (which
    // skips NaN) produces correct results without index remapping.
    void set_reduced_indicator_ids(const std::vector<int>& ids) { m_reduced_indicator_ids = ids; }

    // Out-of-sample windows start no earlier than this bar. Set it to the end
    // of the screen range: the reduced set was chosen on those bars, so
    // trading them "out of sample" would reuse the selection data.
    // In-sample windows may still look back into the screen range.
    void set_min_oos_start(int bar) { m_min_oos_start = bar > 0 ? bar : 0; }

    // Set MarketData pointer for session signal computation.
    // When non-null, session signals are computed and appended to the
    // technical indicator signals in each WFO window.
    // Invalidates the cached strategy grid since signal count changes.
    void set_market_data(const MarketData* md) {
        m_market_data = md;
        m_strategy_grid.reset();  // invalidate: signal count changed
    }

    // Precompute all reduced-set indicators and session signals over the
    // full bar dataset. After this call, compute_signals_for_range() returns
    // slices into the precomputed data instead of recomputing per window.
    // Must be called after set_reduced_indicator_ids() and set_market_data().
    void precompute_all_signals(const Bar* bars, int num_bars, CLContext& ctx,
                                CLIndicators& cl_ind);

    // Read-only access to the cached indicator grid
    const IndicatorGrid& indicator_grid() const { return m_indicator_grid; }

    // Read-only access to the cached strategy grid (includes session signals if MarketData set)
    const StrategyGrid& strategy_grid() const { return get_strategy_grid(); }

    // Run a single WFO schedule with a specific selection strategy.
    // Uses pre-computed window evaluations for efficiency.
    WFORunResult run(const Bar* bars, int num_bars, const WFOConfig& schedule,
                     const std::string&      selection_strategy,
                     const std::vector<int>& reduced_strategy_ids, const CostModel& cost,
                     const SessionInfo* sessions = nullptr);

    // Evaluate all windows for a given schedule (the expensive part).
    // Returns one WindowEvaluation per window. This is shared across
    // all 7 selection strategies.
    std::vector<WindowEvaluation> evaluate_windows(const Bar* bars, int num_bars,
                                                   const WFOConfig&        schedule,
                                                   const std::vector<int>& reduced_strategy_ids,
                                                   const CostModel&        cost);

    // Apply a selection strategy to pre-evaluated windows to produce
    // a complete WFORunResult. This is the cheap post-processing step.
    // sessions: optional per-bar SessionInfo array for DST-correct daily PnL
    //           bucketing in the stitched equity Sharpe computation. Pass
    //           &MarketData::session(0) when available; nullptr falls back to
    //           the approximate fixed UTC shift.
    WFORunResult apply_selection(const Bar* bars, int num_bars, const WFOConfig& schedule,
                                 SelectionStrategy                    selection,
                                 const std::vector<WindowEvaluation>& evaluations,
                                 const std::vector<int>&              reduced_strategy_ids,
                                 const CostModel& cost, const SessionInfo* sessions = nullptr);

private:
    // Per-window IS quality floors
    static constexpr int   IS_MIN_TRADES = 30;
    static constexpr float IS_MIN_PF     = 1.0f;
    static constexpr float IS_MIN_ROC    = 0.0f;

    // Apply IS quality floors, returning indices of strategies that pass
    std::vector<int> apply_is_floors(const std::vector<BacktestResult>& results,
                                     const CostModel&                   cost) const;

    // Evaluate a single window's IS period
    void evaluate_is_window(const Bar* bars, int num_bars, int is_start, int is_end,
                            const std::vector<int>& reduced_strategy_ids, const CostModel& cost,
                            CLContext& ctx, CLIndicators& cl_ind, CLBacktest& cl_bt,
                            WindowEvaluation& eval);

    // Compute parameter stability (variance of strategy IDs across windows)
    static float compute_parameter_stability(const std::vector<WFOWindowResult>& windows);

    // Compute aggregate OOS metrics from stitched equity curve trades
    BacktestResult compute_stitched_metrics(const Bar* bars, int first_oos_start, int last_oos_end,
                                            const std::vector<float>&   equity_curve,
                                            const std::vector<int64_t>& equity_timestamps,
                                            const std::vector<int32_t>& equity_session_days,
                                            const CostModel&            cost) const;

    // Compute session signals for a bar range and append to signal arrays.
    // Returns the number of session signal series appended.
    // abs_bar_start/abs_bar_end are in MarketData global bar coordinates.
    // load_count is the number of bars in the local load window.
    int compute_session_signals(int abs_bar_start, int abs_bar_end, int load_count,
                                std::vector<std::vector<float>>& signal_storage,
                                std::vector<float*>&             signal_ptrs) const;

    // Pre-built indicator spec set (reusable across windows).
    // build_indicator_specs() populates this from indicator_grid + reduced set.
    struct IndicatorSpecSet {
        std::vector<IndicatorSpec>       specs;
        std::vector<std::pair<int, int>> series_map;  // (indicator_id, series_within)
        int                              total_series    = 0;
        int                              num_signals     = 0;  // = indicator_grid.size()
        bool                             use_reduced_set = false;
    };

    IndicatorSpecSet build_indicator_specs() const;

    // Compute indicators and normalize to signals for a bar range.
    // Populates signal_storage and signal_ptrs (sized to num_signals, NaN-filled
    // for unused indicators). Appends session signals. Returns total signal count.
    int compute_signals_for_range(const IndicatorSpecSet& spec_set, const Bar* bars, int load_start,
                                  int load_count, int abs_bar_end, CLContext& ctx,
                                  CLIndicators&                    cl_ind,
                                  std::vector<std::vector<float>>& signal_storage,
                                  std::vector<float*>&             signal_ptrs) const;

    // Reduced indicator IDs for walk-forward optimization (empty = compute all)
    std::vector<int> m_reduced_indicator_ids;

    // Optional MarketData for session signal computation (not owned)
    const MarketData* m_market_data = nullptr;

    // Earliest bar an out-of-sample window may start at (see set_min_oos_start).
    int m_min_oos_start = 0;

    // Full-dataset precomputed signals (walk-forward optimization).
    // When valid, compute_signals_for_range() returns slices into this
    // data instead of recomputing indicators per window.
    struct PrecomputedSignals {
        // Per-indicator signal data (only active indicators are stored;
        // inactive slots share nan_buffer via signal_ptrs)
        std::vector<std::vector<float>> tech_signals;  // one per active indicator
        std::vector<float>              nan_buffer;    // shared by inactive slots

        // Session signal per-day values (cached, broadcast per-window)
        std::vector<std::vector<float>> session_day_values;     // per grid entry x num_days
        std::vector<int32_t>            session_avail_after;    // per series
        std::vector<bool>               session_suppress_halt;  // per grid entry
        int                             session_day_start = 0;
        int                             session_day_end   = 0;

        // Full pointer array: [0..indicator_grid.size()) = tech signals,
        // then session signals appended. Same indexing as StrategyGrid.
        std::vector<float*> signal_ptrs;
        int                 num_tech_signals    = 0;  // = indicator_grid.size()
        int                 num_session_signals = 0;
        int                 total_signals       = 0;  // tech + session
        int                 total_bars          = 0;
        bool                valid               = false;
    };
    PrecomputedSignals m_precomputed;

    // Cached grids (constructed once, reused across all windows/selections)
    IndicatorGrid m_indicator_grid;

    // Cached session signal grid (reused across all windows)
    SessionSignalGrid m_session_signal_grid;

    // Cached StrategyGrid -- lazily initialized on first use since it depends
    // on the total signal count (technical + session).
    mutable std::unique_ptr<StrategyGrid> m_strategy_grid;
    const StrategyGrid&                   get_strategy_grid() const {
        if (!m_strategy_grid) {
            int total_signals = m_indicator_grid.size();
            if (m_market_data) {
                total_signals += m_session_signal_grid.total_signals();
            }
            m_strategy_grid = std::make_unique<StrategyGrid>(total_signals);
        }
        return *m_strategy_grid;
    }

public:
    // Overloads that accept shared CL instances to avoid redundant
    // device discovery + program compilation across calls.
    std::vector<WindowEvaluation> evaluate_windows(const Bar* bars, int num_bars,
                                                   const WFOConfig&        schedule,
                                                   const std::vector<int>& reduced_strategy_ids,
                                                   const CostModel& cost, CLContext& shared_ctx,
                                                   CLIndicators& shared_cl_ind,
                                                   CLBacktest&   shared_cl_bt);

    WFORunResult apply_selection(const Bar* bars, int num_bars, const WFOConfig& schedule,
                                 SelectionStrategy                    selection,
                                 const std::vector<WindowEvaluation>& evaluations,
                                 const std::vector<int>&              reduced_strategy_ids,
                                 const CostModel& cost, CLContext& shared_ctx,
                                 CLIndicators& shared_cl_ind, CLBacktest& shared_cl_bt,
                                 const SessionInfo* sessions = nullptr);

    // Precompute OOS indicator signals for all windows in a schedule.
    // Returns one PrecomputedOOSWindow per evaluation. This is computed
    // once per schedule and shared across all 7 apply_selection() calls,
    // eliminating 6x redundant OOS indicator computation.
    PrecomputedOOSData precompute_oos_signals(const Bar* bars, int num_bars,
                                              const std::vector<WindowEvaluation>& evaluations,
                                              CLContext& shared_ctx, CLIndicators& shared_cl_ind);

    // apply_selection overload that uses precomputed OOS signals
    WFORunResult apply_selection(const Bar* bars, int num_bars, const WFOConfig& schedule,
                                 SelectionStrategy                    selection,
                                 const std::vector<WindowEvaluation>& evaluations,
                                 const std::vector<int>&              reduced_strategy_ids,
                                 const CostModel& cost, CLContext& shared_ctx,
                                 CLIndicators& shared_cl_ind, CLBacktest& shared_cl_bt,
                                 PrecomputedOOSData& precomputed_oos,
                                 const SessionInfo*  sessions = nullptr);
};
