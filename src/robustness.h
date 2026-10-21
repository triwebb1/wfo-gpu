#pragma once
#include <vector>
#include <functional>
#include "metrics.h"

// ---------------------------------------------------------------------------
// Robustness -- parameter neighborhood stability scoring
//
// For each strategy S with parameter vector P, check all neighboring parameter
// values (one step in each direction on the sweep grid) while holding other
// params fixed.
//
// retention(N) = metric(N) / metric(S)
// sensitivity(p) = 1 - min(retention(N) for neighbors of p)
// robustness = 1 - max(sensitivity(p) for all params)
//
// Uses green_weeks_pct as the primary metric (spike-safe).
//
// Interpretation:
//   >= 0.70  Plateau -- robust, likely real edge
//   0.50-0.70  Ridge -- some sensitivity, proceed with caution
//   0.30-0.50  Hill -- fragile, likely partially overfit
//   < 0.30  Spike -- extremely fragile, probably a fluke
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// NeighborResult -- result for one neighboring parameter configuration
// ---------------------------------------------------------------------------
struct NeighborResult {
    int   param_axis   = 0;     // Which parameter dimension was varied
    float metric_value = 0.0f;  // green_weeks_pct of the neighbor
    // Tracks whether the neighbor's CPU-only metrics are fully
    // populated. compute_robustness() skips neighbors with this false.
    bool metrics_complete = false;
};

// ---------------------------------------------------------------------------
// Compute robustness score for a single strategy.
//
// Parameters:
//   center_metric:  green_weeks_pct of the strategy being scored
//   neighbors:      vector of NeighborResult, one per neighbor
//                   (can have multiple entries per param_axis for +1 and -1 steps)
//   num_param_axes: number of distinct parameter axes
//
// Returns robustness score in [0, 1].
// If center_metric is zero or negative, returns 0 (no meaningful edge).
// ---------------------------------------------------------------------------
float compute_robustness(float center_metric, const std::vector<NeighborResult>& neighbors,
                         int num_param_axes);

// ---------------------------------------------------------------------------
// Batch robustness computation for a grid of results.
//
// This operates as a lookup: for each strategy, its neighbors already exist
// in the results vector. The caller provides a function that, given a
// strategy index, returns the list of (param_axis, neighbor_index) pairs.
//
// Parameters:
//   results:        all backtest results from the sweep
//   get_neighbors:  given a strategy index, returns vector of
//                   (param_axis, neighbor_result_index) pairs
//   num_param_axes: number of distinct parameter axes
//
// Returns a vector of robustness scores, one per result.
//
// is_usable_neighbor: optional predicate that decides whether a
// candidate neighbor is allowed into the score. Defaults to "always true"
// when null. landscape.cpp passes `r.viable && r.metrics_complete` to enforce
// structural validity (drop non-economic and CPU-incomplete neighbors).
// ---------------------------------------------------------------------------
std::vector<float> compute_robustness_batch(
    const std::vector<BacktestResult>&                          results,
    const std::function<std::vector<std::pair<int, int>>(int)>& get_neighbors, int num_param_axes,
    const std::function<bool(const BacktestResult&)>& is_usable_neighbor = nullptr);

// ---------------------------------------------------------------------------
// Batch with axis coverage (REPORTING-ONLY).
//
// Same as compute_robustness_batch() but additionally returns:
//   - coverage[i] = (axes_with_neighbors / num_param_axes), in [0, 1]
//   - incomplete[i] = (coverage[i] < 0.5f)
// reduced-set gating does not use it; coverage is purely diagnostic.
// ---------------------------------------------------------------------------
struct RobustnessBatchResult {
    std::vector<float> scores;
    std::vector<float> coverage;
    std::vector<bool>  incomplete;
};
RobustnessBatchResult compute_robustness_batch_full(
    const std::vector<BacktestResult>&                          results,
    const std::function<std::vector<std::pair<int, int>>(int)>& get_neighbors, int num_param_axes,
    const std::function<bool(const BacktestResult&)>& is_usable_neighbor = nullptr);
