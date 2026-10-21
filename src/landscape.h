#pragma once
// ---------------------------------------------------------------------------
// LandscapeAnalysis -- screen result characterization
//
// Given a vector of BacktestResult from the broad scan, this module:
//   1. Applies sanity floors (step 1)
//   2. Profiles survivors across the 6 key dimensions (step 2)
//   3. Delegates Pareto frontier extraction + archetype classification (step 3)
//   4. Computes distribution statistics for visualization data (step 4)
//   5. Builds the reduced parameter set (step 5):
//      frontier + parameter neighborhoods, robustness >= floor
//
// The reduced set becomes the input for walk-forward validation.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

#include "metrics.h"
#include "pareto.h"
#include "robustness.h"

// ---------------------------------------------------------------------------
// SanityFloors -- configurable sanity thresholds (see docs/METHODOLOGY.md)
// ---------------------------------------------------------------------------
struct SanityFloors {
    float min_total_pnl     = 0.0f;  // total_pnl > 0
    int   min_total_trades  = 100;   // total_trades >= 100
    float min_profit_factor = 1.0f;  // profit_factor > 1.0
    float min_annual_roc    = 0.0f;  // annual_roc > 0
    // Gate on calendar-year losses, not month losses: strategies with
    // occasional losing months but no losing calendar year pass.
    int max_negative_years = 0;  // negative_years == 0
};

// ---------------------------------------------------------------------------
// DistributionStats -- summary statistics for a metric across survivors
// ---------------------------------------------------------------------------
struct DistributionStats {
    float min    = 0.0f;
    float p25    = 0.0f;
    float median = 0.0f;
    float p75    = 0.0f;
    float max    = 0.0f;
    float mean   = 0.0f;
};

// ---------------------------------------------------------------------------
// LandscapeAnalysis -- complete screen result characterization
// ---------------------------------------------------------------------------
struct LandscapeAnalysis {
    // Survivors whose average trade clears the viability floor (covers the
    // spread with margin); only these compete for the Pareto frontier.
    int viable_survivors = 0;

    // Counts
    int total_candidates     = 0;
    int passed_sanity_floors = 0;
    int on_pareto_frontier   = 0;
    int grinder_count        = 0;
    int sniper_count         = 0;
    int balanced_count       = 0;

    // Distribution stats for survivors
    DistributionStats roc_stats;
    DistributionStats green_weeks_stats;
    DistributionStats robustness_stats;
    DistributionStats profit_factor_stats;
    DistributionStats trr_stats;

    // Pareto frontier entries with archetypes
    std::vector<ParetoEntry> frontier;

    // Robustness scores for ALL results (indexed same as input results vector)
    std::vector<float> robustness_scores;

    // REPORTING-ONLY: axis coverage diagnostic, parallel
    // to robustness_scores. coverage = (axes with usable neighbors) /
    // num_param_axes. incomplete = (coverage < 0.5). Reduced-set gating still
    // uses ONLY robustness_score >= floor; nothing filters on incomplete.
    std::vector<float> robustness_coverage;
    std::vector<bool>  robustness_incomplete;

    // Indices of results that passed sanity floors
    std::vector<int> sanity_passed_indices;

    // Reduced set as positions in the results vector (frontier +
    // neighborhoods, robustness >= floor). Map through results[i].strategy_id
    // to get the strategy-grid ids the walk-forward stage consumes.
    std::vector<int> reduced_set_ids;
};

// ---------------------------------------------------------------------------
// Apply sanity floors to a set of results.
// Returns indices into `results` that pass all floor checks.
// ---------------------------------------------------------------------------
std::vector<int> apply_sanity_floors(const std::vector<BacktestResult>& results,
                                     const SanityFloors&                floors = SanityFloors{});

// ---------------------------------------------------------------------------
// Compute distribution statistics for a vector of float values.
// ---------------------------------------------------------------------------
DistributionStats compute_distribution(const std::vector<float>& values);

// ---------------------------------------------------------------------------
// Build the reduced parameter set from the Pareto frontier.
//
// The reduced set includes:
//   1. All strategies on the Pareto frontier with robustness >= robustness_floor
//   2. Their parameter neighborhood indices (provided by get_neighbors)
//      that also have robustness >= robustness_floor
//
// Parameters:
//   frontier_indices:  indices on the Pareto frontier (into the full results vector)
//   robustness_scores: robustness score per result (indexed into full results vector)
//   robustness_floor:  minimum robustness for inclusion (default 0.50)
//   get_neighbors:     given a result index, returns neighbor result indices
//
// Returns: sorted, deduplicated vector of result indices for walk-forward.
// ---------------------------------------------------------------------------
std::vector<int> build_reduced_set(const std::vector<int>&                     frontier_indices,
                                   const std::vector<float>&                   robustness_scores,
                                   float                                       robustness_floor,
                                   const std::function<std::vector<int>(int)>& get_neighbors);

// ---------------------------------------------------------------------------
// Full landscape analysis pipeline.
//
// Parameters:
//   results:            all backtest results from the broad scan
//   robustness_floor:   minimum robustness for walk-forward inclusion (0.50)
//   get_neighbors_for_robustness: for robustness batch computation,
//       given index -> vector of (param_axis, neighbor_index) pairs
//   num_param_axes:     number of parameter axes for robustness
//   get_neighbors_for_reduced_set: for reduced set construction,
//       given index -> vector of neighbor indices
//
// Returns: fully populated LandscapeAnalysis.
// ---------------------------------------------------------------------------
LandscapeAnalysis analyze_landscape(
    const std::vector<BacktestResult>& results, float robustness_floor,
    const std::function<std::vector<std::pair<int, int>>(int)>& get_neighbors_for_robustness,
    int num_param_axes, const std::function<std::vector<int>(int)>& get_neighbors_for_reduced_set,
    const SanityFloors& floors = SanityFloors{});
