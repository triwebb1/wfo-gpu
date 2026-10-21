#pragma once
#include <limits>
#include <vector>
#include "metrics.h"

// ---------------------------------------------------------------------------
// Archetype classification for strategies on the Pareto frontier
// ---------------------------------------------------------------------------
enum class Archetype : uint8_t {
    GRINDER  = 0,  // High WR >= 90%, high green_weeks >= 75%, moderate ROC
    SNIPER   = 1,  // WR 30-55%, high ROC
    BALANCED = 2   // Everything else
};

inline const char* archetype_name(Archetype a) {
    switch (a) {
        case Archetype::GRINDER:
            return "grinder";
        case Archetype::SNIPER:
            return "sniper";
        case Archetype::BALANCED:
            return "balanced";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// ParetoEntry -- a strategy on the Pareto frontier with its archetype
// ---------------------------------------------------------------------------
struct ParetoEntry {
    int       original_index = -1;  // Index into the source BacktestResult vector
    Archetype archetype      = Archetype::BALANCED;
};

// Pareto frontier over 5 dims (all "higher is better"):
//   annual_roc, green_weeks_pct, 1/(1+max_red_weeks),
//   trade_recovery_ratio, profit_factor.
// Returns indices of non-dominated results. O(N^2 * D).
std::vector<int> extract_pareto_frontier(const std::vector<BacktestResult>& results);

// ---------------------------------------------------------------------------
// Pareto dominance check (exposed for testing)
//
// Returns true if A dominates B: A >= B on all dimensions AND A > B on at least one.
// ---------------------------------------------------------------------------
bool dominates(const BacktestResult& a, const BacktestResult& b);

// ---------------------------------------------------------------------------
// Get the 5 Pareto dimensions as a float array for a BacktestResult.
// dims[0..4] = { annual_roc, green_weeks_pct, 1/(1+max_red_weeks),
//                trade_recovery_ratio, profit_factor }
// ---------------------------------------------------------------------------
void          get_pareto_dimensions(const BacktestResult& r, float dims[5]);
constexpr int PARETO_DIM_COUNT = 5;

// Archetype classification thresholds.
// Derived from the full IS pool (all sanity-floor survivors), NOT frontier-only.
//   grinder_wr_min = max(0.90, pool_p75_wr)
//   grinder_gw_min = max(75,   pool_p75_green_weeks)
//   sniper_wr_max  = pool_p25_wr      (floored to 0.15 + eps)
//   sniper_roc_min = pool_p75_roc
// Grinder: WR >= grinder_wr_min AND green_weeks >= grinder_gw_min.
// Sniper:  WR <= sniper_wr_max AND ROC >= sniper_roc_min.
// Balanced: everything else.
struct ArchetypeThresholds {
    float grinder_wr_min = 0.90f;
    float grinder_gw_min = 75.0f;  // green_weeks_pct is 0-100 scale
    // Empty-pool defaults must make the Sniper check fail for ALL
    // candidates. Sniper requires `WR >= 0.15 && WR <= sniper_wr_max &&
    // ROC >= sniper_roc_min`. Setting sniper_wr_max = 0.0f makes the
    // combined WR constraint [0.15, 0.0] empty, so no candidate qualifies
    // on an uninitialized-thresholds instance. compute_archetype_thresholds()
    // overrides with pool P25 (bounded below by 0.15 + eps).
    float sniper_wr_max  = 0.0f;
    float sniper_roc_min = std::numeric_limits<float>::infinity();
};

// Compute thresholds from pool distribution
ArchetypeThresholds compute_archetype_thresholds(const std::vector<BacktestResult>& pool);

// Classify a single result given thresholds
Archetype classify_archetype(const BacktestResult& r, const ArchetypeThresholds& thresholds);

// Classify all frontier entries in-place.
// results: the full result set (indices reference into this).
// frontier_indices: indices into results that are on the frontier.
// full_is_pool: the full set of IS candidates that passed sanity floors,
//               used for computing archetype thresholds (percentiles).
//               If empty, falls back to using frontier results for thresholds.
// Returns ParetoEntry vector with archetype assigned.
std::vector<ParetoEntry> classify_frontier(const std::vector<BacktestResult>& results,
                                           const std::vector<int>&            frontier_indices,
                                           const std::vector<BacktestResult>& full_is_pool = {});
