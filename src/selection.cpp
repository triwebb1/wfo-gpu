#include "selection.h"
#include <cmath>
#include <limits>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helper: find the entry with the best (highest) value of a metric
// among frontier entries matching an optional archetype filter.
// Returns the original_index, or -1 if no match.
// ---------------------------------------------------------------------------
static int best_by_metric(const std::vector<BacktestResult>& results,
                          const std::vector<ParetoEntry>&    frontier,
                          float (*metric_fn)(const BacktestResult&),
                          const Archetype* archetype_filter = nullptr) {
    int   best_idx = -1;
    float best_val = -std::numeric_limits<float>::infinity();

    for (const auto& pe : frontier) {
        if (archetype_filter && pe.archetype != *archetype_filter) continue;
        if (pe.original_index < 0 || pe.original_index >= static_cast<int>(results.size()))
            continue;
        float val = metric_fn(results[pe.original_index]);
        // Tiebreak on lowest original_index for deterministic selection.
        if (val > best_val || (val == best_val && pe.original_index < best_idx)) {
            best_val = val;
            best_idx = pe.original_index;
        }
    }
    return best_idx;
}

// Metric accessors
static float metric_annual_roc(const BacktestResult& r) {
    return r.annual_roc;
}
static float metric_green_weeks(const BacktestResult& r) {
    return r.green_weeks_pct;
}
static float metric_trr(const BacktestResult& r) {
    return r.trade_recovery_ratio;
}
static float metric_roc_x_consistency(const BacktestResult& r) {
    return r.annual_roc * r.green_weeks_pct;
}

namespace selection {

int best_roc(const std::vector<BacktestResult>& results, const std::vector<ParetoEntry>& frontier) {
    return best_by_metric(results, frontier, metric_annual_roc);
}

int best_consistency(const std::vector<BacktestResult>& results,
                     const std::vector<ParetoEntry>&    frontier) {
    return best_by_metric(results, frontier, metric_green_weeks);
}

int best_trr(const std::vector<BacktestResult>& results, const std::vector<ParetoEntry>& frontier) {
    return best_by_metric(results, frontier, metric_trr);
}

int best_grinder(const std::vector<BacktestResult>& results,
                 const std::vector<ParetoEntry>&    frontier) {
    Archetype arch = Archetype::GRINDER;
    int       idx  = best_by_metric(results, frontier, metric_annual_roc, &arch);
    if (idx < 0) {
        // Fallback to frontier_centroid
        return frontier_centroid(results, frontier);
    }
    return idx;
}

int best_sniper(const std::vector<BacktestResult>& results,
                const std::vector<ParetoEntry>&    frontier) {
    Archetype arch = Archetype::SNIPER;
    int       idx  = best_by_metric(results, frontier, metric_annual_roc, &arch);
    if (idx < 0) {
        // Fallback to frontier_centroid
        return frontier_centroid(results, frontier);
    }
    return idx;
}

int frontier_centroid(const std::vector<BacktestResult>& results,
                      const std::vector<ParetoEntry>&    frontier) {
    if (frontier.empty()) return -1;

    // Step 1: Compute min/max for each dimension across the frontier
    float dim_min[PARETO_DIM_COUNT];
    float dim_max[PARETO_DIM_COUNT];
    for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
        dim_min[d] = std::numeric_limits<float>::max();
        dim_max[d] = -std::numeric_limits<float>::max();
    }

    // Collect dimensions for all frontier entries, tracking which are valid
    std::vector<float> all_dims(frontier.size() * PARETO_DIM_COUNT, 0.0f);
    std::vector<bool>  valid_entry(frontier.size(), false);
    int                valid_count = 0;
    for (size_t i = 0; i < frontier.size(); ++i) {
        if (frontier[i].original_index < 0 ||
            frontier[i].original_index >= static_cast<int>(results.size()))
            continue;
        float* dims = &all_dims[i * PARETO_DIM_COUNT];
        get_pareto_dimensions(results[frontier[i].original_index], dims);
        // Guard against NaN/Inf from zero-trade or zero-loss strategies
        bool has_nan = false;
        for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
            if (std::isnan(dims[d]) || std::isinf(dims[d])) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) continue;
        valid_entry[i] = true;
        ++valid_count;
        for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
            dim_min[d] = std::min(dim_min[d], dims[d]);
            dim_max[d] = std::max(dim_max[d], dims[d]);
        }
    }
    if (valid_count == 0) return -1;

    // Step 2: Normalize to [0,1] and compute centroid (skip invalid entries)
    float centroid[PARETO_DIM_COUNT] = {};
    float range[PARETO_DIM_COUNT];
    for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
        range[d] = dim_max[d] - dim_min[d];
        if (range[d] < 1e-12f) range[d] = 1.0f;  // Avoid division by zero
    }

    for (size_t i = 0; i < frontier.size(); ++i) {
        if (!valid_entry[i]) continue;
        const float* dims = &all_dims[i * PARETO_DIM_COUNT];
        for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
            centroid[d] += (dims[d] - dim_min[d]) / range[d];
        }
    }
    float inv_n = 1.0f / static_cast<float>(valid_count);
    for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
        centroid[d] *= inv_n;
    }

    // Step 3: Find the frontier entry closest to the centroid (Euclidean distance)
    int   best_idx  = -1;
    float best_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < frontier.size(); ++i) {
        if (!valid_entry[i]) continue;
        const float* dims    = &all_dims[i * PARETO_DIM_COUNT];
        float        dist_sq = 0.0f;
        for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
            float norm = (dims[d] - dim_min[d]) / range[d];
            float diff = norm - centroid[d];
            dist_sq += diff * diff;
        }
        // Tiebreak on lowest original_index for deterministic selection.
        if (dist_sq < best_dist ||
            (dist_sq == best_dist && frontier[i].original_index < best_idx)) {
            best_dist = dist_sq;
            best_idx  = frontier[i].original_index;
        }
    }
    return best_idx;
}

int roc_x_consistency(const std::vector<BacktestResult>& results,
                      const std::vector<ParetoEntry>&    frontier) {
    return best_by_metric(results, frontier, metric_roc_x_consistency);
}

}  // namespace selection

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
int select_winner(SelectionStrategy strategy, const std::vector<BacktestResult>& results,
                  const std::vector<ParetoEntry>& frontier) {
    if (frontier.empty()) return -1;

    switch (strategy) {
        case SelectionStrategy::BEST_ROC:
            return selection::best_roc(results, frontier);
        case SelectionStrategy::BEST_CONSISTENCY:
            return selection::best_consistency(results, frontier);
        case SelectionStrategy::BEST_TRR:
            return selection::best_trr(results, frontier);
        case SelectionStrategy::BEST_GRINDER:
            return selection::best_grinder(results, frontier);
        case SelectionStrategy::BEST_SNIPER:
            return selection::best_sniper(results, frontier);
        case SelectionStrategy::FRONTIER_CENTROID:
            return selection::frontier_centroid(results, frontier);
        case SelectionStrategy::ROC_X_CONSISTENCY:
            return selection::roc_x_consistency(results, frontier);
    }
    return -1;
}
