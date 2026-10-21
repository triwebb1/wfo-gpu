#include "robustness.h"
#include <algorithm>
#include <cmath>
#include <limits>

float compute_robustness(float center_metric, const std::vector<NeighborResult>& neighbors,
                         int num_param_axes) {
    // NaN/inf guard at the center. A non-finite or non-positive
    // center has no meaningful edge to measure stability of -- early-return 0.
    if (!std::isfinite(center_metric) || center_metric <= 0.0f) return 0.0f;

    // If no neighbors, robustness cannot be assessed -- return 0 (worst)
    // to avoid artificially favoring grid-boundary strategies
    if (neighbors.empty() || num_param_axes <= 0) return 0.0f;

    // Compute sensitivity for each parameter axis.
    // sensitivity(p) = 1 - min(retention(N) for all neighbors of p)
    // where retention(N) = metric(N) / metric(S), clamped to [0, 1]
    float max_sensitivity = 0.0f;
    // Track whether ANY neighbor survived the usability filters.
    // If every neighbor was filtered (all !metrics_complete or all non-finite),
    // we must NOT return 1.0 (falsely maximum-robustness). Return 0.0 instead,
    // matching the neighbors.empty() early-return semantics.
    bool any_usable_neighbor = false;

    for (int axis = 0; axis < num_param_axes; ++axis) {
        float min_retention = 1.0f;  // Best case: all neighbors match
        // has_neighbor stays true if EITHER +1 or -1 step exists for this
        // axis. Grid-boundary strategies (only one neighbor on an axis) are
        // NOT marked incomplete -- they get scored from the side that exists.
        bool has_neighbor = false;

        for (const auto& nb : neighbors) {
            if (nb.param_axis != axis) continue;

            // Skip neighbors whose CPU-only metrics aren't filled in.
            // Without this guard, a GPU-only neighbor (green_weeks_pct = 0
            // by default) would falsely appear as a sensitivity spike.
            if (!nb.metrics_complete) continue;

            // NaN/inf guard at neighbor read.
            if (!std::isfinite(nb.metric_value)) continue;

            has_neighbor        = true;
            any_usable_neighbor = true;

            float retention = nb.metric_value / center_metric;
            // Clamp retention to [0, 1] -- neighbor can't be "better" for stability
            retention     = std::clamp(retention, 0.0f, 1.0f);
            min_retention = std::min(min_retention, retention);
        }

        if (has_neighbor) {
            float sensitivity = 1.0f - min_retention;
            max_sensitivity   = std::max(max_sensitivity, sensitivity);
        }
    }

    // If no neighbor survived filtering, robustness is unassessable.
    // Return 0.0 (worst) to match the neighbors.empty() early-return.
    if (!any_usable_neighbor) return 0.0f;

    // robustness = 1 - max_sensitivity
    return 1.0f - max_sensitivity;
}

// Internal helper: full pass over results, optionally tracking axis coverage.
static RobustnessBatchResult compute_robustness_batch_impl(
    const std::vector<BacktestResult>&                          results,
    const std::function<std::vector<std::pair<int, int>>(int)>& get_neighbors, int num_param_axes,
    const std::function<bool(const BacktestResult&)>& is_usable_neighbor, bool track_coverage) {
    int                   n = static_cast<int>(results.size());
    RobustnessBatchResult out;
    out.scores.assign(n, 0.0f);
    if (track_coverage) {
        out.coverage.assign(n, 0.0f);
        out.incomplete.assign(n, true);
    }

    for (int i = 0; i < n; ++i) {
        // NaN guard at center read.
        float center_metric = results[i].green_weeks_pct;
        if (!std::isfinite(center_metric) || center_metric <= 0.0f) {
            // out.scores[i] already 0; coverage left at 0 / incomplete = true
            continue;
        }

        auto                        neighbor_pairs = get_neighbors(i);
        std::vector<NeighborResult> nbrs;
        nbrs.reserve(neighbor_pairs.size());

        // Per-axis presence tracking for coverage.
        std::vector<uint8_t> axis_seen;
        if (track_coverage && num_param_axes > 0) {
            axis_seen.assign(num_param_axes, 0);
        }

        for (const auto& [axis, nb_idx] : neighbor_pairs) {
            if (nb_idx < 0 || nb_idx >= n) continue;
            const BacktestResult& nb = results[nb_idx];

            // Structural-validity filter -- exclude neighbors
            // lacking economic viability or CPU-complete metrics. Without
            // this filter, robustness collapses toward 0 for any frontier
            // entry whose neighbors include broken/incomplete results.
            if (is_usable_neighbor && !is_usable_neighbor(nb)) continue;

            // NaN guard at neighbor read.
            if (!std::isfinite(nb.green_weeks_pct)) continue;

            // Defensive even when caller didn't supply a predicate.
            if (!nb.metrics_complete) continue;

            NeighborResult nr;
            nr.param_axis       = axis;
            nr.metric_value     = nb.green_weeks_pct;
            nr.metrics_complete = nb.metrics_complete;
            nbrs.push_back(nr);

            if (track_coverage && axis >= 0 && axis < num_param_axes) {
                axis_seen[axis] = 1;
            }
        }

        out.scores[i] = compute_robustness(center_metric, nbrs, num_param_axes);

        if (track_coverage) {
            int seen = 0;
            for (uint8_t v : axis_seen)
                if (v) ++seen;
            float cov       = (num_param_axes > 0)
                                  ? static_cast<float>(seen) / static_cast<float>(num_param_axes)
                                  : 0.0f;
            out.coverage[i] = cov;
            // An axis with a single neighbor still counts toward coverage
            // (boundary strategies aren't marked incomplete on that basis).
            // "Incomplete" here means coverage < 50% of axes had ANY usable
            // neighbor -- a much weaker signal than per-axis boundary status.
            out.incomplete[i] = (cov < 0.5f);
        }
    }

    return out;
}

std::vector<float> compute_robustness_batch(
    const std::vector<BacktestResult>&                          results,
    const std::function<std::vector<std::pair<int, int>>(int)>& get_neighbors, int num_param_axes,
    const std::function<bool(const BacktestResult&)>& is_usable_neighbor) {
    auto r =
        compute_robustness_batch_impl(results, get_neighbors, num_param_axes, is_usable_neighbor,
                                      /*track_coverage=*/false);
    return std::move(r.scores);
}

RobustnessBatchResult compute_robustness_batch_full(
    const std::vector<BacktestResult>&                          results,
    const std::function<std::vector<std::pair<int, int>>(int)>& get_neighbors, int num_param_axes,
    const std::function<bool(const BacktestResult&)>& is_usable_neighbor) {
    return compute_robustness_batch_impl(results, get_neighbors, num_param_axes, is_usable_neighbor,
                                         /*track_coverage=*/true);
}
