#include "landscape.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

// ---------------------------------------------------------------------------
// apply_sanity_floors
// ---------------------------------------------------------------------------

std::vector<int> apply_sanity_floors(const std::vector<BacktestResult>& results,
                                     const SanityFloors&                floors) {
    std::vector<int> passed;
    passed.reserve(results.size() / 2);  // rough estimate

    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        const auto& r = results[i];

        if (r.total_pnl <= floors.min_total_pnl) continue;
        if (r.total_trades < floors.min_total_trades) continue;
        if (r.profit_factor <= floors.min_profit_factor) continue;
        if (r.annual_roc <= floors.min_annual_roc) continue;
        // Gate on calendar-year losses (not month losses).
        if (r.negative_years > floors.max_negative_years) continue;

        passed.push_back(i);
    }

    return passed;
}

// ---------------------------------------------------------------------------
// compute_distribution
// ---------------------------------------------------------------------------

DistributionStats compute_distribution(const std::vector<float>& values) {
    DistributionStats stats;
    if (values.empty()) return stats;

    std::vector<float> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    int n = static_cast<int>(sorted.size());

    stats.min = sorted.front();
    stats.max = sorted.back();

    // Mean
    double sum = 0.0;
    for (float v : sorted) sum += static_cast<double>(v);
    stats.mean = static_cast<float>(sum / n);

    stats.p25    = ::percentile(sorted, 0.25f);
    stats.median = ::percentile(sorted, 0.50f);
    stats.p75    = ::percentile(sorted, 0.75f);

    return stats;
}

// ---------------------------------------------------------------------------
// build_reduced_set
// ---------------------------------------------------------------------------

std::vector<int> build_reduced_set(const std::vector<int>&                     frontier_indices,
                                   const std::vector<float>&                   robustness_scores,
                                   float                                       robustness_floor,
                                   const std::function<std::vector<int>(int)>& get_neighbors) {
    std::set<int> reduced;

    for (int idx : frontier_indices) {
        // Only include frontier strategies that meet the robustness floor
        if (idx < 0 || idx >= static_cast<int>(robustness_scores.size())) continue;
        if (robustness_scores[idx] < robustness_floor) continue;

        reduced.insert(idx);

        // Include neighbors that also meet the robustness floor
        if (get_neighbors) {
            auto neighbors = get_neighbors(idx);
            for (int nb_idx : neighbors) {
                if (nb_idx < 0 || nb_idx >= static_cast<int>(robustness_scores.size())) continue;
                if (robustness_scores[nb_idx] >= robustness_floor) {
                    reduced.insert(nb_idx);
                }
            }
        }
    }

    // Convert to sorted vector
    return std::vector<int>(reduced.begin(), reduced.end());
}

// ---------------------------------------------------------------------------
// analyze_landscape -- full pipeline
// ---------------------------------------------------------------------------

LandscapeAnalysis analyze_landscape(
    const std::vector<BacktestResult>& results, float robustness_floor,
    const std::function<std::vector<std::pair<int, int>>(int)>& get_neighbors_for_robustness,
    int num_param_axes, const std::function<std::vector<int>(int)>& get_neighbors_for_reduced_set,
    const SanityFloors& floors) {
    LandscapeAnalysis la;
    la.total_candidates = static_cast<int>(results.size());

    if (results.empty()) return la;

    // Step 1: Sanity floors
    la.sanity_passed_indices = apply_sanity_floors(results, floors);
    la.passed_sanity_floors  = static_cast<int>(la.sanity_passed_indices.size());

    if (la.sanity_passed_indices.empty()) return la;

    // Step 2: Compute robustness scores for ALL results (lookup-based).
    // Pass an is_usable_neighbor predicate enforcing structural
    // validity -- exclude neighbors lacking economic viability or
    // CPU-complete metrics. Without this, a frontier strategy whose neighbors
    // include broken/incomplete results gets falsely scored as fragile.
    auto is_usable = [](const BacktestResult& r) { return r.viable && r.metrics_complete; };
    auto rb = compute_robustness_batch_full(results, get_neighbors_for_robustness, num_param_axes,
                                            is_usable);
    la.robustness_scores     = std::move(rb.scores);
    la.robustness_coverage   = std::move(rb.coverage);    // reporting-only
    la.robustness_incomplete = std::move(rb.incomplete);  // reporting-only

    // Step 3: Build a filtered results vector for Pareto extraction
    // (only those passing sanity floors AND marked viable; exclude
    // non-viable strategies that lack the avg_pnl >= 2x spread floor here).
    std::vector<BacktestResult> survivors;
    std::vector<int>            survivor_original_indices;
    survivors.reserve(la.sanity_passed_indices.size());
    survivor_original_indices.reserve(la.sanity_passed_indices.size());
    for (int idx : la.sanity_passed_indices) {
        const BacktestResult& r = results[idx];
        if (!r.viable) continue;
        survivors.push_back(r);
        survivor_original_indices.push_back(idx);
    }

    la.viable_survivors = static_cast<int>(survivors.size());

    // Extract Pareto frontier among survivors
    std::vector<int> survivor_frontier = extract_pareto_frontier(survivors);

    // Map survivor frontier indices back to original result indices.
    // Use survivor_original_indices (which tracks the !viable filter)
    // instead of la.sanity_passed_indices, since survivors is a subset.
    std::vector<int> original_frontier;
    original_frontier.reserve(survivor_frontier.size());
    for (int sf_idx : survivor_frontier) {
        original_frontier.push_back(survivor_original_indices[sf_idx]);
    }

    la.on_pareto_frontier = static_cast<int>(original_frontier.size());

    // Step 4: Classify archetypes on the frontier, using full survivor pool for thresholds
    la.frontier = classify_frontier(results, original_frontier, survivors);

    la.grinder_count  = 0;
    la.sniper_count   = 0;
    la.balanced_count = 0;
    for (const auto& pe : la.frontier) {
        switch (pe.archetype) {
            case Archetype::GRINDER:
                ++la.grinder_count;
                break;
            case Archetype::SNIPER:
                ++la.sniper_count;
                break;
            case Archetype::BALANCED:
                ++la.balanced_count;
                break;
        }
    }

    // Step 5: Distribution statistics for survivors
    {
        std::vector<float> rocs, gws, robs, pfs, trrs;
        rocs.reserve(la.sanity_passed_indices.size());
        gws.reserve(la.sanity_passed_indices.size());
        robs.reserve(la.sanity_passed_indices.size());
        pfs.reserve(la.sanity_passed_indices.size());
        trrs.reserve(la.sanity_passed_indices.size());

        for (int idx : la.sanity_passed_indices) {
            rocs.push_back(results[idx].annual_roc);
            gws.push_back(results[idx].green_weeks_pct);
            robs.push_back(la.robustness_scores[idx]);
            pfs.push_back(results[idx].profit_factor);
            trrs.push_back(results[idx].trade_recovery_ratio);
        }

        la.roc_stats           = compute_distribution(rocs);
        la.green_weeks_stats   = compute_distribution(gws);
        la.robustness_stats    = compute_distribution(robs);
        la.profit_factor_stats = compute_distribution(pfs);
        la.trr_stats           = compute_distribution(trrs);
    }

    // Step 6: Build reduced set (frontier + neighborhoods, robustness >= floor).
    // A neighbor must also be usable in the sense robustness scoring uses:
    // viable, with complete metrics. Every viable result is fully computed
    // on every device, so the reduced set does not depend on the device.
    std::function<std::vector<int>(int)> usable_neighbors;
    if (get_neighbors_for_reduced_set) {
        usable_neighbors = [&](int idx) {
            std::vector<int> out;
            for (int nb : get_neighbors_for_reduced_set(idx)) {
                if (nb >= 0 && nb < static_cast<int>(results.size()) && is_usable(results[nb]))
                    out.push_back(nb);
            }
            return out;
        };
    }
    la.reduced_set_ids = build_reduced_set(original_frontier, la.robustness_scores,
                                           robustness_floor, usable_neighbors);

    return la;
}
