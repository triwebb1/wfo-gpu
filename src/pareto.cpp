#include "pareto.h"
#include "common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

void get_pareto_dimensions(const BacktestResult& r, float dims[5]) {
    dims[0] = r.annual_roc;
    dims[1] = r.green_weeks_pct;
    dims[2] = 1.0f / (1.0f + static_cast<float>(r.max_red_weeks));
    // TRR and PF are clamped at 20, which keeps their sentinel values
    // (100 and 1e6 when nothing went negative) from dominating real ranges.
    dims[3] = std::min(r.trade_recovery_ratio, 20.0f);
    dims[4] = std::min(r.profit_factor, 20.0f);

    // Replace any non-finite dimension with -inf so the entry is
    // strictly dominated on that axis (cannot pollute the frontier).
    for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
        if (!std::isfinite(dims[d])) {
            dims[d] = -std::numeric_limits<float>::infinity();
        }
    }
}

bool dominates(const BacktestResult& a, const BacktestResult& b) {
    float da[PARETO_DIM_COUNT], db[PARETO_DIM_COUNT];
    get_pareto_dimensions(a, da);
    get_pareto_dimensions(b, db);

    bool at_least_one_strictly_better = false;
    for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
        if (da[d] < db[d]) return false;  // A is worse on dimension d
        if (da[d] > db[d]) at_least_one_strictly_better = true;
    }
    return at_least_one_strictly_better;
}

std::vector<int> extract_pareto_frontier(const std::vector<BacktestResult>& results) {
    int n = static_cast<int>(results.size());
    if (n == 0) return {};

    // Pre-extract all 5 Pareto dimensions into a compact array to avoid
    // scattered BacktestResult field access inside the O(N^2) inner loop.
    std::vector<std::array<float, PARETO_DIM_COUNT>> dims(n);
    for (int i = 0; i < n; ++i) {
        get_pareto_dimensions(results[i], dims[i].data());
    }

    // Build an index array and pre-sort by dimension 0 (annual_roc) descending.
    // This improves early termination: strategies with highest ROC are checked
    // first, so dominated strategies are eliminated sooner.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    // stable_sort with original-index tiebreak so iteration order is
    // deterministic across runs (no dependence on input ordering of equal-ROC
    // strategies). a, b are original indices from std::iota above.
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return dims[a][0] > dims[b][0] || (dims[a][0] == dims[b][0] && a < b);
    });

    // is_dominated[i] = true if some other strategy dominates i
    std::vector<bool> is_dominated(n, false);

    // O(N^2 x D) dominance check with precomputed dimensions and
    // reverse dominance for early break.
    for (int ii = 0; ii < n; ++ii) {
        int i = order[ii];
        if (is_dominated[i]) continue;

        const auto& di = dims[i];

        for (int jj = 0; jj < n; ++jj) {
            int j = order[jj];
            if (i == j || is_dominated[j]) continue;

            // Forward check: does i dominate j?
            {
                bool i_at_least_one = false;
                bool i_dominates    = true;
                for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
                    if (di[d] < dims[j][d]) {
                        i_dominates = false;
                        break;
                    }
                    if (di[d] > dims[j][d]) i_at_least_one = true;
                }
                if (i_dominates && i_at_least_one) {
                    is_dominated[j] = true;
                    continue;
                }
            }

            // Reverse check: does j dominate i?
            // If so, mark i dominated and break out of the inner loop immediately.
            {
                bool j_at_least_one = false;
                bool j_dominates    = true;
                for (int d = 0; d < PARETO_DIM_COUNT; ++d) {
                    if (dims[j][d] < di[d]) {
                        j_dominates = false;
                        break;
                    }
                    if (dims[j][d] > di[d]) j_at_least_one = true;
                }
                if (j_dominates && j_at_least_one) {
                    is_dominated[i] = true;
                    break;  // i is dominated; no need to check more j's
                }
            }
        }
    }

    std::vector<int> frontier;
    for (int i = 0; i < n; ++i) {
        if (!is_dominated[i]) {
            frontier.push_back(i);
        }
    }
    return frontier;
}

ArchetypeThresholds compute_archetype_thresholds(const std::vector<BacktestResult>& pool) {
    ArchetypeThresholds t;
    if (pool.empty()) return t;

    // Collect win rates, green_weeks, and ROC from the pool
    std::vector<float> win_rates;
    std::vector<float> green_weeks;
    std::vector<float> rocs;
    win_rates.reserve(pool.size());
    green_weeks.reserve(pool.size());
    rocs.reserve(pool.size());

    for (const auto& r : pool) {
        win_rates.push_back(r.win_rate);
        green_weeks.push_back(r.green_weeks_pct);
        rocs.push_back(r.annual_roc);
    }

    // Sort for percentile computation
    std::sort(win_rates.begin(), win_rates.end());
    std::sort(green_weeks.begin(), green_weeks.end());
    std::sort(rocs.begin(), rocs.end());

    float wr_p75  = ::percentile(win_rates, 0.75f);
    float wr_p25  = ::percentile(win_rates, 0.25f);
    float gw_p75  = ::percentile(green_weeks, 0.75f);
    float roc_p75 = ::percentile(rocs, 0.75f);

    // Grinder thresholds: max of hardcoded minimum and pool percentile
    t.grinder_wr_min = std::max(0.90f, wr_p75);
    t.grinder_gw_min = std::max(75.0f, gw_p75);

    // Sniper thresholds: bottom quartile WR (P25), top quartile ROC (P75).
    // Floor the WR cap at 0.15 + eps so a degenerate pool with
    // wr_p25 == 0 doesn't produce sniper_wr_max == 0 (which would exclude
    // every candidate from the Sniper archetype, even legitimate
    // low-WR/high-ROC strategies).
    constexpr float SNIPER_WR_FLOOR = 0.15f + 1e-6f;
    t.sniper_wr_max                 = std::max(wr_p25, SNIPER_WR_FLOOR);
    t.sniper_roc_min                = roc_p75;

    return t;
}

Archetype classify_archetype(const BacktestResult& r, const ArchetypeThresholds& thresholds) {
    // Grinder: high WR and high green weeks
    if (r.win_rate >= thresholds.grinder_wr_min && r.green_weeks_pct >= thresholds.grinder_gw_min) {
        return Archetype::GRINDER;
    }

    // Sniper: bottom quartile WR AND top quartile ROC, with a floor to
    // exclude degenerate strategies (e.g., 2% WR with a single huge winner)
    if (r.win_rate >= 0.15f && r.win_rate <= thresholds.sniper_wr_max &&
        r.annual_roc >= thresholds.sniper_roc_min) {
        return Archetype::SNIPER;
    }

    return Archetype::BALANCED;
}

std::vector<ParetoEntry> classify_frontier(const std::vector<BacktestResult>& results,
                                           const std::vector<int>&            frontier_indices,
                                           const std::vector<BacktestResult>& full_is_pool) {
    // Use full IS pool for threshold computation if provided,
    // otherwise fall back to frontier results.
    const std::vector<BacktestResult>* threshold_pool = &full_is_pool;

    std::vector<BacktestResult> frontier_pool;
    if (full_is_pool.empty()) {
        frontier_pool.reserve(frontier_indices.size());
        for (int idx : frontier_indices) {
            frontier_pool.push_back(results[idx]);
        }
        threshold_pool = &frontier_pool;
    }

    ArchetypeThresholds thresholds = compute_archetype_thresholds(*threshold_pool);

    std::vector<ParetoEntry> entries;
    entries.reserve(frontier_indices.size());
    for (int idx : frontier_indices) {
        ParetoEntry pe;
        pe.original_index = idx;
        pe.archetype      = classify_archetype(results[idx], thresholds);
        entries.push_back(pe);
    }
    return entries;
}
