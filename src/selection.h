#pragma once
#include <vector>
#include "metrics.h"
#include "pareto.h"
#include "grids/selection_grid.h"

// Selection: pick one winner from the classified Pareto frontier.
// Archetype-specific strategies fall back to frontier_centroid when their
// archetype is empty. Returns -1 only on empty frontier.

// Main entry point: select the winner using a given strategy.
int select_winner(SelectionStrategy strategy, const std::vector<BacktestResult>& results,
                  const std::vector<ParetoEntry>& frontier);

// Individual selection strategies (exposed for testing)
namespace selection {

// Highest annual_roc on the frontier
int best_roc(const std::vector<BacktestResult>& results, const std::vector<ParetoEntry>& frontier);

// Highest green_weeks_pct on the frontier
int best_consistency(const std::vector<BacktestResult>& results,
                     const std::vector<ParetoEntry>&    frontier);

// Highest trade_recovery_ratio on the frontier
int best_trr(const std::vector<BacktestResult>& results, const std::vector<ParetoEntry>& frontier);

// Highest annual_roc among Grinder archetype on the frontier.
// Falls back to frontier_centroid if no Grinders.
int best_grinder(const std::vector<BacktestResult>& results,
                 const std::vector<ParetoEntry>&    frontier);

// Highest annual_roc among Sniper archetype on the frontier.
// Falls back to frontier_centroid if no Snipers.
int best_sniper(const std::vector<BacktestResult>& results,
                const std::vector<ParetoEntry>&    frontier);

// Closest to the centroid of the frontier.
// Dimensions normalized to [0,1] before distance computation.
int frontier_centroid(const std::vector<BacktestResult>& results,
                      const std::vector<ParetoEntry>&    frontier);

// Highest annual_roc * green_weeks_pct product on the frontier.
int roc_x_consistency(const std::vector<BacktestResult>& results,
                      const std::vector<ParetoEntry>&    frontier);

}  // namespace selection
