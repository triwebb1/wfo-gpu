#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// SelectionStrategy -- the 7 ways to pick a winner from the Pareto frontier
// ---------------------------------------------------------------------------
enum class SelectionStrategy : uint8_t {
    BEST_ROC          = 0,  // Highest annual_roc
    BEST_CONSISTENCY  = 1,  // Highest green_weeks_pct
    BEST_TRR          = 2,  // Highest trade_recovery_ratio
    BEST_GRINDER      = 3,  // Highest annual_roc among Grinder archetype
    BEST_SNIPER       = 4,  // Highest annual_roc among Sniper archetype
    FRONTIER_CENTROID = 5,  // Closest to centroid of frontier (normalized)
    ROC_X_CONSISTENCY = 6   // Highest annual_roc * green_weeks_pct
};

// Total number of selection strategies
constexpr int SELECTION_STRATEGY_COUNT = 7;

inline const char* selection_strategy_name(SelectionStrategy s) {
    switch (s) {
        case SelectionStrategy::BEST_ROC:
            return "best_roc";
        case SelectionStrategy::BEST_CONSISTENCY:
            return "best_consistency";
        case SelectionStrategy::BEST_TRR:
            return "best_trr";
        case SelectionStrategy::BEST_GRINDER:
            return "best_grinder";
        case SelectionStrategy::BEST_SNIPER:
            return "best_sniper";
        case SelectionStrategy::FRONTIER_CENTROID:
            return "frontier_centroid";
        case SelectionStrategy::ROC_X_CONSISTENCY:
            return "roc_x_consistency";
    }
    return "unknown";
}

// All selection strategies in order
inline SelectionStrategy all_selection_strategies(int index) {
    return static_cast<SelectionStrategy>(index);
}
