#pragma once
#include <array>
#include <cstdint>
#include <utility>
#include <vector>
#include "../backtest_engine.h"

// ---------------------------------------------------------------------------
// ExitModePreset -- discrete exit mode presets for grid enumeration
// These map to StrategyConfig::ExitMode + timeout_bars
// ---------------------------------------------------------------------------
enum class ExitModePreset : uint8_t {
    SIGNAL_REVERSAL     = 0,
    BAR_TIMEOUT_10      = 1,
    BAR_TIMEOUT_20      = 2,
    BAR_TIMEOUT_50      = 3,
    BAR_TIMEOUT_100     = 4,
    TRAILING_STOP_1_0   = 5,  // 1.0 x ATR
    TRAILING_STOP_1_5   = 6,  // 1.5 x ATR
    TRAILING_STOP_2_0   = 7,  // 2.0 x ATR
    TRAILING_STOP_2_5   = 8,  // 2.5 x ATR
    OPPOSING_SIGNAL     = 9,
    MAX_LOSS_ATR_2_0    = 10,  // 2.0 x ATR fixed stop
    MAX_LOSS_ATR_3_0    = 11,  // 3.0 x ATR fixed stop
    MAX_LOSS_ATR_4_0    = 12,  // 4.0 x ATR fixed stop
    MAX_LOSS_ATR_5_0    = 13,  // 5.0 x ATR fixed stop
    TAKE_PROFIT_ATR_1_0 = 14,  // TP at 1.0 x ATR
    TAKE_PROFIT_ATR_1_5 = 15,  // TP at 1.5 x ATR
    TAKE_PROFIT_ATR_2_0 = 16,  // TP at 2.0 x ATR
    TAKE_PROFIT_ATR_3_0 = 17,  // TP at 3.0 x ATR
    TP_SL_1_0_1_0       = 18,  // TP=1.0, SL=1.0 (1:1 R:R)
    TP_SL_1_5_1_0       = 19,  // TP=1.5, SL=1.0
    TP_SL_2_0_1_0       = 20,  // TP=2.0, SL=1.0 (2:1 R:R)
    TP_SL_3_0_1_0       = 21,  // TP=3.0, SL=1.0 (3:1 R:R)
    TP_SL_2_0_1_5       = 22,  // TP=2.0, SL=1.5
    TP_SL_2_0_2_0       = 23   // TP=2.0, SL=2.0 (1:1 R:R but wider)
};

// ---------------------------------------------------------------------------
// StrategyGridEntry -- one strategy composition from the grid
// ---------------------------------------------------------------------------
struct StrategyGridEntry {
    // Active signal subset (signal indices, see to_strategy_config)
    int indicator_ids[4] = {0, 0, 0, 0};
    int num_indicators   = 0;

    // Signal weights per indicator
    float weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // Indicator roles (SIGNAL or FILTER)
    StrategyConfig::IndicatorRole indicator_roles[4] = {
        StrategyConfig::SIGNAL, StrategyConfig::SIGNAL, StrategyConfig::SIGNAL,
        StrategyConfig::SIGNAL};

    // Aggregation mode
    StrategyConfig::AggMode aggregation = StrategyConfig::WEIGHTED_SUM;

    // Entry threshold
    float entry_threshold = 0.5f;

    // Exit mode
    ExitModePreset exit_mode = ExitModePreset::SIGNAL_REVERSAL;

    // Trading hours
    StrategyConfig::TradingHours trading_hours = StrategyConfig::ETH;

    // Returns true if this entry trades identically to another entry that
    // is kept (see the rules in strategy_grid.cpp). The screen skips these.
    bool is_redundant() const;

    // Convert to a StrategyConfig suitable for the backtest engine.
    // Maps each ExitModePreset to the appropriate ExitMode + parameters.
    // indicator_ids are signal indices: the technical indicators in
    // IndicatorGrid order, then the session signals.
    StrategyConfig to_strategy_config() const;
};

// ---------------------------------------------------------------------------
// StrategyGrid -- enumerates all strategy compositions
//
// The grid is far too large to store (about 4e15 entries over the 643
// signals of the default configuration), so it is never materialized:
//   - total_count(): total number of combos (computed, not stored)
//   - entry_at(index): rebuild a combo from its flat index by
//     combinatorial unranking
//   - subset_count(): number of signal subsets (up to 4 of N)
//
// Ids are ordered by subset size: every 1-signal entry comes before every
// 2-signal entry, and so on. The screen enumerates ids in order, so its
// strategy budget decides how far into the grid it reaches.
//
// The screen and the walk-forward stage both build the grid over the same N
// signals (Orchestrator::screen_signal_count), so an id means the same
// strategy in both.
// ---------------------------------------------------------------------------
class StrategyGrid {
public:
    // Construct with the number of available signals.
    explicit StrategyGrid(int n_indicators);

    // Total number of strategy compositions (can be very large)
    int64_t total_count() const { return m_total_count; }

    // Number of indicator subsets C(N,1)+C(N,2)+C(N,3)+C(N,4)
    int64_t subset_count() const { return m_subset_count; }

    // Number of weight combos per indicator count (4^k for k indicators)
    static int weight_combos(int k);

    // Number of aggregation modes
    static constexpr int AGG_MODES = 3;

    // Number of entry thresholds
    static constexpr int ENTRY_THRESHOLDS = 4;

    // Number of exit modes
    static constexpr int EXIT_MODES = 24;

    // Number of trading hours options
    static constexpr int TRADING_HOURS_OPTIONS = 4;  // ETH, RTH, MORNING, AFTERNOON

    // Number of role patterns: ALL_SIGNAL, LAST_FILTER
    static constexpr int ROLE_PATTERNS = 2;

    // Weight values
    static constexpr float WEIGHT_VALUES[] = {0.25f, 0.50f, 0.75f, 1.0f};

    // Entry threshold values
    static constexpr float THRESHOLD_VALUES[] = {0.5f, 0.6f, 0.7f, 0.8f};

    // Per-weight-combo block size: product of all axes EXCEPT weight_rank.
    // entry_at()'s inner layout (innermost first):
    //   role(2) > hours(4) > exit(24) > thresh(4) > agg(3) > weight_rank(4^k)
    // PER_WEIGHT = AGG * THRESH * EXIT * HOURS * ROLE = 3*4*24*4*2 = 2304.
    // The static_assert below pins this layout: any future grid edit that
    // changes this product MUST also update orchestrator.cpp neighbor geometry.
    static constexpr int PER_WEIGHT =
        AGG_MODES * ENTRY_THRESHOLDS * EXIT_MODES * TRADING_HOURS_OPTIONS * ROLE_PATTERNS;
    static_assert(PER_WEIGHT == 2304,
                  "grid layout drift -- review neighbor geometry at orchestrator.cpp");

    // ---- Axis descriptor for neighbor scoring -----------------------------
    // Each AxisDescriptor describes one parameter axis used by robustness
    // neighbor lookup. The 6 axes (in declaration order) are:
    //   0: role         stride=1     range=2          divisor=1
    //   1: hours        stride=2     range=4          divisor=2
    //   2: exit         stride=8     range=24         divisor=8
    //   3: threshold    stride=192   range=4          divisor=192
    //   4: agg          stride=768   range=3          divisor=768
    //   5: weight_rank  stride=2304  range=4^k        divisor=2304
    // Only weight_rank's range varies with k (1..4). All other axes are
    // identical across k-blocks because every block size is a multiple of
    // PER_WEIGHT.
    struct AxisDescriptor {
        int         stride;   // grid-index step to move +/- 1 on this axis
        int         range;    // number of values on this axis (4^k for weight_rank)
        int         divisor;  // grid_id / divisor % range = position on axis
        const char* name;     // human-readable axis name
    };

    // Returns the 6 AxisDescriptors for a given subset size k (1..4).
    // weight_rank.range = weight_combos(k) = 4^k.
    static std::array<AxisDescriptor, 6> axis_descriptors(int k);

    // Reconstruct a StrategyGridEntry from a flat index.
    // Returns the entry. Precondition: 0 <= index < total_count().
    StrategyGridEntry entry_at(int64_t index) const;

    // Number of indicators k (1..4) of the strategy at a flat index.
    int subset_size_at(int64_t index) const;

    // One-step parameter neighbors of a flat index: (axis, neighbor index)
    // for each axis of axis_descriptors(k), stepping -1 and +1 where the
    // position on that axis allows. Positions are taken within the
    // k-indicator block, so a step never leaves the strategy's indicator
    // subset.
    std::vector<std::pair<int, int64_t>> neighbors(int64_t index) const;

    // Number of available indicators
    int n_indicators() const { return m_n_indicators; }

private:
    int     m_n_indicators = 0;
    int64_t m_subset_count = 0;
    int64_t m_total_count  = 0;

    // Precomputed cumulative subset counts: m_cum_subsets[k] = total subsets of size <= k
    // m_cum_subsets[0] = 0, m_cum_subsets[1] = C(N,1), etc.
    int64_t m_cum_subsets[5] = {};

    // Combinatorial helpers
    static int64_t binomial(int n, int k);

    // Unrank a k-combination from N elements.
    // Returns the combination as a vector of indices in increasing order.
    static std::vector<int> unrank_combination(int64_t rank, int n, int k);
};
