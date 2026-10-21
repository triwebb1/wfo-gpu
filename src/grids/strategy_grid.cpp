#include "strategy_grid.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cassert>

// Static constexpr member definitions (needed pre-C++17 inline for some compilers)
constexpr float StrategyGrid::WEIGHT_VALUES[];
constexpr float StrategyGrid::THRESHOLD_VALUES[];

bool StrategyGridEntry::is_redundant() const {
    // An entry is redundant when another kept entry trades identically. Each
    // equivalence class keeps exactly one canonical member.

    // A single filter with no signal never trades.
    if (num_indicators == 1 && indicator_roles[0] == StrategyConfig::FILTER) return true;

    // With one indicator, UNANIMOUS and MAJORITY_VOTE both reduce to the
    // signal's sign; keep MAJORITY_VOTE.
    if (aggregation == StrategyConfig::UNANIMOUS && num_indicators == 1) return true;

    constexpr float FULL_WEIGHT = 1.0f;

    // A filter's weight is never read; keep weight 1.0.
    for (int i = 0; i < num_indicators; ++i) {
        if (indicator_roles[i] == StrategyConfig::FILTER && weights[i] != FULL_WEIGHT) return true;
    }

    if (aggregation != StrategyConfig::WEIGHTED_SUM) {
        // Vote modes ignore weights and output -1, 0 or +1, which clears
        // every entry threshold: keep weight 1.0 and the first threshold.
        if (entry_threshold != StrategyGrid::THRESHOLD_VALUES[0]) return true;
        for (int i = 0; i < num_indicators; ++i) {
            if (weights[i] != FULL_WEIGHT) return true;
        }
        return false;
    }

    // WEIGHTED_SUM divides by the total weight, so proportional weight
    // vectors give the same composite. In quarter units (weights 1..4) each
    // proportional class is a primitive vector p times a scale; keep the
    // largest member, p * floor(4 / max(p)).
    int q[4] = {0, 0, 0, 0};
    int g = 0, max_q = 0;
    for (int i = 0; i < num_indicators; ++i) {
        if (indicator_roles[i] == StrategyConfig::FILTER) continue;
        q[i]  = static_cast<int>(std::lround(weights[i] * 4.0f));
        g     = std::gcd(g, q[i]);
        max_q = std::max(max_q, q[i]);
    }
    if (g == 0) return false;
    const int scale = 4 / (max_q / g);  // largest scale that stays in the grid
    return g != scale;
}

StrategyConfig StrategyGridEntry::to_strategy_config() const {
    StrategyConfig sc;
    sc.num_indicators = num_indicators;
    for (int i = 0; i < num_indicators; ++i) {
        sc.indicator_indices[i] = indicator_ids[i];
        sc.weights[i]           = weights[i];
        sc.indicator_roles[i]   = indicator_roles[i];
    }
    sc.aggregation     = aggregation;
    sc.entry_threshold = entry_threshold;

    // Lookup table: ExitModePreset -> {exit_mode, timeout, trailing, max_loss, tp, sl}
    struct ExitPresetRow {
        StrategyConfig::ExitMode mode;
        int                      timeout;
        float                    trailing, max_loss, tp, sl;
    };
    static constexpr ExitPresetRow EXIT_PRESETS[] = {
        // preset                          mode                          timeout  trail  mloss  tp    sl
        /*  0 SIGNAL_REVERSAL    */ {StrategyConfig::SIGNAL_REVERSAL, 0, 0, 0, 0, 0},
        /*  1 BAR_TIMEOUT_10     */ {StrategyConfig::BAR_TIMEOUT, 10, 0, 0, 0, 0},
        /*  2 BAR_TIMEOUT_20     */ {StrategyConfig::BAR_TIMEOUT, 20, 0, 0, 0, 0},
        /*  3 BAR_TIMEOUT_50     */ {StrategyConfig::BAR_TIMEOUT, 50, 0, 0, 0, 0},
        /*  4 BAR_TIMEOUT_100    */ {StrategyConfig::BAR_TIMEOUT, 100, 0, 0, 0, 0},
        /*  5 TRAILING_STOP_1_0  */ {StrategyConfig::TRAILING_STOP, 0, 1.0f, 0, 0, 0},
        /*  6 TRAILING_STOP_1_5  */ {StrategyConfig::TRAILING_STOP, 0, 1.5f, 0, 0, 0},
        /*  7 TRAILING_STOP_2_0  */ {StrategyConfig::TRAILING_STOP, 0, 2.0f, 0, 0, 0},
        /*  8 TRAILING_STOP_2_5  */ {StrategyConfig::TRAILING_STOP, 0, 2.5f, 0, 0, 0},
        /*  9 OPPOSING_SIGNAL    */ {StrategyConfig::OPPOSING_SIGNAL, 0, 0, 0, 0, 0},
        /* 10 MAX_LOSS_ATR_2_0   */ {StrategyConfig::MAX_LOSS_ATR, 0, 0, 2.0f, 0, 0},
        /* 11 MAX_LOSS_ATR_3_0   */ {StrategyConfig::MAX_LOSS_ATR, 0, 0, 3.0f, 0, 0},
        /* 12 MAX_LOSS_ATR_4_0   */ {StrategyConfig::MAX_LOSS_ATR, 0, 0, 4.0f, 0, 0},
        /* 13 MAX_LOSS_ATR_5_0   */ {StrategyConfig::MAX_LOSS_ATR, 0, 0, 5.0f, 0, 0},
        /* 14 TAKE_PROFIT_1_0    */ {StrategyConfig::TAKE_PROFIT_ATR, 0, 0, 0, 1.0f, 0},
        /* 15 TAKE_PROFIT_1_5    */ {StrategyConfig::TAKE_PROFIT_ATR, 0, 0, 0, 1.5f, 0},
        /* 16 TAKE_PROFIT_2_0    */ {StrategyConfig::TAKE_PROFIT_ATR, 0, 0, 0, 2.0f, 0},
        /* 17 TAKE_PROFIT_3_0    */ {StrategyConfig::TAKE_PROFIT_ATR, 0, 0, 0, 3.0f, 0},
        /* 18 TP_SL_1_0_1_0      */ {StrategyConfig::TP_SL_ATR, 0, 0, 0, 1.0f, 1.0f},
        /* 19 TP_SL_1_5_1_0      */ {StrategyConfig::TP_SL_ATR, 0, 0, 0, 1.5f, 1.0f},
        /* 20 TP_SL_2_0_1_0      */ {StrategyConfig::TP_SL_ATR, 0, 0, 0, 2.0f, 1.0f},
        /* 21 TP_SL_3_0_1_0      */ {StrategyConfig::TP_SL_ATR, 0, 0, 0, 3.0f, 1.0f},
        /* 22 TP_SL_2_0_1_5      */ {StrategyConfig::TP_SL_ATR, 0, 0, 0, 2.0f, 1.5f},
        /* 23 TP_SL_2_0_2_0      */ {StrategyConfig::TP_SL_ATR, 0, 0, 0, 2.0f, 2.0f},
    };
    const auto& row      = EXIT_PRESETS[static_cast<uint8_t>(exit_mode)];
    sc.exit_mode         = row.mode;
    sc.exit_timeout_bars = row.timeout;
    sc.trailing_atr_mult = row.trailing;
    sc.max_loss_atr_mult = row.max_loss;
    sc.tp_atr_mult       = row.tp;
    sc.sl_atr_mult       = row.sl;
    sc.trading_hours     = trading_hours;
    return sc;
}

int64_t StrategyGrid::binomial(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;  // Optimization: C(n,k) = C(n,n-k)

    int64_t result = 1;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

std::vector<int> StrategyGrid::unrank_combination(int64_t rank, int n, int k) {
    std::vector<int> result;
    result.reserve(k);

    int start = 0;
    for (int i = 0; i < k; ++i) {
        // Find the smallest c >= start such that C(n-1-c, k-1-i) <= rank
        for (int c = start; c < n; ++c) {
            int64_t count = binomial(n - 1 - c, k - 1 - i);
            if (rank < count) {
                result.push_back(c);
                start = c + 1;
                break;
            }
            rank -= count;
        }
    }
    return result;
}

int StrategyGrid::weight_combos(int k) {
    // 4 weight values per indicator, k indicators
    int result = 1;
    for (int i = 0; i < k; ++i) result *= 4;
    return result;
}

// ---------------------------------------------------------------------------
// axis_descriptors
//
// Build the 6 neighbor-scoring axes derived from the same constants that
// drive entry_at() decoding. Strides are computed bottom-up from the
// innermost-first layout:
//   role(1) > hours(2) > exit(8) > thresh(192) > agg(768) > weight_rank(2304)
// PER_WEIGHT (= 2304) is the static_assert-guarded product.
// Only weight_rank.range varies with k (= 4^k = weight_combos(k)).
// ---------------------------------------------------------------------------
std::array<StrategyGrid::AxisDescriptor, 6> StrategyGrid::axis_descriptors(int k) {
    constexpr int role_stride   = 1;
    constexpr int hours_stride  = role_stride * ROLE_PATTERNS;           // 2
    constexpr int exit_stride   = hours_stride * TRADING_HOURS_OPTIONS;  // 8
    constexpr int thresh_stride = exit_stride * EXIT_MODES;              // 192
    constexpr int agg_stride    = thresh_stride * ENTRY_THRESHOLDS;      // 768
    constexpr int wt_stride     = agg_stride * AGG_MODES;                // 2304 (= PER_WEIGHT)
    static_assert(wt_stride == PER_WEIGHT, "axis stride layout drift");

    return {{
        {role_stride, ROLE_PATTERNS, role_stride, "role"},
        {hours_stride, TRADING_HOURS_OPTIONS, hours_stride, "hours"},
        {exit_stride, EXIT_MODES, exit_stride, "exit"},
        {thresh_stride, ENTRY_THRESHOLDS, thresh_stride, "threshold"},
        {agg_stride, AGG_MODES, agg_stride, "agg"},
        {wt_stride, weight_combos(k), wt_stride, "weight_rank"},
    }};
}

StrategyGrid::StrategyGrid(int n_indicators) : m_n_indicators(n_indicators) {
    // Compute cumulative subset counts
    m_cum_subsets[0] = 0;
    for (int k = 1; k <= 4; ++k) {
        m_cum_subsets[k] = m_cum_subsets[k - 1] + binomial(n_indicators, k);
    }
    m_subset_count = m_cum_subsets[4];

    // Total = sum over k=1..4 of:
    //   C(N,k) * weight_combos(k) * AGG_MODES * ENTRY_THRESHOLDS * EXIT_MODES * TRADING_HOURS_OPTIONS * ROLE_PATTERNS
    m_total_count = 0;
    for (int k = 1; k <= 4; ++k) {
        m_total_count += binomial(n_indicators, k) * weight_combos(k) * AGG_MODES *
                         ENTRY_THRESHOLDS * EXIT_MODES * TRADING_HOURS_OPTIONS * ROLE_PATTERNS;
    }
}

StrategyGridEntry StrategyGrid::entry_at(int64_t index) const {
    assert(index >= 0 && index < m_total_count);

    StrategyGridEntry e;

    // Determine which subset size k this index falls into
    int64_t remaining = index;
    int     k         = 0;
    for (k = 1; k <= 4; ++k) {
        int64_t block_size = binomial(m_n_indicators, k) * weight_combos(k) * AGG_MODES *
                             ENTRY_THRESHOLDS * EXIT_MODES * TRADING_HOURS_OPTIONS * ROLE_PATTERNS;
        if (remaining < block_size) break;
        remaining -= block_size;
    }

    // Safety clamp: if the loop exits without breaking (should not happen with
    // a valid index due to the assert above), clamp k to the valid range.
    if (k > 4) k = 4;

    e.num_indicators = k;

    // Decompose remaining into: subset_rank, weight_rank, agg, threshold, exit, trading_hours, role_pattern
    int64_t wt_combos  = weight_combos(k);
    int64_t per_subset = wt_combos * AGG_MODES * ENTRY_THRESHOLDS * EXIT_MODES *
                         TRADING_HOURS_OPTIONS * ROLE_PATTERNS;
    int64_t subset_rank = remaining / per_subset;
    remaining %= per_subset;

    // Unrank the k-subset
    auto subset = unrank_combination(subset_rank, m_n_indicators, k);
    for (int i = 0; i < k; ++i) {
        e.indicator_ids[i] = subset[i];
    }

    // Decompose remaining into weight_rank, agg, threshold, exit, trading_hours, role_pattern
    int64_t per_weight = static_cast<int64_t>(AGG_MODES) * ENTRY_THRESHOLDS * EXIT_MODES *
                         TRADING_HOURS_OPTIONS * ROLE_PATTERNS;
    int64_t weight_rank = remaining / per_weight;
    remaining %= per_weight;

    // Decode weight indices (base-4 number with k digits)
    for (int i = k - 1; i >= 0; --i) {
        int wi       = static_cast<int>(weight_rank % 4);
        e.weights[i] = WEIGHT_VALUES[wi];
        weight_rank /= 4;
    }

    // Aggregation mode
    int64_t per_agg =
        static_cast<int64_t>(ENTRY_THRESHOLDS) * EXIT_MODES * TRADING_HOURS_OPTIONS * ROLE_PATTERNS;
    int agg_idx = static_cast<int>(remaining / per_agg);
    remaining %= per_agg;

    static constexpr StrategyConfig::AggMode AGG_TABLE[] = {
        StrategyConfig::WEIGHTED_SUM, StrategyConfig::MAJORITY_VOTE, StrategyConfig::UNANIMOUS};
    e.aggregation = AGG_TABLE[agg_idx];

    // Entry threshold
    int64_t per_thresh = static_cast<int64_t>(EXIT_MODES) * TRADING_HOURS_OPTIONS * ROLE_PATTERNS;
    int     thresh_idx = static_cast<int>(remaining / per_thresh);
    remaining %= per_thresh;
    e.entry_threshold = THRESHOLD_VALUES[thresh_idx];

    // Exit mode
    static constexpr ExitModePreset EXIT_TABLE[] = {
        ExitModePreset::SIGNAL_REVERSAL,     ExitModePreset::BAR_TIMEOUT_10,
        ExitModePreset::BAR_TIMEOUT_20,      ExitModePreset::BAR_TIMEOUT_50,
        ExitModePreset::BAR_TIMEOUT_100,     ExitModePreset::TRAILING_STOP_1_0,
        ExitModePreset::TRAILING_STOP_1_5,   ExitModePreset::TRAILING_STOP_2_0,
        ExitModePreset::TRAILING_STOP_2_5,   ExitModePreset::OPPOSING_SIGNAL,
        ExitModePreset::MAX_LOSS_ATR_2_0,    ExitModePreset::MAX_LOSS_ATR_3_0,
        ExitModePreset::MAX_LOSS_ATR_4_0,    ExitModePreset::MAX_LOSS_ATR_5_0,
        ExitModePreset::TAKE_PROFIT_ATR_1_0, ExitModePreset::TAKE_PROFIT_ATR_1_5,
        ExitModePreset::TAKE_PROFIT_ATR_2_0, ExitModePreset::TAKE_PROFIT_ATR_3_0,
        ExitModePreset::TP_SL_1_0_1_0,       ExitModePreset::TP_SL_1_5_1_0,
        ExitModePreset::TP_SL_2_0_1_0,       ExitModePreset::TP_SL_3_0_1_0,
        ExitModePreset::TP_SL_2_0_1_5,       ExitModePreset::TP_SL_2_0_2_0};
    int64_t per_exit = static_cast<int64_t>(TRADING_HOURS_OPTIONS) * ROLE_PATTERNS;
    int     exit_idx = static_cast<int>(remaining / per_exit);
    remaining %= per_exit;
    e.exit_mode = EXIT_TABLE[exit_idx];

    // Trading hours
    int64_t per_hours = static_cast<int64_t>(ROLE_PATTERNS);
    int     hours_idx = static_cast<int>(remaining / per_hours);
    remaining %= per_hours;
    static constexpr StrategyConfig::TradingHours HOURS_TABLE[] = {
        StrategyConfig::ETH, StrategyConfig::RTH, StrategyConfig::MORNING,
        StrategyConfig::AFTERNOON};
    e.trading_hours = HOURS_TABLE[hours_idx];

    // Role pattern
    int role_idx = static_cast<int>(remaining);
    if (role_idx == 0) {
        // ALL_SIGNAL: all slots are SIGNAL (default)
        for (int i = 0; i < 4; ++i) e.indicator_roles[i] = StrategyConfig::SIGNAL;
    } else {
        // LAST_FILTER: last active indicator is FILTER, rest are SIGNAL
        for (int i = 0; i < k - 1; ++i) e.indicator_roles[i] = StrategyConfig::SIGNAL;
        e.indicator_roles[k - 1] = StrategyConfig::FILTER;
    }

    return e;
}

// ---------------------------------------------------------------------------
// subset_size_at / neighbors
// ---------------------------------------------------------------------------
int StrategyGrid::subset_size_at(int64_t index) const {
    int64_t start = 0;
    for (int k = 1; k <= 4; ++k) {
        start += binomial(m_n_indicators, k) * weight_combos(k) * PER_WEIGHT;
        if (index < start) return k;
    }
    return 4;
}

std::vector<std::pair<int, int64_t>> StrategyGrid::neighbors(int64_t index) const {
    std::vector<std::pair<int, int64_t>> out;
    if (index < 0 || index >= m_total_count) return out;

    int64_t block_start = 0;
    int     k           = 1;
    for (; k <= 4; ++k) {
        const int64_t size = binomial(m_n_indicators, k) * weight_combos(k) * PER_WEIGHT;
        if (index < block_start + size) break;
        block_start += size;
    }
    const int64_t local = index - block_start;
    const auto    axes  = axis_descriptors(k);
    for (int axis = 0; axis < static_cast<int>(axes.size()); ++axis) {
        const auto&   ad  = axes[axis];
        const int64_t pos = (local / ad.divisor) % ad.range;
        if (pos > 0) out.emplace_back(axis, index - ad.stride);
        if (pos < ad.range - 1) out.emplace_back(axis, index + ad.stride);
    }
    return out;
}
