#include "backtest_engine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Composite signal computation (unified: optional time-gating)
//
// When time_gates is non-null, each series value is gated: if the bar's
// time_of_day_sec is before the gate threshold, the value is treated as 0.0
// (neutral / not yet available). When time_gates is null, values are read
// directly (no gating overhead).
// ---------------------------------------------------------------------------

float BacktestEngine::compute_composite_signal(const float* const* indicator_values, int num_series,
                                               int bar_offset, const StrategyConfig& config,
                                               const SignalTimeGate* time_gates,
                                               int32_t               bar_time_of_day_sec) const {
    // Read indicator value, applying time gate if configured
    auto read_val = [&](int series_idx) -> float {
        float val = indicator_values[series_idx][bar_offset];
        if (time_gates && !std::isnan(val)) {
            const SignalTimeGate& g = time_gates[series_idx];
            if (g.available_after_et_sec > 0 &&
                seconds_since_open(bar_time_of_day_sec, g.session_open_sec) <
                    seconds_since_open(g.available_after_et_sec, g.session_open_sec))
                return 0.0f;
        }
        return val;
    };

    switch (config.aggregation) {
        case StrategyConfig::WEIGHTED_SUM: {
            float weighted_sum  = 0.0f;
            float weight_total  = 0.0f;
            bool  filter_active = true;
            for (int i = 0; i < config.num_indicators; ++i) {
                const int idx = config.indicator_indices[i];
                if (idx < 0 || idx >= num_series) return 0.0f;
                const float val = read_val(idx);
                if (config.indicator_roles[i] == StrategyConfig::FILTER) {
                    if (std::isnan(val) || val == 0.0f) filter_active = false;
                    continue;
                }
                if (std::isnan(val)) continue;
                weighted_sum += val * config.weights[i];
                weight_total += config.weights[i];
            }
            if (!filter_active) return 0.0f;
            if (weight_total <= 0.0f) return 0.0f;
            return weighted_sum / weight_total;
        }

        case StrategyConfig::MAJORITY_VOTE: {
            int  votes_long    = 0;
            int  votes_short   = 0;
            bool filter_active = true;
            for (int i = 0; i < config.num_indicators; ++i) {
                const int idx = config.indicator_indices[i];
                if (idx < 0 || idx >= num_series) return 0.0f;
                const float val = read_val(idx);
                if (config.indicator_roles[i] == StrategyConfig::FILTER) {
                    if (std::isnan(val) || val == 0.0f) filter_active = false;
                    continue;
                }
                if (std::isnan(val)) continue;
                if (val > 0.0f)
                    ++votes_long;
                else if (val < 0.0f)
                    ++votes_short;
            }
            if (!filter_active) return 0.0f;
            if (votes_long > votes_short) return 1.0f;
            if (votes_short > votes_long) return -1.0f;
            return 0.0f;
        }

        case StrategyConfig::UNANIMOUS: {
            int  direction     = 0;
            bool all_agree     = true;
            bool filter_active = true;
            for (int i = 0; i < config.num_indicators; ++i) {
                const int idx = config.indicator_indices[i];
                if (idx < 0 || idx >= num_series) return 0.0f;
                const float val = read_val(idx);
                if (config.indicator_roles[i] == StrategyConfig::FILTER) {
                    if (std::isnan(val) || val == 0.0f) filter_active = false;
                    continue;
                }
                if (std::isnan(val)) continue;
                if (val == 0.0f) continue;

                const int this_dir = (val > 0.0f) ? 1 : -1;
                if (direction == 0) {
                    direction = this_dir;
                } else if (this_dir != direction) {
                    all_agree = false;
                    break;
                }
            }
            if (!filter_active) return 0.0f;
            return (all_agree && direction != 0) ? static_cast<float>(direction) : 0.0f;
        }

        default:
            return 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Public run() without sessions or time gates
// ---------------------------------------------------------------------------

BacktestResult BacktestEngine::run(const Bar* bars, int bar_start, int bar_end,
                                   const float* const* indicator_values, int num_indicator_series,
                                   const StrategyConfig& config, const CostModel& cost,
                                   int entry_lag, const EarlyTermination& early_term) const {
    return run_impl(bars, nullptr, bar_start, bar_end, indicator_values, num_indicator_series,
                    nullptr, config, cost, entry_lag, early_term);
}

// ---------------------------------------------------------------------------
// Public run() -- time-gated interface
// ---------------------------------------------------------------------------

BacktestResult BacktestEngine::run(const Bar* bars, const SessionInfo* sessions, int bar_start,
                                   int bar_end, const float* const* indicator_values,
                                   int num_indicator_series, const SignalTimeGate* time_gates,
                                   const StrategyConfig& config, const CostModel& cost,
                                   int entry_lag, const EarlyTermination& early_term) const {
    return run_impl(bars, sessions, bar_start, bar_end, indicator_values, num_indicator_series,
                    time_gates, config, cost, entry_lag, early_term);
}

// ---------------------------------------------------------------------------
// Core backtest loop implementation
// ---------------------------------------------------------------------------

BacktestResult BacktestEngine::run_impl(const Bar* bars, const SessionInfo* sessions, int bar_start,
                                        int bar_end, const float* const* indicator_values,
                                        int num_indicator_series, const SignalTimeGate* time_gates,
                                        const StrategyConfig& config, const CostModel& cost,
                                        int entry_lag, const EarlyTermination& early_term) const {
    assert(bar_start >= 0);
    assert(bar_end > bar_start);
    assert(entry_lag >= 1);
    assert(config.num_indicators > 0 && config.num_indicators <= 4);
    // CostModel structural invariants. BacktestEngine is stateless so
    // "once per construction" is not meaningful; cost is passed per-run.
    // The check is O(1) and cheap relative to the run itself.
    assert(cost.is_valid() && "CostModel invariants violated");

    const int num_bars = bar_end - bar_start;

    // Determine whether we need time-gated signal reads
    const bool use_gating = (sessions != nullptr && time_gates != nullptr);

    // Helper: compute composite signal at bar offset i, applying time gates
    // if sessions and time_gates are provided.
    auto get_composite = [&](int i) -> float {
        if (use_gating) {
            const int abs_bar = bar_start + i;
            int32_t   tod     = sessions[abs_bar].time_of_day_sec;
            return compute_composite_signal(indicator_values, num_indicator_series, i, config,
                                            time_gates, tod);
        } else {
            return compute_composite_signal(indicator_values, num_indicator_series, i, config);
        }
    };

    // Pre-allocate trade log.  Estimate: one trade per 20 bars is generous.
    m_trades.clear();
    m_trades.reserve(static_cast<size_t>(num_bars / 20 + 16));

    // -- Position state ---------------------------------------------------
    int   pos_direction      = 0;  // 0 = flat, +1 = long, -1 = short
    float pos_entry_price    = 0.0f;
    int   pos_entry_bar      = 0;     // absolute bar index of entry fill
    float pos_min_unrealized = 0.0f;  // worst M2M PnL for current position
    float pos_max_unrealized = 0.0f;  // best M2M PnL for current position
    float pos_best_price     = 0.0f;  // best price since entry (for trailing stop)

    // Pending entry state (signal fires on bar N, entry fills on bar N + entry_lag)
    int pending_direction  = 0;   // direction of pending entry
    int pending_signal_bar = -1;  // bar offset when signal fired (-1 = none)

    // Pending exit state (exit signal on bar N, fill at bar N+1 open)
    bool pending_exit = false;
    // Discriminator for Step 0 when the pending_exit path fires.
    // Session-close exits and signal-driven exits share the same fill-at-next-open
    // mechanics but must be labelled distinctly for PnL decomposition.
    FillType pending_exit_fill_type = FillType::NORMAL;

    // Exit ATR: Wilder ATR(EXIT_ATR_PERIOD) over the evaluated range, updated
    // at the top of each bar from the previous bar only, so stop and target
    // distances on bar i never use bar i's own range. ATR exits stay inactive
    // until the first EXIT_ATR_PERIOD bars have closed. A config that names
    // atr_series_index uses that series instead.
    float exit_atr     = 0.0f;
    float exit_atr_sum = 0.0f;
    int   exit_atr_n   = 0;

    // Early termination state
    bool      checked_early_term = false;
    const int early_term_bar =
        early_term.enabled
            ? static_cast<int>(static_cast<float>(num_bars) * early_term.check_at_pct)
            : num_bars;  // effectively disabled

    // -- Main loop --------------------------------------------------------
    int last_bar = num_bars - 1;  // last bar processed; earlier if terminated early
    for (int i = 0; i < num_bars; ++i) {
        const int abs_bar = bar_start + i;

        if (i > 0) {
            const Bar& pb = bars[abs_bar - 1];
            float      tr = pb.high - pb.low;
            if (i > 1) {
                const float pc = bars[abs_bar - 2].close;
                tr = std::max(tr, std::max(std::abs(pb.high - pc), std::abs(pb.low - pc)));
            }
            if (exit_atr_n < EXIT_ATR_PERIOD) {
                exit_atr_sum += tr;
                if (++exit_atr_n == EXIT_ATR_PERIOD)
                    exit_atr = exit_atr_sum / static_cast<float>(EXIT_ATR_PERIOD);
            } else {
                exit_atr = (exit_atr * static_cast<float>(EXIT_ATR_PERIOD - 1) + tr) /
                           static_cast<float>(EXIT_ATR_PERIOD);
            }
        }

        // -----------------------------------------------------------------
        // Step 0: Fill pending exit (signal was on previous bar; fill at
        //         this bar's open to avoid lookahead bias)
        // -----------------------------------------------------------------
        if (pending_exit && pos_direction != 0) {
            const float exit_price = bars[abs_bar].open;

            // Fold the fill bar's intrabar extrema into the running
            // pos_min/max_unrealized BEFORE recording the trade.
            // Otherwise the fill-bar low/high would be excluded from the
            // trade's worst/best M2M, understating MAE and overstating TRR
            // for signal-based exits that clear within the fill bar.
            {
                const float fill_worst_price =
                    (pos_direction > 0) ? bars[abs_bar].low : bars[abs_bar].high;
                const float fill_worst_unrealized = cost.price_to_pnl(
                    static_cast<float>(pos_direction) * (fill_worst_price - pos_entry_price));
                if (fill_worst_unrealized < pos_min_unrealized) {
                    pos_min_unrealized = fill_worst_unrealized;
                }
                const float fill_best_price =
                    (pos_direction > 0) ? bars[abs_bar].high : bars[abs_bar].low;
                const float fill_best_unrealized = cost.price_to_pnl(
                    static_cast<float>(pos_direction) * (fill_best_price - pos_entry_price));
                if (fill_best_unrealized > pos_max_unrealized) {
                    pos_max_unrealized = fill_best_unrealized;
                }
            }

            // Record completed trade
            Trade trade;
            trade.entry_bar   = pos_entry_bar;
            trade.exit_bar    = abs_bar;
            trade.entry_price = pos_entry_price;
            trade.exit_price  = exit_price;
            trade.direction   = pos_direction;

            trade.gross_pnl = cost.price_to_pnl(static_cast<float>(pos_direction) *
                                                (exit_price - pos_entry_price));
            trade.net_pnl   = trade.gross_pnl - cost.round_trip_cost();

            trade.min_unrealized = pos_min_unrealized;
            trade.max_unrealized = pos_max_unrealized;
            // Honour the session-close marker set on the prior bar.
            // Labelling is CPU-only; fill semantics match the GPU kernel
            // (backtest.cl).
            trade.fill_type = pending_exit_fill_type;

            m_trades.push_back(trade);

            // Reset position
            pos_direction          = 0;
            pos_entry_price        = 0.0f;
            pos_entry_bar          = 0;
            pos_min_unrealized     = 0.0f;
            pos_max_unrealized     = 0.0f;
            pos_best_price         = 0.0f;
            pending_exit           = false;
            pending_exit_fill_type = FillType::NORMAL;

            // If a pending entry was scheduled
            // to fire on exactly THIS fill bar, it would re-enter in the same
            // step the exit just cleared. Suppress only that exact alignment;
            // earlier carry-over pendings (e.g., from entry_lag=2 chains) are
            // preserved.
            if (pending_direction != 0 && pending_signal_bar == i - entry_lag) {
                pending_direction  = 0;
                pending_signal_bar = -1;
            }
        }

        // -----------------------------------------------------------------
        // Step 1: Fill pending entry
        // -----------------------------------------------------------------
        if (pending_direction != 0 && i == pending_signal_bar + entry_lag) {
            // Enter only while flat, and never between the close and the next open.
            const bool after_close =
                sessions != nullptr && (sessions[abs_bar].flags & SESSION_AFTER_CLOSE);
            if (pos_direction == 0 && !after_close) {
                pos_direction      = pending_direction;
                pos_entry_price    = bars[abs_bar].open;
                pos_entry_bar      = abs_bar;
                pos_min_unrealized = 0.0f;
                pos_max_unrealized = 0.0f;
                pos_best_price     = pos_entry_price;
            }
            pending_direction  = 0;
            pending_signal_bar = -1;
        }

        // -----------------------------------------------------------------
        // Step 2: If in position, update M2M and check exit conditions
        // -----------------------------------------------------------------
        if (pos_direction != 0) {
            // Mark-to-market using intrabar extreme for worst-case drawdown
            const float worst_price = (pos_direction > 0) ? bars[abs_bar].low : bars[abs_bar].high;
            const float worst_unrealized = cost.price_to_pnl(static_cast<float>(pos_direction) *
                                                             (worst_price - pos_entry_price));
            if (worst_unrealized < pos_min_unrealized) {
                pos_min_unrealized = worst_unrealized;
            }

            // Mark-to-market using intrabar extreme for best-case peak
            const float best_price = (pos_direction > 0) ? bars[abs_bar].high : bars[abs_bar].low;
            const float best_unrealized = cost.price_to_pnl(static_cast<float>(pos_direction) *
                                                            (best_price - pos_entry_price));
            if (best_unrealized > pos_max_unrealized) {
                pos_max_unrealized = best_unrealized;
            }

            // Check exit conditions.
            //
            // Two exit mechanisms:
            //   1. Signal-based (SIGNAL_REVERSAL, OPPOSING_SIGNAL, BAR_TIMEOUT,
            //      session close): set pending_exit, fill at next bar's open.
            //      These are market orders placed after the signal fires.
            //   2. Level-triggered (TRAILING_STOP, MAX_LOSS_ATR, TAKE_PROFIT_ATR,
            //      TP_SL_ATR): fill immediately at the trigger level on the
            //      detection bar. These model resting stop/limit orders that
            //      fill when price reaches the level.
            //
            // Backtesting rules (see docs/METHODOLOGY.md):
            //   "Stop orders: fill at the stop level or worse"
            //   "Limit orders: fill only if price trades through the limit level"

            bool  should_exit      = false;
            float level_exit_price = 0.0f;  // >0 = immediate fill at this level

            // ATR for level exits: the caller's series if named, else the
            // engine's exit ATR (0 until warm, which disables level exits).
            // A caller-supplied series is read at the previous bar, so a
            // stop on bar i never uses a value that includes bar i.
            float atr_val = exit_atr;
            if (config.atr_series_index >= 0 && config.atr_series_index < num_indicator_series) {
                atr_val = (i > 0) ? indicator_values[config.atr_series_index][i - 1] : 0.0f;
                if (std::isnan(atr_val)) atr_val = 0.0f;
            }

            // A stop that the bar opens through fills at the open, not at the
            // stop level. Targets fill at their level.
            const float bar_open        = bars[abs_bar].open;
            auto        long_stop_fill  = [bar_open](float lvl) { return std::min(bar_open, lvl); };
            auto        short_stop_fill = [bar_open](float lvl) { return std::max(bar_open, lvl); };

            // Level-triggered exits (TRAILING_STOP,
            // MAX_LOSS_ATR, TAKE_PROFIT_ATR, TP_SL_ATR) must not fire on the
            // entry bar itself -- that would produce a same-bar round trip
            // with the entry-bar intrabar extreme as fill price, violating
            // the no same-bar round trip rule. Signal-based exits route through
            // pending_exit and fill at next-bar open, so they are harmless
            // on the entry bar.
            const bool post_entry = (abs_bar > pos_entry_bar);

            switch (config.exit_mode) {
                case StrategyConfig::SIGNAL_REVERSAL: {
                    const float sig = get_composite(i);
                    if ((pos_direction > 0 && sig <= 0.0f) || (pos_direction < 0 && sig >= 0.0f)) {
                        should_exit = true;
                        // Signal-based: level_exit_price stays 0 -> pending_exit
                    }
                    break;
                }

                case StrategyConfig::BAR_TIMEOUT: {
                    const int bars_held = abs_bar - pos_entry_bar;
                    if (bars_held >= config.exit_timeout_bars) {
                        should_exit = true;
                    }
                    break;
                }

                case StrategyConfig::TRAILING_STOP: {
                    if (post_entry && atr_val > 0.0f) {
                        const float stop_dist = config.trailing_atr_mult * atr_val;
                        if (pos_direction > 0) {
                            const float stop_level = pos_best_price - stop_dist;
                            if (bars[abs_bar].low <= stop_level) {
                                should_exit      = true;
                                level_exit_price = long_stop_fill(stop_level);
                            }
                        } else {
                            const float stop_level = pos_best_price + stop_dist;
                            if (bars[abs_bar].high >= stop_level) {
                                should_exit      = true;
                                level_exit_price = short_stop_fill(stop_level);
                            }
                        }
                    }
                    break;
                }

                case StrategyConfig::OPPOSING_SIGNAL: {
                    const float sig = get_composite(i);
                    if ((pos_direction > 0 && sig < -config.entry_threshold) ||
                        (pos_direction < 0 && sig > config.entry_threshold)) {
                        should_exit = true;
                    }
                    break;
                }

                case StrategyConfig::MAX_LOSS_ATR: {
                    if (post_entry && atr_val > 0.0f) {
                        const float stop_dist = config.max_loss_atr_mult * atr_val;
                        if (pos_direction > 0) {
                            const float stop_level = pos_entry_price - stop_dist;
                            if (bars[abs_bar].low <= stop_level) {
                                should_exit      = true;
                                level_exit_price = long_stop_fill(stop_level);
                            }
                        } else {
                            const float stop_level = pos_entry_price + stop_dist;
                            if (bars[abs_bar].high >= stop_level) {
                                should_exit      = true;
                                level_exit_price = short_stop_fill(stop_level);
                            }
                        }
                    }
                    break;
                }

                case StrategyConfig::TAKE_PROFIT_ATR: {
                    if (post_entry && atr_val > 0.0f) {
                        const float tp_dist = config.tp_atr_mult * atr_val;
                        if (pos_direction > 0) {
                            const float tp_level = pos_entry_price + tp_dist;
                            if (bars[abs_bar].high >= tp_level) {
                                should_exit      = true;
                                level_exit_price = tp_level;
                            }
                        } else {
                            const float tp_level = pos_entry_price - tp_dist;
                            if (bars[abs_bar].low <= tp_level) {
                                should_exit      = true;
                                level_exit_price = tp_level;
                            }
                        }
                    }
                    break;
                }

                case StrategyConfig::TP_SL_ATR: {
                    if (post_entry && atr_val > 0.0f) {
                        const float tp_dist  = config.tp_atr_mult * atr_val;
                        const float sl_dist  = config.sl_atr_mult * atr_val;
                        bool        sl_hit   = false;
                        bool        tp_hit   = false;
                        float       sl_level = 0.0f;
                        float       tp_level = 0.0f;
                        if (pos_direction > 0) {
                            sl_level = pos_entry_price - sl_dist;
                            tp_level = pos_entry_price + tp_dist;
                            sl_hit   = (bars[abs_bar].low <= sl_level);
                            tp_hit   = (bars[abs_bar].high >= tp_level);
                        } else {
                            sl_level = pos_entry_price + sl_dist;
                            tp_level = pos_entry_price - tp_dist;
                            sl_hit   = (bars[abs_bar].high >= sl_level);
                            tp_hit   = (bars[abs_bar].low <= tp_level);
                        }
                        // SL takes precedence when both trigger on the same bar
                        // (conservative: cannot determine intra-bar ordering).
                        if (sl_hit) {
                            should_exit      = true;
                            level_exit_price = (pos_direction > 0) ? long_stop_fill(sl_level)
                                                                   : short_stop_fill(sl_level);
                        } else if (tp_hit) {
                            should_exit      = true;
                            level_exit_price = tp_level;
                        }
                    }
                    break;
                }

            }  // switch exit_mode

            // Flat at the daily close: on the close bar the position exits at
            // that bar's close (a market-on-close order) instead of carrying
            // through the break. A signal exit raised on this bar would fill
            // at the next session's open, so it becomes the close fill too;
            // only a stop or target that already filled intrabar takes
            // precedence. Matches the GPU kernel.
            bool session_close_fired = false;
            if (sessions != nullptr && (sessions[abs_bar].flags & SESSION_CLOSE_BAR) &&
                level_exit_price <= 0.0f) {
                should_exit         = true;
                session_close_fired = true;
                level_exit_price    = bars[abs_bar].close;
            }

            if (should_exit) {
                if (level_exit_price > 0.0f) {
                    // Level-triggered: fill immediately at the resting order level.
                    Trade trade;
                    trade.entry_bar      = pos_entry_bar;
                    trade.exit_bar       = abs_bar;
                    trade.entry_price    = pos_entry_price;
                    trade.exit_price     = level_exit_price;
                    trade.direction      = pos_direction;
                    trade.gross_pnl      = cost.price_to_pnl(static_cast<float>(pos_direction) *
                                                             (level_exit_price - pos_entry_price));
                    trade.net_pnl        = trade.gross_pnl - cost.round_trip_cost();
                    trade.min_unrealized = pos_min_unrealized;
                    trade.max_unrealized = pos_max_unrealized;
                    trade.fill_type =
                        session_close_fired ? FillType::SESSION_CLOSE : FillType::LEVEL_EXIT;
                    m_trades.push_back(trade);

                    pos_direction      = 0;
                    pos_entry_price    = 0.0f;
                    pos_entry_bar      = 0;
                    pos_min_unrealized = 0.0f;
                    pos_max_unrealized = 0.0f;
                    pos_best_price     = 0.0f;
                } else {
                    // Signal-based: defer to next bar's open
                    if (i + 1 < num_bars) {
                        pending_exit           = true;
                        pending_exit_fill_type = FillType::NORMAL;
                    }
                }
            }

            // Advance the trailing-stop reference with this bar's extreme
            // only after this bar's exits were checked against the previous
            // reference: the stop on bar i never uses bar i's own high/low.
            if (pos_direction != 0 && !pending_exit) {
                if (pos_direction > 0) {
                    if (bars[abs_bar].high > pos_best_price) pos_best_price = bars[abs_bar].high;
                } else {
                    if (bars[abs_bar].low < pos_best_price) pos_best_price = bars[abs_bar].low;
                }
            }
        }

        // -----------------------------------------------------------------
        // Step 3: If flat and no pending entry, evaluate signal
        // -----------------------------------------------------------------
        if (pos_direction == 0 && pending_direction == 0 && !pending_exit) {
            // No new entries during a halt or between the close and the next open.
            if (sessions != nullptr &&
                (sessions[abs_bar].halt_flag || (sessions[abs_bar].flags & SESSION_AFTER_CLOSE))) {
                // Skip entry evaluation
            } else {
                // Trading hours filter: suppress new entries outside the configured window
                bool in_trading_hours = true;
                if (config.trading_hours != StrategyConfig::ETH && sessions != nullptr) {
                    const uint8_t flags = sessions[abs_bar].flags;
                    switch (config.trading_hours) {
                        case StrategyConfig::RTH:
                            in_trading_hours = (flags & SESSION_IN_CORE) != 0;
                            break;
                        case StrategyConfig::MORNING:
                            in_trading_hours = (flags & SESSION_IN_MORNING) != 0;
                            break;
                        case StrategyConfig::AFTERNOON:
                            in_trading_hours = (flags & SESSION_IN_AFTERNOON) != 0;
                            break;
                        default:
                            break;
                    }
                }

                if (in_trading_hours && i + entry_lag < num_bars) {
                    const float sig = get_composite(i);

                    if (sig > config.entry_threshold) {
                        pending_direction  = 1;  // long
                        pending_signal_bar = i;
                    } else if (sig < -config.entry_threshold) {
                        pending_direction  = -1;  // short
                        pending_signal_bar = i;
                    }
                }
            }  // else (not halted)
        }

        // -----------------------------------------------------------------
        // Step 4: Early termination check (screen-stage optimization)
        //
        // After reaching the checkpoint percentage of bars, check whether
        // the strategy has any chance of passing screen floors. If running
        // PnL is negative or trade count is too low, abort early. The
        // resulting poor metrics will fail the sanity floors.
        // -----------------------------------------------------------------
        if (early_term.enabled && !checked_early_term && i >= early_term_bar) {
            checked_early_term = true;
            // Accumulate in double to avoid float precision loss
            // when summing many small per-trade net_pnl values at the early-
            // termination checkpoint. The cutoff is a sign test so the
            // precision of the sum directly affects termination decisions.
            double running_pnl = 0.0;
            for (const auto& t : m_trades) running_pnl += static_cast<double>(t.net_pnl);
            if (running_pnl < 0.0 ||
                static_cast<int>(m_trades.size()) < early_term.min_trades_at_check) {
                last_bar = i;
                break;  // abort -- won't pass screen floors
            }
        }
    }

    // -- Force-close any open position at the last processed bar -----------
    // This ensures positions are not silently discarded at OOS window
    // boundaries, consistent with the OpenCL backtest.cl kernel. After an
    // early termination that is the checkpoint bar, so an aborted strategy
    // never books the price move of bars it did not process.
    if (pos_direction != 0 && num_bars > 0) {
        const int   last_abs_bar = bar_start + last_bar;
        const float exit_price   = bars[last_abs_bar].close;

        Trade trade;
        trade.entry_bar   = pos_entry_bar;
        trade.exit_bar    = last_abs_bar;
        trade.entry_price = pos_entry_price;
        trade.exit_price  = exit_price;
        trade.direction   = pos_direction;

        trade.gross_pnl =
            cost.price_to_pnl(static_cast<float>(pos_direction) * (exit_price - pos_entry_price));
        trade.net_pnl = trade.gross_pnl - cost.round_trip_cost();

        trade.min_unrealized = pos_min_unrealized;
        trade.max_unrealized = pos_max_unrealized;
        trade.fill_type      = FillType::FORCED_CLOSE;

        m_trades.push_back(trade);
    }

    // -- Compute metrics from the trade log -------------------------------
    // Pass sessions for DST-correct daily PnL bucketing (Sharpe ratio).
    // sessions may be nullptr when called via the non-gated run() overload;
    // compute_metrics falls back to the approximate UTC shift in that case.
    BacktestResult result =
        compute_metrics(m_trades, bars, bar_start, bar_start + last_bar + 1, cost, sessions);

    return result;
}
