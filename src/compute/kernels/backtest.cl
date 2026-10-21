// =============================================================================
// OpenCL Backtest Evaluation Kernel
//
// One work-item per strategy configuration.  All work-items in a sub-group
// read the same indicator values at each bar (broadcast pattern -- maximum
// coalescing).  Strategy parameters (weights, thresholds, exit mode) are
// loaded from a parameter buffer into private memory.
//
// Execution strategy: grouped by indicator set.  All strategies sharing the
// same indicator set run in one clEnqueueNDRangeKernel call.
//
// Output: per-strategy metrics written to a result buffer.
// =============================================================================

// No fused multiply-add contraction: results must match the CPU engine.
#pragma OPENCL FP_CONTRACT OFF

// ---------------------------------------------------------------------------
// Strategy parameter struct (flat, passed via global buffer)
//
// IMPORTANT: Field ordering and sizes must exactly match the host-side
// StrategyParamsGPU struct in cl_backtest.cpp.  Both are plain C structs
// with int/float fields -- no enum, no uint8_t, no padding surprises.
// ---------------------------------------------------------------------------
typedef struct {
    int   indicator_indices[4];
    int   num_indicators;
    float weights[4];
    int   indicator_roles[4]; // 0=SIGNAL, 1=FILTER
    int   aggregation;       // 0=WEIGHTED_SUM, 1=MAJORITY_VOTE, 2=UNANIMOUS
    float entry_threshold;
    int   exit_mode;         // 0=SIGNAL_REVERSAL, 1=BAR_TIMEOUT, 2=TRAILING_STOP, 3=OPPOSING_SIGNAL, 4=MAX_LOSS_ATR, 5=TAKE_PROFIT_ATR, 6=TP_SL_ATR
    int   exit_timeout_bars;
    float trailing_atr_mult;
    float max_loss_atr_mult;
    float tp_atr_mult;
    float sl_atr_mult;
    int   atr_series_index;
    int   trading_hours;     // 0=ETH, 1=RTH, 2=MORNING, 3=AFTERNOON
} StrategyParams;

// ---------------------------------------------------------------------------
// Result struct (one per work-item, written to global result buffer)
//
// Raw accumulators only -- the host fills in derived metrics (Sharpe, Calmar,
// etc.) from these.  Must match BacktestResultGPU in cl_backtest.cpp.
// ---------------------------------------------------------------------------
typedef struct {
    float total_pnl;
    int   total_trades;
    int   win_count;
    float gross_wins;
    float gross_losses;
    float max_drawdown;
    float max_live_drawdown;
    float peak_equity;
    int   total_bars_in_position;  // for avg hold time: total_bars_in_position / total_trades
    float sum_min_unreal;          // sum of per-trade worst unrealized (negative); host negates to get avg_trade_dd
    float sum_max_unreal;          // sum of per-trade peak unrealized (positive); host divides for avg_trade_peak
} BacktestResultGPU;

// ---------------------------------------------------------------------------
// Per-bar session flags (SessionFlag in session_model.h)
// ---------------------------------------------------------------------------
#define SESSION_IN_CORE      1
#define SESSION_IN_MORNING   2
#define SESSION_IN_AFTERNOON 4
#define SESSION_CLOSE_BAR    8
#define SESSION_HALTED       16
#define SESSION_AFTER_CLOSE  32

// Period of the kernel-computed exit ATR (EXIT_ATR_PERIOD in backtest_engine.h)
#define EXIT_ATR_PERIOD 14

// ---------------------------------------------------------------------------
// Trading hours filter
//
// Returns 1 if entries are allowed on a bar with the given session flags.
// Matches the CPU-side BacktestEngine::run_impl() logic exactly.
// ---------------------------------------------------------------------------
int is_entry_allowed(int trading_hours, int flags) {
    if (trading_hours == 1) return (flags & SESSION_IN_CORE) ? 1 : 0;       // RTH
    if (trading_hours == 2) return (flags & SESSION_IN_MORNING) ? 1 : 0;    // MORNING
    if (trading_hours == 3) return (flags & SESSION_IN_AFTERNOON) ? 1 : 0;  // AFTERNOON
    return 1;                                                               // ETH
}

// ---------------------------------------------------------------------------
// Composite signal computation
//
// Uses flat indicator layout: indicator_data[indicator_index * stride + bar]
// ---------------------------------------------------------------------------
float compute_signal(
    __global const float* indicator_data,
    int indicator_stride,
    int num_indicators_total,
    int bar_index,
    __private const StrategyParams* params
) {
    // WEIGHTED_SUM
    if (params->aggregation == 0) {
        float weighted_sum = 0.0f;
        float weight_total = 0.0f;
        int filter_active = 1;
        for (int i = 0; i < params->num_indicators; ++i) {
            int idx = params->indicator_indices[i];
            if (idx < 0 || idx >= num_indicators_total) return 0.0f;  // as the CPU engine
            float val = indicator_data[(size_t)idx * (size_t)indicator_stride + (size_t)bar_index];
            if (params->indicator_roles[i] == 1) {  // FILTER
                if (isnan(val) || val == 0.0f) filter_active = 0;
                continue;
            }
            if (isnan(val)) continue;
            weighted_sum += val * params->weights[i];
            weight_total += params->weights[i];
        }
        if (!filter_active) return 0.0f;
        if (weight_total <= 0.0f) return 0.0f;
        return weighted_sum / weight_total;
    }
    // MAJORITY_VOTE
    else if (params->aggregation == 1) {
        int votes_long = 0;
        int votes_short = 0;
        int filter_active = 1;
        for (int i = 0; i < params->num_indicators; ++i) {
            int idx = params->indicator_indices[i];
            if (idx < 0 || idx >= num_indicators_total) return 0.0f;  // as the CPU engine
            float val = indicator_data[(size_t)idx * (size_t)indicator_stride + (size_t)bar_index];
            if (params->indicator_roles[i] == 1) {  // FILTER
                if (isnan(val) || val == 0.0f) filter_active = 0;
                continue;
            }
            if (isnan(val)) continue;
            if (val > 0.0f) votes_long++;
            else if (val < 0.0f) votes_short++;
        }
        if (!filter_active) return 0.0f;
        if (votes_long > votes_short)  return  1.0f;
        if (votes_short > votes_long)  return -1.0f;
        return 0.0f;
    }
    // UNANIMOUS
    else {
        int direction = 0;
        int all_agree = 1;
        int filter_active = 1;
        for (int i = 0; i < params->num_indicators; ++i) {
            int idx = params->indicator_indices[i];
            if (idx < 0 || idx >= num_indicators_total) return 0.0f;  // as the CPU engine
            float val = indicator_data[(size_t)idx * (size_t)indicator_stride + (size_t)bar_index];
            if (params->indicator_roles[i] == 1) {  // FILTER
                if (isnan(val) || val == 0.0f) filter_active = 0;
                continue;
            }
            if (isnan(val)) continue;
            if (val == 0.0f) continue;
            int this_dir = (val > 0.0f) ? 1 : -1;
            if (direction == 0) {
                direction = this_dir;
            } else if (this_dir != direction) {
                all_agree = 0;
                break;
            }
        }
        if (!filter_active) return 0.0f;
        return (all_agree && direction != 0) ? (float)direction : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Main backtest evaluation kernel
//
// One work-item per strategy.  Reads shared indicator arrays (broadcast).
// Writes one BacktestResultGPU to the result buffer.
//
// Indicator data layout: flat buffer, indicator_data[k * indicator_stride + bar]
// where indicator_stride = num_bars (total bars in the window including warmup).
//
// bar_flags: SessionFlag bits for each bar. has_bar_flags enables the
// trading-hours filter; has_session_data enables the flat-at-close exit and
// halt suppression. With both 0 every bar is treated as ETH.
// ---------------------------------------------------------------------------
__kernel void evaluate_backtest(
    __global const float*   bar_open,           // SoA open prices
    __global const float*   bar_high,           // SoA high prices
    __global const float*   bar_low,            // SoA low prices
    __global const float*   bar_close,          // SoA close prices
    __global const int*     bar_flags,          // SessionFlag bits per bar
    __global const float*   indicator_data,     // flat: [num_indicators * indicator_stride]
    int                     indicator_stride,   // = num_bars
    int                     num_indicators,     // total indicator series count
    __global const StrategyParams* strategy_params,  // one per work-item
    __global BacktestResultGPU*    results,          // one per work-item
    int num_bars,
    int eval_start,
    int eval_end,
    float point_value,
    float rt_cost_arg,
    int   entry_lag,
    int   has_bar_flags,    // 1 if the hours bits of bar_flags are valid
    int   has_session_data  // 1 if the close/halt bits of bar_flags are valid
) {
    int gid = get_global_id(0);
    __private StrategyParams params = strategy_params[gid];

    float pt_val  = point_value;
    float rt_cost = rt_cost_arg;

    // Position state (private memory)
    int   pos_dir         = 0;
    float pos_entry_price = 0.0f;
    int   pos_entry_bar   = 0;
    float pos_min_unreal  = 0.0f;
    float pos_max_unreal  = 0.0f;
    float pos_best_price  = 0.0f;

    int   pend_dir        = 0;
    int   pend_signal_bar = -1;
    int   pend_exit       = 0;

    // Metric accumulators (private memory)
    float total_pnl      = 0.0f;
    float gross_wins     = 0.0f;
    float gross_losses   = 0.0f;
    float peak_equity    = 0.0f;
    float max_dd         = 0.0f;
    float max_live_dd    = 0.0f;
    int   trade_count    = 0;
    int   win_count      = 0;
    int   bars_in_pos    = 0;
    float sum_min_unreal = 0.0f;
    float sum_max_unreal = 0.0f;

    int num_eval_bars = eval_end - eval_start;

    // Exit ATR over the evaluated range from completed bars only; mirrors
    // BacktestEngine::run_impl.
    float exit_atr     = 0.0f;
    float exit_atr_sum = 0.0f;
    int   exit_atr_n   = 0;

    for (int i = 0; i < num_eval_bars; ++i) {
        int abs_i = eval_start + i;

        if (i > 0) {
            int pb = abs_i - 1;
            float tr = bar_high[pb] - bar_low[pb];
            if (i > 1) {
                float pc = bar_close[pb - 1];
                tr = fmax(tr, fmax(fabs(bar_high[pb] - pc), fabs(bar_low[pb] - pc)));
            }
            if (exit_atr_n < EXIT_ATR_PERIOD) {
                exit_atr_sum += tr;
                exit_atr_n += 1;
                if (exit_atr_n == EXIT_ATR_PERIOD) exit_atr = exit_atr_sum / (float)EXIT_ATR_PERIOD;
            } else {
                exit_atr = (exit_atr * (float)(EXIT_ATR_PERIOD - 1) + tr) / (float)EXIT_ATR_PERIOD;
            }
        }

        // Step 0: Fill pending exit
        if (pend_exit != 0 && pos_dir != 0) {
            // Fold the fill bar's extremes into the trade's worst and best
            // unrealized PnL before recording it, as the CPU engine does.
            {
                float fill_worst_price = (pos_dir > 0) ? bar_low[abs_i] : bar_high[abs_i];
                float fill_worst_unreal = (float)pos_dir * (fill_worst_price - pos_entry_price) * pt_val;
                if (fill_worst_unreal < pos_min_unreal) pos_min_unreal = fill_worst_unreal;
                float fill_best_price = (pos_dir > 0) ? bar_high[abs_i] : bar_low[abs_i];
                float fill_best_unreal = (float)pos_dir * (fill_best_price - pos_entry_price) * pt_val;
                if (fill_best_unreal > pos_max_unreal) pos_max_unreal = fill_best_unreal;
            }

            float exit_price = bar_open[abs_i];
            float price_diff = (float)pos_dir * (exit_price - pos_entry_price);
            float gross_pnl = price_diff * pt_val;
            float net_pnl = gross_pnl - rt_cost;

            total_pnl += net_pnl;
            if (net_pnl > 0.0f) {
                gross_wins += net_pnl;
                win_count++;
            } else {
                gross_losses += net_pnl;
            }
            trade_count++;
            bars_in_pos += abs_i - pos_entry_bar;
            sum_min_unreal += pos_min_unreal;
            sum_max_unreal += pos_max_unreal;

            if (total_pnl > peak_equity) peak_equity = total_pnl;
            float dd = peak_equity - total_pnl;
            if (dd > max_dd) max_dd = dd;

            if (-pos_min_unreal > max_live_dd) max_live_dd = -pos_min_unreal;

            pos_dir = 0;
            pos_entry_price = 0.0f;
            pos_min_unreal = 0.0f;
            pos_max_unreal = 0.0f;
            pos_best_price = 0.0f;
            pend_exit = 0;

            // A pending entry scheduled for this same fill bar is dropped,
            // so the position never re-enters on the bar it exited on.
            if (pend_dir != 0 && pend_signal_bar == i - entry_lag) {
                pend_dir = 0;
                pend_signal_bar = -1;
            }
        }

        // Step 1: Fill pending entry (never between the close and the next open)
        if (pend_dir != 0 && i == pend_signal_bar + entry_lag) {
            int after_close = (has_session_data != 0 && (bar_flags[abs_i] & SESSION_AFTER_CLOSE) != 0) ? 1 : 0;
            if (pos_dir == 0 && after_close == 0) {
                pos_dir = pend_dir;
                pos_entry_price = bar_open[abs_i];
                pos_entry_bar = abs_i;
                pos_min_unreal = 0.0f;
                pos_max_unreal = 0.0f;
                pos_best_price = pos_entry_price;
            }
            pend_dir = 0;
            pend_signal_bar = -1;
        }

        // Step 2: If in position, update M2M and check exit
        if (pos_dir != 0) {
            float worst_price = (pos_dir > 0) ? bar_low[abs_i] : bar_high[abs_i];
            float worst_unreal = (float)pos_dir * (worst_price - pos_entry_price) * pt_val;
            if (worst_unreal < pos_min_unreal) pos_min_unreal = worst_unreal;

            float best_price_bar = (pos_dir > 0) ? bar_high[abs_i] : bar_low[abs_i];
            float best_unreal = (float)pos_dir * (best_price_bar - pos_entry_price) * pt_val;
            if (best_unreal > pos_max_unreal) pos_max_unreal = best_unreal;

            // Exit check: signal-based exits use pending_exit (next bar open),
            // level-triggered exits fill immediately at the trigger level.
            int should_exit = 0;
            float level_exit_price = 0.0f;  // >0 = immediate fill at this level

            // Level-triggered exits never fire on the entry bar itself.
            int post_entry = (abs_i > pos_entry_bar) ? 1 : 0;

            // A stop the bar opens through fills at the open; targets fill at
            // their level.
            float bar_o = bar_open[abs_i];

            if (params.exit_mode == 0) {  // SIGNAL_REVERSAL
                float sig = compute_signal(indicator_data, indicator_stride, num_indicators, abs_i, &params);
                if ((pos_dir > 0 && sig <= 0.0f) || (pos_dir < 0 && sig >= 0.0f)) {
                    should_exit = 1;
                }
            } else if (params.exit_mode == 1) {  // BAR_TIMEOUT
                if (abs_i - pos_entry_bar >= params.exit_timeout_bars) {
                    should_exit = 1;
                }
            } else if (params.exit_mode == 2) {  // TRAILING_STOP
                float atr_val = exit_atr;
                if (params.atr_series_index >= 0 && params.atr_series_index < num_indicators) {
                    atr_val = (i > 0) ? indicator_data[(size_t)params.atr_series_index * (size_t)indicator_stride + (size_t)(abs_i - 1)] : 0.0f;
                }
                if (post_entry != 0 && !isnan(atr_val) && atr_val > 0.0f) {
                    float stop_dist = params.trailing_atr_mult * atr_val;
                    if (pos_dir > 0) {
                        float stop_level = pos_best_price - stop_dist;
                        if (bar_low[abs_i] <= stop_level) { should_exit = 1; level_exit_price = fmin(bar_o, stop_level); }
                    } else {
                        float stop_level = pos_best_price + stop_dist;
                        if (bar_high[abs_i] >= stop_level) { should_exit = 1; level_exit_price = fmax(bar_o, stop_level); }
                    }
                }
            } else if (params.exit_mode == 3) {  // OPPOSING_SIGNAL
                float sig = compute_signal(indicator_data, indicator_stride, num_indicators, abs_i, &params);
                if ((pos_dir > 0 && sig < -params.entry_threshold) ||
                    (pos_dir < 0 && sig >  params.entry_threshold)) {
                    should_exit = 1;
                }
            } else if (params.exit_mode == 4) {  // MAX_LOSS_ATR
                float atr_val = exit_atr;
                if (params.atr_series_index >= 0 && params.atr_series_index < num_indicators) {
                    atr_val = (i > 0) ? indicator_data[(size_t)params.atr_series_index * (size_t)indicator_stride + (size_t)(abs_i - 1)] : 0.0f;
                }
                if (post_entry != 0 && !isnan(atr_val) && atr_val > 0.0f) {
                    float stop_dist = params.max_loss_atr_mult * atr_val;
                    if (pos_dir > 0) {
                        float sl = pos_entry_price - stop_dist;
                        if (bar_low[abs_i] <= sl) { should_exit = 1; level_exit_price = fmin(bar_o, sl); }
                    } else {
                        float sl = pos_entry_price + stop_dist;
                        if (bar_high[abs_i] >= sl) { should_exit = 1; level_exit_price = fmax(bar_o, sl); }
                    }
                }
            } else if (params.exit_mode == 5) {  // TAKE_PROFIT_ATR
                float atr_val = exit_atr;
                if (params.atr_series_index >= 0 && params.atr_series_index < num_indicators) {
                    atr_val = (i > 0) ? indicator_data[(size_t)params.atr_series_index * (size_t)indicator_stride + (size_t)(abs_i - 1)] : 0.0f;
                }
                if (post_entry != 0 && !isnan(atr_val) && atr_val > 0.0f) {
                    float tp_dist = params.tp_atr_mult * atr_val;
                    if (pos_dir > 0) {
                        float tp = pos_entry_price + tp_dist;
                        if (bar_high[abs_i] >= tp) { should_exit = 1; level_exit_price = tp; }
                    } else {
                        float tp = pos_entry_price - tp_dist;
                        if (bar_low[abs_i] <= tp) { should_exit = 1; level_exit_price = tp; }
                    }
                }
            } else if (params.exit_mode == 6) {  // TP_SL_ATR
                float atr_val = exit_atr;
                if (params.atr_series_index >= 0 && params.atr_series_index < num_indicators) {
                    atr_val = (i > 0) ? indicator_data[(size_t)params.atr_series_index * (size_t)indicator_stride + (size_t)(abs_i - 1)] : 0.0f;
                }
                if (post_entry != 0 && !isnan(atr_val) && atr_val > 0.0f) {
                    float tp_dist = params.tp_atr_mult * atr_val;
                    float sl_dist = params.sl_atr_mult * atr_val;
                    // SL takes precedence when both trigger on same bar
                    if (pos_dir > 0) {
                        float sl = pos_entry_price - sl_dist;
                        float tp = pos_entry_price + tp_dist;
                        if (bar_low[abs_i] <= sl) { should_exit = 1; level_exit_price = fmin(bar_o, sl); }
                        else if (bar_high[abs_i] >= tp) { should_exit = 1; level_exit_price = tp; }
                    } else {
                        float sl = pos_entry_price + sl_dist;
                        float tp = pos_entry_price - tp_dist;
                        if (bar_high[abs_i] >= sl) { should_exit = 1; level_exit_price = fmax(bar_o, sl); }
                        else if (bar_low[abs_i] <= tp) { should_exit = 1; level_exit_price = tp; }
                    }
                }
            }

            // Flat at the daily close: exit at the close bar's own close, also
            // when a signal exit was raised on it; an intrabar stop or target
            // fill takes precedence.
            if (has_session_data != 0 && (bar_flags[abs_i] & SESSION_CLOSE_BAR) != 0 &&
                level_exit_price <= 0.0f) {
                should_exit = 1;
                level_exit_price = bar_close[abs_i];
            }

            if (should_exit != 0) {
                if (level_exit_price > 0.0f) {
                    // Level-triggered: fill at trigger level immediately
                    float price_diff = (float)pos_dir * (level_exit_price - pos_entry_price);
                    float gross_pnl = price_diff * pt_val;
                    float net_pnl = gross_pnl - rt_cost;
                    total_pnl += net_pnl;
                    if (net_pnl > 0.0f) { gross_wins += net_pnl; win_count++; }
                    else { gross_losses += net_pnl; }
                    trade_count++;
                    bars_in_pos += abs_i - pos_entry_bar;
                    sum_min_unreal += pos_min_unreal;
                    sum_max_unreal += pos_max_unreal;
                    if (-pos_min_unreal > max_live_dd) max_live_dd = -pos_min_unreal;
                    if (total_pnl > peak_equity) peak_equity = total_pnl;
                    float dd = peak_equity - total_pnl;
                    if (dd > max_dd) max_dd = dd;
                    pos_dir = 0;
                    pos_entry_price = 0.0f;
                    pos_min_unreal = 0.0f;
                    pos_max_unreal = 0.0f;
                    pos_best_price = 0.0f;
                } else if (i + 1 < num_eval_bars) {
                    pend_exit = 1;
                } else {
                    // Last bar: exit at close
                    float exit_price = bar_close[abs_i];
                    float price_diff = (float)pos_dir * (exit_price - pos_entry_price);
                    float gross_pnl = price_diff * pt_val;
                    float net_pnl = gross_pnl - rt_cost;
                    total_pnl += net_pnl;
                    if (net_pnl > 0.0f) { gross_wins += net_pnl; win_count++; }
                    else { gross_losses += net_pnl; }
                    trade_count++;
                    bars_in_pos += abs_i - pos_entry_bar;
                    sum_min_unreal += pos_min_unreal;
                    sum_max_unreal += pos_max_unreal;
                    if (-pos_min_unreal > max_live_dd) max_live_dd = -pos_min_unreal;
                    if (total_pnl > peak_equity) peak_equity = total_pnl;
                    float dd2 = peak_equity - total_pnl;
                    if (dd2 > max_dd) max_dd = dd2;
                    pos_dir = 0;
                }
            }

            // Advance the trailing-stop reference only after this bar's exits
            // were checked against the previous reference.
            if (pos_dir != 0 && pend_exit == 0) {
                if (pos_dir > 0) {
                    if (bar_high[abs_i] > pos_best_price) pos_best_price = bar_high[abs_i];
                } else {
                    if (bar_low[abs_i] < pos_best_price) pos_best_price = bar_low[abs_i];
                }
            }

            // Force exit on last bar
            if (pos_dir != 0 && pend_exit == 0 && i == num_eval_bars - 1) {
                float exit_price = bar_close[abs_i];
                float price_diff = (float)pos_dir * (exit_price - pos_entry_price);
                float gross_pnl = price_diff * pt_val;
                float net_pnl = gross_pnl - rt_cost;
                total_pnl += net_pnl;
                if (net_pnl > 0.0f) { gross_wins += net_pnl; win_count++; }
                else { gross_losses += net_pnl; }
                trade_count++;
                bars_in_pos += abs_i - pos_entry_bar;
                sum_min_unreal += pos_min_unreal;
                sum_max_unreal += pos_max_unreal;
                if (-pos_min_unreal > max_live_dd) max_live_dd = -pos_min_unreal;
                if (total_pnl > peak_equity) peak_equity = total_pnl;
                float dd3 = peak_equity - total_pnl;
                if (dd3 > max_dd) max_dd = dd3;
                pos_dir = 0;
            }
        }

        // Step 3: If flat and no pending, evaluate signal
        if (pos_dir == 0 && pend_dir == 0 && pend_exit == 0) {
            // No new entries during a halt or between the close and the next open
            int halted = 0;
            if (has_session_data != 0 && (bar_flags[abs_i] & (SESSION_HALTED | SESSION_AFTER_CLOSE)) != 0) {
                halted = 1;
            }

            // Trading hours filter
            int allowed = 1;
            if (halted == 0 && has_bar_flags != 0 && params.trading_hours != 0) {
                allowed = is_entry_allowed(params.trading_hours, bar_flags[abs_i]);
            }

            if (halted == 0 && allowed != 0 && i + entry_lag < num_eval_bars) {
                float sig = compute_signal(indicator_data, indicator_stride, num_indicators, abs_i, &params);
                if (sig > params.entry_threshold) {
                    pend_dir = 1;
                    pend_signal_bar = i;
                } else if (sig < -params.entry_threshold) {
                    pend_dir = -1;
                    pend_signal_bar = i;
                }
            }
        }
    }

    // Finalize peak equity
    if (total_pnl > peak_equity) peak_equity = total_pnl;
    float dd = peak_equity - total_pnl;
    if (dd > max_dd) max_dd = dd;

    // Write results
    results[gid].total_pnl         = total_pnl;
    results[gid].total_trades      = trade_count;
    results[gid].win_count         = win_count;
    results[gid].gross_wins        = gross_wins;
    results[gid].gross_losses      = gross_losses;
    results[gid].max_drawdown          = max_dd;
    results[gid].max_live_drawdown     = max_live_dd;
    results[gid].peak_equity           = peak_equity;
    results[gid].total_bars_in_position = bars_in_pos;
    results[gid].sum_min_unreal        = sum_min_unreal;
    results[gid].sum_max_unreal        = sum_max_unreal;
}
