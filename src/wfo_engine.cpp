#include "wfo_engine.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <iostream>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "compute/indicator_warmup.h"
#include "indicators/signal_normalize.h"
#include "resource_monitor.h"

// ---------------------------------------------------------------------------
// WFOConfig::from_schedule
// ---------------------------------------------------------------------------
WFOConfig WFOConfig::from_schedule(const WFOScheduleConfig& sc, float bars_per_day) {
    WFOConfig cfg;
    cfg.frequency_days = frequency_approx_days(sc.frequency);
    cfg.lookback_days  = lookback_approx_days(sc.lookback);
    cfg.frequency_bars = static_cast<int>(static_cast<float>(cfg.frequency_days) * bars_per_day);
    cfg.lookback_bars  = static_cast<int>(static_cast<float>(cfg.lookback_days) * bars_per_day);

    // Ensure minimums
    if (cfg.frequency_bars < 1) cfg.frequency_bars = 1;
    if (cfg.lookback_bars < 1) cfg.lookback_bars = 1;

    // Adaptive trigger settings
    cfg.trigger          = sc.trigger;
    cfg.trigger_dd_frac  = trigger_dd_pct(sc.trigger);
    cfg.trigger_losers_n = trigger_losers_count(sc.trigger);

    return cfg;
}

// ---------------------------------------------------------------------------
// apply_is_floors -- per-window in-sample quality floors
//
//   - total_trades >= 30
//   - profit_factor > 1.0
//   - avg per-trade PnL >= 2x spread (matches BacktestResult::viable threshold)
//   - annual_roc > 0
// ---------------------------------------------------------------------------
std::vector<int> WFOEngine::apply_is_floors(const std::vector<BacktestResult>& results,
                                            const CostModel&                   cost) const {
    const float min_avg_pnl =
        2.0f * cost.round_trip_cost();  // 2x spread -- matches metrics.cpp viable-flag threshold
    std::vector<int> passed;
    passed.reserve(results.size());

    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        const BacktestResult& r = results[i];
        if (r.total_trades < IS_MIN_TRADES) continue;
        if (r.profit_factor <= IS_MIN_PF) continue;
        if (r.annual_roc <= IS_MIN_ROC) continue;

        // avg per-trade PnL >= 2x spread
        if (r.total_trades > 0) {
            float avg_pnl = r.total_pnl / static_cast<float>(r.total_trades);
            if (avg_pnl < min_avg_pnl) continue;
        }

        passed.push_back(i);
    }
    return passed;
}

// ---------------------------------------------------------------------------
// compute_session_signals -- compute session signals for a bar range and
// append to signal arrays. Returns the number of session signal series added.
// ---------------------------------------------------------------------------
int WFOEngine::compute_session_signals(int abs_bar_start, int abs_bar_end, int load_count,
                                       std::vector<std::vector<float>>& signal_storage,
                                       std::vector<float*>&             signal_ptrs) const {
    if (!m_market_data || !m_market_data->session_model().session_signals) return 0;
    (void)load_count;  // the series span exactly [abs_bar_start, abs_bar_end)

    const int num_added = append_session_signal_series(*m_market_data, m_session_signal_grid,
                                                       abs_bar_start, abs_bar_end, signal_storage);

    // Rebuild signal_ptrs from signal_storage to avoid stale pointers
    signal_ptrs.resize(signal_storage.size());
    for (size_t i = 0; i < signal_storage.size(); ++i) {
        signal_ptrs[i] = signal_storage[i].data();
    }

    return num_added;
}

// ---------------------------------------------------------------------------
// precompute_all_signals -- compute all reduced-set indicators + session
// signals once over the full bar dataset.  After this call,
// compute_signals_for_range() returns slices into this data.
// ---------------------------------------------------------------------------
void WFOEngine::precompute_all_signals(const Bar* bars, int num_bars, CLContext& ctx,
                                       CLIndicators& cl_ind) {
    auto t_start = std::chrono::steady_clock::now();
    std::cout << "Precomputing signals over " << num_bars << " bars..." << std::flush;

    const IndicatorSpecSet spec_set = build_indicator_specs();
    const int              num_tech = spec_set.num_signals;  // = indicator_grid.size()

    // -- Step 1: Compute raw indicator series via GPU/CPU --
    CLMemory memory;
    auto indicator_ptrs = memory.allocate_indicator_buffers(ctx, spec_set.total_series, num_bars);

    memory.upload_bars(ctx, bars, 0, num_bars);
    const SoABarData& soa = memory.soa_bars();

    cl_ind.compute_with_bars(ctx, bars, 0, soa.open.data(), soa.high.data(), soa.low.data(),
                             soa.close.data(), num_bars, spec_set.specs, indicator_ptrs);

    // -- Step 2: Allocate signal storage --
    // Active indicators get their own buffer; inactive share a NaN buffer.
    m_precomputed.nan_buffer.assign(num_bars, std::numeric_limits<float>::quiet_NaN());
    m_precomputed.tech_signals.clear();
    m_precomputed.signal_ptrs.resize(num_tech);

    // Initialize all tech slots to NaN buffer
    for (int i = 0; i < num_tech; ++i)
        m_precomputed.signal_ptrs[i] = m_precomputed.nan_buffer.data();

    // -- Step 3: Normalize active indicators into dedicated buffers --
    int raw_idx = 0;
    while (raw_idx < spec_set.total_series) {
        int                    ind_id = spec_set.series_map[raw_idx].first;
        const IndicatorParams& params = m_indicator_grid.params(ind_id);
        int                    ns     = indicator_num_series(params);

        m_precomputed.tech_signals.emplace_back(num_bars);
        float* out = m_precomputed.tech_signals.back().data();

        std::vector<const float*> series_ptrs(ns);
        for (int s = 0; s < ns; ++s) series_ptrs[s] = indicator_ptrs[raw_idx + s];

        normalize_to_signal(params, series_ptrs.data(), ns, bars, 0, num_bars, out);

        m_precomputed.signal_ptrs[ind_id] = out;
        raw_idx += ns;
    }

    // -- Step 4: Precompute session signal per-day values --
    m_precomputed.num_session_signals = 0;
    if (m_market_data && m_market_data->session_model().session_signals) {
        const MarketData& md        = *m_market_data;
        int               day_start = md.first_session_day_in_range(0);
        int               day_end   = md.last_session_day_in_range(num_bars) + 1;

        if (day_start >= 0 && day_end > day_start) {
            m_precomputed.session_day_start = day_start;
            m_precomputed.session_day_end   = day_end;
            int num_days                    = day_end - day_start;

            m_precomputed.session_day_values.clear();
            m_precomputed.session_avail_after.clear();
            m_precomputed.session_suppress_halt.clear();

            for (int gi = 0; gi < m_session_signal_grid.size(); ++gi) {
                const SessionSignalParams& sp   = m_session_signal_grid.params(gi);
                auto                       inst = create_session_signal(sp);
                if (!inst) continue;

                int  ns       = inst->num_signals();
                auto avail    = inst->signal_available_after_et_seconds();
                bool suppress = inst->suppress_on_halt();

                std::vector<float> day_output(ns * num_days);
                inst->compute(md, day_start, day_end, day_output.data());

                // Store per-series day values
                for (int si = 0; si < ns; ++si) {
                    std::vector<float> series_days(num_days);
                    std::memcpy(series_days.data(), &day_output[si * num_days],
                                num_days * sizeof(float));
                    m_precomputed.session_day_values.push_back(std::move(series_days));
                    m_precomputed.session_avail_after.push_back(avail[si]);
                    m_precomputed.session_suppress_halt.push_back(suppress);
                    m_precomputed.num_session_signals++;
                }
            }
        }
    }

    m_precomputed.num_tech_signals = num_tech;
    m_precomputed.total_signals    = num_tech + m_precomputed.num_session_signals;
    m_precomputed.total_bars       = num_bars;
    m_precomputed.valid            = true;

    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::cout << " done (" << static_cast<int>(elapsed) << "s, "
              << m_precomputed.tech_signals.size() << " tech + "
              << m_precomputed.num_session_signals << " session signals, "
              << (m_precomputed.tech_signals.size() * num_bars * 4 / (1024 * 1024)) << " MB)"
              << std::endl;
}

// ---------------------------------------------------------------------------
// build_indicator_specs -- build the set of indicator specs to compute,
// optionally filtered to only the reduced indicator set for the walk-forward stage.
// ---------------------------------------------------------------------------
WFOEngine::IndicatorSpecSet WFOEngine::build_indicator_specs() const {
    IndicatorSpecSet ss;
    ss.num_signals = m_indicator_grid.size();

    std::unordered_set<int> active;
    if (!m_reduced_indicator_ids.empty()) {
        active.insert(m_reduced_indicator_ids.begin(), m_reduced_indicator_ids.end());
        ss.use_reduced_set = true;
    }

    for (int i = 0; i < m_indicator_grid.size(); ++i) {
        if (ss.use_reduced_set && active.find(i) == active.end()) continue;
        const IndicatorParams& params = m_indicator_grid.params(i);
        int                    ns     = indicator_num_series(params);
        for (int s = 0; s < ns; ++s) {
            ss.specs.push_back(IndicatorSpec::from_params(params, s));
            ss.series_map.push_back({i, s});
        }
    }
    ss.total_series = static_cast<int>(ss.specs.size());
    return ss;
}

// ---------------------------------------------------------------------------
// compute_signals_for_range -- compute indicators, normalize to signals,
// append session signals for a given bar range. Returns total signal count.
// ---------------------------------------------------------------------------
int WFOEngine::compute_signals_for_range(const IndicatorSpecSet& spec_set, const Bar* bars,
                                         int load_start, int load_count, int abs_bar_end,
                                         CLContext& ctx, CLIndicators& cl_ind,
                                         std::vector<std::vector<float>>& signal_storage,
                                         std::vector<float*>&             signal_ptrs) const {
    // ---- Fast path: use precomputed full-dataset signals ----
    if (m_precomputed.valid) {
        const int num_tech    = m_precomputed.num_tech_signals;
        const int num_session = m_precomputed.num_session_signals;
        const int total       = num_tech + num_session;

        // Point technical signals into precomputed arrays at load_start offset
        signal_storage.clear();
        signal_ptrs.resize(total);
        for (int i = 0; i < num_tech; ++i) {
            signal_ptrs[i] = m_precomputed.signal_ptrs[i] + load_start;
        }

        // Broadcast session signals from cached per-day values to this
        // window's bar range.  This is O(load_count) per signal -- fast.
        if (m_market_data && num_session > 0) {
            const MarketData& md       = *m_market_data;
            int               load_end = load_start + load_count;
            if (load_end > abs_bar_end) load_end = abs_bar_end;

            // Reuse a single storage block for session signals
            signal_storage.resize(num_session);
            for (int si = 0; si < num_session; ++si) {
                signal_storage[si].resize(load_count);
                broadcast_session_signal_to_bars(
                    md, m_precomputed.session_day_values[si].data(),
                    m_precomputed.session_day_start, m_precomputed.session_day_end,
                    m_precomputed.session_avail_after[si], load_start, load_end,
                    signal_storage[si].data(), m_precomputed.session_suppress_halt[si]);
                signal_ptrs[num_tech + si] = signal_storage[si].data();
            }
        }

        return total;
    }

    // ---- Without precomputed signals: compute this range from scratch ----
    CLMemory memory;
    auto indicator_ptrs = memory.allocate_indicator_buffers(ctx, spec_set.total_series, load_count);

    memory.upload_bars(ctx, &bars[load_start], 0, load_count);
    const SoABarData& soa = memory.soa_bars();

    cl_ind.compute_with_bars(ctx, &bars[load_start], 0, soa.open.data(), soa.high.data(),
                             soa.low.data(), soa.close.data(), load_count, spec_set.specs,
                             indicator_ptrs);

    // Allocate signal arrays for ALL indicators (indexed by original ID).
    // Fill with NaN so unused indicators are neutral in composite signal.
    signal_storage.resize(spec_set.num_signals);
    signal_ptrs.resize(spec_set.num_signals);
    for (int i = 0; i < spec_set.num_signals; ++i) {
        signal_storage[i].assign(load_count, std::numeric_limits<float>::quiet_NaN());
        signal_ptrs[i] = signal_storage[i].data();
    }

    // Normalize: reduced set normalizes only active indicators into their
    // correct signal slots; full set normalizes all.
    if (spec_set.use_reduced_set) {
        int raw_idx = 0;
        while (raw_idx < spec_set.total_series) {
            int                    ind_id = spec_set.series_map[raw_idx].first;
            const IndicatorParams& params = m_indicator_grid.params(ind_id);
            int                    ns     = indicator_num_series(params);

            std::vector<const float*> series_ptrs(ns);
            for (int s = 0; s < ns; ++s) series_ptrs[s] = indicator_ptrs[raw_idx + s];

            normalize_to_signal(params, series_ptrs.data(), ns, &bars[load_start], 0, load_count,
                                signal_ptrs[ind_id]);
            raw_idx += ns;
        }
    } else {
        normalize_all_indicators(m_indicator_grid, indicator_ptrs, &bars[load_start], load_count,
                                 signal_ptrs);
    }

    // Append session signals
    int num_signals = spec_set.num_signals;
    int load_end    = load_start + load_count;
    if (load_end > abs_bar_end) load_end = abs_bar_end;
    int session_sigs =
        compute_session_signals(load_start, load_end, load_count, signal_storage, signal_ptrs);
    num_signals += session_sigs;

    return num_signals;
}

// ---------------------------------------------------------------------------
// evaluate_is_window -- evaluate all reduced-grid strategies on an IS window
// ---------------------------------------------------------------------------
void WFOEngine::evaluate_is_window(const Bar* bars, int num_bars, int is_start, int is_end,
                                   const std::vector<int>& reduced_strategy_ids,
                                   const CostModel& cost, CLContext& ctx, CLIndicators& cl_ind,
                                   CLBacktest& cl_bt, WindowEvaluation& eval) {
    eval.is_start_bar = is_start;
    eval.is_end_bar   = is_end;
    eval.is_results.clear();
    eval.is_results.reserve(reduced_strategy_ids.size());

    const int warmup     = max_indicator_warmup(m_indicator_grid);
    const int load_start = std::max(0, is_start - warmup);
    const int load_end   = std::min(num_bars, is_end);
    const int load_count = load_end - load_start;

    if (load_count <= 0 || is_start >= is_end) {
        eval.empty_frontier = true;
        return;
    }

    // The whole window is evaluated in one pass: splitting it would close
    // positions and restart the exit ATR at every split point.
    const IndicatorSpecSet spec_set = build_indicator_specs();
    {
        std::vector<std::vector<float>> signal_storage;
        std::vector<float*>             signal_ptrs;
        int num_signals = compute_signals_for_range(spec_set, bars, load_start, load_count, is_end,
                                                    ctx, cl_ind, signal_storage, signal_ptrs);

        int eval_start_local = is_start - load_start;
        int eval_end_local   = is_end - load_start;

        const StrategyGrid& strategy_grid = get_strategy_grid();

        // Batch all valid strategies into a single evaluate() call.
        // This lets evaluate_cpu() parallelize across strategies internally
        // instead of paying per-call overhead for each strategy.
        std::vector<StrategyConfig> batch_configs;
        std::vector<int>            batch_strategy_ids;  // original strategy IDs, in order
        std::vector<int> invalid_indices;  // indices into reduced_strategy_ids that are invalid

        batch_configs.reserve(reduced_strategy_ids.size());
        batch_strategy_ids.reserve(reduced_strategy_ids.size());

        for (int i = 0; i < static_cast<int>(reduced_strategy_ids.size()); ++i) {
            int strategy_id = reduced_strategy_ids[i];
            if (strategy_id < 0 || strategy_id >= strategy_grid.total_count()) {
                invalid_indices.push_back(i);
                continue;
            }
            StrategyGridEntry entry = strategy_grid.entry_at(strategy_id);
            batch_configs.push_back(entry.to_strategy_config());
            batch_strategy_ids.push_back(strategy_id);
        }

        // Evaluate all valid strategies in one batched call
        std::vector<BacktestResult> batch_results;
        if (!batch_configs.empty()) {
            // Thread sessions (shifted to the load_start base used by
            // &bars[load_start]) so session-close exit, halt-flag
            // suppression, and RTH/MORNING/AFTERNOON hours filter activate in
            // the walk-forward stage as they do in the screen stage.
            const SessionInfo* sessions_full = m_market_data ? m_market_data->sessions() : nullptr;
            const SessionInfo* sessions_slice =
                sessions_full ? sessions_full + load_start : nullptr;
            cl_bt.evaluate(ctx, &bars[load_start], 0,
                           const_cast<const float* const*>(signal_ptrs.data()), num_signals,
                           load_count, eval_start_local, eval_end_local, batch_configs, cost,
                           1,  // lag-1 for IS
                           batch_results, EarlyTermination{}, sessions_slice);
        }

        // Rebuild eval.is_results in the original order, inserting empty
        // results for invalid strategy IDs.
        {
            int batch_idx   = 0;
            int invalid_idx = 0;
            for (int i = 0; i < static_cast<int>(reduced_strategy_ids.size()); ++i) {
                if (invalid_idx < static_cast<int>(invalid_indices.size()) &&
                    invalid_indices[invalid_idx] == i) {
                    BacktestResult empty_r;
                    empty_r.strategy_id = reduced_strategy_ids[i];
                    eval.is_results.push_back(empty_r);
                    ++invalid_idx;
                } else {
                    if (batch_idx < static_cast<int>(batch_results.size())) {
                        batch_results[batch_idx].strategy_id = batch_strategy_ids[batch_idx];
                        eval.is_results.push_back(batch_results[batch_idx]);
                    } else {
                        BacktestResult empty_r;
                        empty_r.strategy_id = reduced_strategy_ids[i];
                        eval.is_results.push_back(empty_r);
                    }
                    ++batch_idx;
                }
            }
        }

        // The GPU kernel returns raw accumulators; consistency metrics used
        // by the Pareto objectives and selection (green weeks, red-week
        // streaks) come only from the CPU engine. Re-run the in-sample floor
        // survivors on the CPU path so selection sees identical inputs on
        // every device.
        if (ctx.is_available()) {
            const std::vector<int> survivors = apply_is_floors(eval.is_results, cost);
            if (!survivors.empty()) {
                std::vector<StrategyConfig> survivor_configs;
                survivor_configs.reserve(survivors.size());
                for (int idx : survivors)
                    survivor_configs.push_back(
                        strategy_grid.entry_at(eval.is_results[idx].strategy_id)
                            .to_strategy_config());

                const SessionInfo* sessions_full =
                    m_market_data ? m_market_data->sessions() : nullptr;
                CLContext  cpu_only;  // never initialized: CLBacktest takes the CPU path
                CLBacktest cpu_bt;
                std::vector<BacktestResult> full;
                cpu_bt.evaluate(cpu_only, &bars[load_start], 0,
                                const_cast<const float* const*>(signal_ptrs.data()), num_signals,
                                load_count, eval_start_local, eval_end_local, survivor_configs,
                                cost, 1, full, EarlyTermination{},
                                sessions_full ? sessions_full + load_start : nullptr);
                for (size_t k = 0; k < survivors.size(); ++k) {
                    const int idx                    = survivors[k];
                    const int id                     = eval.is_results[idx].strategy_id;
                    eval.is_results[idx]             = full[k];
                    eval.is_results[idx].strategy_id = id;
                }
            }
        }
    }

    // Apply quality floors
    eval.passed_indices = apply_is_floors(eval.is_results, cost);

    if (eval.passed_indices.empty()) {
        eval.empty_frontier = true;
        return;
    }

    // Build a filtered results vector for Pareto extraction
    std::vector<BacktestResult> passed_results;
    passed_results.reserve(eval.passed_indices.size());
    for (int idx : eval.passed_indices) {
        passed_results.push_back(eval.is_results[idx]);
    }

    // Extract Pareto frontier from the passed results
    std::vector<int> frontier_in_passed = extract_pareto_frontier(passed_results);

    if (frontier_in_passed.empty()) {
        eval.empty_frontier = true;
        return;
    }

    // Map frontier indices back to the is_results index space
    eval.frontier_indices.clear();
    for (int fi : frontier_in_passed) {
        eval.frontier_indices.push_back(eval.passed_indices[fi]);
    }

    // Classify frontier using full IS pool (passed_results) for thresholds
    eval.frontier       = classify_frontier(eval.is_results, eval.frontier_indices, passed_results);
    eval.empty_frontier = false;
}

// ---------------------------------------------------------------------------
// evaluate_windows -- evaluate all windows for a schedule (expensive part)
//
// Always computes IS for every window. The adaptive re-optimization trigger
// logic (DD / LOSERS) lives in apply_selection, where it can correctly track
// THAT selection's deployed strategy's OOS performance. IS evaluation stays
// shared across all 7 selection strategies.
// ---------------------------------------------------------------------------
std::vector<WindowEvaluation> WFOEngine::evaluate_windows(
    const Bar* bars, int num_bars, const WFOConfig& schedule,
    const std::vector<int>& reduced_strategy_ids, const CostModel& cost) {
    // Delegate to shared-CL overload with locally-created instances
    CLContext ctx;
    ctx.initialize(CLContext::default_device());
    CLIndicators cl_ind;
    CLBacktest   cl_bt;
    return evaluate_windows(bars, num_bars, schedule, reduced_strategy_ids, cost, ctx, cl_ind,
                            cl_bt);
}

std::vector<WindowEvaluation> WFOEngine::evaluate_windows(
    const Bar* bars, int num_bars, const WFOConfig& schedule,
    const std::vector<int>& reduced_strategy_ids, const CostModel& cost, CLContext& ctx,
    CLIndicators& cl_ind, CLBacktest& cl_bt) {
    std::vector<WindowEvaluation> evaluations;

    if (num_bars <= 0 || reduced_strategy_ids.empty()) return evaluations;

    const int L = schedule.lookback_bars;
    const int F = schedule.frequency_bars;

    // t = current IS end = OOS start. The first OOS window starts after a
    // full lookback and no earlier than m_min_oos_start.
    int  t             = std::max(L, m_min_oos_start);
    int  window_idx    = 0;
    int  total_windows = std::max(0, (num_bars - t) / F);
    auto t_windows     = std::chrono::steady_clock::now();

    while (t + F <= num_bars) {
        if (total_windows > 3) {
            double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_windows).count();
            double frac = static_cast<double>(window_idx) / static_cast<double>(total_windows);
            if (window_idx == 0 || (window_idx % std::max(1, total_windows / 10)) == 0) {
                int remain_sec = (frac > 0.0 && elapsed > 0.0)
                                     ? static_cast<int>((elapsed / frac) - elapsed)
                                     : 0;
                std::cout << "  Window " << (window_idx + 1) << "/" << total_windows;
                if (window_idx > 0) std::cout << "  ETA " << remain_sec << "s";
                std::cout << std::endl;
            }
        }
        WindowEvaluation eval;
        eval.is_start_bar  = t - L;
        eval.is_end_bar    = t;
        eval.oos_start_bar = t;
        eval.oos_end_bar   = t + F;

        // Always re-evaluate IS for each window. The adaptive trigger
        // decision (whether to USE this IS frontier or stick with the cached
        // one) is made later, per-selection, in apply_selection.
        evaluate_is_window(bars, num_bars, eval.is_start_bar, eval.is_end_bar, reduced_strategy_ids,
                           cost, ctx, cl_ind, cl_bt, eval);

        evaluations.push_back(std::move(eval));

        t += F;
        ++window_idx;
    }

    return evaluations;
}

// ---------------------------------------------------------------------------
// apply_selection -- apply a selection strategy to pre-evaluated windows
// ---------------------------------------------------------------------------
WFORunResult WFOEngine::apply_selection(const Bar* bars, int num_bars, const WFOConfig& schedule,
                                        SelectionStrategy                    selection,
                                        const std::vector<WindowEvaluation>& evaluations,
                                        const std::vector<int>& /*reduced_strategy_ids*/,
                                        const CostModel& cost, const SessionInfo* sessions) {
    // Delegate to shared-CL overload with locally-created instances
    CLContext ctx;
    ctx.initialize(CLContext::default_device());
    CLIndicators cl_ind;
    CLBacktest   cl_bt;
    return apply_selection(bars, num_bars, schedule, selection, evaluations, {}, cost, ctx, cl_ind,
                           cl_bt, sessions);
}

WFORunResult WFOEngine::apply_selection(const Bar* bars, int num_bars, const WFOConfig& schedule,
                                        SelectionStrategy                    selection,
                                        const std::vector<WindowEvaluation>& evaluations,
                                        const std::vector<int>&              reduced_strategy_ids,
                                        const CostModel& cost, CLContext& ctx, CLIndicators& cl_ind,
                                        CLBacktest& cl_bt, const SessionInfo* sessions) {
    PrecomputedOOSData oos = precompute_oos_signals(bars, num_bars, evaluations, ctx, cl_ind);
    return apply_selection(bars, num_bars, schedule, selection, evaluations, reduced_strategy_ids,
                           cost, ctx, cl_ind, cl_bt, oos, sessions);
}

// ---------------------------------------------------------------------------
// precompute_oos_signals -- compute OOS indicators/signals once for all windows.
// This is the expensive part, done once rather than once per selection
// strategy. The returned PrecomputedOOSData is shared across
// all apply_selection() calls for the same schedule.
// ---------------------------------------------------------------------------
PrecomputedOOSData WFOEngine::precompute_oos_signals(
    const Bar* bars, int num_bars, const std::vector<WindowEvaluation>& evaluations, CLContext& ctx,
    CLIndicators& cl_ind) {
    PrecomputedOOSData result;
    result.windows.resize(evaluations.size());

    const IndicatorSpecSet spec_set = build_indicator_specs();
    const int              warmup   = max_indicator_warmup(m_indicator_grid);

    for (int wi = 0; wi < static_cast<int>(evaluations.size()); ++wi) {
        const WindowEvaluation& eval = evaluations[wi];
        PrecomputedOOSWindow&   pw   = result.windows[wi];
        pw.num_signals               = spec_set.num_signals;

        // Precompute OOS signals for EVERY window (regardless of this
        // window's own empty_frontier). Under the adaptive-trigger
        // architecture, a selection may deploy a cached
        // strategy from a prior window onto this window's OOS range even
        // when this window's own IS frontier is empty.
        const int load_start = std::max(0, eval.oos_start_bar - warmup);
        const int load_end   = std::min(num_bars, eval.oos_end_bar);
        const int load_count = load_end - load_start;

        if (load_count <= 0) {
            pw.valid = false;
            continue;
        }

        pw.load_start = load_start;
        pw.load_count = load_count;
        pw.eval_start = eval.oos_start_bar - load_start;
        pw.eval_end   = eval.oos_end_bar - load_start;

        // Compute signals directly into pw's storage, caching the full
        // pointer array (tech + session) so apply_selection doesn't rebuild
        // from signal_storage alone (which only holds session signals in
        // the precomputed fast path).
        pw.num_signals =
            compute_signals_for_range(spec_set, bars, load_start, load_count, eval.oos_end_bar, ctx,
                                      cl_ind, pw.signal_storage, pw.cached_signal_ptrs);

        pw.valid = true;
    }

    return result;
}

// ---------------------------------------------------------------------------
// apply_selection -- pick a winner per window with one selection strategy,
// trade it out of sample, and stitch the results. The out-of-sample runs use
// the CPU engine, so every OOS metric is the same on every device.
// ---------------------------------------------------------------------------
WFORunResult WFOEngine::apply_selection(const Bar* bars, int num_bars, const WFOConfig& schedule,
                                        SelectionStrategy                    selection,
                                        const std::vector<WindowEvaluation>& evaluations,
                                        const std::vector<int>& /*reduced_strategy_ids*/,
                                        const CostModel& cost, CLContext& /*ctx*/,
                                        CLIndicators& /*cl_ind*/, CLBacktest& /*cl_bt*/,
                                        PrecomputedOOSData& precomputed_oos,
                                        const SessionInfo*  sessions) {
    WFORunResult result;
    result.schedule           = schedule;
    result.selection_strategy = selection_strategy_name(selection);
    result.windows_skipped    = 0;

    // Collect all OOS trades across non-skipped windows for stitched metrics
    std::vector<Trade> all_oos_trades;

    // ---- Per-selection adaptive trigger state ----
    const bool adaptive               = (schedule.trigger != WFOTrigger::FIXED_INTERVAL);
    float      cumulative_oos_equity  = 0.0f;
    float      peak_oos_equity        = 0.0f;
    int        consecutive_losers     = 0;
    int        last_used_eval_idx     = -1;
    bool       trigger_fired_for_next = false;

    for (int wi = 0; wi < static_cast<int>(evaluations.size()); ++wi) {
        const WindowEvaluation& eval = evaluations[wi];

        // Decide which evaluation's frontier to use for this window
        int eff_idx;
        if (!adaptive || last_used_eval_idx < 0 || trigger_fired_for_next) {
            eff_idx = wi;
            if (adaptive && (trigger_fired_for_next || last_used_eval_idx < 0)) {
                cumulative_oos_equity = 0.0f;
                peak_oos_equity       = 0.0f;
                consecutive_losers    = 0;
            }
            trigger_fired_for_next = false;
        } else {
            eff_idx = last_used_eval_idx;
        }
        const WindowEvaluation& src = evaluations[eff_idx];

        WFOWindowResult wr;
        wr.window_index  = wi;
        wr.is_start_bar  = src.is_start_bar;
        wr.is_end_bar    = src.is_end_bar;
        wr.oos_start_bar = eval.oos_start_bar;
        wr.oos_end_bar   = eval.oos_end_bar;

        if (src.empty_frontier) {
            // Deployed frontier has no survivors -- skip window
            wr.skipped   = true;
            wr.archetype = "none";
            result.windows_skipped++;

            // Flat equity for this OOS segment
            for (int b = eval.oos_start_bar; b < eval.oos_end_bar && b < num_bars; ++b) {
                result.equity_curve.push_back(
                    result.equity_curve.empty() ? 0.0f : result.equity_curve.back());
                result.equity_timestamps.push_back(bars[b].timestamp);
                if (sessions) result.equity_session_days.push_back(sessions[b].session_day);
            }

            last_used_eval_idx = -1;  // nothing deployed: re-optimize next window
            result.windows.push_back(std::move(wr));
            continue;
        }

        // Apply selection strategy to the chosen frontier
        int winner_idx = select_winner(selection, src.is_results, src.frontier);

        if (winner_idx < 0) {
            // Selection found nothing (shouldn't happen if frontier is non-empty)
            wr.skipped   = true;
            wr.archetype = "none";
            result.windows_skipped++;

            for (int b = eval.oos_start_bar; b < eval.oos_end_bar && b < num_bars; ++b) {
                result.equity_curve.push_back(
                    result.equity_curve.empty() ? 0.0f : result.equity_curve.back());
                result.equity_timestamps.push_back(bars[b].timestamp);
                if (sessions) result.equity_session_days.push_back(sessions[b].session_day);
            }

            last_used_eval_idx = -1;  // nothing deployed: re-optimize next window
            result.windows.push_back(std::move(wr));
            continue;
        }

        wr.strategy_id = src.is_results[winner_idx].strategy_id;
        wr.is_result   = src.is_results[winner_idx];

        // Determine archetype from the frontier we selected from
        for (const auto& pe : src.frontier) {
            if (pe.original_index == winner_idx) {
                wr.archetype = archetype_name(pe.archetype);
                break;
            }
        }

        // ---------------------------------------------------------------
        // Use precomputed OOS indicators instead of computing from scratch
        // ---------------------------------------------------------------
        PrecomputedOOSWindow& pw                   = precomputed_oos.windows[wi];
        const bool            oos_indicators_valid = pw.valid;
        const int             oos_num_signals      = pw.num_signals;
        const int             oos_load_start       = pw.load_start;
        const int             oos_es               = pw.eval_start;
        const int             oos_ee               = pw.eval_end;

        // Get signal pointers from precomputed data (cached full array)
        auto& oos_sig_ptrs = pw.cached_signal_ptrs;

        const StrategyGrid& oos_sg = get_strategy_grid();

        // OOS uses SHIFTED base pointer (&bars[oos_load_start]
        // + relative bar indices). Sessions must shift the same way so that
        // run_impl's `sessions[bar_start + i]` lookup aligns with bars[].
        const SessionInfo* sessions_full = m_market_data ? m_market_data->sessions() : nullptr;
        const SessionInfo* sessions_shifted =
            sessions_full ? sessions_full + oos_load_start : nullptr;

        // OOS lag-1 validation (reusing precomputed indicators).
        // Use BacktestEngine directly (not cl_bt.evaluate) to capture the
        // trade log in one pass. The trades are reused for equity curve
        // stitching below, eliminating a redundant second backtest run.
        BacktestEngine            lag1_engine;
        std::vector<const float*> lag1_offset_ptrs(oos_num_signals);
        if (oos_indicators_valid && wr.strategy_id >= 0 && wr.strategy_id < oos_sg.total_count()) {
            StrategyGridEntry sge = oos_sg.entry_at(wr.strategy_id);
            StrategyConfig    sc  = sge.to_strategy_config();

            for (int i = 0; i < oos_num_signals; ++i) {
                lag1_offset_ptrs[i] = oos_sig_ptrs[i] + oos_es;
            }

            wr.oos_lag1 = lag1_engine.run(&bars[oos_load_start], sessions_shifted, oos_es, oos_ee,
                                          lag1_offset_ptrs.data(), oos_num_signals,
                                          nullptr /* time_gates */, sc, cost, 1);
            wr.oos_lag1.strategy_id = wr.strategy_id;
        }

        // OOS lag-2 validation: the same strategy entering one bar later.
        // A large Sharpe drop means the edge depends on next-bar fills, so
        // the lag-2 trades are decomposed for the report.
        if (oos_indicators_valid && wr.strategy_id >= 0 && wr.strategy_id < oos_sg.total_count()) {
            const StrategyConfig sc = oos_sg.entry_at(wr.strategy_id).to_strategy_config();
            BacktestEngine       lag2_engine;
            wr.oos_lag2 = lag2_engine.run(&bars[oos_load_start], sessions_shifted, oos_es, oos_ee,
                                          lag1_offset_ptrs.data(), oos_num_signals,
                                          nullptr /* time_gates */, sc, cost, 2);
            wr.oos_lag2.strategy_id = wr.strategy_id;

            if (wr.oos_lag1.sharpe_ratio > 0.0f &&
                wr.oos_lag2.sharpe_ratio < wr.oos_lag1.sharpe_ratio * 0.5f) {
                wr.pnl_decomp =
                    decompose_pnl(lag2_engine.last_trades(), &bars[oos_load_start], cost);
            }
        }

        // Stitch OOS equity curve from actual per-trade PnL (staircase).
        // Reuse the lag-1 trade log -- same strategy, lag, and data, so
        // trades are identical. Eliminates a redundant backtest run.
        float base_equity = result.equity_curve.empty() ? 0.0f : result.equity_curve.back();
        int   oos_bars    = eval.oos_end_bar - eval.oos_start_bar;
        if (oos_bars > 0) {
            std::vector<float> bar_pnl(oos_bars, 0.0f);

            {
                const auto& trades = lag1_engine.last_trades();
                for (const auto& t : trades) {
                    int oos_local = t.exit_bar - oos_es;
                    if (oos_local >= 0 && oos_local < oos_bars) {
                        bar_pnl[oos_local] += t.net_pnl;
                    }

                    Trade abs_trade     = t;
                    abs_trade.entry_bar = t.entry_bar + oos_load_start;
                    abs_trade.exit_bar  = t.exit_bar + oos_load_start;
                    all_oos_trades.push_back(abs_trade);
                }
            }

            // Build staircase equity from per-bar trade PnL
            for (int b = 0; b < oos_bars && (eval.oos_start_bar + b) < num_bars; ++b) {
                const int abs_bar = eval.oos_start_bar + b;
                base_equity += bar_pnl[b];
                result.equity_curve.push_back(base_equity);
                result.equity_timestamps.push_back(bars[abs_bar].timestamp);
                if (sessions) result.equity_session_days.push_back(sessions[abs_bar].session_day);
            }
        }

        // ---- Adaptive trigger evaluation ----
        // Watches THIS selection's deployed strategy's OOS performance.
        if (adaptive) {
            const auto& dep_trades = lag1_engine.last_trades();
            float       window_pnl = 0.0f;
            for (const auto& t : dep_trades) window_pnl += t.net_pnl;
            cumulative_oos_equity += window_pnl;
            if (cumulative_oos_equity > peak_oos_equity) {
                peak_oos_equity = cumulative_oos_equity;
            }

            if (schedule.trigger == WFOTrigger::DD_TRIGGER_5PCT ||
                schedule.trigger == WFOTrigger::DD_TRIGGER_10PCT) {
                float dd        = peak_oos_equity - cumulative_oos_equity;
                float threshold = schedule.trigger_dd_frac * cost.capital_required;
                if (dd >= threshold) trigger_fired_for_next = true;
            } else if (schedule.trigger == WFOTrigger::LOSERS_5 ||
                       schedule.trigger == WFOTrigger::LOSERS_10) {
                for (const auto& tr : dep_trades) {
                    if (tr.net_pnl < 0.0f)
                        consecutive_losers++;
                    else
                        consecutive_losers = 0;
                    if (consecutive_losers >= schedule.trigger_losers_n) {
                        trigger_fired_for_next = true;
                        break;
                    }
                }
            }
        }

        last_used_eval_idx = eff_idx;
        result.windows.push_back(std::move(wr));
    }

    // Viability: <50% windows skipped
    int total_windows = static_cast<int>(result.windows.size());
    if (total_windows > 0) {
        result.sufficient_windows = (result.windows_skipped * 2 < total_windows);
    }

    // Parameter stability: variance of chosen strategy IDs across non-skipped windows
    result.parameter_stability = compute_parameter_stability(result.windows);

    // Aggregate OOS metrics from stitched trades
    {
        int first_oos_start = 0;
        int last_oos_end    = 0;
        if (!evaluations.empty()) {
            first_oos_start = evaluations.front().oos_start_bar;
            last_oos_end    = evaluations.back().oos_end_bar;
            if (last_oos_end > num_bars) last_oos_end = num_bars;
        }

        if (!all_oos_trades.empty() && first_oos_start < last_oos_end) {
            std::sort(all_oos_trades.begin(), all_oos_trades.end(),
                      [](const Trade& a, const Trade& b) { return a.entry_bar < b.entry_bar; });

            result.oos_aggregate = compute_metrics(all_oos_trades, bars, first_oos_start,
                                                   last_oos_end, cost, sessions);
        }

        BacktestResult eq_metrics =
            compute_stitched_metrics(bars, first_oos_start, last_oos_end, result.equity_curve,
                                     result.equity_timestamps, result.equity_session_days, cost);

        result.oos_aggregate.sharpe_ratio = eq_metrics.sharpe_ratio;
        result.oos_aggregate.max_drawdown = eq_metrics.max_drawdown;

        if (result.oos_aggregate.max_drawdown > 0.0f && !result.equity_timestamps.empty()) {
            int64_t first_ts = result.equity_timestamps.front();
            int64_t last_ts  = result.equity_timestamps.back();
            double  duration_years =
                static_cast<double>(last_ts - first_ts) / (365.25 * 86400.0 * 1e9);
            if (duration_years > 0.0) {
                float annualized_pnl = static_cast<float>(
                    static_cast<double>(result.oos_aggregate.total_pnl) / duration_years);
                result.oos_aggregate.calmar_ratio =
                    annualized_pnl / result.oos_aggregate.max_drawdown;
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// run -- convenience method (evaluate windows + apply selection)
// ---------------------------------------------------------------------------
WFORunResult WFOEngine::run(const Bar* bars, int num_bars, const WFOConfig& schedule,
                            const std::string&      selection_strategy_name,
                            const std::vector<int>& reduced_strategy_ids, const CostModel& cost,
                            const SessionInfo* sessions) {
    // Map string to enum
    SelectionStrategy sel = SelectionStrategy::FRONTIER_CENTROID;
    for (int i = 0; i < SELECTION_STRATEGY_COUNT; ++i) {
        auto s = static_cast<SelectionStrategy>(i);
        if (selection_strategy_name == ::selection_strategy_name(s)) {
            sel = s;
            break;
        }
    }

    auto evaluations = evaluate_windows(bars, num_bars, schedule, reduced_strategy_ids, cost);
    return apply_selection(bars, num_bars, schedule, sel, evaluations, reduced_strategy_ids, cost,
                           sessions);
}

// ---------------------------------------------------------------------------
// compute_parameter_stability
//
// Fraction of consecutive OOS windows that selected the same strategy ID
// as the previous window.  1.0 = same strategy every window (maximally
// stable).  0.0 = a different strategy chosen every single time.
//
// NOTE: This deliberately avoids the coefficient of variation of ordinal
// strategy IDs, which is meaningless because adjacent IDs can
// map to completely different parameter sets.
// ---------------------------------------------------------------------------
float WFOEngine::compute_parameter_stability(const std::vector<WFOWindowResult>& windows) {
    std::vector<int> ids;
    for (const auto& w : windows) {
        if (!w.skipped && w.strategy_id >= 0) {
            ids.push_back(w.strategy_id);
        }
    }

    if (ids.size() < 2) return 1.0f;

    int matches = 0;
    for (size_t i = 1; i < ids.size(); ++i) {
        if (ids[i] == ids[i - 1]) {
            ++matches;
        }
    }

    return static_cast<float>(matches) / static_cast<float>(ids.size() - 1);
}

// ---------------------------------------------------------------------------
// compute_stitched_metrics -- compute Sharpe and max_drawdown from the
// stitched equity curve.
//
// The stitched equity curve includes flat segments for skipped windows,
// which the trade-based compute_metrics() doesn't know about. This method
// provides the equity-curve-derived overrides for those metrics.
// ---------------------------------------------------------------------------
BacktestResult WFOEngine::compute_stitched_metrics(const Bar* /*bars*/, int /*first_oos_start*/,
                                                   int /*last_oos_end*/,
                                                   const std::vector<float>&   equity_curve,
                                                   const std::vector<int64_t>& equity_timestamps,
                                                   const std::vector<int32_t>& equity_session_days,
                                                   const CostModel& /*cost*/) const {
    BacktestResult r;

    // Sharpe from stitched equity curve -- compute daily PnL by
    // bucketing into session days. When equity_session_days is
    // populated (sessions were passed in), use the DST-correct
    // session day index. Otherwise fall back to the approximate
    // fixed UTC timestamp shift.
    if (equity_curve.size() >= 2) {
        static constexpr int64_t NS_PER_SEC_WFO            = 1'000'000'000LL;
        static constexpr int64_t NS_PER_DAY_WFO            = 86400LL * NS_PER_SEC_WFO;
        static constexpr double  TRADING_DAYS_PER_YEAR_WFO = 252.0;

        const bool have_session_days =
            !equity_session_days.empty() && equity_session_days.size() == equity_curve.size();

        // Compute per-bar PnL delta from the stitched equity curve
        // then bucket into session days
        std::unordered_map<int64_t, float> daily_pnl;

        float prev_eq = 0.0f;
        if (have_session_days) {
            // DST-correct path
            for (size_t ei = 0; ei < equity_curve.size(); ++ei) {
                float bar_delta = equity_curve[ei] - prev_eq;
                prev_eq         = equity_curve[ei];

                if (bar_delta != 0.0f) {
                    int64_t day_key = static_cast<int64_t>(equity_session_days[ei]);
                    daily_pnl[day_key] += bar_delta;
                }
            }

            // Include all session days that have bars (0 PnL for no-trade days)
            {
                int32_t prev_day = -1;
                for (size_t ei = 0; ei < equity_session_days.size(); ++ei) {
                    int32_t sd = equity_session_days[ei];
                    if (sd != prev_day) {
                        int64_t day_key = static_cast<int64_t>(sd);
                        if (daily_pnl.find(day_key) == daily_pnl.end()) {
                            daily_pnl[day_key] = 0.0f;
                        }
                        prev_day = sd;
                    }
                }
            }
        } else {
            // Fallback: fixed UTC timestamp shift (approximate)
            static constexpr int64_t SESSION_SHIFT_NS = 22LL * 3600LL * NS_PER_SEC_WFO;

            for (size_t ei = 0; ei < equity_curve.size(); ++ei) {
                float bar_delta = equity_curve[ei] - prev_eq;
                prev_eq         = equity_curve[ei];

                if (bar_delta != 0.0f) {
                    int64_t day_key = (equity_timestamps[ei] - SESSION_SHIFT_NS) / NS_PER_DAY_WFO;
                    daily_pnl[day_key] += bar_delta;
                }
            }

            {
                int64_t prev_day = -1;
                for (size_t ei = 0; ei < equity_timestamps.size(); ++ei) {
                    int64_t day_key = (equity_timestamps[ei] - SESSION_SHIFT_NS) / NS_PER_DAY_WFO;
                    if (day_key != prev_day) {
                        if (daily_pnl.find(day_key) == daily_pnl.end()) {
                            daily_pnl[day_key] = 0.0f;
                        }
                        prev_day = day_key;
                    }
                }
            }
        }

        if (daily_pnl.size() >= 2) {
            const int n_days = static_cast<int>(daily_pnl.size());
            double    sum    = 0.0;
            for (const auto& [_, pnl] : daily_pnl) {
                sum += static_cast<double>(pnl);
            }
            const double mean = sum / n_days;

            double var = 0.0;
            for (const auto& [_, pnl] : daily_pnl) {
                const double diff = static_cast<double>(pnl) - mean;
                var += diff * diff;
            }
            var /= (n_days - 1);  // sample variance
            const double std_dev = std::sqrt(var);

            r.sharpe_ratio =
                (std_dev > 0.0)
                    ? static_cast<float>(mean / std_dev * std::sqrt(TRADING_DAYS_PER_YEAR_WFO))
                    : 0.0f;
        }
    }

    // Max drawdown from stitched equity. The curve starts from zero equity,
    // so a loss on the very first bar counts as drawdown too.
    if (!equity_curve.empty()) {
        float peak   = 0.0f;
        float max_dd = 0.0f;
        for (float eq : equity_curve) {
            if (eq > peak) peak = eq;
            float dd = peak - eq;
            if (dd > max_dd) max_dd = dd;
        }
        r.max_drawdown = max_dd;
    }

    return r;
}
