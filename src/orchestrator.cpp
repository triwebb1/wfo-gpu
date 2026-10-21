#include "orchestrator.h"
#include "common.h"
#include "durable_file.h"
#include "results.h"
#include "results_serde.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "compute/indicator_warmup.h"
#include "indicators/signal_normalize.h"
#include "resource_monitor.h"

// ---------------------------------------------------------------------------
// Progress helper -- prints "  42% (1234 done) ETA 3m 12s"
// ---------------------------------------------------------------------------
static std::string format_eta(double elapsed_sec, double fraction_done) {
    if (fraction_done <= 0.0 || elapsed_sec <= 0.0) return "estimating...";
    double total_est = elapsed_sec / fraction_done;
    double remain    = total_est - elapsed_sec;
    if (remain < 0.0) remain = 0.0;
    int sec = static_cast<int>(remain);
    if (sec < 60) return std::to_string(sec) + "s";
    int min = sec / 60;
    sec %= 60;
    if (min < 60) return std::to_string(min) + "m " + std::to_string(sec) + "s";
    int hr = min / 60;
    min %= 60;
    return std::to_string(hr) + "h " + std::to_string(min) + "m";
}

using SteadyClock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// refine_with_cpu_result
//
// GPU results carry the kernel's accumulators only: no Sharpe, no weekly,
// monthly or yearly consistency metrics, and no calendar-time return. The
// screen re-runs every result a later decision reads on the CPU engine and
// takes the CPU result wholesale, so those decisions are the ones a CPU-only
// run makes. Counters that both engines produce are compared first; a
// mismatch is a parity bug and is reported.
// ---------------------------------------------------------------------------
static void refine_with_cpu_result(BacktestResult& target, BacktestResult cpu_r) {
    auto approx_eq = [](float a, float b) {
        const float diff = std::fabs(a - b);
        return diff <= 0.01f || diff <= 0.01f * std::max(std::fabs(a), std::fabs(b));
    };
    if (target.total_trades != cpu_r.total_trades ||
        !approx_eq(target.total_pnl, cpu_r.total_pnl) ||
        !approx_eq(target.max_drawdown, cpu_r.max_drawdown)) {
        std::cerr << "[WARN] GPU/CPU parity drift for strategy " << target.strategy_id
                  << ": trades=" << target.total_trades << "/" << cpu_r.total_trades
                  << " pnl=" << target.total_pnl << "/" << cpu_r.total_pnl << std::endl;
    }
    cpu_r.strategy_id = target.strategy_id;
    target            = cpu_r;
}

// Landscape analysis works in result positions; downstream stages need the
// strategy-grid ids of those results. They differ whenever the screen skips
// grid entries (redundant combinations, truncated grids), so convert here.
static std::vector<int> grid_ids_of(const std::vector<BacktestResult>& results,
                                    const std::vector<int>&            positions) {
    std::vector<int> ids;
    ids.reserve(positions.size());
    for (int pos : positions) {
        if (pos >= 0 && pos < static_cast<int>(results.size()))
            ids.push_back(results[pos].strategy_id);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

// ---------------------------------------------------------------------------
// run_screen -- full pipeline with file I/O
// ---------------------------------------------------------------------------

Orchestrator::ScreenResult Orchestrator::run_screen(const ScreenConfig& config) {
    // Load bar data
    MarketData market_data;
    if (!market_data.load(config.bar_file, config.session)) {
        std::cerr << "Orchestrator: Failed to load bar file: " << config.bar_file << std::endl;
        return ScreenResult{};
    }

    int total_bars = market_data.bar_count();
    int scan_start = config.scan_start_bar;
    int scan_end   = config.scan_end_bar;

    if (scan_start < 0) scan_start = 0;
    if (scan_end <= 0 || scan_end > total_bars) scan_end = total_bars;
    if (scan_start >= scan_end) {
        std::cerr << "Orchestrator: Invalid scan range [" << scan_start << ", " << scan_end << ")"
                  << std::endl;
        return ScreenResult{};
    }

    std::cout << "Scan range: bars [" << scan_start << ", " << scan_end
              << ") = " << (scan_end - scan_start) << " bars" << std::endl;

    // Build indicator grid
    IndicatorGrid indicator_grid;

    // Compute warmup requirement
    int max_warmup = 0;
    for (int i = 0; i < indicator_grid.size(); ++i) {
        IndicatorSpec spec = IndicatorSpec::from_params(indicator_grid.params(i), 0);
        max_warmup         = std::max(max_warmup, spec.warmup_bars);
    }
    max_warmup *= 2;  // safety factor on indicator warmup

    // Extend bar window backward by warmup
    int load_start = std::max(0, scan_start - max_warmup);
    int load_count = scan_end - load_start;

    // eval_start/eval_end are relative to load_start
    int eval_start = scan_start - load_start;
    int eval_end   = scan_end - load_start;

    const Bar* bars = &market_data.bar(load_start);

    // Pass session info so screen backtests respect trading hours and halts.
    // IMPORTANT: sessions pointer must be aligned with bars pointer -- both start
    // at load_start so that sessions[i] corresponds to bars[i].
    const SessionInfo* sessions = (total_bars > 0) ? &market_data.session(load_start) : nullptr;

    // Stash progress_cb so run_screen_with_data /
    // evaluate_strategies can dispatch to it at batch boundaries.
    m_screen_progress_cb = config.progress_cb;

    // Run the screen with pre-loaded data (pass MarketData for session signals)
    ScreenResult result = run_screen_with_data(bars, load_count, eval_start, eval_end,
                                               indicator_grid, config, sessions, &market_data);

    // Clear the stash so it doesn't leak across runs.
    m_screen_progress_cb = nullptr;

    // Write results if output_dir is specified
    if (!config.output_dir.empty()) {
        result.output_written = write_results(config.output_dir, result);
    }

    return result;
}

// ---------------------------------------------------------------------------
// run_screen_with_data -- core pipeline, no file I/O
// ---------------------------------------------------------------------------

Orchestrator::ScreenResult Orchestrator::run_screen_with_data(const Bar* bars, int num_bars,
                                                              int eval_start, int eval_end,
                                                              const IndicatorGrid& indicator_grid,
                                                              const ScreenConfig&  config,
                                                              const SessionInfo*   sessions,
                                                              const MarketData*    md) {
    ScreenResult result;

    if (!bars || num_bars <= 0 || eval_start >= eval_end || eval_end > num_bars) {
        std::cerr << "Orchestrator: Invalid data range (bars=" << num_bars << ", eval=["
                  << eval_start << "," << eval_end << "))" << std::endl;
        return result;
    }

    std::cout << "Computing indicators on " << num_bars << " bars..." << std::flush;

    // Initialize OpenCL context (CPU fallback if not available)
    CLContext ctx;
    ctx.initialize(CLContext::default_device());
    // Surface device selection so silent CPU fallback is visible.
    if (ctx.is_available()) {
        std::cerr << "[INFO] OpenCL device selected (screen): " << ctx.device_name() << std::endl;
    } else {
        std::cerr << "[INFO] No OpenCL device in use (screen); "
                     "running the native CPU path"
                  << std::endl;
    }

    CLMemory     memory;
    CLIndicators cl_ind;

    // Compute every indicator over the whole range in one pass: the screen
    // needs them aligned bar for bar. CLMemory falls back to host buffers
    // when the device cannot hold them.
    std::vector<float*>        indicator_ptrs;
    std::vector<IndicatorSpec> indicator_specs;
    compute_indicators(ctx, bars, 0, num_bars, indicator_grid, memory, cl_ind, indicator_ptrs,
                       indicator_specs);
    const int num_indicator_series = static_cast<int>(indicator_ptrs.size());

    std::cout << " done (" << num_indicator_series << " series)." << std::endl;

    // Normalize raw indicator series to [-1, +1] signal arrays.
    // Each indicator in the grid produces ONE signal (not one per raw series).
    // This puts RSI (0-100), EMA price levels, etc. on a common scale.
    int                             num_technical_signals = indicator_grid.size();
    std::vector<std::vector<float>> signal_storage(num_technical_signals);
    std::vector<float*>             signal_ptrs(num_technical_signals);
    for (int i = 0; i < num_technical_signals; ++i) {
        signal_storage[i].resize(num_bars);
        signal_ptrs[i] = signal_storage[i].data();
    }
    normalize_all_indicators(indicator_grid, indicator_ptrs, bars, num_bars, signal_ptrs);

    // Compute session signals and append to signal arrays.
    // Session signals are already normalized to [-1,+1] and don't need
    // normalize_to_signal(). They are broadcast from per-day to per-bar
    // with time-gating via broadcast_session_signal_to_bars().
    int num_session_signals = 0;
    if (md && sessions && md->session_model().session_signals) {
        // bars may start part-way into md; the series cover the same bars.
        const int abs_bar_start = static_cast<int>(bars - &md->bar(0));
        num_session_signals     = append_session_signal_series(
            *md, SessionSignalGrid(), abs_bar_start, abs_bar_start + num_bars, signal_storage);
    }

    // Rebuild signal_ptrs from signal_storage to avoid stale pointers
    // (push_back to signal_storage may have reallocated despite the reserve)
    int num_signals = num_technical_signals + num_session_signals;
    signal_ptrs.resize(num_signals);
    for (int i = 0; i < num_signals; ++i) {
        signal_ptrs[i] = signal_storage[i].data();
    }

    // Evaluate all strategy combos using normalized signals.
    // Early termination is opt-in. It cuts losing strategies short on the
    // CPU path; CLBacktest runs early-terminated batches on the CPU even when
    // a GPU is present, so a given config produces the same survivors on
    // every device.
    EarlyTermination screen_early_term;
    screen_early_term.enabled = config.early_termination;

    std::cout << "Evaluating strategies (" << num_signals << " signals, eval range [" << eval_start
              << "," << eval_end << "))..." << std::endl;

    evaluate_strategies(ctx, bars, 0, num_bars, eval_start, eval_end,
                        const_cast<const float* const*>(signal_ptrs.data()), num_signals,
                        indicator_grid, config, result.all_results, screen_early_term, sessions);

    result.total_evaluated = static_cast<int>(result.all_results.size());

    if (result.all_results.empty()) return result;

    // GPU results carry raw accumulators only; consistency metrics (green
    // weeks and months, red-week streaks, negative years) and Sharpe need
    // the CPU engine. Two passes: the GPU sweeps
    // the whole grid, the CPU refines every result a later decision reads.
    // Frontier membership and robustness neighbors only ever use viable
    // results, so refining all viable results plus the floor survivors makes
    // every decision identical to a CPU-only run.
    if (ctx.is_available()) {
        std::vector<int> floor_survivors =
            apply_sanity_floors(result.all_results, config.sanity_floors);
        {
            std::vector<char> pick(result.all_results.size(), 0);
            for (int i : floor_survivors) pick[i] = 1;
            for (size_t i = 0; i < result.all_results.size(); ++i)
                if (result.all_results[i].viable) pick[i] = 1;
            floor_survivors.clear();
            for (size_t i = 0; i < pick.size(); ++i)
                if (pick[i]) floor_survivors.push_back(static_cast<int>(i));
        }

        if (!floor_survivors.empty()) {
            std::cout << "Re-evaluating " << floor_survivors.size()
                      << " viable or floor-passing strategies on CPU for full metrics..."
                      << std::flush;
            auto t_cpu = SteadyClock::now();

            // Build strategy grid to decode strategy_ids back to configs
            StrategyGrid strategy_grid(num_signals);

            // Thread-parallel CPU re-evaluation
            auto reeval_range = [&](int range_start, int range_end) {
                BacktestEngine            engine;
                std::vector<const float*> offset_ptrs_local(num_signals);
                for (int k = 0; k < num_signals; ++k)
                    offset_ptrs_local[k] = signal_ptrs[k] + eval_start;

                for (int ri = range_start; ri < range_end; ++ri) {
                    int result_idx  = floor_survivors[ri];
                    int strategy_id = result.all_results[result_idx].strategy_id;
                    if (strategy_id < 0 || strategy_id >= strategy_grid.total_count()) continue;

                    StrategyGridEntry entry = strategy_grid.entry_at(strategy_id);
                    StrategyConfig    sc    = entry.to_strategy_config();

                    BacktestResult cpu_r;
                    if (sessions) {
                        cpu_r = engine.run(bars, sessions, eval_start, eval_end,
                                           offset_ptrs_local.data(), num_signals, nullptr, sc,
                                           config.cost, config.entry_lag);
                    } else {
                        cpu_r = engine.run(bars, eval_start, eval_end, offset_ptrs_local.data(),
                                           num_signals, sc, config.cost, config.entry_lag);
                    }
                    refine_with_cpu_result(result.all_results[result_idx], cpu_r);
                }
            };

            int             num_survivors = static_cast<int>(floor_survivors.size());
            ResourceMonitor rm;
            int             max_threads = rm.max_threads();
            if (max_threads <= 1 || num_survivors < max_threads * 4) {
                reeval_range(0, num_survivors);
            } else {
                int                        chunk = (num_survivors + max_threads - 1) / max_threads;
                std::vector<JoiningThread> threads;
                for (int t = 0; t < max_threads; ++t) {
                    int start = t * chunk;
                    int end   = std::min(start + chunk, num_survivors);
                    if (start >= end) break;
                    threads.emplace_back([&, start, end]() { reeval_range(start, end); });
                }
            }

            double elapsed = std::chrono::duration<double>(SteadyClock::now() - t_cpu).count();
            std::cout << " done (" << static_cast<int>(elapsed) << "s)" << std::endl;
        }
    }

    // Landscape analysis
    result.num_signals   = num_signals;
    NeighborFunctions nf = build_neighbor_functions(result.all_results, num_signals);

    result.landscape =
        analyze_landscape(result.all_results, config.robustness_floor, nf.for_robustness,
                          nf.num_param_axes, nf.for_reduced_set, config.sanity_floors);

    result.passed_floors        = result.landscape.passed_sanity_floors;
    result.on_frontier          = result.landscape.on_pareto_frontier;
    result.reduced_set_size     = static_cast<int>(result.landscape.reduced_set_ids.size());
    result.reduced_strategy_ids = grid_ids_of(result.all_results, result.landscape.reduced_set_ids);
    result.grinder_count        = result.landscape.grinder_count;
    result.sniper_count         = result.landscape.sniper_count;
    result.balanced_count       = result.landscape.balanced_count;

    return result;
}

// ---------------------------------------------------------------------------
// reanalyze_results -- re-derive landscape from existing backtest results
// Used for checkpoint resume when results are loaded but analysis isn't stored.
// ---------------------------------------------------------------------------

Orchestrator::ScreenResult Orchestrator::reanalyze_results(std::vector<BacktestResult> results,
                                                           float               robustness_floor,
                                                           const SanityFloors& floors,
                                                           int                 num_signals) const {
    ScreenResult out;
    out.all_results     = std::move(results);
    out.total_evaluated = static_cast<int>(out.all_results.size());

    if (out.all_results.empty()) return out;

    auto nf       = build_neighbor_functions(out.all_results, num_signals);
    out.landscape = analyze_landscape(out.all_results, robustness_floor, nf.for_robustness,
                                      nf.num_param_axes, nf.for_reduced_set, floors);

    out.passed_floors        = out.landscape.passed_sanity_floors;
    out.on_frontier          = out.landscape.on_pareto_frontier;
    out.reduced_set_size     = static_cast<int>(out.landscape.reduced_set_ids.size());
    out.reduced_strategy_ids = grid_ids_of(out.all_results, out.landscape.reduced_set_ids);
    out.grinder_count        = out.landscape.grinder_count;
    out.sniper_count         = out.landscape.sniper_count;
    out.balanced_count       = out.landscape.balanced_count;

    return out;
}

// ---------------------------------------------------------------------------
// compute_indicators
// ---------------------------------------------------------------------------

void Orchestrator::compute_indicators(const CLContext& ctx, const Bar* bars, int bar_offset,
                                      int num_bars, const IndicatorGrid& indicator_grid,
                                      CLMemory& memory, CLIndicators& cl_ind,
                                      std::vector<float*>&        indicator_ptrs,
                                      std::vector<IndicatorSpec>& indicator_specs) {
    (void)bar_offset;  // bars already offset by caller

    // Build indicator specs from the grid.
    // Each indicator grid entry may produce multiple series.
    indicator_specs.clear();
    for (int i = 0; i < indicator_grid.size(); ++i) {
        const IndicatorParams& params = indicator_grid.params(i);
        int                    ns     = indicator_num_series(params);

        for (int s = 0; s < ns; ++s) {
            indicator_specs.push_back(IndicatorSpec::from_params(params, s));
        }
    }

    // Allocate output buffers
    int total_series = static_cast<int>(indicator_specs.size());
    indicator_ptrs   = memory.allocate_indicator_buffers(ctx, total_series, num_bars);

    // Convert bars to SoA for the compute path
    memory.upload_bars(ctx, bars, 0, num_bars);

    const SoABarData& soa = memory.soa_bars();

    // Compute all indicators (with full Bar data for VWAP/OBV/PivotPoints)
    cl_ind.compute_with_bars(ctx, bars, 0, soa.open.data(), soa.high.data(), soa.low.data(),
                             soa.close.data(), num_bars, indicator_specs, indicator_ptrs);
}

// ---------------------------------------------------------------------------
// evaluate_strategies
// ---------------------------------------------------------------------------

void Orchestrator::evaluate_strategies(
    const CLContext& ctx, const Bar* bars, int bar_offset, int num_bars, int eval_start,
    int eval_end, const float* const* indicator_series, int num_indicator_series,
    const IndicatorGrid& /*indicator_grid*/, const ScreenConfig& config,
    std::vector<BacktestResult>& results, const EarlyTermination& early_term,
    const SessionInfo* sessions) {
    // Build WindowModeGrid from indicator grid
    // For the screen on CPU, we skip the window mode complexity and
    // use rolling mode for all indicators (series already computed above).
    // The strategy grid operates on the indicator series directly.

    // Determine the number of available indicator series
    int n_series = num_indicator_series;

    // Build a strategy grid over the available indicator series.
    // For testing with small grids, n_series may be very small.
    StrategyGrid strategy_grid(n_series);

    // The budget counts distinct strategies evaluated; redundant grid
    // entries are skipped without counting. Strategy ids are ints, which
    // bounds how far into the grid a screen can reach.
    const int64_t grid_size =
        std::min<int64_t>(strategy_grid.total_count(), std::numeric_limits<int>::max());
    const int64_t total_strategies = (config.max_strategies > 0)
                                         ? std::min<int64_t>(config.max_strategies, grid_size)
                                         : grid_size;

    std::cout << "Strategy grid: " << strategy_grid.total_count()
              << " combinations; evaluating up to " << total_strategies << " distinct strategies"
              << std::endl;

    // Evaluate strategies in batches for memory efficiency
    const int64_t BATCH_SIZE = 1024;
    results.clear();
    results.reserve(static_cast<size_t>(std::min<int64_t>(total_strategies, 1 << 24)));

    CLBacktest cl_bt;
    int64_t    last_progress_pct = -1;
    auto       t_start           = SteadyClock::now();
    int64_t    next_index        = 0;

    // The screen walks the grid in order, so a partial checkpoint resumes
    // just after the last grid entry it holds.
    if (!config.resumed_results.empty()) {
        results    = config.resumed_results;
        next_index = static_cast<int64_t>(results.back().strategy_id) + 1;
    }

    while (static_cast<int64_t>(results.size()) < total_strategies && next_index < grid_size) {
        // Print progress every 5%
        const int64_t done_count = static_cast<int64_t>(results.size());
        int64_t       pct        = done_count * 100 / total_strategies;
        if (pct / 5 != last_progress_pct / 5) {
            last_progress_pct = pct;
            double elapsed    = std::chrono::duration<double>(SteadyClock::now() - t_start).count();
            double frac = static_cast<double>(done_count) / static_cast<double>(total_strategies);
            std::cout << "  " << pct << "% (" << results.size() << " evaluated)  ETA "
                      << format_eta(elapsed, frac) << std::endl;
        }
        const int64_t want = std::min(BATCH_SIZE, total_strategies - done_count);

        // Build strategy configs for this batch, tracking original grid indices
        std::vector<StrategyConfig> strat_configs;
        std::vector<int64_t>        grid_indices;  // original index per non-redundant entry
        strat_configs.reserve(static_cast<size_t>(want));
        grid_indices.reserve(static_cast<size_t>(want));

        for (; next_index < grid_size && static_cast<int64_t>(strat_configs.size()) < want;
             ++next_index) {
            StrategyGridEntry entry = strategy_grid.entry_at(next_index);
            if (entry.is_redundant()) continue;  // skip duplicate/useless strategies
            strat_configs.push_back(entry.to_strategy_config());
            grid_indices.push_back(next_index);
        }

        if (strat_configs.empty()) break;  // grid exhausted

        // Evaluate the batch
        std::vector<BacktestResult> batch_results;
        cl_bt.evaluate(ctx, bars, bar_offset, indicator_series, num_indicator_series, num_bars,
                       eval_start, eval_end, strat_configs, config.cost, config.entry_lag,
                       batch_results, early_term, sessions);

        // Assign global strategy IDs from original grid indices
        for (size_t i = 0; i < batch_results.size(); ++i) {
            batch_results[i].strategy_id = static_cast<int>(grid_indices[i]);
        }

        results.insert(results.end(), batch_results.begin(), batch_results.end());

        // Progress dispatch once per batch; the callback throttles itself.
        // Synchronous contract -- cb must not defer to another thread.
        if (m_screen_progress_cb) {
            const int done  = static_cast<int>(results.size());
            const int total = static_cast<int>(total_strategies);
            m_screen_progress_cb(done, total, results);
        }
    }
}

// ---------------------------------------------------------------------------
// build_neighbor_functions
//
// Builds per-axis +/-1-step neighbor lookup tables for robustness scoring using
// the real 6-axis strategy-grid layout returned by StrategyGrid::axis_descriptors
// (role > hours > exit > thresh > agg > weight_rank). Strategy ids are decoded
// per-k via strategy_id_to_k, then each axis's stride is applied in flat index
// space. This is not a flat +/-1 heuristic.
// ---------------------------------------------------------------------------

int Orchestrator::screen_signal_count(const SessionModel& session) {
    const int technical = IndicatorGrid().size();
    return session.session_signals ? technical + SessionSignalGrid().total_signals() : technical;
}

Orchestrator::NeighborFunctions Orchestrator::build_neighbor_functions(
    const std::vector<BacktestResult>& results, int num_signals) const {
    NeighborFunctions nf;
    const int         num_results = static_cast<int>(results.size());

    // Neighbor geometry lives in StrategyGrid::neighbors(); the lambdas below
    // only map grid ids to result indices.
    nf.num_param_axes = 6;

    // The grid size must be the one the screen enumerated. When the caller
    // does not know it, fall back to the smallest grid that contains every
    // evaluated id (exact only when the screen covered the whole grid).
    int n_indicators = num_signals;
    if (n_indicators <= 0) {
        int max_sid = -1;
        for (const auto& r : results) max_sid = std::max(max_sid, r.strategy_id);
        for (int n = 1; n <= 4096 && max_sid >= 0; ++n) {
            if (StrategyGrid(n).total_count() > static_cast<int64_t>(max_sid)) {
                n_indicators = n;
                break;
            }
        }
    }

    auto grid = std::make_shared<StrategyGrid>(std::max(n_indicators, 1));

    // Build grid_id -> result_idx lookup (once, shared by both lambdas)
    auto grid_to_result = std::make_shared<std::unordered_map<int, int>>();
    grid_to_result->reserve(num_results);
    for (int i = 0; i < num_results; ++i) {
        (*grid_to_result)[results[i].strategy_id] = i;
    }

    // Visit the evaluated one-step neighbors of a result: (axis, result index).
    auto for_each_neighbor = [&results, grid, grid_to_result, num_results](int         idx,
                                                                           const auto& visit) {
        if (idx < 0 || idx >= num_results) return;
        for (const auto& [axis, nb_id] : grid->neighbors(results[idx].strategy_id)) {
            if (nb_id > std::numeric_limits<int>::max()) continue;  // never screened
            auto it = grid_to_result->find(static_cast<int>(nb_id));
            if (it != grid_to_result->end()) visit(axis, it->second);
        }
    };

    nf.for_robustness = [for_each_neighbor](int idx) {
        std::vector<std::pair<int, int>> neighbors;
        for_each_neighbor(idx, [&](int axis, int nb) { neighbors.push_back({axis, nb}); });
        return neighbors;
    };
    nf.for_reduced_set = [for_each_neighbor](int idx) {
        std::vector<int> neighbors;
        for_each_neighbor(idx, [&](int, int nb) { neighbors.push_back(nb); });
        return neighbors;
    };

    return nf;
}

// ---------------------------------------------------------------------------
// write_results -- serialize screen output to disk
// ---------------------------------------------------------------------------

bool Orchestrator::write_results(const std::string& output_dir, const ScreenResult& result) {
    // Ensure output directory exists (best effort)
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);

    // Pre-write free-disk check. A late-stage disk-full
    // failure would lose the final landscape summary; warn early so the
    // user can reroute output or free space.
    {
        std::error_code disk_ec;
        auto            si = std::filesystem::space(output_dir, disk_ec);
        if (!disk_ec) {
            const int64_t one_gb = 1LL << 30;
            if (static_cast<int64_t>(si.available) < one_gb) {
                std::cerr << "[WARN] free disk space at " << output_dir << " is "
                          << (si.available / (1024 * 1024))
                          << " MB (< 1 GB); "
                             "screen results write may fail"
                          << std::endl;
            }
        }
    }

    bool success = true;

    // All 3 output files are routed through DurableFileWriter
    // (Win32 atomic rename + FlushFileBuffers). On any Win32 failure the
    // writer throws, the .tmp file is unlinked, and success is set false.

    // 1. Write reduced parameter set (binary)
    try {
        std::string path = output_dir + "/reduced_params.bin";
        DurableFileWriter::atomic_write(path, [&](DurableFileWriter& w) {
            int32_t count = static_cast<int32_t>(result.reduced_strategy_ids.size());
            w.write_bytes(&count, sizeof(count));
            if (!result.reduced_strategy_ids.empty()) {
                w.write_bytes(result.reduced_strategy_ids.data(),
                              static_cast<size_t>(count) * sizeof(int32_t));
            }
        });
    } catch (const std::exception& e) {
        std::cerr << "Orchestrator: reduced_params.bin: " << e.what() << std::endl;
        success = false;
    }

    // 2. Write landscape summary (CSV)
    try {
        std::string path = output_dir + "/landscape_summary.csv";
        // Build CSV into an in-memory string first so DurableFileWriter only
        // needs to stream one write. CSV is small (< 1 KB) so this is trivial.
        std::string csv;
        csv.reserve(2048);
        auto append_line = [&](const std::string& key, auto val) {
            csv += key;
            csv += ',';
            csv += std::to_string(val);
            csv += '\n';
        };
        csv += "metric,value\n";
        append_line("total_evaluated", result.total_evaluated);
        append_line("passed_floors", result.passed_floors);
        append_line("on_frontier", result.on_frontier);
        append_line("reduced_set_size", result.reduced_set_size);
        append_line("grinder_count", result.grinder_count);
        append_line("sniper_count", result.sniper_count);
        append_line("balanced_count", result.balanced_count);
        const auto& la = result.landscape;
        append_line("roc_median", la.roc_stats.median);
        append_line("roc_p25", la.roc_stats.p25);
        append_line("roc_p75", la.roc_stats.p75);
        append_line("green_weeks_median", la.green_weeks_stats.median);
        append_line("robustness_median", la.robustness_stats.median);
        append_line("profit_factor_min", la.profit_factor_stats.min);
        append_line("profit_factor_p25", la.profit_factor_stats.p25);
        append_line("profit_factor_median", la.profit_factor_stats.median);
        append_line("profit_factor_p75", la.profit_factor_stats.p75);
        append_line("profit_factor_max", la.profit_factor_stats.max);
        append_line("profit_factor_mean", la.profit_factor_stats.mean);
        append_line("trr_min", la.trr_stats.min);
        append_line("trr_p25", la.trr_stats.p25);
        append_line("trr_median", la.trr_stats.median);
        append_line("trr_p75", la.trr_stats.p75);
        append_line("trr_max", la.trr_stats.max);
        append_line("trr_mean", la.trr_stats.mean);

        DurableFileWriter::atomic_write(
            path, [&](DurableFileWriter& w) { w.write_bytes(csv.data(), csv.size()); });
    } catch (const std::exception& e) {
        std::cerr << "Orchestrator: landscape_summary.csv: " << e.what() << std::endl;
        success = false;
    }

    // 3. Write all results (binary, for debugging/analysis)
    //
    // Length-prefixed, field-by-field records (not a raw struct memcpy) via
    // results_serde::write_all_results_file. Format: magic "WFAR" + version
    // uint32_t + count int32_t + count * field-by-field record.
    try {
        std::string path = output_dir + "/all_results.bin";
        DurableFileWriter::atomic_write(path, [&](DurableFileWriter& w) {
            results_serde::write_all_results_file(w, result.all_results);
        });
    } catch (const std::exception& e) {
        std::cerr << "Orchestrator: all_results.bin: " << e.what() << std::endl;
        success = false;
    }

    return success;
}

// ---------------------------------------------------------------------------
// run_walk_forward -- full pipeline with file I/O
// ---------------------------------------------------------------------------

Orchestrator::WalkForwardResult Orchestrator::run_walk_forward(const WalkForwardConfig& config) {
    // Load bar data
    MarketData market_data;
    if (!market_data.load(config.bar_file, config.session)) {
        std::cerr << "Orchestrator: Failed to load bar file: " << config.bar_file << std::endl;
        return WalkForwardResult{};
    }

    int        total_bars = market_data.bar_count();
    const Bar* bars       = &market_data.bar(0);

    // Pass session info for DST-correct daily PnL bucketing (Sharpe ratio)
    const SessionInfo* sessions = (total_bars > 0) ? &market_data.session(0) : nullptr;

    // Thread config.progress_cb through the inner schedule
    // loop via a member stash (avoids expanding run_walk_forward_with_data's
    // test-friendly signature). Cleared after the call.
    m_walk_forward_progress_cb = config.progress_cb;
    m_oos_start_bar            = config.oos_start_bar;
    m_resumed_runs             = &config.resumed_runs;

    const float bars_per_day = config.bars_per_day > 0.0f
                                   ? config.bars_per_day
                                   : static_cast<float>(market_data.bars_per_calendar_day());

    WalkForwardResult result = run_walk_forward_with_data(
        bars, total_bars, config.reduced_strategy_ids, config.cost, bars_per_day,
        config.reduced_indicator_ids, sessions, config.start_schedule_idx, &market_data);

    m_walk_forward_progress_cb = nullptr;
    m_oos_start_bar            = 0;
    m_resumed_runs             = nullptr;

    // Write results if output_dir is specified
    if (!config.output_dir.empty()) {
        result.output_written = results::write_walk_forward_results(
            config.output_dir, result.all_runs, result.best_selection_strategy,
            result.best_selection_avg_oos_roc);
    }

    return result;
}

// ---------------------------------------------------------------------------
// run_walk_forward_with_data -- core walk-forward pipeline, no file I/O
//
// Iterates every WFO config (each schedule x each selection strategy).
// Shares backtest evaluations across selection strategies per schedule.
// ---------------------------------------------------------------------------

Orchestrator::WalkForwardResult Orchestrator::run_walk_forward_with_data(
    const Bar* bars, int num_bars, const std::vector<int>& reduced_strategy_ids,
    const CostModel& cost, float bars_per_day, const std::vector<int>& reduced_indicator_ids,
    const SessionInfo* sessions, int start_schedule_idx, const MarketData* md) {
    WalkForwardResult result;

    if (!bars || num_bars <= 0 || reduced_strategy_ids.empty()) {
        return result;
    }

    // Runs restored from a checkpoint come first, exactly as they were.
    if (m_resumed_runs) result.all_runs = *m_resumed_runs;

    WFOGrid   wfo_grid;
    WFOEngine engine;
    engine.set_min_oos_start(m_oos_start_bar);

    // Set MarketData for session signal computation (when available)
    if (md) {
        engine.set_market_data(md);
    }

    // Walk-forward optimization: compute only the indicators actually used by
    // the reduced strategy set. If reduced_indicator_ids is provided, use it
    // directly. Otherwise, extract indicator IDs from the strategy set.
    std::vector<int> active_indicator_ids = reduced_indicator_ids;
    if (active_indicator_ids.empty()) {
        // Extract indicator IDs from all reduced strategies using the
        // engine's cached grids to avoid redundant construction.
        // Use the engine's strategy grid which includes session signals
        // (if MarketData was set) for correct strategy ID decoding.
        const StrategyGrid&     strategy_grid = engine.strategy_grid();
        std::unordered_set<int> indicator_set;
        for (int sid : reduced_strategy_ids) {
            if (sid >= 0 && sid < strategy_grid.total_count()) {
                StrategyGridEntry e = strategy_grid.entry_at(sid);
                for (int k = 0; k < e.num_indicators; ++k) {
                    // Only technical indicators need CL computation;
                    // session signals are computed separately
                    if (e.indicator_ids[k] < engine.indicator_grid().size()) {
                        indicator_set.insert(e.indicator_ids[k]);
                    }
                }
            }
        }
        active_indicator_ids.assign(indicator_set.begin(), indicator_set.end());
        // Sort for reproducibility -- hash-seed-independent iteration order.
        std::sort(active_indicator_ids.begin(), active_indicator_ids.end());
    }

    if (!active_indicator_ids.empty()) {
        engine.set_reduced_indicator_ids(active_indicator_ids);
    }

    // Create shared CL instances ONCE for the entire walk-forward run, so
    // device discovery and program compilation are not repeated by every
    // evaluate_windows() and apply_selection() call.
    CLContext shared_ctx;
    shared_ctx.initialize(CLContext::default_device());
    // Surface device selection for the walk-forward shared context.
    if (shared_ctx.is_available()) {
        std::cerr << "[INFO] OpenCL device selected (walk-forward): " << shared_ctx.device_name()
                  << std::endl;
    } else {
        std::cerr << "[INFO] No OpenCL device in use (walk-forward); "
                     "running the native CPU path"
                  << std::endl;
    }
    CLIndicators shared_cl_ind;
    CLBacktest   shared_cl_bt;

    // Precompute all reduced-set indicators + session signals once over the
    // full bar dataset.  After this, every per-window compute_signals_for_range()
    // call returns slices into this data instead of recomputing.
    engine.precompute_all_signals(bars, num_bars, shared_ctx, shared_cl_ind);

    // For each WFO schedule config, evaluate windows once, then apply all
    // selection strategies as post-processing.
    // Skip schedules already completed during a previous run (resume support).
    int  total_schedules = wfo_grid.size();
    auto t_walk_forward  = SteadyClock::now();
    std::cout << "Walk-forward: " << (total_schedules - start_schedule_idx)
              << " schedules to evaluate" << std::endl;

    for (int si = start_schedule_idx; si < total_schedules; ++si) {
        const WFOScheduleConfig& sc  = wfo_grid.config(si);
        WFOConfig                cfg = WFOConfig::from_schedule(sc, bars_per_day);

        {
            double elapsed =
                std::chrono::duration<double>(SteadyClock::now() - t_walk_forward).count();
            double frac = static_cast<double>(si - start_schedule_idx) /
                          static_cast<double>(total_schedules - start_schedule_idx);
            std::cout << "Schedule " << (si + 1) << "/" << total_schedules
                      << " (freq=" << cfg.frequency_days << "d, lb=" << cfg.lookback_days
                      << "d)  ETA " << format_eta(elapsed, frac) << std::endl;
        }

        // Skip if lookback is larger than available data
        if (cfg.lookback_bars + cfg.frequency_bars > num_bars) {
            std::cout << "Walk-forward: Skipping schedule " << (si + 1)
                      << " (freq=" << cfg.frequency_days << "d, lb=" << cfg.lookback_days << "d)"
                      << " -- insufficient data (" << (cfg.lookback_bars + cfg.frequency_bars)
                      << " bars needed, " << num_bars << " available)" << std::endl;
            // Emit stub runs so skipped schedules appear in the output
            for (int sel_i = 0; sel_i < SELECTION_STRATEGY_COUNT; ++sel_i) {
                WFORunResult stub;
                stub.schedule           = cfg;
                stub.selection_strategy = selection_strategy_name(all_selection_strategies(sel_i));
                stub.sufficient_windows = false;
                stub.windows_skipped    = 0;
                result.all_runs.push_back(std::move(stub));
            }
            continue;
        }

        // Evaluate all windows (the expensive part -- shared across selections)
        std::vector<WindowEvaluation> evaluations =
            engine.evaluate_windows(bars, num_bars, cfg, reduced_strategy_ids, cost, shared_ctx,
                                    shared_cl_ind, shared_cl_bt);

        if (evaluations.empty()) {
            // Mirror the stub-emit pattern used above so a
            // silent engine failure (empty evaluations) produces visible
            // output rows and an explanatory stderr line rather than an
            // invisible schedule-skip.
            std::cerr << "[WARN] Walk-forward: schedule " << (si + 1)
                      << " (freq=" << cfg.frequency_days << "d, lb=" << cfg.lookback_days
                      << "d) produced no window evaluations; emitting stub rows" << std::endl;
            for (int sel_i = 0; sel_i < SELECTION_STRATEGY_COUNT; ++sel_i) {
                WFORunResult stub;
                stub.schedule           = cfg;
                stub.selection_strategy = selection_strategy_name(all_selection_strategies(sel_i));
                stub.sufficient_windows = false;
                stub.windows_skipped    = 0;
                result.all_runs.push_back(std::move(stub));
            }
            continue;
        }

        // Precompute OOS indicator signals once for this schedule.
        // All selection strategies use the same OOS indicators -- only
        // the winner strategy differs, so OOS indicators are computed
        // once per schedule.
        PrecomputedOOSData precomputed_oos =
            engine.precompute_oos_signals(bars, num_bars, evaluations, shared_ctx, shared_cl_ind);

        // Apply all selection strategies (sharing CL instances + precomputed OOS)
        for (int sel_i = 0; sel_i < SELECTION_STRATEGY_COUNT; ++sel_i) {
            SelectionStrategy sel = all_selection_strategies(sel_i);

            WFORunResult run_result = engine.apply_selection(
                bars, num_bars, cfg, sel, evaluations, reduced_strategy_ids, cost, shared_ctx,
                shared_cl_ind, shared_cl_bt, precomputed_oos, sessions);

            result.all_runs.push_back(std::move(run_result));
        }

        // Per-schedule progress dispatch. Called synchronously
        // (see progress_cb contract). Argument `done` is the number of schedules fully
        // completed including this one; `total` is the grid size.
        if (m_walk_forward_progress_cb) {
            m_walk_forward_progress_cb(si + 1, total_schedules, result.all_runs);
        }

        // Periodic resource-limit check at schedule boundaries.
        // Warn (not abort) when the RAM ceiling is breached so the user
        // can intervene. Once per schedule is cheap and bounded.
        {
            ResourceMonitor rm;
            if (!rm.within_limits()) {
                std::cerr << "[WARN] resource limits breached after "
                             "schedule "
                          << (si + 1) << "/" << total_schedules << " (RAM " << rm.current_ram_gb()
                          << " GB / " << rm.max_ram_gb() << " GB limit)" << std::endl;
            }
        }
    }

    // Meta-analysis: which selection strategy produces the best average OOS ROC?
    // Score each selection strategy across all schedule configs.
    {
        std::unordered_map<std::string, float> sel_roc_sum;
        std::unordered_map<std::string, int>   sel_count;

        for (const auto& run : result.all_runs) {
            if (run.sufficient_windows) {
                sel_roc_sum[run.selection_strategy] += run.oos_aggregate.annual_roc;
                sel_count[run.selection_strategy]++;
            }
        }

        // Sort entries deterministically before picking the winner so
        // the chosen selection strategy does not depend on unordered_map
        // iteration order (hash seed). Sort by (-avg, name) ascending.
        std::vector<std::pair<float, std::string>> sel_avg;
        sel_avg.reserve(sel_roc_sum.size());
        for (const auto& [name, sum] : sel_roc_sum) {
            int cnt = sel_count[name];
            if (cnt > 0) {
                float avg = sum / static_cast<float>(cnt);
                sel_avg.emplace_back(avg, name);
            }
        }
        std::sort(
            sel_avg.begin(), sel_avg.end(),
            [](const std::pair<float, std::string>& a, const std::pair<float, std::string>& b) {
                if (a.first != b.first) return a.first > b.first;  // higher avg first
                return a.second < b.second;                        // ascending name tiebreak
            });
        if (!sel_avg.empty()) {
            result.best_selection_strategy    = sel_avg.front().second;
            result.best_selection_avg_oos_roc = sel_avg.front().first;
        }
    }

    return result;
}
