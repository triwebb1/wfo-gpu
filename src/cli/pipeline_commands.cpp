// wfo-gpu screen | walk-forward | run | reanalyze | detail

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "bar_io.h"
#include "checkpoint.h"
#include "cli/commands.h"
#include "compute/cl_context.h"
#include "config.h"
#include "detail_runner.h"
#include "market_data.h"
#include "orchestrator.h"
#include "resource_monitor.h"
#include "run_lock.h"

namespace cli {

namespace {

using Clock = std::chrono::steady_clock;

struct Context {
    RunConfig   cfg;
    std::string checkpoint_dir;
    bool        resume       = false;
    bool        fresh        = false;
    bool        force_unlock = false;
};

// Load --config, apply command-line overrides, and set process-wide limits.
Context load_context(Args& args) {
    Context           c;
    const std::string path   = args.required("--config");
    ConfigResult      loaded = load_run_config(path);
    if (!loaded.ok()) throw UsageError(loaded.error);
    c.cfg = std::move(loaded.config);

    if (auto v = args.value("--bars")) c.cfg.bars = *v;
    if (auto v = args.value("--output")) c.cfg.output = *v;
    if (auto v = args.value("--device")) {
        if (!parse_device(*v, c.cfg.device))
            throw UsageError("--device must be auto, gpu, opencl-cpu or native");
    }
    if (auto v = args.integer("--max-strategies")) {
        if (*v <= 0) throw UsageError("--max-strategies must be positive");
        c.cfg.max_strategies = *v;
    }
    c.resume       = args.flag("--resume");
    c.fresh        = args.flag("--fresh");
    c.force_unlock = args.flag("--force-unlock");
    if (c.resume && c.fresh) throw UsageError("--resume and --fresh are mutually exclusive");
    if (c.cfg.bars.empty()) throw UsageError("no bar file: set [data] bars or pass --bars");

    ResourceMonitor::Limits limits = ResourceMonitor::default_limits();
    if (c.cfg.max_cpu_pct) limits.max_cpu_pct = *c.cfg.max_cpu_pct;
    if (c.cfg.max_ram_gb) limits.max_ram_gb = *c.cfg.max_ram_gb;
    ResourceMonitor::set_default_limits(limits);
    CLContext::set_default_device(c.cfg.device);
    CLIndicators::set_default_on_device(c.cfg.indicators_on_device);

    std::error_code ec;
    std::filesystem::create_directories(c.cfg.output, ec);
    if (ec)
        throw UsageError("cannot create output directory " + c.cfg.output + ": " + ec.message());
    c.checkpoint_dir = c.cfg.output + "/checkpoints";
    std::filesystem::create_directories(c.checkpoint_dir, ec);
    return c;
}

void print_banner(const Context& c, const char* stage) {
    ResourceMonitor rm;
    std::cout << "wfo-gpu " << stage << "\n"
              << "  bars      " << c.cfg.bars << "\n"
              << "  output    " << c.cfg.output << "\n"
              << "  session   " << c.cfg.session.name
              << (c.cfg.session.session_signals ? "" : " (no session signals)") << "\n"
              << "  device    " << CLContext::device_type_name(c.cfg.device)
              << (c.cfg.indicators_on_device ? " (indicators on the device)" : "") << "\n"
              << "  threads   " << rm.max_threads() << " of "
              << ResourceMonitor::hardware_concurrency() << "\n";
    const CostModel& k = c.cfg.cost;
    std::printf("  costs     tick %g = %.2f, round trip %.2f, capital %.0f%s\n", k.tick_size,
                k.tick_value, k.round_trip_cost(), k.capital_required,
                c.cfg.instrument_configured ? "" : "  (no [instrument] section: E-mini defaults)");
}

int resolve_bound(const MarketData& md, const BarBound& b, int fallback, const char* name) {
    if (!b.is_set()) return fallback;
    if (b.index) {
        if (*b.index > md.bar_count())
            throw UsageError(std::string("[screen] ") + name + " is past the end of the data");
        return static_cast<int>(*b.index);
    }
    int64_t ns = 0;
    if (!parse_timestamp_ns(b.timestamp, ns))
        throw UsageError(std::string("[screen] ") + name + " is not a date: " + b.timestamp);
    const int idx = md.find_bar_ge(ns);
    return idx < 0 ? md.bar_count() : idx;
}

uint64_t bar_file_hash(const Context& c, MarketData& md) {
    if (!md.load(c.cfg.bars, c.cfg.session)) throw UsageError("cannot load bar file " + c.cfg.bars);
    return md.data_hash();
}

std::optional<RunLock> acquire_lock(const Context& c, const char* name) {
    const std::string path = c.checkpoint_dir + "/" + name + ".lock";
    auto              lock = RunLock::acquire(path, c.force_unlock);
    if (!lock) {
        std::cerr << "error: another " << name << " run holds " << path
                  << ".\nIf a previous run crashed, re-run with --force-unlock.\n";
    }
    return lock;
}

std::string seconds_text(Clock::time_point t0) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(1)
       << std::chrono::duration<double>(Clock::now() - t0).count() << " s";
    return os.str();
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

std::optional<Orchestrator::ScreenResult> run_screen_stage(Context& c, Orchestrator& orch) {
    auto lock = acquire_lock(c, "screen");
    if (!lock) return std::nullopt;

    MarketData     md;
    const uint64_t bar_hash = bar_file_hash(c, md);

    Orchestrator::ScreenConfig sc;
    sc.bar_file          = c.cfg.bars;
    sc.output_dir        = c.cfg.output;
    sc.cost              = c.cfg.cost;
    sc.session           = c.cfg.session;
    sc.entry_lag         = c.cfg.entry_lag;
    sc.early_termination = c.cfg.early_termination;
    sc.max_strategies    = c.cfg.max_strategies;
    sc.robustness_floor  = c.cfg.robustness_floor;
    sc.sanity_floors     = c.cfg.floors;
    sc.scan_start_bar    = resolve_bound(md, c.cfg.screen_start, 0, "start");
    sc.scan_end_bar      = resolve_bound(md, c.cfg.screen_end, md.bar_count(), "end");
    if (sc.scan_end_bar <= sc.scan_start_bar)
        throw UsageError("[screen] range is empty: start must come before end");

    std::cout << "\n== Screen: bars [" << sc.scan_start_bar << ", " << sc.scan_end_bar << ") of "
              << md.bar_count() << " ==\n";

    const std::string cp_path = c.checkpoint_dir + "/screen.wfck";

    // Mid-run checkpoint every five minutes, so a crash loses at most that
    // much work. Invoked synchronously on the orchestrator thread.
    {
        auto last_save = std::make_shared<Clock::time_point>(Clock::now());
        sc.progress_cb = [cp_path, last_save, bar_hash, &sc](
                             int done, int total, const std::vector<BacktestResult>& partial) {
            if (Clock::now() - *last_save < std::chrono::minutes(5)) return;
            *last_save = Clock::now();
            Checkpoint cp;
            cp.run_id               = "screen";
            cp.phase                = 1;
            cp.config_hash          = compute_config_hash_screen(sc);
            cp.bar_data_hash        = bar_hash;
            cp.scan_start_bar       = sc.scan_start_bar;
            cp.scan_end_bar         = sc.scan_end_bar;
            cp.strategies_evaluated = done;
            cp.total_strategies     = total;
            cp.completed_results    = partial;
            cp.save(cp_path);
        };
    }

    std::optional<Orchestrator::ScreenResult> resumed;
    if (c.resume) {
        auto cp = Checkpoint::load(cp_path, compute_config_hash_screen(sc), bar_hash);
        if (cp && cp->strategies_evaluated >= cp->total_strategies &&
            !cp->completed_results.empty()) {
            resumed = orch.reanalyze_results(std::move(cp->completed_results), sc.robustness_floor,
                                             sc.sanity_floors,
                                             Orchestrator::screen_signal_count(sc.session));
            std::cout << "Resumed the completed screen from " << cp_path << "\n";
        } else if (cp && !cp->completed_results.empty()) {
            std::cout << "Resuming the screen after " << cp->completed_results.size()
                      << " strategies from " << cp_path << "\n";
            sc.resumed_results = std::move(cp->completed_results);
        } else if (std::filesystem::exists(cp_path)) {
            std::cerr << "error: " << cp_path
                      << " does not match this config or bar file, or is corrupt.\n"
                         "Re-run with --fresh to discard it.\n";
            return std::nullopt;
        }
    } else if (c.fresh) {
        std::error_code ec;
        std::filesystem::remove(cp_path, ec);
    }

    const auto                 t0 = Clock::now();
    Orchestrator::ScreenResult r  = resumed ? std::move(*resumed) : orch.run_screen(sc);
    if (resumed) r.output_written = Orchestrator::write_results(c.cfg.output, r);

    Checkpoint cp;
    cp.run_id               = "screen";
    cp.phase                = 1;
    cp.config_hash          = compute_config_hash_screen(sc);
    cp.bar_data_hash        = bar_hash;
    cp.scan_start_bar       = sc.scan_start_bar;
    cp.scan_end_bar         = sc.scan_end_bar;
    cp.strategies_evaluated = r.total_evaluated;
    cp.total_strategies     = r.total_evaluated;
    cp.completed_results    = r.all_results;
    cp.save(cp_path);

    std::cout << "\nScreen finished in " << seconds_text(t0) << "\n"
              << "  strategies evaluated   " << r.total_evaluated << "\n"
              << "  passed sanity floors   " << r.passed_floors << "\n"
              << "  economically viable    " << r.landscape.viable_survivors << "\n"
              << "  on Pareto frontier     " << r.on_frontier << "\n"
              << "  kept for walk-forward  " << r.reduced_set_size << "\n"
              << "  archetypes            " << r.grinder_count << " grinder, " << r.sniper_count
              << " sniper, " << r.balanced_count << " balanced\n"
              << "  results                " << (r.output_written ? c.cfg.output : "not written")
              << "\n";
    return r;
}

// ---------------------------------------------------------------------------
// Walk-forward
// ---------------------------------------------------------------------------

bool read_reduced_set(const std::string& path, std::vector<int>& ids) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    int32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || count <= 0 || count > 10'000'000) return false;
    ids.resize(static_cast<size_t>(count));
    in.read(reinterpret_cast<char*>(ids.data()),
            static_cast<std::streamsize>(count * sizeof(int32_t)));
    return static_cast<bool>(in);
}

void print_walk_forward_table(const Orchestrator::WalkForwardResult& r) {
    // Rank only runs with enough out-of-sample windows to mean something.
    constexpr size_t                 MIN_WINDOWS = 4;
    std::vector<const WFORunResult*> runs;
    for (const auto& run : r.all_runs)
        if (run.sufficient_windows && run.windows.size() >= MIN_WINDOWS) runs.push_back(&run);
    if (runs.empty()) {
        std::cout << "  no schedule produced " << MIN_WINDOWS << " or more out-of-sample windows\n";
        return;
    }

    // The distribution first: the best row was picked on out-of-sample data
    // and is optimistic by construction.
    std::vector<float> rocs;
    int                profitable = 0;
    for (const auto* run : runs) {
        rocs.push_back(run->oos_aggregate.annual_roc);
        if (run->oos_aggregate.total_pnl > 0.0f) ++profitable;
    }
    std::sort(rocs.begin(), rocs.end());
    const size_t mid    = rocs.size() / 2;
    const float  median = rocs.size() % 2 ? rocs[mid] : 0.5f * (rocs[mid - 1] + rocs[mid]);
    std::printf(
        "\n  Out-of-sample across %zu runs with %zu+ windows: median ROC %.2f%%, "
        "%d of %zu profitable\n",
        runs.size(), MIN_WINDOWS, median, profitable, runs.size());

    std::sort(runs.begin(), runs.end(), [](const WFORunResult* a, const WFORunResult* b) {
        return a->oos_aggregate.annual_roc > b->oos_aggregate.annual_roc;
    });
    std::printf(
        "\n  Top runs by out-of-sample return on capital\n"
        "  %-6s %-8s %-9s %-19s %9s %7s %6s %9s %7s %7s\n",
        "step", "lookback", "trigger", "selection", "OOS ROC%", "Sharpe", "PF", "MaxDD$", "trades",
        "windows");
    const size_t shown = std::min<size_t>(runs.size(), 10);
    for (size_t i = 0; i < shown; ++i) {
        const WFORunResult&   run      = *runs[i];
        const BacktestResult& m        = run.oos_aggregate;
        const std::string     step     = std::to_string(run.schedule.frequency_days) + "d";
        const std::string     lookback = std::to_string(run.schedule.lookback_days) + "d";
        std::printf("  %-6s %-8s %-9s %-19s %9.2f %7.2f %6.2f %9.0f %7d %7zu\n", step.c_str(),
                    lookback.c_str(), trigger_name(run.schedule.trigger),
                    run.selection_strategy.c_str(), m.annual_roc, m.sharpe_ratio,
                    std::min(m.profit_factor, 99.99f), m.max_drawdown, m.total_trades,
                    run.windows.size());
    }
}

int run_walk_forward_stage(Context& c, Orchestrator& orch, std::vector<int> reduced_ids) {
    auto lock = acquire_lock(c, "walk_forward");
    if (!lock) return 1;

    MarketData     md;
    const uint64_t bar_hash = bar_file_hash(c, md);

    Orchestrator::WalkForwardConfig wc;
    wc.bar_file     = c.cfg.bars;
    wc.output_dir   = c.cfg.output;
    wc.cost         = c.cfg.cost;
    wc.session      = c.cfg.session;
    wc.bars_per_day = static_cast<float>(md.bars_per_calendar_day());
    // Out-of-sample trading starts where the screen range ends.
    wc.oos_start_bar = resolve_bound(md, c.cfg.screen_end, md.bar_count(), "end");
    if (wc.oos_start_bar >= md.bar_count()) {
        std::cerr << "error: the screen range covers the whole file, leaving no "
                     "out-of-sample bars.\nSet [screen] end before the end of the data.\n";
        return 1;
    }

    if (reduced_ids.empty()) {
        const std::string path = c.cfg.output + "/reduced_params.bin";
        if (!read_reduced_set(path, reduced_ids)) {
            std::cerr << "error: cannot read " << path
                      << ".\nRun `wfo-gpu screen` with this config first.\n";
            return 1;
        }
    }
    wc.reduced_strategy_ids = std::move(reduced_ids);
    std::cout << "\n== Walk-forward: " << wc.reduced_strategy_ids.size()
              << " screened strategies, out-of-sample from bar " << wc.oos_start_bar << " of "
              << md.bar_count() << " ==\n";

    const std::string cp_path = c.checkpoint_dir + "/walk_forward.wfck";
    wc.progress_cb            = [cp_path, bar_hash, &wc](int                              done, int,
                                              const std::vector<WFORunResult>& partial) {
        Checkpoint cp;
        cp.run_id               = "walk_forward";
        cp.phase                = 2;
        cp.config_hash          = compute_config_hash_walk_forward(wc);
        cp.bar_data_hash        = bar_hash;
        cp.current_schedule_idx = done;
        cp.completed_runs       = partial;
        cp.save(cp_path);
    };

    if (c.resume) {
        auto cp = Checkpoint::load(cp_path, compute_config_hash_walk_forward(wc), bar_hash);
        if (cp) {
            if (cp->current_schedule_idx >= WFOGrid().size())
                std::cout << "Walk-forward already complete; rewriting its outputs from " << cp_path
                          << "\n";
            else
                std::cout << "Resuming after schedule " << cp->current_schedule_idx << " ("
                          << cp->completed_runs.size() << " runs restored)\n";
            wc.start_schedule_idx = cp->current_schedule_idx;
            wc.resumed_runs       = std::move(cp->completed_runs);
        } else if (std::filesystem::exists(cp_path)) {
            std::cerr << "error: " << cp_path
                      << " does not match this config or bar file, or is corrupt.\n"
                         "Re-run with --fresh to discard it.\n";
            return 1;
        }
    } else if (c.fresh) {
        std::error_code ec;
        std::filesystem::remove(cp_path, ec);
    }

    const auto t0 = Clock::now();
    const auto r  = orch.run_walk_forward(wc);

    // Mark every schedule done, so a later --resume only rewrites the outputs.
    Checkpoint cp;
    cp.run_id               = "walk_forward";
    cp.phase                = 2;
    cp.config_hash          = compute_config_hash_walk_forward(wc);
    cp.bar_data_hash        = bar_hash;
    cp.current_schedule_idx = WFOGrid().size();
    cp.completed_runs       = r.all_runs;
    cp.save(cp_path);

    std::cout << "\nWalk-forward finished in " << seconds_text(t0) << "\n"
              << "  schedule x selection runs  " << r.all_runs.size() << "\n"
              << "  top selection rule         " << r.best_selection_strategy << " (" << std::fixed
              << std::setprecision(2) << r.best_selection_avg_oos_roc
              << "% avg OOS ROC; ranked on OOS data, so optimistic)\n"
              << "  results                    "
              << (r.output_written ? c.cfg.output : "not written") << "\n";
    print_walk_forward_table(r);
    return 0;
}

}  // namespace

int cmd_screen(Args& args) {
    Context c = load_context(args);
    args.finish();
    print_banner(c, "screen");
    Orchestrator orch;
    return run_screen_stage(c, orch) ? 0 : 1;
}

int cmd_walk_forward(Args& args) {
    Context c = load_context(args);
    args.finish();
    print_banner(c, "walk-forward");
    Orchestrator orch;
    return run_walk_forward_stage(c, orch, {});
}

int cmd_run(Args& args) {
    Context c = load_context(args);
    args.finish();
    print_banner(c, "run (screen + walk-forward)");
    Orchestrator orch;
    auto         screened = run_screen_stage(c, orch);
    if (!screened) return 1;
    if (screened->reduced_strategy_ids.empty()) {
        std::cout << "\nNo strategy survived the screen; nothing to walk forward.\n";
        return 0;
    }
    return run_walk_forward_stage(c, orch, screened->reduced_strategy_ids);
}

int cmd_reanalyze(Args& args) {
    Context    c     = load_context(args);
    const auto floor = args.number("--robustness-floor");
    args.finish();

    const std::string cp_path = c.checkpoint_dir + "/screen.wfck";
    // Reanalysis reuses stored results, so config and data hashes are not
    // checked: changing the floors is the point.
    auto cp = Checkpoint::load(cp_path);
    if (!cp) {
        std::cerr << "error: no valid screen checkpoint at " << cp_path << "\n";
        return 1;
    }
    const float  robustness = floor ? static_cast<float>(*floor) : c.cfg.robustness_floor;
    Orchestrator orch;
    auto r = orch.reanalyze_results(std::move(cp->completed_results), robustness, c.cfg.floors,
                                    Orchestrator::screen_signal_count(c.cfg.session));
    r.output_written = Orchestrator::write_results(c.cfg.output, r);

    std::cout << "Reanalyzed " << r.total_evaluated << " screen results (robustness floor "
              << robustness << ")\n"
              << "  passed sanity floors   " << r.passed_floors << "\n"
              << "  on Pareto frontier     " << r.on_frontier << "\n"
              << "  kept for walk-forward  " << r.reduced_set_size << "\n";
    return r.output_written ? 0 : 1;
}

int cmd_detail(Args& args) {
    Context           c        = load_context(args);
    const std::string ids_path = args.required("--ids");
    const auto        start    = args.integer("--start");
    const auto        end      = args.integer("--end");
    const int         lag      = static_cast<int>(args.integer("--lag").value_or(1));
    args.finish();
    if (lag < 1 || lag > 2) throw UsageError("--lag must be 1 or 2");

    const auto ids = detail::load_strategy_ids(ids_path);
    if (ids.empty()) {
        std::cerr << "error: no strategy ids in " << ids_path << "\n";
        return 1;
    }
    MarketData md;
    if (!md.load(c.cfg.bars, c.cfg.session)) {
        std::cerr << "error: cannot load " << c.cfg.bars << "\n";
        return 1;
    }
    const int n = md.bar_count();
    const int s = static_cast<int>(std::clamp<int64_t>(start.value_or(0), 0, n));
    const int e = static_cast<int>(std::clamp<int64_t>(end.value_or(n), s, n));

    const int64_t grid_size =
        StrategyGrid(Orchestrator::screen_signal_count(c.cfg.session)).total_count();
    for (int id : ids) {
        if (id < 0 || id >= grid_size) {
            std::cerr << "error: strategy id " << id << " is outside the strategy grid (0 to "
                      << grid_size - 1 << ")\n";
            return 1;
        }
    }

    std::cout << "Evaluating " << ids.size() << " strategies on bars [" << s << ", " << e
              << ") with entry lag " << lag << "\n";
    const auto results =
        detail::run_detail(md.bars(), n, s, e, ids, c.cfg.cost, lag, md.sessions(), &md);
    const std::string dir = c.cfg.output + "/detail";
    const bool        ok  = detail::write_all_detail_outputs(dir, results, md.bars());

    std::cout << "\n  strategy   trades   ROC%    Sharpe  PF     MaxDD$    GreenWk%\n";
    for (const auto& dr : results) {
        const auto& m = dr.metrics;
        char        line[160];
        std::snprintf(line, sizeof(line), "  %-9d %7d %7.2f %7.2f %6.2f %9.0f %9.1f\n",
                      dr.strategy_id, m.total_trades, m.annual_roc, m.sharpe_ratio,
                      std::min(m.profit_factor, 99.99f), m.max_drawdown, m.green_weeks_pct);
        std::cout << line;
    }
    std::cout << "\nTrade logs and summaries written to " << dir << "\n";
    return ok ? 0 : 1;
}

}  // namespace cli
