// wfo-gpu-bench: backtest throughput on a fixed synthetic workload.
//
// Builds a seeded synthetic bar series, computes and normalizes the full
// technical-signal grid once, then times CLBacktest::evaluate on a fixed,
// set of distinct strategies (two workloads, below) for each compute device.
// The reported figures are:
//
//   backtests/s       strategies evaluated per second
//   bar-evals/s       strategies * evaluated bars per second
//
// Timings cover strategy evaluation only (host transfers included on the
// GPU path); signal computation is reported separately. Each device runs one
// warm-up pass and reports the median of --repeat timed passes.
//
// Usage: wfo-gpu-bench [--days N] [--strategies N] [--repeat N] [--markdown]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "compute/cl_backtest.h"
#include "compute/cl_context.h"
#include "compute/cl_indicators.h"
#include "cost_model.h"
#include "grids/indicator_grid.h"
#include "grids/strategy_grid.h"
#include "indicators/signal_normalize.h"
#include "resource_monitor.h"
#include "synthetic.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define WFO_HAVE_CPUID 1
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#define WFO_HAVE_CPUID 1
#endif

namespace {

using Clock = std::chrono::steady_clock;

std::string cpu_brand() {
#ifdef WFO_HAVE_CPUID
    unsigned int regs[12] = {};
    for (unsigned int i = 0; i < 3; ++i) {
#ifdef _MSC_VER
        int r[4];
        __cpuid(r, static_cast<int>(0x80000002u + i));
        std::memcpy(regs + 4 * i, r, sizeof(r));
#else
        __get_cpuid(0x80000002u + i, &regs[4 * i], &regs[4 * i + 1], &regs[4 * i + 2],
                    &regs[4 * i + 3]);
#endif
    }
    std::string s(reinterpret_cast<const char*>(regs), sizeof(regs));
    s            = s.c_str();
    const auto b = s.find_first_not_of(' ');
    return b == std::string::npos ? "unknown CPU" : s.substr(b);
#else
    return "unknown CPU";
#endif
}

std::string os_name() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "unknown OS";
#endif
}

std::string compiler_name() {
#if defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "unknown compiler";
#endif
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int arg_int(int argc, char** argv, const char* name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return std::max(1, std::atoi(argv[i + 1]));
    return fallback;
}

bool arg_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const int  days           = arg_int(argc, argv, "--days", 30);
    const int  num_strategies = arg_int(argc, argv, "--strategies", 20000);
    const int  repeat         = arg_int(argc, argv, "--repeat", 5);
    const bool markdown       = arg_flag(argc, argv, "--markdown");

    // ---- Workload: fixed seed, fixed size ---------------------------------
    SyntheticConfig cfg;
    cfg.seed                    = 424242;
    cfg.num_days                = days;
    cfg.start_year              = 2024;
    cfg.start_month             = 1;
    cfg.start_day               = 2;
    const std::vector<Bar> bars = generate_synthetic_bars(cfg);
    const int              n    = static_cast<int>(bars.size());

    std::vector<float> o(n), h(n), l(n), c(n);
    for (int i = 0; i < n; ++i) {
        o[i] = bars[i].open;
        h[i] = bars[i].high;
        l[i] = bars[i].low;
        c[i] = bars[i].close;
    }

    // ---- Signals: full technical grid, computed once ----------------------
    const IndicatorGrid        grid;
    std::vector<IndicatorSpec> specs;
    for (int i = 0; i < grid.size(); ++i)
        for (int s = 0; s < indicator_num_series(grid.params(i)); ++s)
            specs.push_back(IndicatorSpec::from_params(grid.params(i), s));
    std::vector<std::vector<float>> raw(specs.size(), std::vector<float>(n));
    std::vector<float*>             raw_ptrs(specs.size());
    for (size_t i = 0; i < raw.size(); ++i) raw_ptrs[i] = raw[i].data();

    const auto t_sig = Clock::now();
    {
        CLContext    native;  // signals on the native path, so every device sees identical inputs
        CLIndicators ind;
        ind.compute_with_bars(native, bars.data(), 0, o.data(), h.data(), l.data(), c.data(), n,
                              specs, raw_ptrs);
    }
    std::vector<std::vector<float>> sig(grid.size(), std::vector<float>(n));
    std::vector<float*>             sig_ptrs(grid.size());
    for (int i = 0; i < grid.size(); ++i) sig_ptrs[i] = sig[i].data();
    normalize_all_indicators(grid, raw_ptrs, bars.data(), n, sig_ptrs);
    const double signal_seconds = std::chrono::duration<double>(Clock::now() - t_sig).count();

    // ---- Strategies: two fixed workloads ----------------------------------
    // "screen":  the first distinct strategies in grid order, as the screen
    //            enumerates them; neighbouring strategies share signals.
    // "sampled": a deterministic stride across the whole grid, as in a
    //            walk-forward reduced set where strategies share little.
    // Each further pass starts one entry later, so redundant entries skipped
    // on the first pass are made up without leaving the stride pattern.
    const StrategyGrid sgrid(grid.size());
    const int64_t      total = sgrid.total_count();
    auto               build = [&](int64_t stride) {
        std::vector<StrategyConfig> out;
        out.reserve(num_strategies);
        for (int64_t first = 0; first < stride; ++first) {
            for (int64_t id = first; id < total && static_cast<int>(out.size()) < num_strategies;
                 id += stride) {
                const StrategyGridEntry e = sgrid.entry_at(id);
                if (!e.is_redundant()) out.push_back(e.to_strategy_config());
            }
            if (static_cast<int>(out.size()) == num_strategies) break;
        }
        return out;
    };
    struct Workload {
        std::string                 name;
        std::vector<StrategyConfig> strategies;
    };
    const std::vector<Workload> workloads = {
        {"screen", build(1)},
        {"sampled", build(std::max<int64_t>(1, total / num_strategies))},
    };

    const int       warmup     = 1000;  // bars of signal warmup before evaluation starts
    const int       eval_start = std::min(warmup, n - 1);
    const int       eval_end   = n;
    const int64_t   eval_bars  = eval_end - eval_start;
    const CostModel cost       = CostModel::es();

    auto time_device = [&](const CLContext& ctx, const std::vector<StrategyConfig>& strategies) {
        std::vector<double>         runs;
        CLBacktest                  bt;
        std::vector<BacktestResult> results;
        for (int r = 0; r <= repeat; ++r) {
            const auto t0 = Clock::now();
            bt.evaluate(ctx, bars.data(), 0, const_cast<const float* const*>(sig_ptrs.data()),
                        grid.size(), n, eval_start, eval_end, strategies, cost, 1, results);
            const double s = std::chrono::duration<double>(Clock::now() - t0).count();
            if (r > 0) runs.push_back(s);  // pass 0 is warm-up (kernel build, buffer setup)
        }
        return median(runs);
    };

    struct Device {
        std::string label, hardware;
        CLContext   ctx;
    };
    std::vector<Device> devices;
    {
        ResourceMonitor rm;
        Device          d;
        d.label    = "native (" + std::to_string(rm.max_threads()) + " threads)";
        d.hardware = cpu_brand() + ", " + std::to_string(rm.max_threads()) + " of " +
                     std::to_string(ResourceMonitor::hardware_concurrency()) + " threads";
        devices.push_back(std::move(d));
    }
    for (const DeviceType type : {DeviceType::GPU, DeviceType::CPU}) {
        Device d;
        if (!d.ctx.initialize(type)) continue;
        d.label    = std::string("opencl ") + CLContext::device_type_name(type);
        d.hardware = d.ctx.device_name();
        if (type == DeviceType::GPU)
            d.hardware +=
                ", " + std::to_string(d.ctx.device_memory_bytes() / (1024 * 1024)) + " MB";
        devices.push_back(std::move(d));
    }

    struct Row {
        const Device*   device;
        const Workload* workload;
        double          seconds;
    };
    std::vector<Row> rows;
    for (const auto& w : workloads)
        for (const auto& d : devices) rows.push_back({&d, &w, time_device(d.ctx, w.strategies)});

    // ---- Report -------------------------------------------------------------
    if (markdown) {
        for (const auto& r : rows) {
            const double count = static_cast<double>(r.workload->strategies.size());
            char         line[512];
            std::snprintf(line, sizeof(line),
                          "| %s | %s | %s | %s %s | %d | %lld | %.0f | %.3g |\n",
                          r.workload->name.c_str(), r.device->label.c_str(),
                          r.device->hardware.c_str(), os_name().c_str(), compiler_name().c_str(),
                          static_cast<int>(count), static_cast<long long>(eval_bars),
                          count / r.seconds, count * eval_bars / r.seconds);
            std::cout << line;
        }
        return 0;
    }

    std::cout << "wfo-gpu-bench\n"
              << "  bars        " << n << " synthetic 30s bars (" << days << " sessions), "
              << eval_bars << " evaluated\n"
              << "  strategies  up to " << num_strategies << " distinct per workload over "
              << grid.size() << " signals, entry lag 1\n"
              << "  host        " << cpu_brand() << ", " << os_name() << ", " << compiler_name()
              << "\n"
              << "  signals     computed in " << signal_seconds << " s (native path)\n";
    for (const auto& w : workloads) {
        int by_k[5] = {};
        for (const auto& s : w.strategies) ++by_k[s.num_indicators];
        std::printf("  %-11s %zu strategies; 1/2/3/4 signals: %d/%d/%d/%d\n", w.name.c_str(),
                    w.strategies.size(), by_k[1], by_k[2], by_k[3], by_k[4]);
    }
    std::cout << "\n";
    std::printf("  %-9s %-24s %10s %14s %16s\n", "workload", "device", "median s", "backtests/s",
                "bar-evals/s");
    for (const auto& r : rows) {
        const double count = static_cast<double>(r.workload->strategies.size());
        std::printf("  %-9s %-24s %10.3f %14.0f %16.3g\n", r.workload->name.c_str(),
                    r.device->label.c_str(), r.seconds, count / r.seconds,
                    count * eval_bars / r.seconds);
    }
    for (const auto& d : devices) std::cout << "\n  " << d.label << ": " << d.hardware;
    std::cout << "\n";
    return 0;
}
