// ---------------------------------------------------------------------------
// End-to-end device parity: the same config reaches the same decisions on an
// OpenCL device as on the native CPU path.
//
// Kernel-level parity (one batch, GPU vs CPU totals) is covered in
// test_opencl.cpp. These tests run whole stages, so they also cover what the
// stages do with GPU results: refinement on the CPU engine, floors, the
// reduced set, and the walk-forward's out-of-sample runs.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "test_helpers.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "bar_io.h"
#include "compute/cl_context.h"
#include "orchestrator.h"
#include "synthetic.h"

namespace {

const std::string& parity_file() {
    static const std::string path = [] {
        SyntheticConfig cfg;
        cfg.seed           = 777;
        cfg.num_days       = 60;
        cfg.start_month    = 3;
        cfg.start_day      = 4;
        cfg.trend_strength = 0.03;  // some strategies pass the floors
        const std::string p =
            (std::filesystem::temp_directory_path() / "device_parity.bin").string();
        if (!write_bar_file(p, cfg.symbol, cfg.timeframe_sec, generate_synthetic_bars(cfg)))
            return std::string();
        return p;
    }();
    return path;
}

struct DefaultDevice {
    DeviceType saved = CLContext::default_device();
    explicit DefaultDevice(DeviceType d) { CLContext::set_default_device(d); }
    ~DefaultDevice() { CLContext::set_default_device(saved); }
};

bool opencl_available() {
    CLContext probe;
    return probe.initialize(DeviceType::AUTO);
}

bool approx(float a, float b) {
    return std::fabs(a - b) <= 0.01f + 1e-4f * std::max(std::fabs(a), std::fabs(b));
}

Orchestrator::ScreenConfig screen_config() {
    Orchestrator::ScreenConfig sc;
    sc.bar_file                       = parity_file();
    sc.max_strategies                 = 4000;
    sc.sanity_floors.min_total_trades = 20;
    return sc;
}

}  // namespace

TEST(DeviceParity, ScreenReachesTheSameDecisions) {
    ASSERT_FALSE(parity_file().empty());
    if (!opencl_available()) SKIP_UNLESS_OPENCL("no OpenCL device is available");

    Orchestrator::ScreenResult native, device;
    {
        DefaultDevice d(DeviceType::NATIVE);
        native = Orchestrator().run_screen(screen_config());
    }
    {
        DefaultDevice d(DeviceType::AUTO);
        device = Orchestrator().run_screen(screen_config());
    }
    ASSERT_EQ(native.total_evaluated, device.total_evaluated);
    ASSERT_GT(native.passed_floors, 0) << "setup: some strategies must pass the floors";

    for (int i = 0; i < native.total_evaluated; ++i) {
        const BacktestResult& n = native.all_results[i];
        const BacktestResult& g = device.all_results[i];
        ASSERT_EQ(n.strategy_id, g.strategy_id);
        ASSERT_EQ(n.total_trades, g.total_trades) << "strategy " << n.strategy_id;
        ASSERT_TRUE(approx(n.total_pnl, g.total_pnl)) << "strategy " << n.strategy_id;
        ASSERT_TRUE(approx(n.annual_roc, g.annual_roc)) << "strategy " << n.strategy_id;
        ASSERT_EQ(n.viable, g.viable) << "strategy " << n.strategy_id;
    }
    EXPECT_EQ(native.passed_floors, device.passed_floors);
    EXPECT_EQ(native.on_frontier, device.on_frontier);
    EXPECT_EQ(native.reduced_strategy_ids, device.reduced_strategy_ids);
}

TEST(DeviceParity, WalkForwardProducesTheSameRuns) {
    ASSERT_FALSE(parity_file().empty());
    if (!opencl_available()) SKIP_UNLESS_OPENCL("no OpenCL device is available");

    Orchestrator::WalkForwardConfig wc;
    wc.bar_file = parity_file();
    for (int id = 0; id < 9216 * 6; id += 4099) wc.reduced_strategy_ids.push_back(id);

    Orchestrator::WalkForwardResult native, device;
    {
        DefaultDevice d(DeviceType::NATIVE);
        native = Orchestrator().run_walk_forward(wc);
    }
    {
        DefaultDevice d(DeviceType::AUTO);
        device = Orchestrator().run_walk_forward(wc);
    }
    ASSERT_EQ(native.all_runs.size(), device.all_runs.size());
    for (size_t i = 0; i < native.all_runs.size(); ++i) {
        const WFORunResult& n = native.all_runs[i];
        const WFORunResult& g = device.all_runs[i];
        SCOPED_TRACE("run " + std::to_string(i));
        ASSERT_EQ(n.windows.size(), g.windows.size());
        for (size_t w = 0; w < n.windows.size(); ++w)
            ASSERT_EQ(n.windows[w].strategy_id, g.windows[w].strategy_id) << "window " << w;
        EXPECT_EQ(n.oos_aggregate.total_trades, g.oos_aggregate.total_trades);
        EXPECT_FLOAT_EQ(n.oos_aggregate.total_pnl, g.oos_aggregate.total_pnl);
    }
    EXPECT_EQ(native.best_selection_strategy, device.best_selection_strategy);
}
