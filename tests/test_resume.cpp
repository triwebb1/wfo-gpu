// ---------------------------------------------------------------------------
// Resuming interrupted runs.
//
// A resumed run must produce exactly what an uninterrupted run produces:
// no stage repeated, no result lost, however many times it was interrupted.
// The checkpoints themselves are covered in test_checkpoint_resources.cpp;
// these tests drive the orchestrator through the same state handoff the CLI
// performs (progress callback -> checkpoint -> resumed config).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "test_helpers.h"

#include <filesystem>
#include <string>
#include <vector>

#include "bar_io.h"
#include "compute/cl_context.h"
#include "grids/wfo_grid.h"
#include "orchestrator.h"
#include "synthetic.h"

namespace {

// Sixty synthetic sessions: enough for the one- and two-month lookback
// schedules to produce real windows.
const std::string& sixty_day_file() {
    static const std::string path = [] {
        SyntheticConfig cfg;
        cfg.seed        = 4242;
        cfg.num_days    = 60;
        cfg.start_month = 3;
        cfg.start_day   = 4;
        const std::string p =
            (std::filesystem::temp_directory_path() / "resume_sixty_days.bin").string();
        if (!write_bar_file(p, cfg.symbol, cfg.timeframe_sec, generate_synthetic_bars(cfg)))
            return std::string();
        return p;
    }();
    return path;
}

Orchestrator::WalkForwardConfig walk_forward_config() {
    Orchestrator::WalkForwardConfig wc;
    wc.bar_file = sixty_day_file();
    // A spread of single-signal strategies: ids of the first few signals.
    for (int id = 0; id < 9216 * 6; id += 4099) wc.reduced_strategy_ids.push_back(id);
    return wc;
}

void expect_same_runs(const std::vector<WFORunResult>& a, const std::vector<WFORunResult>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        SCOPED_TRACE("run " + std::to_string(i));
        EXPECT_EQ(a[i].schedule.frequency_days, b[i].schedule.frequency_days);
        EXPECT_EQ(a[i].schedule.lookback_days, b[i].schedule.lookback_days);
        EXPECT_EQ(a[i].schedule.trigger, b[i].schedule.trigger);
        EXPECT_EQ(a[i].selection_strategy, b[i].selection_strategy);
        EXPECT_EQ(a[i].windows.size(), b[i].windows.size());
        EXPECT_EQ(a[i].oos_aggregate.total_trades, b[i].oos_aggregate.total_trades);
        EXPECT_FLOAT_EQ(a[i].oos_aggregate.total_pnl, b[i].oos_aggregate.total_pnl);
    }
}

// Runs on the native path so the comparison does not depend on a device.
struct NativeDevice {
    DeviceType saved = CLContext::default_device();
    NativeDevice() { CLContext::set_default_device(DeviceType::NATIVE); }
    ~NativeDevice() { CLContext::set_default_device(saved); }
};

}  // namespace

TEST(Resume, WalkForwardInterruptedTwiceMatchesAnUninterruptedRun) {
    ASSERT_FALSE(sixty_day_file().empty());
    NativeDevice native;

    // Uninterrupted, recording what each checkpoint would have held.
    std::vector<std::pair<int, std::vector<WFORunResult>>> checkpoints;
    auto                                                   wc = walk_forward_config();
    wc.progress_cb = [&](int done, int, const std::vector<WFORunResult>& runs) {
        checkpoints.emplace_back(done, runs);
    };
    const auto full = Orchestrator().run_walk_forward(wc);
    ASSERT_GE(checkpoints.size(), 3u) << "setup: need several completed schedules";
    ASSERT_EQ(static_cast<int>(full.all_runs.size()), WFOGrid().size() * SELECTION_STRATEGY_COUNT);

    // First interruption after the first checkpoint; the resumed run's own
    // checkpoints must include the restored runs.
    const auto& first      = checkpoints.front();
    auto        wc2        = walk_forward_config();
    wc2.start_schedule_idx = first.first;
    wc2.resumed_runs       = first.second;
    std::vector<std::pair<int, std::vector<WFORunResult>>> checkpoints2;
    wc2.progress_cb = [&](int done, int, const std::vector<WFORunResult>& runs) {
        checkpoints2.emplace_back(done, runs);
    };
    const auto resumed = Orchestrator().run_walk_forward(wc2);
    expect_same_runs(resumed.all_runs, full.all_runs);
    EXPECT_EQ(resumed.best_selection_strategy, full.best_selection_strategy);
    EXPECT_FLOAT_EQ(resumed.best_selection_avg_oos_roc, full.best_selection_avg_oos_roc);
    ASSERT_FALSE(checkpoints2.empty());

    // Second interruption, from a checkpoint the resumed run wrote.
    const auto& second     = checkpoints2.front();
    auto        wc3        = walk_forward_config();
    wc3.start_schedule_idx = second.first;
    wc3.resumed_runs       = second.second;
    const auto twice       = Orchestrator().run_walk_forward(wc3);
    expect_same_runs(twice.all_runs, full.all_runs);
    EXPECT_EQ(twice.best_selection_strategy, full.best_selection_strategy);
}

// Resuming a finished run (every schedule done) repeats nothing.
TEST(Resume, CompletedWalkForwardResumesToTheSameResult) {
    ASSERT_FALSE(sixty_day_file().empty());
    NativeDevice native;

    const auto full       = Orchestrator().run_walk_forward(walk_forward_config());
    auto       wc         = walk_forward_config();
    wc.start_schedule_idx = WFOGrid().size();
    wc.resumed_runs       = full.all_runs;
    int callbacks         = 0;
    wc.progress_cb        = [&](int, int, const std::vector<WFORunResult>&) { ++callbacks; };
    const auto again      = Orchestrator().run_walk_forward(wc);
    EXPECT_EQ(callbacks, 0);
    expect_same_runs(again.all_runs, full.all_runs);
    EXPECT_EQ(again.best_selection_strategy, full.best_selection_strategy);
}

// A screen resumed from a mid-run checkpoint evaluates the rest of the grid
// and reaches the same reduced set.
TEST(Resume, PartialScreenContinuesAfterItsLastResult) {
    ASSERT_FALSE(sixty_day_file().empty());
    NativeDevice native;

    Orchestrator::ScreenConfig sc;
    sc.bar_file                       = sixty_day_file();
    sc.max_strategies                 = 3000;
    sc.sanity_floors.min_total_trades = 20;

    std::vector<BacktestResult> partial;
    sc.progress_cb = [&](int done, int, const std::vector<BacktestResult>& results) {
        if (partial.empty() && done >= 1000) partial = results;
    };
    const auto full = Orchestrator().run_screen(sc);
    ASSERT_EQ(full.total_evaluated, 3000);
    ASSERT_FALSE(partial.empty());
    ASSERT_LT(partial.size(), 3000u);

    sc.progress_cb     = nullptr;
    sc.resumed_results = partial;
    const auto resumed = Orchestrator().run_screen(sc);
    ASSERT_EQ(resumed.total_evaluated, full.total_evaluated);
    for (int i = 0; i < full.total_evaluated; ++i) {
        ASSERT_EQ(resumed.all_results[i].strategy_id, full.all_results[i].strategy_id) << i;
        ASSERT_FLOAT_EQ(resumed.all_results[i].total_pnl, full.all_results[i].total_pnl) << i;
    }
    EXPECT_EQ(resumed.reduced_strategy_ids, full.reduced_strategy_ids);
}
