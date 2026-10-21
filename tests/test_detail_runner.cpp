// ---------------------------------------------------------------------------
// Tests for detail_runner: strategy decoding, session plumbing, quarterly
// buckets and the per-strategy output files written by `wfo-gpu detail`.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "test_fixture.h"
#include "test_helpers.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "detail_runner.h"
#include "grids/strategy_grid.h"
#include "market_data.h"
#include "orchestrator.h"

namespace {

int count_lines(const std::string& path) {
    std::ifstream f(path);
    int           n = 0;
    std::string   line;
    while (std::getline(f, line))
        if (!line.empty()) ++n;
    return n;
}

std::string first_line(const std::string& path) {
    std::ifstream f(path);
    std::string   line;
    std::getline(f, line);
    return line;
}

// Detail runs on the committed fixture, evaluated after a day of warmup.
struct FixtureRun {
    MarketData md;
    int        eval_start = 0;
    int        eval_end   = 0;

    bool load() {
        if (!md.load(test_fixture::fixture_path())) return false;
        eval_start = md.bar_count() / 3;
        eval_end   = md.bar_count();
        return true;
    }

    std::vector<detail::DetailResult> run(const std::vector<int>& ids, bool with_md) const {
        return detail::run_detail(md.bars(), md.bar_count(), eval_start, eval_end, ids,
                                  CostModel::es(), 1, md.sessions(), with_md ? &md : nullptr);
    }
};

}  // namespace

// Sessions must reach BacktestEngine::run from run_detail. An RTH strategy on
// bars stamped at 00:00 (outside core hours) trades without sessions and
// never with them. Strategy id 2 decodes to hours = RTH, everything else at
// its first value.
TEST(DetailRunner, SessionsActivateTradingLogic) {
    const int        num_bars = 500;
    std::vector<Bar> bars(num_bars);
    const int64_t    start_ns = static_cast<int64_t>(1704153600LL) * NS_PER_SEC;
    for (int i = 0; i < num_bars; ++i) {
        const float base = 4000.0f + static_cast<float>(i % 100) * 0.1f;
        bars[i]          = make_bar(start_ns + static_cast<int64_t>(i) * 30LL * NS_PER_SEC, base,
                                    base + 0.5f, base - 0.5f, base + ((i % 2 == 0) ? 0.3f : -0.3f));
    }

    std::vector<SessionInfo> sessions(num_bars);
    for (int i = 0; i < num_bars; ++i) {
        sessions[i].session_day       = 0;
        sessions[i].intra_session_idx = i;
        sessions[i].time_of_day_sec   = 0;  // outside core hours
        sessions[i].halt_flag         = 0;
        sessions[i].early_close       = 0;
        sessions[i].session_close_sec = -1;  // no daily close in this test
    }
    stamp_session_flags(sessions.data(), num_bars, SessionModel::us_futures(), 30);

    const std::vector<int> ids = {2};
    ASSERT_EQ(StrategyGrid(IndicatorGrid().size()).entry_at(2).to_strategy_config().trading_hours,
              StrategyConfig::RTH);

    auto without =
        detail::run_detail(bars.data(), num_bars, 200, num_bars, ids, CostModel::es(), 1, nullptr);
    auto with = detail::run_detail(bars.data(), num_bars, 200, num_bars, ids, CostModel::es(), 1,
                                   sessions.data());
    ASSERT_EQ(without.size(), 1u);
    ASSERT_EQ(with.size(), 1u);
    EXPECT_GT(without[0].metrics.total_trades, 0) << "setup: the baseline must trade";
    EXPECT_EQ(with[0].metrics.total_trades, 0) << "the RTH filter must block every entry";
}

// With MarketData, ids decode against the screen's grid, which spans the
// session signals after the technical ones.
TEST(DetailRunner, SessionSignalIdsDecodeAsInTheScreen) {
    FixtureRun f;
    ASSERT_TRUE(f.load());
    const int    num_signals = Orchestrator::screen_signal_count(f.md.session_model());
    const int    technical   = IndicatorGrid().size();
    StrategyGrid grid(num_signals);
    ASSERT_GT(num_signals, technical);

    // The first single-signal entry built on a session signal.
    int64_t id = -1;
    for (int64_t i = 0; i < grid.total_count() && id < 0; ++i) {
        const StrategyGridEntry e = grid.entry_at(i);
        if (!e.is_redundant() && e.num_indicators == 1 && e.indicator_ids[0] >= technical) id = i;
    }
    ASSERT_GE(id, 0);
    ASSERT_LE(id, std::numeric_limits<int>::max());

    const auto results = f.run({static_cast<int>(id)}, true);
    ASSERT_EQ(results.size(), 1u);
    const StrategyConfig expected = grid.entry_at(id).to_strategy_config();
    EXPECT_EQ(results[0].config.num_indicators, 1);
    EXPECT_EQ(results[0].config.indicator_indices[0], expected.indicator_indices[0]);
    EXPECT_EQ(results[0].config.exit_mode, expected.exit_mode);
    EXPECT_EQ(results[0].config.trading_hours, expected.trading_hours);
}

TEST(DetailRunner, IdOutsideTheGridGivesAnEmptyResult) {
    FixtureRun f;
    ASSERT_TRUE(f.load());
    const auto results = f.run({-1}, true);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].strategy_id, -1);
    EXPECT_EQ(results[0].metrics.total_trades, 0);
    EXPECT_TRUE(results[0].trades.empty());
}

// Quarterly buckets partition the trade log: their PnL and trade counts add
// up to the strategy totals.
TEST(DetailRunner, QuarterlyBucketsAddUpToTheTotals) {
    FixtureRun f;
    ASSERT_TRUE(f.load());
    const auto results = f.run({0, 1, 2, 3, 4, 5}, true);
    int        traded  = 0;
    for (const auto& dr : results) {
        if (dr.trades.empty()) continue;
        ++traded;
        double pnl    = 0.0;
        int    trades = 0;
        for (const auto& q : dr.quarterly) {
            pnl += q.pnl;
            trades += q.trade_count;
        }
        EXPECT_EQ(trades, static_cast<int>(dr.trades.size()));
        EXPECT_EQ(trades, dr.metrics.total_trades);
        EXPECT_NEAR(pnl, dr.metrics.total_pnl, 0.01 + 1e-4 * std::fabs(dr.metrics.total_pnl));
    }
    EXPECT_GT(traded, 0) << "setup: at least one strategy must trade";
}

TEST(DetailRunner, WritesSummaryTradeLogsAndQuarterlyFiles) {
    FixtureRun f;
    ASSERT_TRUE(f.load());
    const auto results = f.run({0, 3}, true);

    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "wfo_gpu_test_detail_outputs";
    std::filesystem::remove_all(dir);
    ASSERT_TRUE(detail::write_all_detail_outputs(dir.string(), results, f.md.bars()));

    const std::string summary = (dir / "detail_summary.csv").string();
    EXPECT_EQ(first_line(summary).rfind("strategy_id,total_trades,", 0), 0u);
    EXPECT_EQ(count_lines(summary), 1 + static_cast<int>(results.size()));

    for (const auto& dr : results) {
        const std::string id     = std::to_string(dr.strategy_id);
        const std::string trades = (dir / "trades" / ("strategy_" + id + ".csv")).string();
        const std::string quarts = (dir / "quarterly" / ("strategy_" + id + ".csv")).string();
        EXPECT_EQ(first_line(trades).rfind("trade_idx,entry_bar,exit_bar,", 0), 0u);
        EXPECT_EQ(count_lines(trades), 1 + static_cast<int>(dr.trades.size()));
        EXPECT_EQ(first_line(quarts), "year,quarter,pnl,trades");
        EXPECT_EQ(count_lines(quarts), 1 + static_cast<int>(dr.quarterly.size()));
    }
    std::filesystem::remove_all(dir);
}
