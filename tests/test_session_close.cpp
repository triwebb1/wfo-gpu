// ---------------------------------------------------------------------------
// Daily-close behaviour, session flags, session-signal timing, calendar keys
// and bar-file validation.
//
// Flat at the close is a guarantee of the engine: no position carries
// through a daily break, whatever exit the strategy uses and whether or not
// the bar in the final slot exists. Both engines must agree, so every engine
// case also runs through CLBacktest when an OpenCL device is present.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "test_fixture.h"
#include "test_helpers.h"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "backtest_engine.h"
#include "bar_io.h"
#include "compute/cl_backtest.h"
#include "compute/cl_context.h"
#include "market_data.h"
#include "session_model.h"
#include "session_signals/session_signal_base.h"

namespace {

constexpr int32_t HOUR = 3600;

// A hand-built market: each bar is (session day, local time of day), with
// the session's close time. Prices rise by 1.0 a bar so every exit price
// identifies its bar.
struct Market {
    std::vector<Bar>         bars;
    std::vector<SessionInfo> sessions;

    void add(int session_day, int32_t time_of_day_sec, int32_t close_sec = 17 * HOUR,
             bool early_close = false) {
        const int     i  = static_cast<int>(bars.size());
        const int64_t ts = (1'704'153'600LL + static_cast<int64_t>(i) * 30) * NS_PER_SEC;
        const float   o  = 100.0f + static_cast<float>(i);
        bars.push_back(make_bar(ts, o, o + 0.75f, o - 0.25f, o + 0.5f));
        SessionInfo s{};
        s.session_day       = session_day;
        s.intra_session_idx = i;
        s.time_of_day_sec   = time_of_day_sec;
        s.session_close_sec = close_sec;
        s.early_close       = early_close ? 1 : 0;
        sessions.push_back(s);
    }

    void stamp() {
        stamp_session_flags(sessions.data(), static_cast<int>(sessions.size()),
                            SessionModel::us_futures(), 30);
    }

    int size() const { return static_cast<int>(bars.size()); }
};

StrategyConfig one_signal(StrategyConfig::ExitMode exit_mode, int timeout = 1000) {
    StrategyConfig c;
    c.num_indicators       = 1;
    c.indicator_indices[0] = 0;
    c.weights[0]           = 1.0f;
    c.entry_threshold      = 0.5f;
    c.exit_mode            = exit_mode;
    c.exit_timeout_bars    = timeout;
    return c;
}

std::vector<Trade> run_cpu(const Market& m, const std::vector<float>& signal,
                           const StrategyConfig& config, BacktestResult* result = nullptr) {
    const float*   series[] = {signal.data()};
    BacktestEngine engine;
    BacktestResult r = engine.run(m.bars.data(), m.sessions.data(), 0, m.size(), series, 1, nullptr,
                                  config, CostModel::es(), 1);
    if (result) *result = r;
    return engine.last_trades();
}

// The kernel must reach the same totals as the CPU engine.
void expect_gpu_matches(const Market& m, const std::vector<float>& signal,
                        const StrategyConfig& config) {
#ifdef HAS_OPENCL
    CLContext ctx;
    if (!ctx.initialize(DeviceType::AUTO)) return;  // covered by the CPU assertions
    BacktestResult cpu;
    run_cpu(m, signal, config, &cpu);

    const float*                series[] = {signal.data()};
    CLBacktest                  bt;
    std::vector<BacktestResult> gpu;
    ASSERT_TRUE(bt.build_kernel(ctx));
    bt.evaluate(ctx, m.bars.data(), 0, series, 1, m.size(), 0, m.size(), {config}, CostModel::es(),
                1, gpu, EarlyTermination{}, m.sessions.data());
    ASSERT_EQ(gpu.size(), 1u);
    EXPECT_EQ(gpu[0].total_trades, cpu.total_trades);
    EXPECT_NEAR(gpu[0].total_pnl, cpu.total_pnl, 0.01f);
#else
    (void)m;
    (void)signal;
    (void)config;
#endif
}

}  // namespace

// ---- Flags ---------------------------------------------------------------

TEST(SessionFlags, MarkCloseBarAndBarsAfterTheClose) {
    Market m;
    m.add(0, 17 * HOUR - 60);    // 16:59:00
    m.add(0, 17 * HOUR - 30);    // 16:59:30, final slot
    m.add(0, 17 * HOUR + 1800);  // 17:30, after the close
    m.add(1, 18 * HOUR);         // next session opens
    m.add(1, 16 * HOUR + 1800);  // 16:30, last bar of a session missing its final slot
    m.add(2, 18 * HOUR);
    m.stamp();

    EXPECT_FALSE(m.sessions[0].flags & SESSION_CLOSE_BAR);
    EXPECT_TRUE(m.sessions[1].flags & SESSION_CLOSE_BAR);
    EXPECT_TRUE(m.sessions[2].flags & SESSION_AFTER_CLOSE);
    EXPECT_FALSE(m.sessions[2].flags & SESSION_CLOSE_BAR);
    EXPECT_FALSE(m.sessions[3].flags & (SESSION_CLOSE_BAR | SESSION_AFTER_CLOSE));
    EXPECT_TRUE(m.sessions[4].flags & SESSION_CLOSE_BAR) << "the session ends after this bar";
}

TEST(SessionFlags, EarlyCloseMovesTheCloseAndClearsCoreHours) {
    Market m;
    m.add(0, 13 * HOUR - 30, 13 * HOUR, true);  // 12:59:30, final slot of a 13:00 close
    m.add(0, 14 * HOUR, 13 * HOUR, true);       // 14:00, after the early close
    m.stamp();
    EXPECT_TRUE(m.sessions[0].flags & SESSION_CLOSE_BAR);
    EXPECT_TRUE(m.sessions[0].flags & SESSION_IN_CORE);
    EXPECT_EQ(m.sessions[1].flags, SESSION_AFTER_CLOSE);
}

// ---- Engine --------------------------------------------------------------

// A signal exit raised on the close bar fills at that bar's close, not at
// the next session's open.
TEST(SessionClose, SignalExitOnTheCloseBarFillsAtTheClose) {
    Market m;
    for (int k = 5; k >= 0; --k) m.add(0, 17 * HOUR - 30 * (k + 1));  // 16:57:00 .. 16:59:30
    for (int k = 0; k < 4; ++k) m.add(1, 18 * HOUR + 30 * k);
    m.stamp();
    const int close_bar = 5;
    ASSERT_TRUE(m.sessions[close_bar].flags & SESSION_CLOSE_BAR);

    std::vector<float> signal(m.size(), 1.0f);
    signal[close_bar] = -1.0f;  // reversal raised on the close bar
    for (int i = close_bar + 1; i < m.size(); ++i) signal[i] = 0.0f;

    const auto trades = run_cpu(m, signal, one_signal(StrategyConfig::SIGNAL_REVERSAL));
    ASSERT_FALSE(trades.empty());
    EXPECT_EQ(trades[0].exit_bar, close_bar);
    EXPECT_FLOAT_EQ(trades[0].exit_price, m.bars[close_bar].close);
    EXPECT_EQ(trades[0].fill_type, FillType::SESSION_CLOSE);
    expect_gpu_matches(m, signal, one_signal(StrategyConfig::SIGNAL_REVERSAL));
}

// Without the final-slot bar, the last bar before the break flattens.
TEST(SessionClose, MissingFinalSlotStillFlattens) {
    Market m;
    for (int k = 5; k >= 1; --k) m.add(0, 17 * HOUR - 30 * (k + 1));  // 16:57:00 .. 16:59:00
    for (int k = 0; k < 4; ++k) m.add(1, 18 * HOUR + 30 * k);
    m.stamp();
    const int last_bar = 4;

    const std::vector<float> signal(m.size(), 1.0f);
    const auto               trades = run_cpu(m, signal, one_signal(StrategyConfig::BAR_TIMEOUT));
    ASSERT_FALSE(trades.empty());
    EXPECT_EQ(trades[0].exit_bar, last_bar);
    EXPECT_EQ(trades[0].fill_type, FillType::SESSION_CLOSE);
    for (const auto& t : trades)
        EXPECT_EQ(m.sessions[t.entry_bar].session_day, m.sessions[t.exit_bar].session_day)
            << "a position crossed the daily break";
    expect_gpu_matches(m, signal, one_signal(StrategyConfig::BAR_TIMEOUT));
}

// Bars after an early close take no entries, including one signalled on the
// close bar that would fill on the first bar after it.
TEST(SessionClose, NoEntriesBetweenTheCloseAndTheNextOpen) {
    Market m;
    m.add(0, 13 * HOUR - 60, 13 * HOUR, true);  // 12:59:00
    m.add(0, 13 * HOUR - 30, 13 * HOUR, true);  // 12:59:30, close bar
    for (int k = 0; k < 4; ++k) m.add(0, 13 * HOUR + 30 * k, 13 * HOUR, true);  // stray bars
    for (int k = 0; k < 3; ++k) m.add(1, 18 * HOUR + 30 * k);
    m.stamp();

    const std::vector<float> signal(m.size(), 1.0f);
    const auto trades = run_cpu(m, signal, one_signal(StrategyConfig::BAR_TIMEOUT, 1));
    for (const auto& t : trades) {
        EXPECT_FALSE(m.sessions[t.entry_bar].flags & SESSION_AFTER_CLOSE)
            << "entered on bar " << t.entry_bar << " after the close";
    }
    expect_gpu_matches(m, signal, one_signal(StrategyConfig::BAR_TIMEOUT, 1));
}

// An early-terminated run closes its open position at the checkpoint bar;
// it never books the price move of bars it did not process.
TEST(SessionClose, EarlyTerminationClosesAtTheCheckpointBar) {
    std::vector<float> closes(100);
    for (int i = 0; i < 100; ++i) closes[i] = 100.0f + static_cast<float>(i);
    std::vector<Bar> bars;
    for (int i = 0; i < 100; ++i)
        bars.push_back(make_bar((1'262'304'000LL + i * 30LL) * NS_PER_SEC, closes[i],
                                closes[i] + 0.5f, closes[i] - 0.5f, closes[i]));

    const std::vector<float> signal(100, 1.0f);
    const float*             series[] = {signal.data()};
    EarlyTermination         et;
    et.enabled = true;  // no closed trades at 60%: below min_trades_at_check, so it aborts
    BacktestEngine engine;
    engine.run(bars.data(), 0, 100, series, 1, one_signal(StrategyConfig::BAR_TIMEOUT),
               CostModel::es(), 1, et);
    const auto& trades = engine.last_trades();
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades[0].fill_type, FillType::FORCED_CLOSE);
    EXPECT_EQ(trades[0].exit_bar, 60);
}

// ---- Session-signal timing -------------------------------------------------

// A signal available only from the core close describes the whole regular
// session: bars see the previous session's value until it is available,
// then the current one. Earlier signals are neutral until available.
TEST(SessionSignalTiming, PostCloseSignalsCarryIntoTheNextSession) {
    MarketData md;
    ASSERT_TRUE(md.load(test_fixture::fixture_path()));
    const int days = md.session_count();
    ASSERT_GE(days, 3);
    std::vector<float> values(days);
    for (int d = 0; d < days; ++d) values[d] = static_cast<float>(d + 1) / 10.0f;

    const int          n    = md.bar_count();
    const int32_t      open = md.session_model().session_open_sec;
    std::vector<float> post_close(n), morning(n);
    broadcast_session_signal_to_bars(md, values.data(), 0, days, session_time::RTH_CLOSE, 0, n,
                                     post_close.data());
    broadcast_session_signal_to_bars(md, values.data(), 0, days, 10 * HOUR, 0, n, morning.data());

    for (int i = 0; i < n; ++i) {
        const SessionInfo& s       = md.session(i);
        const int32_t      elapsed = seconds_since_open(s.time_of_day_sec, open);
        const bool  after_close    = elapsed >= seconds_since_open(session_time::RTH_CLOSE, open);
        const bool  after_ten      = elapsed >= seconds_since_open(10 * HOUR, open);
        const float carried        = s.session_day > 0 ? values[s.session_day - 1] : 0.0f;
        ASSERT_FLOAT_EQ(post_close[i], after_close ? values[s.session_day] : carried) << i;
        ASSERT_FLOAT_EQ(morning[i], after_ten ? values[s.session_day] : 0.0f) << i;
    }
}

// ---- Calendar keys -----------------------------------------------------------

// A session is named by the date it closes on. The fixture's first session
// opens Sunday 2024-01-07 at 18:00 ET and closes Monday 2024-01-08, which is
// in ISO week 2.
TEST(CalendarKeys, SessionsAreKeyedByTheirCloseDate) {
    MarketData md;
    ASSERT_TRUE(md.load(test_fixture::fixture_path()));
    ASSERT_GE(md.session_count(), 1);
    EXPECT_EQ(md.session_day_to_week()[0], 202402);
    EXPECT_EQ(md.session_day_to_month()[0], 202401);
    EXPECT_EQ(md.session_day_to_year()[0], 2024);
}

// ---- Bar-file validation -----------------------------------------------------

// The header's bar length must not exceed the spacing of the bars: an
// overstated length would mark higher-timeframe bars complete too early.
TEST(BarFileValidation, RejectsATimeframeLongerThanTheBarSpacing) {
    std::vector<Bar> bars;
    for (int i = 0; i < 10; ++i)
        bars.push_back(make_bar((1'704'153'600LL + i * 30LL) * NS_PER_SEC, 100, 101, 99, 100));
    const std::string path =
        (std::filesystem::temp_directory_path() / "overstated_timeframe.bin").string();

    std::string error;
    EXPECT_FALSE(write_bar_file(path, "T", 60, bars, &error));  // 30 s bars labelled 60 s
    EXPECT_NE(error.find("closer together"), std::string::npos) << error;

    // A file written elsewhere with the same mistake is rejected on load.
    ASSERT_TRUE(write_bar_file(path, "T", 30, bars));
    {
        MarketData md;
        EXPECT_TRUE(md.load(path));
    }
    {
        std::fstream   f(path, std::ios::in | std::ios::out | std::ios::binary);
        const uint32_t sixty = 60;
        f.seekp(offsetof(FileHeader, timeframe_sec));
        f.write(reinterpret_cast<const char*>(&sixty), sizeof(sixty));
    }
    {
        MarketData md;
        EXPECT_FALSE(md.load(path));
    }
    std::filesystem::remove(path);
}
