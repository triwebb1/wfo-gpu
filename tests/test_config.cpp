// Run-config parsing: defaults, every section, and error reporting.

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "config.h"

#ifndef WFO_GPU_SOURCE_DIR
#error "WFO_GPU_SOURCE_DIR must be defined by the build"
#endif

TEST(Config, EmptyDocumentGivesDefaults) {
    const auto r = parse_run_config("");
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(r.config.output, "output");
    EXPECT_EQ(r.config.session.name, "us_futures");
    EXPECT_EQ(r.config.device, DeviceType::AUTO);
    EXPECT_FALSE(r.config.screen_start.is_set());
    EXPECT_FALSE(r.config.max_cpu_pct.has_value());
}

TEST(Config, ParsesEverySection) {
    const auto r = parse_run_config(R"(
[data]
bars = "data/nq.bin"
output = "out/nq"

[instrument]
symbol = "NQ"
tick_size = 0.25
tick_value = 5.0
spread_ticks = 1
slippage_ticks = 1.5
capital = 25000

[session]
model = "us_futures"
core_hours = ["09:30", "11:30", "16:00"]
halt_gap_minutes = 10

[screen]
start = "2024-01-01"
end = 50000
max_strategies = 250000
robustness_floor = 0.4
entry_lag = 2

[floors]
min_trades = 40
min_profit_factor = 1.1
min_annual_roc = 5.0
max_negative_years = 1

[compute]
device = "native"
indicators_on_device = true

[resources]
max_cpu_pct = 75
max_ram_gb = 12.5
)");
    ASSERT_TRUE(r.ok()) << r.error;
    const RunConfig& c = r.config;
    EXPECT_EQ(c.bars, "data/nq.bin");
    EXPECT_EQ(c.output, "out/nq");
    EXPECT_EQ(c.symbol, "NQ");
    EXPECT_FLOAT_EQ(c.cost.tick_size, 0.25f);
    EXPECT_FLOAT_EQ(c.cost.tick_value, 5.0f);
    EXPECT_FLOAT_EQ(c.cost.point_value, 20.0f);  // derived
    EXPECT_EQ(c.cost.round_trip_ticks, 1);
    EXPECT_FLOAT_EQ(c.cost.slippage_ticks, 1.5f);
    EXPECT_FLOAT_EQ(c.cost.capital_required, 25000.0f);
    EXPECT_TRUE(c.cost.is_valid());
    EXPECT_EQ(c.session.core_split_sec, 11 * 3600 + 1800);
    EXPECT_EQ(c.session.halt_gap_sec, 600);
    EXPECT_EQ(c.screen_start.timestamp, "2024-01-01");
    EXPECT_EQ(c.screen_end.index.value_or(-1), 50000);
    EXPECT_EQ(c.max_strategies, 250000);
    EXPECT_FLOAT_EQ(c.robustness_floor, 0.4f);
    EXPECT_EQ(c.entry_lag, 2);
    EXPECT_EQ(c.floors.min_total_trades, 40);
    EXPECT_FLOAT_EQ(c.floors.min_profit_factor, 1.1f);
    EXPECT_EQ(c.floors.max_negative_years, 1);
    EXPECT_EQ(c.device, DeviceType::NATIVE);
    EXPECT_TRUE(c.indicators_on_device);
    EXPECT_FLOAT_EQ(c.max_cpu_pct.value_or(0), 75.0f);
    EXPECT_FLOAT_EQ(c.max_ram_gb.value_or(0), 12.5f);
}

TEST(Config, ContinuousSessionModel) {
    const auto r = parse_run_config(R"(
[session]
model = "continuous"
core_hours = ["12:00", "15:00", "20:00"]
)");
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(r.config.session.clock, SessionModel::Clock::UTC);
    EXPECT_FALSE(r.config.session.flat_at_close);
    EXPECT_FALSE(r.config.session.session_signals);
    EXPECT_EQ(r.config.session.core_open_sec, 12 * 3600);
}

// The session signals assume the 09:30-16:00 ET equity session; other core
// hours switch them off. Moving only the split keeps them.
TEST(Config, CustomCoreHoursDisableSessionSignals) {
    const auto gold = parse_run_config(R"(
[session]
model = "us_futures"
core_hours = ["08:20", "11:00", "13:30"]
)");
    ASSERT_TRUE(gold.ok()) << gold.error;
    EXPECT_FALSE(gold.config.session.session_signals);

    const auto split = parse_run_config(R"(
[session]
core_hours = ["09:30", "11:30", "16:00"]
)");
    ASSERT_TRUE(split.ok()) << split.error;
    EXPECT_TRUE(split.config.session.session_signals);
}

TEST(Config, RejectsTyposAndBadValues) {
    struct Case {
        const char* toml;
        const char* expect;
    };
    const Case cases[] = {
        {"[dta]\nbars = 'x'\n", "unknown section [dta]"},
        {"[data]\nbar = 'x'\n", "unknown key [data] bar"},
        {"[instrument]\ntick_size = 0.25\n", "needs tick_size and tick_value"},
        {"[instrument]\ntick_size = -1\ntick_value = 1\n", "tick_size must be positive"},
        {"[instrument]\ntick_size = 0.25\ntick_value = 12.5\npoint_value = 40\n",
         "tick_value must equal"},
        {"[session]\nmodel = 'lunar'\n", "model must be"},
        {"[session]\ncore_hours = ['9:30', '12:00', '16:00']\n", "HH:MM"},
        {"[session]\ncore_hours = ['12:00', '10:00', '16:00']\n", "open <= split <= close"},
        {"[screen]\nentry_lag = 3\n", "entry_lag must be 1 or 2"},
        {"[screen]\nstart = -5\n", "must not be negative"},
        {"[screen]\nmax_strategies = 'many'\n", "must be an integer"},
        {"[compute]\ndevice = 'tpu'\n", "device must be"},
        {"[resources]\nmax_cpu_pct = 150\n", "max_cpu_pct must be in"},
        {"[data]\nbars = \n", "config:"},
    };
    for (const auto& c : cases) {
        const auto r = parse_run_config(c.toml);
        EXPECT_FALSE(r.ok()) << c.toml;
        EXPECT_NE(r.error.find(c.expect), std::string::npos) << "for:\n"
                                                             << c.toml << "got: " << r.error;
    }
}

TEST(Config, ClockTimes) {
    int32_t s = 0;
    EXPECT_TRUE(parse_clock_time("00:00", s));
    EXPECT_EQ(s, 0);
    EXPECT_TRUE(parse_clock_time("23:59", s));
    EXPECT_EQ(s, 86340);
    for (const char* bad : {"24:00", "9:30", "09:60", "0930", "ab:cd", ""})
        EXPECT_FALSE(parse_clock_time(bad, s)) << bad;
}

// Every shipped example config parses.
TEST(Config, ShippedExamplesParse) {
    const auto dir   = std::filesystem::path(WFO_GPU_SOURCE_DIR) / "config";
    int        count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".toml") continue;
        const auto r = load_run_config(entry.path().string());
        EXPECT_TRUE(r.ok()) << entry.path() << ": " << r.error;
        ++count;
    }
    EXPECT_GE(count, 3);
}

TEST(Config, TracksWhetherInstrumentWasConfigured) {
    EXPECT_FALSE(parse_run_config("").config.instrument_configured);
    EXPECT_TRUE(parse_run_config("[instrument]\ntick_size = 1\ntick_value = 1\n")
                    .config.instrument_configured);
    for (const char* bad : {"[floors]\nmin_trades = -1\n", "[floors]\nmax_negative_years = -2\n",
                            "[floors]\nmin_profit_factor = -0.5\n"}) {
        const auto r = parse_run_config(bad);
        EXPECT_FALSE(r.ok()) << bad;
        EXPECT_NE(r.error.find("must not be negative"), std::string::npos) << r.error;
    }
}
