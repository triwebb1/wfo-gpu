// Synthetic bar generator: determinism, calendar layout, and bar sanity.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <vector>

#include "bar_io.h"
#include "civil_time.h"
#include "market_data.h"
#include "synthetic.h"
#include "test_fixture.h"

namespace {

uint64_t fnv1a(const void* data, size_t len) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t    h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t hash_bars(const std::vector<Bar>& bars) {
    return fnv1a(bars.data(), bars.size() * sizeof(Bar));
}

std::string temp_path(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

}  // namespace

TEST(Synthetic, SameSeedSameBytes) {
    SyntheticConfig cfg;
    cfg.num_days = 2;
    EXPECT_EQ(hash_bars(generate_synthetic_bars(cfg)), hash_bars(generate_synthetic_bars(cfg)));
    SyntheticConfig other = cfg;
    other.seed            = cfg.seed + 1;
    EXPECT_NE(hash_bars(generate_synthetic_bars(cfg)), hash_bars(generate_synthetic_bars(other)));
}

// The generator avoids platform-dependent math, so a fixed config hashes to
// the same value on every compiler and OS. CI runs this on Linux and Windows.
TEST(Synthetic, OutputIsIdenticalAcrossPlatforms) {
    const auto bars = generate_synthetic_bars(test_fixture::fixture_config());
    EXPECT_EQ(hash_bars(bars), test_fixture::FIXTURE_BARS_FNV1A);
}

// The committed fixture is exactly what the generator produces.
TEST(Synthetic, CommittedFixtureMatchesGenerator) {
    const auto        cfg         = test_fixture::fixture_config();
    const auto        bars        = generate_synthetic_bars(cfg);
    const std::string regenerated = temp_path("wfo_gpu_fixture_regen.bin");
    ASSERT_TRUE(write_bar_file(regenerated, cfg.symbol, cfg.timeframe_sec, bars));

    auto read_all = [](const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(f), {});
    };
    const auto committed = read_all(test_fixture::fixture_path());
    ASSERT_FALSE(committed.empty()) << "missing " << test_fixture::fixture_path();
    EXPECT_TRUE(committed == read_all(regenerated))
        << "tests/data fixture is stale; regenerate it with tools/make_fixtures";
}

TEST(Synthetic, BarsAreWellFormed) {
    SyntheticConfig cfg;
    cfg.num_days    = 5;
    cfg.jump_prob   = 0.01;  // exercise the jump path
    const auto bars = generate_synthetic_bars(cfg);
    ASSERT_FALSE(bars.empty());
    for (size_t i = 0; i < bars.size(); ++i) {
        const Bar& b = bars[i];
        ASSERT_LE(b.low, std::min(b.open, b.close)) << i;
        ASSERT_GE(b.high, std::max(b.open, b.close)) << i;
        ASSERT_GT(b.low, 0.0f) << i;
        ASSERT_GT(b.volume, 0u) << i;
        for (float px : {b.open, b.high, b.low, b.close}) {
            const double ticks = px / cfg.tick_size;
            ASSERT_NEAR(ticks, std::round(ticks), 1e-6) << "price off the tick grid at " << i;
        }
        if (i > 0) ASSERT_GT(b.timestamp, bars[i - 1].timestamp) << i;
    }
}

TEST(Synthetic, UsFuturesCalendarLayout) {
    SyntheticConfig cfg;
    cfg.start_year  = 2024;
    cfg.start_month = 3;
    cfg.start_day   = 4;  // Monday; the run spans the March DST change
    cfg.num_days    = 10;
    const auto bars = generate_synthetic_bars(cfg);
    EXPECT_EQ(bars.size(), static_cast<size_t>(10 * 23 * 120));

    for (const Bar& b : bars) {
        const int64_t sec   = b.timestamp / civil::NS_PER_SEC;
        const int64_t local = sec + civil::us_eastern_offset_sec(sec);
        const int64_t tod   = ((local % 86400) + 86400) % 86400;
        const int     wd    = civil::weekday_from_days(civil::floor_div(local, 86400));
        ASSERT_FALSE(tod >= 17 * 3600 && tod < 18 * 3600) << "bar inside the daily break";
        ASSERT_NE(wd, 6) << "bar on Saturday";
        if (wd == 5) ASSERT_LT(tod, 17 * 3600) << "bar after Friday close";
        if (wd == 0) ASSERT_GE(tod, 18 * 3600) << "bar before Sunday open";
    }

    const std::string path = temp_path("wfo_gpu_synth_us.bin");
    ASSERT_TRUE(write_bar_file(path, cfg.symbol, cfg.timeframe_sec, bars));
    MarketData md;
    ASSERT_TRUE(md.load(path));
    EXPECT_EQ(md.session_count(), 10);
    for (int sd = 0; sd < md.session_count(); ++sd) {
        const int first = md.first_bar_of_session(sd);
        EXPECT_EQ(md.session(first).time_of_day_sec, 18 * 3600) << "session " << sd;
    }
}

TEST(Synthetic, ContinuousCalendarLayout) {
    SyntheticConfig cfg;
    cfg.calendar          = SyntheticCalendar::CONTINUOUS;
    cfg.timeframe_sec     = 60;
    cfg.start_price       = 60000.0;
    cfg.tick_size         = 0.5;
    cfg.annual_volatility = 0.6;
    cfg.num_days          = 9;  // includes a weekend
    const auto bars       = generate_synthetic_bars(cfg);
    ASSERT_EQ(bars.size(), static_cast<size_t>(9 * 1440));
    for (size_t i = 1; i < bars.size(); ++i)
        ASSERT_EQ(bars[i].timestamp - bars[i - 1].timestamp, 60 * civil::NS_PER_SEC);

    const std::string path = temp_path("wfo_gpu_synth_cont.bin");
    ASSERT_TRUE(write_bar_file(path, "BTC", 60, bars));
    MarketData md;
    ASSERT_TRUE(md.load(path, SessionModel::continuous()));
    EXPECT_EQ(md.session_count(), 9);
    // No daily close in a 24/7 market, so no bar forces a flat exit.
    for (int i = 0; i < md.bar_count(); ++i)
        ASSERT_EQ(md.session(i).flags & SESSION_CLOSE_BAR, 0) << i;
}

// Realized volatility lands near the configured value and the zero-drift
// series has no significant mean return.
TEST(Synthetic, StatisticsMatchConfiguration) {
    SyntheticConfig cfg;
    cfg.num_days           = 120;
    cfg.jump_prob          = 0.0;
    cfg.regime_switch_prob = 0.0;
    const auto bars        = generate_synthetic_bars(cfg);

    double sum = 0.0, sum2 = 0.0;
    int    n = 0;
    for (size_t i = 1; i < bars.size(); ++i) {
        const double r = std::log(static_cast<double>(bars[i].close) / bars[i - 1].close);
        sum += r;
        sum2 += r * r;
        ++n;
    }
    const double mean          = sum / n;
    const double var           = sum2 / n - mean * mean;
    const double bars_per_year = 252.0 * 23.0 * 3600.0 / cfg.timeframe_sec;
    const double annual_vol    = std::sqrt(var * bars_per_year);
    EXPECT_GT(annual_vol, 0.7 * cfg.annual_volatility);
    EXPECT_LT(annual_vol, 1.5 * cfg.annual_volatility);
    EXPECT_LT(std::abs(mean), 4.0 * std::sqrt(var / n));
}

TEST(Synthetic, RejectsBadConfig) {
    SyntheticConfig cfg;
    cfg.timeframe_sec = 7;  // does not divide a day
    EXPECT_THROW(generate_synthetic_bars(cfg), std::invalid_argument);
    cfg           = SyntheticConfig{};
    cfg.tick_size = 0.0;
    EXPECT_THROW(generate_synthetic_bars(cfg), std::invalid_argument);
    cfg          = SyntheticConfig{};
    cfg.num_days = 0;
    EXPECT_THROW(generate_synthetic_bars(cfg), std::invalid_argument);
}

TEST(CivilTime, KnownDates) {
    EXPECT_EQ(civil::days_from_civil(1970, 1, 1), 0);
    EXPECT_EQ(civil::days_from_civil(2000, 3, 1), 11017);
    const auto d = civil::civil_from_days(19723);
    EXPECT_EQ(d.year, 2024);
    EXPECT_EQ(d.month, 1);
    EXPECT_EQ(d.day, 1);
    EXPECT_EQ(civil::weekday_from_days(0), 4);  // 1970-01-01 was a Thursday
    // 2024 DST: starts 2024-03-10 07:00 UTC, ends 2024-11-03 06:00 UTC.
    const int64_t start = civil::days_from_civil(2024, 3, 10) * 86400 + 7 * 3600;
    const int64_t end   = civil::days_from_civil(2024, 11, 3) * 86400 + 6 * 3600;
    EXPECT_EQ(civil::us_eastern_offset_sec(start - 1), civil::US_EASTERN_EST);
    EXPECT_EQ(civil::us_eastern_offset_sec(start), civil::US_EASTERN_EDT);
    EXPECT_EQ(civil::us_eastern_offset_sec(end - 1), civil::US_EASTERN_EDT);
    EXPECT_EQ(civil::us_eastern_offset_sec(end), civil::US_EASTERN_EST);
    EXPECT_EQ(civil::us_eastern_to_utc_sec(2024, 7, 1, 9 * 3600 + 1800),
              civil::days_from_civil(2024, 7, 1) * 86400 + 13 * 3600 + 1800);
}

// Planted trend regimes show up as a variance ratio well above 1: returns
// over long horizons vary more than the sum of their one-bar parts. The
// plain random walk stays near 1.
TEST(Synthetic, TrendRegimesPlantPersistentDrift) {
    auto variance_ratio = [](const std::vector<Bar>& bars, int horizon) {
        std::vector<double> lp(bars.size());
        for (size_t i = 0; i < bars.size(); ++i)
            lp[i] = std::log(static_cast<double>(bars[i].close));
        double v1 = 0.0, vh = 0.0;
        int    n1 = 0, nh = 0;
        for (size_t i = 1; i < lp.size(); ++i, ++n1)
            v1 += (lp[i] - lp[i - 1]) * (lp[i] - lp[i - 1]);
        for (size_t i = horizon; i < lp.size(); i += horizon, ++nh)
            vh += (lp[i] - lp[i - horizon]) * (lp[i] - lp[i - horizon]);
        return (vh / nh) / (horizon * v1 / n1);
    };
    SyntheticConfig cfg;
    cfg.num_days          = 120;
    cfg.jump_prob         = 0.0;
    cfg.gap_sigmas        = 0.0;
    const double plain    = variance_ratio(generate_synthetic_bars(cfg), 500);
    cfg.trend_strength    = 0.15;
    const double trending = variance_ratio(generate_synthetic_bars(cfg), 500);
    EXPECT_GT(plain, 0.6);
    EXPECT_LT(plain, 1.6);
    EXPECT_GT(trending, 2.0 * plain);
}
