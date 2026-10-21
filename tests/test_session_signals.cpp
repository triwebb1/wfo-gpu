// =============================================================================
// Session-Structure Signal Tests
//
// At least 2 tests per module (16+ total):
//   1. Known-day test with synthetic data verifying correct signal values
//   2. No-lookahead test verifying signal doesn't change when future bars are modified
//
// Framework: Google Test v1.14
// =============================================================================

// Suppress MSVC deprecation warning for fopen (used in test helper)
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "market_data.h"
#include "session_signals/session_signal_base.h"
#include "session_signals/opening_range.h"
#include "session_signals/gap_analysis.h"
#include "session_signals/multiday_patterns.h"
#include "session_signals/intraday_structure.h"
#include "session_signals/level_interaction.h"
#include "session_signals/overnight_context.h"
#include "session_signals/calendar.h"
#include "session_signals/volatility_regime.h"

// =============================================================================
// Helpers
// =============================================================================

static constexpr int64_t NS_PER_SEC = 1'000'000'000LL;

static bool is_nan(float v) {
    return std::isnan(v);
}
static bool is_valid(float v) {
    return !std::isnan(v);
}
static bool approx_eq(float a, float b, float tol = 0.01f) {
    if (is_nan(a) && is_nan(b)) return true;
    if (is_nan(a) || is_nan(b)) return false;
    return std::abs(a - b) <= tol;
}

// ---------------------------------------------------------------------------
// SyntheticMarketBuilder -- creates in-memory MarketData from synthetic bars.
//
// Constructs a binary file on disk, loads it via MarketData::load().
// Each session day has ETH (18:00-09:30) + RTH (09:30-16:00) bars at 30s intervals.
// ---------------------------------------------------------------------------
class SyntheticMarketBuilder {
public:
    // DaySpec: per-day price specification
    struct DaySpec {
        float eth_open  = 4000.0f;  // overnight open
        float eth_close = 4001.0f;  // overnight close (just before RTH)
        float rth_open  = 4001.0f;
        float rth_high  = 4010.0f;
        float rth_low   = 3990.0f;
        float rth_close = 4005.0f;

        // Optional: specific time-period prices for intraday signals
        float or_close         = 0.0f;  // close at 10:00 (0 = use interpolation)
        float ib_close         = 0.0f;  // close at 10:30
        float lunch_high       = 0.0f;  // lunch period high
        float lunch_low        = 0.0f;  // lunch period low
        float afternoon_close  = 0.0f;
        float power_hour_close = 0.0f;
    };

    void add_day(const DaySpec& spec) { m_days.push_back(spec); }

    // Build and return a loaded MarketData.
    // Returns true on success.  The MarketData is moved into `out`.
    bool build(MarketData& out) {
        // Generate bars for each day
        std::vector<Bar> all_bars;

        // Base date: 2024-01-02 (Tuesday).
        // 2024-01-02 00:00:00 UTC = 1704153600 unix seconds.
        // Session day 0 starts at 18:00 ET Jan 2 = 23:00 UTC Jan 2.
        // EST = UTC-5, so ET_to_UTC = ET + 5 hours.
        static constexpr int64_t JAN2_MIDNIGHT_UTC = 1704153600LL;  // 2024-01-02 00:00 UTC
        static constexpr int64_t EST_OFFSET        = 5LL * 3600;    // add 5h to ET to get UTC

        for (size_t day = 0; day < m_days.size(); ++day) {
            const DaySpec& spec = m_days[day];

            // Each session day: ETH from 18:00 ET day N to RTH close at 16:00 ET day N+1
            // Calendar date of the session open = Jan 2 + day
            int64_t cal_day_utc_midnight =
                (JAN2_MIDNIGHT_UTC + static_cast<int64_t>(day) * 86400LL);

            // --- ETH bars: 18:00 ET to 09:29:30 ET next calendar day ---
            // 18:00 ET = 18*3600 seconds since midnight ET
            // UTC = ET + 5h, so 18:00 ET = 23:00 UTC same calendar day
            int64_t eth_start_utc = cal_day_utc_midnight + 18LL * 3600 + EST_OFFSET;
            // ETH ends just before RTH open at 09:30 ET next day
            // 09:30 ET next day = (cal_day + 1) midnight + 9.5h ET = (cal_day + 1) midnight + 9.5h + 5h UTC
            int64_t rth_open_utc =
                cal_day_utc_midnight + 86400LL + 9LL * 3600 + 30 * 60 + EST_OFFSET;

            // Generate ETH bars from eth_start to just before rth_open
            for (int64_t t = eth_start_utc; t < rth_open_utc; t += 30) {
                Bar b{};
                b.timestamp = t * NS_PER_SEC;

                float frac =
                    static_cast<float>(t - eth_start_utc) /
                    static_cast<float>(std::max<int64_t>(1, rth_open_utc - eth_start_utc - 30));
                float price = spec.eth_open + (spec.eth_close - spec.eth_open) * frac;
                b.open      = price - 0.25f;
                b.high      = price + 0.5f;
                b.low       = price - 0.5f;
                b.close     = price;
                b.volume    = 100;
                b._pad      = 0;
                all_bars.push_back(b);
            }

            // --- RTH bars: 09:30 ET to 15:59:30 ET (same next calendar day) ---
            // 16:00 ET = (cal_day+1) midnight + 16h ET = ... + 16h + 5h UTC
            int64_t rth_close_utc = cal_day_utc_midnight + 86400LL + 16LL * 3600 + EST_OFFSET;
            int     rth_bar_idx   = 0;
            int     rth_bars      = static_cast<int>((rth_close_utc - rth_open_utc) / 30);

            for (int64_t t = rth_open_utc; t < rth_close_utc; t += 30) {
                Bar b{};
                b.timestamp = t * NS_PER_SEC;

                // ET seconds since midnight for this bar
                int64_t et_sec_since_midnight = (t - EST_OFFSET) % 86400LL;
                if (et_sec_since_midnight < 0) et_sec_since_midnight += 86400LL;

                float time_pct = static_cast<float>(et_sec_since_midnight - 34200) /
                                 static_cast<float>(57600 - 34200);

                float price;
                if (time_pct < 0.077f) {
                    // First 30 min (OR)
                    float target     = (spec.or_close != 0.0f)
                                           ? spec.or_close
                                           : spec.rth_open + (spec.rth_close - spec.rth_open) * 0.3f;
                    float local_frac = time_pct / 0.077f;
                    price            = spec.rth_open + (target - spec.rth_open) * local_frac;
                } else if (time_pct < 0.154f) {
                    // 10:00-10:30 (IB second half)
                    float or_end_price =
                        (spec.or_close != 0.0f)
                            ? spec.or_close
                            : spec.rth_open + (spec.rth_close - spec.rth_open) * 0.3f;
                    float ib_target  = (spec.ib_close != 0.0f)
                                           ? spec.ib_close
                                           : spec.rth_open + (spec.rth_close - spec.rth_open) * 0.5f;
                    float local_frac = (time_pct - 0.077f) / 0.077f;
                    price            = or_end_price + (ib_target - or_end_price) * local_frac;
                } else {
                    // Rest of day: smooth interpolation to close
                    float ib_end_price =
                        (spec.ib_close != 0.0f)
                            ? spec.ib_close
                            : spec.rth_open + (spec.rth_close - spec.rth_open) * 0.5f;
                    float local_frac = (time_pct - 0.154f) / (1.0f - 0.154f);
                    price            = ib_end_price + (spec.rth_close - ib_end_price) * local_frac;
                }

                // Add some range around the price
                float range_scale = (spec.rth_high - spec.rth_low) / 40.0f;
                b.open            = price;
                b.high            = price + range_scale;
                b.low             = price - range_scale;
                b.close           = price;
                b.volume          = 200;
                b._pad            = 0;

                // Ensure overall high/low are reached at some point
                if (rth_bar_idx == rth_bars / 4) {
                    b.high = std::max(b.high, spec.rth_high);
                }
                if (rth_bar_idx == rth_bars * 3 / 4) {
                    b.low = std::min(b.low, spec.rth_low);
                }

                all_bars.push_back(b);
                rth_bar_idx++;
            }
        }

        return write_and_load(all_bars, out);
    }

    // Access the temporary file path (valid after build())
    const std::string& path() const { return m_path; }

    ~SyntheticMarketBuilder() {
        // Clean up temp file
        if (!m_path.empty()) {
            std::remove(m_path.c_str());
        }
    }

private:
    std::vector<DaySpec> m_days;
    std::string          m_path;
    static int           s_counter;

    bool write_and_load(const std::vector<Bar>& bars, MarketData& out) {
        // Create temp file with unique name
        m_path = "test_session_signals_temp_" + std::to_string(s_counter++) + ".bin";

        FileHeader hdr{};
        std::memcpy(hdr.magic, "WFBT", 4);
        hdr.version   = 1;
        hdr.bar_count = static_cast<uint32_t>(bars.size());
        std::memcpy(hdr.symbol, "ES\0\0\0\0\0\0", 8);
        hdr.timeframe_sec = 30;
        hdr.start_ts      = bars.front().timestamp;
        hdr.end_ts        = bars.back().timestamp;

        FILE* f = fopen(m_path.c_str(), "wb");
        if (!f) return false;

        fwrite(&hdr, sizeof(hdr), 1, f);
        fwrite(bars.data(), sizeof(Bar), bars.size(), f);
        fclose(f);

        return out.load(m_path);
    }
};

int SyntheticMarketBuilder::s_counter = 0;

// ---------------------------------------------------------------------------
// Build a standard 30-day market with consistent pricing for tests.
// Each day has similar range ~20 points, with slight variations.
// ---------------------------------------------------------------------------
static bool build_standard_market(SyntheticMarketBuilder& builder, MarketData& md,
                                  int num_days = 30) {
    for (int d = 0; d < num_days; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + static_cast<float>(d) * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close = base + (d % 2 == 0 ? 5.0f : -3.0f);  // alternating up/down
        builder.add_day(spec);
    }
    return builder.build(md);
}

// =============================================================================
// OPENING RANGE TESTS
// =============================================================================

TEST(OpeningRange, KnownDaySignals) {
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md));

    OpeningRangeSignals sig(5);
    ASSERT_EQ(sig.num_signals(), 8);

    // Compute for day 15 (enough warmup)
    int                day = 15;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // Signal 0: OR direction should be valid
    EXPECT_TRUE(is_valid(out[0]));
    EXPECT_GE(out[0], -1.0f);
    EXPECT_LE(out[0], 1.0f);

    // Signal 1: OR range size should be valid and in [0,1]
    EXPECT_TRUE(is_valid(out[1]));
    EXPECT_GE(out[1], 0.0f);
    EXPECT_LE(out[1], 1.0f);

    // Signal 4: IB direction should be valid
    EXPECT_TRUE(is_valid(out[4]));

    // Signal 5: IB range size should be valid
    EXPECT_TRUE(is_valid(out[5]));
    EXPECT_GE(out[5], 0.0f);
}

TEST(OpeningRange, NoLookahead) {
    // Build two identical markets, modify future bars in second one,
    // verify signals for earlier days remain identical.
    SyntheticMarketBuilder builder1, builder2;
    MarketData             md1, md2;

    int num_days = 25;
    for (int d = 0; d < num_days; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder1.add_day(spec);

        // Same spec for first 13 days, drastically different after
        if (d >= 13) {
            spec.rth_high  = base + 50.0f;
            spec.rth_low   = base - 50.0f;
            spec.rth_close = base + 40.0f;
        }
        builder2.add_day(spec);
    }

    ASSERT_TRUE(builder1.build(md1));
    ASSERT_TRUE(builder2.build(md2));

    OpeningRangeSignals sig(5);
    int                 test_day = 12;  // Day to test (well before modification at day 15)

    std::vector<float> out1(sig.num_signals()), out2(sig.num_signals());
    sig.compute(md1, test_day, test_day + 1, out1.data());
    sig.compute(md2, test_day, test_day + 1, out2.data());

    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(approx_eq(out1[s], out2[s]))
            << "Signal " << s << " differs: " << out1[s] << " vs " << out2[s];
    }
}

// =============================================================================
// GAP ANALYSIS TESTS
// =============================================================================

TEST(GapAnalysis, KnownDaySignals) {
    SyntheticMarketBuilder builder;

    // Create a specific gap-up scenario on day 15
    for (int d = 0; d < 20; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;

        if (d == 15) {
            // Gap up: open significantly above prev close
            spec.rth_open  = base + 15.0f;
            spec.rth_close = base + 18.0f;
        }
        builder.add_day(spec);
    }

    MarketData md;
    ASSERT_TRUE(builder.build(md));

    GapAnalysisSignals sig(5);
    ASSERT_EQ(sig.num_signals(), 6);

    int                day = 15;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // Signal 0: Gap direction should be positive (gap up)
    EXPECT_TRUE(is_valid(out[0]));
    EXPECT_GT(out[0], 0.0f);

    // Signal 1: Gap size should be positive (gap up, some bucket)
    EXPECT_TRUE(is_valid(out[1]));
    EXPECT_GT(out[1], 0.0f);

    // Signal 3: Gap into prev value area
    EXPECT_TRUE(is_valid(out[3]));
}

TEST(GapAnalysis, NoLookahead) {
    SyntheticMarketBuilder builder1, builder2;
    MarketData             md1, md2;

    int num_days = 25;
    for (int d = 0; d < num_days; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder1.add_day(spec);

        if (d >= 13) {
            spec.rth_open  = base + 30.0f;
            spec.rth_close = base + 35.0f;
        }
        builder2.add_day(spec);
    }

    ASSERT_TRUE(builder1.build(md1));
    ASSERT_TRUE(builder2.build(md2));

    GapAnalysisSignals sig(5);
    int                test_day = 12;

    std::vector<float> out1(sig.num_signals()), out2(sig.num_signals());
    sig.compute(md1, test_day, test_day + 1, out1.data());
    sig.compute(md2, test_day, test_day + 1, out2.data());

    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(approx_eq(out1[s], out2[s]))
            << "Signal " << s << " differs: " << out1[s] << " vs " << out2[s];
    }
}

// =============================================================================
// MULTI-DAY PATTERN TESTS
// =============================================================================

TEST(MultidayPatterns, KnownDaySignals) {
    SyntheticMarketBuilder builder;

    // Create an inside day scenario: day 14 range inside day 13
    for (int d = 0; d < 20; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;

        if (d == 13) {
            // Wide range day
            spec.rth_high  = base + 20.0f;
            spec.rth_low   = base - 20.0f;
            spec.rth_close = base + 10.0f;
        } else if (d == 14) {
            // Inside day (narrow range inside day 13)
            spec.rth_high  = base + 5.0f;
            spec.rth_low   = base - 5.0f;
            spec.rth_close = base + 2.0f;
        } else {
            spec.rth_high  = base + 10.0f;
            spec.rth_low   = base - 10.0f;
            spec.rth_close = base + 5.0f;
        }
        builder.add_day(spec);
    }

    MarketData md;
    ASSERT_TRUE(builder.build(md));

    MultidayPatternSignals sig(5);
    ASSERT_EQ(sig.num_signals(), 12);

    // Test day 15: yesterday (14) was inside day relative to day 13
    int                day = 15;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // Signal 0: Inside day should be 1.0 (yesterday was an inside day)
    EXPECT_TRUE(is_valid(out[0]));
    EXPECT_FLOAT_EQ(out[0], 1.0f);

    // Signal 10: Prev day direction should be valid
    EXPECT_TRUE(is_valid(out[10]));
}

TEST(MultidayPatterns, RunningDays) {
    SyntheticMarketBuilder builder;

    // Create 5 consecutive up-close days
    for (int d = 0; d < 20; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 5.0f;  // consistent uptrend
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;  // always closes up from prev day
        builder.add_day(spec);
    }

    MarketData md;
    ASSERT_TRUE(builder.build(md));

    MultidayPatternSignals sig(5);

    int                day = 15;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // Signal 6: Running days should be positive (consecutive up closes)
    EXPECT_TRUE(is_valid(out[6]));
    EXPECT_GT(out[6], 0.0f);
}

TEST(MultidayPatterns, NoLookahead) {
    SyntheticMarketBuilder builder1, builder2;
    MarketData             md1, md2;

    int num_days = 25;
    for (int d = 0; d < num_days; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder1.add_day(spec);

        if (d >= 13) {
            spec.rth_high  = base + 50.0f;
            spec.rth_low   = base - 50.0f;
            spec.rth_close = base + 40.0f;
        }
        builder2.add_day(spec);
    }

    ASSERT_TRUE(builder1.build(md1));
    ASSERT_TRUE(builder2.build(md2));

    MultidayPatternSignals sig(5);
    int                    test_day = 12;

    std::vector<float> out1(sig.num_signals()), out2(sig.num_signals());
    sig.compute(md1, test_day, test_day + 1, out1.data());
    sig.compute(md2, test_day, test_day + 1, out2.data());

    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(approx_eq(out1[s], out2[s]))
            << "Signal " << s << " differs: " << out1[s] << " vs " << out2[s];
    }
}

// =============================================================================
// INTRADAY STRUCTURE TESTS
// =============================================================================

TEST(IntradayStructure, KnownDaySignals) {
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md));

    IntradayStructureSignals sig(5);
    ASSERT_EQ(sig.num_signals(), 10);

    int                day = 15;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // Signal 0: Morning direction should be valid
    EXPECT_TRUE(is_valid(out[0]));
    EXPECT_GE(out[0], -1.0f);
    EXPECT_LE(out[0], 1.0f);

    // Signal 5: Range consumed should be in [0, 1]
    EXPECT_TRUE(is_valid(out[5]));
    EXPECT_GE(out[5], 0.0f);
    EXPECT_LE(out[5], 1.0f);

    // Signal 9: Opening drive quality should be valid
    EXPECT_TRUE(is_valid(out[9]));
    EXPECT_GE(out[9], -1.0f);
    EXPECT_LE(out[9], 1.0f);
}

TEST(IntradayStructure, NoLookahead) {
    SyntheticMarketBuilder builder1, builder2;
    MarketData             md1, md2;

    int num_days = 25;
    for (int d = 0; d < num_days; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder1.add_day(spec);

        if (d >= 13) {
            spec.rth_close = base - 30.0f;
            spec.rth_low   = base - 40.0f;
        }
        builder2.add_day(spec);
    }

    ASSERT_TRUE(builder1.build(md1));
    ASSERT_TRUE(builder2.build(md2));

    IntradayStructureSignals sig(5);
    int                      test_day = 12;

    std::vector<float> out1(sig.num_signals()), out2(sig.num_signals());
    sig.compute(md1, test_day, test_day + 1, out1.data());
    sig.compute(md2, test_day, test_day + 1, out2.data());

    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(approx_eq(out1[s], out2[s]))
            << "Signal " << s << " differs: " << out1[s] << " vs " << out2[s];
    }
}

// =============================================================================
// LEVEL INTERACTION TESTS
// =============================================================================

TEST(LevelInteraction, KnownDaySignals) {
    SyntheticMarketBuilder builder;

    // Create a scenario where today tests prev high
    for (int d = 0; d < 20; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder.add_day(spec);
    }

    MarketData md;
    ASSERT_TRUE(builder.build(md));

    LevelInteractionSignals sig(2.0f);
    ASSERT_EQ(sig.num_signals(), 7);

    int                day = 15;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // All signals should be valid (not NaN) with enough history
    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(is_valid(out[s])) << "Signal " << s << " is NaN";
        EXPECT_GE(out[s], -1.0f) << "Signal " << s << " below -1.0";
        EXPECT_LE(out[s], 1.0f) << "Signal " << s << " above 1.0";
    }
}

TEST(LevelInteraction, NoLookahead) {
    SyntheticMarketBuilder builder1, builder2;
    MarketData             md1, md2;

    int num_days = 25;
    for (int d = 0; d < num_days; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder1.add_day(spec);

        if (d >= 13) {
            spec.rth_high  = base + 50.0f;
            spec.rth_close = base + 45.0f;
        }
        builder2.add_day(spec);
    }

    ASSERT_TRUE(builder1.build(md1));
    ASSERT_TRUE(builder2.build(md2));

    LevelInteractionSignals sig(2.0f);
    int                     test_day = 12;

    std::vector<float> out1(sig.num_signals()), out2(sig.num_signals());
    sig.compute(md1, test_day, test_day + 1, out1.data());
    sig.compute(md2, test_day, test_day + 1, out2.data());

    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(approx_eq(out1[s], out2[s]))
            << "Signal " << s << " differs: " << out1[s] << " vs " << out2[s];
    }
}

// =============================================================================
// OVERNIGHT CONTEXT TESTS
// =============================================================================

TEST(OvernightContext, KnownDaySignals) {
    SyntheticMarketBuilder builder;

    // Create a bullish overnight scenario: ETH trends up, closes near high
    for (int d = 0; d < 20; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 8.0f;  // ETH starts low
        spec.eth_close                       = base + 2.0f;  // ETH closes high (bullish)
        spec.rth_open                        = base + 3.0f;  // RTH opens above ETH close
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder.add_day(spec);
    }

    MarketData md;
    ASSERT_TRUE(builder.build(md));

    OvernightContextSignals sig(5);
    ASSERT_EQ(sig.num_signals(), 7);

    int                day = 15;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // Signal 0: ETH direction should be positive (bullish overnight)
    EXPECT_TRUE(is_valid(out[0]));
    EXPECT_GT(out[0], 0.0f);

    // Signal 1: ETH range should be valid and positive
    EXPECT_TRUE(is_valid(out[1]));
    EXPECT_GT(out[1], 0.0f);

    // Signal 2: ETH closed near extreme should be +1 (near high)
    EXPECT_TRUE(is_valid(out[2]));
    EXPECT_GT(out[2], 0.0f);

    // Signal 4: ETH trend steady depends on whether range < 0.5 ATR
    EXPECT_TRUE(is_valid(out[4]));
}

TEST(OvernightContext, NoLookahead) {
    SyntheticMarketBuilder builder1, builder2;
    MarketData             md1, md2;

    int num_days = 25;
    for (int d = 0; d < num_days; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder1.add_day(spec);

        if (d >= 13) {
            spec.eth_open  = base - 30.0f;
            spec.eth_close = base - 25.0f;
        }
        builder2.add_day(spec);
    }

    ASSERT_TRUE(builder1.build(md1));
    ASSERT_TRUE(builder2.build(md2));

    OvernightContextSignals sig(5);
    int                     test_day = 12;

    std::vector<float> out1(sig.num_signals()), out2(sig.num_signals());
    sig.compute(md1, test_day, test_day + 1, out1.data());
    sig.compute(md2, test_day, test_day + 1, out2.data());

    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(approx_eq(out1[s], out2[s]))
            << "Signal " << s << " differs: " << out1[s] << " vs " << out2[s];
    }
}

// =============================================================================
// CALENDAR TESTS
// =============================================================================

TEST(Calendar, KnownDaySignals) {
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md, 10));

    CalendarSignals sig;
    ASSERT_EQ(sig.num_signals(), 4);

    // Test day 3
    int                day = 3;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // Signal 0: Day of week -- should be a valid weekday mapping
    EXPECT_TRUE(is_valid(out[0]));
    EXPECT_GE(out[0], -1.0f);
    EXPECT_LE(out[0], 1.0f);

    // Signal 1: Week of month
    EXPECT_TRUE(is_valid(out[1]));
    EXPECT_GE(out[1], -1.0f);
    EXPECT_LE(out[1], 1.0f);

    // Signal 2: OPEX week
    EXPECT_TRUE(is_valid(out[2]));
    EXPECT_TRUE(out[2] == 0.0f || out[2] == 1.0f);

    // Signal 3: Quarter
    EXPECT_TRUE(is_valid(out[3]));
    EXPECT_GE(out[3], -1.0f);
    EXPECT_LE(out[3], 1.0f);
}

TEST(Calendar, NoLookahead) {
    // Calendar signals are purely date-derived, so they trivially can't look ahead.
    // Still verify consistency across two data sets where only future days differ.
    SyntheticMarketBuilder builder1, builder2;
    MarketData             md1, md2;

    for (int d = 0; d < 20; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder1.add_day(spec);

        if (d >= 10) {
            spec.rth_close = base + 50.0f;
        }
        builder2.add_day(spec);
    }

    ASSERT_TRUE(builder1.build(md1));
    ASSERT_TRUE(builder2.build(md2));

    CalendarSignals sig;
    int             test_day = 5;

    std::vector<float> out1(sig.num_signals()), out2(sig.num_signals());
    sig.compute(md1, test_day, test_day + 1, out1.data());
    sig.compute(md2, test_day, test_day + 1, out2.data());

    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(approx_eq(out1[s], out2[s]))
            << "Signal " << s << " differs: " << out1[s] << " vs " << out2[s];
    }
}

// =============================================================================
// VOLATILITY REGIME TESTS
// =============================================================================

TEST(VolatilityRegime, KnownDaySignals) {
    SyntheticMarketBuilder builder;
    MarketData             md;

    // Create 30 days with expanding volatility
    for (int d = 0; d < 30; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base  = 4000.0f + d * 2.0f;
        float                           range = 10.0f + d * 0.5f;  // Expanding range
        spec.eth_open                         = base - 5.0f;
        spec.eth_close                        = base;
        spec.rth_open                         = base;
        spec.rth_high                         = base + range;
        spec.rth_low                          = base - range;
        spec.rth_close                        = base + range * 0.5f;
        builder.add_day(spec);
    }
    ASSERT_TRUE(builder.build(md));

    VolatilityRegimeSignals sig;
    ASSERT_EQ(sig.num_signals(), 4);

    int                day = 25;
    std::vector<float> out(sig.num_signals());
    sig.compute(md, day, day + 1, out.data());

    // Signal 0: ATR ratio -- with expanding vol, 5d ATR > 20d ATR -> positive
    EXPECT_TRUE(is_valid(out[0]));
    EXPECT_GT(out[0], 0.0f);  // Expanding volatility

    // Signal 1: Prev day range expansion
    EXPECT_TRUE(is_valid(out[1]));

    // Signal 2: Prev day range contraction
    EXPECT_TRUE(is_valid(out[2]));

    // Signal 3: Consecutive expansion/contraction
    EXPECT_TRUE(is_valid(out[3]));
}

TEST(VolatilityRegime, NoLookahead) {
    SyntheticMarketBuilder builder1, builder2;
    MarketData             md1, md2;

    int num_days = 30;
    for (int d = 0; d < num_days; ++d) {
        SyntheticMarketBuilder::DaySpec spec;
        float                           base = 4000.0f + d * 2.0f;
        spec.eth_open                        = base - 5.0f;
        spec.eth_close                       = base;
        spec.rth_open                        = base;
        spec.rth_high                        = base + 10.0f;
        spec.rth_low                         = base - 10.0f;
        spec.rth_close                       = base + 5.0f;
        builder1.add_day(spec);

        if (d >= 25) {
            spec.rth_high = base + 50.0f;
            spec.rth_low  = base - 50.0f;
        }
        builder2.add_day(spec);
    }

    ASSERT_TRUE(builder1.build(md1));
    ASSERT_TRUE(builder2.build(md2));

    VolatilityRegimeSignals sig;
    int                     test_day = 22;

    std::vector<float> out1(sig.num_signals()), out2(sig.num_signals());
    sig.compute(md1, test_day, test_day + 1, out1.data());
    sig.compute(md2, test_day, test_day + 1, out2.data());

    for (int s = 0; s < sig.num_signals(); ++s) {
        EXPECT_TRUE(approx_eq(out1[s], out2[s]))
            << "Signal " << s << " differs: " << out1[s] << " vs " << out2[s];
    }
}

// =============================================================================
// SIGNAL BASE / UTILITY TESTS
// =============================================================================

TEST(SessionSignalBase, ClampSignal) {
    EXPECT_FLOAT_EQ(clamp_signal(0.5f), 0.5f);
    EXPECT_FLOAT_EQ(clamp_signal(1.5f), 1.0f);
    EXPECT_FLOAT_EQ(clamp_signal(-1.5f), -1.0f);
    EXPECT_FLOAT_EQ(clamp_signal(0.0f), 0.0f);
    EXPECT_TRUE(is_nan(clamp_signal(SIGNAL_NAN)));
}

TEST(SessionSignalBase, DirectionSignal) {
    EXPECT_FLOAT_EQ(direction_signal(100.0f, 105.0f), 1.0f);
    EXPECT_FLOAT_EQ(direction_signal(105.0f, 100.0f), -1.0f);
    EXPECT_FLOAT_EQ(direction_signal(100.0f, 100.0f), 0.0f);

    // With dead zone
    EXPECT_FLOAT_EQ(direction_signal(100.0f, 100.5f, 1.0f), 0.0f);
    EXPECT_FLOAT_EQ(direction_signal(100.0f, 102.0f, 1.0f), 1.0f);
}

TEST(SessionSignalBase, MultiDayCompute) {
    // Test computing over a range of days
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md));

    OpeningRangeSignals sig(5);

    // Compute for days 10-15 (6 days)
    int                day_start = 10, day_end = 16;
    int                num_days = day_end - day_start;
    std::vector<float> out(sig.num_signals() * num_days);
    sig.compute(md, day_start, day_end, out.data());

    // Verify layout: output[signal_idx * num_days + day_offset]
    for (int di = 0; di < num_days; ++di) {
        // OR direction (signal 0) for each day should be valid
        float val = out[0 * num_days + di];
        EXPECT_TRUE(is_valid(val)) << "Day " << (day_start + di) << " OR direction is NaN";
    }
}

// =============================================================================
// TIME-GATING TESTS (session-signal lookahead prevention)
// =============================================================================

#include "backtest_engine.h"
#include "cost_model.h"

// ---------------------------------------------------------------------------
// Test: verify each signal module reports the correct number of availability
// entries, and that the values are within sensible ranges.
// ---------------------------------------------------------------------------
TEST(TimeGating, SignalAvailabilityMetadata) {
    // OpeningRange: 8 signals
    {
        OpeningRangeSignals sig;
        auto                avail = sig.signal_available_after_et_seconds();
        ASSERT_EQ(static_cast<int>(avail.size()), sig.num_signals());
        // OR direction/range (0,1) -> 36000 (10:00)
        EXPECT_EQ(avail[0], 36000);
        EXPECT_EQ(avail[1], 36000);
        // OR breakout/false breakout (2,3) -> 57600 (16:00, retrospective)
        EXPECT_EQ(avail[2], 57600);
        EXPECT_EQ(avail[3], 57600);
        // IB direction/range (4,5) -> 37800 (10:30)
        EXPECT_EQ(avail[4], 37800);
        EXPECT_EQ(avail[5], 37800);
        // IB tight-extend/extension (6,7) -> 57600 (16:00, retrospective)
        EXPECT_EQ(avail[6], 57600);
        EXPECT_EQ(avail[7], 57600);
    }

    // IntradayStructure: 10 signals
    {
        IntradayStructureSignals sig;
        auto                     avail = sig.signal_available_after_et_seconds();
        ASSERT_EQ(static_cast<int>(avail.size()), sig.num_signals());
        // Morning signals (0-3) -> 37800 (10:30)
        EXPECT_EQ(avail[0], 37800);
        EXPECT_EQ(avail[1], 37800);
        EXPECT_EQ(avail[2], 37800);
        EXPECT_EQ(avail[3], 37800);
        // Morning first direction (4) -> 35100 (09:45)
        EXPECT_EQ(avail[4], 35100);
        // Range consumed (5) -> 39600 (11:00)
        EXPECT_EQ(avail[5], 39600);
        // Lunch compression (6) -> 46800 (13:00)
        EXPECT_EQ(avail[6], 46800);
        // Afternoon direction (7) -> 54000 (15:00)
        EXPECT_EQ(avail[7], 54000);
        // Power hour direction (8) -> 57600 (16:00)
        EXPECT_EQ(avail[8], 57600);
        // Opening drive quality (9) -> 35100 (09:45)
        EXPECT_EQ(avail[9], 35100);
    }

    // GapAnalysis: 6 signals
    {
        GapAnalysisSignals sig;
        auto               avail = sig.signal_available_after_et_seconds();
        ASSERT_EQ(static_cast<int>(avail.size()), sig.num_signals());
        // Most gap signals -> 34200 (09:30)
        EXPECT_EQ(avail[0], 34200);
        EXPECT_EQ(avail[1], 34200);
        // Gap fill timing -> 39600 (11:00)
        EXPECT_EQ(avail[2], 39600);
        EXPECT_EQ(avail[3], 34200);
        EXPECT_EQ(avail[4], 34200);
        EXPECT_EQ(avail[5], 34200);
    }

    // MultidayPatterns: 12 signals, all 0 (session open)
    {
        MultidayPatternSignals sig;
        auto                   avail = sig.signal_available_after_et_seconds();
        ASSERT_EQ(static_cast<int>(avail.size()), sig.num_signals());
        for (int i = 0; i < sig.num_signals(); ++i) {
            EXPECT_EQ(avail[i], 0) << "MultidayPattern signal " << i;
        }
    }

    // LevelInteraction: 7 signals
    //   0-6: all retrospective -> 57600 (16:00, RTH close)
    {
        LevelInteractionSignals sig;
        auto                    avail = sig.signal_available_after_et_seconds();
        ASSERT_EQ(static_cast<int>(avail.size()), sig.num_signals());
        for (int i = 0; i < sig.num_signals(); ++i) {
            EXPECT_EQ(avail[i], 57600) << "LevelInteraction signal " << i;
        }
    }

    // OvernightContext: 7 signals, all 34200 (RTH open)
    {
        OvernightContextSignals sig;
        auto                    avail = sig.signal_available_after_et_seconds();
        ASSERT_EQ(static_cast<int>(avail.size()), sig.num_signals());
        for (int i = 0; i < sig.num_signals(); ++i) {
            EXPECT_EQ(avail[i], 34200) << "OvernightContext signal " << i;
        }
    }

    // Calendar: 4 signals, all 0
    {
        CalendarSignals sig;
        auto            avail = sig.signal_available_after_et_seconds();
        ASSERT_EQ(static_cast<int>(avail.size()), sig.num_signals());
        for (int i = 0; i < sig.num_signals(); ++i) {
            EXPECT_EQ(avail[i], 0) << "Calendar signal " << i;
        }
    }

    // VolatilityRegime: 4 signals, all 0
    {
        VolatilityRegimeSignals sig;
        auto                    avail = sig.signal_available_after_et_seconds();
        ASSERT_EQ(static_cast<int>(avail.size()), sig.num_signals());
        for (int i = 0; i < sig.num_signals(); ++i) {
            EXPECT_EQ(avail[i], 0) << "VolatilityRegime signal " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Test: broadcast_session_signal_to_bars correctly time-gates signals.
// ---------------------------------------------------------------------------
TEST(TimeGating, BroadcastSessionSignalTimegating) {
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md));

    // Compute OR signals for day 15
    OpeningRangeSignals or_sig(5);
    int                 day = 15;
    std::vector<float>  day_signals(or_sig.num_signals());
    or_sig.compute(md, day, day + 1, day_signals.data());

    // Signal 0: OR direction, available_after = 36000 (10:00)
    float or_direction_value = day_signals[0];
    ASSERT_TRUE(is_valid(or_direction_value));
    ASSERT_NE(or_direction_value, 0.0f);  // Should have a definite direction

    auto    avail        = or_sig.signal_available_after_et_seconds();
    int32_t or_dir_avail = avail[0];  // 36000

    // Find bar range for day 15
    SessionDayView view = build_session_view(md, day);
    ASSERT_GT(view.bar_count, 0);

    int bar_start = view.first_bar;
    int bar_end   = view.first_bar + view.bar_count;
    int num_bars  = bar_end - bar_start;

    // Broadcast signal 0 to bars with time-gating
    std::vector<float> bar_values(num_bars);
    broadcast_session_signal_to_bars(md, &or_direction_value, day, day + 1, or_dir_avail, bar_start,
                                     bar_end, bar_values.data());

    // Verify: bars before 10:00 ET of the session day get 0.0 (gated) and
    // bars from 10:00 ET on get the signal value. The session opens at
    // 18:00 the evening before, so the evening bars (clock time later than
    // 10:00) come *before* the availability time and must be gated too.
    const int32_t open                = md.session_model().session_open_sec;
    bool          found_gated         = false;
    bool          found_ungated       = false;
    bool          found_evening_gated = false;
    for (int i = 0; i < num_bars; ++i) {
        int     abs_bar = bar_start + i;
        int32_t tod     = md.session(abs_bar).time_of_day_sec;

        if (seconds_since_open(tod, open) < seconds_since_open(or_dir_avail, open)) {
            if (tod >= open) found_evening_gated = true;
            EXPECT_FLOAT_EQ(bar_values[i], 0.0f)
                << "Bar at tod=" << tod << " should be gated (0.0)";
            found_gated = true;
        } else {
            EXPECT_FLOAT_EQ(bar_values[i], or_direction_value)
                << "Bar at tod=" << tod << " should have signal value " << or_direction_value;
            found_ungated = true;
        }
    }

    // Ensure we actually tested both cases (bars before and after 10:00)
    EXPECT_TRUE(found_gated) << "No bars found before 10:00 ET";
    EXPECT_TRUE(found_evening_gated || md.session(bar_start).time_of_day_sec < open)
        << "evening bars of the session must be gated";
    EXPECT_TRUE(found_ungated) << "No bars found at/after 10:00 ET";
}

// ---------------------------------------------------------------------------
// Test: broadcast with available_after=0 passes all values through (no gating).
// ---------------------------------------------------------------------------
TEST(TimeGating, BroadcastNoGatingWhenAvailableAfterZero) {
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md));

    // Use a calendar signal (available_after=0)
    CalendarSignals    cal_sig;
    int                day = 10;
    std::vector<float> day_signals(cal_sig.num_signals());
    cal_sig.compute(md, day, day + 1, day_signals.data());

    float dow_value = day_signals[0];
    ASSERT_TRUE(is_valid(dow_value));

    auto    avail     = cal_sig.signal_available_after_et_seconds();
    int32_t dow_avail = avail[0];  // 0
    EXPECT_EQ(dow_avail, 0);

    SessionDayView view = build_session_view(md, day);
    ASSERT_GT(view.bar_count, 0);

    int bar_start = view.first_bar;
    int bar_end   = view.first_bar + view.bar_count;
    int num_bars  = bar_end - bar_start;

    std::vector<float> bar_values(num_bars);
    broadcast_session_signal_to_bars(md, &dow_value, day, day + 1, dow_avail, bar_start, bar_end,
                                     bar_values.data());

    // All bars should have the signal value (no gating)
    for (int i = 0; i < num_bars; ++i) {
        EXPECT_FLOAT_EQ(bar_values[i], dow_value)
            << "Bar " << i << " should have dow_value=" << dow_value;
    }
}

// ---------------------------------------------------------------------------
// Test: backtest engine time-gating prevents using OR breakout signal
// for bars before 10:00 ET.
//
// Setup:
//   - Two indicator series: one always-available (weight 0), one OR-gated
//   - The OR-gated series has a strong bullish signal (+1.0) for all bars
//   - With gating (available_after=36000), bars before 10:00 should not see it
//   - Without gating, the signal is visible to all bars
//
// We verify that the gated run produces different (fewer) trades than un-gated.
// ---------------------------------------------------------------------------
TEST(TimeGating, BacktestEngineTimeGating) {
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md, 20));

    // Pick a session day with enough bars
    int            day  = 15;
    SessionDayView view = build_session_view(md, day);
    ASSERT_GT(view.rth_count, 0);

    // Use the RTH portion as our backtest range
    int bar_start = view.rth_first;
    int bar_end   = view.rth_first + view.rth_count;
    int num_bars  = bar_end - bar_start;
    ASSERT_GT(num_bars, 10);

    // Create a single indicator series: constant +1.0 (strong bullish signal)
    std::vector<float> signal_data(num_bars, 1.0f);
    const float*       series_ptrs[1] = {signal_data.data()};

    // Strategy config: one indicator, entry threshold 0.5
    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 5;

    CostModel cost = CostModel::es();

    BacktestEngine engine;

    // --- Run WITHOUT time-gating ---
    BacktestResult result_ungated =
        engine.run(&md.bar(0), bar_start, bar_end, series_ptrs, 1, config, cost, 1);
    int trades_ungated = result_ungated.total_trades;

    // --- Run WITH time-gating (signal available after 10:00 ET = 36000) ---
    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = session_time::OR_END;  // 36000

    BacktestResult result_gated = engine.run(&md.bar(0), &md.session(0), bar_start, bar_end,
                                             series_ptrs, 1, gates, config, cost, 1);
    int            trades_gated = result_gated.total_trades;

    // The gated version should have fewer or equal trades because some
    // early-session bars see the signal as 0 instead of 1.0.
    // Specifically, bars before 10:00 ET won't trigger entries.
    EXPECT_LE(trades_gated, trades_ungated);

    // Verify that gated bars before 10:00 did not produce any entry signals.
    // Check the trade log: no trade should have an entry bar before 10:00 ET.
    for (const auto& trade : engine.last_trades()) {
        // The entry fill is at entry_bar (which is signal_bar + entry_lag).
        // The signal was evaluated at (entry_bar - 1), which should be >= 10:00
        // to have seen the signal.
        int signal_bar = trade.entry_bar - 1;  // entry_lag = 1
        if (signal_bar >= bar_start) {
            int32_t signal_tod = md.session(signal_bar).time_of_day_sec;
            EXPECT_GE(signal_tod, session_time::OR_END)
                << "Trade generated from signal at tod=" << signal_tod
                << " which is before OR end (10:00 ET)";
        }
    }
}

// ---------------------------------------------------------------------------
// Test: time-gating with RTH_CLOSE availability effectively prevents
// all intra-RTH signals (only usable for next session's bars).
// ---------------------------------------------------------------------------
TEST(TimeGating, RthCloseGatingBlocksAllRthBars) {
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md, 20));

    int            day  = 15;
    SessionDayView view = build_session_view(md, day);
    ASSERT_GT(view.rth_count, 0);

    int bar_start = view.rth_first;
    int bar_end   = view.rth_first + view.rth_count;
    int num_bars  = bar_end - bar_start;

    // Create a signal that would trigger trades if visible
    std::vector<float> signal_data(num_bars, 1.0f);
    const float*       series_ptrs[1] = {signal_data.data()};

    StrategyConfig config;
    config.num_indicators       = 1;
    config.indicator_indices[0] = 0;
    config.weights[0]           = 1.0f;
    config.aggregation          = StrategyConfig::WEIGHTED_SUM;
    config.entry_threshold      = 0.5f;
    config.exit_mode            = StrategyConfig::BAR_TIMEOUT;
    config.exit_timeout_bars    = 3;

    CostModel      cost = CostModel::es();
    BacktestEngine engine;

    // Gate with RTH_CLOSE (57600) -- no RTH bar has tod >= 57600
    SignalTimeGate gates[1];
    gates[0].available_after_et_sec = session_time::RTH_CLOSE;

    BacktestResult result = engine.run(&md.bar(0), &md.session(0), bar_start, bar_end, series_ptrs,
                                       1, gates, config, cost, 1);

    // All RTH bars have tod < RTH_CLOSE, so the signal is gated to 0 for
    // every bar.  No entries should be triggered.
    EXPECT_EQ(result.total_trades, 0)
        << "RTH_CLOSE gating should block all signals within RTH bars";
}

// ---------------------------------------------------------------------------
// HalfDaySuppressionTest: On an early-close session (session_close_sec=46800,
// 13:00 ET), a signal gated at 14:00 ET (50400) must be suppressed because
// 50400 > 46800: a signal that becomes available after the session has
// closed must never reach that session's bars.
//
// Strategy: Build a standard market, manually mark one session as early_close
// with session_close_sec=46800, then broadcast a signal with
// available_after=50400 (14:00 ET).  All output bars for that session must
// be 0.0 (suppressed).
// ---------------------------------------------------------------------------
TEST(SessionSignals, HalfDaySuppressionTest) {
    SyntheticMarketBuilder builder;
    MarketData             md;
    ASSERT_TRUE(build_standard_market(builder, md, 20));

    // Pick a session day with enough warmup
    const int      day  = 10;
    SessionDayView view = build_session_view(md, day);
    ASSERT_GT(view.bar_count, 0);

    // Mark all bars in this session as early_close with session_close at 13:00 ET
    const int32_t early_close_sec = 46800;  // 13:00 ET
    for (int i = view.first_bar; i < view.first_bar + view.bar_count; ++i) {
        SessionInfo& si      = const_cast<SessionInfo&>(md.session(i));
        si.early_close       = 1;
        si.session_close_sec = early_close_sec;
    }

    // Create a signal value for this day: a strong positive signal
    float signal_value = 0.8f;

    // Broadcast with available_after = 50400 (14:00 ET)
    // Since 50400 > 46800 (session_close_sec), the signal must be suppressed
    const int32_t available_after = 50400;  // 14:00 ET

    int bar_start = view.first_bar;
    int bar_end   = view.first_bar + view.bar_count;
    int num_bars  = bar_end - bar_start;

    std::vector<float> output(num_bars);
    broadcast_session_signal_to_bars(md, &signal_value, day, day + 1, available_after, bar_start,
                                     bar_end, output.data());

    // All bars should be 0.0 because the signal availability time (14:00)
    // exceeds the early-close session close time (13:00)
    int suppressed_count = 0;
    for (int i = 0; i < num_bars; ++i) {
        if (output[i] == 0.0f) ++suppressed_count;
    }
    EXPECT_EQ(suppressed_count, num_bars)
        << "All bars in an early-close (13:00) session must suppress a signal "
           "gated at 14:00 ET";

    // Verify that without early_close, the signal would NOT be suppressed for
    // bars after 14:00 ET.  Restore normal session to confirm the test is valid.
    for (int i = view.first_bar; i < view.first_bar + view.bar_count; ++i) {
        SessionInfo& si      = const_cast<SessionInfo&>(md.session(i));
        si.early_close       = 0;
        si.session_close_sec = 61200;  // normal 17:00 ET
    }

    std::vector<float> output_normal(num_bars);
    broadcast_session_signal_to_bars(md, &signal_value, day, day + 1, available_after, bar_start,
                                     bar_end, output_normal.data());

    // Some bars after 14:00 ET should now have the signal value
    int non_zero_count = 0;
    for (int i = 0; i < num_bars; ++i) {
        if (output_normal[i] != 0.0f) ++non_zero_count;
    }
    EXPECT_GT(non_zero_count, 0)
        << "On a normal session, bars after 14:00 ET should receive the signal "
           "(confirms test validity)";
}

// =============================================================================
// Level Interaction Warmup Tests
// =============================================================================

TEST(LevelInteraction, WarmupDaysDefault) {
    // Default ema_period is 20; warmup_days should be max(3, 20) = 20
    LevelInteractionSignals lis;
    EXPECT_EQ(lis.warmup_days(), 20);
}

TEST(LevelInteraction, WarmupDaysSmallEma) {
    // ema_period=2 is below the minimum floor of 3
    LevelInteractionSignals lis(2.0f, 0.25f, 2);
    EXPECT_EQ(lis.warmup_days(), 3);
}

TEST(LevelInteraction, WarmupDaysLargeEma) {
    // ema_period=50 exceeds the floor; warmup_days should be 50
    LevelInteractionSignals lis(2.0f, 0.25f, 50);
    EXPECT_EQ(lis.warmup_days(), 50);
}
