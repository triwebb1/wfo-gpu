// Bar file I/O: CSV import, timestamp parsing, and binary round trips.

#include <gtest/gtest.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "bar_io.h"
#include "civil_time.h"
#include "market_data.h"

namespace {

constexpr int64_t NS = civil::NS_PER_SEC;
// 2024-01-02T14:30:00Z
constexpr int64_t T0_SEC = 1704205800;

CsvImportResult parse(const std::string& text) {
    std::istringstream in(text);
    return read_bars_csv(in);
}

}  // namespace

TEST(BarIo, ParsesEpochTimestampsByMagnitude) {
    int64_t ns = 0;
    ASSERT_TRUE(parse_timestamp_ns("1704205800", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
    ASSERT_TRUE(parse_timestamp_ns("1704205800000", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
    ASSERT_TRUE(parse_timestamp_ns("1704205800000000", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
    ASSERT_TRUE(parse_timestamp_ns("1704205800000000000", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
}

TEST(BarIo, ParsesIso8601Timestamps) {
    int64_t ns = 0;
    ASSERT_TRUE(parse_timestamp_ns("2024-01-02T14:30:00Z", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
    ASSERT_TRUE(parse_timestamp_ns("2024-01-02 14:30:00", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
    ASSERT_TRUE(parse_timestamp_ns("2024-01-02T14:30", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
    ASSERT_TRUE(parse_timestamp_ns("2024-01-02T09:30:00-05:00", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
    ASSERT_TRUE(parse_timestamp_ns("2024-01-02T15:30:00+0100", ns));
    EXPECT_EQ(ns, T0_SEC * NS);
    ASSERT_TRUE(parse_timestamp_ns("2024-01-02T14:30:00.250Z", ns));
    EXPECT_EQ(ns, T0_SEC * NS + 250'000'000);
    ASSERT_TRUE(parse_timestamp_ns("2024-01-02", ns));
    EXPECT_EQ(ns, (T0_SEC - (14 * 3600 + 1800)) * NS);
}

TEST(BarIo, RejectsMalformedTimestamps) {
    int64_t ns = 0;
    for (const char* bad : {"", "abc", "2024-13-01", "2024-01-02T25:00", "2024-01-02T14:30:00Q",
                            "2024/01/02", "12.5"}) {
        EXPECT_FALSE(parse_timestamp_ns(bad, ns)) << bad;
    }
}

TEST(BarIo, ReadsCsvWithHeaderInAnyColumnOrder) {
    const auto r = parse(
        "# exported bars\n"
        "Volume,Close,Low,High,Open,Timestamp\n"
        "10,101.5,100,102,100.5,2024-01-02T14:30:00Z\n"
        "12,101.0,100.5,102.5,101.5,2024-01-02T14:31:00Z\n"
        "\n"
        "9,101.25,100.75,101.5,101.0,2024-01-02T14:32:00Z\n");
    ASSERT_TRUE(r.ok()) << r.error;
    ASSERT_EQ(r.bars.size(), 3u);
    EXPECT_EQ(r.timeframe_sec, 60);
    EXPECT_EQ(r.bars[0].timestamp, T0_SEC * NS);
    EXPECT_FLOAT_EQ(r.bars[0].open, 100.5f);
    EXPECT_FLOAT_EQ(r.bars[0].high, 102.0f);
    EXPECT_FLOAT_EQ(r.bars[0].low, 100.0f);
    EXPECT_FLOAT_EQ(r.bars[0].close, 101.5f);
    EXPECT_EQ(r.bars[0].volume, 10u);
}

TEST(BarIo, ReadsHeaderlessCsvAndInfersTimeframe) {
    // One missing bar: the most common spacing still wins.
    const auto r = parse(
        "1704205800,1,2,0.5,1.5,100\n"
        "1704205830,1.5,2,1,1.75,100\n"
        "1704205860,1.75,2,1,1.5,100\n"
        "1704205920,1.5,2,1,1.25,100.4\n");
    ASSERT_TRUE(r.ok()) << r.error;
    EXPECT_EQ(r.rows_read, 4);
    EXPECT_EQ(r.timeframe_sec, 30);
    EXPECT_EQ(r.bars[3].volume, 100u);
}

TEST(BarIo, ReportsTheOffendingLine) {
    auto r = parse(
        "timestamp,open,high,low,close,volume\n"
        "1704205800,1,2,0.5,1.5,100\n"
        "1704205800,1,2,0.5,1.5,100\n");
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.error.find("line 3"), std::string::npos) << r.error;
    EXPECT_NE(r.error.find("increase"), std::string::npos) << r.error;

    r = parse("1704205800,1,0.9,0.5,1.5,100\n");  // high below close
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.error.find("line 1"), std::string::npos) << r.error;

    r = parse("1704205800,1,2,0.5,nan,100\n");
    EXPECT_FALSE(r.ok());
    r = parse("1704205800,1,2,0.5,1.5,-3\n");
    EXPECT_FALSE(r.ok());
    r = parse("time,open,high,low,close\n1704205800,1,2,0.5,1.5\n");
    EXPECT_FALSE(r.ok());
    EXPECT_NE(r.error.find("missing a column"), std::string::npos) << r.error;
    r = parse("");
    EXPECT_FALSE(r.ok());
}

TEST(BarIo, BinaryRoundTripThroughMarketData) {
    const auto r = parse(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02T14:30:00Z,4700.25,4701,4699.5,4700.75,120\n"
        "2024-01-02T14:30:30Z,4700.75,4702,4700.5,4701.5,95\n"
        "2024-01-02T14:31:00Z,4701.5,4701.75,4700,4700.25,140\n");
    ASSERT_TRUE(r.ok()) << r.error;
    const auto path =
        (std::filesystem::temp_directory_path() / "wfo_gpu_bar_io_roundtrip.bin").string();
    std::string err;
    ASSERT_TRUE(write_bar_file(path, "ESTEST", r.timeframe_sec, r.bars, &err)) << err;

    MarketData md;
    ASSERT_TRUE(md.load(path));
    ASSERT_EQ(md.bar_count(), 3);
    EXPECT_EQ(md.symbol(), "ESTEST");
    EXPECT_EQ(md.header().timeframe_sec, 30u);
    EXPECT_EQ(md.header().start_ts, r.bars.front().timestamp);
    EXPECT_EQ(md.header().end_ts, r.bars.back().timestamp);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(md.bar(i).timestamp, r.bars[i].timestamp);
        EXPECT_EQ(md.bar(i).close, r.bars[i].close);
        EXPECT_EQ(md.bar(i).volume, r.bars[i].volume);
    }
}

TEST(BarIo, WriterRejectsBadInput) {
    const auto  path = (std::filesystem::temp_directory_path() / "wfo_gpu_bar_io_bad.bin").string();
    std::string err;
    EXPECT_FALSE(write_bar_file(path, "X", 30, {}, &err));
    std::vector<Bar> bars(2);
    bars[0].timestamp = 10;
    bars[1].timestamp = 10;
    EXPECT_FALSE(write_bar_file(path, "X", 30, bars, &err));
    EXPECT_NE(err.find("strictly increase"), std::string::npos);
    bars[1].timestamp = 20;
    EXPECT_FALSE(write_bar_file(path, "X", 0, bars, &err));
}
