#pragma once
// ---------------------------------------------------------------------------
// Bar file I/O: write the binary bar format and import plain CSV.
//
// Binary format: 64-byte FileHeader followed by bar_count 32-byte Bar
// records, little-endian (see docs/DATA_FORMAT.md). MarketData::load reads
// it back via a memory map.
//
// CSV format: one bar per row, columns timestamp,open,high,low,close,volume.
// A header row is optional. Timestamps are UTC and may be integer epoch
// seconds, milliseconds, microseconds or nanoseconds (detected by
// magnitude), or ISO 8601 "YYYY-MM-DD[T ]HH:MM[:SS[.fraction]]" with an
// optional "Z" or "+HH:MM" / "-HH:MM" offset.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "market_data.h"

// Write bars to a binary bar file. Bars must have strictly increasing
// timestamps. Returns false and fills *error on failure.
bool write_bar_file(const std::string& path, const std::string& symbol, int timeframe_sec,
                    const std::vector<Bar>& bars, std::string* error = nullptr);

struct CsvImportResult {
    std::vector<Bar> bars;
    int              timeframe_sec = 0;  // most common spacing between consecutive bars
    int              rows_read     = 0;
    std::string      error;  // empty on success
    bool             ok() const { return error.empty(); }
};

// Parse bars from CSV text. Rejects rows with unparseable fields,
// non-finite or non-positive prices, high < max(open, close),
// low > min(open, close), negative volume, or timestamps that do not
// strictly increase. Errors name the offending line.
CsvImportResult read_bars_csv(std::istream& in);
CsvImportResult read_bars_csv_file(const std::string& path);

// Parse a single timestamp field to UTC nanoseconds. Exposed for tests.
bool parse_timestamp_ns(const std::string& field, int64_t& out_ns);
