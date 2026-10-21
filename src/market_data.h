#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "session_model.h"

// NOTE: pragma pack(push, 1) is kept around both Bar and FileHeader.
// Bar is naturally aligned at 32 bytes (int64_t(8) + 4*float(16) + 2*uint32_t(8) = 32),
// so pack(1) is harmless but kept for consistency with FileHeader, which REQUIRES
// packing to hit exactly 64 bytes (without it, padding after timeframe_sec inflates
// the struct to 68 bytes).
#pragma pack(push, 1)
struct Bar {
    int64_t  timestamp;  // Unix epoch nanoseconds, UTC
    float    open;
    float    high;
    float    low;
    float    close;
    uint32_t volume;
    // Reserved; explicit tail padding keeps Bar at exactly 32 bytes /
    // 8-byte aligned. Do NOT repurpose
    // without bumping FileHeader::version -- binary bar files on disk rely on
    // this exact layout.
    uint32_t _pad;
};
static_assert(sizeof(Bar) == 32, "Bar must be exactly 32 bytes");

struct FileHeader {
    char     magic[4];  // "WFBT"
    uint32_t version;
    uint32_t bar_count;
    char     symbol[8];
    uint32_t timeframe_sec;
    int64_t  start_ts;
    int64_t  end_ts;
    char     reserved[24];
};
static_assert(sizeof(FileHeader) == 64, "FileHeader must be exactly 64 bytes");
#pragma pack(pop)

// Session info for one bar
struct SessionInfo {
    int32_t session_day;        // session day index (0-based, incrementing)
    int32_t intra_session_idx;  // bar index within this session (0 = first bar after the open)
    int32_t time_of_day_sec;    // seconds since local midnight in the session model's clock
    uint8_t halt_flag;          // 1 if the session has an abnormal core-hours gap
    uint8_t early_close;        // 1 if this is a known early-close session, 0 otherwise
    uint8_t flags;              // SessionFlag bits, stamped from the session model
    int32_t
        session_close_sec;  // local close time (17:00 ET = 61200 for US futures, earlier on early-close days; -1 = none)
    // Seconds between this bar and the previous bar in the
    // dataset (0 for the first bar). Callers can inspect this to detect gaps
    // without re-reading raw timestamps. Gap semantics are independent of
    // session boundaries: weekend/holiday gaps also surface here.
    int32_t gap_to_prev_sec;
};
static_assert(sizeof(SessionInfo) == 24,
              "SessionInfo layout assumption broken; review all callers for padding/order");

// ---------------------------------------------------------------------------
// MarketData -- immutable bar store with mmap-backed price arrays and a
// precomputed session/timeframe index.
//
// Thread-safety contract:
//   * load() / move-assign mutate internal state and are NOT thread-safe.
//   * After load() returns successfully, the instance is post-load IMMUTABLE:
//     no public method mutates any observable field. All accessors
//     (bars(), bar(), session(), timeframe(), find_bar(),
//     first_bar_of_session(), ...) are safe to call concurrently from
//     multiple reader threads.
//   * data_hash() computes and caches the hash on its first call, so that
//     first call must not race with another; later calls are plain reads.
//   * Callers MUST NOT invoke load() while readers are outstanding.
// ---------------------------------------------------------------------------
class MarketData {
public:
    MarketData() = default;
    ~MarketData();

    // Non-copyable, movable
    MarketData(const MarketData&)            = delete;
    MarketData& operator=(const MarketData&) = delete;
    MarketData(MarketData&& other) noexcept;
    MarketData& operator=(MarketData&& other) noexcept;

    // Load from binary file (mmap)
    bool load(const std::string& filepath, const SessionModel& model = SessionModel::us_futures());

    const SessionModel& session_model() const { return m_model; }

    // Bar access
    const Bar*        bars() const { return m_bars; }
    const Bar&        bar(int index) const;
    int               bar_count() const;
    const FileHeader& header() const;

    // Session index
    const SessionInfo& session(int bar_index) const;
    const SessionInfo* sessions() const { return m_sessions.data(); }
    int                session_count() const;

    // O(1) lookup: first bar index for a given session day.
    // Returns -1 if session_day is out of range.
    int first_bar_of_session(int session_day) const;

    // Precomputed calendar mappings per session day (built in build_session_index).
    // These avoid repeated gmtime_s() / iso_year_week() calls during metrics.
    const std::vector<int>& session_day_to_week() const { return m_session_day_to_week; }
    const std::vector<int>& session_day_to_month() const { return m_session_day_to_month; }
    const std::vector<int>& session_day_to_year() const { return m_session_day_to_year; }

    // Binary search: first session_day index whose first bar >= bar_start
    int first_session_day_in_range(int bar_start) const;
    // Binary search: last session_day index whose first bar < bar_end
    int last_session_day_in_range(int bar_end) const;

    // Find bar index for a given UTC timestamp (nanoseconds). Returns -1 if not found.
    int find_bar(int64_t timestamp_ns) const;

    // Find the first bar index on or after a given timestamp
    int find_bar_ge(int64_t timestamp_ns) const;

    // Pre-aggregated timeframe bars
    struct TimeframeData {
        std::vector<Bar> bars;
        // parent_map[i] = index of the TF bar that contains base bar i. That
        // TF bar is still forming while bar i is open; its OHLC includes
        // later base bars.
        std::vector<int> parent_map;
        // completed_map[i] = index of the latest TF bar that is complete at
        // the close of base bar i, or -1 if none yet. Anything a strategy
        // acts on at bar i must use this map, not parent_map, or it sees the
        // future.
        std::vector<int> completed_map;
        int              timeframe_sec = 0;
    };

    // Fetch a pre-aggregated timeframe. Throws std::invalid_argument if
    // tf_sec is not one of the built-in target timeframes.
    const TimeframeData& timeframe(int tf_sec) const;

    // Non-throwing variant. Returns nullptr when tf_sec is unknown
    // (e.g. a window_mode whose target timeframe was not pre-aggregated).
    // Callers should prefer this in code paths that already handle an
    // "unsupported timeframe" fallback, to avoid exception-driven control
    // flow on a cold configuration error.
    const TimeframeData* try_timeframe(int tf_sec) const;

    // Average bars per calendar day across the file (weekends and holidays
    // included), for converting day-based window lengths into bar counts.
    double bars_per_calendar_day() const;

    // Get symbol string (null-terminated)
    std::string symbol() const;

    // FNV-1a 64-bit hash over the mapped bar payload
    // (FileHeader + Bar array). Computed lazily on first call and cached.
    // Used as the bar_data_hash invariant stamped in checkpoints: if the
    // underlying bar file changed (re-built, extended, edited) the hash
    // differs and resume is rejected.
    uint64_t data_hash() const;

private:
    // Memory-mapped file
    void*  m_mapped_data = nullptr;
    size_t m_mapped_size = 0;
#ifdef _WIN32
    void* m_file_handle    = nullptr;
    void* m_mapping_handle = nullptr;
#else
    int m_fd = -1;
#endif

    const FileHeader* m_header = nullptr;
    const Bar*        m_bars   = nullptr;

    // Session index
    std::vector<SessionInfo> m_sessions;
    int                      m_session_count = 0;

    // Reverse mapping: session_day -> first bar index (built in build_session_index)
    std::vector<int> m_session_first_bar;

    // Precomputed calendar keys per session day (built in build_session_index)
    std::vector<int> m_session_day_to_week;   // ISO year*100+week
    std::vector<int> m_session_day_to_month;  // year*100+month
    std::vector<int> m_session_day_to_year;   // year

    // Timeframe aggregations
    std::vector<TimeframeData> m_timeframes;  // indexed by position, not by tf_sec

    void unmap();
    void build_session_index();
    void build_timeframe_bars();

    // Holiday / early-close calendar: maps "YYYY-MM-DD" -> close time in ET seconds
    std::unordered_map<std::string, int32_t> m_early_close_map;
    void                                     load_holiday_calendar(const std::string& csv_path);

    // Session layout and local clock.
    SessionModel m_model;

    // UTC offset (seconds) of the session model's clock at a UTC instant.
    int32_t local_offset_sec(int64_t ts_ns) const;

    // Cached FNV-1a 64-bit hash of the mapped bar payload.
    // Mutable because data_hash() is a logically-const query.
    mutable uint64_t m_data_hash        = 0;
    mutable bool     m_data_hash_cached = false;
};
