#include "market_data.h"
#include "common.h"
#include "civil_time.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int64_t NS_PER_SEC = 1'000'000'000LL;
static constexpr int64_t NS_PER_DAY = 86400LL * NS_PER_SEC;

// Target timeframes for pre-aggregation (seconds)
static constexpr int TARGET_TFS[]   = {60, 120, 300, 900, 1800, 3600};
static constexpr int NUM_TARGET_TFS = 6;

// ---------------------------------------------------------------------------
// Destructor / move
// ---------------------------------------------------------------------------
MarketData::~MarketData() {
    unmap();
}

MarketData::MarketData(MarketData&& other) noexcept
    : m_mapped_data(other.m_mapped_data),
      m_mapped_size(other.m_mapped_size)
#ifdef _WIN32
      ,
      m_file_handle(other.m_file_handle),
      m_mapping_handle(other.m_mapping_handle)
#else
      ,
      m_fd(other.m_fd)
#endif
      ,
      m_header(other.m_header),
      m_bars(other.m_bars),
      m_sessions(std::move(other.m_sessions)),
      m_session_count(other.m_session_count),
      m_session_first_bar(std::move(other.m_session_first_bar)),
      m_session_day_to_week(std::move(other.m_session_day_to_week)),
      m_session_day_to_month(std::move(other.m_session_day_to_month)),
      m_session_day_to_year(std::move(other.m_session_day_to_year)),
      m_timeframes(std::move(other.m_timeframes)),
      m_early_close_map(std::move(other.m_early_close_map)),
      m_model(std::move(other.m_model)) {
    other.m_mapped_data = nullptr;
    other.m_mapped_size = 0;
    other.m_header      = nullptr;
    other.m_bars        = nullptr;
#ifdef _WIN32
    other.m_file_handle    = nullptr;
    other.m_mapping_handle = nullptr;
#else
    other.m_fd = -1;
#endif
}

MarketData& MarketData::operator=(MarketData&& other) noexcept {
    if (this != &other) {
        unmap();
        m_mapped_data = other.m_mapped_data;
        m_mapped_size = other.m_mapped_size;
#ifdef _WIN32
        m_file_handle    = other.m_file_handle;
        m_mapping_handle = other.m_mapping_handle;
#else
        m_fd = other.m_fd;
#endif
        m_header               = other.m_header;
        m_bars                 = other.m_bars;
        m_sessions             = std::move(other.m_sessions);
        m_session_count        = other.m_session_count;
        m_session_first_bar    = std::move(other.m_session_first_bar);
        m_session_day_to_week  = std::move(other.m_session_day_to_week);
        m_session_day_to_month = std::move(other.m_session_day_to_month);
        m_session_day_to_year  = std::move(other.m_session_day_to_year);
        m_timeframes           = std::move(other.m_timeframes);
        m_model                = std::move(other.m_model);
        m_early_close_map      = std::move(other.m_early_close_map);

        other.m_mapped_data = nullptr;
        other.m_mapped_size = 0;
        other.m_header      = nullptr;
        other.m_bars        = nullptr;
#ifdef _WIN32
        other.m_file_handle    = nullptr;
        other.m_mapping_handle = nullptr;
#else
        other.m_fd = -1;
#endif
    }
    return *this;
}

// ---------------------------------------------------------------------------
// unmap
// ---------------------------------------------------------------------------
void MarketData::unmap() {
#ifdef _WIN32
    if (m_mapped_data) {
        UnmapViewOfFile(m_mapped_data);
        m_mapped_data = nullptr;
    }
    if (m_mapping_handle) {
        CloseHandle(m_mapping_handle);
        m_mapping_handle = nullptr;
    }
    if (m_file_handle && m_file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(m_file_handle);
        m_file_handle = nullptr;
    }
#else
    if (m_mapped_data && m_mapped_data != MAP_FAILED) {
        munmap(m_mapped_data, m_mapped_size);
        m_mapped_data = nullptr;
    }
    if (m_fd >= 0) {
        close(m_fd);
        m_fd = -1;
    }
#endif
    m_mapped_size = 0;
    m_header      = nullptr;
    m_bars        = nullptr;
    // Invalidate the cached data_hash; a subsequent load() will recompute.
    m_data_hash        = 0;
    m_data_hash_cached = false;
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
bool MarketData::load(const std::string& filepath, const SessionModel& model) {
    std::string model_error;
    if (!model.validate(&model_error)) {
        std::cerr << "[MarketData] Invalid session model '" << model.name << "': " << model_error
                  << std::endl;
        return false;
    }
    // Clean up any previous mapping
    unmap();
    m_sessions.clear();
    m_session_count = 0;
    m_session_first_bar.clear();
    m_session_day_to_week.clear();
    m_session_day_to_month.clear();
    m_session_day_to_year.clear();
    m_timeframes.clear();

#ifdef _WIN32
    // Open file
    m_file_handle = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (m_file_handle == INVALID_HANDLE_VALUE) {
        std::cerr << "[MarketData] Failed to open file: " << filepath << std::endl;
        m_file_handle = nullptr;
        return false;
    }

    // Get file size
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(m_file_handle, &file_size)) {
        std::cerr << "[MarketData] Failed to get file size." << std::endl;
        CloseHandle(m_file_handle);
        m_file_handle = nullptr;
        return false;
    }
    m_mapped_size = static_cast<size_t>(file_size.QuadPart);

    // Create file mapping
    m_mapping_handle =
        CreateFileMappingA(m_file_handle, nullptr, PAGE_READONLY, 0, 0,  // map entire file
                           nullptr);
    if (!m_mapping_handle) {
        std::cerr << "[MarketData] Failed to create file mapping." << std::endl;
        CloseHandle(m_file_handle);
        m_file_handle = nullptr;
        return false;
    }

    // Map view
    m_mapped_data = MapViewOfFile(m_mapping_handle, FILE_MAP_READ, 0, 0, 0  // map entire file
    );
    if (!m_mapped_data) {
        std::cerr << "[MarketData] Failed to map view of file." << std::endl;
        CloseHandle(m_mapping_handle);
        m_mapping_handle = nullptr;
        CloseHandle(m_file_handle);
        m_file_handle = nullptr;
        return false;
    }
#else
    // Open file
    m_fd = open(filepath.c_str(), O_RDONLY);
    if (m_fd < 0) {
        std::cerr << "[MarketData] Failed to open file: " << filepath << std::endl;
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(m_fd, &st) != 0) {
        std::cerr << "[MarketData] Failed to stat file." << std::endl;
        close(m_fd);
        m_fd = -1;
        return false;
    }
    m_mapped_size = static_cast<size_t>(st.st_size);

    // Memory-map
    m_mapped_data = mmap(nullptr, m_mapped_size, PROT_READ, MAP_PRIVATE, m_fd, 0);
    if (m_mapped_data == MAP_FAILED) {
        std::cerr << "[MarketData] Failed to mmap file." << std::endl;
        m_mapped_data = nullptr;
        close(m_fd);
        m_fd = -1;
        return false;
    }
#endif

    // Validate minimum size
    if (m_mapped_size < sizeof(FileHeader)) {
        std::cerr << "[MarketData] File too small for header." << std::endl;
        unmap();
        return false;
    }

    // Set header pointer and validate
    m_header = reinterpret_cast<const FileHeader*>(m_mapped_data);

    if (std::memcmp(m_header->magic, "WFBT", 4) != 0) {
        std::cerr << "[MarketData] Invalid magic bytes (expected 'WFBT')." << std::endl;
        unmap();
        return false;
    }

    if (m_header->version != 1) {
        std::cerr << "[MarketData] Unsupported file version: " << m_header->version << std::endl;
        unmap();
        return false;
    }

    // Validate bar_count before computing expected size.
    // 16 years of 30s data ~ 8M bars.  500M is a generous upper bound that
    // catches corrupted headers without rejecting any plausible dataset.
    if (m_header->bar_count > 500'000'000u) {
        std::cerr << "[MarketData] Unreasonable bar_count: " << m_header->bar_count << std::endl;
        unmap();
        return false;
    }

    // Validate file size matches header bar count
    const size_t expected_size =
        sizeof(FileHeader) + static_cast<size_t>(m_header->bar_count) * sizeof(Bar);
    if (m_mapped_size < expected_size) {
        std::cerr << "[MarketData] File size mismatch. Expected at least " << expected_size
                  << " bytes, got " << m_mapped_size << std::endl;
        unmap();
        return false;
    }

    // Set bar pointer (bars start immediately after header)
    m_bars =
        reinterpret_cast<const Bar*>(static_cast<const char*>(m_mapped_data) + sizeof(FileHeader));

    // Bar-timestamp monotonicity scan. Done AFTER m_bars is assigned
    // (the scan needs m_bars to exist). Reject strictly
    // non-increasing timestamps -- duplicate or out-of-order bars would
    // silently corrupt the session index and all downstream aggregations.
    //
    // The header's bar length drives the daily-close detection and decides
    // when a higher-timeframe bar is complete, so it must be positive and no
    // longer than the spacing of any two bars: an overstated length would
    // mark higher-timeframe bars complete too early.
    {
        const uint32_t bar_count = m_header->bar_count;
        if (bar_count == 0) {
            std::cerr << "[MarketData] File contains no bars." << std::endl;
            unmap();
            return false;
        }
        const int64_t tf_ns = static_cast<int64_t>(m_header->timeframe_sec) * 1'000'000'000LL;
        if (tf_ns <= 0) {
            std::cerr << "[MarketData] Header timeframe_sec must be positive." << std::endl;
            unmap();
            return false;
        }
        for (uint32_t i = 1; i < bar_count; ++i) {
            if (m_bars[i].timestamp <= m_bars[i - 1].timestamp) {
                std::cerr << "[MarketData] ERROR: bar timestamps not strictly "
                             "monotonic at index "
                          << i << " (prev=" << m_bars[i - 1].timestamp
                          << " ns, curr=" << m_bars[i].timestamp << " ns). Rebuild the .bin file."
                          << std::endl;
                unmap();
                return false;
            }
            if (m_bars[i].timestamp - m_bars[i - 1].timestamp < tf_ns) {
                std::cerr << "[MarketData] ERROR: bars " << i - 1 << " and " << i
                          << " are closer together than the header's timeframe_sec ("
                          << m_header->timeframe_sec << " s). Rebuild the .bin file." << std::endl;
                unmap();
                return false;
            }
        }
    }

    std::cout << "[MarketData] Loaded " << m_header->bar_count << " bars"
              << " (" << m_mapped_size / (1024 * 1024) << " MB mapped)"
              << ", symbol=" << symbol() << ", tf=" << m_header->timeframe_sec << "s" << std::endl;

    // The US Eastern clock applies the DST rules in force since 2007.
    constexpr int64_t DST_RULES_FROM_NS = 1'173'596'400LL * NS_PER_SEC;  // 2007-03-11 07:00 UTC
    if (model.clock == SessionModel::Clock::US_EASTERN && m_bars[0].timestamp < DST_RULES_FROM_NS) {
        std::cerr << "[MarketData] WARNING: bars before 2007-03-11 use the post-2007 US DST "
                     "rules; local times can be off by an hour around the old DST dates."
                  << std::endl;
    }

    // Early-close calendar: search next to the bar file, then ../config/,
    // then config/ relative to the working directory.
    m_model = model;
    m_early_close_map.clear();
    if (!m_model.early_close_calendar.empty()) {
        std::string dir;
        auto        last_sep = filepath.find_last_of("/\\");
        if (last_sep != std::string::npos) {
            dir = filepath.substr(0, last_sep + 1);
        }

        const std::string& name         = m_model.early_close_calendar;
        bool               loaded       = false;
        const std::string  candidates[] = {
            dir + name,
            dir + "../config/" + name,
            "config/" + name,
        };
        for (const auto& path : candidates) {
            std::ifstream test(path);
            if (test.good()) {
                test.close();
                load_holiday_calendar(path);
                loaded = true;
                break;
            }
        }
        if (!loaded) {
            std::cerr << "[MarketData] WARNING: " << name << " not found "
                      << "-- early-close detection disabled." << std::endl;
        }
    }

    // Build derived structures
    build_session_index();
    build_timeframe_bars();

    std::cout << "[MarketData] Session index built: " << m_session_count << " sessions"
              << ", " << m_timeframes.size() << " timeframe aggregations" << std::endl;

    return true;
}

// ---------------------------------------------------------------------------
// Holiday / early-close calendar
// ---------------------------------------------------------------------------
void MarketData::load_holiday_calendar(const std::string& csv_path) {
    m_early_close_map.clear();

    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "[MarketData] WARNING: Could not open holiday calendar: " << csv_path
                  << " -- early-close detection disabled." << std::endl;
        return;
    }

    std::string line;
    // Skip header line
    if (!std::getline(file, line)) return;

    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Parse "YYYY-MM-DD,HH:MM"
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;

        std::string date_str = line.substr(0, comma);
        std::string time_str = line.substr(comma + 1);

        // Trim whitespace
        while (!time_str.empty() && (time_str.back() == '\r' || time_str.back() == ' '))
            time_str.pop_back();

        // Parse HH:MM to seconds since midnight
        int                hour = 0, minute = 0;
        char               colon = 0;
        std::istringstream tss(time_str);
        if (!(tss >> hour >> colon >> minute) || colon != ':') continue;

        int32_t close_sec = hour * 3600 + minute * 60;
        // Reject out-of-range close times. Normal CME close is
        // 17:00 ET (61200); a row with that value is a do-nothing entry.
        // Accept the RTH window [09:00, 17:00) ET = [32400, 61200). Sub-9am
        // or >=17:00 closes would either be nonsensical early closes or
        // normal-close entries bloating the map.
        if (close_sec < 32400 || close_sec >= m_model.session_close_sec) {
            std::cerr << "[MarketData] WARNING: early-close row '" << line
                      << "' has close_sec=" << close_sec
                      << " (expected 9:00 <= t < 17:00 ET, i.e. 32400..61199); "
                         "skipping"
                      << std::endl;
            continue;
        }
        m_early_close_map[date_str] = close_sec;
        ++count;
    }

    std::cout << "[MarketData] Loaded " << count << " early-close dates from " << csv_path
              << std::endl;
}

// Convert a UTC nanosecond timestamp to (year, month, day)
static void utc_ns_to_ymd(int64_t ts_ns, int& year, int& month, int& day) {
    int64_t total_days = ts_ns / NS_PER_DAY;
    if (ts_ns < 0) total_days -= 1;  // floor division for negative timestamps

    // Days since Unix epoch (1970-01-01) using a civil calendar algorithm
    // Based on Howard Hinnant's chrono-compatible algorithm
    int64_t z   = total_days + 719468;  // shift epoch from 1970-01-01 to 0000-03-01
    int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    int64_t doe = z - era * 146097;                                       // day of era [0, 146096]
    int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // year of era
    int64_t y   = yoe + era * 400;
    int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);  // day of year [0, 365]
    int64_t mp  = (5 * doy + 2) / 153;                      // month [0, 11]
    day         = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
    month       = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
    year        = static_cast<int>(y + (month <= 2 ? 1 : 0));
}

// ---------------------------------------------------------------------------
// Local clock helpers
// ---------------------------------------------------------------------------
int32_t MarketData::local_offset_sec(int64_t ts_ns) const {
    return m_model.utc_offset_sec(civil::floor_div(ts_ns, NS_PER_SEC));
}

// ---------------------------------------------------------------------------
// Session index
// ---------------------------------------------------------------------------
void MarketData::build_session_index() {
    if (!m_header || m_header->bar_count == 0) return;

    const int n = static_cast<int>(m_header->bar_count);
    m_sessions.resize(n);

    // Deterministic session_day. A session starts at the model's local
    // session-open time (18:00 ET for US futures, 00:00 UTC for 24/7):
    //   session_day_raw = floor((utc_ns + local_offset_ns - open_ns) / 86400_ns)
    // which produces the same value for every bar in one session and a
    // strictly monotonic-by-session value across bars (because UTC timestamps
    // are strictly monotonic, enforced in load()). Gap-based heuristics
    // (e.g. 3h gap + time-sec backwards) can split or merge sessions around
    // holidays / DST, so they are not used.
    //
    // Raw session_day values are large epoch-relative integers; we compress
    // to dense 0-based sequential indices (dense compression rather than
    // "subtract min", so that m_session_first_bar[] has no empty slots across
    // weekends/holidays and remains a contiguous monotonic array suitable for
    // binary search).
    const int64_t SESSION_OPEN_SEC_NS = static_cast<int64_t>(m_model.session_open_sec) * NS_PER_SEC;

    std::vector<int64_t> raw_sd(n);
    int64_t              min_sd = INT64_MAX;

    // First pass: compute raw session_day, time_of_day_sec, and gap_to_prev_sec.
    for (int i = 0; i < n; ++i) {
        int64_t ts_ns  = m_bars[i].timestamp;
        int64_t off_ns = static_cast<int64_t>(local_offset_sec(ts_ns)) * NS_PER_SEC;
        int64_t et_ns  = ts_ns + off_ns;

        // Seconds since local midnight (time_of_day_sec).
        int64_t et_day_ns = et_ns % NS_PER_DAY;
        if (et_day_ns < 0) et_day_ns += NS_PER_DAY;
        m_sessions[i].time_of_day_sec = static_cast<int32_t>(et_day_ns / NS_PER_SEC);

        // Session day: bars in [open today, open tomorrow) share one
        // session_day. We subtract SESSION_OPEN_SEC_NS so the open boundary
        // becomes the day boundary, then floor-divide by 86400 ns.
        int64_t shifted = et_ns - SESSION_OPEN_SEC_NS;
        int64_t sd      = shifted / NS_PER_DAY;
        if (shifted < 0 && shifted % NS_PER_DAY != 0) sd -= 1;  // floor for negatives
        raw_sd[i] = sd;
        if (sd < min_sd) min_sd = sd;

        // gap_to_prev_sec. First bar = 0.
        if (i == 0) {
            m_sessions[i].gap_to_prev_sec = 0;
        } else {
            int64_t gap_ns = m_bars[i].timestamp - m_bars[i - 1].timestamp;
            // Clamp to int32_t range defensively (monotonicity scan already
            // guarantees gap_ns >= 1 ns since timestamps are strictly
            // increasing; giant gaps capped at INT32_MAX seconds = ~68 yrs).
            int64_t gap_s = gap_ns / NS_PER_SEC;
            if (gap_s > INT32_MAX) gap_s = INT32_MAX;
            m_sessions[i].gap_to_prev_sec = static_cast<int32_t>(gap_s);
        }
    }

    // Second pass: compress raw_sd into DENSE session_day indices. Iterate
    // bars in order; each time raw_sd changes (relative to the previous bar),
    // increment current_session. This yields session_day values [0, K-1]
    // where K = number of distinct sessions actually present in the data
    // (weekends / holidays produce NO empty slots -- they simply aren't
    // materialized). Downstream code (binary searches on m_session_first_bar,
    // calendar mapping arrays, etc.) keeps its "dense, strictly increasing
    // first_bar" invariant.
    //
    // Correctness: because bar timestamps are strictly monotonic,
    // raw_sd is weakly monotonic per bar (DST aside, which only nudges a
    // single bar by +/- 1 relative to its neighbors in a way that can't
    // re-order the floor'd daily bucket). Thus same-raw_sd bars are always
    // contiguous.
    int                      current_session = -1;
    int                      intra_idx       = 0;
    int64_t                  prev_raw_sd     = INT64_MIN;
    int                      warn_count      = 0;
    static constexpr int64_t GAP_WARN_NS     = 3LL * 3600 * NS_PER_SEC;  // 3 hours
    (void)min_sd;  // retained for diagnostics; not needed with dense compression

    for (int i = 0; i < n; ++i) {
        bool new_session = (i == 0) || (raw_sd[i] != prev_raw_sd);

        if (new_session) {
            ++current_session;
            intra_idx = 0;
        }

        m_sessions[i].session_day       = current_session;
        m_sessions[i].intra_session_idx = intra_idx;
        m_sessions[i].halt_flag         = 0;
        m_sessions[i].early_close       = 0;
        m_sessions[i].session_close_sec = m_model.session_close_sec;
        m_sessions[i].flags             = 0;
        ++intra_idx;

        // Warn-only on large same-session gaps; they do not split sessions.
        if (i > 0 && raw_sd[i] == prev_raw_sd) {
            int64_t gap_ns = m_bars[i].timestamp - m_bars[i - 1].timestamp;
            if (gap_ns > GAP_WARN_NS && warn_count < 10) {
                std::cerr << "[MarketData] WARNING: large intra-session gap ("
                          << (gap_ns / (60LL * NS_PER_SEC)) << " min) in "
                          << "session_day=" << current_session << " between bars " << (i - 1)
                          << " and " << i << std::endl;
                ++warn_count;
            }
        }
        prev_raw_sd = raw_sd[i];
    }

    m_session_count = current_session + 1;

    // Build reverse mapping: session_day -> first bar index. Dense layout
    // (no -1 sentinels) preserves the monotonic invariant required by
    // first_session_day_in_range / last_session_day_in_range.
    m_session_first_bar.assign(m_session_count, -1);
    for (int i = 0; i < n; ++i) {
        int sd = m_sessions[i].session_day;
        if (m_session_first_bar[sd] < 0) {
            m_session_first_bar[sd] = i;
        }
    }

    // -----------------------------------------------------------------------
    // Early-close detection: check each session's date against holiday calendar.
    // The session date = the calendar date of the RTH portion (bars between
    // 00:00 and 17:00 ET).  We find the first bar in [09:30, 16:00) ET and
    // convert its UTC timestamp to an ET date string for lookup.
    // -----------------------------------------------------------------------
    if (!m_early_close_map.empty()) {
        int early_close_count = 0;
        for (int sd = 0; sd < m_session_count; ++sd) {
            int start = m_session_first_bar[sd];
            if (start < 0) continue;

            // Find a bar in the daytime portion to determine the calendar date.
            // Use the first bar with time_of_day_sec in [09:30, 17:00) (i.e., RTH hours).
            // If no RTH bar exists, try any bar with tod < 18:00 (daytime portion).
            int64_t ref_ts = -1;
            for (int i = start; i < n && m_sessions[i].session_day == sd; ++i) {
                int32_t tod = m_sessions[i].time_of_day_sec;
                if (tod >= m_model.core_open_sec && tod < m_model.session_close_sec) {
                    ref_ts = m_bars[i].timestamp;
                    break;
                }
            }
            // Fallback: use any daytime bar (tod < 18:00 and tod >= 0)
            if (ref_ts < 0) {
                for (int i = start; i < n && m_sessions[i].session_day == sd; ++i) {
                    int32_t tod = m_sessions[i].time_of_day_sec;
                    if (tod < m_model.session_open_sec) {
                        ref_ts = m_bars[i].timestamp;
                        break;
                    }
                }
            }
            if (ref_ts < 0) continue;  // session has no daytime bars (shouldn't happen)

            // Convert UTC timestamp to ET, then to calendar date
            int32_t offset = local_offset_sec(ref_ts);
            int64_t et_ns  = ref_ts + static_cast<int64_t>(offset) * NS_PER_SEC;
            int     year, month, day;
            utc_ns_to_ymd(et_ns, year, month, day);

            // Format as YYYY-MM-DD
            char date_buf[16];
            std::snprintf(date_buf, sizeof(date_buf), "%04d-%02d-%02d", year, month, day);
            std::string date_str(date_buf);

            auto it = m_early_close_map.find(date_str);
            if (it != m_early_close_map.end()) {
                int32_t close_sec = it->second;
                // Mark all bars in this session
                for (int i = start; i < n && m_sessions[i].session_day == sd; ++i) {
                    m_sessions[i].early_close       = 1;
                    m_sessions[i].session_close_sec = close_sec;
                }
                ++early_close_count;
            }
        }
        if (early_close_count > 0) {
            std::cout << "[MarketData] Marked " << early_close_count << " sessions as early-close."
                      << std::endl;
        }

        // Warn if dataset extends beyond the last date in cme_early_closes.csv.
        // In that case early-close sessions won't be flagged and afternoon
        // session signals may incorrectly fire on shortened days.
        if (!m_early_close_map.empty()) {
            std::string max_csv_date;
            for (const auto& [date_str, _] : m_early_close_map) {
                if (date_str > max_csv_date) max_csv_date = date_str;
            }
            // Determine the last bar's ET date
            int64_t last_ts     = m_bars[n - 1].timestamp;
            int32_t last_offset = local_offset_sec(last_ts);
            int64_t last_et_ns  = last_ts + static_cast<int64_t>(last_offset) * NS_PER_SEC;
            int     last_year, last_month, last_day;
            utc_ns_to_ymd(last_et_ns, last_year, last_month, last_day);
            char last_date_buf[16];
            std::snprintf(last_date_buf, sizeof(last_date_buf), "%04d-%02d-%02d", last_year,
                          last_month, last_day);
            std::string last_date(last_date_buf);

            if (last_date > max_csv_date) {
                std::cerr << "[MarketData] WARNING: Dataset extends to " << last_date
                          << " but cme_early_closes.csv only covers through " << max_csv_date
                          << ". Early-close sessions after that date "
                          << "will NOT be flagged. Update the CSV to avoid missed "
                          << "early-close handling." << std::endl;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Halt detection: a gap longer than halt_gap_sec between two consecutive
    // core-hours bars of the same session indicates a halt or missing data.
    // Bars from the first such gap to the end of the session get
    // halt_flag = 1 so entries can be suppressed. Earlier bars of the session
    // are left alone: when they trade, nobody knows a halt is coming. The
    // daily maintenance break lies outside core hours.
    // -----------------------------------------------------------------------
    const int32_t RTH_OPEN_SEC   = m_model.core_open_sec;
    const int32_t RTH_CLOSE_SEC  = m_model.core_close_sec;
    const int64_t MAX_RTH_GAP_NS = static_cast<int64_t>(m_model.halt_gap_sec) * NS_PER_SEC;

    // First pass: first bar after a halt gap in each session (-1 = none)
    std::vector<int> halt_from(m_session_count, -1);

    int rth_gap_warnings = 0;
    for (int i = 1; i < n && m_model.detect_halts; ++i) {
        // Only check consecutive bars in the same session
        if (m_sessions[i].session_day != m_sessions[i - 1].session_day) continue;

        int32_t prev_tod = m_sessions[i - 1].time_of_day_sec;
        int32_t curr_tod = m_sessions[i].time_of_day_sec;

        // Both bars must be within core hours
        if (prev_tod < RTH_OPEN_SEC || prev_tod >= RTH_CLOSE_SEC) continue;
        if (curr_tod < RTH_OPEN_SEC || curr_tod >= RTH_CLOSE_SEC) continue;

        int64_t gap_ns = m_bars[i].timestamp - m_bars[i - 1].timestamp;
        if (gap_ns > MAX_RTH_GAP_NS) {
            int& first = halt_from[m_sessions[i].session_day];
            if (first < 0) first = i;
            if (rth_gap_warnings < 10) {
                double gap_min = static_cast<double>(gap_ns) / (60.0 * 1e9);
                std::cerr << "[MarketData] WARNING: core-hours gap of " << gap_min
                          << " min in session " << m_sessions[i].session_day << " between bars "
                          << (i - 1) << " and " << i << std::endl;
            }
            rth_gap_warnings++;
        }
    }
    if (rth_gap_warnings > 10) {
        std::cerr << "[MarketData] WARNING: " << rth_gap_warnings
                  << " total core-hours gaps detected (showing first 10)" << std::endl;
    }

    // Second pass: flag the rest of each affected session
    int halt_session_count = 0;
    for (int sd = 0; sd < m_session_count; ++sd) {
        const int start = halt_from[sd];
        if (start < 0) continue;
        ++halt_session_count;
        for (int i = start; i < n && m_sessions[i].session_day == sd; ++i) {
            m_sessions[i].halt_flag = 1;
        }
    }
    if (halt_session_count > 0) {
        std::cout << "[MarketData] Flagged " << halt_session_count
                  << " sessions with halt_flag (core-hours gap)." << std::endl;
    }

    // Per-bar flags consumed by the backtest engines (CPU and GPU).
    stamp_session_flags(m_sessions.data(), n, m_model, static_cast<int>(m_header->timeframe_sec));

    // -----------------------------------------------------------------------
    // Build precomputed calendar mappings per session day.
    // A session is named by the local date it closes on, so a session that
    // opens in the evening belongs to the next calendar day: Sunday evening
    // starts Monday's session, in Monday's ISO week and month. These keys
    // avoid repeated date arithmetic in metrics.cpp.
    // -----------------------------------------------------------------------
    m_session_day_to_week.resize(m_session_count, 0);
    m_session_day_to_month.resize(m_session_count, 0);
    m_session_day_to_year.resize(m_session_count, 0);

    for (int sd = 0; sd < m_session_count; ++sd) {
        int bar_idx = m_session_first_bar[sd];
        if (bar_idx < 0) continue;

        // The first bar's local date, moved to the close date when the
        // session opened the evening before.
        int64_t ts     = m_bars[bar_idx].timestamp;
        int32_t offset = local_offset_sec(ts);
        int64_t et_ns  = ts + static_cast<int64_t>(offset) * NS_PER_SEC;
        if (m_model.session_open_sec > 0 &&
            m_sessions[bar_idx].time_of_day_sec >= m_model.session_open_sec)
            et_ns += NS_PER_DAY;

        int year, month, day;
        utc_ns_to_ymd(et_ns, year, month, day);

        // Year and month keys
        m_session_day_to_month[sd] = year * 100 + month;
        m_session_day_to_year[sd]  = year;

        // ISO week calculation (same algorithm as metrics.cpp iso_year_week)
        // Need day-of-week and day-of-year from (year, month, day).

        // Day of week via Tomohiko Sakamoto's algorithm (0=Sunday..6=Saturday)
        {
            static constexpr int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
            int                  y   = year;
            if (month < 3) y -= 1;
            int dow_sun = (y + y / 4 - y / 100 + y / 400 + t[month - 1] + day) % 7;
            // Convert to Monday=0 .. Sunday=6
            int dow = (dow_sun + 6) % 7;

            // Day of year (1-based -> 0-based for ISO calculation)
            static constexpr int days_before_month[] = {0,   31,  59,  90,  120, 151,
                                                        181, 212, 243, 273, 304, 334};
            int                  yday                = days_before_month[month - 1] + day - 1;
            // Add leap day if applicable
            if (month > 2 && is_leap_year(year)) {
                yday += 1;
            }

            int week     = (yday - dow + 10) / 7;
            int iso_year = year;

            if (week < 1) {
                iso_year -= 1;
                week = iso_weeks_in_year(iso_year);
            } else if (week > 52) {
                // Check if Dec 31 is Thursday or later for week 53
                int days_in_year  = 365 + (is_leap_year(year) ? 1 : 0);
                int days_to_dec31 = days_in_year - 1 - yday;
                int dec31_wday    = (dow_sun + days_to_dec31) % 7;
                int dec31_dow_mon = (dec31_wday + 6) % 7;
                if (week > 53 || (week == 53 && dec31_dow_mon < 3)) {
                    iso_year += 1;
                    week = 1;
                }
            }

            m_session_day_to_week[sd] = iso_year * 100 + week;
        }
    }
}

// ---------------------------------------------------------------------------
// Timeframe aggregation
// ---------------------------------------------------------------------------

// Floor ET seconds to a timeframe boundary
static int32_t floor_to_tf(int32_t et_sec, int tf_sec) {
    // Floor to tf_sec boundary.  For TFs that are factors of 3600 this
    // gives clean boundaries (e.g., 300 -> 0, 300, 600, ...).
    // We floor within the day.
    return (et_sec / tf_sec) * tf_sec;
}

void MarketData::build_timeframe_bars() {
    if (!m_header || m_header->bar_count == 0) return;

    const int     n           = static_cast<int>(m_header->bar_count);
    const int32_t base_tf_sec = static_cast<int32_t>(m_header->timeframe_sec);
    m_timeframes.resize(NUM_TARGET_TFS);

    for (int tfi = 0; tfi < NUM_TARGET_TFS; ++tfi) {
        const int tf_sec  = TARGET_TFS[tfi];
        auto&     tfd     = m_timeframes[tfi];
        tfd.timeframe_sec = tf_sec;
        tfd.parent_map.resize(n);
        tfd.completed_map.resize(n);
        tfd.bars.reserve(n / (tf_sec / 30) + 100);  // rough estimate

        int     current_tf_bar   = -1;
        int32_t current_boundary = -1;
        int32_t current_session  = -1;

        for (int i = 0; i < n; ++i) {
            int32_t et_sec      = m_sessions[i].time_of_day_sec;
            int32_t session_day = m_sessions[i].session_day;
            int32_t boundary    = floor_to_tf(et_sec, tf_sec);

            // Start a new TF bar when the boundary changes OR the session changes
            bool new_tf_bar = (current_tf_bar < 0) || (boundary != current_boundary) ||
                              (session_day != current_session);

            if (new_tf_bar) {
                // Start a new aggregated bar
                Bar agg{};
                agg.timestamp = m_bars[i].timestamp;
                agg.open      = m_bars[i].open;
                agg.high      = m_bars[i].high;
                agg.low       = m_bars[i].low;
                agg.close     = m_bars[i].close;
                agg.volume    = m_bars[i].volume;
                agg._pad      = 0;
                tfd.bars.push_back(agg);
                current_tf_bar   = static_cast<int>(tfd.bars.size()) - 1;
                current_boundary = boundary;
                current_session  = session_day;
            } else {
                // Aggregate into existing bar
                Bar& agg = tfd.bars[current_tf_bar];
                if (m_bars[i].high > agg.high) agg.high = m_bars[i].high;
                if (m_bars[i].low < agg.low) agg.low = m_bars[i].low;
                agg.close = m_bars[i].close;
                // Saturate rather than wrap on pathological volumes.
                const uint64_t volume =
                    static_cast<uint64_t>(agg.volume) + static_cast<uint64_t>(m_bars[i].volume);
                agg.volume = static_cast<uint32_t>(std::min<uint64_t>(volume, UINT32_MAX));
            }

            tfd.parent_map[i] = current_tf_bar;

            // The current TF bar is complete once this base bar closes at or
            // past the TF boundary; until then only the previous TF bar is
            // final. The clock, not the next bar's arrival, decides, so the
            // map stays causal across missing bars.
            const int32_t bar_end_sec = et_sec + base_tf_sec;
            tfd.completed_map[i] =
                (bar_end_sec >= current_boundary + tf_sec) ? current_tf_bar : current_tf_bar - 1;
        }

        tfd.bars.shrink_to_fit();
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
const Bar& MarketData::bar(int index) const {
    assert(m_bars != nullptr && "MarketData::bar() called on unloaded MarketData");
    assert(index >= 0 && index < bar_count() && "MarketData::bar() index out of range");
    return m_bars[index];
}

int MarketData::bar_count() const {
    return m_header ? static_cast<int>(m_header->bar_count) : 0;
}

const FileHeader& MarketData::header() const {
    if (!m_header) {
        throw std::runtime_error(
            "[MarketData] header() called on unloaded MarketData (null header)");
    }
    return *m_header;
}

const SessionInfo& MarketData::session(int bar_index) const {
    assert(bar_index >= 0 && bar_index < static_cast<int>(m_sessions.size()) &&
           "MarketData::session() index out of range");
    return m_sessions[bar_index];
}

int MarketData::session_count() const {
    return m_session_count;
}

int MarketData::first_bar_of_session(int session_day) const {
    if (session_day < 0 || session_day >= m_session_count) return -1;
    return m_session_first_bar[session_day];
}

int MarketData::first_session_day_in_range(int bar_start) const {
    // Binary search for the first session_day whose first_bar >= bar_start.
    // m_session_first_bar is monotonically increasing by construction.
    if (m_session_count == 0) return -1;
    int lo = 0, hi = m_session_count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (m_session_first_bar[mid] < bar_start)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (lo < m_session_count) ? lo : -1;
}

int MarketData::last_session_day_in_range(int bar_end) const {
    // Binary search for the last session_day whose first_bar < bar_end.
    if (m_session_count == 0) return -1;
    int lo = 0, hi = m_session_count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (m_session_first_bar[mid] < bar_end)
            lo = mid + 1;
        else
            hi = mid;
    }
    // lo is the first session_day with first_bar >= bar_end, so last valid is lo - 1
    return (lo > 0) ? lo - 1 : -1;
}

double MarketData::bars_per_calendar_day() const {
    const int n = bar_count();
    if (n < 2) return static_cast<double>(n);
    const double span_days = static_cast<double>(m_bars[n - 1].timestamp - m_bars[0].timestamp) /
                             static_cast<double>(NS_PER_DAY);
    return span_days > 0.0 ? static_cast<double>(n) / std::max(1.0, span_days)
                           : static_cast<double>(n);
}

std::string MarketData::symbol() const {
    if (!m_header) return {};
    // symbol field is char[8], may not be null-terminated if 8 chars long
    return std::string(m_header->symbol, strnlen(m_header->symbol, sizeof(m_header->symbol)));
}

// ---------------------------------------------------------------------------
// data_hash -- FNV-1a 64 over the mapped bar payload.
//
// Hashes the FileHeader (sans the `reserved` region, which may differ between
// identical datasets written by different tool versions) plus the Bar array.
// Computed once on first call and cached.
// Used as a resume invariant: changing the underlying bar file between runs
// will change this hash and cause Checkpoint::load to reject the old file.
// ---------------------------------------------------------------------------
uint64_t MarketData::data_hash() const {
    if (m_data_hash_cached) return m_data_hash;

    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;
    uint64_t           h          = FNV_OFFSET;

    auto mix_bytes = [&](const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; ++i) {
            h ^= p[i];
            h *= FNV_PRIME;
        }
    };

    if (m_header) {
        // Hash the stable header prefix: magic + version + bar_count + symbol +
        // timeframe_sec + start_ts + end_ts. Skip `reserved[24]`.
        const size_t stable_header_bytes = sizeof(FileHeader) - sizeof(m_header->reserved);
        mix_bytes(m_header, stable_header_bytes);
    }
    if (m_bars && m_header) {
        const size_t nbytes = static_cast<size_t>(m_header->bar_count) * sizeof(Bar);
        mix_bytes(m_bars, nbytes);
    }

    m_data_hash        = h;
    m_data_hash_cached = true;
    return h;
}

// ---------------------------------------------------------------------------
// Binary search
// ---------------------------------------------------------------------------
int MarketData::find_bar(int64_t timestamp_ns) const {
    if (!m_bars || !m_header || m_header->bar_count == 0) return -1;

    const int n = static_cast<int>(m_header->bar_count);

    // Binary search for exact match
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (m_bars[mid].timestamp == timestamp_ns) return mid;
        if (m_bars[mid].timestamp < timestamp_ns)
            lo = mid + 1;
        else
            hi = mid - 1;
    }
    return -1;
}

int MarketData::find_bar_ge(int64_t timestamp_ns) const {
    if (!m_bars || !m_header || m_header->bar_count == 0) return -1;

    const int n = static_cast<int>(m_header->bar_count);

    // Binary search for lower_bound (first bar >= timestamp_ns)
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (m_bars[mid].timestamp < timestamp_ns)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (lo < n) ? lo : -1;
}

// ---------------------------------------------------------------------------
// Timeframe lookup
// ---------------------------------------------------------------------------
const MarketData::TimeframeData& MarketData::timeframe(int tf_sec) const {
    for (const auto& tfd : m_timeframes) {
        if (tfd.timeframe_sec == tf_sec) return tfd;
    }
    // Should not happen -- caller must use a valid timeframe
    throw std::invalid_argument("[MarketData] Unknown timeframe: " + std::to_string(tf_sec) + "s");
}

// Non-throwing lookup.
const MarketData::TimeframeData* MarketData::try_timeframe(int tf_sec) const {
    for (const auto& tfd : m_timeframes) {
        if (tfd.timeframe_sec == tf_sec) return &tfd;
    }
    return nullptr;
}
