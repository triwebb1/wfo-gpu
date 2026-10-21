#include "bar_io.h"

#include <cstdint>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#include "civil_time.h"
#include "durable_file.h"

bool write_bar_file(const std::string& path, const std::string& symbol, int timeframe_sec,
                    const std::vector<Bar>& bars, std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };
    if (bars.empty()) return fail("no bars to write");
    if (bars.size() > std::numeric_limits<uint32_t>::max()) return fail("too many bars");
    if (timeframe_sec <= 0) return fail("timeframe_sec must be positive");
    const int64_t tf_ns = static_cast<int64_t>(timeframe_sec) * 1'000'000'000LL;
    for (size_t i = 1; i < bars.size(); ++i) {
        if (bars[i].timestamp <= bars[i - 1].timestamp)
            return fail("timestamps must strictly increase (bar " + std::to_string(i) + ")");
        if (bars[i].timestamp - bars[i - 1].timestamp < tf_ns)
            return fail("bars " + std::to_string(i - 1) + " and " + std::to_string(i) +
                        " are closer together than the timeframe (" +
                        std::to_string(timeframe_sec) + " s)");
    }

    FileHeader hdr{};
    std::memcpy(hdr.magic, "WFBT", 4);
    hdr.version   = 1;
    hdr.bar_count = static_cast<uint32_t>(bars.size());
    std::memcpy(hdr.symbol, symbol.data(), std::min(symbol.size(), sizeof(hdr.symbol)));
    hdr.timeframe_sec = static_cast<uint32_t>(timeframe_sec);
    hdr.start_ts      = bars.front().timestamp;
    hdr.end_ts        = bars.back().timestamp;

    try {
        DurableFileWriter::atomic_write(path, [&](DurableFileWriter& w) {
            w.write_bytes(&hdr, sizeof(hdr));
            w.write_bytes(bars.data(), bars.size() * sizeof(Bar));
        });
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    return true;
}

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (std::isspace(static_cast<unsigned char>(s[a])) || s[a] == '"')) ++a;
    while (b > a && (std::isspace(static_cast<unsigned char>(s[b - 1])) || s[b - 1] == '"')) --b;
    return s.substr(a, b - a);
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string              cur;
    for (char c : line) {
        if (c == ',') {
            out.push_back(trim(cur));
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    out.push_back(trim(cur));
    return out;
}

bool parse_int64(const std::string& s, int64_t& out) {
    if (s.empty()) return false;
    char* end         = nullptr;
    errno             = 0;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || end != s.c_str() + s.size()) return false;
    out = v;
    return true;
}

bool parse_double(const std::string& s, double& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    out       = std::strtod(s.c_str(), &end);
    return end == s.c_str() + s.size();
}

// Parse exactly n digits at s[pos].
bool digits(const std::string& s, size_t pos, size_t n, int& out) {
    if (pos + n > s.size()) return false;
    out = 0;
    for (size_t i = pos; i < pos + n; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
        out = out * 10 + (s[i] - '0');
    }
    return true;
}

bool parse_iso8601(const std::string& s, int64_t& out_ns) {
    int y, mo, d, h = 0, mi = 0, sec = 0;
    if (!digits(s, 0, 4, y) || s.size() < 10 || s[4] != '-' || !digits(s, 5, 2, mo) ||
        s[7] != '-' || !digits(s, 8, 2, d))
        return false;
    if (mo < 1 || mo > 12 || d < 1 || d > 31) return false;
    size_t  pos     = 10;
    int64_t frac_ns = 0;
    if (pos < s.size() && (s[pos] == 'T' || s[pos] == ' ')) {
        if (!digits(s, pos + 1, 2, h) || pos + 3 >= s.size() || s[pos + 3] != ':' ||
            !digits(s, pos + 4, 2, mi))
            return false;
        pos += 6;
        if (pos < s.size() && s[pos] == ':') {
            if (!digits(s, pos + 1, 2, sec)) return false;
            pos += 3;
            if (pos < s.size() && s[pos] == '.') {
                ++pos;
                int64_t scale = 100'000'000;
                while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
                    frac_ns += (s[pos] - '0') * scale;
                    scale /= 10;
                    ++pos;
                }
            }
        }
    }
    if (h > 23 || mi > 59 || sec > 60) return false;
    int64_t offset_sec = 0;
    if (pos < s.size()) {
        if (s[pos] == 'Z' && pos + 1 == s.size()) {
            ++pos;
        } else if (s[pos] == '+' || s[pos] == '-') {
            int        oh, om;
            const bool colon = pos + 3 < s.size() && s[pos + 3] == ':';
            if (!digits(s, pos + 1, 2, oh) || !digits(s, pos + (colon ? 4 : 3), 2, om))
                return false;
            offset_sec = (oh * 3600 + om * 60) * (s[pos] == '-' ? -1 : 1);
            pos += colon ? 6 : 5;
        }
    }
    if (pos != s.size()) return false;
    const int64_t days    = civil::days_from_civil(y, mo, d);
    const int64_t utc_sec = days * civil::SEC_PER_DAY + h * 3600 + mi * 60 + sec - offset_sec;
    out_ns                = utc_sec * civil::NS_PER_SEC + frac_ns;
    return true;
}

int column_for(const std::string& name) {
    const std::string n = lower(name);
    if (n == "timestamp" || n == "time" || n == "date" || n == "datetime" || n == "ts" ||
        n == "ts_event")
        return 0;
    if (n == "open" || n == "o") return 1;
    if (n == "high" || n == "h") return 2;
    if (n == "low" || n == "l") return 3;
    if (n == "close" || n == "c") return 4;
    if (n == "volume" || n == "vol" || n == "v") return 5;
    return -1;
}

}  // namespace

bool parse_timestamp_ns(const std::string& field, int64_t& out_ns) {
    const std::string s = trim(field);
    int64_t           v;
    if (parse_int64(s, v)) {
        // Seconds, milliseconds, microseconds or nanoseconds, told apart by
        // magnitude. A value whose nanosecond equivalent overflows is rejected.
        const int64_t a     = (v == INT64_MIN) ? INT64_MAX : (v < 0 ? -v : v);
        const int64_t scale = a < 100'000'000'000LL           ? 1'000'000'000LL
                              : a < 100'000'000'000'000LL     ? 1'000'000LL
                              : a < 100'000'000'000'000'000LL ? 1'000LL
                                                              : 1LL;
        if (a > INT64_MAX / scale) return false;
        out_ns = v * scale;
        return true;
    }
    return parse_iso8601(s, out_ns);
}

CsvImportResult read_bars_csv(std::istream& in) {
    CsvImportResult        r;
    int                    col[6] = {0, 1, 2, 3, 4, 5};
    std::string            line;
    int                    line_no    = 0;
    bool                   first_data = true;
    std::map<int64_t, int> spacing_counts;

    auto fail = [&](const std::string& msg) {
        r.error = "line " + std::to_string(line_no) + ": " + msg;
        r.bars.clear();
        return r;
    };

    while (std::getline(in, line)) {
        ++line_no;
        if (trim(line).empty() || line[0] == '#') continue;
        const auto f = split_csv(line);

        if (first_data) {
            first_data = false;
            int64_t probe;
            if (!parse_timestamp_ns(f[0], probe)) {
                // Header row: map columns by name.
                int found[6] = {-1, -1, -1, -1, -1, -1};
                for (size_t i = 0; i < f.size(); ++i) {
                    const int c = column_for(f[i]);
                    if (c >= 0 && found[c] < 0) found[c] = static_cast<int>(i);
                }
                for (int c = 0; c < 6; ++c) {
                    if (found[c] < 0)
                        return fail(
                            "header is missing a column (need timestamp,open,high,"
                            "low,close,volume)");
                    col[c] = found[c];
                }
                continue;
            }
        }

        const int needed = *std::max_element(col, col + 6) + 1;
        if (static_cast<int>(f.size()) < needed)
            return fail("expected at least " + std::to_string(needed) + " fields");

        Bar b{};
        if (!parse_timestamp_ns(f[col[0]], b.timestamp))
            return fail("cannot parse timestamp '" + f[col[0]] + "'");
        double px[4];
        for (int k = 0; k < 4; ++k) {
            if (!parse_double(f[col[1 + k]], px[k]) || !std::isfinite(px[k]) || px[k] <= 0.0)
                return fail("invalid price '" + f[col[1 + k]] + "'");
        }
        double vol;
        if (!parse_double(f[col[5]], vol) || !std::isfinite(vol) || vol < 0.0 ||
            vol > static_cast<double>(std::numeric_limits<uint32_t>::max()))
            return fail("invalid volume '" + f[col[5]] + "'");
        if (px[1] < std::max(px[0], px[3]) || px[2] > std::min(px[0], px[3]) || px[1] < px[2])
            return fail("high/low inconsistent with open/close");

        b.open   = static_cast<float>(px[0]);
        b.high   = static_cast<float>(px[1]);
        b.low    = static_cast<float>(px[2]);
        b.close  = static_cast<float>(px[3]);
        b.volume = static_cast<uint32_t>(std::floor(vol + 0.5));

        if (!r.bars.empty()) {
            const int64_t gap = b.timestamp - r.bars.back().timestamp;
            if (gap <= 0) return fail("timestamp does not increase");
            ++spacing_counts[gap];
        }
        r.bars.push_back(b);
        ++r.rows_read;
    }

    if (r.bars.empty()) {
        r.error = "no bars found";
        return r;
    }
    int64_t best_gap   = 0;
    int     best_count = 0;
    for (const auto& [gap, count] : spacing_counts) {
        if (count > best_count) {
            best_gap   = gap;
            best_count = count;
        }
    }
    r.timeframe_sec = static_cast<int>(best_gap / civil::NS_PER_SEC);
    if (r.bars.size() > 1 && r.timeframe_sec <= 0) {
        r.error = "bar spacing is below one second; only whole-second timeframes are supported";
        r.bars.clear();
    }
    return r;
}

CsvImportResult read_bars_csv_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        CsvImportResult r;
        r.error = "cannot open " + path;
        return r;
    }
    return read_bars_csv(in);
}
