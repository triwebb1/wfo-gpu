#include "synthetic.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "civil_time.h"

namespace {

// xoshiro256** seeded through splitmix64. Fixed algorithm, so streams are
// identical everywhere (std:: distributions are implementation-defined).
class Rng {
public:
    explicit Rng(uint64_t seed) {
        for (auto& s : m_state) s = splitmix64(seed);
    }

    uint64_t next() {
        const uint64_t result = rotl(m_state[1] * 5, 7) * 9;
        const uint64_t t      = m_state[1] << 17;
        m_state[2] ^= m_state[0];
        m_state[3] ^= m_state[1];
        m_state[1] ^= m_state[2];
        m_state[0] ^= m_state[3];
        m_state[2] ^= t;
        m_state[3] = rotl(m_state[3], 45);
        return result;
    }

    // Uniform in [0, 1).
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

    // Approximately standard normal (Irwin-Hall, 12 uniforms). Avoids
    // transcendental functions, whose last-bit results vary by platform.
    double normal() {
        double sum = 0.0;
        for (int i = 0; i < 12; ++i) sum += uniform();
        return sum - 6.0;
    }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    static uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z          = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    uint64_t m_state[4];
};

struct Session {
    int64_t open_utc_sec;
    int64_t close_utc_sec;
    int32_t open_local_sec;  // local seconds since midnight at session open
    bool    after_weekend;   // first session after a non-trading stretch
};

constexpr int32_t HOUR = 3600;

// Relative volatility by local time of day.
double us_futures_season(int32_t et_sec) {
    if (et_sec >= 18 * HOUR || et_sec < 3 * HOUR) return 0.55;
    if (et_sec < 8 * HOUR + 1800) return 0.75;
    if (et_sec < 9 * HOUR + 1800) return 1.10;
    if (et_sec < 10 * HOUR + 1800) return 1.90;
    if (et_sec < 14 * HOUR) return 1.00;
    if (et_sec < 15 * HOUR) return 1.10;
    if (et_sec < 16 * HOUR) return 1.50;
    return 0.60;
}

double continuous_season(int32_t utc_sec) {
    if (utc_sec >= 13 * HOUR && utc_sec < 21 * HOUR) return 1.30;
    if (utc_sec < 2 * HOUR) return 1.00;
    return 0.80;
}

std::vector<Session> build_sessions(const SyntheticConfig& c) {
    std::vector<Session> sessions;
    int64_t              day = civil::days_from_civil(c.start_year, c.start_month, c.start_day);
    int64_t              prev_day = day - 1;
    while (static_cast<int>(sessions.size()) < c.num_days) {
        if (c.calendar == SyntheticCalendar::CONTINUOUS) {
            sessions.push_back(
                {day * civil::SEC_PER_DAY, (day + 1) * civil::SEC_PER_DAY, 0, false});
            ++day;
            continue;
        }
        const int wd = civil::weekday_from_days(day);
        if (wd != 0 && wd != 6) {
            const civil::Date d = civil::civil_from_days(day);
            const civil::Date p = civil::civil_from_days(day - 1);
            Session           s;
            s.open_utc_sec   = civil::us_eastern_to_utc_sec(p.year, p.month, p.day, 18 * HOUR);
            s.close_utc_sec  = civil::us_eastern_to_utc_sec(d.year, d.month, d.day, 17 * HOUR);
            s.open_local_sec = 18 * HOUR;
            s.after_weekend  = (day - prev_day) > 1;
            sessions.push_back(s);
            prev_day = day;
        }
        ++day;
    }
    return sessions;
}

}  // namespace

std::vector<Bar> generate_synthetic_bars(const SyntheticConfig& c) {
    if (c.timeframe_sec <= 0 || c.timeframe_sec > 86400 || 86400 % c.timeframe_sec != 0)
        throw std::invalid_argument("timeframe_sec must evenly divide one day");
    if (!(c.tick_size > 0.0) || !(c.start_price > c.tick_size))
        throw std::invalid_argument("start_price must exceed a positive tick_size");
    if (c.num_days <= 0 || c.substeps <= 0)
        throw std::invalid_argument("num_days and substeps must be positive");
    if (c.trend_strength < 0.0 || !(c.trend_duration_bars >= 1.0))
        throw std::invalid_argument("trend_strength must be >= 0 and trend_duration_bars >= 1");

    const bool   continuous     = c.calendar == SyntheticCalendar::CONTINUOUS;
    const double tf             = static_cast<double>(c.timeframe_sec);
    const double bars_per_year  = continuous ? 365.0 * 86400.0 / tf : 252.0 * 23.0 * 3600.0 / tf;
    const double sigma_bar      = c.annual_volatility / std::sqrt(bars_per_year);
    const double mu_step        = c.annual_drift / bars_per_year / c.substeps;
    const double sigma_hour     = sigma_bar * std::sqrt(3600.0 / tf);
    const double inv_sqrt_steps = 1.0 / std::sqrt(static_cast<double>(c.substeps));

    auto season = [&](int32_t local_sec) {
        return continuous ? continuous_season(local_sec) : us_futures_season(local_sec);
    };

    // Normalize seasonality so its mean over a session is 1, keeping the
    // configured annual volatility accurate.
    double        season_sum  = 0.0;
    int           season_n    = 0;
    const int32_t session_len = continuous ? 24 * HOUR : 23 * HOUR;
    const int32_t first_local = continuous ? 0 : 18 * HOUR;
    for (int32_t t = 0; t < session_len; t += c.timeframe_sec) {
        season_sum += season((first_local + t) % (24 * HOUR));
        ++season_n;
    }
    const double season_norm = season_n / season_sum;

    const double regime_mult[3] = {c.low_vol_mult, 1.0, c.high_vol_mult};
    int          regime         = 1;

    Rng        rng(c.seed);
    const auto sessions = build_sessions(c);
    const auto quantize = [&](double x) { return std::floor(x / c.tick_size + 0.5) * c.tick_size; };

    std::vector<Bar> bars;
    bars.reserve(sessions.size() * static_cast<size_t>(session_len / c.timeframe_sec));
    double price     = c.start_price;
    int    trend_dir = 1;  // sign of the hidden trend regime

    for (size_t si = 0; si < sessions.size(); ++si) {
        const Session& s = sessions[si];
        if (si > 0 && !continuous) {
            const double gap = c.gap_sigmas * sigma_hour * (s.after_weekend ? 2.0 : 1.0);
            price *= 1.0 + gap * rng.normal();
        }

        for (int64_t t = s.open_utc_sec; t < s.close_utc_sec; t += c.timeframe_sec) {
            if (rng.uniform() < c.regime_switch_prob)
                regime = (regime == 1) ? (rng.uniform() < 0.5 ? 0 : 2) : 1;

            const int32_t local_sec =
                static_cast<int32_t>((s.open_local_sec + (t - s.open_utc_sec)) % (24 * HOUR));
            const double sea        = season(local_sec) * season_norm;
            const double sigma      = sigma_bar * sea * regime_mult[regime];
            const double step_sigma = sigma * inv_sqrt_steps;

            const double open = price;
            // Draw only when trends are enabled, so trend_strength = 0
            // reproduces the plain random walk byte for byte.
            double carry = 0.0;
            if (c.trend_strength > 0.0) {
                if (rng.uniform() < 1.0 / c.trend_duration_bars) trend_dir = -trend_dir;
                carry = trend_dir * c.trend_strength * sigma / c.substeps;
            }
            double hi = price, lo = price;
            for (int k = 0; k < c.substeps; ++k) {
                double r = mu_step + carry + step_sigma * rng.normal();
                if (rng.uniform() < c.jump_prob / c.substeps)
                    r += (rng.uniform() < 0.5 ? -1.0 : 1.0) * c.jump_sigmas * sigma;
                price *= 1.0 + r;
                price = std::max(price, 2.0 * c.tick_size);
                hi    = std::max(hi, price);
                lo    = std::min(lo, price);
            }

            const double qo = quantize(open), qc = quantize(price);
            Bar          b{};
            b.timestamp = t * civil::NS_PER_SEC;
            b.open      = static_cast<float>(qo);
            b.close     = static_cast<float>(qc);
            b.high      = static_cast<float>(std::max({quantize(hi), qo, qc}));
            b.low       = static_cast<float>(std::min({quantize(lo), qo, qc}));

            const double move = sigma > 0.0 ? std::abs(qc - qo) / (open * sigma) : 0.0;
            const double vol  = c.base_volume * sea * std::sqrt(sea) * (0.6 + 0.8 * rng.uniform()) *
                               (1.0 + 0.5 * move);
            b.volume = static_cast<uint32_t>(std::max(1.0, std::floor(vol + 0.5)));
            bars.push_back(b);
        }
    }
    return bars;
}
