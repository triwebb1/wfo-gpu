#pragma once
// ---------------------------------------------------------------------------
// Common utilities shared across modules.
// ---------------------------------------------------------------------------

#include <cmath>
#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

// Clamp a float to [-1, +1], preserving NaN for warmup.
inline float clamp_signal(float v) {
    if (std::isnan(v)) return v;
    return std::max(-1.0f, std::min(1.0f, v));
}

// ---------------------------------------------------------------------------
// Percentile -- linear interpolation on a pre-sorted vector
// ---------------------------------------------------------------------------
inline float percentile(const std::vector<float>& sorted, float p) {
    if (sorted.size() <= 1) return sorted.empty() ? 0.0f : sorted[0];
    float idx = p * static_cast<float>(sorted.size() - 1);
    int   lo  = static_cast<int>(idx);
    int   hi  = lo + 1;
    if (hi >= static_cast<int>(sorted.size())) return sorted.back();
    float frac = idx - static_cast<float>(lo);
    return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

// ---------------------------------------------------------------------------
// Calendar helpers
// ---------------------------------------------------------------------------

inline bool is_leap_year(int year) {
    return (year % 4 == 0) && (year % 100 != 0 || year % 400 == 0);
}

// Number of ISO 8601 weeks in a year (52 or 53).
// A year has 53 weeks if Jan 1 is Thursday, or if it is a leap year
// whose Jan 1 is Wednesday.
inline int iso_weeks_in_year(int year) {
    // Tomohiko Sakamoto: day-of-week for Jan 1 (0=Sunday)
    static constexpr int t[]     = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    int                  y       = year - 1;  // month=1 < 3
    int                  dow_sun = (y + y / 4 - y / 100 + y / 400 + t[0] + 1) % 7;
    int                  dow_mon = (dow_sun + 6) % 7;  // Monday=0
    return (dow_mon == 3 || (is_leap_year(year) && dow_mon == 2)) ? 53 : 52;
}

// ---------------------------------------------------------------------------
// Metric helpers -- single source of truth for sentinel-safe ratios
// ---------------------------------------------------------------------------

// Profit factor: gross_wins / |gross_losses|.  Returns 1e6f sentinel when
// all trades are winners (no losses), 0 when no trades.
inline float compute_profit_factor(float gross_wins, float gross_losses) {
    float abs_losses = std::abs(gross_losses);
    return (abs_losses > 0.0f) ? (gross_wins / abs_losses) : (gross_wins > 0.0f) ? 1e6f : 0.0f;
}

// Payoff ratio: avg_win / |avg_loss|.  Same sentinel logic.
inline float compute_payoff_ratio(float avg_win, float avg_loss) {
    float abs_avg_loss = std::abs(avg_loss);
    return (abs_avg_loss > 0.0f) ? (avg_win / abs_avg_loss) : (avg_win > 0.0f) ? 1e6f : 0.0f;
}

// ---------------------------------------------------------------------------
// JoiningThread -- a std::thread that joins on destruction. std::jthread
// would do, but some standard libraries (libc++ without experimental
// features) do not ship it yet.
// ---------------------------------------------------------------------------
class JoiningThread {
public:
    template <class F>
    explicit JoiningThread(F&& f) : m_thread(std::forward<F>(f)) {}
    JoiningThread(JoiningThread&&) noexcept            = default;
    JoiningThread& operator=(JoiningThread&&) noexcept = delete;
    JoiningThread(const JoiningThread&)                = delete;
    JoiningThread& operator=(const JoiningThread&)     = delete;
    ~JoiningThread() {
        if (m_thread.joinable()) m_thread.join();
    }

private:
    std::thread m_thread;
};
