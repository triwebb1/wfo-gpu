#include "connors_rsi.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

ConnorsRSI::ConnorsRSI(int rsi_period, int streak_rsi_period, int percentile_period)
    : m_rsi_period(rsi_period),
      m_streak_rsi_period(streak_rsi_period),
      m_percentile_period(percentile_period) {
    assert(rsi_period > 0);
    assert(streak_rsi_period > 0);
    assert(percentile_period > 0);
}

int ConnorsRSI::num_series() const {
    return 1;
}

int ConnorsRSI::warmup_bars() const {
    // RSI components use Wilder smoothing (3x period for stabilization).
    // PercentRank needs percentile_period bars of change history.
    return std::max({3 * m_rsi_period, 3 * m_streak_rsi_period, m_percentile_period});
}

const char* ConnorsRSI::name() const {
    return "ConnorsRSI";
}

// Compute Wilder-smoothed RSI on arbitrary sequential data.
// This follows the same algorithm as the RSI indicator class but operates
// on a generic float array rather than Bar close prices.
void ConnorsRSI::compute_rsi_on_data(const float* data, int count, int period, float* rsi_out) {
    const float period_f  = static_cast<float>(period);
    const float period_m1 = period_f - 1.0f;

    float avg_gain        = 0.0f;
    float avg_loss        = 0.0f;
    bool  initialized     = false;
    int   changes_counted = 0;

    for (int i = 0; i < count; ++i) {
        if (i == 0) {
            rsi_out[i] = INDICATOR_NAN;
            continue;
        }

        const float diff = data[i] - data[i - 1];
        const float gain = (diff > 0.0f) ? diff : 0.0f;
        const float loss = (diff < 0.0f) ? -diff : 0.0f;

        if (!initialized) {
            avg_gain += gain;
            avg_loss += loss;
            changes_counted++;

            if (changes_counted == period) {
                avg_gain /= period_f;
                avg_loss /= period_f;
                initialized = true;

                if (avg_gain == 0.0f && avg_loss == 0.0f) {
                    rsi_out[i] = 50.0f;
                } else if (avg_loss == 0.0f) {
                    rsi_out[i] = 100.0f;
                } else {
                    const float rs = avg_gain / avg_loss;
                    rsi_out[i]     = 100.0f - 100.0f / (1.0f + rs);
                }
            } else {
                rsi_out[i] = INDICATOR_NAN;
            }
        } else {
            avg_gain = (avg_gain * period_m1 + gain) / period_f;
            avg_loss = (avg_loss * period_m1 + loss) / period_f;

            if (avg_gain == 0.0f && avg_loss == 0.0f) {
                rsi_out[i] = 50.0f;
            } else if (avg_loss == 0.0f) {
                rsi_out[i] = 100.0f;
            } else {
                const float rs = avg_gain / avg_loss;
                rsi_out[i]     = 100.0f - 100.0f / (1.0f + rs);
            }
        }
    }
}

void ConnorsRSI::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = warmup_bars();

    // -- Step 1: Extract close prices ------------------------------------
    std::vector<float> closes(N);
    for (int i = 0; i < N; ++i) {
        closes[i] = bars[start + i].close;
    }

    // -- Step 2: Compute close-to-close changes -------------------------
    std::vector<float> changes(N);
    changes[0] = INDICATOR_NAN;  // No prior bar; NaN avoids polluting PercentRank
    for (int i = 1; i < N; ++i) {
        changes[i] = closes[i] - closes[i - 1];
    }

    // -- Step 3: Compute streak series ----------------------------------
    // Streak: count consecutive up closes (positive) or down closes (negative).
    // Reset on direction change.
    std::vector<float> streaks(N);
    streaks[0] = 0.0f;
    for (int i = 1; i < N; ++i) {
        float chg = changes[i];
        if (chg > 0.0f) {
            streaks[i] = (streaks[i - 1] > 0.0f) ? streaks[i - 1] + 1.0f : 1.0f;
        } else if (chg < 0.0f) {
            streaks[i] = (streaks[i - 1] < 0.0f) ? streaks[i - 1] - 1.0f : -1.0f;
        } else {
            streaks[i] = 0.0f;  // No change = reset streak
        }
    }

    // -- Step 4: RSI(rsi_period) on closes ------------------------------
    std::vector<float> rsi_vals(N);
    compute_rsi_on_data(closes.data(), N, m_rsi_period, rsi_vals.data());

    // -- Step 5: StreakRSI: RSI(streak_rsi_period) on streak values -----
    std::vector<float> streak_rsi(N);
    compute_rsi_on_data(streaks.data(), N, m_streak_rsi_period, streak_rsi.data());

    // -- Step 6: PercentRank --------------------------------------------
    // For each bar i, count how many of the previous percentile_period
    // close changes are less than the current change. Divide by the
    // number of valid (non-NaN) comparisons. Multiply by 100.
    // NaN values in changes[] (e.g. changes[0]) are skipped, reducing
    // the effective window size for early bars.
    std::vector<float> pct_rank(N);
    for (int i = 0; i < N; ++i) {
        if (i < m_percentile_period || i == 0) {
            pct_rank[i] = INDICATOR_NAN;
            continue;
        }

        float current_change = changes[i];
        int   count_less     = 0;
        int   count_valid    = 0;
        for (int j = i - m_percentile_period; j < i; ++j) {
            if (std::isnan(changes[j])) continue;
            count_valid++;
            if (changes[j] < current_change) {
                count_less++;
            }
        }
        if (count_valid == 0) {
            pct_rank[i] = INDICATOR_NAN;
        } else {
            pct_rank[i] =
                (static_cast<float>(count_less) / static_cast<float>(count_valid)) * 100.0f;
        }
    }

    // -- Step 7: ConnorsRSI = (RSI + StreakRSI + PercentRank) / 3 ------
    for (int i = 0; i < N; ++i) {
        if (i < wu || std::isnan(rsi_vals[i]) || std::isnan(streak_rsi[i]) ||
            std::isnan(pct_rank[i])) {
            output[i] = INDICATOR_NAN;
        } else {
            output[i] = (rsi_vals[i] + streak_rsi[i] + pct_rank[i]) / 3.0f;
        }
    }
}
