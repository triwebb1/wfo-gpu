#include "rsi.h"
#include <algorithm>
#include <cassert>

RSI::RSI(int period, float overbought, float oversold)
    : m_period(period), m_overbought(overbought), m_oversold(oversold) {
    assert(period > 0);
    assert(overbought > oversold);
}

int RSI::num_series() const {
    return 1;
}

int RSI::warmup_bars() const {
    // Wilder smoothing warmup = 3 * period
    return 3 * m_period;
}

const char* RSI::name() const {
    return "RSI";
}

void RSI::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int   wu             = warmup_bars();
    const float period_f       = static_cast<float>(m_period);
    const float period_minus_1 = period_f - 1.0f;

    // We need at least period+1 bars to compute the first valid RSI
    // (period price changes require period+1 prices)

    float avg_gain        = 0.0f;
    float avg_loss        = 0.0f;
    bool  initialized     = false;
    int   changes_counted = 0;  // number of price changes accumulated

    for (int i = 0; i < N; ++i) {
        const int bar_idx = start + i;

        if (i == 0) {
            // First bar: no previous close to compare against
            output[i] = INDICATOR_NAN;
            continue;
        }

        // Compute gain and loss for this bar
        const float diff = bars[bar_idx].close - bars[bar_idx - 1].close;
        const float gain = (diff > 0.0f) ? diff : 0.0f;
        const float loss = (diff < 0.0f) ? -diff : 0.0f;

        if (!initialized) {
            // Accumulate for the initial average
            avg_gain += gain;
            avg_loss += loss;
            changes_counted++;

            if (changes_counted == m_period) {
                // We have enough changes to compute the first average
                avg_gain /= period_f;
                avg_loss /= period_f;
                initialized = true;

                // Compute RSI but only output if past warmup
                if (i < wu) {
                    output[i] = INDICATOR_NAN;
                } else {
                    if (avg_gain == 0.0f && avg_loss == 0.0f) {
                        output[i] = 50.0f;  // neutral when price is flat
                    } else if (avg_loss == 0.0f) {
                        output[i] = 100.0f;
                    } else {
                        const float rs = avg_gain / avg_loss;
                        output[i]      = 100.0f - 100.0f / (1.0f + rs);
                    }
                }
            } else {
                output[i] = INDICATOR_NAN;
            }
        } else {
            // Wilder smoothing update
            avg_gain = (avg_gain * period_minus_1 + gain) / period_f;
            avg_loss = (avg_loss * period_minus_1 + loss) / period_f;

            if (i < wu) {
                output[i] = INDICATOR_NAN;
            } else {
                if (avg_gain == 0.0f && avg_loss == 0.0f) {
                    output[i] = 50.0f;  // neutral when price is flat
                } else if (avg_loss == 0.0f) {
                    output[i] = 100.0f;
                } else {
                    const float rs = avg_gain / avg_loss;
                    output[i]      = 100.0f - 100.0f / (1.0f + rs);
                }
            }
        }
    }
}
