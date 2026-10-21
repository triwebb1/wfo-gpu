#include "obv.h"
#include <cassert>
#include <cmath>
#include <vector>

OBV::OBV(int smoothing_ma_period) : m_smoothing_period(smoothing_ma_period) {
    assert(smoothing_ma_period >= 0);
}

int OBV::num_series() const {
    return 1;
}

int OBV::warmup_bars() const {
    // Raw OBV has no warmup. Smoothed OBV needs (smoothing_period - 1) bars
    // because the SMA can produce its first valid value once it has accumulated
    // exactly smoothing_period data points (indices 0..smoothing_period-1).
    return (m_smoothing_period > 0) ? (m_smoothing_period - 1) : 0;
}

const char* OBV::name() const {
    return "OBV";
}

void OBV::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = warmup_bars();

    // Use double for OBV accumulation to avoid float precision issues
    // with large cumulative volume values. Cast to float at output time.
    double obv = 0.0;

    if (m_smoothing_period <= 0) {
        // Raw OBV -- no warmup needed
        for (int i = 0; i < N; ++i) {
            const int bar_idx = start + i;

            if (i == 0) {
                obv = 0.0;
            } else {
                const float curr_close = bars[bar_idx].close;
                const float prev_close = bars[bar_idx - 1].close;

                if (curr_close > prev_close) {
                    obv += static_cast<double>(bars[bar_idx].volume);
                } else if (curr_close < prev_close) {
                    obv -= static_cast<double>(bars[bar_idx].volume);
                }
                // Equal: OBV unchanged
            }

            output[i] = static_cast<float>(obv);
        }
    } else {
        // Smoothed OBV: compute raw OBV, then SMA over smoothing_period.
        // Use a circular buffer to avoid overwriting raw values needed
        // by the running sum.
        const float        sp_f = static_cast<float>(m_smoothing_period);
        std::vector<float> circ(m_smoothing_period, 0.0f);
        int                circ_idx    = 0;
        float              running_sum = 0.0f;
        int                count       = 0;

        for (int i = 0; i < N; ++i) {
            const int bar_idx = start + i;

            if (i == 0) {
                obv = 0.0;
            } else {
                const float curr_close = bars[bar_idx].close;
                const float prev_close = bars[bar_idx - 1].close;
                if (curr_close > prev_close) {
                    obv += static_cast<double>(bars[bar_idx].volume);
                } else if (curr_close < prev_close) {
                    obv -= static_cast<double>(bars[bar_idx].volume);
                }
            }

            const float obv_f = static_cast<float>(obv);

            // Running sum for SMA using circular buffer
            running_sum += obv_f;
            count++;

            if (count > m_smoothing_period) {
                running_sum -= circ[circ_idx];
            }

            // Store in circular buffer
            circ[circ_idx] = obv_f;
            circ_idx       = (circ_idx + 1) % m_smoothing_period;

            if (i < wu || count < m_smoothing_period) {
                output[i] = INDICATOR_NAN;
            } else {
                output[i] = running_sum / sp_f;
            }
        }
    }
}
