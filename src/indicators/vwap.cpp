#include "vwap.h"
#include <cassert>
#include <cmath>

VWAP::VWAP(float band_multiplier) : m_band_multiplier(band_multiplier) {
    assert(band_multiplier > 0.0f);
}

int VWAP::num_series() const {
    return 3;
}

int VWAP::warmup_bars() const {
    // VWAP resets each session -- warmup is 0
    return 0;
}

const char* VWAP::name() const {
    return "VWAP";
}

void VWAP::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    // Series layout: VWAP [0..N), upper [N..2N), lower [2N..3N)
    float* vwap_out  = output;
    float* upper_out = output + N;
    float* lower_out = output + 2 * N;

    // Session detection via timestamp gaps.
    // CME maintenance window is 17:00-18:00 ET (60 min gap).
    // If gap between bars > 55 minutes, treat as session boundary.
    constexpr int64_t GAP_THRESHOLD_NS = 55LL * 60 * 1'000'000'000LL;  // 55 min in nanoseconds

    // Cumulative accumulators (reset at session boundary)
    double cum_tp_vol    = 0.0;
    double cum_vol       = 0.0;
    double cum_tp_vol_sq = 0.0;  // for variance: sum(vol * tp^2)

    for (int i = 0; i < N; ++i) {
        const int bar_idx = start + i;

        // Detect session boundary
        bool new_session = (i == 0);
        if (i > 0) {
            const int64_t gap = bars[bar_idx].timestamp - bars[bar_idx - 1].timestamp;
            if (gap > GAP_THRESHOLD_NS) {
                new_session = true;
            }
        }

        if (new_session) {
            cum_tp_vol    = 0.0;
            cum_vol       = 0.0;
            cum_tp_vol_sq = 0.0;
        }

        const float  tp   = (bars[bar_idx].high + bars[bar_idx].low + bars[bar_idx].close) / 3.0f;
        const double vol  = static_cast<double>(bars[bar_idx].volume);
        const double tp_d = static_cast<double>(tp);

        cum_tp_vol += tp_d * vol;
        cum_vol += vol;
        cum_tp_vol_sq += vol * tp_d * tp_d;

        if (cum_vol == 0.0) {
            vwap_out[i]  = tp;
            upper_out[i] = tp;
            lower_out[i] = tp;
        } else {
            const float vwap = static_cast<float>(cum_tp_vol / cum_vol);
            vwap_out[i]      = vwap;

            // Rolling standard deviation of VWAP
            // Variance = sum(vol * (tp - vwap)^2) / sum(vol)
            //          = sum(vol * tp^2) / sum(vol) - vwap^2
            const double vwap_d   = static_cast<double>(vwap);
            double       variance = (cum_tp_vol_sq / cum_vol) - (vwap_d * vwap_d);
            if (variance < 0.0) variance = 0.0;  // numerical safety
            const float stddev = static_cast<float>(std::sqrt(variance));

            upper_out[i] = vwap + m_band_multiplier * stddev;
            lower_out[i] = vwap - m_band_multiplier * stddev;
        }
    }
}
