#include "pivot_points.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

PivotPoints::PivotPoints(PivotType type) : m_type(type) {}

int PivotPoints::num_series() const {
    return 7;  // PP, S1, S2, S3, R1, R2, R3
}

int PivotPoints::warmup_bars() const {
    // Actual validity is controlled by the have_prior_session flag in compute(),
    // which correctly detects session boundaries via timestamp gaps.  We return 0
    // here so the warmup check doesn't redundantly mask bars that already have
    // valid pivot levels from a completed prior session.
    return 0;
}

const char* PivotPoints::name() const {
    return "PivotPoints";
}

void PivotPoints::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int wu = warmup_bars();

    // Series layout: PP[0..N), S1[N..2N), S2[2N..3N), S3[3N..4N),
    //                R1[4N..5N), R2[5N..6N), R3[6N..7N)
    float* pp_out = output;
    float* s1_out = output + N;
    float* s2_out = output + 2 * N;
    float* s3_out = output + 3 * N;
    float* r1_out = output + 4 * N;
    float* r2_out = output + 5 * N;
    float* r3_out = output + 6 * N;

    // Session boundary detection via timestamp gaps.
    // CME maintenance window is 17:00-18:00 ET (60 min gap).
    // If gap between bars > 55 minutes, treat as session boundary.
    constexpr int64_t GAP_THRESHOLD_NS = 55LL * 60 * 1'000'000'000LL;  // 55 min in nanoseconds

    // Track prior session's H, L, C
    float prev_session_high  = 0.0f;
    float prev_session_low   = 0.0f;
    float prev_session_close = 0.0f;
    bool  have_prior_session = false;

    // Track current session's H, L, C
    float curr_session_high  = -std::numeric_limits<float>::infinity();
    float curr_session_low   = std::numeric_limits<float>::infinity();
    float curr_session_close = 0.0f;
    bool  in_first_session   = true;

    // Current pivot levels (computed at session boundary, applied to all bars in session)
    float pp = INDICATOR_NAN;
    float s1 = INDICATOR_NAN, s2 = INDICATOR_NAN, s3 = INDICATOR_NAN;
    float r1 = INDICATOR_NAN, r2 = INDICATOR_NAN, r3 = INDICATOR_NAN;

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

        if (new_session && !in_first_session) {
            // The session that just ended becomes the "prior" session
            prev_session_high  = curr_session_high;
            prev_session_low   = curr_session_low;
            prev_session_close = curr_session_close;
            have_prior_session = true;

            // Compute pivot levels from the prior session
            const float H     = prev_session_high;
            const float L     = prev_session_low;
            const float C     = prev_session_close;
            const float range = H - L;

            pp = (H + L + C) / 3.0f;

            switch (m_type) {
                case PivotType::CLASSIC:
                    s1 = 2.0f * pp - H;
                    r1 = 2.0f * pp - L;
                    s2 = pp - range;
                    r2 = pp + range;
                    s3 = L - 2.0f * (H - pp);
                    r3 = H + 2.0f * (pp - L);
                    break;

                case PivotType::FIBONACCI:
                    s1 = pp - 0.382f * range;
                    r1 = pp + 0.382f * range;
                    s2 = pp - 0.618f * range;
                    r2 = pp + 0.618f * range;
                    s3 = pp - 1.000f * range;
                    r3 = pp + 1.000f * range;
                    break;

                case PivotType::CAMARILLA:
                    s1 = C - 1.1f * range / 12.0f;
                    r1 = C + 1.1f * range / 12.0f;
                    s2 = C - 1.1f * range / 6.0f;
                    r2 = C + 1.1f * range / 6.0f;
                    s3 = C - 1.1f * range / 4.0f;
                    r3 = C + 1.1f * range / 4.0f;
                    break;
            }

            // Reset current session tracking
            curr_session_high  = -std::numeric_limits<float>::infinity();
            curr_session_low   = std::numeric_limits<float>::infinity();
            curr_session_close = 0.0f;
        }

        if (new_session && in_first_session) {
            in_first_session = false;
        }

        // Update current session tracking
        curr_session_high  = std::max(curr_session_high, bars[bar_idx].high);
        curr_session_low   = std::min(curr_session_low, bars[bar_idx].low);
        curr_session_close = bars[bar_idx].close;

        // Output: NaN during warmup or before we have a prior session
        if (i < wu || !have_prior_session) {
            pp_out[i] = INDICATOR_NAN;
            s1_out[i] = INDICATOR_NAN;
            s2_out[i] = INDICATOR_NAN;
            s3_out[i] = INDICATOR_NAN;
            r1_out[i] = INDICATOR_NAN;
            r2_out[i] = INDICATOR_NAN;
            r3_out[i] = INDICATOR_NAN;
        } else {
            pp_out[i] = pp;
            s1_out[i] = s1;
            s2_out[i] = s2;
            s3_out[i] = s3;
            r1_out[i] = r1;
            r2_out[i] = r2;
            r3_out[i] = r3;
        }
    }
}
