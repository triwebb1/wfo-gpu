#include "adx.h"
#include <algorithm>
#include <cassert>
#include <cmath>

ADX::ADX(int period) : m_period(period) {
    assert(period > 0);
}

int ADX::num_series() const {
    return 3;
}

int ADX::warmup_bars() const {
    // ADX warmup = 4 * period
    return 4 * m_period;
}

const char* ADX::name() const {
    return "ADX";
}

void ADX::compute(const Bar* bars, int start, int end, float* output) const {
    const int N = end - start;
    assert(N > 0);

    const int   wu             = warmup_bars();
    const float period_f       = static_cast<float>(m_period);
    const float period_minus_1 = period_f - 1.0f;

    // Series layout: ADX [0..N), +DI [N..2N), -DI [2N..3N)
    float* adx_out      = output;
    float* plus_di_out  = output + N;
    float* minus_di_out = output + 2 * N;

    // Wilder-smoothed accumulators for +DM, -DM, TR
    float smooth_plus_dm  = 0.0f;
    float smooth_minus_dm = 0.0f;
    float smooth_tr       = 0.0f;

    // For initial averaging of +DM, -DM, TR
    float sum_plus_dm  = 0.0f;
    float sum_minus_dm = 0.0f;
    float sum_tr       = 0.0f;

    // ADX is Wilder smooth of DX
    float adx_val = 0.0f;
    float sum_dx  = 0.0f;

    // Stages:
    // Stage 0: i=0 (no previous bar, no DM/TR)
    // Stage 1: i=1..period-1 (accumulating first period of DM/TR)
    // Stage 2: i=period-1 (first smoothed DM/TR/DI values)
    // Stage 3: i=period..2*period-2 (accumulating first period of DX for ADX seed)
    // Stage 4: i=2*period-1 (first ADX = mean of first period DX values)
    // Stage 5: i>=2*period (Wilder smoothing of ADX)

    bool dm_initialized  = false;
    bool adx_initialized = false;
    int  dx_count        = 0;

    for (int i = 0; i < N; ++i) {
        const int bar_idx = start + i;

        if (i == 0) {
            // First bar: no previous bar to compute DM/TR
            adx_out[i]      = INDICATOR_NAN;
            plus_di_out[i]  = INDICATOR_NAN;
            minus_di_out[i] = INDICATOR_NAN;
            continue;
        }

        // Compute +DM, -DM
        const float up_move   = bars[bar_idx].high - bars[bar_idx - 1].high;
        const float down_move = bars[bar_idx - 1].low - bars[bar_idx].low;

        float plus_dm  = 0.0f;
        float minus_dm = 0.0f;

        if (up_move > down_move && up_move > 0.0f) {
            plus_dm = up_move;
        }
        if (down_move > up_move && down_move > 0.0f) {
            minus_dm = down_move;
        }

        const float tr = compute_true_range(bars[bar_idx], bars[bar_idx - 1].close);

        if (!dm_initialized) {
            // Accumulate for initial average
            sum_plus_dm += plus_dm;
            sum_minus_dm += minus_dm;
            sum_tr += tr;

            // i goes from 1..m_period for the first period changes (need m_period changes)
            if (i == m_period) {
                // First smoothed values = simple average * period (Wilder uses sum, not mean)
                smooth_plus_dm  = sum_plus_dm;
                smooth_minus_dm = sum_minus_dm;
                smooth_tr       = sum_tr;
                dm_initialized  = true;

                // Compute first +DI, -DI
                float plus_di  = 0.0f;
                float minus_di = 0.0f;
                if (smooth_tr > 0.0f) {
                    plus_di  = (smooth_plus_dm / smooth_tr) * 100.0f;
                    minus_di = (smooth_minus_dm / smooth_tr) * 100.0f;
                }

                // Compute DX
                const float di_sum = plus_di + minus_di;
                float       dx     = 0.0f;
                if (di_sum > 0.0f) {
                    dx = (std::abs(plus_di - minus_di) / di_sum) * 100.0f;
                }
                sum_dx += dx;
                dx_count++;

                if (i < wu) {
                    adx_out[i]      = INDICATOR_NAN;
                    plus_di_out[i]  = INDICATOR_NAN;
                    minus_di_out[i] = INDICATOR_NAN;
                } else {
                    adx_out[i]      = INDICATOR_NAN;  // ADX not ready yet
                    plus_di_out[i]  = plus_di;
                    minus_di_out[i] = minus_di;
                }
            } else {
                adx_out[i]      = INDICATOR_NAN;
                plus_di_out[i]  = INDICATOR_NAN;
                minus_di_out[i] = INDICATOR_NAN;
            }
        } else {
            // Wilder smoothing of +DM, -DM, TR
            smooth_plus_dm  = smooth_plus_dm - (smooth_plus_dm / period_f) + plus_dm;
            smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period_f) + minus_dm;
            smooth_tr       = smooth_tr - (smooth_tr / period_f) + tr;

            // Compute +DI, -DI
            float plus_di  = 0.0f;
            float minus_di = 0.0f;
            if (smooth_tr > 0.0f) {
                plus_di  = (smooth_plus_dm / smooth_tr) * 100.0f;
                minus_di = (smooth_minus_dm / smooth_tr) * 100.0f;
            }

            // Compute DX
            const float di_sum = plus_di + minus_di;
            float       dx     = 0.0f;
            if (di_sum > 0.0f) {
                dx = (std::abs(plus_di - minus_di) / di_sum) * 100.0f;
            }

            if (!adx_initialized) {
                sum_dx += dx;
                dx_count++;

                if (dx_count == m_period) {
                    // First ADX = simple average of first 'period' DX values
                    adx_val         = sum_dx / period_f;
                    adx_initialized = true;
                }
            } else {
                // Wilder smoothing of ADX
                adx_val = (adx_val * period_minus_1 + dx) / period_f;
            }

            if (i < wu) {
                adx_out[i]      = INDICATOR_NAN;
                plus_di_out[i]  = INDICATOR_NAN;
                minus_di_out[i] = INDICATOR_NAN;
            } else {
                adx_out[i]      = adx_initialized ? adx_val : INDICATOR_NAN;
                plus_di_out[i]  = plus_di;
                minus_di_out[i] = minus_di;
            }
        }
    }
}
