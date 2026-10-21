#pragma once
#include "indicator.h"

// VWAP (Volume Weighted Average Price)
//
// Computation:
//   typical_price = (high + low + close) / 3
//   cumulative_tp_vol += tp * volume
//   cumulative_vol += volume
//   VWAP = cumulative_tp_vol / cumulative_vol
//   Bands: upper = VWAP + mult * rolling_stddev, lower = VWAP - mult * rolling_stddev
//   Rolling stddev = sqrt(cum(volume * (tp - vwap)^2) / cum_vol)
//
// Session reset: cumulative accumulators reset at the start of each session.
// The session is anchored to the CME Globex session (18:00 ET to 17:00 ET
// next day), NOT the RTH session (09:30-16:00).  Session boundaries are
// detected via timestamp gaps: a gap > 55 minutes between consecutive bars
// triggers a reset.  The CME maintenance window (17:00-18:00 ET) is the only
// gap exceeding this threshold during normal trading, so the VWAP resets at
// 18:00 ET (session open) and computes through 17:00 ET (session close),
// covering both ETH and RTH.
//
// Output: 3 series -- VWAP, upper band, lower band.
//
// Warmup: 0 (resets each session).
//
// Parameters: band_multiplier.
// Note: reset_period is inherently session-based; the indicator always resets
// at session boundaries detected from the timestamp gaps.
class VWAP : public Indicator {
public:
    explicit VWAP(float band_multiplier = 2.0f);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    float band_multiplier() const { return m_band_multiplier; }

private:
    float m_band_multiplier;
};
