#pragma once
#include "indicator.h"

// Keltner Channels (ATR Envelope)
//
// Computation:
//   center = EMA(close, ma_period) or SMA(close, ma_period)
//   upper = center + atr_multiplier * ATR(atr_period)
//   lower = center - atr_multiplier * ATR(atr_period)
//
// Output: 3 series -- center, upper, lower.
//
// Warmup: max(3*ma_period, 3*atr_period) for EMA center;
//         max(ma_period, 3*atr_period) for SMA center.
//
// Parameters: ma_type, ma_period, atr_period, atr_multiplier.
class KeltnerChannels : public Indicator {
public:
    KeltnerChannels(MAType ma_type, int ma_period, int atr_period, float atr_multiplier = 2.0f);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    MAType ma_type() const { return m_ma_type; }
    int    ma_period() const { return m_ma_period; }
    int    atr_period() const { return m_atr_period; }
    float  atr_multiplier() const { return m_atr_multiplier; }

private:
    MAType m_ma_type;
    int    m_ma_period;
    int    m_atr_period;
    float  m_atr_multiplier;
};
