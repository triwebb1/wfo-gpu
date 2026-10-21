#include "indicator_grid.h"

#include <cstdio>
#include <algorithm>
#include <cassert>
#include <cmath>

// ---------------------------------------------------------------------------
// indicator_num_series -- central helper used by all pipeline stages
// ---------------------------------------------------------------------------

int indicator_num_series(const IndicatorParams& params) {
    switch (params.type) {
        case IndicatorType::EMA_SMA_CROSS:
            return 2;  // fast MA, slow MA
        case IndicatorType::RSI:
            return 1;
        case IndicatorType::ATR:
            return 2;  // short period, long period
        case IndicatorType::MACD:
            return 3;  // MACD line, signal, histogram
        case IndicatorType::ADX:
            return 3;  // ADX, +DI, -DI
        case IndicatorType::BOLLINGER:
            return 3;  // middle, upper, lower
        case IndicatorType::KELTNER:
            return 3;  // center, upper, lower
        case IndicatorType::STOCHASTIC:
            return 2;  // %K, %D
        case IndicatorType::CCI:
            return 1;
        case IndicatorType::ROC:
            return 1;  // raw ROC only (no smoothing in grid)
        case IndicatorType::VWAP:
            return 3;  // VWAP, upper band, lower band
        case IndicatorType::OBV:
            return 1;
        case IndicatorType::DONCHIAN:
            return 2;  // upper, lower
        case IndicatorType::PIVOT_POINTS:
            return 7;  // PP, S1, S2, S3, R1, R2, R3
        case IndicatorType::HMA:
            return 1;  // HMA value
        case IndicatorType::CONNORS_RSI:
            return 1;  // ConnorsRSI value
    }
    return 1;
}

// ---------------------------------------------------------------------------
// describe_indicator
// ---------------------------------------------------------------------------
std::string describe_indicator(const IndicatorParams& p) {
    auto ma = [](MAType t, int period) {
        return std::string(t == MAType::EMA ? "EMA(" : "SMA(") + std::to_string(period) + ")";
    };
    auto num = [](float v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(v));
        return std::string(buf);
    };
    auto i = [](int v) { return std::to_string(v); };
    switch (p.type) {
        case IndicatorType::EMA_SMA_CROSS:
            return ma(p.fast_ma_type, p.fast_period) + " x " + ma(p.slow_ma_type, p.slow_period);
        case IndicatorType::RSI:
            return "RSI(" + i(p.rsi_period) + ") " + num(p.rsi_overbought) + "/" +
                   num(p.rsi_oversold);
        case IndicatorType::ATR:
            return "ATR ratio(" + i(p.atr_short_period) + "/" + i(p.atr_long_period) + ")";
        case IndicatorType::MACD:
            return "MACD(" + i(p.macd_fast_period) + "," + i(p.macd_slow_period) + "," +
                   i(p.macd_signal_period) + ")";
        case IndicatorType::ADX:
            return "ADX(" + i(p.adx_period) + ") > " + num(p.adx_threshold);
        case IndicatorType::BOLLINGER:
            return "Bollinger(" + i(p.bollinger_period) + ", " + num(p.bollinger_multiplier) + ")";
        case IndicatorType::KELTNER:
            return "Keltner(" + ma(p.keltner_ma_type, p.keltner_ma_period) + ", ATR " +
                   i(p.keltner_atr_period) + ", " + num(p.keltner_multiplier) + ")";
        case IndicatorType::STOCHASTIC:
            return "Stochastic(" + i(p.stoch_k_period) + "," + i(p.stoch_d_period) + ") " +
                   num(p.stoch_overbought) + "/" + num(p.stoch_oversold);
        case IndicatorType::CCI:
            return "CCI(" + i(p.cci_period) + ") +/-" + num(p.cci_threshold);
        case IndicatorType::ROC:
            return "ROC(" + i(p.roc_period) + ")";
        case IndicatorType::VWAP:
            return "VWAP bands(" + num(p.vwap_band_mult) + ")";
        case IndicatorType::OBV:
            return "OBV(" + i(p.obv_smoothing) + ")";
        case IndicatorType::DONCHIAN:
            return "Donchian(" + i(p.donchian_period) + ")";
        case IndicatorType::PIVOT_POINTS:
            return "Pivot points(" + i(static_cast<int>(p.pivot_type)) + ")";
        case IndicatorType::HMA:
            return "HMA(" + i(p.hma_period) + ")";
        case IndicatorType::CONNORS_RSI:
            return "Connors RSI(" + i(p.connors_rsi_period) + "," + i(p.connors_streak_period) +
                   "," + i(p.connors_percentile_period) + ")";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// IndicatorParams::operator==
// ---------------------------------------------------------------------------

bool IndicatorParams::operator==(const IndicatorParams& o) const {
    if (type != o.type) return false;
    switch (type) {
        case IndicatorType::EMA_SMA_CROSS:
            return fast_ma_type == o.fast_ma_type && fast_period == o.fast_period &&
                   slow_ma_type == o.slow_ma_type && slow_period == o.slow_period;
        case IndicatorType::RSI:
            return rsi_period == o.rsi_period && rsi_overbought == o.rsi_overbought &&
                   rsi_oversold == o.rsi_oversold;
        case IndicatorType::ATR:
            return atr_short_period == o.atr_short_period && atr_long_period == o.atr_long_period;
        case IndicatorType::MACD:
            return macd_fast_period == o.macd_fast_period &&
                   macd_slow_period == o.macd_slow_period &&
                   macd_signal_period == o.macd_signal_period;
        case IndicatorType::ADX:
            return adx_period == o.adx_period && adx_threshold == o.adx_threshold;
        case IndicatorType::BOLLINGER:
            return bollinger_period == o.bollinger_period &&
                   bollinger_multiplier == o.bollinger_multiplier;
        case IndicatorType::KELTNER:
            return keltner_ma_type == o.keltner_ma_type &&
                   keltner_ma_period == o.keltner_ma_period &&
                   keltner_atr_period == o.keltner_atr_period &&
                   keltner_multiplier == o.keltner_multiplier;
        case IndicatorType::STOCHASTIC:
            return stoch_k_period == o.stoch_k_period && stoch_d_period == o.stoch_d_period &&
                   stoch_overbought == o.stoch_overbought && stoch_oversold == o.stoch_oversold;
        case IndicatorType::CCI:
            return cci_period == o.cci_period && cci_threshold == o.cci_threshold;
        case IndicatorType::ROC:
            return roc_period == o.roc_period;
        case IndicatorType::VWAP:
            return vwap_band_mult == o.vwap_band_mult;
        case IndicatorType::OBV:
            return obv_smoothing == o.obv_smoothing;
        case IndicatorType::DONCHIAN:
            return donchian_period == o.donchian_period;
        case IndicatorType::PIVOT_POINTS:
            return pivot_type == o.pivot_type;
        case IndicatorType::HMA:
            return hma_period == o.hma_period;
        case IndicatorType::CONNORS_RSI:
            return connors_rsi_period == o.connors_rsi_period &&
                   connors_streak_period == o.connors_streak_period &&
                   connors_percentile_period == o.connors_percentile_period &&
                   connors_overbought == o.connors_overbought &&
                   connors_oversold == o.connors_oversold;
    }
    return false;
}

// ---------------------------------------------------------------------------
// IndicatorGrid constructor
// ---------------------------------------------------------------------------

IndicatorGrid::IndicatorGrid() {
    m_ema_start = 0;
    build_ema_sma_cross();

    m_rsi_start = static_cast<int>(m_entries.size());
    build_rsi();

    m_atr_start = static_cast<int>(m_entries.size());
    build_atr();

    m_macd_start = static_cast<int>(m_entries.size());
    build_macd();

    m_adx_start = static_cast<int>(m_entries.size());
    build_adx();

    m_bollinger_start = static_cast<int>(m_entries.size());
    build_bollinger();

    m_keltner_start = static_cast<int>(m_entries.size());
    build_keltner();

    m_stochastic_start = static_cast<int>(m_entries.size());
    build_stochastic();

    m_cci_start = static_cast<int>(m_entries.size());
    build_cci();

    m_roc_start = static_cast<int>(m_entries.size());
    build_roc();

    m_vwap_start = static_cast<int>(m_entries.size());
    build_vwap();

    m_obv_start = static_cast<int>(m_entries.size());
    build_obv();

    m_donchian_start = static_cast<int>(m_entries.size());
    build_donchian();

    m_pivot_start = static_cast<int>(m_entries.size());
    build_pivot_points();

    m_hma_start = static_cast<int>(m_entries.size());
    build_hma();

    m_connors_rsi_start = static_cast<int>(m_entries.size());
    build_connors_rsi();

    // Paranoia guard: verify no zero-period entries slipped through.
    // Current grids never produce period==0, but this catches future mistakes.
    for (const auto& p : m_entries) {
        switch (p.type) {
            case IndicatorType::EMA_SMA_CROSS:
                assert(p.fast_period > 0 && p.slow_period > 0);
                break;
            case IndicatorType::RSI:
                assert(p.rsi_period > 0);
                break;
            case IndicatorType::ATR:
                assert(p.atr_short_period > 0 && p.atr_long_period > 0);
                break;
            case IndicatorType::MACD:
                assert(p.macd_fast_period > 0 && p.macd_slow_period > 0 &&
                       p.macd_signal_period > 0);
                break;
            case IndicatorType::ADX:
                assert(p.adx_period > 0);
                break;
            case IndicatorType::BOLLINGER:
                assert(p.bollinger_period > 0);
                break;
            case IndicatorType::KELTNER:
                assert(p.keltner_ma_period > 0 && p.keltner_atr_period > 0);
                break;
            case IndicatorType::STOCHASTIC:
                assert(p.stoch_k_period > 0 && p.stoch_d_period > 0);
                break;
            case IndicatorType::CCI:
                assert(p.cci_period > 0);
                break;
            case IndicatorType::ROC:
                assert(p.roc_period > 0);
                break;
            case IndicatorType::VWAP:
                break;  // VWAP has no period parameter
            case IndicatorType::OBV:
                assert(p.obv_smoothing > 0);
                break;
            case IndicatorType::DONCHIAN:
                assert(p.donchian_period > 0);
                break;
            case IndicatorType::PIVOT_POINTS:
                break;  // Pivot Points have no period parameter
            case IndicatorType::HMA:
                assert(p.hma_period >= 2);
                break;
            case IndicatorType::CONNORS_RSI:
                assert(p.connors_rsi_period > 0 && p.connors_streak_period > 0 &&
                       p.connors_percentile_period > 0);
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// EMA/SMA Crossover: ~136 combos
// Fast type: {EMA, SMA}, fast period: {3, 5, 8, 13, 21, 33}
// Slow type: {EMA, SMA}, slow period: {20, 34, 50, 100, 200, 233}
// Prune: fast_period >= slow_period (nonsensical -- fast must be faster)
// ---------------------------------------------------------------------------
void IndicatorGrid::build_ema_sma_cross() {
    constexpr MAType fast_types[]   = {MAType::EMA, MAType::SMA};
    constexpr int    fast_periods[] = {3, 5, 8, 13, 21, 33};
    constexpr MAType slow_types[]   = {MAType::EMA, MAType::SMA};
    constexpr int    slow_periods[] = {20, 34, 50, 100, 200, 233};

    for (MAType ft : fast_types) {
        for (int fp : fast_periods) {
            for (MAType st : slow_types) {
                for (int sp : slow_periods) {
                    // Prune: fast period must be strictly less than slow period
                    if (fp >= sp) continue;

                    IndicatorParams p;
                    p.type         = IndicatorType::EMA_SMA_CROSS;
                    p.fast_ma_type = ft;
                    p.fast_period  = fp;
                    p.slow_ma_type = st;
                    p.slow_period  = sp;
                    m_entries.push_back(p);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RSI: 63 combos
// Period: {2, 3, 5, 7, 9, 14, 21}
// Overbought: {70, 75, 80}
// Oversold: {20, 25, 30}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_rsi() {
    constexpr int   periods[]    = {2, 3, 5, 7, 9, 14, 21};
    constexpr float overbought[] = {70.0f, 75.0f, 80.0f};
    constexpr float oversold[]   = {20.0f, 25.0f, 30.0f};

    for (int per : periods) {
        for (float ob : overbought) {
            for (float os : oversold) {
                IndicatorParams p;
                p.type           = IndicatorType::RSI;
                p.rsi_period     = per;
                p.rsi_overbought = ob;
                p.rsi_oversold   = os;
                m_entries.push_back(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ATR: ~16 combos
// Short period: {5, 7, 10, 14}
// Long period: {20, 34, 50, 100}
// Prune: short_period >= long_period
// ---------------------------------------------------------------------------
void IndicatorGrid::build_atr() {
    constexpr int short_periods[] = {5, 7, 10, 14};
    constexpr int long_periods[]  = {20, 34, 50, 100};

    for (int sp : short_periods) {
        for (int lp : long_periods) {
            // Prune: short must be strictly less than long
            if (sp >= lp) continue;

            IndicatorParams p;
            p.type             = IndicatorType::ATR;
            p.atr_short_period = sp;
            p.atr_long_period  = lp;
            m_entries.push_back(p);
        }
    }
}

// ---------------------------------------------------------------------------
// MACD: 27 combos
// fast {8,12,16} x slow {20,26,34} x signal {5,9,12}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_macd() {
    constexpr int fast_periods[]   = {8, 12, 16};
    constexpr int slow_periods[]   = {20, 26, 34};
    constexpr int signal_periods[] = {5, 9, 12};

    for (int fp : fast_periods) {
        for (int sp : slow_periods) {
            // Prune: fast period must be strictly less than slow period
            if (fp >= sp) continue;

            for (int sig : signal_periods) {
                IndicatorParams p;
                p.type               = IndicatorType::MACD;
                p.macd_fast_period   = fp;
                p.macd_slow_period   = sp;
                p.macd_signal_period = sig;
                m_entries.push_back(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ADX: 20 combos
// period {7, 14, 21, 28} x threshold {15, 20, 25, 30, 35}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_adx() {
    constexpr int   periods[]    = {7, 14, 21, 28};
    constexpr float thresholds[] = {15.0f, 20.0f, 25.0f, 30.0f, 35.0f};

    for (int per : periods) {
        for (float thr : thresholds) {
            IndicatorParams p;
            p.type          = IndicatorType::ADX;
            p.adx_period    = per;
            p.adx_threshold = thr;
            m_entries.push_back(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Bollinger Bands: 16 combos
// period {10, 20, 34, 50} x mult {1.5, 2.0, 2.5, 3.0}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_bollinger() {
    constexpr int   periods[] = {10, 20, 34, 50};
    constexpr float mults[]   = {1.5f, 2.0f, 2.5f, 3.0f};

    for (int per : periods) {
        for (float m : mults) {
            IndicatorParams p;
            p.type                 = IndicatorType::BOLLINGER;
            p.bollinger_period     = per;
            p.bollinger_multiplier = m;
            m_entries.push_back(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Keltner Channels: 64 combos
// ma_type {EMA} x ma_period {10,20,34,50} x atr_period {7,10,14,20}
//   x mult {1.5, 2.0, 2.5, 3.0}
// = 1 x 4 x 4 x 4 = 64
// ---------------------------------------------------------------------------
void IndicatorGrid::build_keltner() {
    constexpr MAType ma_types[]    = {MAType::EMA};
    constexpr int    ma_periods[]  = {10, 20, 34, 50};
    constexpr int    atr_periods[] = {7, 10, 14, 20};
    constexpr float  mults[]       = {1.5f, 2.0f, 2.5f, 3.0f};

    for (MAType mt : ma_types) {
        for (int mp : ma_periods) {
            for (int ap : atr_periods) {
                for (float m : mults) {
                    IndicatorParams p;
                    p.type               = IndicatorType::KELTNER;
                    p.keltner_ma_type    = mt;
                    p.keltner_ma_period  = mp;
                    p.keltner_atr_period = ap;
                    p.keltner_multiplier = m;
                    m_entries.push_back(p);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stochastic: 72 combos
// k {5, 9, 14, 21} x d {3, 5} x OB {75, 80, 85} x OS {15, 20, 25}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_stochastic() {
    constexpr int   k_periods[]  = {5, 9, 14, 21};
    constexpr int   d_periods[]  = {3, 5};
    constexpr float overbought[] = {75.0f, 80.0f, 85.0f};
    constexpr float oversold[]   = {15.0f, 20.0f, 25.0f};

    for (int kp : k_periods) {
        for (int dp : d_periods) {
            for (float ob : overbought) {
                for (float os : oversold) {
                    // Guard: skip degenerate zero-period combos
                    if (kp == 0 || dp == 0) continue;

                    IndicatorParams p;
                    p.type             = IndicatorType::STOCHASTIC;
                    p.stoch_k_period   = kp;
                    p.stoch_d_period   = dp;
                    p.stoch_overbought = ob;
                    p.stoch_oversold   = os;
                    m_entries.push_back(p);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CCI: 16 combos
// period {10, 14, 20, 34} x threshold {80, 100, 150, 200}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_cci() {
    constexpr int   periods[]    = {10, 14, 20, 34};
    constexpr float thresholds[] = {80.0f, 100.0f, 150.0f, 200.0f};

    for (int per : periods) {
        for (float thr : thresholds) {
            IndicatorParams p;
            p.type          = IndicatorType::CCI;
            p.cci_period    = per;
            p.cci_threshold = thr;
            m_entries.push_back(p);
        }
    }
}

// ---------------------------------------------------------------------------
// ROC: 5 combos
// period {5, 10, 14, 20, 50}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_roc() {
    constexpr int periods[] = {5, 10, 14, 20, 50};

    for (int per : periods) {
        IndicatorParams p;
        p.type       = IndicatorType::ROC;
        p.roc_period = per;
        m_entries.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// VWAP: 3 combos
// band_mult {1.0, 1.5, 2.0}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_vwap() {
    constexpr float mults[] = {1.0f, 1.5f, 2.0f};

    for (float m : mults) {
        IndicatorParams p;
        p.type           = IndicatorType::VWAP;
        p.vwap_band_mult = m;
        m_entries.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// OBV: 3 combos
// smoothing {10, 20, 50}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_obv() {
    constexpr int smoothings[] = {10, 20, 50};

    for (int s : smoothings) {
        IndicatorParams p;
        p.type          = IndicatorType::OBV;
        p.obv_smoothing = s;
        m_entries.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// Donchian Channel: 5 combos
// period {10, 20, 34, 50, 100}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_donchian() {
    constexpr int periods[] = {10, 20, 34, 50, 100};

    for (int per : periods) {
        IndicatorParams p;
        p.type            = IndicatorType::DONCHIAN;
        p.donchian_period = per;
        m_entries.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// Pivot Points: 3 combos
// type {0=Classic, 1=Fibonacci, 2=Camarilla}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_pivot_points() {
    constexpr PivotType types[] = {PivotType::CLASSIC, PivotType::FIBONACCI, PivotType::CAMARILLA};

    for (PivotType pt : types) {
        IndicatorParams p;
        p.type       = IndicatorType::PIVOT_POINTS;
        p.pivot_type = pt;
        m_entries.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// HMA (Hull Moving Average): 6 combos
// period {5, 9, 14, 21, 34, 50}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_hma() {
    constexpr int periods[] = {5, 9, 14, 21, 34, 50};

    for (int per : periods) {
        IndicatorParams p;
        p.type       = IndicatorType::HMA;
        p.hma_period = per;
        m_entries.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// Connors RSI: 8 combos
// rsi {2,3} x streak {2,3} x percentile {50,100}
// ---------------------------------------------------------------------------
void IndicatorGrid::build_connors_rsi() {
    constexpr int   rsi_periods[]        = {2, 3};
    constexpr int   streak_periods[]     = {2, 3};
    constexpr int   percentile_periods[] = {50, 100};
    constexpr float ob_thresholds[]      = {90.0f, 95.0f};
    constexpr float os_thresholds[]      = {5.0f, 10.0f};

    for (int rp : rsi_periods) {
        for (int sp : streak_periods) {
            for (int pp : percentile_periods) {
                for (float ob : ob_thresholds) {
                    for (float os : os_thresholds) {
                        IndicatorParams p;
                        p.type                      = IndicatorType::CONNORS_RSI;
                        p.connors_rsi_period        = rp;
                        p.connors_streak_period     = sp;
                        p.connors_percentile_period = pp;
                        p.connors_overbought        = ob;
                        p.connors_oversold          = os;
                        m_entries.push_back(p);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// find -- reverse lookup
// ---------------------------------------------------------------------------

int IndicatorGrid::find(const IndicatorParams& p) const {
    for (int i = 0; i < static_cast<int>(m_entries.size()); ++i) {
        if (m_entries[i] == p) return i;
    }
    return -1;
}
