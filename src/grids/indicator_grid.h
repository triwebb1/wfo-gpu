#pragma once
#include <cstdint>
#include <vector>
#include <optional>
#include <string>
#include "../indicators/indicator.h"
#include "../indicators/pivot_points.h"

// ---------------------------------------------------------------------------
// Indicator type enumeration -- all 16 supported indicator types
// ---------------------------------------------------------------------------
enum class IndicatorType : uint8_t {
    EMA_SMA_CROSS = 0,
    RSI           = 1,
    ATR           = 2,
    MACD          = 3,
    ADX           = 4,
    BOLLINGER     = 5,
    KELTNER       = 6,
    STOCHASTIC    = 7,
    CCI           = 8,
    ROC           = 9,
    VWAP          = 10,
    OBV           = 11,
    DONCHIAN      = 12,
    PIVOT_POINTS  = 13,
    HMA           = 14,
    CONNORS_RSI   = 15
};

// ---------------------------------------------------------------------------
// IndicatorParams -- flat struct holding all possible indicator parameters.
// Only fields relevant to the indicator_type are meaningful.
// ---------------------------------------------------------------------------
struct IndicatorParams {
    IndicatorType type = IndicatorType::EMA_SMA_CROSS;

    // EMA/SMA Crossover
    MAType fast_ma_type = MAType::EMA;
    int    fast_period  = 3;
    MAType slow_ma_type = MAType::EMA;
    int    slow_period  = 20;

    // RSI
    int   rsi_period     = 14;
    float rsi_overbought = 70.0f;
    float rsi_oversold   = 30.0f;

    // ATR
    int atr_short_period = 5;
    int atr_long_period  = 20;

    // MACD
    int macd_fast_period   = 12;
    int macd_slow_period   = 26;
    int macd_signal_period = 9;

    // ADX
    int   adx_period    = 14;
    float adx_threshold = 25.0f;

    // Bollinger Bands
    int   bollinger_period     = 20;
    float bollinger_multiplier = 2.0f;

    // Keltner Channels
    MAType keltner_ma_type    = MAType::EMA;
    int    keltner_ma_period  = 20;
    int    keltner_atr_period = 10;
    float  keltner_multiplier = 2.0f;

    // Stochastic
    int   stoch_k_period   = 14;
    int   stoch_d_period   = 3;
    float stoch_overbought = 80.0f;
    float stoch_oversold   = 20.0f;

    // CCI
    int   cci_period    = 20;
    float cci_threshold = 100.0f;

    // ROC
    int roc_period = 10;

    // VWAP
    float vwap_band_mult = 2.0f;

    // OBV
    int obv_smoothing = 20;

    // Donchian Channel
    int donchian_period = 20;

    // Pivot Points
    PivotType pivot_type = PivotType::CLASSIC;

    // HMA (Hull Moving Average)
    int hma_period = 20;

    // Connors RSI
    int   connors_rsi_period        = 3;
    int   connors_streak_period     = 2;
    int   connors_percentile_period = 100;
    float connors_overbought        = 90.0f;
    float connors_oversold          = 10.0f;

    bool operator==(const IndicatorParams& o) const;
    bool operator!=(const IndicatorParams& o) const { return !(*this == o); }
};

// ---------------------------------------------------------------------------
// Helper: return the number of output series for a given indicator type/params
// ---------------------------------------------------------------------------
int indicator_num_series(const IndicatorParams& params);

// Short human-readable description, e.g. "EMA(5) x SMA(50)" or "RSI(14) 70/30".
std::string describe_indicator(const IndicatorParams& params);

// ---------------------------------------------------------------------------
// IndicatorGrid -- deterministic enumeration of all indicator param combos
//
// Each combo gets a unique integer ID (0-based, contiguous).
// Given ID -> reconstruct params.  Given params -> find ID.
// ---------------------------------------------------------------------------
class IndicatorGrid {
public:
    IndicatorGrid();

    // Total number of indicator combos in the grid
    int size() const { return static_cast<int>(m_entries.size()); }

    // Lookup: ID -> params (precondition: 0 <= id < size())
    const IndicatorParams& params(int id) const { return m_entries[id]; }

    // Reverse lookup: params -> ID, or -1 if not found
    int find(const IndicatorParams& p) const;

    // Ranges per indicator type
    int ema_sma_cross_start() const { return m_ema_start; }
    int ema_sma_cross_end() const { return m_rsi_start; }
    int rsi_start() const { return m_rsi_start; }
    int rsi_end() const { return m_atr_start; }
    int atr_start() const { return m_atr_start; }
    int atr_end() const { return m_macd_start; }

    // Ranges for the extended indicator set
    int macd_start() const { return m_macd_start; }
    int macd_end() const { return m_adx_start; }
    int adx_start() const { return m_adx_start; }
    int adx_end() const { return m_bollinger_start; }
    int bollinger_start() const { return m_bollinger_start; }
    int bollinger_end() const { return m_keltner_start; }
    int keltner_start() const { return m_keltner_start; }
    int keltner_end() const { return m_stochastic_start; }
    int stochastic_start() const { return m_stochastic_start; }
    int stochastic_end() const { return m_cci_start; }
    int cci_start() const { return m_cci_start; }
    int cci_end() const { return m_roc_start; }
    int roc_start() const { return m_roc_start; }
    int roc_end() const { return m_vwap_start; }
    int vwap_start() const { return m_vwap_start; }
    int vwap_end() const { return m_obv_start; }
    int obv_start() const { return m_obv_start; }
    int obv_end() const { return m_donchian_start; }
    int donchian_start() const { return m_donchian_start; }
    int donchian_end() const { return m_pivot_start; }
    int pivot_start() const { return m_pivot_start; }
    int pivot_end() const { return m_hma_start; }
    int hma_start() const { return m_hma_start; }
    int hma_end() const { return m_connors_rsi_start; }
    int connors_rsi_start() const { return m_connors_rsi_start; }
    int connors_rsi_end() const { return size(); }

    // Count per type
    int ema_sma_cross_count() const { return m_rsi_start - m_ema_start; }
    int rsi_count() const { return m_atr_start - m_rsi_start; }
    int atr_count() const { return m_macd_start - m_atr_start; }
    int macd_count() const { return m_adx_start - m_macd_start; }
    int adx_count() const { return m_bollinger_start - m_adx_start; }
    int bollinger_count() const { return m_keltner_start - m_bollinger_start; }
    int keltner_count() const { return m_stochastic_start - m_keltner_start; }
    int stochastic_count() const { return m_cci_start - m_stochastic_start; }
    int cci_count() const { return m_roc_start - m_cci_start; }
    int roc_count() const { return m_vwap_start - m_roc_start; }
    int vwap_count() const { return m_obv_start - m_vwap_start; }
    int obv_count() const { return m_donchian_start - m_obv_start; }
    int donchian_count() const { return m_pivot_start - m_donchian_start; }
    int pivot_count() const { return m_hma_start - m_pivot_start; }
    int hma_count() const { return m_connors_rsi_start - m_hma_start; }
    int connors_rsi_count() const { return size() - m_connors_rsi_start; }

private:
    std::vector<IndicatorParams> m_entries;

    int m_ema_start         = 0;
    int m_rsi_start         = 0;
    int m_atr_start         = 0;
    int m_macd_start        = 0;
    int m_adx_start         = 0;
    int m_bollinger_start   = 0;
    int m_keltner_start     = 0;
    int m_stochastic_start  = 0;
    int m_cci_start         = 0;
    int m_roc_start         = 0;
    int m_vwap_start        = 0;
    int m_obv_start         = 0;
    int m_donchian_start    = 0;
    int m_pivot_start       = 0;
    int m_hma_start         = 0;
    int m_connors_rsi_start = 0;

    void build_ema_sma_cross();
    void build_rsi();
    void build_atr();
    void build_macd();
    void build_adx();
    void build_bollinger();
    void build_keltner();
    void build_stochastic();
    void build_cci();
    void build_roc();
    void build_vwap();
    void build_obv();
    void build_donchian();
    void build_pivot_points();
    void build_hma();
    void build_connors_rsi();
};
