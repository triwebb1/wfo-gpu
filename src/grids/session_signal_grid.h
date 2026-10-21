#pragma once
// ---------------------------------------------------------------------------
// SessionSignalGrid -- deterministic enumeration of all session signal
// parameter combos.
//
// Analogous to IndicatorGrid but for session-structure signals. Each entry
// specifies a session signal type + its constructor parameters. A factory
// function creates the concrete SessionSignal instance from the params.
//
// Grid contents (~19 entries producing ~156 signal series):
//   OpeningRange:        2 x 2 x 2 = 8 combos  (8 signals each)
//   IntradayStructure:   2 x 2     = 4 combos  (10 signals each)
//   MultidayPatterns:    2          = 2 combos  (12 signals each)
//   LevelInteraction:    1                      (7 signals)
//   VolatilityRegime:    1                      (4 signals)
//   GapAnalysis:         1                      (6 signals)
//   OvernightContext:     1                      (7 signals)
//   Calendar:            1                      (4 signals)
// ---------------------------------------------------------------------------

#include <cstdint>
#include <memory>
#include <vector>

#include "../session_signals/session_signal_base.h"

// ---------------------------------------------------------------------------
// Session signal type enumeration -- all 8 session signal modules
// ---------------------------------------------------------------------------
enum class SessionSignalType : uint8_t {
    OPENING_RANGE      = 0,
    INTRADAY_STRUCTURE = 1,
    MULTIDAY_PATTERNS  = 2,
    LEVEL_INTERACTION  = 3,
    VOLATILITY_REGIME  = 4,
    GAP_ANALYSIS       = 5,
    OVERNIGHT_CONTEXT  = 6,
    CALENDAR           = 7
};

// ---------------------------------------------------------------------------
// SessionSignalParams -- flat struct holding all possible session signal
// parameters. Only fields relevant to the signal_type are meaningful.
// ---------------------------------------------------------------------------
struct SessionSignalParams {
    SessionSignalType type = SessionSignalType::OPENING_RANGE;

    // Common
    int atr_lookback = 10;

    // OpeningRange
    int   or_period_minutes        = 30;
    float false_breakout_threshold = 0.25f;
    float ib_tight_threshold       = 0.5f;

    // IntradayStructure
    int32_t range_consumed_checkpoint_et = 39600;  // 11:00 ET
    float   lunch_compression_threshold  = 0.3f;

    // MultidayPatterns
    float poor_threshold = 0.25f;

    // LevelInteraction
    float touch_ticks = 2.0f;
    float tick_size   = 0.25f;
    int   ema_period  = 20;

    // VolatilityRegime
    int   atr_short_lookback    = 5;
    int   atr_long_lookback     = 20;
    float expansion_threshold   = 1.1f;
    float contraction_threshold = 0.7f;

    // GapAnalysis -- uses atr_lookback only

    // OvernightContext -- uses atr_lookback only

    // Calendar -- no params

    bool operator==(const SessionSignalParams& o) const;
    bool operator!=(const SessionSignalParams& o) const { return !(*this == o); }
};

// ---------------------------------------------------------------------------
// Helper: return the number of output signals for a given session signal type
// ---------------------------------------------------------------------------
int session_signal_num_outputs(const SessionSignalParams& p);

// ---------------------------------------------------------------------------
// Factory: create a SessionSignal instance from params
// ---------------------------------------------------------------------------
std::unique_ptr<SessionSignal> create_session_signal(const SessionSignalParams& p);

// ---------------------------------------------------------------------------
// SessionSignalGrid -- deterministic enumeration of all session signal combos
// ---------------------------------------------------------------------------
class SessionSignalGrid {
public:
    SessionSignalGrid();

    // Total number of session signal combos in the grid
    int size() const { return static_cast<int>(m_entries.size()); }

    // Lookup: ID -> params (precondition: 0 <= id < size())
    const SessionSignalParams& params(int id) const { return m_entries[id]; }

    // Reverse lookup: params -> ID, or -1 if not found
    int find(const SessionSignalParams& p) const;

    // Total number of signal series across all entries
    int total_signals() const { return m_total_signals; }

private:
    std::vector<SessionSignalParams> m_entries;
    int                              m_total_signals = 0;

    void build_opening_range();
    void build_intraday_structure();
    void build_multiday_patterns();
    void build_level_interaction();
    void build_volatility_regime();
    void build_gap_analysis();
    void build_overnight_context();
    void build_calendar();
};

// ---------------------------------------------------------------------------
// append_session_signal_series -- compute every signal in `grid` over the
// session days that bars [abs_bar_start, abs_bar_end) of `md` start, expand
// each series to those bars with broadcast_session_signal_to_bars, and
// append the per-bar arrays to `out` in grid order. Returns the number of
// series appended (grid.total_signals(), or 0 when no session starts in the
// range).
// ---------------------------------------------------------------------------
int append_session_signal_series(const MarketData& md, const SessionSignalGrid& grid,
                                 int abs_bar_start, int abs_bar_end,
                                 std::vector<std::vector<float>>& out);
