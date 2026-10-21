#include "session_signal_grid.h"

#include "../session_signals/opening_range.h"
#include "../session_signals/intraday_structure.h"
#include "../session_signals/multiday_patterns.h"
#include "../session_signals/level_interaction.h"
#include "../session_signals/volatility_regime.h"
#include "../session_signals/gap_analysis.h"
#include "../session_signals/overnight_context.h"
#include "../session_signals/calendar.h"

#include <cassert>

// ---------------------------------------------------------------------------
// SessionSignalParams::operator==
// ---------------------------------------------------------------------------
bool SessionSignalParams::operator==(const SessionSignalParams& o) const {
    if (type != o.type) return false;
    switch (type) {
        case SessionSignalType::OPENING_RANGE:
            return atr_lookback == o.atr_lookback && or_period_minutes == o.or_period_minutes &&
                   false_breakout_threshold == o.false_breakout_threshold &&
                   ib_tight_threshold == o.ib_tight_threshold;
        case SessionSignalType::INTRADAY_STRUCTURE:
            return atr_lookback == o.atr_lookback &&
                   range_consumed_checkpoint_et == o.range_consumed_checkpoint_et &&
                   lunch_compression_threshold == o.lunch_compression_threshold;
        case SessionSignalType::MULTIDAY_PATTERNS:
            return atr_lookback == o.atr_lookback && poor_threshold == o.poor_threshold;
        case SessionSignalType::LEVEL_INTERACTION:
            return touch_ticks == o.touch_ticks && tick_size == o.tick_size &&
                   ema_period == o.ema_period;
        case SessionSignalType::VOLATILITY_REGIME:
            return atr_short_lookback == o.atr_short_lookback &&
                   atr_long_lookback == o.atr_long_lookback &&
                   expansion_threshold == o.expansion_threshold &&
                   contraction_threshold == o.contraction_threshold;
        case SessionSignalType::GAP_ANALYSIS:
            return atr_lookback == o.atr_lookback;
        case SessionSignalType::OVERNIGHT_CONTEXT:
            return atr_lookback == o.atr_lookback;
        case SessionSignalType::CALENDAR:
            return true;  // no params
    }
    return false;
}

// ---------------------------------------------------------------------------
// session_signal_num_outputs
// ---------------------------------------------------------------------------
int session_signal_num_outputs(const SessionSignalParams& p) {
    switch (p.type) {
        case SessionSignalType::OPENING_RANGE:
            return 8;
        case SessionSignalType::INTRADAY_STRUCTURE:
            return 10;
        case SessionSignalType::MULTIDAY_PATTERNS:
            return 12;
        case SessionSignalType::LEVEL_INTERACTION:
            return 7;
        case SessionSignalType::VOLATILITY_REGIME:
            return 4;
        case SessionSignalType::GAP_ANALYSIS:
            return 6;
        case SessionSignalType::OVERNIGHT_CONTEXT:
            return 7;
        case SessionSignalType::CALENDAR:
            return 4;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// create_session_signal -- factory
// ---------------------------------------------------------------------------
std::unique_ptr<SessionSignal> create_session_signal(const SessionSignalParams& p) {
    switch (p.type) {
        case SessionSignalType::OPENING_RANGE:
            return std::make_unique<OpeningRangeSignals>(p.atr_lookback, p.or_period_minutes,
                                                         p.false_breakout_threshold,
                                                         p.ib_tight_threshold);
        case SessionSignalType::INTRADAY_STRUCTURE:
            return std::make_unique<IntradayStructureSignals>(
                p.atr_lookback, p.range_consumed_checkpoint_et, p.lunch_compression_threshold);
        case SessionSignalType::MULTIDAY_PATTERNS:
            return std::make_unique<MultidayPatternSignals>(p.atr_lookback, p.poor_threshold);
        case SessionSignalType::LEVEL_INTERACTION:
            return std::make_unique<LevelInteractionSignals>(p.touch_ticks, p.tick_size,
                                                             p.ema_period);
        case SessionSignalType::VOLATILITY_REGIME:
            return std::make_unique<VolatilityRegimeSignals>(
                p.atr_short_lookback, p.atr_long_lookback, p.expansion_threshold,
                p.contraction_threshold);
        case SessionSignalType::GAP_ANALYSIS:
            return std::make_unique<GapAnalysisSignals>(p.atr_lookback);
        case SessionSignalType::OVERNIGHT_CONTEXT:
            return std::make_unique<OvernightContextSignals>(p.atr_lookback);
        case SessionSignalType::CALENDAR:
            return std::make_unique<CalendarSignals>();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// SessionSignalGrid constructor
// ---------------------------------------------------------------------------
SessionSignalGrid::SessionSignalGrid() {
    build_opening_range();
    build_intraday_structure();
    build_multiday_patterns();
    build_level_interaction();
    build_volatility_regime();
    build_gap_analysis();
    build_overnight_context();
    build_calendar();

    // Compute total signal count
    m_total_signals = 0;
    for (const auto& e : m_entries) {
        m_total_signals += session_signal_num_outputs(e);
    }
}

// ---------------------------------------------------------------------------
// OpeningRange: or_period{15,30} x false_breakout{0.25,0.5} x ib_tight{0.5,0.7}
// = 2 x 2 x 2 = 8 entries, each producing 8 signals
// ---------------------------------------------------------------------------
void SessionSignalGrid::build_opening_range() {
    constexpr int   or_periods[]          = {15, 30};
    constexpr float false_breakout_vals[] = {0.25f, 0.50f};
    constexpr float ib_tight_vals[]       = {0.5f, 0.7f};

    for (int orp : or_periods) {
        for (float fb : false_breakout_vals) {
            for (float ibt : ib_tight_vals) {
                SessionSignalParams p;
                p.type                     = SessionSignalType::OPENING_RANGE;
                p.atr_lookback             = 10;
                p.or_period_minutes        = orp;
                p.false_breakout_threshold = fb;
                p.ib_tight_threshold       = ibt;
                m_entries.push_back(p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IntradayStructure: checkpoint{36000,39600} x lunch{0.3,0.5}
// = 2 x 2 = 4 entries, each producing 10 signals
// ---------------------------------------------------------------------------
void SessionSignalGrid::build_intraday_structure() {
    constexpr int32_t checkpoints[]      = {36000, 39600};  // 10:00 ET, 11:00 ET
    constexpr float   lunch_thresholds[] = {0.3f, 0.5f};

    for (int32_t cp : checkpoints) {
        for (float lt : lunch_thresholds) {
            SessionSignalParams p;
            p.type                         = SessionSignalType::INTRADAY_STRUCTURE;
            p.atr_lookback                 = 10;
            p.range_consumed_checkpoint_et = cp;
            p.lunch_compression_threshold  = lt;
            m_entries.push_back(p);
        }
    }
}

// ---------------------------------------------------------------------------
// MultidayPatterns: poor{0.10,0.25} = 2 entries, each producing 12 signals
// ---------------------------------------------------------------------------
void SessionSignalGrid::build_multiday_patterns() {
    constexpr float poor_vals[] = {0.10f, 0.25f};

    for (float pv : poor_vals) {
        SessionSignalParams p;
        p.type           = SessionSignalType::MULTIDAY_PATTERNS;
        p.atr_lookback   = 10;
        p.poor_threshold = pv;
        m_entries.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// LevelInteraction: 1 entry (default params), 7 signals
// ---------------------------------------------------------------------------
void SessionSignalGrid::build_level_interaction() {
    SessionSignalParams p;
    p.type = SessionSignalType::LEVEL_INTERACTION;
    m_entries.push_back(p);
}

// ---------------------------------------------------------------------------
// VolatilityRegime: 1 entry (default params), 4 signals
// ---------------------------------------------------------------------------
void SessionSignalGrid::build_volatility_regime() {
    SessionSignalParams p;
    p.type = SessionSignalType::VOLATILITY_REGIME;
    m_entries.push_back(p);
}

// ---------------------------------------------------------------------------
// GapAnalysis: 1 entry (default params), 6 signals
// ---------------------------------------------------------------------------
void SessionSignalGrid::build_gap_analysis() {
    SessionSignalParams p;
    p.type = SessionSignalType::GAP_ANALYSIS;
    m_entries.push_back(p);
}

// ---------------------------------------------------------------------------
// OvernightContext: 1 entry (default params), 7 signals
// ---------------------------------------------------------------------------
void SessionSignalGrid::build_overnight_context() {
    SessionSignalParams p;
    p.type = SessionSignalType::OVERNIGHT_CONTEXT;
    m_entries.push_back(p);
}

// ---------------------------------------------------------------------------
// Calendar: 1 entry (no params), 4 signals
// ---------------------------------------------------------------------------
void SessionSignalGrid::build_calendar() {
    SessionSignalParams p;
    p.type = SessionSignalType::CALENDAR;
    m_entries.push_back(p);
}

// ---------------------------------------------------------------------------
// find -- reverse lookup
// ---------------------------------------------------------------------------
int SessionSignalGrid::find(const SessionSignalParams& p) const {
    for (int i = 0; i < static_cast<int>(m_entries.size()); ++i) {
        if (m_entries[i] == p) return i;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// append_session_signal_series
// ---------------------------------------------------------------------------
int append_session_signal_series(const MarketData& md, const SessionSignalGrid& grid,
                                 int abs_bar_start, int abs_bar_end,
                                 std::vector<std::vector<float>>& out) {
    const int day_start = md.first_session_day_in_range(abs_bar_start);
    const int day_end   = md.last_session_day_in_range(abs_bar_end) + 1;
    if (day_start < 0 || day_end <= day_start) return 0;

    const int num_days  = day_end - day_start;
    const int num_bars  = abs_bar_end - abs_bar_start;
    int       num_added = 0;
    out.reserve(out.size() + static_cast<size_t>(grid.total_signals()));

    for (int gi = 0; gi < grid.size(); ++gi) {
        auto signal = create_session_signal(grid.params(gi));
        if (!signal) continue;

        const int  ns       = signal->num_signals();
        const auto avail    = signal->signal_available_after_et_seconds();
        const bool suppress = signal->suppress_on_halt();

        std::vector<float> day_output(static_cast<size_t>(ns) * num_days);
        signal->compute(md, day_start, day_end, day_output.data());

        for (int si = 0; si < ns; ++si) {
            std::vector<float> bar_signal(static_cast<size_t>(num_bars), 0.0f);
            broadcast_session_signal_to_bars(md, &day_output[static_cast<size_t>(si) * num_days],
                                             day_start, day_end, avail[si], abs_bar_start,
                                             abs_bar_end, bar_signal.data(), suppress);
            out.push_back(std::move(bar_signal));
            ++num_added;
        }
    }
    return num_added;
}
