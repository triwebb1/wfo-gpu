#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include "market_data.h"
#include "cost_model.h"
#include "metrics.h"

// ---------------------------------------------------------------------------
// StrategyConfig -- defines how indicator signals are combined and acted upon
//
// The backtest engine receives pre-computed indicator value arrays (float*)
// and uses this config to interpret them as trading signals.
// ---------------------------------------------------------------------------
// Period of the engine-computed ATR used by level exits (see
// StrategyConfig::atr_series_index). The GPU kernel uses the same constant.
inline constexpr int EXIT_ATR_PERIOD = 14;

struct StrategyConfig {
    // Which indicator series to use (indices into the indicator_values array)
    int indicator_indices[4] = {0, 0, 0, 0};
    int num_indicators       = 0;

    // Indicator roles: SIGNAL contributes to composite direction, FILTER gates entry
    enum IndicatorRole : uint8_t {
        SIGNAL = 0,  // Contributes to directional composite
        FILTER = 1   // Must be non-zero to allow entry, does not affect direction
    };
    IndicatorRole indicator_roles[4] = {SIGNAL, SIGNAL, SIGNAL, SIGNAL};

    // Signal weights for weighted-sum aggregation
    float weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // Aggregation mode
    enum AggMode : uint8_t { WEIGHTED_SUM = 0, MAJORITY_VOTE = 1, UNANIMOUS = 2 };
    AggMode aggregation = WEIGHTED_SUM;

    // Entry threshold: composite > threshold -> long, < -threshold -> short
    float entry_threshold = 0.5f;

    // Trading hours filter
    enum TradingHours : uint8_t {
        ETH       = 0,  // all session hours
        RTH       = 1,  // core hours (09:30-16:00 ET for US futures)
        MORNING   = 2,  // core open to core split (09:30-12:00 ET)
        AFTERNOON = 3   // core split to core close (12:00-16:00 ET)
    };
    TradingHours trading_hours = ETH;

    // Exit mode
    enum ExitMode : uint8_t {
        SIGNAL_REVERSAL = 0,  // exit when composite crosses zero / reverses sign
        BAR_TIMEOUT     = 1,  // exit after N bars
        TRAILING_STOP   = 2,  // exit when price retraces by X * ATR from best
        OPPOSING_SIGNAL = 3,  // exit when composite exceeds threshold in opposite direction
        MAX_LOSS_ATR    = 4,  // exit when unrealized loss exceeds X * ATR from entry
        TAKE_PROFIT_ATR = 5,  // exit at X*ATR profit from entry
        TP_SL_ATR       = 6   // combined: TP at T*ATR, SL at S*ATR from entry
    };
    ExitMode exit_mode         = SIGNAL_REVERSAL;
    int      exit_timeout_bars = 20;    // for BAR_TIMEOUT
    float    trailing_atr_mult = 2.0f;  // for TRAILING_STOP (multiplied by current ATR)
    float    max_loss_atr_mult = 3.0f;  // for MAX_LOSS_ATR (max loss in ATR multiples)
    float    tp_atr_mult       = 2.0f;  // for TAKE_PROFIT_ATR and TP_SL_ATR (TP level)
    float    sl_atr_mult       = 1.0f;  // for TP_SL_ATR (SL level)

    // ATR source for the level exits (TRAILING_STOP, MAX_LOSS_ATR,
    // TAKE_PROFIT_ATR, TP_SL_ATR). -1 (the default) uses the engine's own
    // Wilder ATR(EXIT_ATR_PERIOD) computed from completed bars of the
    // evaluated range; >= 0 names a series in indicator_values whose value at
    // bar i is used on bar i.
    int atr_series_index = -1;
};

// ---------------------------------------------------------------------------
// FillType -- discriminator for how a Trade was closed.
//
// Downstream consumers (PnL decomposition in particular) need to distinguish
// a same-bar LEVEL_EXIT (e.g., stop triggered intra-bar at the stop price)
// from a same-bar SESSION_CLOSE (flat-at-close) to allocate gap vs body PnL
// correctly.  Default initialization is NORMAL; trade-recording paths
// stamp a different type explicitly where needed.
// ---------------------------------------------------------------------------
enum class FillType : uint8_t {
    NORMAL        = 0,  // standard signal-driven exit at next-bar open
    LEVEL_EXIT    = 1,  // stop/TP/trailing triggered intra-bar at the level
    SESSION_CLOSE = 2,  // flat-at-close (ETH/RTH session boundary)
    FORCED_CLOSE  = 3   // end-of-range forced exit
};

// ---------------------------------------------------------------------------
// Trade -- one completed round-trip
//
// Populated by the backtest engine during evaluation.
// ---------------------------------------------------------------------------
struct Trade {
    int      entry_bar      = 0;  // bar index of entry fill
    int      exit_bar       = 0;  // bar index of exit fill
    float    entry_price    = 0.0f;
    float    exit_price     = 0.0f;
    int      direction      = 0;     // +1 long, -1 short
    float    gross_pnl      = 0.0f;  // before costs (dollars)
    float    net_pnl        = 0.0f;  // after costs (dollars)
    float    min_unrealized = 0.0f;  // worst M2M PnL during the trade (dollars, can be negative)
    float    max_unrealized = 0.0f;  // peak M2M PnL during the trade (dollars, can be positive)
    FillType fill_type      = FillType::NORMAL;
};

// ---------------------------------------------------------------------------
// SignalTimeGate -- describes time-availability for one indicator series.
//
// When a series is a session signal broadcast to per-bar arrays, its value
// may be invalid before a certain time-of-day within the session (e.g., the
// OR direction signal is meaningless before 10:00 ET because the OR period
// hasn't completed yet).
//
// If available_after_et_sec == 0, no gating is applied (series is always
// valid -- this is the default for technical indicators like EMA/RSI).
// If available_after_et_sec > 0, the series value at bar offset j is treated
// as 0.0 (neutral) whenever the bar's time_of_day_sec < available_after_et_sec.
// ---------------------------------------------------------------------------
struct SignalTimeGate {
    int32_t available_after_et_sec = 0;  // 0 = always available
    // Local time the session opens; availability is measured as elapsed time
    // since the open, so bars late in the previous calendar day (an evening
    // open) are correctly "before" a daytime availability time.
    int32_t session_open_sec = 18 * 3600;
};

// ---------------------------------------------------------------------------
// EarlyTermination -- optional early abort for screen-stage optimization
//
// When enabled, checks at a configurable percentage of bars whether the
// strategy is worth continuing. If running PnL is negative or trade count
// is too low at the checkpoint, the backtest aborts early. The resulting
// poor metrics will fail screen sanity floors, which is the desired outcome.
// ---------------------------------------------------------------------------
struct EarlyTermination {
    bool  enabled             = false;
    float check_at_pct        = 0.6f;  // check after 60% of bars evaluated
    int   min_trades_at_check = 10;    // need at least this many trades by checkpoint
};

// ---------------------------------------------------------------------------
// BacktestEngine -- stateless evaluation
//
// THREAD SAFETY:
//   Each thread MUST use its own BacktestEngine instance. The engine stores
//   trade results in a mutable member (m_trades) during run(), so sharing
//   a single instance across threads would race. All call sites create
//   local instances.
//
// Backtesting rules (see docs/METHODOLOGY.md):
//   1. Signal on bar N -> entry at bar N + entry_lag open.  NEVER same-bar.
//   2. Deduct round_trip_cost() on every trade close.
//   3. Track min_unrealized (worst M2M PnL) per trade for TRR computation.
//   4. Only one position at a time.  Must exit before reversing.
//
// The engine works with float* indicator arrays, NOT indicator objects.
// This decouples it from the indicator computation layer.
// ---------------------------------------------------------------------------
class BacktestEngine {
public:
    // Run a single backtest over the given bar range.
    //
    // Parameters:
    //   bars                 - the full bar array (mmap'd or loaded)
    //   bar_start            - first bar index of the evaluation range
    //   bar_end              - one past the last bar index (exclusive)
    //   indicator_values     - array of float* pointers, one per indicator series
    //                          Each float* has (bar_end - bar_start) values,
    //                          aligned with the evaluation bar range.
    //                          indicator_values[i][j] = value of series i at bar (bar_start + j)
    //   num_indicator_series - number of entries in indicator_values
    //   config               - strategy configuration
    //   cost                 - cost model
    //   entry_lag            - 1 (standard) or 2 (lag-2 validation)
    //
    // Returns a BacktestResult with all required metrics.
    BacktestResult run(const Bar* bars, int bar_start, int bar_end,
                       const float* const* indicator_values, int num_indicator_series,
                       const StrategyConfig& config, const CostModel& cost, int entry_lag = 1,
                       const EarlyTermination& early_term = EarlyTermination{}) const;

    // Run a backtest with time-gated session signals.
    //
    // Same as run() but additionally accepts:
    //   sessions   - SessionInfo array aligned with bars[] (same indexing)
    //   time_gates - per-series availability times (size == num_indicator_series).
    //                If time_gates[i].available_after_et_sec > 0, indicator_values[i][j]
    //                is replaced with 0.0 whenever sessions[bar_start+j].time_of_day_sec
    //                < time_gates[i].available_after_et_sec.
    //
    // This prevents intra-session lookahead: a bar at 09:35 ET will not see
    // a signal that depends on data through 10:00 ET (e.g., OR direction).
    BacktestResult run(const Bar* bars, const SessionInfo* sessions, int bar_start, int bar_end,
                       const float* const* indicator_values, int num_indicator_series,
                       const SignalTimeGate* time_gates, const StrategyConfig& config,
                       const CostModel& cost, int entry_lag = 1,
                       const EarlyTermination& early_term = EarlyTermination{}) const;

    // Access the trade log from the most recent run() call.
    // Useful for PnL decomposition, equity curve construction, etc.
    const std::vector<Trade>& last_trades() const { return m_trades; }

private:
    // Compute composite signal from indicator values at a given bar offset.
    // bar_offset is relative to bar_start (0-based into indicator_values arrays).
    // When time_gates is non-null, applies time-gating: series values are treated
    // as 0.0 (neutral) when bar_time_of_day_sec < the gate threshold.
    float compute_composite_signal(const float* const* indicator_values, int num_series,
                                   int bar_offset, const StrategyConfig& config,
                                   const SignalTimeGate* time_gates          = nullptr,
                                   int32_t               bar_time_of_day_sec = 0) const;

    // Core backtest loop implementation shared by both run() overloads.
    BacktestResult run_impl(const Bar*         bars,
                            const SessionInfo* sessions,  // may be nullptr (no gating)
                            int bar_start, int bar_end, const float* const* indicator_values,
                            int                   num_indicator_series,
                            const SignalTimeGate* time_gates,  // may be nullptr (no gating)
                            const StrategyConfig& config, const CostModel& cost, int entry_lag,
                            const EarlyTermination& early_term) const;

    // Mutable trade log -- populated during run().
    // Making this mutable allows run() to be const-qualified while still
    // producing a trade log for post-analysis (PnL decomposition, etc.).
    mutable std::vector<Trade> m_trades;
};
