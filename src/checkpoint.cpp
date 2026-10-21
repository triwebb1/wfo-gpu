#include "checkpoint.h"

#include "crc32c.h"
#include "durable_file.h"
#include "grids/strategy_grid.h"
#include "landscape.h"
#include "results_serde.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <type_traits>

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash helpers (for config_hash, bar_data_hash)
// ---------------------------------------------------------------------------
namespace {

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr uint64_t FNV_PRIME  = 0x100000001b3ULL;

// ---------------------------------------------------------------------------
// Named sanity caps for deserialization length fields.
//
// All user-supplied length prefixes are bounded before any allocation to
// guard against a corrupt or crafted checkpoint forcing an OOM. The caps
// are deliberately loose (several orders of magnitude above any realistic
// workload) -- their only job is to keep deserialization O(realistic).
// ---------------------------------------------------------------------------

// Upper bound for a single string's byte length (run_id, archetype, etc.).
constexpr uint32_t MAX_STRING_LENGTH = 10'000'000u;

// Upper bound for PerTrade record count per WFOWindowResult.
constexpr uint32_t MAX_PER_TRADE_COUNT = 10'000'000u;

// Upper bound for WFO windows in a single run.
constexpr uint32_t MAX_WINDOW_COUNT = 100'000u;

// Upper bound for equity curve sample count (float per bar) per run.
constexpr uint32_t MAX_EQUITY_SAMPLES = 100'000'000u;

// Upper bound for completed BacktestResult count in a checkpoint body.
constexpr uint32_t MAX_RESULTS_COUNT = 100'000'000u;

// Upper bound for WFORunResult count in a checkpoint body.
constexpr uint32_t MAX_RUNS_COUNT = 100'000u;

inline void fnv1a_update(uint64_t& h, const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
}

template <typename T>
inline void fnv1a_pod(uint64_t& h, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "fnv1a_pod requires POD");
    fnv1a_update(h, &v, sizeof(T));
}

void fnv1a_string(uint64_t& h, const std::string& v) {
    fnv1a_pod(h, static_cast<uint64_t>(v.size()));
    fnv1a_update(h, v.data(), v.size());
}

// Every SessionModel field changes which bars trade, where sessions break
// and which signals exist, so all of them are part of the run's identity.
void fnv1a_session(uint64_t& h, const SessionModel& m) {
    fnv1a_string(h, m.name);
    fnv1a_pod(h, static_cast<uint8_t>(m.clock));
    fnv1a_pod(h, m.session_open_sec);
    fnv1a_pod(h, m.session_close_sec);
    fnv1a_pod(h, m.flat_at_close);
    fnv1a_pod(h, m.core_open_sec);
    fnv1a_pod(h, m.core_split_sec);
    fnv1a_pod(h, m.core_close_sec);
    fnv1a_pod(h, m.detect_halts);
    fnv1a_pod(h, m.halt_gap_sec);
    fnv1a_string(h, m.early_close_calendar);
    fnv1a_pod(h, m.session_signals);
}

}  // namespace

// ---------------------------------------------------------------------------
// compute_grid_layout_fingerprint
// ---------------------------------------------------------------------------
uint64_t compute_grid_layout_fingerprint() {
    uint64_t h = FNV_OFFSET;

    auto mix_u32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<uint8_t>(v & 0xFF);
            h *= FNV_PRIME;
            v >>= 8;
        }
    };
    auto mix_f32 = [&](float v) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        mix_u32(bits);
    };

    mix_u32(static_cast<uint32_t>(StrategyGrid::AGG_MODES));
    mix_u32(static_cast<uint32_t>(StrategyGrid::ENTRY_THRESHOLDS));
    mix_u32(static_cast<uint32_t>(StrategyGrid::EXIT_MODES));
    mix_u32(static_cast<uint32_t>(StrategyGrid::TRADING_HOURS_OPTIONS));
    mix_u32(static_cast<uint32_t>(StrategyGrid::ROLE_PATTERNS));
    for (float w : StrategyGrid::WEIGHT_VALUES) mix_f32(w);
    for (float t : StrategyGrid::THRESHOLD_VALUES) mix_f32(t);

    return h;
}

// ---------------------------------------------------------------------------
// config_hash -- FNV-1a over every input that changes a stage's results.
// File paths are excluded: moving the data or the output keeps a run valid.
// ---------------------------------------------------------------------------
uint64_t compute_config_hash_screen(const Orchestrator::ScreenConfig& c) {
    uint64_t h = FNV_OFFSET;
    // Domain tag: distinguishes screen from walk-forward with identical numeric fields.
    const uint32_t tag = 0x5343524eu;  // "SCRN"
    fnv1a_pod(h, tag);

    fnv1a_pod(h, c.scan_start_bar);
    fnv1a_pod(h, c.scan_end_bar);
    fnv1a_pod(h, c.robustness_floor);
    fnv1a_pod(h, c.max_active_indicators);
    fnv1a_pod(h, c.entry_lag);
    fnv1a_pod(h, c.early_termination);
    fnv1a_pod(h, c.max_strategies);
    // CostModel and SanityFloors are 4-byte fields only, so no padding bytes.
    fnv1a_update(h, &c.cost, sizeof(c.cost));
    fnv1a_update(h, &c.sanity_floors, sizeof(c.sanity_floors));
    fnv1a_session(h, c.session);
    return h;
}

uint64_t compute_config_hash_walk_forward(const Orchestrator::WalkForwardConfig& c) {
    uint64_t       h   = FNV_OFFSET;
    const uint32_t tag = 0x57414c4bu;  // "WALK"
    fnv1a_pod(h, tag);

    fnv1a_update(h, &c.cost, sizeof(c.cost));
    fnv1a_pod(h, c.bars_per_day);
    fnv1a_pod(h, c.oos_start_bar);
    fnv1a_session(h, c.session);
    // The strategies under test; reduced_indicator_ids is derived from them.
    fnv1a_pod(h, static_cast<uint64_t>(c.reduced_strategy_ids.size()));
    for (int id : c.reduced_strategy_ids) fnv1a_pod(h, id);
    return h;
}

// ---------------------------------------------------------------------------
// Streaming write helpers (DurableFileWriter-backed)
// ---------------------------------------------------------------------------
namespace {

template <typename T>
inline void write_pod(DurableFileWriter& w, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "write_pod requires POD");
    w.write_bytes(&v, sizeof(T));
}

inline void write_bool(DurableFileWriter& w, bool v) {
    uint8_t b = v ? 1 : 0;
    w.write_bytes(&b, 1);
}

inline void write_string(DurableFileWriter& w, const std::string& s) {
    uint32_t len = static_cast<uint32_t>(s.size());
    write_pod(w, len);
    if (len > 0) w.write_bytes(s.data(), len);
}

// ---------------------------------------------------------------------------
// Streaming read helpers -- std::ifstream + running CRC32C side-channel.
// ---------------------------------------------------------------------------
struct CrcReader {
    std::istream& in;
    uint32_t      crc_state = CRC32C_INITIAL_STATE;

    explicit CrcReader(std::istream& s) : in(s) {}

    bool read(void* buf, size_t len) {
        in.read(static_cast<char*>(buf), static_cast<std::streamsize>(len));
        if (!in.good()) return false;
        crc_state = crc32c_update(crc_state, buf, len);
        return true;
    }

    uint32_t crc() const { return crc32c_finalize(crc_state); }
};

template <typename T>
inline bool read_pod(CrcReader& r, T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "read_pod requires POD");
    return r.read(&v, sizeof(T));
}

inline bool read_bool(CrcReader& r, bool& v) {
    uint8_t b = 0;
    if (!r.read(&b, 1)) return false;
    v = (b != 0);
    return true;
}

inline bool read_string(CrcReader& r, std::string& s) {
    uint32_t len = 0;
    if (!read_pod(r, len)) return false;
    if (len > MAX_STRING_LENGTH) return false;  // sanity cap
    s.resize(len);
    if (len > 0) {
        if (!r.read(s.data(), len)) return false;
    }
    return true;
}

// BacktestResult write: delegate to shared results_serde.
inline void write_backtest_result(DurableFileWriter& w, const BacktestResult& r) {
    results_serde::write_backtest_result(w, r);
}

// BacktestResult read: inline field-by-field via CrcReader (must match
// results_serde::write_backtest_result byte-for-byte).
bool read_backtest_result(CrcReader& r, BacktestResult& b) {
    if (!read_pod(r, b.start_ts)) return false;
    if (!read_pod(r, b.end_ts)) return false;
    if (!read_pod(r, b.total_trades)) return false;
    if (!read_pod(r, b.trades_per_week)) return false;
    if (!read_pod(r, b.win_rate)) return false;
    if (!read_pod(r, b.avg_win)) return false;
    if (!read_pod(r, b.avg_loss)) return false;
    if (!read_pod(r, b.avg_hold_bars)) return false;
    if (!read_pod(r, b.total_pnl)) return false;
    if (!read_pod(r, b.gross_wins)) return false;
    if (!read_pod(r, b.gross_losses)) return false;
    if (!read_pod(r, b.max_drawdown)) return false;
    if (!read_pod(r, b.max_live_drawdown)) return false;
    if (!read_pod(r, b.avg_trade_dd)) return false;
    if (!read_pod(r, b.avg_trade_peak)) return false;
    if (!read_pod(r, b.sharpe_ratio)) return false;
    if (!read_pod(r, b.calmar_ratio)) return false;
    if (!read_pod(r, b.profit_factor)) return false;
    if (!read_pod(r, b.payoff_ratio)) return false;
    if (!read_pod(r, b.green_weeks_pct)) return false;
    if (!read_pod(r, b.green_months_pct)) return false;
    if (!read_pod(r, b.max_red_weeks)) return false;
    if (!read_pod(r, b.negative_years)) return false;
    if (!read_pod(r, b.negative_months)) return false;
    if (!read_pod(r, b.annual_roc)) return false;
    if (!read_pod(r, b.reward_risk_trade)) return false;
    if (!read_pod(r, b.daily_profit_rate)) return false;
    if (!read_pod(r, b.time_to_double)) return false;
    if (!read_pod(r, b.trade_recovery_ratio)) return false;
    if (!read_pod(r, b.strategy_id)) return false;
    if (!read_bool(r, b.viable)) return false;
    if (!read_bool(r, b.metrics_complete)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// WFOWindowResult serialization (including per_trade).
// ---------------------------------------------------------------------------
void write_wfo_window(DurableFileWriter& w, const WFOWindowResult& win) {
    write_pod(w, win.window_index);
    write_pod(w, win.is_start_bar);
    write_pod(w, win.is_end_bar);
    write_pod(w, win.oos_start_bar);
    write_pod(w, win.oos_end_bar);
    write_pod(w, win.strategy_id);
    write_string(w, win.archetype);
    write_backtest_result(w, win.is_result);
    write_backtest_result(w, win.oos_lag1);
    write_backtest_result(w, win.oos_lag2);

    write_pod(w, win.pnl_decomp.total_bar_body_pnl);
    write_pod(w, win.pnl_decomp.total_gap_pnl);
    write_pod(w, win.pnl_decomp.total_gross_pnl);
    write_pod(w, win.pnl_decomp.bar_body_pct);
    write_pod(w, win.pnl_decomp.gap_pct);
    write_bool(w, win.pnl_decomp.is_real_edge);
    write_bool(w, win.pnl_decomp.is_gap_artifact);

    // Length-prefixed per_trade vector.
    uint32_t pt_len = static_cast<uint32_t>(win.pnl_decomp.per_trade.size());
    write_pod(w, pt_len);
    for (const auto& pt : win.pnl_decomp.per_trade) {
        int32_t trade_index = pt.trade_index;
        write_pod(w, trade_index);
        write_pod(w, pt.bar_body_pnl);
        write_pod(w, pt.gap_pnl);
        write_pod(w, pt.remaining_pnl);
    }

    write_bool(w, win.skipped);
}

bool read_wfo_window(CrcReader& r, WFOWindowResult& win) {
    if (!read_pod(r, win.window_index)) return false;
    if (!read_pod(r, win.is_start_bar)) return false;
    if (!read_pod(r, win.is_end_bar)) return false;
    if (!read_pod(r, win.oos_start_bar)) return false;
    if (!read_pod(r, win.oos_end_bar)) return false;
    if (!read_pod(r, win.strategy_id)) return false;
    if (!read_string(r, win.archetype)) return false;
    if (!read_backtest_result(r, win.is_result)) return false;
    if (!read_backtest_result(r, win.oos_lag1)) return false;
    if (!read_backtest_result(r, win.oos_lag2)) return false;

    if (!read_pod(r, win.pnl_decomp.total_bar_body_pnl)) return false;
    if (!read_pod(r, win.pnl_decomp.total_gap_pnl)) return false;
    if (!read_pod(r, win.pnl_decomp.total_gross_pnl)) return false;
    if (!read_pod(r, win.pnl_decomp.bar_body_pct)) return false;
    if (!read_pod(r, win.pnl_decomp.gap_pct)) return false;
    if (!read_bool(r, win.pnl_decomp.is_real_edge)) return false;
    if (!read_bool(r, win.pnl_decomp.is_gap_artifact)) return false;

    // per_trade vector.
    uint32_t pt_len = 0;
    if (!read_pod(r, pt_len)) return false;
    if (pt_len > MAX_PER_TRADE_COUNT) return false;  // sanity cap
    win.pnl_decomp.per_trade.resize(pt_len);
    for (uint32_t i = 0; i < pt_len; ++i) {
        int32_t trade_index = 0;
        if (!read_pod(r, trade_index)) return false;
        win.pnl_decomp.per_trade[i].trade_index = trade_index;
        if (!read_pod(r, win.pnl_decomp.per_trade[i].bar_body_pnl)) return false;
        if (!read_pod(r, win.pnl_decomp.per_trade[i].gap_pnl)) return false;
        if (!read_pod(r, win.pnl_decomp.per_trade[i].remaining_pnl)) return false;
    }

    if (!read_bool(r, win.skipped)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// WFORunResult serialization
// ---------------------------------------------------------------------------
void write_wfo_run(DurableFileWriter& w, const WFORunResult& run) {
    write_pod(w, run.schedule.frequency_bars);
    write_pod(w, run.schedule.lookback_bars);
    write_pod(w, run.schedule.frequency_days);
    write_pod(w, run.schedule.lookback_days);

    int trigger_int = static_cast<int>(run.schedule.trigger);
    write_pod(w, trigger_int);
    write_pod(w, run.schedule.trigger_dd_frac);
    write_pod(w, run.schedule.trigger_losers_n);

    write_string(w, run.selection_strategy);

    uint32_t num_windows = static_cast<uint32_t>(run.windows.size());
    write_pod(w, num_windows);
    for (const auto& win : run.windows) {
        write_wfo_window(w, win);
    }

    uint32_t eq_len = static_cast<uint32_t>(run.equity_curve.size());
    write_pod(w, eq_len);
    if (eq_len > 0) {
        w.write_bytes(run.equity_curve.data(), eq_len * sizeof(float));
    }

    uint32_t ts_len = static_cast<uint32_t>(run.equity_timestamps.size());
    write_pod(w, ts_len);
    if (ts_len > 0) {
        w.write_bytes(run.equity_timestamps.data(), ts_len * sizeof(int64_t));
    }

    uint32_t sd_len = static_cast<uint32_t>(run.equity_session_days.size());
    write_pod(w, sd_len);
    if (sd_len > 0) {
        w.write_bytes(run.equity_session_days.data(), sd_len * sizeof(int32_t));
    }

    write_backtest_result(w, run.oos_aggregate);

    write_pod(w, run.parameter_stability);
    write_pod(w, run.windows_skipped);
    write_bool(w, run.sufficient_windows);
}

bool read_wfo_run(CrcReader& r, WFORunResult& run) {
    if (!read_pod(r, run.schedule.frequency_bars)) return false;
    if (!read_pod(r, run.schedule.lookback_bars)) return false;
    if (!read_pod(r, run.schedule.frequency_days)) return false;
    if (!read_pod(r, run.schedule.lookback_days)) return false;

    int trigger_int = 0;
    if (!read_pod(r, trigger_int)) return false;
    if (trigger_int < 0 || trigger_int > 4) return false;
    run.schedule.trigger = static_cast<WFOTrigger>(trigger_int);
    if (!read_pod(r, run.schedule.trigger_dd_frac)) return false;
    if (!read_pod(r, run.schedule.trigger_losers_n)) return false;

    if (!read_string(r, run.selection_strategy)) return false;

    uint32_t num_windows = 0;
    if (!read_pod(r, num_windows)) return false;
    if (num_windows > MAX_WINDOW_COUNT) return false;  // sanity cap
    run.windows.resize(num_windows);
    for (uint32_t i = 0; i < num_windows; ++i) {
        if (!read_wfo_window(r, run.windows[i])) return false;
    }

    uint32_t eq_len = 0;
    if (!read_pod(r, eq_len)) return false;
    if (eq_len > MAX_EQUITY_SAMPLES) return false;  // sanity cap
    run.equity_curve.resize(eq_len);
    if (eq_len > 0 && !r.read(run.equity_curve.data(), eq_len * sizeof(float))) return false;

    uint32_t ts_len = 0;
    if (!read_pod(r, ts_len)) return false;
    if (ts_len > MAX_EQUITY_SAMPLES) return false;  // sanity cap
    run.equity_timestamps.resize(ts_len);
    if (ts_len > 0 && !r.read(run.equity_timestamps.data(), ts_len * sizeof(int64_t))) return false;

    uint32_t sd_len = 0;
    if (!read_pod(r, sd_len)) return false;
    if (sd_len > MAX_EQUITY_SAMPLES) return false;  // sanity cap
    run.equity_session_days.resize(sd_len);
    if (sd_len > 0 && !r.read(run.equity_session_days.data(), sd_len * sizeof(int32_t)))
        return false;

    if (!read_backtest_result(r, run.oos_aggregate)) return false;

    if (!read_pod(r, run.parameter_stability)) return false;
    if (!read_pod(r, run.windows_skipped)) return false;
    if (!read_bool(r, run.sufficient_windows)) return false;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Checkpoint::save -- durable atomic write via DurableFileWriter + trailing CRC.
// ---------------------------------------------------------------------------
bool Checkpoint::save(const std::string& path) const {
    try {
        DurableFileWriter::atomic_write(path, [this](DurableFileWriter& w) {
            // Header
            w.write_bytes(CHECKPOINT_MAGIC, 4);
            write_pod(w, CHECKPOINT_VERSION);
            const uint64_t fp = compute_grid_layout_fingerprint();
            write_pod(w, fp);
            write_pod(w, this->config_hash);
            write_pod(w, this->bar_data_hash);
            write_pod(w, this->scan_start_bar);
            write_pod(w, this->scan_end_bar);

            // Body
            write_string(w, this->run_id);
            write_pod(w, this->phase);
            write_pod(w, this->strategies_evaluated);
            write_pod(w, this->total_strategies);

            uint32_t num_results = static_cast<uint32_t>(this->completed_results.size());
            write_pod(w, num_results);
            for (const auto& r : this->completed_results) {
                write_backtest_result(w, r);
            }

            write_pod(w, this->current_schedule_idx);
            write_pod(w, this->current_window_idx);

            uint32_t num_runs = static_cast<uint32_t>(this->completed_runs.size());
            write_pod(w, num_runs);
            for (const auto& run : this->completed_runs) {
                write_wfo_run(w, run);
            }

            // Trailing CRC32C of every byte written so far.
            // Snapshot before writing the CRC field itself (CRC covers magic
            // through end-of-body, NOT the trailing CRC bytes).
            const uint32_t crc = w.crc32c();
            w.write_bytes(&crc, sizeof(crc));
        });
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Checkpoint::save: " << e.what() << std::endl;
        return false;
    }
}

// ---------------------------------------------------------------------------
// Checkpoint::load -- parse + validate; std::nullopt on any failure.
// ---------------------------------------------------------------------------
std::optional<Checkpoint> Checkpoint::load(const std::string& path, uint64_t expected_config_hash,
                                           uint64_t expected_bar_data_hash) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return std::nullopt;
    }

    CrcReader  r(in);
    Checkpoint cp;

    // --- Header ---
    char magic[4] = {};
    if (!r.read(magic, 4)) return std::nullopt;
    if (std::memcmp(magic, CHECKPOINT_MAGIC, 4) != 0) {
        std::cerr << "Checkpoint::load: invalid magic bytes in " << path << std::endl;
        return std::nullopt;
    }

    uint32_t version = 0;
    if (!read_pod(r, version)) return std::nullopt;
    if (version < CHECKPOINT_VERSION_MIN_COMPAT || version > CHECKPOINT_VERSION) {
        std::cerr << "Checkpoint::load: version " << version << " outside compat band ["
                  << CHECKPOINT_VERSION_MIN_COMPAT << ", " << CHECKPOINT_VERSION << "] in " << path
                  << std::endl;
        return std::nullopt;
    }

    uint64_t saved_fp = 0;
    if (!read_pod(r, saved_fp)) return std::nullopt;
    const uint64_t expected_fp = compute_grid_layout_fingerprint();
    if (saved_fp != expected_fp) {
        std::cerr << "Checkpoint::load: grid layout fingerprint mismatch in " << path
                  << " (saved=0x" << std::hex << saved_fp << " expected=0x" << expected_fp
                  << std::dec << ")" << std::endl;
        return std::nullopt;
    }
    cp.grid_layout_fingerprint = saved_fp;

    if (!read_pod(r, cp.config_hash)) return std::nullopt;
    if (!read_pod(r, cp.bar_data_hash)) return std::nullopt;
    if (!read_pod(r, cp.scan_start_bar)) return std::nullopt;
    if (!read_pod(r, cp.scan_end_bar)) return std::nullopt;

    if (expected_config_hash != 0 && cp.config_hash != expected_config_hash) {
        std::cerr << "Checkpoint::load: config_hash mismatch (saved=0x" << std::hex
                  << cp.config_hash << " expected=0x" << expected_config_hash << std::dec << ") in "
                  << path << std::endl;
        return std::nullopt;
    }
    if (expected_bar_data_hash != 0 && cp.bar_data_hash != expected_bar_data_hash) {
        std::cerr << "Checkpoint::load: bar_data_hash mismatch (saved=0x" << std::hex
                  << cp.bar_data_hash << " expected=0x" << expected_bar_data_hash << std::dec
                  << ") in " << path << std::endl;
        return std::nullopt;
    }

    // --- Body ---
    if (!read_string(r, cp.run_id)) return std::nullopt;
    if (!read_pod(r, cp.phase)) return std::nullopt;
    if (!read_pod(r, cp.strategies_evaluated)) return std::nullopt;
    if (!read_pod(r, cp.total_strategies)) return std::nullopt;

    uint32_t num_results = 0;
    if (!read_pod(r, num_results)) return std::nullopt;
    if (num_results > MAX_RESULTS_COUNT) return std::nullopt;  // sanity cap
    cp.completed_results.resize(num_results);
    for (uint32_t i = 0; i < num_results; ++i) {
        if (!read_backtest_result(r, cp.completed_results[i])) return std::nullopt;
    }

    if (!read_pod(r, cp.current_schedule_idx)) return std::nullopt;
    if (!read_pod(r, cp.current_window_idx)) return std::nullopt;

    uint32_t num_runs = 0;
    if (!read_pod(r, num_runs)) return std::nullopt;
    if (num_runs > MAX_RUNS_COUNT) return std::nullopt;  // sanity cap
    cp.completed_runs.resize(num_runs);
    for (uint32_t i = 0; i < num_runs; ++i) {
        if (!read_wfo_run(r, cp.completed_runs[i])) return std::nullopt;
    }

    // --- Trailing CRC32C ---
    // Snapshot running CRC BEFORE reading the 4-byte CRC field. The CRC
    // covers every byte up to (but not including) the trailer.
    const uint32_t computed = r.crc();
    uint32_t       stored   = 0;
    in.read(reinterpret_cast<char*>(&stored), sizeof(stored));
    if (!in.good()) {
        std::cerr << "Checkpoint::load: truncated trailing CRC in " << path << std::endl;
        return std::nullopt;
    }
    if (stored != computed) {
        std::cerr << "Checkpoint::load: CRC32C mismatch in " << path << " (stored=0x" << std::hex
                  << stored << " computed=0x" << computed << std::dec << ")" << std::endl;
        return std::nullopt;
    }

    return cp;
}

// ---------------------------------------------------------------------------
// Checkpoint::is_valid -- thin wrapper over load(path, 0, 0).has_value().
// ---------------------------------------------------------------------------
bool Checkpoint::is_valid(const std::string& path) {
    return Checkpoint::load(path, /*expected_config_hash=*/0,
                            /*expected_bar_data_hash=*/0)
        .has_value();
}

// ---------------------------------------------------------------------------
// Checkpoint::default_directory
// ---------------------------------------------------------------------------
std::string Checkpoint::default_directory() {
    return "./output/checkpoints";
}
