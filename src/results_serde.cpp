#include "results_serde.h"
#include "durable_file.h"

#include <cstring>

namespace results_serde {

namespace {

template <typename T>
void write_pod(DurableFileWriter& w, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "write_pod requires trivially copyable");
    w.write_bytes(&v, sizeof(T));
}

void write_bool(DurableFileWriter& w, bool v) {
    uint8_t b = v ? 1 : 0;
    w.write_bytes(&b, 1);
}

template <typename T>
bool read_pod(std::istream& in, T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "read_pod requires trivially copyable");
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return in.good();
}

bool read_bool(std::istream& in, bool& v) {
    uint8_t b = 0;
    in.read(reinterpret_cast<char*>(&b), 1);
    if (!in.good()) return false;
    v = (b != 0);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// write_backtest_result -- field-by-field emission via DurableFileWriter.
// Byte-layout MUST match the read path in checkpoint.cpp
// (read_backtest_result) for format-parity.
// ---------------------------------------------------------------------------
void write_backtest_result(DurableFileWriter& w, const BacktestResult& r) {
    write_pod(w, r.start_ts);
    write_pod(w, r.end_ts);
    write_pod(w, r.total_trades);
    write_pod(w, r.trades_per_week);
    write_pod(w, r.win_rate);
    write_pod(w, r.avg_win);
    write_pod(w, r.avg_loss);
    write_pod(w, r.avg_hold_bars);
    write_pod(w, r.total_pnl);
    write_pod(w, r.gross_wins);
    write_pod(w, r.gross_losses);
    write_pod(w, r.max_drawdown);
    write_pod(w, r.max_live_drawdown);
    write_pod(w, r.avg_trade_dd);
    write_pod(w, r.avg_trade_peak);
    write_pod(w, r.sharpe_ratio);
    write_pod(w, r.calmar_ratio);
    write_pod(w, r.profit_factor);
    write_pod(w, r.payoff_ratio);
    write_pod(w, r.green_weeks_pct);
    write_pod(w, r.green_months_pct);
    write_pod(w, r.max_red_weeks);
    write_pod(w, r.negative_years);
    write_pod(w, r.negative_months);
    write_pod(w, r.annual_roc);
    write_pod(w, r.reward_risk_trade);
    write_pod(w, r.daily_profit_rate);
    write_pod(w, r.time_to_double);
    write_pod(w, r.trade_recovery_ratio);
    write_pod(w, r.strategy_id);
    write_bool(w, r.viable);
    write_bool(w, r.metrics_complete);
}

bool read_backtest_result(std::istream& in, BacktestResult& r) {
    if (!read_pod(in, r.start_ts)) return false;
    if (!read_pod(in, r.end_ts)) return false;
    if (!read_pod(in, r.total_trades)) return false;
    if (!read_pod(in, r.trades_per_week)) return false;
    if (!read_pod(in, r.win_rate)) return false;
    if (!read_pod(in, r.avg_win)) return false;
    if (!read_pod(in, r.avg_loss)) return false;
    if (!read_pod(in, r.avg_hold_bars)) return false;
    if (!read_pod(in, r.total_pnl)) return false;
    if (!read_pod(in, r.gross_wins)) return false;
    if (!read_pod(in, r.gross_losses)) return false;
    if (!read_pod(in, r.max_drawdown)) return false;
    if (!read_pod(in, r.max_live_drawdown)) return false;
    if (!read_pod(in, r.avg_trade_dd)) return false;
    if (!read_pod(in, r.avg_trade_peak)) return false;
    if (!read_pod(in, r.sharpe_ratio)) return false;
    if (!read_pod(in, r.calmar_ratio)) return false;
    if (!read_pod(in, r.profit_factor)) return false;
    if (!read_pod(in, r.payoff_ratio)) return false;
    if (!read_pod(in, r.green_weeks_pct)) return false;
    if (!read_pod(in, r.green_months_pct)) return false;
    if (!read_pod(in, r.max_red_weeks)) return false;
    if (!read_pod(in, r.negative_years)) return false;
    if (!read_pod(in, r.negative_months)) return false;
    if (!read_pod(in, r.annual_roc)) return false;
    if (!read_pod(in, r.reward_risk_trade)) return false;
    if (!read_pod(in, r.daily_profit_rate)) return false;
    if (!read_pod(in, r.time_to_double)) return false;
    if (!read_pod(in, r.trade_recovery_ratio)) return false;
    if (!read_pod(in, r.strategy_id)) return false;
    if (!read_bool(in, r.viable)) return false;
    if (!read_bool(in, r.metrics_complete)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// write_all_results_file -- self-describing all_results.bin layout:
//   bytes 0-3:   magic "WFAR"
//   bytes 4-7:   version (uint32_t, currently 1)
//   bytes 8-11:  count (int32_t)
//   bytes 12+:   count * BacktestResult (field-by-field via write_backtest_result)
// ---------------------------------------------------------------------------
void write_all_results_file(DurableFileWriter& w, const std::vector<BacktestResult>& results) {
    w.write_bytes(ALL_RESULTS_MAGIC, 4);
    write_pod(w, ALL_RESULTS_VERSION);
    int32_t count = static_cast<int32_t>(results.size());
    write_pod(w, count);
    for (const auto& r : results) {
        write_backtest_result(w, r);
    }
}

}  // namespace results_serde
