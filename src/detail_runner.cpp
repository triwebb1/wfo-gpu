#include "detail_runner.h"

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

#include "compute/cl_backtest.h"
#include "compute/cl_context.h"
#include "compute/cl_indicators.h"
#include "compute/cl_memory.h"
#include "compute/indicator_warmup.h"
#include "grids/indicator_grid.h"
#include "grids/session_signal_grid.h"
#include "grids/strategy_grid.h"
#include "indicators/signal_normalize.h"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int64_t NS_PER_SEC = 1'000'000'000LL;

// ---------------------------------------------------------------------------
// Session-day-aware date helpers for quarterly PnL bucketing.
//
// CME futures sessions span two UTC calendar days (18:00 ET to 17:00 ET).
// The "trade date" for a session is the date of the RTH portion (the day
// portion from 09:30-16:00 ET).  For bars in the evening ETH portion
// (18:00-23:59 ET), the trade date is the following calendar day.
//
// When SessionInfo is available, we derive the ET offset from time_of_day_sec
// to correctly convert UTC timestamps to the session's trade date.  When not
// available, we fall back to UTC (gmtime), which is off by up to 5 hours.
// ---------------------------------------------------------------------------

// Compute the ET UTC offset (in seconds) from a bar's UTC timestamp and
// its SessionInfo time_of_day_sec.  Returns -18000 (EST) or -14400 (EDT).
static int32_t compute_et_offset(int64_t ts_ns, int32_t time_of_day_sec) {
    const int64_t utc_sec = ts_ns / NS_PER_SEC;
    const int32_t utc_day_sec =
        static_cast<int32_t>(((utc_sec % 86400) + 86400) % 86400);  // handle negative timestamps
    int32_t offset = time_of_day_sec - utc_day_sec;
    // Normalize to [-43200, +43200) range to handle day boundary wrapping
    if (offset > 43200) offset -= 86400;
    if (offset < -43200) offset += 86400;
    return offset;
}

// Get the session trade date components (year, month) from a bar's timestamp
// and SessionInfo.  The trade date is the calendar date of the RTH portion:
// for bars at 18:00 ET or later (evening session), the trade date is the
// following calendar day.
static void session_trade_date(int64_t ts_ns, const SessionInfo& si, int& out_year,
                               int& out_month) {
    int32_t et_offset = compute_et_offset(ts_ns, si.time_of_day_sec);
    // Convert to ET seconds since epoch
    time_t et_sec = static_cast<time_t>(ts_ns / NS_PER_SEC) + et_offset;
    // If evening session (>= 18:00 ET), trade date is next calendar day
    if (si.time_of_day_sec >= 18 * 3600) {
        et_sec += 86400;
    }
    std::tm t{};
#ifdef _WIN32
    gmtime_s(&t, &et_sec);
#else
    gmtime_r(&et_sec, &t);
#endif
    out_year  = t.tm_year + 1900;
    out_month = t.tm_mon + 1;
}

// Fallback: year/month from UTC timestamp (used when no SessionInfo available)
static int year_of(int64_t ts_ns) {
    const time_t secs = static_cast<time_t>(ts_ns / NS_PER_SEC);
    std::tm      t{};
#ifdef _WIN32
    gmtime_s(&t, &secs);
#else
    gmtime_r(&secs, &t);
#endif
    return t.tm_year + 1900;
}

static int month_of(int64_t ts_ns) {
    const time_t secs = static_cast<time_t>(ts_ns / NS_PER_SEC);
    std::tm      t{};
#ifdef _WIN32
    gmtime_s(&t, &secs);
#else
    gmtime_r(&secs, &t);
#endif
    return t.tm_mon + 1;
}

static int quarter_of(int month) {
    return (month - 1) / 3 + 1;
}

namespace detail {

// ---------------------------------------------------------------------------
// Bucket trades into quarterly PnL
//
// When sessions is non-null, uses DST-aware session date computation for
// correct year/quarter assignment at session boundaries.
// ---------------------------------------------------------------------------
static std::vector<QuarterlyPnL> bucket_quarterly(const std::vector<Trade>& trades, const Bar* bars,
                                                  const SessionInfo* sessions = nullptr) {
    // Accumulate by (year, quarter)
    std::map<std::pair<int, int>, QuarterlyPnL> buckets;

    for (const auto& t : trades) {
        int64_t ts = bars[t.exit_bar].timestamp;
        int     y, m;
        if (sessions) {
            session_trade_date(ts, sessions[t.exit_bar], y, m);
        } else {
            y = year_of(ts);
            m = month_of(ts);
        }
        int q = quarter_of(m);

        auto  key = std::make_pair(y, q);
        auto& b   = buckets[key];
        b.year    = y;
        b.quarter = q;
        b.pnl += t.net_pnl;
        b.trade_count += 1;
    }

    std::vector<QuarterlyPnL> result;
    result.reserve(buckets.size());
    for (const auto& [_, qp] : buckets) {
        result.push_back(qp);
    }
    return result;
}

// ---------------------------------------------------------------------------
// run_detail
// ---------------------------------------------------------------------------
std::vector<DetailResult> run_detail(const Bar* bars, int num_bars, int eval_start, int eval_end,
                                     const std::vector<int>& strategy_ids, const CostModel& cost,
                                     int entry_lag, const SessionInfo* sessions,
                                     const MarketData* md) {
    std::vector<DetailResult> results;
    results.reserve(strategy_ids.size());

    if (strategy_ids.empty() || eval_start >= eval_end) return results;

    // Set up compute pipeline
    CLContext ctx;
    ctx.initialize(CLContext::default_device());
    CLIndicators cl_ind;
    CLBacktest   cl_bt;

    IndicatorGrid indicator_grid;
    const int     warmup     = max_indicator_warmup(indicator_grid);
    const int     load_start = std::max(0, eval_start - warmup);
    const int     load_end   = std::min(num_bars, eval_end);
    const int     load_count = load_end - load_start;

    if (load_count <= 0) return results;

    // Build indicator specs
    std::vector<IndicatorSpec> indicator_specs;
    for (int i = 0; i < indicator_grid.size(); ++i) {
        const IndicatorParams& params = indicator_grid.params(i);
        int                    ns     = indicator_num_series(params);
        for (int s = 0; s < ns; ++s) {
            indicator_specs.push_back(IndicatorSpec::from_params(params, s));
        }
    }
    int total_raw_series = static_cast<int>(indicator_specs.size());
    int num_signals      = indicator_grid.size();

    // Allocate and compute raw indicators
    CLMemory            memory;
    std::vector<float*> raw_ptrs =
        memory.allocate_indicator_buffers(ctx, total_raw_series, load_count);
    memory.upload_bars(ctx, &bars[load_start], 0, load_count);
    const SoABarData& soa = memory.soa_bars();

    cl_ind.compute_with_bars(ctx, &bars[load_start], 0, soa.open.data(), soa.high.data(),
                             soa.low.data(), soa.close.data(), load_count, indicator_specs,
                             raw_ptrs);

    // Normalize raw series -> signal arrays (one per indicator)
    std::vector<std::vector<float>> signal_storage(num_signals);
    std::vector<float*>             signal_ptrs(num_signals);
    for (int i = 0; i < num_signals; ++i) {
        signal_storage[i].resize(load_count);
        signal_ptrs[i] = signal_storage[i].data();
    }
    normalize_all_indicators(indicator_grid, raw_ptrs, &bars[load_start], load_count, signal_ptrs);

    // Session signals follow the technical ones, as in the screen, so the
    // strategy ids decode to the same strategies.
    if (md && md->session_model().session_signals && md->bar_count() == num_bars) {
        append_session_signal_series(*md, SessionSignalGrid(), load_start, load_end,
                                     signal_storage);
        num_signals = static_cast<int>(signal_storage.size());
        signal_ptrs.resize(signal_storage.size());
        for (size_t i = 0; i < signal_storage.size(); ++i)
            signal_ptrs[i] = signal_storage[i].data();
    }

    const int    eval_start_local = eval_start - load_start;
    StrategyGrid strategy_grid(num_signals);

    // Evaluate each strategy and collect trades
    for (int sid : strategy_ids) {
        DetailResult dr;
        dr.strategy_id = sid;

        if (sid < 0 || sid >= strategy_grid.total_count()) {
            results.push_back(dr);
            continue;
        }

        const StrategyConfig sc = strategy_grid.entry_at(sid).to_strategy_config();
        dr.config               = sc;

        // Build offset pointers for the eval range
        std::vector<const float*> offset_ptrs(num_signals);
        for (int k = 0; k < num_signals; ++k) {
            offset_ptrs[k] = signal_ptrs[k] + eval_start_local;
        }

        BacktestEngine engine;
        // Thread `sessions` so RTH/MORNING/AFTERNOON filter,
        // halt suppression, and session-close exit activate in detail runs.
        dr.metrics = engine.run(bars, sessions, eval_start, eval_end, offset_ptrs.data(),
                                num_signals, nullptr /* time_gates */, sc, cost, entry_lag);
        dr.metrics.strategy_id = sid;

        // Capture the trade log
        dr.trades = engine.last_trades();

        // Bucket quarterly PnL (DST-correct when sessions are available)
        dr.quarterly = bucket_quarterly(dr.trades, bars, sessions);

        results.push_back(std::move(dr));
    }

    return results;
}

// ---------------------------------------------------------------------------
// load_strategy_ids
// ---------------------------------------------------------------------------
std::vector<int> load_strategy_ids(const std::string& filepath) {
    std::vector<int> ids;
    std::ifstream    fin(filepath);
    if (!fin.is_open()) {
        std::cerr << "DetailRunner: Cannot open " << filepath << std::endl;
        return ids;
    }

    std::string line;
    while (std::getline(fin, line)) {
        // Strip comments
        auto pos = line.find('#');
        if (pos != std::string::npos) line.resize(pos);

        // Strip whitespace
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r'))
            line.pop_back();
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
            line.erase(line.begin());

        if (line.empty()) continue;

        try {
            ids.push_back(std::stoi(line));
        } catch (...) {
            std::cerr << "DetailRunner: Skipping invalid line: " << line << std::endl;
        }
    }
    return ids;
}

// ---------------------------------------------------------------------------
// write_trade_csv
// ---------------------------------------------------------------------------
bool write_trade_csv(const std::string& filepath, const DetailResult& result, const Bar* bars) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "trade_idx,entry_bar,exit_bar,entry_ts,exit_ts,"
           "entry_price,exit_price,direction,gross_pnl,net_pnl,"
           "min_unrealized,max_unrealized,hold_bars\n";

    for (int i = 0; i < static_cast<int>(result.trades.size()); ++i) {
        const Trade& t = result.trades[i];
        out << i << "," << t.entry_bar << "," << t.exit_bar << "," << bars[t.entry_bar].timestamp
            << "," << bars[t.exit_bar].timestamp << "," << t.entry_price << "," << t.exit_price
            << "," << t.direction << "," << t.gross_pnl << "," << t.net_pnl << ","
            << t.min_unrealized << "," << t.max_unrealized << "," << (t.exit_bar - t.entry_bar)
            << "\n";
    }
    out << std::setprecision(9);
    out.close();
    if (out.fail()) {
        std::cerr << "DetailRunner: error writing " << filepath << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_quarterly_csv
// ---------------------------------------------------------------------------
bool write_quarterly_csv(const std::string& filepath, const DetailResult& result) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "year,quarter,pnl,trades\n";

    for (const auto& q : result.quarterly) {
        out << q.year << "," << q.quarter << "," << q.pnl << "," << q.trade_count << "\n";
    }
    out << std::setprecision(9);
    out.close();
    if (out.fail()) {
        std::cerr << "DetailRunner: error writing " << filepath << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_detail_summary_csv
// ---------------------------------------------------------------------------
bool write_detail_summary_csv(const std::string&               filepath,
                              const std::vector<DetailResult>& results) {
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << "strategy_id,total_trades,trades_per_week,win_rate,"
           "avg_win,avg_loss,total_pnl,max_drawdown,max_live_drawdown,"
           "avg_trade_dd,avg_trade_peak,"
           "sharpe,calmar,profit_factor,green_weeks_pct,max_red_weeks,"
           "annual_roc,trade_recovery_ratio,viable\n";

    for (const auto& dr : results) {
        const BacktestResult& m = dr.metrics;
        out << dr.strategy_id << "," << m.total_trades << "," << m.trades_per_week << ","
            << m.win_rate << "," << m.avg_win << "," << m.avg_loss << "," << m.total_pnl << ","
            << m.max_drawdown << "," << m.max_live_drawdown << "," << m.avg_trade_dd << ","
            << m.avg_trade_peak << "," << m.sharpe_ratio << "," << m.calmar_ratio << ","
            << m.profit_factor << "," << m.green_weeks_pct << "," << m.max_red_weeks << ","
            << m.annual_roc << "," << m.trade_recovery_ratio << "," << (m.viable ? 1 : 0) << "\n";
    }
    out << std::setprecision(9);
    out.close();
    if (out.fail()) {
        std::cerr << "DetailRunner: error writing " << filepath << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_all_detail_outputs
// ---------------------------------------------------------------------------
bool write_all_detail_outputs(const std::string&               output_dir,
                              const std::vector<DetailResult>& results, const Bar* bars) {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    std::filesystem::create_directories(output_dir + "/trades", ec);
    std::filesystem::create_directories(output_dir + "/quarterly", ec);

    bool ok = true;

    if (!write_detail_summary_csv(output_dir + "/detail_summary.csv", results)) ok = false;

    for (const auto& dr : results) {
        std::string id_str = std::to_string(dr.strategy_id);

        if (!write_trade_csv(output_dir + "/trades/strategy_" + id_str + ".csv", dr, bars))
            ok = false;

        if (!write_quarterly_csv(output_dir + "/quarterly/strategy_" + id_str + ".csv", dr))
            ok = false;
    }

    if (ok) {
        std::cout << "Detail outputs written to " << output_dir << "/\n"
                  << "  detail_summary.csv          (" << results.size() << " strategies)\n"
                  << "  trades/strategy_*.csv       (per-trade logs)\n"
                  << "  quarterly/strategy_*.csv    (quarterly PnL)" << std::endl;
    }

    return ok;
}

}  // namespace detail
