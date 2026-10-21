#include "results.h"
#include "wfo_engine.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace results {

// ---------------------------------------------------------------------------
// Helper: make a descriptive filename stem from a WFO run
// ---------------------------------------------------------------------------
static std::string run_filename_stem(const WFORunResult& run) {
    std::string stem = "freq" + std::to_string(run.schedule.frequency_days) + "d" + "_lb" +
                       std::to_string(run.schedule.lookback_days) + "d";

    // Include trigger type for adaptive configs
    if (run.schedule.trigger != WFOTrigger::FIXED_INTERVAL) {
        stem += "_";
        stem += trigger_name(run.schedule.trigger);
    }

    stem += "_" + run.selection_strategy;
    return stem;
}

// ---------------------------------------------------------------------------
// write_window_csv
// ---------------------------------------------------------------------------
bool write_window_csv(const std::string& filepath, const WFORunResult& run) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Results: Cannot write " << filepath << std::endl;
        return false;
    }
    out << std::setprecision(9);

    // Header
    out << "window_index,is_start_bar,is_end_bar,oos_start_bar,oos_end_bar,"
           "strategy_id,archetype,skipped,"
           "is_total_pnl,is_sharpe,is_annual_roc,is_profit_factor,"
           "oos_lag1_total_pnl,oos_lag1_sharpe,oos_lag1_annual_roc,"
           "oos_lag2_total_pnl,oos_lag2_sharpe,oos_lag2_annual_roc,"
           "pnl_decomp_bar_body_pct,pnl_decomp_gap_pct\n";

    for (const auto& w : run.windows) {
        out << w.window_index << "," << w.is_start_bar << "," << w.is_end_bar << ","
            << w.oos_start_bar << "," << w.oos_end_bar << "," << w.strategy_id << "," << w.archetype
            << "," << (w.skipped ? 1 : 0) << ",";

        if (!w.skipped) {
            out << w.is_result.total_pnl << "," << w.is_result.sharpe_ratio << ","
                << w.is_result.annual_roc << "," << w.is_result.profit_factor << ","
                << w.oos_lag1.total_pnl << "," << w.oos_lag1.sharpe_ratio << ","
                << w.oos_lag1.annual_roc << "," << w.oos_lag2.total_pnl << ","
                << w.oos_lag2.sharpe_ratio << "," << w.oos_lag2.annual_roc << ","
                << w.pnl_decomp.bar_body_pct << "," << w.pnl_decomp.gap_pct;
        } else {
            out << "0,0,0,0,0,0,0,0,0,0,0,0";
        }

        out << "\n";
    }

    out.close();
    if (out.fail()) {
        std::cerr << "Results: error writing " << filepath << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_summary_csv
// ---------------------------------------------------------------------------
bool write_summary_csv(const std::string& filepath, const std::vector<WFORunResult>& runs) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Results: Cannot write " << filepath << std::endl;
        return false;
    }
    out << std::setprecision(9);

    // Header
    out << "schedule_freq_days,schedule_lookback_days,trigger,selection_strategy,"
           "total_windows,windows_skipped,viable,parameter_stability,"
           "oos_total_pnl,oos_total_trades,oos_annual_roc,oos_sharpe,"
           "oos_profit_factor,oos_max_drawdown\n";

    for (const auto& r : runs) {
        out << r.schedule.frequency_days << "," << r.schedule.lookback_days << ","
            << trigger_name(r.schedule.trigger) << "," << r.selection_strategy << ","
            << static_cast<int>(r.windows.size()) << "," << r.windows_skipped << ","
            << (r.sufficient_windows ? 1 : 0) << "," << r.parameter_stability << ","
            << r.oos_aggregate.total_pnl << "," << r.oos_aggregate.total_trades << ","
            << r.oos_aggregate.annual_roc << "," << r.oos_aggregate.sharpe_ratio << ","
            << r.oos_aggregate.profit_factor << "," << r.oos_aggregate.max_drawdown << "\n";
    }

    out.close();
    if (out.fail()) {
        std::cerr << "Results: error writing " << filepath << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_equity_csv
// ---------------------------------------------------------------------------
bool write_equity_csv(const std::string& filepath, const WFORunResult& run) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Results: Cannot write " << filepath << std::endl;
        return false;
    }
    out << std::setprecision(9);

    out << "timestamp_ns,cumulative_pnl\n";

    if (run.equity_curve.size() != run.equity_timestamps.size()) {
        std::cerr << "Results: equity_curve size (" << run.equity_curve.size()
                  << ") != equity_timestamps size (" << run.equity_timestamps.size()
                  << "); writing "
                  << std::min(run.equity_curve.size(), run.equity_timestamps.size()) << " rows"
                  << std::endl;
    }

    const size_t n = std::min(run.equity_curve.size(), run.equity_timestamps.size());
    for (size_t i = 0; i < n; ++i) {
        out << run.equity_timestamps[i] << "," << run.equity_curve[i] << "\n";
    }

    out.close();
    if (out.fail()) {
        std::cerr << "Results: error writing " << filepath << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// write_walk_forward_results
// ---------------------------------------------------------------------------
bool write_walk_forward_results(const std::string&               output_dir,
                                const std::vector<WFORunResult>& runs,
                                const std::string&               best_selection_strategy,
                                float                            best_selection_avg_oos_roc) {
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    std::filesystem::create_directories(output_dir + "/details", ec);
    std::filesystem::create_directories(output_dir + "/equity", ec);

    bool success = true;

    // Summary CSV
    if (!write_summary_csv(output_dir + "/summary.csv", runs)) {
        success = false;
    }

    // Per-config detail and equity CSVs
    for (const auto& run : runs) {
        std::string stem = run_filename_stem(run);

        if (!write_window_csv(output_dir + "/details/" + stem + ".csv", run)) {
            success = false;
        }

        if (!write_equity_csv(output_dir + "/equity/" + stem + ".csv", run)) {
            success = false;
        }
    }

    // Meta-analysis CSV: best selection strategy across all schedules
    if (!best_selection_strategy.empty()) {
        std::string   path = output_dir + "/meta_analysis.csv";
        std::ofstream out(path);
        if (!out.is_open()) {
            std::cerr << "Results: Cannot write " << path << std::endl;
            success = false;
        } else {
            out << std::setprecision(9) << "metric,value\n";
            out << "best_selection_strategy," << best_selection_strategy << "\n";
            out << "best_selection_avg_oos_roc," << best_selection_avg_oos_roc << "\n";
            out << "total_runs," << runs.size() << "\n";

            // Per-selection-strategy breakdown
            int viable_count = 0;
            for (const auto& r : runs) {
                if (r.sufficient_windows) ++viable_count;
            }
            out << "viable_runs," << viable_count << "\n";
            out.close();
            if (out.fail()) {
                std::cerr << "Results: error writing " << path << std::endl;
                success = false;
            }
        }
    }

    return success;
}

}  // namespace results
