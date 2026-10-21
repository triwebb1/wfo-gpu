// wfo-gpu generate | convert | info

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

#include "bar_io.h"
#include "civil_time.h"
#include "cli/commands.h"
#include "compute/cl_context.h"
#include "grids/indicator_grid.h"
#include "grids/session_signal_grid.h"
#include "grids/strategy_grid.h"
#include "orchestrator.h"
#include "market_data.h"
#include "resource_monitor.h"
#include "synthetic.h"

namespace cli {

namespace {

std::string format_utc(int64_t ts_ns) {
    const int64_t     sec  = civil::floor_div(ts_ns, civil::NS_PER_SEC);
    const int64_t     days = civil::floor_div(sec, civil::SEC_PER_DAY);
    const int64_t     tod  = sec - days * civil::SEC_PER_DAY;
    const civil::Date d    = civil::civil_from_days(days);
    char              buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d UTC", d.year, d.month, d.day,
                  static_cast<int>(tod / 3600), static_cast<int>(tod / 60 % 60),
                  static_cast<int>(tod % 60));
    return buf;
}

std::string signal_name(int id, const SessionModel& model) {
    static const IndicatorGrid grid;
    if (id < grid.size()) return describe_indicator(grid.params(id));
    if (!model.session_signals) return "?";
    static const SessionSignalGrid ss;
    int                            next = grid.size();
    for (int g = 0; g < ss.size(); ++g) {
        auto sig = create_session_signal(ss.params(g));
        if (!sig) continue;
        if (id < next + sig->num_signals())
            return std::string("session: ") + sig->name() + " #" + std::to_string(id - next);
        next += sig->num_signals();
    }
    return "?";
}

int describe_strategy(int64_t id, const SessionModel& model) {
    const int          n = Orchestrator::screen_signal_count(model);
    const StrategyGrid grid(n);
    if (id < 0 || id >= grid.total_count())
        throw UsageError("--strategy must be in [0, " + std::to_string(grid.total_count()) + ")");
    const StrategyGridEntry e       = grid.entry_at(id);
    const StrategyConfig    c       = e.to_strategy_config();
    static const char*      AGG[]   = {"weighted sum", "majority vote", "unanimous"};
    static const char*      HOURS[] = {"all session hours", "core hours", "morning", "afternoon"};
    static const char*      EXIT[]  = {"signal reversal",
                                       "bar timeout",
                                       "trailing stop",
                                       "opposing signal",
                                       "max loss (ATR)",
                                       "take profit (ATR)",
                                       "take profit / stop loss (ATR)"};
    std::cout << "Strategy " << id << " (grid of " << n << " signals, " << model.name << ")\n";
    for (int i = 0; i < e.num_indicators; ++i) {
        std::cout << "  signal " << e.indicator_ids[i] << ": "
                  << signal_name(e.indicator_ids[i], model) << "  weight " << e.weights[i]
                  << (c.indicator_roles[i] == StrategyConfig::FILTER ? "  (filter)" : "") << "\n";
    }
    std::cout << "  aggregation      " << AGG[c.aggregation] << "\n"
              << "  entry threshold  " << c.entry_threshold << "\n"
              << "  trading hours    " << HOURS[c.trading_hours] << "\n"
              << "  exit             " << EXIT[c.exit_mode];
    switch (c.exit_mode) {
        case StrategyConfig::BAR_TIMEOUT:
            std::cout << ", " << c.exit_timeout_bars << " bars";
            break;
        case StrategyConfig::TRAILING_STOP:
            std::cout << ", " << c.trailing_atr_mult << " x ATR";
            break;
        case StrategyConfig::MAX_LOSS_ATR:
            std::cout << ", " << c.max_loss_atr_mult << " x ATR";
            break;
        case StrategyConfig::TAKE_PROFIT_ATR:
            std::cout << ", " << c.tp_atr_mult << " x ATR";
            break;
        case StrategyConfig::TP_SL_ATR:
            std::cout << ", TP " << c.tp_atr_mult << " / SL " << c.sl_atr_mult << " x ATR";
            break;
        default:
            break;
    }
    std::cout << "\n";
    return 0;
}

bool parse_date(const std::string& text, int& y, int& m, int& d) {
    int64_t ns = 0;
    if (text.size() != 10 || !parse_timestamp_ns(text, ns)) return false;
    const civil::Date date =
        civil::civil_from_days(civil::floor_div(ns / civil::NS_PER_SEC, 86400));
    y = date.year;
    m = date.month;
    d = date.day;
    return true;
}

}  // namespace

int cmd_generate(Args& args) {
    SyntheticConfig   cfg;
    const std::string out = args.required("--out");
    if (auto v = args.value("--calendar")) {
        if (*v == "us_futures")
            cfg.calendar = SyntheticCalendar::US_FUTURES;
        else if (*v == "continuous")
            cfg.calendar = SyntheticCalendar::CONTINUOUS;
        else
            throw UsageError("--calendar must be us_futures or continuous");
    }
    if (auto v = args.value("--start")) {
        if (!parse_date(*v, cfg.start_year, cfg.start_month, cfg.start_day))
            throw UsageError("--start expects YYYY-MM-DD");
    }
    if (auto v = args.value("--symbol")) cfg.symbol = *v;
    if (auto v = args.integer("--days")) cfg.num_days = static_cast<int>(*v);
    if (auto v = args.integer("--seed")) cfg.seed = static_cast<uint64_t>(*v);
    if (auto v = args.integer("--timeframe")) cfg.timeframe_sec = static_cast<int>(*v);
    if (auto v = args.number("--price")) cfg.start_price = *v;
    if (auto v = args.number("--tick")) cfg.tick_size = *v;
    if (auto v = args.number("--vol")) cfg.annual_volatility = *v;
    if (auto v = args.number("--drift")) cfg.annual_drift = *v;
    if (auto v = args.number("--trend")) cfg.trend_strength = *v;
    if (auto v = args.number("--trend-bars")) cfg.trend_duration_bars = *v;
    args.finish();

    const auto       t0 = std::chrono::steady_clock::now();
    std::vector<Bar> bars;
    try {
        bars = generate_synthetic_bars(cfg);
    } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
    }
    std::string err;
    if (!write_bar_file(out, cfg.symbol, cfg.timeframe_sec, bars, &err)) {
        std::cerr << "error: " << err << "\n";
        return 1;
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "Wrote " << bars.size() << " synthetic " << cfg.timeframe_sec << "s bars ("
              << cfg.num_days << " days, seed " << cfg.seed << ") to " << out << " in "
              << std::fixed << std::setprecision(2) << secs << " s\n"
              << "  " << format_utc(bars.front().timestamp) << "  ->  "
              << format_utc(bars.back().timestamp) << "\n";
    return 0;
}

int cmd_convert(Args& args) {
    const std::string in          = args.required("--csv");
    const std::string out         = args.required("--out");
    const std::string symbol      = args.value("--symbol").value_or("BARS");
    const auto        tf_override = args.integer("--timeframe");
    args.finish();

    const CsvImportResult r = read_bars_csv_file(in);
    if (!r.ok()) {
        std::cerr << "error: " << in << ": " << r.error << "\n";
        return 1;
    }
    const int tf = tf_override ? static_cast<int>(*tf_override) : r.timeframe_sec;
    if (tf <= 0) {
        std::cerr << "error: cannot infer the bar timeframe; pass --timeframe <seconds>\n";
        return 1;
    }
    std::string err;
    if (!write_bar_file(out, symbol, tf, r.bars, &err)) {
        std::cerr << "error: " << err << "\n";
        return 1;
    }
    std::cout << "Converted " << r.bars.size() << " bars (" << tf << "s) from " << in << " to "
              << out << "\n";
    return 0;
}

int cmd_info(Args& args) {
    const auto        bars         = args.value("--bars");
    const std::string session_name = args.value("--session").value_or("us_futures");
    const bool        list_signals = args.flag("--signals");
    const auto        strategy_id  = args.integer("--strategy");
    args.finish();

    SessionModel model;
    if (!SessionModel::from_name(session_name, model))
        throw UsageError("--session must be us_futures or continuous");

    if (strategy_id) return describe_strategy(*strategy_id, model);

    if (list_signals) {
        // Signal ids are the indicator ids that strategy ids decode to.
        const IndicatorGrid grid;
        std::cout << "id    signal\n";
        for (int i = 0; i < grid.size(); ++i)
            std::cout << std::left << std::setw(6) << i << describe_indicator(grid.params(i))
                      << "\n";
        if (model.session_signals) {
            const SessionSignalGrid ss;
            int                     id = grid.size();
            for (int g = 0; g < ss.size(); ++g) {
                auto sig = create_session_signal(ss.params(g));
                if (!sig) continue;
                for (int j = 0; j < sig->num_signals(); ++j)
                    std::cout << std::left << std::setw(6) << id++ << "session: " << sig->name()
                              << " #" << j << "\n";
            }
        }
        std::cout << std::right;
        return 0;
    }

    ResourceMonitor monitor;
    std::cout << "System\n"
              << "  logical cpus      " << ResourceMonitor::hardware_concurrency() << "\n"
              << "  worker threads    " << monitor.max_threads() << "\n"
              << "  physical memory   " << std::fixed << std::setprecision(1)
              << ResourceMonitor::physical_ram_gb() << " GB\n";

    CLContext ctx;
    if (ctx.initialize(DeviceType::AUTO)) {
        const size_t free_bytes = ctx.available_memory_bytes();  // may log how it was measured
        std::cout << "  opencl device     " << ctx.device_name() << " ("
                  << free_bytes / (1024 * 1024) << " MB free of "
                  << ctx.device_memory_bytes() / (1024 * 1024) << " MB, largest buffer "
                  << ctx.max_alloc_bytes() / (1024 * 1024) << " MB)\n";
    } else {
#ifdef HAS_OPENCL
        std::cout << "  opencl device     none found; using the native CPU path\n";
#else
        std::cout << "  opencl device     not built with OpenCL; using the native CPU path\n";
#endif
    }

    if (!bars) return 0;
    MarketData md;
    if (!md.load(*bars, model)) {
        std::cerr << "error: cannot load " << *bars << "\n";
        return 1;
    }
    std::cout << "\nBar file " << *bars << "\n"
              << "  symbol            " << md.symbol() << "\n"
              << "  timeframe         " << md.header().timeframe_sec << " s\n"
              << "  bars              " << md.bar_count() << "\n"
              << "  sessions          " << md.session_count() << " (" << model.name << ")\n"
              << "  bars/calendar day " << std::setprecision(1) << md.bars_per_calendar_day()
              << "\n"
              << "  first bar         " << format_utc(md.bar(0).timestamp) << "\n"
              << "  last bar          " << format_utc(md.bar(md.bar_count() - 1).timestamp) << "\n";
    return 0;
}

}  // namespace cli
