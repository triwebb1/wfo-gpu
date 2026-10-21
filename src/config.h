#pragma once
// ---------------------------------------------------------------------------
// RunConfig -- everything a run needs, loaded from a TOML file.
//
// See config/*.toml for annotated examples. Unknown sections and keys are
// rejected so a typo cannot silently fall back to a default.
//
//   [data]        bars, output
//   [instrument]  symbol, tick_size, tick_value, point_value,
//                 spread_ticks, slippage_ticks, capital
//   [session]     model, session_open, session_close, flat_at_close,
//                 core_hours, halt_gap_minutes, early_close_calendar
//   [screen]      start, end, max_strategies, robustness_floor, entry_lag,
//                 early_termination
//   [floors]      min_trades, min_profit_factor, min_annual_roc,
//                 min_total_pnl, max_negative_years
//   [compute]     device, indicators_on_device
//   [resources]   max_cpu_pct, max_ram_gb
// ---------------------------------------------------------------------------

#include <cstdint>
#include <optional>
#include <string>

#include "compute/cl_context.h"
#include "cost_model.h"
#include "landscape.h"
#include "session_model.h"

// A position in the bar file: either a bar index or a UTC date/time string
// ("2024-03-01", "2024-03-01T14:30:00Z").
struct BarBound {
    std::optional<int64_t> index;
    std::string            timestamp;
    bool                   is_set() const { return index.has_value() || !timestamp.empty(); }
};

struct RunConfig {
    // [data]
    std::string bars;
    std::string output = "output";

    // [instrument]; without the section the cost model defaults to an
    // E-mini-style contract and instrument_configured stays false.
    std::string symbol;
    CostModel   cost                  = CostModel::es();
    bool        instrument_configured = false;

    // [session]
    SessionModel session = SessionModel::us_futures();

    // [screen]
    BarBound screen_start;
    BarBound screen_end;
    int64_t  max_strategies    = 10'000'000;
    float    robustness_floor  = 0.50f;
    int      entry_lag         = 1;
    bool     early_termination = false;

    // [floors]
    SanityFloors floors;

    // [compute]
    DeviceType device               = DeviceType::AUTO;
    bool       indicators_on_device = false;  // see CLIndicators::set_default_on_device

    // [resources] (unset = keep process defaults)
    std::optional<float> max_cpu_pct;
    std::optional<float> max_ram_gb;
};

struct ConfigResult {
    RunConfig   config;
    std::string error;  // empty on success
    bool        ok() const { return error.empty(); }
};

// Parse a TOML document. `source_name` is used in error messages.
ConfigResult parse_run_config(const std::string& toml_text,
                              const std::string& source_name = "config");
ConfigResult load_run_config(const std::string& path);

// Parse "HH:MM" into seconds after midnight. Returns false if malformed.
bool parse_clock_time(const std::string& text, int32_t& seconds);

// Parse a device name: auto, gpu, opencl-cpu, native.
bool parse_device(const std::string& text, DeviceType& out);
