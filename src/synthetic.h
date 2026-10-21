#pragma once
// ---------------------------------------------------------------------------
// Deterministic synthetic bar generator.
//
// Produces realistic-looking OHLCV bars without any market data: a random
// walk with Markov-switching volatility regimes, rare jumps, intraday
// volatility/volume seasonality, and overnight/weekend gaps. Prices are
// quantized to the tick size.
//
// The generator uses only integer arithmetic and IEEE-exact floating-point
// operations (+, -, *, /, sqrt), so the same config produces byte-identical
// output on every platform. Tests and benchmarks rely on that.
//
// With annual_drift = 0 and trend_strength = 0 the series has no exploitable
// structure, which makes it a useful null hypothesis: a sound walk-forward
// test should find no out-of-sample edge on it after costs.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

#include "market_data.h"

enum class SyntheticCalendar : uint8_t {
    US_FUTURES = 0,  // Sun-Fri, 18:00-17:00 US Eastern with a one-hour daily break
    CONTINUOUS = 1   // 24/7, UTC days, no breaks
};

struct SyntheticConfig {
    uint64_t          seed          = 1;
    std::string       symbol        = "SYN";
    int               timeframe_sec = 30;
    SyntheticCalendar calendar      = SyntheticCalendar::US_FUTURES;

    // First trading day (US_FUTURES: the date of the first session's close;
    // weekends are skipped) and number of trading days to generate.
    int start_year  = 2024;
    int start_month = 1;
    int start_day   = 2;
    int num_days    = 5;

    double start_price       = 5000.0;
    double tick_size         = 0.25;
    double annual_volatility = 0.18;  // fraction, e.g. 0.18 = 18%
    double annual_drift      = 0.0;   // fraction per year

    // Hidden trend regimes: a drift of +/- trend_strength bar-sigmas per bar
    // whose sign flips on average every trend_duration_bars. 0 gives a pure
    // random walk; e.g. 0.03 plants persistent trends that trend-following
    // strategies can exploit after costs. Useful for checking that the
    // optimizer finds real structure as well as rejecting noise.
    double trend_strength      = 0.0;
    double trend_duration_bars = 2000.0;

    // Volatility regimes: LOW / NORMAL / HIGH multipliers on base volatility.
    double low_vol_mult       = 0.6;
    double high_vol_mult      = 2.2;
    double regime_switch_prob = 0.002;  // per bar

    // Jumps: probability per bar and size in units of the bar's sigma.
    double jump_prob   = 0.0004;
    double jump_sigmas = 8.0;

    // Session-open gap size in units of one hour of volatility.
    double gap_sigmas = 1.0;

    uint32_t base_volume = 400;  // mean volume per bar at seasonality 1.0
    int      substeps    = 6;    // intra-bar path steps used for high/low
};

// Generate bars for the config. Timestamps are bar-open times in UTC
// nanoseconds and strictly increasing.
std::vector<Bar> generate_synthetic_bars(const SyntheticConfig& config);
