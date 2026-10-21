#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "../market_data.h"

// Sentinel for invalid/warmup values
constexpr float INDICATOR_NAN = std::numeric_limits<float>::quiet_NaN();

// True Range -- shared by ATR, ADX, Keltner. First bar (no prev close) uses H-L.
inline float compute_true_range(const Bar& cur, float prev_close) {
    float hl  = cur.high - cur.low;
    float hpc = std::abs(cur.high - prev_close);
    float lpc = std::abs(cur.low - prev_close);
    return std::max({hl, hpc, lpc});
}

// Moving average type used by crossover and other indicators
enum class MAType : uint8_t { EMA = 0, SMA = 1 };

// Base indicator interface
class Indicator {
public:
    virtual ~Indicator() = default;

    // Compute indicator over bars[start..end), writing to output[0..end-start)
    // Each indicator may produce multiple series (e.g., MACD produces 3)
    // output is a flat array: series0[0..N), series1[0..N), series2[0..N)
    // where N = end - start
    virtual void compute(const Bar* bars, int start, int end, float* output) const = 0;

    // Number of output series (EMA=1, Bollinger=3, etc.)
    virtual int num_series() const = 0;

    // Minimum warmup bars before output is valid
    virtual int warmup_bars() const = 0;

    // Name for debugging/logging
    virtual const char* name() const = 0;
};

// ---------------------------------------------------------------------------
// compute_anchored -- compute indicator on TF bars, broadcast to 30s alignment
//
// The indicator is computed on the higher-timeframe bars (tf_bars), producing
// one value per TF bar.  The results are then broadcast back to 30s bar
// granularity using completed_map: output_30s[i] = tf_output[completed_map[i]].
//
// completed_map must point at the latest TF bar that is complete when base
// bar i closes (TimeframeData::completed_map). Passing parent_map instead
// leaks the still-forming TF bar's close into earlier base bars.
//
// Parameters:
//   ind              - the indicator to compute
//   tf_bars          - pre-aggregated timeframe bars (from MarketData::timeframe())
//   tf_bar_count     - number of TF bars
//   completed_map    - completed_map[30s_idx] -> tf_bar_idx or -1 (from TimeframeData)
//   bar_start_30s    - first 30s bar index in the output range
//   bar_end_30s      - one past the last 30s bar index
//   output           - flat output array with (bar_end_30s - bar_start_30s) * num_series floats
//                      laid out as series0[0..N), series1[0..N), ...
// ---------------------------------------------------------------------------
inline void compute_anchored(
    const Indicator& ind, const Bar* tf_bars, int tf_bar_count,
    const int* completed_map,  // completed_map[30s_idx] -> tf_bar_idx or -1
    int bar_start_30s, int bar_end_30s,
    float* output)  // output has (bar_end_30s - bar_start_30s) * num_series floats
{
    const int num_series = ind.num_series();
    const int num_30s    = bar_end_30s - bar_start_30s;

    // Compute indicator on TF bars (full range to allow warmup)
    std::vector<float> tf_output(static_cast<size_t>(tf_bar_count) * num_series);
    ind.compute(tf_bars, 0, tf_bar_count, tf_output.data());

    // Broadcast TF-granularity results back to 30s alignment
    for (int s = 0; s < num_series; ++s) {
        for (int i = 0; i < num_30s; ++i) {
            int tf_idx = completed_map[bar_start_30s + i];
            if (tf_idx >= 0 && tf_idx < tf_bar_count) {
                output[s * num_30s + i] = tf_output[s * tf_bar_count + tf_idx];
            } else {
                output[s * num_30s + i] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}
