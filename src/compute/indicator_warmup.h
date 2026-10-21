#pragma once
// ---------------------------------------------------------------------------
// Indicator warmup -- how many bars of history an indicator needs before
// its values are trustworthy.
//
// Each stage loads this many bars before the first evaluated bar, so every
// signal is converged when evaluation starts. The figures are the
// recommended warmup (3x the period for exponential smoothing, the period
// for window-based indicators) with a 2x safety factor, in base bars.
// Anchored indicators are scaled from their timeframe assuming 30-second
// base bars.
// ---------------------------------------------------------------------------

#include <vector>

#include "../grids/indicator_grid.h"

struct IndicatorSpec;

// Largest warmup over a parameter list, an indicator grid, or a list of
// indicator specs (the last is anchored-aware).
int max_indicator_warmup(const std::vector<IndicatorParams>& params_list);
int max_indicator_warmup(const IndicatorGrid& grid);
int max_indicator_warmup(const std::vector<IndicatorSpec>& specs);
