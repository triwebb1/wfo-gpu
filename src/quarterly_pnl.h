#pragma once
// ---------------------------------------------------------------------------
// QuarterlyPnL -- PnL bucketed by calendar quarter
//
// One calendar quarter of a strategy's trades (detail_runner.h).
// ---------------------------------------------------------------------------

struct QuarterlyPnL {
    int   year        = 0;
    int   quarter     = 0;  // 1-4
    float pnl         = 0.0f;
    int   trade_count = 0;
};
