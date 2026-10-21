// =============================================================================
// Metrics Edge-Case Tests
//
// Tests covering: ISO week 52/53 boundary, profit_factor / payoff_ratio
// sentinel values for all-winner and no-trade scenarios.
// Framework: Google Test v1.14
// =============================================================================

#include <gtest/gtest.h>
#include "test_helpers.h"

#include <cstdint>
#include <vector>

#include "metrics.h"
#include "market_data.h"
#include "backtest_engine.h"
#include "cost_model.h"

// =============================================================================
// Helpers
// =============================================================================

// =============================================================================
// ISO Week 53 / 52 Tests
// =============================================================================

// ---------------------------------------------------------------------------
// ISOWeek53Year
// Years 2015 and 2020 have 53 ISO weeks. Verify that a trade on Dec 31 of
// such a year is bucketed into week 53 (not clamped to 52).
// ---------------------------------------------------------------------------
TEST(MetricsEdge, ISOWeek53Year_2015) {
    // Dec 31, 2015 = Thursday, ISO week 53 of 2015
    // Create two bars: one in early Dec (safely week 49-52), one on Dec 31.
    // Put a trade on Dec 31 and verify weekly bucketing produces a week key
    // with week == 53.

    // Dec 1, 2015 00:00 UTC
    const int64_t dec1_ts = 1448928000LL * NS_PER_SEC;
    // Dec 31, 2015 12:00 UTC
    const int64_t dec31_ts = 1451563200LL * NS_PER_SEC;

    std::vector<Bar> bars = {
        make_bar(dec1_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
        make_bar(dec31_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
    };

    // One trade exiting on bar 1 (Dec 31, 2015)
    std::vector<Trade> trades(1);
    trades[0].entry_bar      = 0;
    trades[0].exit_bar       = 1;
    trades[0].entry_price    = 4500.0f;
    trades[0].exit_price     = 4500.0f;
    trades[0].direction      = 1;
    trades[0].gross_pnl      = 100.0f;
    trades[0].net_pnl        = 100.0f;
    trades[0].min_unrealized = 0.0f;

    auto weeks = bucket_weekly_pnl(trades, bars.data(), 0, 2);

    // Dec 1 and Dec 31 are in different ISO weeks -> at least 2 buckets
    EXPECT_GE(weeks.size(), 2u) << "Early Dec and late Dec should be in different ISO weeks";

    // Isolate the Dec 31 week key by creating bars only on that date.
    {
        std::vector<Bar> bars2 = {
            make_bar(dec31_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
            make_bar(dec31_ts + 30LL * NS_PER_SEC, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
        };
        std::vector<Trade> trades2(1);
        trades2[0].entry_bar      = 0;
        trades2[0].exit_bar       = 1;
        trades2[0].entry_price    = 4500.0f;
        trades2[0].exit_price     = 4500.0f;
        trades2[0].direction      = 1;
        trades2[0].gross_pnl      = 100.0f;
        trades2[0].net_pnl        = 100.0f;
        trades2[0].min_unrealized = 0.0f;

        auto weeks2 = bucket_weekly_pnl(trades2, bars2.data(), 0, 2);
        // Should be exactly 1 bucket
        EXPECT_EQ(weeks2.size(), 1u);
    }

    // The definitive test: Jan 1, 2016 is a Friday.
    // In ISO 8601, the week containing Jan 4, 2016 (Monday) is week 1 of 2016.
    // So Jan 1, 2016 (Friday) belongs to ISO week 53 of 2015.
    // Create a bar on Jan 1, 2016 and verify it's bucketed to the same week
    // as Dec 31, 2015 (both are week 53 of 2015).
    {
        // Jan 1, 2016 12:00 UTC
        const int64_t jan1_2016_ts = 1451649600LL * NS_PER_SEC;

        std::vector<Bar> bars3 = {
            make_bar(dec31_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
            make_bar(jan1_2016_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
        };

        // Two trades, one on each day
        std::vector<Trade> trades3(2);
        trades3[0].entry_bar      = 0;
        trades3[0].exit_bar       = 0;
        trades3[0].entry_price    = 4500.0f;
        trades3[0].exit_price     = 4500.0f;
        trades3[0].direction      = 1;
        trades3[0].gross_pnl      = 50.0f;
        trades3[0].net_pnl        = 50.0f;
        trades3[0].min_unrealized = 0.0f;

        trades3[1].entry_bar      = 1;
        trades3[1].exit_bar       = 1;
        trades3[1].entry_price    = 4500.0f;
        trades3[1].exit_price     = 4500.0f;
        trades3[1].direction      = 1;
        trades3[1].gross_pnl      = 50.0f;
        trades3[1].net_pnl        = 50.0f;
        trades3[1].min_unrealized = 0.0f;

        auto weeks3 = bucket_weekly_pnl(trades3, bars3.data(), 0, 2);

        // Dec 31, 2015 (Thu) and Jan 1, 2016 (Fri) are both in ISO week 53 of 2015.
        // They should be bucketed into the same week, producing 1 bucket with PnL = 100.
        EXPECT_EQ(weeks3.size(), 1u)
            << "Dec 31 2015 and Jan 1 2016 should both be ISO week 53 of 2015";
        if (!weeks3.empty()) {
            EXPECT_NEAR(weeks3[0].pnl, 100.0f, 0.01f)
                << "Both trades should be summed into the same week 53 bucket";
        }
    }
}

TEST(MetricsEdge, ISOWeek53Year_2020) {
    // Dec 31, 2020 = Thursday, ISO week 53 of 2020
    // Jan 1, 2021 = Friday, also ISO week 53 of 2020
    const int64_t dec31_2020_ts = 1609416000LL * NS_PER_SEC;  // Dec 31, 2020 12:00 UTC
    const int64_t jan1_2021_ts  = 1609502400LL * NS_PER_SEC;  // Jan 1, 2021 12:00 UTC

    std::vector<Bar> bars = {
        make_bar(dec31_2020_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
        make_bar(jan1_2021_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
    };

    std::vector<Trade> trades(2);
    trades[0].entry_bar      = 0;
    trades[0].exit_bar       = 0;
    trades[0].entry_price    = 4500.0f;
    trades[0].exit_price     = 4500.0f;
    trades[0].direction      = 1;
    trades[0].gross_pnl      = 50.0f;
    trades[0].net_pnl        = 50.0f;
    trades[0].min_unrealized = 0.0f;

    trades[1].entry_bar      = 1;
    trades[1].exit_bar       = 1;
    trades[1].entry_price    = 4500.0f;
    trades[1].exit_price     = 4500.0f;
    trades[1].direction      = 1;
    trades[1].gross_pnl      = 50.0f;
    trades[1].net_pnl        = 50.0f;
    trades[1].min_unrealized = 0.0f;

    auto weeks = bucket_weekly_pnl(trades, bars.data(), 0, 2);

    // Both dates are in ISO week 53 of 2020 -> 1 bucket
    EXPECT_EQ(weeks.size(), 1u) << "Dec 31 2020 and Jan 1 2021 should both be ISO week 53 of 2020";
    if (!weeks.empty()) {
        EXPECT_NEAR(weeks[0].pnl, 100.0f, 0.01f);
    }
}

// ---------------------------------------------------------------------------
// ISOWeek52Year
// Normal 52-week years (2016, 2019). Verify week count does not exceed 52.
// ---------------------------------------------------------------------------
TEST(MetricsEdge, ISOWeek52Year) {
    // 2016: Jan 1 is Friday -> ISO week 53 of 2015, NOT week 53 of 2016.
    // 2016 itself has only 52 ISO weeks.
    // Dec 31, 2016 = Saturday -> belongs to ISO week 52 of 2016.

    // Dec 31, 2016 12:00 UTC
    const int64_t dec31_2016_ts = 1483185600LL * NS_PER_SEC;
    // Jan 2, 2017 12:00 UTC (Monday = start of week 1 of 2017)
    const int64_t jan2_2017_ts = 1483358400LL * NS_PER_SEC;

    std::vector<Bar> bars = {
        make_bar(dec31_2016_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
        make_bar(jan2_2017_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
    };

    std::vector<Trade> trades(2);
    trades[0].entry_bar      = 0;
    trades[0].exit_bar       = 0;
    trades[0].entry_price    = 4500.0f;
    trades[0].exit_price     = 4500.0f;
    trades[0].direction      = 1;
    trades[0].gross_pnl      = 50.0f;
    trades[0].net_pnl        = 50.0f;
    trades[0].min_unrealized = 0.0f;

    trades[1].entry_bar      = 1;
    trades[1].exit_bar       = 1;
    trades[1].entry_price    = 4500.0f;
    trades[1].exit_price     = 4500.0f;
    trades[1].direction      = 1;
    trades[1].gross_pnl      = 50.0f;
    trades[1].net_pnl        = 50.0f;
    trades[1].min_unrealized = 0.0f;

    auto weeks = bucket_weekly_pnl(trades, bars.data(), 0, 2);

    // Dec 31, 2016 (Sat) = ISO week 52 of 2016
    // Jan 2, 2017 (Mon) = ISO week 1 of 2017
    // These are different weeks -> 2 buckets
    EXPECT_EQ(weeks.size(), 2u)
        << "Dec 31 2016 (week 52 of 2016) and Jan 2 2017 (week 1 of 2017) are different weeks";
}

TEST(MetricsEdge, ISOWeek52Year_2019) {
    // 2019: Jan 1 is Tuesday -> week 1 of 2019.
    // Dec 31, 2019 = Tuesday -> ISO week 1 of 2020 (since Jan 1 2020 is Wed
    // and the week containing it starts Mon Dec 30 2019).
    // So 2019 has exactly 52 ISO weeks.

    // Dec 29, 2019 12:00 UTC (Sunday) = last day of ISO week 52 of 2019
    const int64_t dec29_2019_ts = 1577620800LL * NS_PER_SEC;
    // Dec 30, 2019 12:00 UTC (Monday) = start of ISO week 1 of 2020
    const int64_t dec30_2019_ts = 1577707200LL * NS_PER_SEC;

    std::vector<Bar> bars = {
        make_bar(dec29_2019_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
        make_bar(dec30_2019_ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f),
    };

    std::vector<Trade> trades(2);
    trades[0].entry_bar      = 0;
    trades[0].exit_bar       = 0;
    trades[0].entry_price    = 4500.0f;
    trades[0].exit_price     = 4500.0f;
    trades[0].direction      = 1;
    trades[0].gross_pnl      = 50.0f;
    trades[0].net_pnl        = 50.0f;
    trades[0].min_unrealized = 0.0f;

    trades[1].entry_bar      = 1;
    trades[1].exit_bar       = 1;
    trades[1].entry_price    = 4500.0f;
    trades[1].exit_price     = 4500.0f;
    trades[1].direction      = 1;
    trades[1].gross_pnl      = 50.0f;
    trades[1].net_pnl        = 50.0f;
    trades[1].min_unrealized = 0.0f;

    auto weeks = bucket_weekly_pnl(trades, bars.data(), 0, 2);

    // Dec 29 (Sun) = week 52 of 2019, Dec 30 (Mon) = week 1 of 2020
    // -> 2 distinct buckets
    EXPECT_EQ(weeks.size(), 2u)
        << "Dec 29 2019 (week 52 of 2019) and Dec 30 2019 (week 1 of 2020) are different weeks";
}

// =============================================================================
// Profit Factor / Payoff Ratio Edge Cases
// =============================================================================

// ---------------------------------------------------------------------------
// ProfitFactor_AllWinners
// All trades are winners -> no losses -> profit_factor should be 1e6f sentinel.
// ---------------------------------------------------------------------------
TEST(MetricsEdge, ProfitFactor_AllWinners) {
    const int        N = 5;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    std::vector<Trade> trades(5);
    for (int i = 0; i < 5; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = 275.0f;
        trades[i].net_pnl        = 200.0f;  // all positive
        trades[i].min_unrealized = 0.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    EXPECT_GE(result.profit_factor, 1e6f)
        << "profit_factor should be 1e6 sentinel when all trades are winners";
}

// ---------------------------------------------------------------------------
// PayoffRatio_AllWinners
// Same setup: all winners -> payoff_ratio should be 1e6f sentinel.
// ---------------------------------------------------------------------------
TEST(MetricsEdge, PayoffRatio_AllWinners) {
    const int        N = 5;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    std::vector<Trade> trades(5);
    for (int i = 0; i < 5; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = 275.0f;
        trades[i].net_pnl        = 200.0f;
        trades[i].min_unrealized = 0.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    EXPECT_GE(result.payoff_ratio, 1e6f)
        << "payoff_ratio should be 1e6 sentinel when all trades are winners";
}

// ---------------------------------------------------------------------------
// ProfitFactor_NoTrades
// No trades -> profit_factor should be 0.0f (default).
// ---------------------------------------------------------------------------
TEST(MetricsEdge, ProfitFactor_NoTrades) {
    const int        N = 10;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    std::vector<Trade> trades;  // empty
    CostModel          cost   = es_cost();
    BacktestResult     result = compute_metrics(trades, bars.data(), 0, N, cost);

    EXPECT_FLOAT_EQ(result.profit_factor, 0.0f)
        << "profit_factor should be 0.0 when there are no trades";
    EXPECT_FLOAT_EQ(result.payoff_ratio, 0.0f)
        << "payoff_ratio should be 0.0 when there are no trades";
}

// ---------------------------------------------------------------------------
// ProfitFactor_Normal
// Normal case with both winners and losers. PF = gross_wins / |gross_losses|.
// ---------------------------------------------------------------------------
TEST(MetricsEdge, ProfitFactor_Normal) {
    const int        N = 8;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    // 5 winning trades at $100 each, 3 losing trades at -$100 each
    // gross_wins = 500, gross_losses = -300, PF = 500/300 = 1.6667
    std::vector<Trade> trades(8);
    for (int i = 0; i < 8; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].min_unrealized = 0.0f;
        if (i < 5) {
            trades[i].gross_pnl = 175.0f;
            trades[i].net_pnl   = 100.0f;
        } else {
            trades[i].gross_pnl = -25.0f;
            trades[i].net_pnl   = -100.0f;
        }
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // PF = 500 / 300 = 1.6667
    EXPECT_NEAR(result.profit_factor, 500.0f / 300.0f, 1e-4f)
        << "profit_factor should be gross_wins / |gross_losses|";

    // Payoff ratio = avg_win / |avg_loss| = (500/5) / (300/3) = 100/100 = 1.0
    EXPECT_NEAR(result.payoff_ratio, 1.0f, 1e-4f) << "payoff_ratio should be avg_win / |avg_loss|";
}

// ---------------------------------------------------------------------------
// ProfitFactor_AllLosers
// All trades are losers -> sum_wins=0, profit_factor=0.0, payoff_ratio=0.0
// ---------------------------------------------------------------------------
TEST(MetricsEdge, ProfitFactor_AllLosers) {
    const int        N = 3;
    std::vector<Bar> bars(N);
    for (int i = 0; i < N; ++i) {
        int64_t ts = (1'262'304'000LL + static_cast<int64_t>(i) * 86400LL) * NS_PER_SEC;
        bars[i]    = make_bar(ts, 4500.0f, 4501.0f, 4499.0f, 4500.0f);
    }

    std::vector<Trade> trades(3);
    for (int i = 0; i < 3; ++i) {
        trades[i].entry_bar      = i;
        trades[i].exit_bar       = i;
        trades[i].entry_price    = 4500.0f;
        trades[i].exit_price     = 4500.0f;
        trades[i].direction      = 1;
        trades[i].gross_pnl      = -50.0f;
        trades[i].net_pnl        = -100.0f;
        trades[i].min_unrealized = -100.0f;
    }

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, N, cost);

    // sum_wins = 0, abs_losses > 0 -> PF = 0/abs_losses = 0
    EXPECT_FLOAT_EQ(result.profit_factor, 0.0f)
        << "profit_factor should be 0 when all trades are losers";
    EXPECT_FLOAT_EQ(result.payoff_ratio, 0.0f)
        << "payoff_ratio should be 0 when all trades are losers";
}

// =============================================================================
// Zero-PnL weeks are NOT green (only strictly positive weeks are green)
// =============================================================================

// ---------------------------------------------------------------------------
// GreenWeeksPct_ZeroPnLWeeksAreNotGreen
// A week with zero PnL should NOT count as green; only strict positive counts.
// This prevents inflated consistency metrics for flat/inactive strategies.
// ---------------------------------------------------------------------------
TEST(MetricsEdge, GreenWeeksPct_ZeroPnLWeeksAreNotGreen) {
    // Create bars spanning 3 distinct ISO weeks.
    // Week 1: Mon Jan 5 2015, Week 2: Mon Jan 12 2015, Week 3: Mon Jan 19 2015
    // We place one trade per week:
    //   Week 1: +100 (green)
    //   Week 2:    0 (zero PnL -- not green)
    //   Week 3: -100 (red)
    // Expected green_weeks_pct = 1/3 * 100 = 33.33%

    const int64_t week1_ts = 1420416000LL * NS_PER_SEC;  // Mon Jan 5 2015 00:00 UTC
    const int64_t week2_ts = 1421020800LL * NS_PER_SEC;  // Mon Jan 12 2015 00:00 UTC
    const int64_t week3_ts = 1421625600LL * NS_PER_SEC;  // Mon Jan 19 2015 00:00 UTC

    // Two bars per week (entry + exit)
    std::vector<Bar> bars = {
        make_bar(week1_ts, 4500, 4501, 4499, 4500),
        make_bar(week1_ts + 30LL * NS_PER_SEC, 4500, 4501, 4499, 4500),
        make_bar(week2_ts, 4500, 4501, 4499, 4500),
        make_bar(week2_ts + 30LL * NS_PER_SEC, 4500, 4501, 4499, 4500),
        make_bar(week3_ts, 4500, 4501, 4499, 4500),
        make_bar(week3_ts + 30LL * NS_PER_SEC, 4500, 4501, 4499, 4500),
    };

    std::vector<Trade> trades(3);
    // Week 1: winning trade (+100)
    trades[0].entry_bar      = 0;
    trades[0].exit_bar       = 1;
    trades[0].entry_price    = 4500.0f;
    trades[0].exit_price     = 4502.0f;
    trades[0].direction      = 1;
    trades[0].gross_pnl      = 175.0f;
    trades[0].net_pnl        = 100.0f;
    trades[0].min_unrealized = 0.0f;
    trades[0].max_unrealized = 100.0f;

    // Week 2: zero PnL trade (exactly 0)
    trades[1].entry_bar      = 2;
    trades[1].exit_bar       = 3;
    trades[1].entry_price    = 4500.0f;
    trades[1].exit_price     = 4500.0f;
    trades[1].direction      = 1;
    trades[1].gross_pnl      = 0.0f;
    trades[1].net_pnl        = 0.0f;
    trades[1].min_unrealized = 0.0f;
    trades[1].max_unrealized = 0.0f;

    // Week 3: losing trade (-100)
    trades[2].entry_bar      = 4;
    trades[2].exit_bar       = 5;
    trades[2].entry_price    = 4500.0f;
    trades[2].exit_price     = 4498.0f;
    trades[2].direction      = 1;
    trades[2].gross_pnl      = -25.0f;
    trades[2].net_pnl        = -100.0f;
    trades[2].min_unrealized = -100.0f;
    trades[2].max_unrealized = 0.0f;

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, 6, cost);

    // 3 weeks total: 1 green (week 1 positive), 2 non-green (week 2 zero, week 3 red)
    // green_weeks_pct = 1/3 * 100 = 33.33%
    EXPECT_NEAR(result.green_weeks_pct, 33.33f, 0.5f)
        << "Only strictly positive weeks count as green";
    EXPECT_EQ(result.max_red_weeks, 2)
        << "Week 2 (zero) and week 3 (negative) are consecutive non-green";
}

// ---------------------------------------------------------------------------
// GreenWeeksPct_AllZeroPnL
// If all weeks have zero PnL (no-trade weeks), green_weeks_pct should be 0%.
// ---------------------------------------------------------------------------
TEST(MetricsEdge, GreenWeeksPct_AllZeroPnL) {
    // Create bars spanning 2 weeks, one trade with net_pnl = 0 in week 1,
    // no trades in week 2 (which gets 0 PnL from the no-trade bucketing).
    const int64_t week1_ts = 1420416000LL * NS_PER_SEC;  // Mon Jan 5 2015
    const int64_t week2_ts = 1421020800LL * NS_PER_SEC;  // Mon Jan 12 2015

    std::vector<Bar> bars = {
        make_bar(week1_ts, 4500, 4501, 4499, 4500),
        make_bar(week1_ts + 30LL * NS_PER_SEC, 4500, 4501, 4499, 4500),
        make_bar(week2_ts, 4500, 4501, 4499, 4500),
        make_bar(week2_ts + 30LL * NS_PER_SEC, 4500, 4501, 4499, 4500),
    };

    // One trade with exactly zero PnL
    std::vector<Trade> trades(1);
    trades[0].entry_bar      = 0;
    trades[0].exit_bar       = 1;
    trades[0].entry_price    = 4500.0f;
    trades[0].exit_price     = 4500.0f;
    trades[0].direction      = 1;
    trades[0].gross_pnl      = 0.0f;
    trades[0].net_pnl        = 0.0f;
    trades[0].min_unrealized = 0.0f;
    trades[0].max_unrealized = 0.0f;

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, 4, cost);

    // Both weeks have PnL == 0, neither is green (only strict positive counts)
    EXPECT_NEAR(result.green_weeks_pct, 0.0f, 0.01f)
        << "Zero-PnL weeks should not be counted as green";
    EXPECT_EQ(result.max_red_weeks, 2)
        << "Both zero-PnL weeks are non-green, forming a streak of 2";
}

// ---------------------------------------------------------------------------
// GreenMonthsPct_ZeroPnLMonthsAreNotGreen
// Same principle applies to monthly bucketing.
// ---------------------------------------------------------------------------
TEST(MetricsEdge, GreenMonthsPct_ZeroPnLMonthsAreNotGreen) {
    // Create bars spanning 2 months: Jan 2015 and Feb 2015.
    // Trade in Jan with +100, trade in Feb with 0.
    const int64_t jan_ts = 1420416000LL * NS_PER_SEC;  // Jan 5 2015
    const int64_t feb_ts = 1423094400LL * NS_PER_SEC;  // Feb 5 2015

    std::vector<Bar> bars = {
        make_bar(jan_ts, 4500, 4501, 4499, 4500),
        make_bar(jan_ts + 30LL * NS_PER_SEC, 4500, 4501, 4499, 4500),
        make_bar(feb_ts, 4500, 4501, 4499, 4500),
        make_bar(feb_ts + 30LL * NS_PER_SEC, 4500, 4501, 4499, 4500),
    };

    std::vector<Trade> trades(2);
    // Jan: winning trade
    trades[0].entry_bar      = 0;
    trades[0].exit_bar       = 1;
    trades[0].entry_price    = 4500.0f;
    trades[0].exit_price     = 4502.0f;
    trades[0].direction      = 1;
    trades[0].gross_pnl      = 175.0f;
    trades[0].net_pnl        = 100.0f;
    trades[0].min_unrealized = 0.0f;
    trades[0].max_unrealized = 100.0f;

    // Feb: zero PnL trade
    trades[1].entry_bar      = 2;
    trades[1].exit_bar       = 3;
    trades[1].entry_price    = 4500.0f;
    trades[1].exit_price     = 4500.0f;
    trades[1].direction      = 1;
    trades[1].gross_pnl      = 0.0f;
    trades[1].net_pnl        = 0.0f;
    trades[1].min_unrealized = 0.0f;
    trades[1].max_unrealized = 0.0f;

    CostModel      cost   = es_cost();
    BacktestResult result = compute_metrics(trades, bars.data(), 0, 4, cost);

    // Jan is green (positive), Feb is not green (zero PnL)
    EXPECT_NEAR(result.green_months_pct, 50.0f, 0.01f)
        << "Only strictly positive months count as green";
    EXPECT_EQ(result.negative_months, 0)
        << "Feb has zero PnL (not negative), so negative_months stays 0";
}
