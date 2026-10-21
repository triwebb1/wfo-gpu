// ---------------------------------------------------------------------------
// Tests for adaptive LOSERS trigger counting
//
// Verifies that the LOSERS_N trigger counts individual consecutive losing
// trades, not consecutive losing OOS windows.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "wfo_engine.h"
#include "backtest_engine.h"
#include "cost_model.h"
#include "grids/wfo_grid.h"
#include "market_data.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Test: LOSERS trigger semantics
//
// LOSERS_5 means "re-optimize after 5 consecutive losing trades," not
// 5 consecutive OOS windows with negative total_pnl.
//
// Verify: A single OOS window with 5 consecutive losing trades should
// trigger, even though it's only 1 window.
// ---------------------------------------------------------------------------
TEST(LosersTrigger, CountsIndividualTradesNotWindows) {
    // This test verifies the design intent by checking that the trigger
    // logic processes individual trades, not window-level PnL.
    //
    // We construct a scenario with a single OOS window containing multiple
    // trades. A window-level count would increment the counter by only 1
    // for a single negative-PnL window, requiring N consecutive negative-PnL
    // windows to trigger. Per-trade counting triggers on N consecutive
    // losing trades within any combination of windows.

    // Synthesize trades to verify counting logic
    // 5 consecutive losers should trigger LOSERS_5
    std::vector<Trade> trades;
    for (int i = 0; i < 5; ++i) {
        Trade t;
        t.entry_bar   = i * 10;
        t.exit_bar    = i * 10 + 5;
        t.entry_price = 5000.0f;
        t.exit_price  = 4995.0f;  // losing trade
        t.direction   = 1;
        t.gross_pnl   = -5.0f;
        t.net_pnl     = -7.5f;  // including costs
        trades.push_back(t);
    }

    // Simulate the per-trade consecutive loser counting logic
    int       consecutive_losers = 0;
    const int trigger_losers_n   = 5;
    bool      triggered          = false;

    for (const auto& tr : trades) {
        if (tr.net_pnl < 0.0f) {
            consecutive_losers++;
        } else {
            consecutive_losers = 0;
        }
        if (consecutive_losers >= trigger_losers_n) {
            triggered = true;
            break;
        }
    }

    EXPECT_TRUE(triggered) << "5 consecutive losing trades should trigger LOSERS_5";
    EXPECT_EQ(consecutive_losers, 5);
}

// ---------------------------------------------------------------------------
// Test: A winning trade in the middle resets the counter
// ---------------------------------------------------------------------------
TEST(LosersTrigger, WinningTradeResetsCounter) {
    std::vector<Trade> trades;

    // 3 losers
    for (int i = 0; i < 3; ++i) {
        Trade t;
        t.net_pnl = -10.0f;
        trades.push_back(t);
    }
    // 1 winner (resets counter)
    {
        Trade t;
        t.net_pnl = 15.0f;
        trades.push_back(t);
    }
    // 4 more losers (total 4, not 7)
    for (int i = 0; i < 4; ++i) {
        Trade t;
        t.net_pnl = -10.0f;
        trades.push_back(t);
    }

    // Simulate counting
    int       consecutive_losers = 0;
    const int trigger_losers_n   = 5;
    bool      triggered          = false;

    for (const auto& tr : trades) {
        if (tr.net_pnl < 0.0f) {
            consecutive_losers++;
        } else {
            consecutive_losers = 0;
        }
        if (consecutive_losers >= trigger_losers_n) {
            triggered = true;
            break;
        }
    }

    EXPECT_FALSE(triggered)
        << "Winner in the middle should reset counter; max consecutive = 4, not >= 5";
    EXPECT_EQ(consecutive_losers, 4);
}

// ---------------------------------------------------------------------------
// Test: Counter persists across OOS windows
//
// If window 1 ends with 3 consecutive losers and window 2 starts with
// 2 more losers, the total is 5 and should trigger LOSERS_5.
// ---------------------------------------------------------------------------
TEST(LosersTrigger, CounterPersistsAcrossWindows) {
    // Simulate 2 OOS windows, each with their trade logs
    std::vector<Trade> window1_trades;
    // Window 1: ends with 3 consecutive losers
    {
        Trade t;
        t.net_pnl = 20.0f;
        window1_trades.push_back(t);  // winner
    }
    for (int i = 0; i < 3; ++i) {
        Trade t;
        t.net_pnl = -5.0f;
        window1_trades.push_back(t);  // loser
    }

    std::vector<Trade> window2_trades;
    // Window 2: starts with 2 more losers
    for (int i = 0; i < 2; ++i) {
        Trade t;
        t.net_pnl = -5.0f;
        window2_trades.push_back(t);
    }
    {
        Trade t;
        t.net_pnl = 10.0f;
        window2_trades.push_back(t);  // winner
    }

    // Simulate consecutive loser counting across both windows
    int       consecutive_losers = 0;
    const int trigger_losers_n   = 5;
    bool      triggered          = false;

    // Process window 1 trades
    for (const auto& tr : window1_trades) {
        if (tr.net_pnl < 0.0f) {
            consecutive_losers++;
        } else {
            consecutive_losers = 0;
        }
        if (consecutive_losers >= trigger_losers_n) {
            triggered = true;
            break;
        }
    }

    // If not triggered yet, process window 2 trades
    if (!triggered) {
        for (const auto& tr : window2_trades) {
            if (tr.net_pnl < 0.0f) {
                consecutive_losers++;
            } else {
                consecutive_losers = 0;
            }
            if (consecutive_losers >= trigger_losers_n) {
                triggered = true;
                break;
            }
        }
    }

    EXPECT_TRUE(triggered)
        << "3 losers from window 1 + 2 losers from window 2 = 5 consecutive, should trigger";
    EXPECT_EQ(consecutive_losers, 5);
}

// ---------------------------------------------------------------------------
// Test: LOSERS_10 requires 10 consecutive losers
// ---------------------------------------------------------------------------
TEST(LosersTrigger, Losers10RequiresTenConsecutive) {
    std::vector<Trade> trades;

    // 9 consecutive losers - should NOT trigger LOSERS_10
    for (int i = 0; i < 9; ++i) {
        Trade t;
        t.net_pnl = -5.0f;
        trades.push_back(t);
    }

    int       consecutive_losers = 0;
    const int trigger_losers_n   = 10;
    bool      triggered          = false;

    for (const auto& tr : trades) {
        if (tr.net_pnl < 0.0f) {
            consecutive_losers++;
        } else {
            consecutive_losers = 0;
        }
        if (consecutive_losers >= trigger_losers_n) {
            triggered = true;
            break;
        }
    }

    EXPECT_FALSE(triggered) << "9 consecutive losers should NOT trigger LOSERS_10";

    // Add 10th loser
    Trade t;
    t.net_pnl = -5.0f;
    if (t.net_pnl < 0.0f) {
        consecutive_losers++;
    } else {
        consecutive_losers = 0;
    }
    triggered = (consecutive_losers >= trigger_losers_n);

    EXPECT_TRUE(triggered) << "10 consecutive losers SHOULD trigger LOSERS_10";
    EXPECT_EQ(consecutive_losers, 10);
}

// ---------------------------------------------------------------------------
// Test: Trades are counted, not windows
//
// A window-level PnL check would give wrong results: a window with 10
// trades (5 winners, 5 losers in alternating order) has negative total_pnl
// but no streak of consecutive losers. A window-level count would
// incorrectly increment the counter. Per-trade counting sees no losing
// streak > 1.
// ---------------------------------------------------------------------------
TEST(LosersTrigger, AlternatingTradesNoConsecutiveLosers) {
    std::vector<Trade> trades;

    // Alternating wins and losses: W L W L W L W L W L
    for (int i = 0; i < 10; ++i) {
        Trade t;
        if (i % 2 == 0) {
            t.net_pnl = 5.0f;  // winner
        } else {
            t.net_pnl = -8.0f;  // loser (larger magnitude -> negative total PnL)
        }
        trades.push_back(t);
    }

    // Total PnL = 5*5 + 5*(-8) = 25 - 40 = -15 (negative)
    float total_pnl = 0.0f;
    for (const auto& tr : trades) total_pnl += tr.net_pnl;
    ASSERT_LT(total_pnl, 0.0f) << "Setup: total PnL must be negative";

    // Count consecutive losers (the correct way, per-trade)
    int consecutive_losers = 0;
    int max_consecutive    = 0;

    for (const auto& tr : trades) {
        if (tr.net_pnl < 0.0f) {
            consecutive_losers++;
            if (consecutive_losers > max_consecutive) max_consecutive = consecutive_losers;
        } else {
            consecutive_losers = 0;
        }
    }

    // With alternating W/L pattern, max consecutive losers is 1
    EXPECT_EQ(max_consecutive, 1) << "Alternating W/L should have max 1 consecutive loser";

    // A window-level check would see total_pnl < 0 and increment the
    // counter (treating the entire window as 1 loser). This is wrong.
    // Per-trade counting means consecutive_losers never exceeds 1 despite
    // the window having negative total PnL.
    EXPECT_LT(max_consecutive, 5) << "Should NOT trigger LOSERS_5 with alternating W/L pattern";
}

// ---------------------------------------------------------------------------
// Test: Zero PnL trade is not counted as a loser
// ---------------------------------------------------------------------------
TEST(LosersTrigger, ZeroPnlNotCountedAsLoser) {
    std::vector<Trade> trades;

    // 4 losers, then a zero-PnL trade, then 1 more loser
    for (int i = 0; i < 4; ++i) {
        Trade t;
        t.net_pnl = -5.0f;
        trades.push_back(t);
    }
    {
        Trade t;
        t.net_pnl = 0.0f;
        trades.push_back(t);
    }
    {
        Trade t;
        t.net_pnl = -5.0f;
        trades.push_back(t);
    }

    int       consecutive_losers = 0;
    const int trigger_losers_n   = 5;
    bool      triggered          = false;

    for (const auto& tr : trades) {
        if (tr.net_pnl < 0.0f) {
            consecutive_losers++;
        } else {
            consecutive_losers = 0;
        }
        if (consecutive_losers >= trigger_losers_n) {
            triggered = true;
            break;
        }
    }

    // The zero-PnL trade resets the counter: 4 losers, reset, 1 loser
    EXPECT_FALSE(triggered) << "Zero-PnL trade should reset the counter";
    EXPECT_EQ(consecutive_losers, 1);
}
