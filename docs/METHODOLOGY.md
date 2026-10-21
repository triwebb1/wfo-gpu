# Methodology

This document describes the backtesting rules wfo-gpu enforces and the reasoning behind each one. Every rule below is implemented in the source under `src/` and exercised by tests under `tests/`. Where a rule has a known weakness, it is listed in [Known limitations](#10-known-limitations).

## 1. Principle

A backtest is informative only if its results would hold in live trading. Backtests most commonly fail in three ways: they use information that was not yet available (lookahead), they select the best of many parameter sets by chance (overfitting), or they understate trading costs.

wfo-gpu is built to make each of these hard to do by accident:

- Fills happen strictly after the bar that produced the signal.
- Every trade pays spread and slippage.
- Parameter sets must clear sanity floors, a cost-based viability floor, and a neighborhood stability test before they reach walk-forward.
- Walk-forward results are judged only on out-of-sample bars, stitched into one equity curve.
- Every walk-forward winner is re-run with one extra bar of delay.

## 2. No lookahead

### Signal timing

A signal computed on bar `N` may use bar `N`'s close and anything earlier. The earliest possible fill is the open of bar `N + entry_lag`.

```
signal on bar N  ->  entry fills at open[N + entry_lag]
```

`entry_lag = 1` is the standard setting. The engine asserts `entry_lag >= 1`, so same-bar entry is impossible. The `[screen] entry_lag` key accepts 1 or 2. A signal is ignored if `N + entry_lag` falls past the end of the evaluation range.

### Higher-timeframe ("anchored") indicators

The indicator layer can compute an indicator on aggregated bars (1, 2, 5, 15, 30 or 60 minutes) and broadcast it back to the base bars. Each timeframe keeps two maps from base bar to aggregated bar:

- `parent_map[i]`: the aggregated bar that contains base bar `i`. That bar is still forming while bar `i` is open, so its OHLC includes later base bars.
- `completed_map[i]`: the latest aggregated bar that is complete at the close of base bar `i`, or `-1` if none exists yet.

Broadcasting uses `completed_map` only. Inside an aggregated block, base bars see the latest finished block, never the one in progress.

### Session signals

Session-level signals (opening range, gaps, overnight context, and similar) are computed per session day and then broadcast to bars. Each signal series declares the time of day after which its value exists. Before that time the broadcast value is `0.0` (neutral). An opening-range direction signal, for example, reads as neutral on every bar until the opening range has finished.

"Before" is measured as time elapsed since the session open, not as clock time. A US futures session opens at 18:00 the evening before its trading day, so its evening bars have a later clock time than a 16:00 availability but happen first; they stay gated. The same rule is available directly in the engine through `SignalTimeGate`.

Signals that only exist once core hours are over (level interactions, some opening-range and intraday-structure outputs, all available from 16:00) describe a whole regular session. They are meant for the next session, so until a session's own value exists its bars see the previous session's value instead of zero. A halt in the source session still suppresses them.

The session signals are written for the 09:30-16:00 ET equity-index session. A config that sets other core hours turns them off rather than feed them the wrong windows.

### Tests that enforce it

The lookahead tests (`tests/test_lookahead.cpp`, plus the `NoLookahead` cases in `tests/test_session_signals.cpp`) build two datasets that are identical up to a cut bar and differ afterwards. They recompute the full production signal set (every technical indicator, normalized, and every gated session signal) with the cut placed in the evening, overnight and during regular hours, and require each output at or before the cut to match bit for bit. Any read of a future bar shows up as a mismatch. A separate canary test confirms that a deliberately non-causal signal (one that reads the next bar) produces implausible results, which shows that the harness detects the effect it guards against.

## 3. Lag-2 re-test

A lag-1 backtest assumes you can act on bar `N`'s close at bar `N + 1`'s open. On short bars, much of a signal's apparent edge can sit in that close-to-open transition, which in practice requires near-instant order placement.

For every walk-forward window, the engine runs the selected winner on the out-of-sample bars twice: once at lag 1 and once at lag 2 (signal on `N`, fill at `open[N + 2]`). Both results are reported per window.

If the lag-1 Sharpe is positive and the lag-2 Sharpe falls below half of it, the engine also decomposes the lag-2 trades' PnL:

```
gap_pnl  = direction * (open[fill_bar]  - close[fill_bar - 1])
body_pnl = direction * (close[fill_bar] - open[fill_bar])
```

A body share above 80% suggests the signal predicts the fill bar's movement (a real short-horizon edge). A gap share above 50% suggests the PnL lives in the inter-bar gap and is unlikely to be capturable.

The lag-2 run is a diagnostic. It does not veto a winner; it tells you how fragile the winner is to one bar of delay.

## 4. Cost model

Each instrument is described by:

| Field | Meaning |
|---|---|
| `tick_size` | Minimum price increment, in price units |
| `tick_value` | Currency value of one tick, per contract |
| `point_value` | Currency value of a 1.00 price move (defaults to `tick_value / tick_size`) |
| `spread_ticks` | Round-trip bid/ask spread, in ticks |
| `slippage_ticks` | Extra slippage per side, in ticks |
| `capital` | Capital per contract; the denominator for return on capital |

Every closed trade, regardless of how it exits, pays:

```
round_trip_cost = (spread_ticks + 2 * slippage_ticks) * tick_value
net_pnl         = direction * (exit_price - entry_price) * point_value - round_trip_cost
```

Example for an equity index future with tick 0.25 and tick value 12.50 (point value 50.00), a 2-tick spread and 1 tick of slippage per side:

```
round_trip_cost = (2 + 2 * 1) * 12.50 = 50.00
long 4.00 points: gross = 4.00 * 50.00 = 200.00, net = 150.00
```

Invalid models (non-positive tick size, tick value, point value or capital, negative spread or slippage) are rejected before a run starts.

## 5. Fills and exits

**One position at a time.** The engine only evaluates entries while flat with no pending entry or exit. It never reverses in one step; a reversal is an exit followed by a fresh signal and a fresh entry.

**Signal-driven exits fill at the next open.** Signal reversal, opposing signal and bar timeout exits are detected on bar `N` and filled at `open[N + 1]`, like a market order sent after the bar closes.

**Level exits are resting orders.** Trailing stop, maximum-loss stop, take-profit, and combined take-profit/stop-loss trigger when the bar's low or high reaches the level.

- Stops fill at the level, or at the bar's open if the bar opens through the level. A gap through a stop costs what it would cost live.
- Targets fill at their level.
- Distances are multiples of an ATR. Unless a strategy supplies its own series, the engine computes a Wilder `ATR(14)` over the evaluated range from completed bars only: the stop on bar `N` uses the ATR as of bar `N - 1`. Level exits stay inactive until 14 bars of the range have closed.
- The trailing stop trails the best high (long) or low (short) of the bars *before* the bar being tested. Bar `N`'s own extreme moves the stop only from bar `N + 1`.
- Level exits are disabled on the entry bar, so no trade opens and closes on intra-bar extremes of one bar.

**Conservative same-bar rule.** When a bar touches both the stop and the target, the stop wins. Bar data cannot say which came first, so the engine assumes the worse outcome.

**Excursion tracking.** While a position is open, the engine marks each bar's worst and best intra-bar price against the entry (low and high for longs, high and low for shorts), including the bar on which a signal exit fills. These feed the drawdown and trade recovery metrics.

**Daily close.** For a session model with a daily close and `flat_at_close` set, the last bar before the close is flagged: the bar in the final slot before the close, or, when that bar is missing, the last bar before the session ends or the close passes. An open position exits at that bar's close, like a market-on-close order, so no position is carried through the daily break. A signal exit raised on that bar would otherwise fill at the next session's open, so it becomes the close fill too; only a stop or target that already filled inside the bar takes precedence. Bars between the close and the next open (after an early close, for example) take no new entries, including an entry signalled on the close bar. Continuous markets have no daily close and nothing is force-closed.

**Early closes.** The bundled calendar (`config/cme_early_closes.csv`) lists shortened sessions by close date: the day before Independence Day, Thanksgiving and the day after, Christmas Eve, and Good Friday. It approximates the CME equity-index schedule, whose exact times vary by year; check it against the exchange's published hours before relying on early-close behaviour.

**End of range.** Any position still open at the last bar of an evaluation range (a screen range, an in-sample window, or an out-of-sample window) is closed at that bar's close and pays full costs. Nothing is silently dropped at a boundary. Ranges are always evaluated in one pass, never split.

**Trading-hours filters.** Session flags mark each bar as inside core hours, the morning half, or the afternoon half. A strategy's `trading_hours` setting (`ETH`, `RTH`, `MORNING`, `AFTERNOON`) restricts the bars on which a new entry signal may fire. It does not force exits. A gap of more than `halt_gap` between two consecutive core-hours bars marks a halt: bars from the gap to the end of that session allow no new entries. Bars before the gap are unaffected, because the halt is not yet observable when they trade.

**GPU parity.** The OpenCL kernel (`src/compute/kernels/backtest.cl`) implements the same entry, exit, cost and session rules as the CPU engine, including the exit ATR. Tests run identical strategies through both paths, across exit modes, trading-hours filters, lag 2, daily closes and early-close sessions, and require matching results. Indicators are computed on the host by default, so both paths see the same signals. Where the pipeline needs metrics only the CPU engine produces (Sharpe, weekly and monthly consistency), it re-runs the GPU's viable and floor-passing results on the CPU and uses those results, and every out-of-sample run uses the CPU engine. End-to-end tests require the screen's reduced set and every walk-forward run to be the same on an OpenCL device and on the native path.

## 6. Metrics

All PnL figures are net of costs.

**Return on capital (primary).** `annual_roc` is the compound annual growth rate of the configured capital over the evaluation period:

```
annual_roc = ((capital + total_pnl) / capital) ^ (1 / years) - 1      (as a percent)
years      = (last bar time - first bar time) / 365.25 days
```

If the account is wiped out (`capital + total_pnl <= 0`), `annual_roc` is -100. Raw PnL is a poor ranking metric: it ignores how much capital the result needed and how long it took, so it cannot compare a short in-sample window to a long one or one instrument to another. Return on capital normalizes both. `capital` should be the amount needed to survive the strategy's drawdowns, not exchange margin.

**Sharpe ratio.** Trade PnL is bucketed into session days by exit bar. Every session day in the range is included, with zero for days without a closed trade. The ratio is `mean(daily_pnl) / stdev(daily_pnl) * sqrt(252)` using sample standard deviation and no risk-free rate.

**Consistency.** Trades are bucketed by exit bar into ISO weeks, months and calendar years of the session's close date (a session that opens Sunday evening belongs to Monday); periods with no trades count as zero.

- `green_weeks_pct`, `green_months_pct`: share of periods with PnL above zero.
- `max_red_weeks`: longest run of weeks with PnL at or below zero.
- `negative_years`, `negative_months`: count of periods with PnL below zero.

These metrics count periods rather than sum PnL, which makes them resistant to a single outsized trade: one large winner can turn at most one week positive and cannot make a strategy appear consistent. For this reason the robustness score and the Pareto frontier rely on them.

**Drawdowns.**

- `max_drawdown`: largest peak-to-trough decline of the closed-trade equity curve, starting from zero.
- `max_live_drawdown`: the worst mark-to-market loss reached inside any single trade, from intra-bar extremes. It shows how deep an open position can go before it is closed.
- `avg_trade_dd`: the average of each trade's worst mark-to-market loss.

**Trade recovery ratio.** How much a typical trade earns relative to how far it typically goes underwater first:

```
trade_recovery_ratio = (avg_win * win_rate) / avg_trade_dd
```

If no trade ever went underwater, the ratio is set to 100.

**Other reported metrics.** Win rate, average win and loss, profit factor, payoff ratio, trades per week, average holding time, Calmar ratio (annualized PnL over `max_drawdown`), and time to double at the measured growth rate.

**Viability floor.** A result is `viable` when its average net trade covers twice the round-trip cost:

```
total_pnl / total_trades >= 2 * round_trip_cost
```

Because `total_pnl` is already net, a viable strategy earns at least three round-trip costs per trade gross. Strategies that barely clear costs in a backtest rarely clear them live, where fills are worse.

## 7. Screen gates

The screen sweeps the strategy grid over the `[screen]` range. Grid entries that trade identically to another entry are skipped: vote-based aggregations ignore weights and entry thresholds, filter-role indicators ignore their weight, and a weighted sum ignores the overall scale of its weights. A property test checks that every skipped entry produces the same trades as the entry kept in its place. `max_strategies` counts distinct strategies evaluated.

Results pass through these gates in order.

**Sanity floors.** Defaults (configurable under `[floors]`):

| Floor | Default |
|---|---|
| `total_pnl` | `> 0` |
| `total_trades` | `>= 100` |
| `profit_factor` | `> 1.0` |
| `annual_roc` | `> 0` |
| `negative_years` | `== 0` |

**Viability.** Only sanity survivors that are also viable compete for the frontier.

**Robustness.** A real edge should not vanish when one parameter moves one step. For a strategy `S`, the engine looks at every neighbor that differs by one grid step on one parameter axis, in either direction, with all other parameters held:

```
retention(N)   = clamp(green_weeks_pct(N) / green_weeks_pct(S), 0, 1)
sensitivity(p) = 1 - min(retention(N) for neighbors N on axis p)
robustness     = 1 - max(sensitivity(p) over all axes)
```

Neighbors count only if they are viable and fully computed. A strategy with no usable neighbors, or with `green_weeks_pct <= 0`, scores 0. The share of axes with at least one usable neighbor is reported as coverage. Interpretation bands:

| Robustness | Shape | Reading |
|---|---|---|
| `>= 0.70` | Plateau | Robust, likely a real edge |
| `0.50 - 0.70` | Ridge | Some sensitivity, proceed with caution |
| `0.30 - 0.50` | Hill | Fragile, likely partly overfit |
| `< 0.30` | Spike | Extremely fragile, probably a fluke |

**Pareto frontier.** Among viable survivors, the engine keeps every result that no other result beats on all five objectives at once (all higher is better):

1. `annual_roc`
2. `green_weeks_pct`
3. `1 / (1 + max_red_weeks)`
4. `trade_recovery_ratio`, capped at 20
5. `profit_factor`, capped at 20

The caps stop sentinel or tiny-denominator values from dominating. Non-finite values are treated as the worst possible on that axis.

**Archetypes.** Frontier members are labeled using thresholds taken from the whole survivor pool:

- Grinder: `win_rate >= max(0.90, pool P75)` and `green_weeks_pct >= max(75, pool P75)`.
- Sniper: `win_rate` between 0.15 and the pool P25 (never below 0.15), and `annual_roc >= pool P75`.
- Balanced: everything else.

**Reduced set.** Frontier members with robustness at or above `robustness_floor` (default 0.50), plus their viable grid neighbors that also meet the floor. This set is the only input to walk-forward.

## 8. Walk-forward

**Windows.** For a schedule with lookback `L` bars and step `F` bars, each window is:

```
in-sample      [t - L, t)
out-of-sample  [t, t + F)
t = t0, t0 + F, t0 + 2F, ...   while t + F <= total bars
t0 = max(L, end of the screen range)
```

Out-of-sample windows never start inside the screen range. The reduced set was chosen on those bars, so trading them again would score the selection on its own training data. In-sample windows may look back into the screen range; that is selection data either way.

In-sample windows roll: each keeps the same length `L`. Day-based lengths are converted to bars using the file's average bars per calendar day. Indicators for each window are loaded with enough warmup bars before the window start; warmup bars are only ever earlier bars.

**Schedule grid.** Re-optimization steps of 1 week, 2 weeks, 1 month or 1 quarter, crossed with lookbacks of 1, 2, 3 or 6 months or 1, 2, 3 or 5 years, keeping only lookbacks at least as long as the step.

**Adaptive triggers.** Instead of re-optimizing on every step, a schedule can step weekly and keep the deployed winner until a trigger fires:

- `dd5pct`, `dd10pct`: out-of-sample equity since the last re-optimization falls 5% or 10% of `capital` below its peak.
- `losers5`, `losers10`: 5 or 10 consecutive losing trades.

The next window then uses its own fresh in-sample result. A window that deploys nothing (no in-sample survivors) also re-optimizes on the next window, since without trades no trigger could ever fire. Trigger state resets on every re-optimization and is tracked separately for each selection strategy.

**Per-window in-sample floors.** Each in-sample window evaluates the reduced set at lag 1 and keeps strategies with at least 30 trades, `profit_factor > 1.0`, `annual_roc > 0`, and an average net trade of at least twice the round-trip cost. The survivors' Pareto frontier is classified into archetypes as in the screen.

**Selection strategies.** One winner per window is picked from the frontier by one of seven rules. Ties go to the lowest index, so selection is deterministic.

| Name | Picks |
|---|---|
| `best_roc` | Highest `annual_roc` |
| `best_consistency` | Highest `green_weeks_pct` |
| `best_trr` | Highest `trade_recovery_ratio` |
| `best_grinder` | Highest `annual_roc` among Grinders |
| `best_sniper` | Highest `annual_roc` among Snipers |
| `frontier_centroid` | Closest to the frontier's centroid, each objective scaled to [0, 1] |
| `roc_x_consistency` | Highest `annual_roc * green_weeks_pct` |

`best_grinder` and `best_sniper` fall back to `frontier_centroid` when their archetype is empty.

**Out-of-sample run and stitching.** The winner trades the out-of-sample window at lag 1 (and again at lag 2, see section 3). Each window starts flat and force-closes at its end. Window results are stitched into one equity curve that steps at each trade's exit bar. Windows with no in-sample survivors are skipped and contribute flat equity.

Aggregate out-of-sample metrics come from the combined trade list. Sharpe and `max_drawdown` are taken from the stitched curve, which starts at zero, so flat skipped periods and an early loss both count. Return on capital is measured over the full out-of-sample span.

**Sufficiency.** A run is marked insufficient unless fewer than half of its windows were skipped (`2 * skipped < total`). Only sufficient runs count toward the per-selection-strategy summary, which averages out-of-sample `annual_roc` across schedules.

**Parameter stability.** The share of consecutive traded windows that kept the same winning strategy. Values near 1 mean re-optimization rarely changes its mind.

## 9. Null-hypothesis check

A pipeline that finds edges in noise is defective, so it should be validated on data with a known answer.

The synthetic generator (`wfo-gpu generate`) produces a random walk with volatility regimes, rare jumps, intraday seasonality and session gaps. With the default `--drift 0` and `--trend 0`, the series has no directional structure. Because every trade pays costs, a correct run should retain nothing, or retain strategies that fail out of sample. Consistent out-of-sample profit on this data indicates that the pipeline leaks future information or understates costs.

`--trend K` plants hidden trend regimes: a drift of `K` bar-sigmas per bar whose sign flips on average every `--trend-bars` bars. This provides a known edge that trend-following strategies can exploit, and confirms that the pipeline finds real structure when it exists.

Both checks are needed: rejecting noise and finding planted structure are separate properties, and a useful optimizer must have both.

## 10. Known limitations

- **Bar data only.** No order book, queue position or partial fills. Every order fills in full at the modeled price.
- **Intra-bar paths are unknown.** When a bar touches both a stop and a target the stop is assumed to fill first, and a stop inside a bar fills at its level even if the real path would have slipped past it.
- **Exit ATR restarts with each range.** The engine's exit ATR starts at the beginning of each evaluation range (a screen range or a walk-forward window), so the first 14 bars of a range cannot take ATR-based exits.
- **One contract.** No position sizing, scaling, pyramiding or compounding of contract count. Return on capital compounds the PnL of a single contract.
- **Constant costs.** Spread and slippage do not change with volatility, time of day or order size.
- **No financing.** No margin interest, borrowing costs or interest on idle capital; Sharpe uses no risk-free rate.
- **Choosing among many runs.** The schedule grid times seven selection strategies produces many out-of-sample runs. The best of them is chosen on out-of-sample data and is an optimistic estimate; look at the whole distribution, not the top row.
- **No formal multiple-testing correction.** The engine does not compute a deflated Sharpe ratio, the probability of backtest overfitting (PBO/CSCV) or a White's reality check. The out-of-sample distribution across all runs and the null-hypothesis runs on synthetic data are the safeguards it offers; a seed ensemble of random-walk runs gives an empirical null for the best-run figure.
- **Robustness skips weak neighbors.** Neighbors that are not viable are left out of the robustness score rather than counted as zero retention. The score measures stability among workable neighbors, not how many neighbors work.
- **Optional early termination.** `[screen] early_termination = true` stops a strategy at 60% of the range if it is losing or has fewer than 10 trades, closing any open position at that bar. It is off by default; when on, the screen runs on the CPU path so results stay the same on every device.
- **The screen reaches the start of the grid.** The screen enumerates strategy ids in order, and ids are ordered by the number of signals, so a budget covers every single-signal strategy (308,640 distinct ones) before any pair. Ids are 32-bit, so at most the first 2^31 grid entries can be screened; three- and four-signal strategies are only reachable from the library API.
- **Schedules cover different spans.** Each schedule's out-of-sample period starts after its own first lookback, so long-lookback schedules trade fewer, later bars. The cross-schedule median mixes spans of different length.
- **Short spans annualize badly.** `annual_roc` compounds the period's return to a year. Over a few weeks of out-of-sample data that turns modest PnL into very large percentages; judge short runs by PnL and drawdown.
- **US DST rules from 2007.** The `us_futures` clock applies the DST rules in force since 2007; local times in older data can be an hour off around the old transition dates. The loader warns when a file starts earlier.
- **30-second assumptions.** Warmup for anchored indicators and the pivot-point warmup are sized assuming 30-second base bars; with longer bars they are conservative, with shorter bars they may be too short.
- **Session detection inside indicators.** VWAP and pivot points find session boundaries from gaps in the bar timestamps rather than from the session model, so on a continuous market VWAP does not reset daily.
- **Data quality and survivorship are your responsibility.** The engine trusts its input bars. Bad ticks, missing sessions, roll artifacts in continuous contracts, inconsistent bar-start versus bar-end timestamps, and instruments chosen because they did well all flow straight into the results.
