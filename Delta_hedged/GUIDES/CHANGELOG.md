# Changelog

Dated record of behaviour changes, so a result table recorded before a given entry
can be reconciled against one recorded after it. For *how the system works*, see
[backtest_process.md](backtest_process.md).

---

# 2026-08-14

One currency fix, one table-preset change, and two pull-path caches. **The currency
fix changes recorded numbers on the four non-USD-base pairs** (EURUSD, GBPUSD,
AUDUSD, NZDUSD). Nothing else changes a number.

## 0. Impact on results you already have

| Metric | Changed? | Cause | Direction |
|---|---|---|---|
| `expectancy`, `median_pnl`, `worst_trade`, `cvar_5pct` (trade), `theta_carry_pnl`, `gamma_pnl`, `vol_pnl`, `spot_tc` | **yes, on EURUSD/GBPUSD/AUDUSD/NZDUSD** | fix #1 | were base ccy, now USD. Not a clean ×spot: the P&L-weighted rate. Measured EURUSD 1M 5y: `expectancy` −797 EUR → −608 USD (×0.763) |
| `ret_on_prem` | **yes, on those four pairs** | fix #1 | numerator now at daily rates, `net_premium` at the entry rate. Measured: −0.0124 → −0.0086 |
| `payoff_ratio`, `t_stat`, `profit_factor`, `pnl_skew`, the `*_share`s | **yes, slightly, on those four pairs** | fix #1 | unit-free but **not invariant**: each trade converts at its own P&L-weighted rate, so any statistic that mixes trades non-linearly (ratio of sums, mean/std) shifts. Measured `payoff_ratio` 0.6557 → 0.6681 (+1.9%) |
| `n_trades`, `win_rate`, `coverage`, `real_vrp`, `vol_spread`, `avg_days_held`, `pct_hold_to_expiry` | **no** | — | exactly invariant: sign-based, count-based, or not money at all. All measured at ratio 1.0000 |
| `net_pnl`, `calmar`, `sharpe`, `sortino`, `max_drawdown`, `var_*`, `cvar_95` (daily), `avg_net_vega`, `avg_net_gamma` | **no** | — | book-sourced, already USD |
| USDJPY / USDCHF / USDCAD | **no** | — | base == USD, factor 1.0, conversion is a no-op |
| gate-sweep `ATTR_COLS` | **no** | — | still base ccy by design; reads the raw log, not `evaluate`'s copy |
| everything, from the new caches (#3) | **no** | — | same data, pulled once instead of once per cell. If anything it removes a source of drift: cells of one table are now guaranteed to share one snapshot |

## 1. FIX — `to_usd=True` converted only the daily book

`evaluate()` called `to_usd_book(book, pair)` and then passed the **raw** `trade_log`
into `scorecard()`, so every `trade_metrics` money figure stayed in the pair's base
currency while every `book_metrics` one was USD. One scorecard, two currencies.
`expectancy` was the visible casualty: a ~0.6–1.3× cross-pair distortion on a column
formatted and coloured as though it were USD.

New `to_usd_trade_log(trade_log, trade_dfs, pair, fx_usd=None)`
([reporting.py](../reporting.py)) converts the per-trade lens, and `evaluate(to_usd=True)`
now applies both.

Each money field is **re-summed from that trade's daily flows** at that day's rate,
`sum_days(factor(day) × flow(day))`, not scaled by a single rate. That is what keeps the
two lenses reconciled — `sum(trade net_pnl) == book_total_pnl` exactly (verified to
0.000000 on 57 real EURUSD trades). `net_premium` is the exception: an entry-date
cashflow, converted at the entry rate.

The FX convention now lives in one helper, `_usd_factor`, shared by both converters so
they cannot drift apart. It also ffills/bfills gaps in the pair's own spot, which
`to_usd_book` previously left to silently NaN out a day's P&L.

**Reading a converted figure:** `usd_total / base_total` is a P&L-**weighted** average
rate, not the average spot, and with gains and losses partly cancelling it can land
outside the spot range entirely. In the measured run `net_pnl` scaled 0.727 against an
average spot of 1.10 and a sample low of 0.959. This is not new — `net_pnl`/`calmar`/
`sharpe` always behaved this way; `expectancy` has simply joined them. See §14.3b of the
process guide.

## 2. CHANGED — `SIGNAL_TABLE_COLS`

```python
SIGNAL_TABLE_COLS = ['expectancy', 'max_drawdown', 'ret_on_prem', 'net_pnl',
                     'calmar', 'sharpe',
                     'n_trades', 'coverage',
                     'win_rate', 'payoff_ratio',
                     'cvar_95',
                     'real_vrp', 'avg_net_vega', 'avg_net_gamma']
```

Added `expectancy` (safe to show now that fix #1 makes it USD), `avg_net_vega`,
`avg_net_gamma`. Dropped `t_stat` (selectivity deflates it mechanically; `n_trades`
already carries the sample-size warning) and `vol_spread` (within the smile premium of
`real_vrp`). 13 columns → 14, and the printed line is ~30 chars wider.

No metric definitions changed — all 14 were already in `CLEAN_METRICS`.

Known cosmetic issue: `expectancy` now leads the table but formats through
`_compact_num`, which is `{v/1e3:.0f}k` above 1000 — so a true 3,375 USD/trade prints
as `3k`. Per-trade figures sit right at the boundary where that formatter loses
precision, unlike the 100k+ book figures it was written for. Not fixed.

## 3. NEW — process-lifetime caches on the two uncached Bloomberg pull paths

Three code paths pull vol/market data. The regime panel was already cached
(`_SERIES_CACHE` / `_CHECK_CACHE` in `regime_filter.py`); the other two were not, so a
grid re-pulled identical data once per cell.

| Path | Was | Now | Measured (USDCAD, 5y) |
|---|---|---|---|
| `FXVolDataset.build` — spot, full vol surface, SOFR, fwd yields | one pull per `run_signal_backtest` call, i.e. per grid cell | memoized on `(pairs, days)` | cold 10.14s → warm 0.000s |
| `get_ImplRealVol` — the batched IV+RV pull every signal builder routes through | one `blp.bdh` per `signal_fn` call, i.e. per cell | memoized on `(currency_list, tenors, days_back)` | cold 0.71s → warm 0.000s |

**Why every cell asks for the same thing.** `dataset_days` is derived from
`sig_dates[0]` — the first date in the signal **index**, not the first date it fires
([backtest_signal.py:94-106](../backtest_signal.py#L94-L106)) — and `apply_regime_gate`
multiplies the gate onto `signal.index` without dropping rows. So an ungated column and
its gated variants all resolve to one cache key. Verified on a 4-column USDCAD sweep:

```
none       : 1305 dates, first 2021-08-16, fires 1305
trend_5d   : 1305 dates, first 2021-08-16, fires  694
trend_10d  : 1305 dates, first 2021-08-16, fires  718
trend_15d  : 1305 dates, first 2021-08-16, fires  742
```

That run went from 4 dataset pulls + 4 signal pulls to 1 + 1, ~32s of a ~5min run.

`get_ImplRealVol` returns a **copy**, so a caller mutating the frame cannot poison the
cache (verified). `FXVolDataset.build` returns the **shared instance** — the same
sharing that already happened across every trade within one backtest, and it lets
per-instance memoized work (smile fits) carry across cells too. The original body is
now `_build_uncached`.

New: `clear_dataset_cache()` / `dataset_cache_info()` in `dataset.py`,
`clear_ivrv_cache()` in `Implied_Realized.py`. Caches live for the process, so a session
left open across a Bloomberg refresh needs all three cleared (those two plus
`clear_regime_cache()`) to force fresh pulls.

## 4. Docs

- `plot_full_report` docstring now warns that it converts **nothing**: pass a
  `to_usd_book` book with a `to_usd_trade_log` log, or neither. The equity curve reads
  the book while the per-trade histogram and worst-trades table read the log, so mixing
  them puts two currencies on one page. Newly reachable now that #1 exists.
- `reporting.py` module docstring, `grid_eval.py` module docstring + `run_grid`/
  `run_combo` docstrings, and the `gate_sweep.py` header comment all corrected — the
  last said "`to_usd` touches only the daily book", which #1 falsified. Its attribution
  **is** still base ccy, but because it reads the raw log it holds, not `evaluate`'s
  converted copy.
- `backtest_process.md`: §15.4 rewritten (was "THE UNIT TRAP"), new §14.3b for
  `to_usd_trade_log`, §14.3 notes `_usd_factor`, §16.1/§16.2 presets updated, trap #1 in
  §20.1 rewritten, and the Layer-6 line in the conventions preamble corrected.

## 5. Files touched

| File | Change |
|---|---|
| `reporting.py` | `_usd_factor`, `to_usd_trade_log`, `_TRADE_MONEY_FROM_FLOW`; `evaluate` converts both lenses + `fx_usd` passthrough; `to_usd_book` refactored onto the shared helper; module + `plot_full_report` docstrings |
| `grid_eval.py` | `SIGNAL_TABLE_COLS` (13 → 14); `expectancy` metric comment; preset rationale block; module / `run_grid` / `run_combo` currency docs |
| `dataset.py` | `build` memoized, body → `_build_uncached`; `_DATASET_CACHE`, `clear_dataset_cache`, `dataset_cache_info` |
| `Signal_Gen/Implied_Realized.py` | `get_ImplRealVol` memoized (returns a copy); `_IVRV_CACHE`, `clear_ivrv_cache` |
| `gate_sweep.py` | header currency comment corrected |
| `TEST.py` | `legs_fn=_custom()` → `_custom` (it is already `fn(s)`, so the parens raised `TypeError`); regime_filter import gained `check_defaults`, `clear_regime_cache`, `regime_cache_info`; §4 block retitled and its redundant `days_back=DAYS_BACK` dropped |
| `GUIDES/backtest_process.md` | §14.3, §14.3b, §15.3, §15.4, §16.1, §16.2, §20.1, conventions preamble |

## 6. Verification performed

**Currency fix**, on a synthetic 2-trade book and a live 57-trade EURUSD 1M 5y run:
`sum(trade net_pnl) == book_total_pnl` to **0.000000**; `expectancy × 56 settled ==`
settled USD P&L; `expectancy`'s conversion factor matches the book's own to 1e-9;
`win_rate` / `real_vrp` / `n_trades` / `coverage` ratios exactly 1.0000; `max_drawdown`,
`cvar_95`, `avg_net_vega`, `avg_net_gamma` all scale inside the sample spot range
(0.959–1.204); USDJPY conversion is a no-op; `net_premium` converts at the entry rate;
vol-point and flag fields pass through untouched. All 14 `SIGNAL_TABLE_COLS` resolve
against `CLEAN_METRICS` with a real `fmt` (no silent fallback).

**Caches:** cold-vs-warm timings above; identical data on the warm call; mutating a
returned frame does not poison the cache; distinct args produce distinct entries; the
4-column index-equality result above.

**End-to-end:** `TEST.py` ran clean on USDJPY 1M (2 combos, 120s, both `ok`), printing
the 14-column table headed `GRID SCORECARD (USD)`.

**Not verified:** the pricing layer remains unreviewed (§11 item 3 of the previous
entry). Multi-pair cross-sections were not re-run after the currency fix — only EURUSD
and USDJPY/USDCAD.

---

# 2026-08-11

Five bug fixes, five new metrics, two metric-semantics corrections, and one new
sweep mode. **Three of the fixes change recorded numbers** — see the impact table
first.

## 0. Impact on results you already have

| Metric | Changed? | Cause | Direction |
|---|---|---|---|
| `coverage` | **yes, always** | fix #3 | was identically `100%`; now real. Back-to-back trades read `rows/(rows+1)` — ~84% at 1W, ~96% at 1M, ~99% at 3M |
| `calmar` | **yes** | fix #3 | reduced by `calendar/deployed`. Negligible for always-on (~1%); ~3× for a 16-trade selective signal |
| `n_trades`, `expectancy`, `win_rate`, `payoff_ratio`, `t_stat`, `ret_on_prem` | **yes, on early-exit configs only** | fix #1 | *increase* in `n_trades` — settled trades were being dropped. No change under `HoldToExpiry` |
| everything for an `xccy` signal | **yes** | fix #2 | the window was ~380 calendar days longer than the baseline's. Old xccy results are not comparable to anything |
| everything for a `DeltaBandHedge` config | **yes** | fix #4 | the rule never did what it claimed. No prior results affected — unused so far |
| `net_pnl`, `max_drawdown`, `sharpe`, `sortino`, `var_95`, `var_99`, `cvar_95` | **no** | — | untouched by any fix. If these moved, it is the window shift (§0.1) |

### 0.1 Not a code change, but read this before comparing old tables

`DAYS_BACK` is anchored to `datetime.now()`, so **re-running on a different day
shifts the whole window by a day.** With `max_concurrent=1` the entry chain is
self-propagating — trade 1's expiry sets trade 2's entry, and so on — so a
one-day shift **re-phases every trade in the sequence**.

Measured on EURUSD 1W, same config, same 5-year length, same 218 trades, one
business day apart: **net_pnl +232k → +102k.** Consistent with a 342k max
drawdown, but it means:

- any config difference under ~130k of 5-year P&L (at 1W) is unreadable;
- **tables run on different days cannot be compared at all.**

A design for pinning the window (an as-of clock, 13 call sites) was scoped but
**not implemented** — see §5.

---

## 1. FIX — `live_trade` dropped settled trades

`backtest_MLeg.py:851`

```python
live_trade = expiry > today        # before
live_trade = exit_reason is None   # after
```

A trade is live only if the loop ended **without a terminal event**. Both terminal
paths (`'expiry'`, or the exit rule's `.name`) fully settle the position, so
`net_pnl` is final.

The old test asked whether *expiry* was ahead — but an exit rule settles a trade
weeks before expiry. So under `ExitAfterNDays`, `ExitAtDaysRemaining` or
`TakeProfitStopLoss`, **every trade entered within one tenor of today was
mislabelled live** and then silently dropped by `trade_metrics(settled_only=True)`.
It always dropped the most recent trades, biasing every per-trade statistic
against the end of the sample. `HoldToExpiry` made the old test coincidentally
correct, which is why baseline runs were unaffected.

Also renamed the unrelated verbose-block local to `expiry_ahead`
(`backtest_MLeg.py:399`) so the two questions aren't confusable.

**Re-run:** any config with an exit rule other than `HoldToExpiry`.

## 2. FIX — the xCCY signal spanned a different window

`Signal_Gen/xCCY_Spread.py:96` and `:69`

```python
val_ts = val_ts.iloc[-days_back:]                    # before — ROW count
cutoff = val_ts.index.max() - pd.Timedelta(days=days_back)   # after — CALENDAR days
val_ts = val_ts[val_ts.index >= cutoff]
```

`days_back` was treated as a row count. At `days_back=1825` there are only ~1500
rows available, so **the trim was a no-op** — the signal returned its entire
pulled history, starting `pct_lookback + 15 = 380` calendar days earlier than an
always-on or IV/RV signal built with the same `days_back`.

Any grid with `signals={'none': _always_on, 'xccy80': _xccy(80)}` was therefore
**comparing a ~6-year sample against a 5-year one.**

Also raised the rolling `min_periods` from `2` to `20`. A percentile from 2
observations can only return 50 or 100, so the earliest stretch of the series was
feeding meaningless values into the threshold. `20` matches
`rolling_percentile`'s floor in `Implied_Realized.py`.

**Re-run:** everything involving `_xccy` / `get_xccy_spread_signal`.

## 3. FIX — `coverage` was always 100%, `calmar` was inflated

`reporting.py:444` and `:475`

`build_daily_book` indexes on the **union of the trades' own date indices**, so a
day with no trade open is **absent** from the book, not a zero row. Both metrics
were computed off `len(book)` — the *deployed*-day count:

```python
ann_pnl  = total / max(len(book)/252, 1e-9)          # before
coverage = active_days / len(book)                   # before  -> identically 1.0

span_days  = (book.index[-1] - book.index[0]).days + 1      # after
span_years = span_days / 365.0
span_bdays = np.busday_count(first, last) + 1
ann_pnl  = total / span_years
coverage = active_days / span_bdays
```

- `calmar` annualized a selective signal's P&L over its *deployed* span,
  overstating it by `calendar/deployed` — ~3× for a signal trading a third of the
  time.
- `coverage` could only ever read `1.0`, because every row in the union belongs to
  at least one trade. It was the one column meant to vary and it never did.

New scorecard key **`span_bdays`** (business days, first row → last row), added to
the `Time / exposure` scorecard group alongside `coverage`, with display labels
`n_days (deployed)` and `span_bdays (first->last)`.

**`sharpe_ann` / `sortino_ann` were deliberately left alone.** With no idle rows to
dilute the mean they are already per-deployed-day — what `active_only=True` would
give. There is **no √coverage penalty**, and they are comparable across signals.

Verified on a synthetic book (12 back-to-back trades vs 4 over the same span):

| | rows | span_bdays | coverage | calmar |
|---|---|---|---|---|
| continuous | 252 | 252 | 100% | −0.65 |
| selective | 84 | 210 | **40%** | −0.34 |

Residual limitation: the span starts at the **first trade**, not at the start of
your requested window, so a signal that doesn't fire for six months isn't charged
for it. And `np.busday_count` ignores FX holidays, making `coverage` a mild
under-estimate. A full fix would reindex the book onto the signal's own calendar
inside `evaluate`.

**Re-run:** everything, if you care about `calmar` or `coverage`.

## 4. FIX — `DeltaBandHedge` ignored the hedge it was holding

`exit_hedge_logic.py:236`

```python
abs(ctx.net_delta_pretrade) / ctx.total_notional > self.band       # before
abs(ctx.target_hedge - ctx.current_hedge) / ctx.total_notional > self.band   # after
```

It tested the option legs' **gross** delta, ignoring the hedge already on. That
isn't a band hedge — it's a delta-*level* trigger, and it degenerates in both
directions:

- a short strangle sits near delta-flat, so with a hedge on, a gross delta below
  the band meant the rule **never rehedged** and held the day-1 hedge to expiry;
- once gross delta sat persistently above the band it rehedged every day, silently
  becoming `DailyHedge`.

The band was never controlling hedge error. It also mixed timings —
`net_delta_pretrade` is a start-of-day greek while `target_hedge` is today's.

Now tests the unhedged residual (`== hedge_gap`), which is the exposure actually
being run. Verified: residual 0 with 3mm gross delta → no hedge; 500k residual
with 0.1mm gross delta → hedge.

**No prior results affected** — the rule hadn't been used yet.

## 5. FIX — NaN in the final row at expiry

`backtest_MLeg.py:600`

The expiry-settlement branch didn't set `vanna_1vp_leg{i}` / `volga_1vp_leg{i}`,
so any trade ending at expiry left NaN in its last row for those two. Nothing
consumes them (`leg_dfs` doesn't include them; `_EXPO_COLS` uses the aggregate
columns, which *were* set), so **no results were affected.** Fixed for consistency.

---

## 6. NEW — five metrics

`grid_eval.py:79` onward. All five already came out of `evaluate()`; nothing new is
computed, which preserves the "reporting.py is the single source of truth"
contract. `CLEAN_METRICS` went from **14 → 19** entries.

| metric | scorecard key | source | why |
|---|---|---|---|
| `ret_on_prem` | `return_on_premium` | trade, unit-free | P&L per unit premium collected. **The primary for signal/gate comparison** — cross-pair safe and deployment-invariant |
| `payoff_ratio` | `payoff_ratio` | trade, unit-free | avg win / avg loss. **`center=1.0`, not 0** — a ratio of 1 means wins and losses are the same size |
| `cvar_95` | `cvar_5pct_daily` | book, USD | mean of the worst 5% of days, vs `var_95`'s quantile |
| `vol_spread` | `avg_vol_spread` | trade, vol points | realized − **vega-weighted** entry implied |
| `real_vrp` | `real_VRP_ave` | trade, vol points | realized − **ATM** entry implied |

`vol_spread` and `real_vrp` both have **inverted signs** relative to every other
`div` metric: both are realized *minus* implied, so for a short book (`direction=-1`)
**more negative is better.** Flips at `direction=+1`.

## 7. NEW — `SIGNAL_TABLE_COLS`

`grid_eval.py:159`. A second console preset, so existing runs print unchanged.

```python
SIGNAL_TABLE_COLS = ['ret_on_prem', 'net_pnl', 'calmar', 'sharpe',
                     'n_trades', 'coverage', 't_stat',
                     'win_rate', 'payoff_ratio',
                     'max_drawdown', 'cvar_95',
                     'vol_spread', 'real_vrp']
```

Leads with the deployment-invariant edge metric. Drops `sortino` (empirically
1.11–1.22× `sharpe` row for row — no incremental information), `var_95`/`var_99`
(superseded by `cvar_95`), and `expectancy` (base-ccy, §9). Added to `__all__`.

> **Superseded 2026-08-14 (§2).** This 13-column list was replaced by a 14-column one;
> `expectancy` is now included, since it is no longer base-ccy. A table printed before
> that date has these columns, in this order.

## 8. NEW — `signals=` sweep mode

`TEST.py:218`. `RUN` gained `signals`, `sort_by`, `ascending`, `cols`.

```python
g = RUN('signals | 1M 25d strangle | Hold to Expiry | Max 1 Trade',
        tenors=['1M'], max_concurrent=1,
        signals={'none': _always_on, 'ivrv80': _ivrv(80), 'xccy80': _xccy(80)},
        sort_by='pair')
```

- Builds one `ComboSpec` per (pair, tenor, signal) with `label=` set, which flips
  `_resolve_col` to lay the grid out as **pair × signal**.
- Routes through `run_grid`, so **`max_concurrent` is honoured** (unlike `gates=`).
- Comprehension order is `for p in P for t in T for lbl, fn in signals.items()` —
  pair-major, signal innermost — so with `print_grid`'s stable sort, `sort_by='pair'`
  puts each pair's baseline directly above its alternatives.
- Two asserts: `signals=` and `gates=` cannot be combined; `signals` must contain a
  `'none'` key.
- Prints `print_gate_consistency(metric='ret_on_prem', label_header='signal')` —
  `ret_on_prem` rather than `calmar` for the reason in §3/§9.
- `ascending` resolves automatically: A→Z on a key column (`pair`/`tenor`/`label`),
  best-first on a metric.

`gate_consistency` / `print_gate_consistency` (`gate_sweep.py:361`) gained
`baseline='none'` and `label_header='gate'`, so a signal sweep reuses them
unchanged. Nothing in that function was ever gate-specific.

## 9. NEW — parameterized signal builders

`TEST.py:119`. `_ivrv(sell_pct)` and `_xccy(sell_pct)` are now factories, following
the `_strangle` pattern. `sell_pct` was hardcoded at 80 — the one parameter
controlling selectivity wasn't sweepable.

`_sig_ivrv = _ivrv(80)` and `_sig_xccy = _xccy(80)` remain as module-level aliases,
so every existing `signal_fn=_sig_ivrv` call site still works.

This enables the **threshold frontier**, the strongest single test available:

```python
signals={'none': _always_on, **{f'ivrv{p}': _ivrv(p) for p in (50,60,70,80,90)}}
```

Read `coverage`, `ret_on_prem` and `vol_spread` together down each pair's block. A
real signal traces a monotone frontier; a peak at one threshold is noise.

---

## 10. CORRECTED — metric semantics (no code change, but the meaning differs)

These were documented wrongly. The code always behaved as described below.

### `vol_spread` is ex-post, not ex-ante

`backtest_MLeg.py:903` defines it as `realised_vol - avg_entry_sigma`. It is **not**
"the implied spread the signal saw at entry." Both vol columns are ex-post; they
differ only in the reference implied:

| | reference | source |
|---|---|---|
| `vol_spread` | `avg_entry_sigma` — vega-weighted across legs, **the vol you actually sold** | `backtest_MLeg.py:903` |
| `real_vrp` | `atm_entry_vol` — pure ATM at inception | `reporting.py:376` |

The **gap** between them is roughly the smile premium collected by selling wings
instead of ATM — empirically a consistent −0.15 to −0.25 vol points across every
row of a 25d strangle grid.

**Consequence:** the grid **cannot** answer "did the signal select rich implied
vol?" Nothing in `trade_log` records the signal's entry condition. For that, use
the signal's own percentile series — the *first* element of the
`(pct_metric, signal_metric, signal_series)` triple that the builders discard with
`[2]`.

### The native currency is BASE, not quote

> **Superseded 2026-08-14 (fix #1).** `evaluate(to_usd=True)` now converts the per-trade
> log as well, so the keys listed below are USD and cross-pair comparable. The rest of
> this subsection — that the native unit is BASE, and why — still holds and is the
> foundation the new converter rests on.

`evaluate(to_usd=True)` converts **only the daily book**; `trade_log` is passed to
`scorecard` unconverted. That unconverted unit is the pair's **base** currency
(`pair[:3]`), which follows from the pricing: option value is
`(intrinsic / S) × notional`, and `intrinsic/S` is dimensionless, so the result
carries `LegSpec.notional`'s unit. `to_usd_book` multiplies by USD-per-base — a
conversion only valid on a base-ccy book, which is the proof.

| pairs | trade-level money unit |
|---|---|
| USDJPY, USDCHF, USDCAD | **already USD** (base = USD → factor 1.0) |
| EURUSD, GBPUSD, AUDUSD, NZDUSD | EUR / GBP / AUD / NZD, ~0.6–1.3 × USD |

Base-ccy keys: `expectancy`, `median_pnl`, `worst_trade`, `cvar_5pct`,
`theta_carry_pnl`, `gamma_pnl`, `vol_pnl`, `spot_tc`. **None appear in either
printed table preset**, so no printed result was ever mis-scaled.

Worst cross-pair distortion is ~±40%, not the order of magnitude a quote-currency
mix would give. Comments corrected in `gate_sweep.py` (header) and `grid_eval.py`
(the `expectancy` warning).

### Docstrings that described absent behaviour

- **`get_always_on_signal`** claimed it trims a tenor off the tail so every trade
  is completed. **It doesn't** — it only drops future dates. Rewritten to describe
  the two downstream guards that actually handle it (`skip_nodata` in
  `run_signal_backtest`, and `live_trade` + `settled_only`), and to note why
  trimming would be wrong: an early exit *can* settle a trade inside that window.
- **`get_vol_signal`** said `metric` accepts `'IV-RV'`. The assert requires
  **`'VD'`**; `'IV-RV'` raises.

---

## 11. NOT fixed — known limitations

| # | Limitation | Why not |
|---|---|---|
| 1 | ~~**`to_usd` doesn't convert `trade_log`**~~ **FIXED 2026-08-14 (#1)** | The open question here — "one FX rate per trade, entry or exit?" — had a third answer: neither. Each field is re-summed from that trade's daily flows at each day's rate, which is the only choice that keeps the trade lens reconciled with the book lens |
| 2 | **The window is anchored to `datetime.now()`** — results are not reproducible day to day | Design scoped (an as-of clock, 13 call sites across 6 files, ~30-line module, no signature changes below `RUN`) but **not implemented**. This is the highest-value outstanding item |
| 3 | **The pricing layer is unreviewed** — `pricer.py`, `option.py`, `dataset.py`, `vol_surface.py`, `trading_calendar.py` (1,631 lines) | Not read. A sign error in a greek or an off-by-one in `add_tenor` would invalidate everything above it. **Validate by pricing one option against Bloomberg OVML** — that checks the whole stack in one shot |
| 4 | **`coverage`'s span starts at the first trade**, not the start of the requested window | A full fix needs the signal's calendar threaded into `evaluate` |
| 5 | **Bloomberg revises history** | Pinning the request window (item 2) gives strong but not absolute reproducibility. True determinism needs the pulls cached to disk |
| 6 | **No transaction cost on the natural-expiry unwind**, but full cost on an early exit | Pre-existing engine asymmetry (§5.6 of the guide), not a regression. It biases exit-rule comparisons toward `HoldToExpiry` by roughly $8k (1W-left) to $25k (weekly) over 5 years. Stress with `tc_fraction=0` |
| 7 | `Signal_Gen/vol_regime.py`, `option.py:222`, `data.py:187`, `Random/*` still anchored to `now()` | Dead or out-of-path code; patching would imply they're live |

---

## 12. Files touched

| File | Change |
|---|---|
| `backtest_MLeg.py` | fixes #1, #5; verbose local rename |
| `reporting.py` | fix #3; new `span_bdays` key; scorecard group + display labels |
| `exit_hedge_logic.py` | fix #4; `GammaScaledHedge` interval-units note |
| `Signal_Gen/xCCY_Spread.py` | fix #2 (calendar cutoff + `min_periods`) |
| `Signal_Gen/Implied_Realized.py` | two docstring corrections (§10) |
| `grid_eval.py` | 5 new metrics, `SIGNAL_TABLE_COLS`, `__all__`, currency comment |
| `gate_sweep.py` | `baseline` / `label_header` params; currency comment |
| `TEST.py` | `signals=` branch, `sort_by`/`ascending`/`cols`, signal factories, import |
| `GUIDES/backtest_process.md` | coverage section rewritten; 6 currency corrections; traps index updated |

## 13. Verification performed

All nine modules compile. Every one of the 19 `CLEAN_METRICS` keys resolves against
`evaluate()`'s output; both table presets and all heatmap panels resolve; no
duplicate metric names; no dangling scorecard-group keys. Fixes #3 and #4 were
verified against synthetic cases with known answers. Edge cases checked: empty
book, single-row book, zero notional, missing baseline label.

**Not verified:** no live backtest was run (no Bloomberg access in the review
environment). Everything above is static reading plus synthetic unit tests on the
metric layer.
