import time
from datetime import date
from typing import Callable, List, Optional, Tuple

import pandas as pd

from backtest_MLeg import LegSpec, run_backtest_multi_leg
from dataset import FXVolDataset
from data import fx_calendar
from trading_calendar import preceding_business_day, add_tenor
from exit_hedge_logic import ExitRule, HedgeRule, HoldToExpiry, DailyHedge


def _as_date(d) -> date:
    return d.date() if hasattr(d, 'date') else d


# ═══════════════════════════════════════════════════════════════════════════════
# Signal-driven runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_signal_backtest(
    signal:             pd.Series,
    legs:                Optional[List[LegSpec]] = None,
    pair:                str            = 'USDJPY',
    tenor                               = '1M',
    hedge_rule_factory:  Callable[[], HedgeRule] = DailyHedge,
    exit_rule_factory:   Callable[[], ExitRule]  = HoldToExpiry,
    tc_fraction:         float          = 0.0001,
    history_days:        int            = 365,
    max_concurrent:      Optional[int]  = 1,
    verbose:             bool           = False,
    progress:            bool           = False,
    legs_factory:        Optional[Callable[[date, FXVolDataset], List[LegSpec]]] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.Series]]:
    """
    Run one instance of `legs` per entry signal.

    signal : Series indexed by date/Timestamp, values truthy (1) on days a new
             trade should be entered.

    legs   : fixed LegSpec structure reused, unmutated, for every trade. Pass
             this when the structure needs no market data to build (fixed-delta
             strangles/straddles/spreads via build_strangle etc).

    legs_factory : callable(entry_date, dataset) -> List[LegSpec], called ONCE
             PER TRADE with that trade's resolved entry_date and the run's
             shared FXVolDataset, to build legs whose sizing depends on the
             entry market snapshot (e.g. vega_neutral_butterfly_factory from
             backtest_MLeg.py). Use this instead of `legs` when the structure
             needs market data at entry. Pass exactly one of legs / legs_factory.

    hedge_rule_factory, exit_rule_factory : zero-arg callables returning a
             FRESH rule instance per trade. Required (not plain instances)
             because rules like SpotMoveHedge/GammaScaledHedge carry mutable
             state across days that must not leak from one trade into the next.

    history_days : passed straight through to the shared FXVolDataset pull —
             see dataset sizing note below.

    max_concurrent : how many trades are allowed to be open at once.
             - 1 (default)   : non-overlapping — a signal while a trade is
               still open is ignored; scanning resumes once that trade exits
             - N > 1         : up to N trades open at once; a signal beyond
               that is ignored until one of the open trades exits, freeing a
               slot.
             - None          : unlimited — a new trade is opened on every
               signal==1 regardless of how many are already open (pure
               stacking).
             NOTE: concurrent trades are still simulated fully independently —
             each gets its own hedge, its own P&L

    progress : print a lightweight run tracker to stdout — signal count, the
             one-off dataset-pull time, a periodic loop heartbeat (dates done /
             trades opened / elapsed / ETA), and an end summary with the skip
             breakdown. Independent of `verbose` (which prints per-trade skip
             reasons). Off by default so batch callers (the grid) stay quiet.


    Returns
    -------
    trade_log      : one row per trade,(entry_date, exit_date, exit_reason,
                      net_pnl, days_held, ...) plus `concurrent_at_entry` 
                      
    trade_df_aggs  : list of each trade's df_agg (daily portfolio detail).
    trade_leg_sums : list of each trade's leg_summaries.
    """
    assert max_concurrent is None or max_concurrent >= 1, \
        "max_concurrent must be None (unlimited) or a positive int"
    assert (legs is None) != (legs_factory is None), \
        "pass exactly one of legs / legs_factory"
    sig = signal.sort_index()
    sig = sig[sig.notna()]
    sig_dates = [_as_date(ts) for ts in sig.index]
    if not any(sig.astype(bool)):
        return pd.DataFrame(), [], []

    # ── One shared dataset for the whole run ────────────────────────────────
    t_start           = time.perf_counter()
    fxc               = fx_calendar(pair)
    earliest_entry    = preceding_business_day(sig_dates[0], fxc.cal_trade)
    sample_expiry     = add_tenor(earliest_entry, tenor, fxc)
    tenor_days_est    = (sample_expiry - earliest_entry).days
    today             = date.today()
    days_back         = (today - earliest_entry).days
    dataset_days      = max(history_days, days_back + tenor_days_est + 10)
    n_entries_est     = int(sig.astype(bool).sum())
    if progress:
        print(f"[backtest] {pair} {tenor}: {n_entries_est} entry signals over "
              f"{len(sig)} dates | max_concurrent={max_concurrent} | "
              f"pulling {dataset_days}d of data...", flush=True)
    t_ds              = time.perf_counter()
    dataset           = FXVolDataset.build(pairs=[pair], days=dataset_days)
    if progress:
        print(f"[backtest] dataset ready in {time.perf_counter() - t_ds:.1f}s "
              f"— starting trade scan", flush=True)

    # ── Sequential entry scan, gated by max_concurrent ──────────────────────
    trade_summaries: List[pd.Series]     = []
    trade_df_aggs:   List[pd.DataFrame]  = []
    trade_leg_sums:  List[List[pd.Series]] = []
    concurrent_at_entry: List[int] = []

    active_exits: List[date] = []   # exit_date of every currently-open trade
    seen_entry_dates: set    = set()   # resolved entry_date already traded, ever

    # Progress tracking — skip-reason tally + periodic heartbeat.
    skip_full = skip_dupe = skip_nodata = 0
    t_loop    = time.perf_counter()
    hb_step   = max(1, len(sig) // 20)   # ~20 heartbeats across the scan

    n = len(sig)
    for i in range(n):
        if progress and i and i % hb_step == 0:
            el   = time.perf_counter() - t_loop
            done = i + 1
            eta  = el / done * (n - done)
            print(f"[backtest] {done:>5}/{n} dates | {len(trade_summaries):>4} trades "
                  f"| {el:6.1f}s elapsed | ~{eta:6.1f}s left", flush=True)

        today_date = sig_dates[i]
        # drop trades that closed strictly before today (exit day itself still
        # counts as occupying its slot, same as the old non-overlap rule)
        active_exits = [d for d in active_exits if d >= today_date]

        if not bool(sig.iloc[i]):
            continue
        if max_concurrent is not None and len(active_exits) >= max_concurrent:
            skip_full += 1
            continue

        # Resolve the actual trading-calendar entry date up front — a raw
        # signal date that isn't itself a business day on this pair's FX
        # calendar rolls back to the preceding one (same resolution
        # run_backtest_multi_leg applies internally). Signal sources whose
        # date index doesn't line up exactly with that calendar (e.g. a
        # Bloomberg print published on a date this pair's calendar treats as
        # a holiday) can have two DIFFERENT raw signal dates resolve to the
        # SAME entry_date — silently producing two identical trades. Skip if
        # this entry_date has already been traded this run, regardless of
        # which raw signal date got there first.
        entry_res = preceding_business_day(today_date, fxc.cal_trade)
        if entry_res in seen_entry_dates:
            skip_dupe += 1
            if verbose:
                print(f"  [skip] {today_date}: resolves to {entry_res}, "
                      f"already traded from an earlier signal date.")
            continue

        # Empty-records guard: skip entries with no market data AFTER inception
        # before min(expiry, today) — e.g. a signal on the most recent date.
        # run_backtest_multi_leg would otherwise build an empty daily frame and
        # raise KeyError on set_index('date'). Mirror its entry-date/window
        # resolution so this check agrees with what the engine would see.
        expiry_res = add_tenor(entry_res, tenor, fxc)
        series_end = min(expiry_res, today)
        window     = dataset.spot.loc[str(entry_res):str(series_end), pair].dropna()
        if not any(_as_date(ts) > entry_res for ts in window.index):
            skip_nodata += 1
            if verbose:
                print(f"  [skip] {today_date}: no market data after inception "
                      f"({entry_res} -> {series_end}); trade cannot run.")
            continue

        seen_entry_dates.add(entry_res)
        concurrent_at_entry.append(len(active_exits))
        trade_legs = legs_factory(entry_res, dataset) if legs_factory is not None else legs
        df_agg, leg_dfs, summary, leg_summaries = run_backtest_multi_leg(
            trade_legs,
            pair=pair,
            tenor=tenor,
            entry_date=entry_res,
            dataset=dataset,
            tc_fraction=tc_fraction,
            verbose=verbose,
            exit_rule=exit_rule_factory(),
            hedge_rule=hedge_rule_factory(),
        )
        trade_summaries.append(summary)
        trade_df_aggs.append(df_agg)
        trade_leg_sums.append(leg_summaries)
        active_exits.append(_as_date(summary['exit_date']))

    if progress:
        loop_s = time.perf_counter() - t_loop
        total_s = time.perf_counter() - t_start
        per = (loop_s / len(trade_summaries)) if trade_summaries else float('nan')
        print(f"[backtest] done: {len(trade_summaries)} trades opened "
              f"({per*1000:.0f}ms/trade) | skips: {skip_full} concurrency, "
              f"{skip_dupe} dup-date, {skip_nodata} no-data | "
              f"loop {loop_s:.1f}s, total {total_s:.1f}s", flush=True)

    if not trade_summaries:
        return pd.DataFrame(), [], []

    trade_log = pd.DataFrame(trade_summaries).reset_index(drop=True)
    trade_log['concurrent_at_entry'] = concurrent_at_entry
    return trade_log, trade_df_aggs, trade_leg_sums


# ═══════════════════════════════════════════════════════════════════════════════
# Trade-log summary helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_trade_log(trade_log: pd.DataFrame) -> None:
    """Pretty-print the trade log with a few headline stats."""
    if trade_log.empty:
        print("No trades triggered by this signal.")
        return

    cols = ['entry_date', 'exit_date', 'exit_reason', 'days_held', 'concurrent_at_entry',
            'net_premium', 'gamma_pnl', 'theta_pnl', 'vega_pnl', 'vanna_pnl', 'volga_pnl',
            'hedge_carry_pnl', 'transaction_costs', 'net_pnl']
    cols = [c for c in cols if c in trade_log.columns]

    print(f"\n{'='*65}")
    print(f"  SIGNAL BACKTEST — {len(trade_log)} trades")
    print(f"{'='*65}")
    print(trade_log[cols].to_string(
        formatters={c: '{:,.2f}'.format for c in cols
                    if trade_log[c].dtype.kind in 'fc'}))

    win_rate = (trade_log['net_pnl'] > 0).mean() * 100
    print(f"\n  Total net P&L:   {trade_log['net_pnl'].sum():>14,.2f}")
    print(f"  Avg net P&L:     {trade_log['net_pnl'].mean():>14,.2f}")
    print(f"  Win rate:        {win_rate:>13.1f}%")
    print(f"  Avg days held:   {trade_log['days_held'].mean():>13.1f}")
