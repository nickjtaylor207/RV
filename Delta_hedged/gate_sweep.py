"""
gate_sweep.py — sweep regime gates over one strategy, cheaply and attributably.

Why this exists instead of just calling run_grid with gated signals
-------------------------------------------------------------------
A regime gate is a pure ENTRY filter: it can only ever remove entry days. And
run_signal_backtest is called by the grid with `max_concurrent=None`, which means
every signal day opens its own trade and — per backtest_signal.py:70-72 —
"concurrent trades are still simulated fully independently". So removing entry
days cannot change any trade that remains.

Therefore, for a fixed (pair, tenor, signal, structure, hedge, exit):

    run the UNGATED backtest once
      -> filter the resulting trade_log / trade_dfs by each gate
      -> re-evaluate each subset

is *exactly* equivalent to re-running the backtest once per gate, at a fraction
of the cost. 20 gate variants become 1 backtest + 20 evaluate() calls.

    IMPORTANT CAVEAT — this equivalence requires max_concurrent=None (which this
    module hardcodes, and asserts on). With max_concurrent=1 or N, removing an
    entry FREES A SLOT and admits a later trade the ungated run had skipped, so
    the surviving trades genuinely differ and you must re-run per gate. If you
    ever move the grid off unlimited stacking, use run_grid with
    gated(signal_fn, spec) instead of this module.

The second thing you get for free is VETO ATTRIBUTION: because both subsets come
from the same run, you can see the P&L of the trades each gate removed. That is
the question that actually matters — comparing gated vs ungated Calmar alone
conflates "the gate found a better sub-sample" with "the gate shrank the sample".
A gate that removes profitable trades is destroying edge no matter what happened
to the ratio; a gate that earns its keep usually does it by truncating the left
tail (visible in max_drawdown / var_95), not by raising the mean.

Usage
-----
    from Signal_Gen.regime_filter import enumerate_gate_specs
    from gate_sweep import run_gate_sweep, print_gate_attribution

    specs = enumerate_gate_specs(sizes=(1, 2))          # none + singles + pairs
    grid, attr = run_gate_sweep(
        pairs=['EURUSD', 'GBPUSD'], tenors=['1M'],
        signal_fn=_always_on, gate_specs=specs,
        legs_fn=_straddle, days_back=504,
        exit_rule_factory=lambda: HoldToExpiry())

    print_grid(grid, sort_by='calmar')                  # pair x gate, calmar-ranked
    plot_grid_heatmaps(grid)                            # same axes, per metric
    print_gate_attribution(attr)                        # what each gate removed
"""

import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest_signal import run_signal_backtest
from exit_hedge_logic import DailyHedge, HoldToExpiry
from grid_eval import CLEAN_METRICS, ComboSpec, _fmt_val
from reporting import evaluate
from Signal_Gen.regime_filter import GateSpec, build_gate


# ═══════════════════════════════════════════════════════════════════════════════
# Attribution contract — what we report about each gate's vetoes
# ═══════════════════════════════════════════════════════════════════════════════
# P&L here is in the pair's BASE currency (= pair[:3]) — read straight off the
# RAW trade_log this module holds. evaluate(to_usd=True) does now convert the
# per-trade log, but it converts its OWN copy, so this attribution is untouched.
# So it is USD for USDJPY/USDCHF/USDCAD but EUR/GBP/AUD/NZD for
# EURUSD/GBPUSD/AUDUSD/NZDUSD. That is fine here: these figures are only ever
# compared kept-vs-removed WITHIN one row, where the FX factor cancels. Do not
# compare them ACROSS pairs — the USD-comparable money metrics live in the grid
# frame.

ATTR_COLS = ['n_all', 'n_kept', 'n_removed', 'veto_rate',
             'kept_mean_pnl', 'removed_mean_pnl', 'removed_total_pnl',
             'worst_kept', 'worst_removed', 'gate_cov']


def _keep_mask(spec: GateSpec, pair: str, tenor: str, days_back: int,
               entry_ts: pd.DatetimeIndex,
               verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Which trades survive `spec`. Returns (bool mask over trade rows, gate coverage).

    Coverage is the fraction of entry dates the gate actually has a read on;
    uncovered dates fall back to spec.on_missing. A low number means the gate
    window doesn't reach the whole sample and most trades are ungated by default
    — the failure mode that makes a filter look inert. We warn rather than guess.
    """
    if not spec.checks:                                    # the 'none' baseline
        return np.ones(len(entry_ts), dtype=bool), 1.0

    gate = build_gate(spec, pair, tenor, days_back, verbose=verbose)
    g    = gate.reindex(entry_ts)
    cov  = float(g.notna().mean()) if len(g) else 1.0
    if cov < 0.95:
        print(f"[gate_sweep] WARNING: {pair} {tenor} gate '{spec.label}' covers "
              f"only {cov:.0%} of the {len(entry_ts)} entry dates; the rest "
              f"default to '{spec.on_missing}'. Widen days_back.")
    fill = 1 if spec.on_missing == 'allow' else 0
    return g.fillna(fill).astype(int).values.astype(bool), cov


def _attr_row(trade_log: pd.DataFrame, keep: np.ndarray, cov: float,
              settled_only: bool = True) -> dict:
    """Kept-vs-removed P&L attribution for one gate on one (pair, tenor)."""
    n_all = len(trade_log)
    row = {c: float('nan') for c in ATTR_COLS}
    row.update(n_all=n_all, n_kept=int(keep.sum()),
               n_removed=int((~keep).sum()), gate_cov=cov,
               veto_rate=(float((~keep).mean()) if n_all else float('nan')))
    if not n_all:
        return row

    # Mirror reporting.trade_metrics: live trades carry only a partial
    # mark-to-market, so they would bias the per-trade comparison.
    if settled_only and 'live_trade' in trade_log.columns:
        settled = ~trade_log['live_trade'].astype(bool).values
    else:
        settled = np.ones(n_all, dtype=bool)
    pnl = trade_log['net_pnl'].astype(float).values

    kept, removed = pnl[keep & settled], pnl[(~keep) & settled]
    if kept.size:
        row['kept_mean_pnl'] = float(kept.mean())
        row['worst_kept']    = float(kept.min())
    if removed.size:
        row['removed_mean_pnl']  = float(removed.mean())
        row['removed_total_pnl'] = float(removed.sum())
        row['worst_removed']     = float(removed.min())
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# The sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_gate_sweep(
    pairs:              Sequence[str],
    tenors:             Sequence[str],
    signal_fn:          Callable[[ComboSpec], pd.Series],
    gate_specs:         Sequence[GateSpec],
    *,
    legs_fn:            Optional[Callable[[ComboSpec], list]] = None,
    legs_factory_fn:    Optional[Callable[[ComboSpec], Callable]] = None,
    hedge_rule_factory: Callable[[], Any] = lambda: DailyHedge(),
    exit_rule_factory:  Callable[[], Any] = lambda: HoldToExpiry(),
    days_back:          int   = 94,
    notional:           float = 10_000_000,
    direction:          int   = -1,
    tc_fraction:        float = 0.0001,
    to_usd:             bool  = True,
    settled_only:       bool  = True,
    verbose:            bool  = True,
    progress:           bool  = False,
    gate_verbose:       bool  = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ONE backtest per (pair, tenor) on the UNGATED signal, then score every
    gate in `gate_specs` as a subset of that run. See the module docstring for
    why that is exact rather than an approximation.

    signal_fn  : the ungated signal, `fn(ComboSpec) -> Series` — the same shape
                 the grid already uses. Do NOT pre-wrap it with `gated(...)`
                 here; the gates come from `gate_specs`.
    gate_specs : list of GateSpec. Include the ungated baseline (GateSpec() /
                 NO_GATE, label 'none') so every table is self-comparing —
                 enumerate_gate_specs() does this for you by default.
    days_back  : history window, passed to signal_fn via the ComboSpec AND used
                 to build every gate, so gate and signal windows always agree.

    Returns (grid, attribution)
      grid        : same schema/index as grid_eval.run_grid — (pair, tenor, label)
                    with label = the gate name — so print_grid /
                    plot_grid_heatmaps work on it unchanged, laid out pair x gate.
      attribution : (pair, tenor, label) x ATTR_COLS — what each gate removed,
                    in the pair's quote currency. Feed to print_gate_attribution.
    """
    assert (legs_fn is None) != (legs_factory_fn is None), \
        "pass exactly one of legs_fn / legs_factory_fn"
    labels = [g.label for g in gate_specs]
    dupes  = {l for l in labels if labels.count(l) > 1}
    assert not dupes, (f"gate labels must be unique (they are the grid's column "
                       f"axis); duplicated: {sorted(dupes)}")

    metric_names = [m.name for m in CLEAN_METRICS]
    nan_metrics  = {m: float('nan') for m in metric_names}
    grid_rows: List[dict] = []
    attr_rows: List[dict] = []

    n_cells  = len(pairs) * len(tenors)
    t_sweep  = time.perf_counter()

    for ci, (pair, tenor) in enumerate(
            ((p, t) for p in pairs for t in tenors), 1):

        base = ComboSpec(pair=pair, tenor=tenor, signal_fn=signal_fn,
                         legs_fn=legs_fn, legs_factory_fn=legs_factory_fn,
                         hedge_rule_factory=hedge_rule_factory,
                         exit_rule_factory=exit_rule_factory,
                         days_back=days_back, notional=notional,
                         direction=direction, tc_fraction=tc_fraction,
                         label='none')
        cfg = {'pair': pair, 'tenor': tenor, 'days_back': days_back,
               'direction': direction, 'notional': notional}

        if verbose:
            print(f"[{ci}/{n_cells}] {pair} {tenor}: one base run + "
                  f"{len(gate_specs)} gate(s) ...", flush=True)
        t_cell = time.perf_counter()

        # ── the single shared backtest ──────────────────────────────────────
        try:
            signal       = signal_fn(base)
            legs         = legs_fn(base) if legs_fn else None
            legs_factory = legs_factory_fn(base) if legs_factory_fn else None
            trade_log, trade_dfs, _ = run_signal_backtest(
                signal, legs=legs, legs_factory=legs_factory,
                pair=pair, tenor=tenor,
                hedge_rule_factory=hedge_rule_factory,
                exit_rule_factory=exit_rule_factory,
                tc_fraction=tc_fraction, history_days=days_back,
                max_concurrent=None,          # REQUIRED — see module docstring
                verbose=False, progress=progress)
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'
            if verbose:
                print(f"      -> [error] base run failed: {msg}", flush=True)
            for spec in gate_specs:
                grid_rows.append({**cfg, 'label': spec.label, 'status': 'error',
                                  'error': msg, **nan_metrics})
                attr_rows.append({**cfg, 'label': spec.label,
                                  **{c: float('nan') for c in ATTR_COLS}})
            continue

        if trade_log is None or len(trade_log) == 0:
            if verbose:
                print("      -> [no_trades] base run produced nothing", flush=True)
            for spec in gate_specs:
                grid_rows.append({**cfg, 'label': spec.label,
                                  'status': 'no_trades', 'error': '',
                                  **nan_metrics})
                attr_rows.append({**cfg, 'label': spec.label,
                                  **{c: float('nan') for c in ATTR_COLS},
                                  'n_all': 0, 'n_kept': 0, 'n_removed': 0})
            continue

        entry_ts = pd.DatetimeIndex(pd.to_datetime(trade_log['entry_date']))
        n_base   = len(trade_log)

        # ── every gate = a subset of that one run ───────────────────────────
        for spec in gate_specs:
            row = {**cfg, 'label': spec.label, 'status': 'ok', 'error': ''}
            try:
                keep, cov = _keep_mask(spec, pair, tenor, days_back, entry_ts,
                                       verbose=gate_verbose)
                attr_rows.append({**cfg, 'label': spec.label,
                                  **_attr_row(trade_log, keep, cov,
                                              settled_only=settled_only)})

                idx = np.flatnonzero(keep)
                if idx.size == 0:
                    grid_rows.append({**row, 'status': 'no_trades', **nan_metrics})
                    continue

                tl  = trade_log.iloc[idx].reset_index(drop=True)
                tdf = [trade_dfs[i] for i in idx]
                sc  = evaluate(tdf, tl, pair=pair, to_usd=to_usd,
                               settled_only=settled_only)
                for m in CLEAN_METRICS:
                    row[m.name] = float(sc.get(m.key, float('nan')))
                grid_rows.append(row)
            except Exception as e:
                grid_rows.append({**row, 'status': 'error',
                                  'error': f'{type(e).__name__}: {e}',
                                  **nan_metrics})
                attr_rows.append({**cfg, 'label': spec.label,
                                  **{c: float('nan') for c in ATTR_COLS}})

        if verbose:
            dt  = time.perf_counter() - t_cell
            el  = time.perf_counter() - t_sweep
            eta = el / ci * (n_cells - ci)
            print(f"      -> {n_base} base trades, {len(gate_specs)} gates "
                  f"scored in {dt:.1f}s | sweep {el:.1f}s elapsed, "
                  f"~{eta:.1f}s left", flush=True)

    if verbose:
        el = time.perf_counter() - t_sweep
        n_ok = sum(1 for r in grid_rows if r.get('status') == 'ok')
        print(f"[gate_sweep] done: {len(grid_rows)} (cell x gate) rows from "
              f"{n_cells} backtest(s) in {el:.1f}s | {n_ok} ok, "
              f"{len(grid_rows) - n_ok} empty/error", flush=True)

    keys = ['pair', 'tenor', 'label']
    cfg_cols = ['days_back', 'direction', 'notional', 'status', 'error']
    grid = pd.DataFrame(grid_rows).set_index(keys)[metric_names + cfg_cols]
    grid.attrs['to_usd'] = to_usd
    attr = pd.DataFrame(attr_rows).set_index(keys)[ATTR_COLS]
    return grid, attr


# ═══════════════════════════════════════════════════════════════════════════════
# Attribution table
# ═══════════════════════════════════════════════════════════════════════════════

def print_gate_attribution(attr: pd.DataFrame, sort_by: str = 'removed_total_pnl',
                           ascending: bool = True, title: Optional[str] = None,
                           pair: Optional[str] = None) -> None:
    """
    What each gate actually removed. P&L is in the pair's QUOTE currency — the
    comparison that matters is kept vs removed within a row, where FX cancels.

    Default sort puts the most NEGATIVE removed_total_pnl first: gates that
    stripped out the biggest cumulative losers, i.e. the ones plausibly earning
    their veto. A gate with POSITIVE removed_total_pnl threw away money — no
    Calmar improvement redeems that, it just means the survivors were smoother.

    pair : optionally restrict to one pair.
    """
    view = attr.reset_index()
    if pair is not None:
        view = view[view['pair'] == pair]
    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=ascending,
                                na_position='last', kind='stable')

    fmt = {'n_all': 'count', 'n_kept': 'count', 'n_removed': 'count',
           'veto_rate': 'pct', 'gate_cov': 'pct',
           'kept_mean_pnl': 'usd', 'removed_mean_pnl': 'usd',
           'removed_total_pnl': 'usd', 'worst_kept': 'usd',
           'worst_removed': 'usd'}
    headers = ['pair', 'tenor', 'gate'] + ATTR_COLS
    body = []
    for _, r in view.iterrows():
        cells = [str(r['pair']), str(r['tenor']), str(r['label'])]
        cells += [_fmt_val(r[c], fmt.get(c, 'ratio')) for c in ATTR_COLS]
        body.append(cells)
    if not body:
        print("  (no attribution rows)")
        return

    widths = [max(len(headers[j]), *(len(b[j]) for b in body))
              for j in range(len(headers))]
    line = '  '.join(h.rjust(w) for h, w in zip(headers, widths))
    bar  = '=' * len(line)
    head = 'GATE VETO ATTRIBUTION (quote ccy)'
    if title:
        head += f'  |  {title}'
    head += f'   (ranked by {sort_by})'
    print(f"\n{bar}\n  {head}\n{bar}")
    print(f"  {line}")
    for b in body:
        print('  ' + '  '.join(x.rjust(w) for x, w in zip(b, widths)))
    print(bar)


def gate_consistency(grid: pd.DataFrame, metric: str = 'calmar',
                     baseline: str = 'none') -> pd.DataFrame:
    """
    Cross-pair consistency of each label on `metric` (default calmar), which is
    the closest thing to an out-of-sample read available inside one sweep.

    Enumerating gate combinations means many looks at one dataset, so the best
    single cell is close to meaningless. A gate that improves 5 of 6 pairs
    modestly is a regime effect; one that transforms a single pair is noise.

    Nothing here is gate-specific — it works on any grid whose `label` axis
    varies and contains a baseline column, so it serves a SIGNAL sweep
    (label = signal name, baseline = the always-on run) unchanged. For signals
    this matters more than for gates: a selective signal shrinks n_trades, which
    deflates t_stat even when per-trade edge improves, so the cross-pair hit rate
    is what has to carry the significance argument. Pass metric='ret_on_prem'
    there — it is the one edge measure invariant to `coverage`.

    baseline : label every other column is differenced against ('none' by
        convention, which is what NO_GATE is named).

    Columns: mean/median `metric` across pairs, its value change vs the baseline
    per pair, `n_better` (pairs improved), and `hit_rate`.
    """
    if metric not in grid.columns:
        raise ValueError(f"unknown metric {metric!r}")
    m = grid[metric].reset_index()
    wide = m.pivot_table(index=['pair', 'tenor'], columns='label',
                         values=metric, aggfunc='first')
    if baseline not in wide.columns:
        raise ValueError(
            f"grid has no {baseline!r} baseline column (labels present: "
            f"{sorted(wide.columns)}) — a sweep needs an unfiltered column to "
            f"measure against: include NO_GATE in gate_specs "
            f"(enumerate_gate_specs does by default), or an always-on entry in "
            f"signals=")
    delta = wide.sub(wide[baseline], axis=0)
    out = pd.DataFrame({
        f'{metric}_mean':   wide.mean(axis=0),
        f'{metric}_median': wide.median(axis=0),
        'delta_mean':       delta.mean(axis=0),
        'n_pairs':          wide.notna().sum(axis=0),
        'n_better':         (delta > 0).sum(axis=0),
    })
    out['hit_rate'] = out['n_better'] / out['n_pairs'].replace(0, np.nan)
    return out.sort_values(['hit_rate', 'delta_mean'], ascending=False)


def print_gate_consistency(grid: pd.DataFrame, metric: str = 'calmar',
                           title: Optional[str] = None,
                           baseline: str = 'none',
                           label_header: str = 'gate') -> pd.DataFrame:
    """gate_consistency() rendered as an aligned table; returns the frame.

    label_header : column heading for the label axis — pass 'signal' when the
        sweep varied signals rather than gates, so the table reads honestly.
    """
    cons = gate_consistency(grid, metric=metric, baseline=baseline)
    head = f'{label_header.upper()} CONSISTENCY on {metric} across pairs'
    if title:
        head += f'  |  {title}'
    print(f"\n{'=' * 78}\n  {head}\n{'=' * 78}")
    print(f"  {label_header:<32} {metric:>9} {'median':>9} "
          f"{f'd_vs_{baseline}':>10} {'better':>8} {'hit':>6}")
    for lbl, r in cons.iterrows():
        print(f"  {str(lbl):<32} {r[f'{metric}_mean']:>9.2f} "
              f"{r[f'{metric}_median']:>9.2f} {r['delta_mean']:>10.2f} "
              f"{int(r['n_better']):>4}/{int(r['n_pairs']):<3} "
              f"{r['hit_rate'] * 100:>5.0f}%")
    print('=' * 78)
    return cons


__all__ = ['run_gate_sweep', 'print_gate_attribution', 'gate_consistency',
           'print_gate_consistency', 'ATTR_COLS']
