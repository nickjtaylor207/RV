"""
Multi-ccy / multi-tenor grid evaluation.

Runs the SAME strategy definition (signal + legs + hedge/exit rules) across a
grid of (pair, tenor) combos, collects a curated set of comparable metrics into
one tidy DataFrame, and presents it two ways:

  - print_grid(grid)            -> ranked console table (one row per combo)
  - plot_grid_heatmaps(grid)    -> one-screen figure, a panel per metric, each
                                   a ccy x tenor colour matrix

Design:
  * evaluate() in reporting.py stays the SINGLE source of truth for every
    metric — nothing is recomputed here. CLEAN_METRICS just SELECTS a subset of
    its scorecard output and tags each with the units / colour treatment the
    display needs.
  * Money metrics are collected in USD (to_usd=True) so they are comparable
    across pairs — BOTH lenses, the daily book and the per-trade log, off the
    same daily FX factor; ratio metrics are unit-free and comparable as-is.
  * run_combo() never raises — a bad combo (missing data, zero trades) becomes a
    NaN row with a `status`, so one failure never kills the sweep.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any

import numpy as np
import pandas as pd

from backtest_signal import run_signal_backtest
from reporting import evaluate, _VIZ, _require_mpl, _compact_num


# ═══════════════════════════════════════════════════════════════════════════════
# 1. The "clean" metric contract — the ONLY definition of the comparison column set
# ═══════════════════════════════════════════════════════════════════════════════
# Each entry maps a display name -> the scorecard key evaluate() already returns,
# plus how to format it and how to colour it in the heatmap:
#   fmt   : 'ratio' | 'usd' | 'pct' | 'count' | 'greek'
#   color : 'div'  diverging around `center` (blue = high/good, red = low/bad)
#           'seq'  sequential magnitude (light -> dark blue)
#           'loss' pure-loss scale (values <= 0; more negative = darker red)
#   center: neutral point for 'div' (0 for P&L / ratios, 0.5 for win rate,
#           0 for greek exposures where colour just encodes long/short sign)
#
# NOTE on greek exposures: colour encodes SIGN (long vs short), not quality —
# a short-vol book sits on the red side by construction, not because it is "bad".

@dataclass(frozen=True)
class Metric:
    name:   str
    key:    str
    fmt:    str
    color:  str
    center: float = 0.0


CLEAN_METRICS: List[Metric] = [
    # performance / edge
    # calmar leads: it is the PRIMARY evaluation metric everywhere (default
    # print_grid sort, first heatmap panel). sortino/sharpe are still reported
    # right beside it as the secondary risk-adjusted reads.
    Metric('calmar',        'calmar',         'ratio', 'div', 0.0),
    Metric('sortino',       'sortino_ann',    'ratio', 'div', 0.0),
    Metric('sharpe',        'sharpe_ann',     'ratio', 'div', 0.0),
    Metric('net_pnl',       'book_total_pnl', 'usd',   'div', 0.0),
    # Mean P&L per trade. USD like every other money column here: evaluate(
    # to_usd=True) converts the per-trade log as well as the daily book, each
    # trade re-summed from its daily flows at that day's FX rate (to_usd_trade_log).
    # So this IS safe to rank across pairs, and expectancy * n_trades reconciles
    # with net_pnl up to settled_only dropping live trades.
    Metric('expectancy',    'expectancy',     'usd',   'div', 0.0),
    # P&L per unit of premium collected. Unit-free, so cross-pair safe, and
    # independent of `coverage` — the two properties that make it the right
    # primary once signals/gates stop the book being always-on.
    Metric('ret_on_prem',   'return_on_premium', 'ratio', 'div', 0.0),
    Metric('win_rate',      'win_rate',       'pct',   'div', 0.5),
    # avg win / avg loss. Centred on 1.0, NOT 0 — a ratio of 1 means wins and
    # losses are the same size, so a 0-centred scale would paint everything good.
    # With win_rate this pins down the whole trade distribution shape.
    Metric('payoff_ratio',  'payoff_ratio',   'ratio', 'div', 1.0),
    # tail / drawdown  (all stored signed <= 0, so higher = better)
    Metric('max_drawdown',  'max_drawdown',   'usd',   'loss'),
    # mean of the worst 5% of days, vs var_95's quantile: where the tail STARTS
    # is less useful than how bad it is once you are in it, for a short-gamma book.
    Metric('cvar_95',       'cvar_5pct_daily','usd',   'loss'),
    Metric('var_95',        'var_95_daily',   'usd',   'loss'),
    Metric('var_99',        'var_99_daily',   'usd',   'loss'),
    # REALIZED-VS-IMPLIED, in vol points — both unit-free, both from trade_metrics,
    # and both EX-POST (measured over each trade's life, then averaged). They
    # differ only in WHICH implied vol they measure against:
    #   vol_spread : realised_vol - avg_entry_sigma  (vega-weighted across legs,
    #                i.e. the vol you actually SOLD)   [backtest_MLeg.py:903]
    #   real_vrp   : realised_vol - atm_entry_vol     (pure ATM at inception,
    #                ignores where on the smile you traded) [reporting.py:376]
    # vol_spread is the more directly relevant one for P&L; the gap between them
    # is roughly the smile premium picked up by trading wings instead of ATM.
    #
    # NOTE both signs are INVERTED vs every other 'div' metric here: they are
    # realised MINUS implied, so for a short book (direction=-1) MORE NEGATIVE
    # IS BETTER. Flips if you ever run direction=+1.
    #
    # NOTE neither is an EX-ANTE read. Nothing in trade_log records the signal's
    # own entry condition, so "did the signal select rich implied?" cannot be
    # answered from the grid — inspect the signal's own pct_metric (the first
    # element of the get_*_signal triple) for that.
    Metric('vol_spread',    'avg_vol_spread', 'ratio', 'div', 0.0),
    Metric('real_vrp',      'real_VRP_ave',   'ratio', 'div', 0.0),
    # exposure (colour = sign)
    Metric('avg_net_vega',  'avg_net_vega',   'greek', 'div', 0.0),
    Metric('avg_net_gamma', 'avg_net_gamma',  'greek', 'div', 0.0),
    # sample-size context — kept beside the ratios so a flattering Sharpe on a
    # handful of overlapping trades is obvious at a glance.
    Metric('n_trades',      'n_trades',       'count', 'seq'),
    Metric('t_stat',        't_stat',         'ratio', 'div', 0.0),
    Metric('coverage',      'coverage',       'pct',   'seq'),
]

_BY_NAME = {m.name: m for m in CLEAN_METRICS}

# Headline panels for the heatmap figure (3x3). The table carries everything.
# calmar sits top-left as the primary metric; sortino/sharpe follow it.
DEFAULT_PANELS = ['calmar', 'sortino', 'sharpe',
                  'net_pnl', 'max_drawdown', 'win_rate',
                  'var_95', 'avg_net_vega', 'avg_net_gamma']

# Default columns for the console table (order matters).
DEFAULT_TABLE_COLS = ['calmar', 'sortino', 'sharpe', 'net_pnl', 'n_trades',
                      'win_rate', 'var_95', 'max_drawdown', 't_stat', 'coverage']

# Columns for comparing SIGNALS or GATES, where deployment varies between cells
# and therefore has to be read alongside everything else.
#
# Why this differs from DEFAULT_TABLE_COLS: with an always-on signal every cell is
# deployed continuously, so any column is comparable. Once a signal fires
# selectively that stops being true, and it is worth knowing exactly how the book
# handles idle time — build_daily_book indexes on the UNION of the trades' own
# date indices, so a day with no trade open is ABSENT from the book, not a zero
# row. Therefore:
#   * sharpe / sortino are already PER-DEPLOYED-DAY (no idle rows dilute the mean),
#     so they are comparable across signals as "quality per day in the market";
#   * net_pnl and n_trades still scale with how often you traded;
#   * calmar and coverage were BOTH broken by this until book_metrics was fixed to
#     work off the book's calendar span rather than its row count — see the span
#     note in reporting.book_metrics.
#
# The columns invariant to deployment — and so the ones a signal comparison rests
# on — are ret_on_prem, win_rate, payoff_ratio and real_vrp.
# calmar/sharpe stay in the table as the "what does this do to my book" read, with
# `coverage` right beside them for context.
#
# Layout: expectancy and max_drawdown lead as the per-trade / worst-case size
# pair, then the edge and book-level reads, then distribution shape, tail, the
# ex-post vol read, and the greek exposures the P&L actually came from.
#
# sortino is dropped: empirically it runs 1.1-1.2x sharpe row for row, so it
# carries no incremental information and just costs a column. var_95/var_99 are
# dropped in favour of cvar_95. t_stat is dropped because selectivity deflates it
# mechanically (see gate_sweep.py:375) — n_trades already carries the sample-size
# warning. vol_spread is dropped in favour of real_vrp alone: the two run within
# the smile premium of each other, so one ex-post vol read is enough.
#
# Every money column here is USD when the grid was built with to_usd=True (the
# default), including expectancy and the two greek exposures — see run_grid's
# currency note. avg_net_vega/avg_net_gamma still colour by SIGN, not quality: a
# short-vol book sits on the red side by construction.
#
# One thing NOT to read into a USD figure: because each day converts at that day's
# rate, usd_total/base_total is a P&L-WEIGHTED average rate, not the average spot,
# and with gains and losses partly cancelling it can land outside the spot range
# entirely. It is not a bug and it is not new (net_pnl/calmar/sharpe always worked
# this way); expectancy has simply joined them. See guide §14.3b.
SIGNAL_TABLE_COLS = ['expectancy', 'max_drawdown', 'ret_on_prem', 'net_pnl',
                     'calmar', 'sharpe',
                     'n_trades', 'coverage',
                     'win_rate', 'payoff_ratio',
                     'cvar_95',
                     'real_vrp', 'avg_net_vega', 'avg_net_gamma']


# ═══════════════════════════════════════════════════════════════════════════════
# 2. One combo = one fully-specified run
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ComboSpec:
    """One cell of the grid, fully specified so the sweep is reproducible.
    signal_fn / legs_fn / legs_factory_fn are callables that take THIS spec, so a
    pair-aware signal ('same signal per pair') is honest rather than accidental.
    Provide exactly one of legs_fn / legs_factory_fn.

    label : the grid's COLUMN axis. Defaults to `tenor`, which is why a plain
        pair x tenor sweep behaves exactly as before. Set it to name a third
        variation you want laid out side by side — a regime gate, an exit rule, a
        parameter setting — and the table/heatmaps switch their column axis to it
        automatically (see _resolve_col). When you vary BOTH tenor and label,
        include the tenor in the label (e.g. '1M|trend') so each cell is unique."""
    pair:  str
    tenor: str
    signal_fn:       Callable[['ComboSpec'], pd.Series]
    legs_fn:         Optional[Callable[['ComboSpec'], list]] = None
    legs_factory_fn: Optional[Callable[['ComboSpec'], Callable]] = None
    hedge_rule_factory: Callable[[], Any] = None
    exit_rule_factory:  Callable[[], Any] = None
    days_back:  int   = 94
    notional:   float = 10_000_000
    direction:  int   = -1
    tc_fraction: float = 0.0001
    max_concurrent: Optional[int] = None
    label:      Optional[str] = None

    def cell(self) -> str:
        """Column-axis value for this spec — the label if given, else the tenor."""
        return self.label if self.label is not None else self.tenor

    def key(self) -> tuple:
        return (self.pair, self.tenor, self.cell())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Run one combo -> one clean row (never raises)
# ═══════════════════════════════════════════════════════════════════════════════
def run_combo(spec: ComboSpec, to_usd: bool = True,
              settled_only: bool = True, progress: bool = False) -> dict:
    """Run a single (pair, tenor) and return one flat dict of clean metrics.

    to_usd : convert money metrics to USD so they are comparable across pairs
             (the whole point of a cross-ccy grid). Covers the per-trade metrics
             as well as the book-level ones — see evaluate(). Requires a USD leg
             in the pair, else that combo is recorded as a 'error' row.

    settled_only : drop still-open (live) trades from the per-trade stats
             (n_trades / expectancy / win_rate / t_stat), whose net_pnl is only
             a partial mark-to-market and would bias them. Defaults to True to
             MATCH plot_full_report / print_scorecard (scorecard()'s own
             default) — evaluate()'s default is False, so passing this through
             explicitly is what keeps the grid consistent with the single-combo
             tearsheet. Book-level metrics (sharpe, drawdown, etc.) are
             unaffected by this flag.

    On any failure or empty run, returns the same schema with NaNs plus a
    `status` ('ok' | 'no_trades' | 'error') and `error` string.
    """
    nan_metrics = {m.name: float('nan') for m in CLEAN_METRICS}
    row = {'pair': spec.pair, 'tenor': spec.tenor, 'label': spec.cell(),
           'days_back': spec.days_back, 'direction': spec.direction,
           'notional': spec.notional, 'status': 'ok', 'error': ''}
    try:
        signal = spec.signal_fn(spec)
        legs         = spec.legs_fn(spec) if spec.legs_fn else None
        legs_factory = spec.legs_factory_fn(spec) if spec.legs_factory_fn else None

        trade_log, trade_dfs, _ = run_signal_backtest(
            signal,
            legs=legs,
            legs_factory=legs_factory,
            pair=spec.pair,
            tenor=spec.tenor,
            hedge_rule_factory=spec.hedge_rule_factory,
            exit_rule_factory=spec.exit_rule_factory,
            tc_fraction=spec.tc_fraction,
            history_days=spec.days_back,
            max_concurrent=spec.max_concurrent,
            verbose=False,
            progress=progress,
        )

        if trade_log is None or len(trade_log) == 0:
            row['status'] = 'no_trades'
            return {**row, **nan_metrics}

        sc = evaluate(trade_dfs, trade_log, pair=spec.pair, to_usd=to_usd,
                      settled_only=settled_only)
        for m in CLEAN_METRICS:
            row[m.name] = float(sc.get(m.key, float('nan')))

    except Exception as e:                       # missing data, no USD leg, etc.
        row['status'], row['error'] = 'error', f'{type(e).__name__}: {e}'
        row.update(nan_metrics)
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Sweep the grid -> tidy DataFrame (rows = combos, cols = clean metrics)
# ═══════════════════════════════════════════════════════════════════════════════
def run_grid(specs: List[ComboSpec], to_usd: bool = True,
             settled_only: bool = True, verbose: bool = True,
             combo_progress: bool = False) -> pd.DataFrame:
    """Run every combo and return a tidy DataFrame indexed by (pair, tenor, label).

    Columns: the CLEAN_METRICS in contract order, then run config + status.
    Row/column order is deterministic (input order of specs, contract order of
    metrics), and axis ordering follows first-seen order for stable plots.
    `label` defaults to `tenor` (see ComboSpec.label), so a plain pair x tenor
    sweep is unchanged; it becomes the display column axis once it varies.

    to_usd : with the default True, EVERY money column in the returned frame is
             USD — book-level (net_pnl, max_drawdown, cvar_95, the greek
             exposures) and per-trade alike (expectancy), because evaluate()
             converts both lenses off one daily FX factor. Recorded on
             grid.attrs['to_usd'], which is what puts '(USD)' in print_grid's
             header. Ratio columns (ret_on_prem, calmar, sharpe, win_rate,
             payoff_ratio) and the vol-point column (real_vrp) are unit-free
             either way. Set False only for a single-pair base-ccy read.

    settled_only : see run_combo — defaults to True to match plot_full_report.
    verbose : print per-combo timing (elapsed, status, trades, running sweep
              total + ETA) and an end summary. This is the grid-level tracker.
    combo_progress : additionally turn on run_signal_backtest's per-trade
              heartbeat INSIDE each combo (dataset-pull time, loop ETA). Off by
              default — verbose per-combo lines are usually enough; switch on
              when a single combo is slow and you want to see inside it.
    """
    rows = []
    n = len(specs)
    t_sweep = time.perf_counter()
    for i, s in enumerate(specs, 1):
        if verbose:
            cell = s.cell()
            tag  = f"{s.pair} {s.tenor}" + (f" [{cell}]" if cell != s.tenor else "")
            print(f"[{i}/{n}] {tag} ...", flush=True)
        t_combo = time.perf_counter()
        row = run_combo(s, to_usd=to_usd, settled_only=settled_only,
                        progress=combo_progress)
        rows.append(row)
        if verbose:
            dt   = time.perf_counter() - t_combo
            el   = time.perf_counter() - t_sweep
            eta  = el / i * (n - i)
            nt   = row.get('n_trades', float('nan'))
            ntxt = '' if nt != nt else f"{int(nt)} trades, "     # nt!=nt -> NaN
            stat = row.get('status', 'ok')
            stxt = '' if stat == 'ok' else f"[{stat}] "
            print(f"      -> {stxt}{ntxt}{dt:.1f}s | "
                  f"sweep {el:.1f}s elapsed, ~{eta:.1f}s left", flush=True)

    if verbose:
        el = time.perf_counter() - t_sweep
        n_ok  = sum(1 for r in rows if r.get('status') == 'ok')
        n_bad = n - n_ok
        print(f"[grid] done: {n} combos in {el:.1f}s "
              f"({el/max(n,1):.1f}s/combo avg) | {n_ok} ok, {n_bad} empty/error",
              flush=True)

    df = pd.DataFrame(rows)
    metric_cols = [m.name for m in CLEAN_METRICS]
    cfg_cols    = ['days_back', 'direction', 'notional', 'status', 'error']
    df = df.set_index(['pair', 'tenor', 'label'])[metric_cols + cfg_cols]
    df.attrs['to_usd'] = to_usd
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Formatting helpers (shared by table + heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
def _fmt_val(v: float, fmt: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'nan'
    if fmt == 'ratio':
        return f"{v:.2f}"
    if fmt == 'pct':
        return f"{v*100:.0f}%"
    if fmt == 'count':
        return f"{v:.0f}"
    return _compact_num(v)                        # usd / greek -> k/M


def _resolve_col(grid: pd.DataFrame, col: str = 'auto') -> str:
    """
    Which index level to lay out as the matrix/table COLUMN axis.

    'auto' (default) picks 'label' when it actually varies — i.e. you swept a
    third dimension such as regime gates — and 'tenor' otherwise. That keeps a
    plain pair x tenor sweep looking exactly as it always did, while a gate sweep
    renders as pair x gate with no extra arguments at the call site.
    """
    if col != 'auto':
        if col not in grid.index.names:
            raise ValueError(f"col must be one of {list(grid.index.names)}, got {col!r}")
        return col
    if 'label' in grid.index.names:
        labels = grid.index.get_level_values('label')
        tenors = grid.index.get_level_values('tenor')
        if len(set(labels)) > 1 and not (labels == tenors).all():
            return 'label'
    return 'tenor'


def _axes(grid: pd.DataFrame, col: str = 'tenor'):
    """First-seen order of row (pair) and column values, for stable layout."""
    pairs = list(dict.fromkeys(grid.index.get_level_values('pair')))
    cols  = list(dict.fromkeys(grid.index.get_level_values(col)))
    return pairs, cols


def _pairs_tenors(grid: pd.DataFrame):
    """Back-compat alias: pairs x tenors axes."""
    return _axes(grid, 'tenor')


def _matrix(grid: pd.DataFrame, metric: str, pairs, cols,
            col: str = 'tenor') -> pd.DataFrame:
    """metric column -> pairs(rows) x `col`(cols) matrix in stable order."""
    view = grid[metric].reset_index()
    dup  = view.duplicated(subset=['pair', col]).any()
    if dup:
        raise ValueError(
            f"grid has multiple rows per (pair, {col}) — the matrix would be "
            f"ambiguous. You are varying tenor AND label; put the tenor into the "
            f"label (e.g. label='1M|trend') so each cell is unique.")
    m = view.pivot(index='pair', columns=col, values=metric)
    return m.reindex(index=pairs, columns=cols)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Console table — ranked one row per combo
# ═══════════════════════════════════════════════════════════════════════════════
def print_grid(grid: pd.DataFrame, sort_by: str = 'calmar',
               ascending: bool = False, cols: Optional[List[str]] = None,
               title: Optional[str] = None) -> None:
    """Aligned, ranked table of the grid, ranked by `sort_by` (default calmar —
    the primary evaluation metric). Money columns are USD when the grid was built
    with to_usd=True (see grid.attrs['to_usd']).

    A 'label' key column is shown whenever it carries information beyond the
    tenor (a gate sweep, an exit sweep, ...)."""
    cols = cols or DEFAULT_TABLE_COLS
    unit = ' (USD)' if grid.attrs.get('to_usd') else ''
    view = grid.reset_index()
    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=ascending,
                                na_position='last', kind='stable')

    show_label = _resolve_col(grid) == 'label'
    keys = ['pair', 'tenor'] + (['label'] if show_label else [])
    headers = keys + cols + ['status']
    fmts = {c: _BY_NAME[c].fmt for c in cols if c in _BY_NAME}
    body = []
    for _, r in view.iterrows():
        cells = [str(r[k]) for k in keys]
        cells += [_fmt_val(r[c], fmts.get(c, 'ratio')) for c in cols]
        cells += [str(r.get('status', ''))]
        body.append(cells)

    widths = [max(len(headers[j]), *(len(b[j]) for b in body)) for j in range(len(headers))]
    line = '  '.join(h.rjust(w) for h, w in zip(headers, widths))
    bar = '=' * len(line)
    print(f"\n{bar}")
    head = f"GRID SCORECARD{unit}"
    if title:
        head += f'  |  {title}'
    head += f'   (ranked by {sort_by})'
    print(f"  {head}")
    print(bar)
    print(f"  {line}")
    for b in body:
        print('  ' + '  '.join(x.rjust(w) for x, w in zip(b, widths)))
    print(bar)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Heatmap figure — one panel per metric, ccy x tenor colour matrix
# ═══════════════════════════════════════════════════════════════════════════════
# Diverging pair (blue <-> red, gray midpoint) and sequential/loss ramps built
# from the data-viz reference palette. Colours are computed per panel, never
# eyeballed; each panel is scaled to its own metric.
def _build_cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    # low = red (bad) .. gray midpoint .. high = blue (good)
    div  = LinearSegmentedColormap.from_list('grid_div',  ['#b02525', '#f0efec', '#2a78d6'])
    seq  = LinearSegmentedColormap.from_list('grid_seq',  ['#e8f1fd', '#3987e5', '#184f95'])
    loss = LinearSegmentedColormap.from_list('grid_loss', ['#b02525', '#e7a6a6', '#f3eeec'])
    for cm in (div, seq, loss):
        cm.set_bad(_VIZ['grid'])                  # NaN cells -> neutral gray
    return {'div': div, 'seq': seq, 'loss': loss}


def _norm_for(meta: Metric, finite: np.ndarray):
    """Pick a matplotlib norm for one panel's finite values."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    if finite.size == 0:
        return Normalize(0.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    if meta.color == 'div':
        c = meta.center
        vmin = min(lo, c) - (1e-9 if lo >= c else 0.0)
        vmax = max(hi, c) + (1e-9 if hi <= c else 0.0)
        # TwoSlopeNorm needs vmin < vcenter < vmax strictly.
        if vmin >= c:
            vmin = c - max(abs(c), 1.0) * 1e-6 - 1e-9
        if vmax <= c:
            vmax = c + max(abs(c), 1.0) * 1e-6 + 1e-9
        return TwoSlopeNorm(vcenter=c, vmin=vmin, vmax=vmax)
    if meta.color == 'loss':
        # values <= 0; most negative (worst) -> dark end of the loss ramp.
        vmax = min(hi, 0.0)
        vmin = lo if lo < vmax else vmax - 1e-9
        return Normalize(vmin, vmax)
    # sequential magnitude
    return Normalize(lo, hi if hi > lo else lo + 1e-9)


def _text_ink(rgba) -> str:
    """Readable label ink for a given cell colour (white on dark, ink on light)."""
    r, g, b, _ = rgba
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return _VIZ['ink'] if lum > 0.6 else 'white'


def _draw_heatmap_panel(ax, mat: pd.DataFrame, meta: Metric, cmap, norm) -> None:
    data = mat.values.astype(float)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect='auto')
    # Disable the toolbar's hover value-readout: for one-sided diverging panels
    # the norm's empty arm inverts to +/-inf, which crashes matplotlib's
    # cursor-data digit formatter (math.floor(log10(inf))). Every cell is
    # already labelled with its value, so the readout is redundant anyway.
    im.get_cursor_data = lambda event: None

    nrows, ncols = data.shape
    # Gate/label columns are far longer than tenor codes — angle them so they
    # stay readable instead of overlapping.
    labs = [str(c) for c in mat.columns]
    rot  = 40 if max((len(l) for l in labs), default=0) > 4 else 0
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(labs, fontsize=8, rotation=rot,
                       ha='right' if rot else 'center')
    ax.set_yticks(range(nrows)); ax.set_yticklabels(mat.index, fontsize=8)
    ax.tick_params(length=0, colors=_VIZ['ink2'])
    for side in ax.spines.values():
        side.set_visible(False)
    # 2px surface gap between cells (data-viz mark spec).
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which='minor', color=_VIZ['surface'], linewidth=2)
    ax.tick_params(which='minor', length=0)

    for i in range(nrows):
        for j in range(ncols):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, '·', ha='center', va='center',
                        color=_VIZ['muted'], fontsize=9)
                continue
            ink = _text_ink(cmap(norm(v)))
            ax.text(j, i, _fmt_val(v, meta.fmt), ha='center', va='center',
                    color=ink, fontsize=8.5)
    ax.set_title(meta.name, fontsize=10, color=_VIZ['ink'], fontweight='bold', pad=6)


def plot_grid_heatmaps(grid: pd.DataFrame, panels: Optional[List[str]] = None,
                       title: Optional[str] = None, ncols: int = 3,
                       save_path: Optional[str] = None, show: bool = False,
                       col: str = 'auto'):
    """One-screen figure: a heatmap panel per metric, each a ccy x `col` matrix.

    panels : metric names to show (default DEFAULT_PANELS, calmar first).
             Unknown names skipped.
    col    : column axis — 'auto' (default) uses 'label' when you swept a third
             dimension such as regime gates, else 'tenor'. See _resolve_col.
    Returns the matplotlib Figure. save_path / show mirror plot_full_report.
    """
    _require_mpl()
    import matplotlib.pyplot as plt

    panels = [p for p in (panels or DEFAULT_PANELS) if p in _BY_NAME]
    if not panels:
        raise ValueError("no valid panels to plot.")
    col_key      = _resolve_col(grid, col)
    pairs, cells = _axes(grid, col_key)
    cmaps = _build_cmaps()

    n = len(panels)
    nrows_fig = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows_fig, ncols,
                             figsize=(max(4.6, 1.1 * len(cells) + 1.6) * ncols,
                                      0.55 * len(pairs) * nrows_fig + 1.4))
    fig.patch.set_facecolor(_VIZ['surface'])
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, panels):
        meta = _BY_NAME[name]
        mat = _matrix(grid, name, pairs, cells, col_key)
        finite = mat.values[np.isfinite(mat.values.astype(float))]
        norm = _norm_for(meta, finite.astype(float))
        _draw_heatmap_panel(ax, mat, meta, cmaps[meta.color], norm)
    for ax in axes[n:]:                            # hide unused cells
        ax.set_visible(False)

    unit = ' (USD)' if grid.attrs.get('to_usd') else ''
    head = f'Grid comparison{unit}' + (f'  |  {title}' if title else '')
    fig.suptitle(head, fontsize=14, color=_VIZ['ink'], fontweight='bold',
                 x=0.01, ha='left', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=_VIZ['surface'], bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Phase robustness — the SAME strategy scored over several sample start dates
# ═══════════════════════════════════════════════════════════════════════════════
# Any single backtest window is one draw. Its calmar/sharpe/expectancy depend on
# where the sample happens to begin, and a window that starts just before or just
# after one large move can move a metric enough to change the conclusion. Running
# the identical spec over several `days_back` values and reading the SPREAD is
# what separates "this strategy works" from "this start date works".
#
# READ THE SPREAD CORRECTLY — the windows are NESTED, not independent:
#   Every signal builder pulls start = today - days_back, end = today (see
#   Implied_Realized.get_ImplRealVol). So all phases END on the same day and
#   overlap on everything except the earliest stretch. days_back=1825 vs 1633
#   share ~89% of their history. The phases are therefore heavily correlated:
#     * std here is a SENSITIVITY measure — "how much does this metric move when
#       I chop the first N days off" — NOT a standard error, and NOT sqrt(n)
#       shrinkable. Do not build a t-stat or a confidence interval out of it.
#     * A small spread is real evidence of stability. A LARGE spread is a red
#       flag, because it means a metric moved a lot on a small change to a shared
#       sample — which is the bias you are testing for.
#   A genuinely independent test needs disjoint windows (a fixed-length window
#   that slides its END date too). Nothing in the pull supports an end date other
#   than today, so that is not available from this route.
#
# AND MIND THE WINDOW LENGTH: different days_back means different sample lengths,
# so _WINDOW_SCALED metrics move for a mechanical reason on top of any real
# instability — a shorter window simply has fewer trades and less cumulative P&L,
# and fewer chances to print a deep drawdown. Their spread is not comparable to
# the spread of the length-invariant metrics (expectancy, ret_on_prem, calmar,
# sharpe, win_rate, payoff_ratio, real_vrp), which are the honest robustness read.

PHASE_SUMMARY_COLS = ['expectancy', 'max_drawdown', 'ret_on_prem', 'net_pnl',
                      'calmar', 'sharpe', 'n_trades', 'win_rate',
                      'payoff_ratio', 'cvar_95', 'real_vrp']

PHASE_STATS = ['mean', 'std', 'min', 'max', 'n']

# Metrics whose LEVEL is a function of how long the window is (see note above).
_WINDOW_SCALED = {'net_pnl', 'n_trades', 'max_drawdown'}


def _as_grid(g) -> pd.DataFrame:
    """RUN's gates route returns (grid, attr) — accept either shape."""
    return g[0] if isinstance(g, tuple) else g


def summarize_phases(grids, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Stack N phase grids and reduce each cell to mean/std/min/max/n per metric.

    grids : dict {phase_label: grid} (insertion order preserved) or a list of
            (phase_label, grid) pairs. A (grid, attr) tuple from the gates route
            is unwrapped automatically.
    cols  : metrics to summarise (default PHASE_SUMMARY_COLS).

    Returns a frame indexed by (pair, tenor, label) — the grid's own cell key, so
    every pair/tenor/signal/gate cell is summarised SEPARATELY and nothing is
    averaged across cells — with MultiIndex columns (metric, stat).

    Cells that failed or found no trades in a given phase are NaN there and are
    skipped by the reduction rather than counted as zero, which is why `n` is
    reported per metric: n < len(grids) means that cell is summarised off fewer
    phases than you asked for. std is ddof=1, so n=1 gives NaN by design.
    """
    items = list(grids.items()) if isinstance(grids, dict) else list(grids)
    if not items:
        raise ValueError("summarize_phases needs at least one grid")
    items = [(str(lbl), _as_grid(g)) for lbl, g in items]
    cols = list(cols or PHASE_SUMMARY_COLS)

    frames = []
    for lbl, g in items:
        missing = [c for c in cols if c not in g.columns]
        if missing:
            raise ValueError(f"phase {lbl!r} is missing column(s) {missing}")
        f = g[cols].copy()
        f['phase'] = lbl
        frames.append(f.set_index('phase', append=True))
    panel = pd.concat(frames)

    # sort=False keeps first-seen cell order, matching print_grid's layout.
    grp = panel.groupby(level=['pair', 'tenor', 'label'], sort=False)
    parts = [grp.mean(), grp.std(ddof=1), grp.min(), grp.max(), grp.count()]
    out = pd.concat(parts, axis=1, keys=PHASE_STATS).swaplevel(axis=1)
    out = out.reindex(columns=pd.MultiIndex.from_product([cols, PHASE_STATS]))
    out.attrs['to_usd'] = bool(items[0][1].attrs.get('to_usd'))
    out.attrs['phases'] = [lbl for lbl, _ in items]
    return out


def _fmt_stat(v: float, metric: str, stat: str) -> str:
    """Format one summary cell. Counts and percentages need more precision on the
    spread stats than on the level — a std of 3.4 trades must not print as '3',
    and a win-rate std of 1.6% must not print as '2%'."""
    if stat == 'n':
        return 'nan' if v != v else f"{v:.0f}"
    fmt = _BY_NAME[metric].fmt if metric in _BY_NAME else 'ratio'
    if stat == 'std':
        if fmt == 'count':
            return 'nan' if v != v else f"{v:.1f}"
        if fmt == 'pct':
            return 'nan' if v != v else f"{v*100:.1f}%"
    return _fmt_val(v, fmt)


def _compact_rows(summary: pd.DataFrame, metrics: List[str]) -> tuple:
    """One row per grid cell, every metric as `mean [std]`.

    The spread travels WITH the level rather than on its own row, which is the
    whole point: a calmar of 1.20 means something different at [0.04] than at
    [0.61], and reading that off two rows several lines apart does not work.
    min/max are dropped here — they are the same information as std at lower
    resolution, and detail=True still has them.
    """
    headers = list(summary.index.names) + metrics + ['n']
    body = []
    for idx, row in summary.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        cells  = [str(x) for x in idx]
        cells += [f"{_fmt_stat(row[(m, 'mean')], m, 'mean')} "
                  f"[{_fmt_stat(row[(m, 'std')], m, 'std')}]" for m in metrics]
        # per-metric n, collapsed to its worst case: run_combo NaNs a failed
        # cell's whole row, so in practice every metric shares one n. Taking the
        # min means a cell that IS ragged under-reports rather than over-reports;
        # detail=True breaks n out per metric.
        cells += [_fmt_stat(min(row[(m, 'n')] for m in metrics), '', 'n')]
        body.append(cells)
    return headers, body


def _detail_rows(summary: pd.DataFrame, metrics: List[str]) -> tuple:
    """One block per grid cell, one row per stat in PHASE_STATS."""
    headers = list(summary.index.names) + ['stat'] + metrics
    body = []
    for idx, row in summary.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        for i, st in enumerate(PHASE_STATS):
            # key columns printed once per block, so the eye tracks the stat rows
            cells  = [str(x) if i == 0 else '' for x in idx]
            cells += [st]
            cells += [_fmt_stat(row[(m, st)], m, st) for m in metrics]
            body.append(cells)
        body.append([''] * len(headers))            # blank line between cells
    if body:
        body.pop()
    return headers, body


def print_phase_summary(summary: pd.DataFrame, title: Optional[str] = None,
                        detail: bool = False, # Toggle True for full stats
                        note: bool = False
                        ) -> None:
    """Aligned table of summarize_phases output. Money columns are USD when the
    phase grids were built to_usd=True.

    detail=False (default) : ONE row per grid cell, each metric as `mean [std]`
             across the sample start dates, with the phase count `n` last.
    detail=True            : one row per stat (mean/std/min/max/n) per cell — the
             full breakdown, and the only view carrying min/max and per-metric n.
    note=True              : print how to read the spread (see module note §8).
    """
    metrics = list(dict.fromkeys(summary.columns.get_level_values(0)))
    headers, body = (_detail_rows if detail else _compact_rows)(summary, metrics)

    widths = [max([len(headers[j])] + [len(b[j]) for b in body])
              for j in range(len(headers))]
    line = '  '.join(h.rjust(w) for h, w in zip(headers, widths))
    bar  = '=' * len(line)

    unit   = ' (USD)' if summary.attrs.get('to_usd') else ''
    phases = summary.attrs.get('phases', [])
    head   = f"PHASE ROBUSTNESS{unit}"
    if title:
        head += f'  |  {title}'
    head += f'   ({len(phases)} sample start dates)'
    print(f"\n{bar}")
    print(f"  {head}")
    if phases:
        print(f"  phases: {', '.join(phases)}")
    if not detail:
        print("  cells: mean [std] across start dates   (detail=True for min/max)")
    print(bar)
    print(f"  {line}")
    for b in body:
        # ASCII only in printed output — a cp1252 console mangles em-dashes.
        print('' if not any(b) else
              '  ' + '  '.join(x.rjust(w) for x, w in zip(b, widths)))
    print(bar)

    if note:
        spread = 'std/min/max are' if detail else 'the [std] beside each mean is'
        print(f"  {spread} ACROSS SAMPLE START DATES, not a standard error:")
        print("  every phase ends today and shares the recent history, so the windows")
        print("  are nested and correlated. Read the spread as sensitivity to where")
        print("  the sample begins - tight = stable, wide = start-date luck.")
        scaled = [m for m in metrics if m in _WINDOW_SCALED]
        if scaled:
            print(f"  {', '.join(scaled)} also scale with window LENGTH, so part of their")
            print("  spread is mechanical. Judge robustness on the length-invariant ones.")
        print("  n < phase count means that cell was empty/errored in some phases.")


__all__ = ['Metric', 'CLEAN_METRICS', 'ComboSpec', 'run_combo', 'run_grid',
           'print_grid', 'plot_grid_heatmaps', 'DEFAULT_PANELS',
           'DEFAULT_TABLE_COLS', 'SIGNAL_TABLE_COLS',
           'PHASE_SUMMARY_COLS', 'PHASE_STATS', 'summarize_phases',
           'print_phase_summary']
