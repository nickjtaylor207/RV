"""
dashboard.py -- one page for a greek-target book.

LAYOUT
------
    SCORECARD  |  equity + drawdown + worst days
               |  vega | gamma | vanna | volga     (4 across, vs target)
    WORST DAYS |  BE cumulative + vega leakage | daily P&L dist | the trades

`run.report.report()` stays the searchable deep-dive; this is the one-glance
view. Call `dashboard(...)` for the figure and `report(...)` for the tiers.


WHY THERE IS NO PER-TRADE PANEL
-------------------------------
The reference dashboards this replaces carried expectancy / win_rate /
payoff_ratio / profit_factor / cvar(trade) / worst-trades / a per-trade
histogram. Those were coherent in the OLD stack because every trade hedged
itself. Here they are not computable honestly, for two independent reasons:

  1. A position reports only its own OPTION P&L. The delta hedge is NETTED at
     book level and lives in `res.hedges` -- book/attribution.py calls that
     separation "the main reason for the rewrite". On the real USDCAD run
     hedge_pnl was -2,295,144 against delta_pnl of +2,295,144, so a per-trade
     P&L would omit the larger half of the position's economics, and there is no
     non-arbitrary way to allocate a netted hedge back to overlapping vintages.

  2. Under mode='top_up' a "trade" is an INCREMENT sized by whatever decayed
     since the last roll -- not a decision with a thesis. Even the count is
     ambiguous: 58 rolls, 232 legs, 216 early closes, all on one strategy.

So the unit of account is the BOOK-DAY, and "understanding the trades" becomes
trade BEHAVIOUR (cadence, increment size, structure mix, turnover, exit mix)
rather than trade P&L. That is the bottom-right panel and the EXECUTION block.


THE TWO UNIT SYSTEMS, AND WHY THE TOGGLE MATTERS
------------------------------------------------
`units='normalised'` plots base-ccy P&L per one-sigma move over `horizon` --
the units the target was written in, so the target is a flat line and the four
panels share a dimension.

`units='trader'` plots GreekVector.as_trader_units(). Read that method's
warning: gamma and vanna come out in NOTIONAL while vega and volga come out in
MONEY. The four panels are then in two different unit families and are NOT
commensurable -- so no y-axis is ever shared across them and every panel is
labelled with its own unit.

A second thing the toggle exposes: a CONSTANT normalised target is a MOVING
target in raw or trader units, because the conversion carries sigma and nu and
both move daily. In trader mode the target is drawn as a line, not a rule. That
wobble is the whole argument for normalising in the first place.

Static images have no toggle, so `units='both'` writes two files.


COLOR
-----
Reference palette, fixed slot order, adjacent pairlist -- the case that palette
validates in both light and dark. Slots are never cycled: a fifth tag folds
into "other" rather than inventing a hue. Every series carries a direct end
label or a legend, so identity is never colour-alone.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.greeks import GreekVector, SPOT_MOVE
from run.report import (PALETTE, GREEK_COL, risk_tracking, _roll_dates,
                        _costs_are_off, _style, _finish, _label_ends, _end_of)


# Converters from a RAW greek to trader units. These mirror
# GreekVector.as_trader_units exactly; that method is the single source of truth
# and is called directly for the plotted series. These exist only to put the
# TARGET into the same space as the line.
_TRADER_FROM_RAW = {
    'vega':  lambda raw, S: raw,
    'volga': lambda raw, S: 2.0 * raw,
    'gamma': lambda raw, S: 2.0 * raw / (SPOT_MOVE * S),
    'vanna': lambda raw, S: raw / (SPOT_MOVE * S),
}
_TRADER_KEY  = {'vega': 'vega_1vp', 'volga': 'volga_1vp',
                'gamma': 'gamma_1pct', 'vanna': 'vanna_1vp'}
_TRADER_UNIT = {'vega': 'base ccy / 1vp', 'volga': 'base ccy / 1vp^2',
                'gamma': 'notional / 1%', 'vanna': 'notional / 1vp'}
_NORM_UNIT   = 'base ccy / 1 sigma'

# Panel order across the middle row, and its fixed colour slot.
MIDDLE = (('vega', 's2'), ('gamma', 's4'), ('vanna', 's3'), ('volga', 's1'))


# ===================================================================== #
# Data prep
# ===================================================================== #
def trader_exposures(res, d: pd.DataFrame) -> pd.DataFrame:
    """
    Add `trader_<greek>` columns to a reset daily frame.

    Rebuilds a GreekVector per day from the net_* columns and calls
    `as_trader_units()` rather than re-deriving those conventions here.
    """
    spot = (res.positions.groupby('date')['spot'].first()
            if len(res.positions) and 'spot' in res.positions.columns
            else pd.Series(dtype=float))
    sp = d['date'].map(spot)
    fields = dict(spot_1pct='net_spot_1pct', gamma_1pct='net_gamma_1pct',
                  vega_1vp='net_vega_1vp', volga_1vp='net_volga_1vp',
                  vanna_1pct_1vp='net_vanna_1pct_1vp',
                  theta_1d='net_theta_1d', delta_hedge='net_delta_hedge')
    rows = []
    for i in range(len(d)):
        kw = {k: float(d[v].iloc[i]) for k, v in fields.items()
              if v in d.columns}
        s_i = sp.iloc[i]
        rows.append(GreekVector(
            **kw, spot=float(s_i) if pd.notna(s_i) else np.nan
        ).as_trader_units())
    t = pd.DataFrame(rows)
    for g, key in _TRADER_KEY.items():
        if key in t.columns:
            d['trader_' + g] = t[key].values
    d['spot_used'] = sp.values
    return d


def constrained_targets(roller) -> Dict[str, float]:
    """{greek: target} for the greeks the roller actually pinned or sized."""
    out: Dict[str, float] = {}
    if roller is None:
        return out
    df = roller.frame()
    if not len(df):
        return out
    for col in df.columns:
        if not col.startswith('want_'):
            continue
        g = col[len('want_'):].split('@')[0]
        v = df[col].dropna()
        if len(v):
            out[g] = float(v.iloc[0])
    return out


def held_days(res) -> Dict[str, float]:
    """Hold-time stats. Only res.book.positions carries entry/exit dates."""
    held, expired, n = [], 0, 0
    for p in res.book.positions.values():
        if p.exit_date is None:
            continue
        n += 1
        held.append((p.exit_date - p.entry_date).days)
        expired += int(p.exit_reason == 'expiry')
    if not n:
        return {'avg_days_held': np.nan, 'pct_to_expiry': np.nan, 'n': 0}
    return {'avg_days_held': float(np.mean(held)),
            'pct_to_expiry': expired / n, 'n': n}


# ===================================================================== #
# Scorecard
# ===================================================================== #
def scorecard(res, d: pd.DataFrame, budget: Optional[float] = None,
              roller=None, greek: str = 'volga',
              band_frac: float = 0.20) -> List[str]:
    """
    Condensed scorecard, ordered by what should drive a decision.

    HARVEST first, RISK-ADJUSTED last and deliberately: sharpe / sortino /
    calmar are statistics of a single path, and engine/loop.py's own note says
    pnl_per_unit_greek "ranks sleeves and configurations far better than calmar,
    which rests on a single order statistic from one path".

    There is no capital base anywhere in this stack, so the ratios are computed
    on daily P&L and labelled (P&L) rather than dressed up as returns.
    """
    D, L = res.daily, []
    A = L.append
    pnl, eq = D['pnl'], D['equity']
    dd = eq - eq.cummax()
    nc = 'norm_' + greek
    live = (d[(d[GREEK_COL[greek]].abs() > 1e-9) & d[nc].notna()]
            if nc in d.columns else d.iloc[0:0])

    A('HARVEST   premium per unit of the risk that earned it')
    for pc, gc, nm in (('volga_pnl_be', 'volga_1vp',      'volga_be/|volga|'),
                       ('gamma_pnl_be', 'gamma_1pct',     'gamma_be/|gamma|'),
                       ('vanna_pnl_be', 'vanna_1pct_1vp', 'vanna_be/|vanna|')):
        A(f'  {nm:<21}{res.pnl_per_unit_greek(pc, gc):>13.4f}')
    vb = D['volga_pnl_be'].sum()
    A(f'  {"vega leak / volga_be":<21}'
      f'{(D["vega_pnl"].sum() / vb if vb else np.nan):>13.1%}')
    prem = res.trades['entry_premium'].sum() if len(res.trades) else 0.0
    A(f'  {"return_on_premium":<21}'
      f'{(pnl.sum() / abs(prem) if prem else np.nan):>13.4f}')

    A('')
    A('TRACKING   did I hold what I said?')
    if len(live) and budget is not None:
        err = (live[nc] - budget).abs()
        A(f'  {"target " + greek:<21}{budget:>13,.0f}')
        A(f'  {"mean |err|":<21}{_k(err.mean()):>13}'
          f'{err.mean() / abs(budget):>11.1%}')
        A(f'  {"p90 / peak |err|":<21}{_k(err.quantile(0.9)):>13}'
          f'{_k(err.max()):>11}')
        A(f'  {("days within " + format(band_frac, ".0%")):<21}'
          f'{(err <= band_frac * abs(budget)).mean():>13.1%}')
        rd = _roll_dates(roller)
        if rd:
            prev = (d['date'].shift(1).isin(rd)
                    .reindex(live.index).fillna(False))
            if prev.any() and (~prev).any():
                A('  post-roll / other:')
                for g in ('vega', 'vanna'):
                    c = 'norm_' + g
                    if c in live.columns:
                        A(f'    {("|" + g + "|"):<19}'
                          f'{_k(live[c][prev].abs().mean()):>11}'
                          f'{_k(live[c][~prev].abs().mean()):>11}')
    else:
        A('  (no budget -- pass budget= or a roller)')

    A('')
    A('EXECUTION')
    if roller is not None and len(roller.frame()):
        lg = roller.frame()
        A('  rolls ' + ' '.join(f'{k}={v}' for k, v in
                                sorted(lg['action'].value_counts().items())))
        tr = lg[lg['action'] == 'traded']
        if len(tr) > 1:
            gaps = pd.Series(pd.to_datetime(tr['date'])).diff().dt.days
            A(f'  {"cadence obs / cfg":<21}{gaps.mean():>13.1f}'
              f'{roller.roll_days * 7 / 5:>9.1f}')
        if 'leverage' in lg.columns and lg['leverage'].notna().any():
            A(f'  {"leverage mean / max":<21}{lg["leverage"].mean():>13.1f}'
              f'{lg["leverage"].max():>9.1f}')
        if 'cond' in lg.columns and lg['cond'].notna().any():
            A(f'  {"cond max":<21}{lg["cond"].max():>13.1f}')
        if len(tr):
            A(f'  {"legs per roll":<21}{tr["legs"].mean():>13.1f}')
            A(f'  {"gross per roll":<21}{_k(tr["gross"].mean()):>13}')
    hd = held_days(res)
    A(f'  {"avg days held":<21}{hd["avg_days_held"]:>13.1f}')
    A(f'  {"pct held to expiry":<21}{hd["pct_to_expiry"]:>13.1%}')
    A(f'  {"n_open med / max":<21}{D["n_open"].median():>13.0f}'
      f'{D["n_open"].max():>11.0f}')

    A('')
    A('COST')
    A(f'  {"option_tc / hedge_tc":<21}{_k(D["option_tc"].sum()):>13}'
      f'{_k(D["hedge_tc"].sum()):>11}')
    carried = res.greek_carried(GREEK_COL[greek][4:]).abs().mean()
    tc = D['option_tc'].sum() + D['hedge_tc'].sum()
    A(f'  {("tc/unit |" + greek + "| (opt+spot)"):<21}'
      f'{(tc / carried if carried else np.nan):>13.4f}')
    if _costs_are_off(res):
        A('  !! cost_model DISABLED')
        A('     this run is GROSS of spread')

    A('')
    A('TAIL')
    A(f'  {"sd / skew (daily)":<21}{_k(pnl.std()):>13}{pnl.skew():>11.2f}')
    q5 = pnl.quantile(0.05)
    A(f'  {"VaR95 / CVaR5":<21}{_k(q5):>13}'
      f'{_k(pnl[pnl <= q5].mean()):>11}')
    A(f'  {"max_dd / dd days":<21}{_k(dd.min()):>13}'
      f'{int((dd < 0).sum()):>11d}')
    r = res.positions['recon_resid']
    A(f'  {"recon mean|r| / peak":<21}{_k(r.abs().mean()):>13}'
      f'{_k(r.abs().max()):>11}')
    A(f'  {"recon % of |opt_pnl|":<21}'
      f'{r.abs().sum() / max(res.positions["option_pnl"].abs().sum(), 1):>13.2%}')

    A('')
    A('RISK-ADJUSTED   one path -- read last')
    ann, down = np.sqrt(252.0), pnl[pnl < 0].std()
    A(f'  {"sharpe_ann (P&L)":<21}'
      f'{(pnl.mean() / pnl.std() * ann if pnl.std() else np.nan):>13.4f}')
    A(f'  {"sortino_ann (P&L)":<21}'
      f'{(pnl.mean() / down * ann if down else np.nan):>13.4f}')
    A(f'  {"calmar (P&L)":<21}'
      f'{(pnl.sum() / abs(dd.min()) if dd.min() else np.nan):>13.4f}')
    A(f'  {"final equity":<21}{_k(eq.iloc[-1]):>13}')
    return L


def worst_days_block(res, d: pd.DataFrame, n: int = 3) -> List[str]:
    """
    WORST n DAYS with BE attribution AND the net exposure that earned it.

    Both halves matter: the attribution says which bucket lost, the exposure
    says whether the book was on target when it happened. A bad day at target
    is the strategy; a bad day at 3x budget is a tracking failure.
    """
    D = res.daily
    L = [f'WORST {n} DAYS    be = attribution, exp = normalised exposure', '']
    L.append(f'  {"date":<11}{"pnl":>8}{"g_be":>8}{"va_be":>8}{"vo_be":>8}'
             f'{"resid":>8}{"vo_exp":>9}{"vg_exp":>9}')
    nm = d.set_index('date') if 'date' in d.columns else d
    for i in D.nsmallest(n, 'pnl').index:
        row = D.loc[i]
        vo = nm['norm_volga'].get(i, np.nan) if 'norm_volga' in nm else np.nan
        vg = nm['norm_vega'].get(i, np.nan) if 'norm_vega' in nm else np.nan
        L.append(f'  {str(i):<11}{_k(row["pnl"]):>8}'
                 f'{_k(row["gamma_pnl_be"]):>8}{_k(row["vanna_pnl_be"]):>8}'
                 f'{_k(row["volga_pnl_be"]):>8}'
                 f'{_k(row["recon_resid"]):>8}{_k(vo):>9}{_k(vg):>9}')
    return L


# ===================================================================== #
# The figure
# ===================================================================== #
def _k(v) -> str:
    """Compact money: -9.8M / -61k / 761. Eight raw money columns do not fit a
    4.75in text panel; the reference dashboards used k/M for the same reason."""
    if v is None or not np.isfinite(v):
        return '-'
    a = abs(v)
    if a >= 1e9:
        return f'{v / 1e9:,.1f}B'
    if a >= 1e6:
        return f'{v / 1e6:,.1f}M'
    if a >= 1e3:
        return f'{v / 1e3:,.0f}k'
    return f'{v:,.0f}'


def _text_panel(ax, lines: List[str], c, size: float = 7.0) -> None:
    ax.axis('off')
    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes,
            va='top', ha='left', family='monospace', fontsize=size,
            color=c['ink2'], linespacing=1.45)


def _date_axis(ax, x, n: int = 4) -> None:
    """
    ONE date-axis policy for every panel on the page.

    Three rules, and each fixes something that was actually wrong:

      * ONE format family, chosen from the span. A fixed MonthLocator(3) is
        wrong on a 26-day window (it yields one tick or none) and the equity
        panel, left on the matplotlib default, printed '2025-12' while its
        neighbours printed 'Jan 26'.
      * NO ROTATION, ever. Rotated ticks are the single biggest source of the
        "messy" look, and they are only needed because there are too many of
        them. Fix the count instead.
      * A tick BUDGET per panel, not a fixed interval, so a quarter-width axes
        gets 3 labels and the full-width equity axes gets more.
    """
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    xs = pd.to_datetime(pd.Series(x)).dropna()
    span = max((xs.max() - xs.min()).days, 1) if len(xs) else 1
    if span <= 45:
        loc, fmt = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1), '%d %b'
    elif span <= 200:
        loc, fmt = mdates.MonthLocator(interval=1), '%b %y'
    elif span <= 800:
        loc, fmt = mdates.MonthLocator(interval=3), '%b %y'
    else:
        loc, fmt = mdates.YearLocator(), '%Y'
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    # Thin whatever the locator produced down to the panel's budget.
    ticks = ax.get_xticks()
    if len(ticks) > n:
        keep = ticks[:: int(np.ceil(len(ticks) / n))]
        ax.set_xticks(keep)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_ha('center')


def _delta_panel(ax, res, c) -> None:
    """
    The spot-hedge delta residual, in base-ccy SPOT NOTIONAL (a quantity, not a
    P&L -- see GreekVector.delta_hedge).

    TWO series, because only together are they interpretable:

      pre-hedge gap   `hedge_gap` = target - current, the options' delta that
                      accumulated since the last rebalance. This is what the
                      hedger saw and what it had to absorb. Always non-zero.

      post-hedge      `hedge_after - hedge_target`. At hedge_fraction=1.0 this
                      is ZERO BY CONSTRUCTION -- book.rebalance_hedge sets
                      hedge_after = target exactly -- which is identity #6.
                      Below 1.0 it becomes the un-hedged delta you are running,
                      and THAT is a number, not a bug.

    The reference dashboard's "Net Delta (resid)" panel plotted only the second
    of these. On a fully-hedged book that is float noise (measured 4.00e-10), so
    the pre-hedge gap is the series that actually carries information.
    """
    h = res.hedges
    if not len(h) or 'hedge_gap' not in h.columns:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no hedge rows', ha='center', va='center',
                color=c['muted'], transform=ax.transAxes)
        return
    x = pd.to_datetime(h['date'])
    gap = h['hedge_gap'].astype(float)
    post = (h['hedge_after'].astype(float) - h['hedge_target'].astype(float)
            if 'hedge_after' in h.columns and 'hedge_target' in h.columns
            else pd.Series(np.zeros(len(h))))
    ax.axhline(0.0, color=c['axis'], lw=1.0, zorder=2)
    ax.plot(x, gap, color=c['s1'], lw=1.4, solid_capstyle='round',
            label='pre-hedge gap', zorder=3)
    ax.plot(x, post, color=c['s2'], lw=1.6, solid_capstyle='round',
            label='post-hedge resid', zorder=4)
    leg = ax.legend(loc='upper left', frameon=False, fontsize=7)
    for t in leg.get_texts():
        t.set_color(c['ink2'])
    frac = res.config.hedge_fraction
    ax.set_title(f'Spot-hedge delta resid, base ccy\n'
                 f'frac={frac:g}  peak|post| {_k(post.abs().max())}  '
                 f'turn {_k(h["hedge_traded"].abs().sum())}',
                 loc='left', color=c['ink'], fontsize=8.5)
    _finish(ax, c)


def _trades_panel(ax, res, roller, c, mode: str = 'traded',
                  max_tags: int = 4) -> None:
    """
    THE TRADES, across the life of the strategy.

    Signed notional by tag -- longs above the line, shorts below -- so the
    structure reads directly: for a wing short you see the ATM body above zero
    and the 10d wings below it, and the roll rhythm as the bar spacing.

    mode='traded' bars what was TRADED on each roll date (what you did).
    mode='held'   areas what was HELD on each date (what you had).

    Tags beyond `max_tags` by gross notional fold into 'other' -- a colour slot
    is never cycled to accommodate a fifth series.
    """
    slots = ['s1', 's2', 's3', 's4']
    if mode == 'held' and len(res.positions):
        src = res.positions.assign(
            signed=res.positions['notional'] * res.positions['direction'])
        piv = src.pivot_table(index='date', columns='tag', values='signed',
                             aggfunc='sum').fillna(0.0)
        kind = 'held'
    elif len(res.trades):
        src = res.trades.assign(
            signed=res.trades['notional'] * res.trades['direction'])
        piv = src.pivot_table(index='date', columns='tag', values='signed',
                             aggfunc='sum').fillna(0.0)
        kind = 'traded'
    else:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no trades', ha='center', va='center',
                color=c['muted'], transform=ax.transAxes)
        return

    order = piv.abs().sum().sort_values(ascending=False).index.tolist()
    if len(order) > max_tags:
        keep, rest = order[:max_tags - 1], order[max_tags - 1:]
        piv = piv[keep].assign(other=piv[rest].sum(axis=1))
        order = keep + ['other']
    else:
        piv = piv[order]

    x = pd.to_datetime(piv.index)
    ax.axhline(0.0, color=c['axis'], lw=1.0, zorder=2)
    pos_b = np.zeros(len(piv))
    neg_b = np.zeros(len(piv))
    width = max(1.0, (x.max() - x.min()).days / max(len(piv), 1) * 0.8)
    for k, tag in enumerate(order):
        v = piv[tag].values
        up, dn = np.clip(v, 0, None), np.clip(v, None, 0)
        if kind == 'traded':
            ax.bar(x, up, bottom=pos_b, width=width, color=c[slots[k]],
                   label=tag, lw=0, zorder=3)
            ax.bar(x, dn, bottom=neg_b, width=width, color=c[slots[k]],
                   lw=0, zorder=3)
        else:
            ax.fill_between(x, pos_b, pos_b + up, color=c[slots[k]],
                            lw=0, alpha=0.95, label=tag, zorder=3)
            ax.fill_between(x, neg_b, neg_b + dn, color=c[slots[k]],
                            lw=0, alpha=0.95, zorder=3)
        pos_b = pos_b + up
        neg_b = neg_b + dn

    # rolls that did NOT trade are invisible in a bar chart of what traded
    if roller is not None and len(roller.frame()):
        lg = roller.frame()
        skip = lg[lg['action'] != 'traded']
        if len(skip):
            ax.plot(pd.to_datetime(skip['date']), np.zeros(len(skip)),
                    marker='x', ms=5, mew=1.2, ls='none', color=c['muted'],
                    label='no trade', zorder=4)
    leg = ax.legend(loc='upper left', frameon=False, ncol=3, fontsize=7)
    for t in leg.get_texts():
        t.set_color(c['ink2'])
    ax.set_title(f'Trades: signed notional {kind}\n'
                 f'long above / short below, base ccy',
                 loc='left', color=c['ink'], fontsize=8.5)
    _finish(ax, c)


def dashboard(res, ds, pair: str, tenor: str, horizon: float,
              budget: Optional[float] = None, roller=None, exiter=None,
              greek: str = 'volga', units: str = 'normalised',
              trades_mode: str = 'traded', band_frac: float = 0.20,
              dark: bool = False, path: Optional[str] = None,
              title: str = '', d: Optional[pd.DataFrame] = None,
              echo: bool = True):
    """
    Build the one-page dashboard.

    units : 'normalised' | 'trader' | 'both'. 'both' writes two files, suffixing
            `path`. A static image has no toggle; two renders is the honest
            translation.
    d     : a frame from `run.report.risk_tracking` / `report()`, to avoid
            rebuilding ~200 snapshots. Built here if omitted.
    echo  : also print the scorecard and worst-days block to the console, so the
            numbers stay searchable and copy-pasteable.
    """
    if units == 'both':
        figs = []
        for u in ('normalised', 'trader'):
            p = None
            if path:
                stem, _, ext = path.rpartition('.')
                p = f'{stem}_{u}.{ext}' if stem else f'{path}_{u}'
            figs.append(dashboard(
                res, ds, pair, tenor, horizon, budget, roller, exiter, greek,
                u, trades_mode, band_frac, dark, p, title, d,
                echo=(u == 'normalised' and echo)))
            d = getattr(figs[-1], '_frame', d)
        return figs
    if units not in ('normalised', 'trader'):
        raise ValueError("units must be 'normalised', 'trader' or 'both'")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    plt, c = _style(dark)

    if d is None:
        d = risk_tracking(res, ds, pair, tenor, horizon)
    if units == 'trader' and 'trader_vega' not in d.columns:
        d = trader_exposures(res, d)

    targets = constrained_targets(roller)
    if budget is None:
        budget = targets.get(greek)

    card  = scorecard(res, d, budget, roller, greek, band_frac)
    worst = worst_days_block(res, d)
    if echo:
        print('\n'.join(card))
        print()
        print('\n'.join(worst))

    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(3, 16, height_ratios=[3.0, 2.3, 2.4],
                          hspace=0.55, wspace=1.35,
                          left=0.035, right=0.975, top=0.945, bottom=0.06)
    ax_card  = fig.add_subplot(gs[0:2, 0:4])
    ax_eq    = fig.add_subplot(gs[0, 4:16])
    ax_mid   = [fig.add_subplot(gs[1, 4 + 3 * i: 7 + 3 * i]) for i in range(4)]
    ax_worst = fig.add_subplot(gs[2, 0:4])
    ax_be    = fig.add_subplot(gs[2, 4:7])
    ax_hist  = fig.add_subplot(gs[2, 7:10])
    ax_trade = fig.add_subplot(gs[2, 10:13])
    ax_delta = fig.add_subplot(gs[2, 13:16])

    _text_panel(ax_card, card, c)
    _text_panel(ax_worst, worst, c, size=6.1)

    money = FuncFormatter(lambda v, _: f'{v:,.0f}')
    x = pd.to_datetime(d['date'])

    # ---- equity ------------------------------------------------------- #
    D  = res.daily
    eq = pd.Series(D['equity'].values, index=range(len(D)))
    xe = pd.to_datetime(D.index)
    dd = eq - eq.cummax()
    ax_eq.axhline(0.0, color=c['axis'], lw=1.0, zorder=1)
    ax_eq.fill_between(xe, 0.0, eq.values, color=c['s1'], alpha=0.10,
                       lw=0, zorder=2)
    ax_eq.plot(xe, eq.values, color=c['s1'], lw=2.0, solid_capstyle='round',
               zorder=3)
    w3 = D.nsmallest(3, 'pnl')
    ax_eq.plot(pd.to_datetime(w3.index), D.loc[w3.index, 'equity'].values,
               marker='o', ms=8, ls='none', color=c['s2'],
               markeredgecolor=c['surface'], markeredgewidth=2.0,
               label='worst 3 days', zorder=5)
    leg = ax_eq.legend(loc='upper left', frameon=False)
    for t in leg.get_texts():
        t.set_color(c['ink2'])
    ax_eq.set_title(f'{title}Equity   final {eq.iloc[-1]:,.0f}   '
                    f'max drawdown {dd.min():,.0f}   '
                    f'({"GROSS of spread" if _costs_are_off(res) else "net of spread"})',
                    loc='left', color=c['ink'])
    _finish(ax_eq, c, 'cumulative P&L')
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
    _date_axis(ax_eq, xe, n=8)

    # ---- the four exposures ------------------------------------------- #
    # NOT a shared y-axis. In trader mode gamma/vanna are NOTIONAL while
    # vega/volga are MONEY, so sharing would be meaningless; in normalised mode
    # the four are the same dimension but different quantities, and forcing one
    # scale flattens three panels. Each carries its own unit label and, in
    # normalised mode, its size as a fraction of budget.
    for ax, (g, slot) in zip(ax_mid, MIDDLE):
        col = ('norm_' if units == 'normalised' else 'trader_') + g
        if col not in d.columns:
            ax.axis('off')
            continue
        y = d[col]
        ax.axhline(0.0, color=c['axis'], lw=1.0, zorder=1)
        ax.plot(x, y, color=c[slot], lw=1.6, solid_capstyle='round', zorder=3)

        tgt = targets.get(g)
        if tgt is not None:
            if units == 'normalised':
                ax.axhline(tgt, color=c['muted'], lw=1.2, ls=(0, (5, 3)),
                           zorder=2)
            else:
                # A CONSTANT normalised target is a MOVING trader-unit target,
                # because the conversion carries sigma and nu. Drawing it as a
                # line rather than a rule is the point, not a nicety.
                sc = d[GREEK_COL[g]].replace(0.0, np.nan)
                raw_over_norm = sc / d['norm_' + g].replace(0.0, np.nan)
                raw_tgt = tgt * raw_over_norm
                ax.plot(x, _TRADER_FROM_RAW[g](raw_tgt, d['spot_used']),
                        color=c['muted'], lw=1.2, ls=(0, (5, 3)), zorder=2)

        note = f'mean {y.mean():,.0f}'
        if units == 'normalised' and budget:
            note += f'   {abs(y).mean() / abs(budget):.0%} of budget'
        # Unit goes in the TITLE, not an xlabel -- in trader mode the four
        # panels are in two different unit families and the label must sit with
        # the number it qualifies, not under rotated date ticks.
        unit = _NORM_UNIT if units == 'normalised' else _TRADER_UNIT[g]
        ax.set_ylabel('')
        ax.set_title(f'{g}' + ('  (pinned 0)' if tgt == 0 else
                               '  (target)' if tgt is not None else '  (free)')
                     + f'\n{note}\n{unit}',
                     loc='left', color=c['ink'], fontsize=9)
        _finish(ax, c)
        _date_axis(ax, x, n=3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
        ax.tick_params(labelsize=7)
        _date_axis(ax, x, n=3)

    # ---- BE cumulative + vega leakage --------------------------------- #
    ax_be.axhline(0.0, color=c['axis'], lw=1.0, zorder=1)
    ends = []
    for col, lab, slot in (('volga_pnl_be', 'volga_be', 's1'),
                           ('gamma_pnl_be', 'gamma_be', 's2'),
                           ('vanna_pnl_be', 'vanna_be', 's3'),
                           ('vega_pnl',     'vega leak', 's4')):
        if col not in D.columns:
            continue
        yy = pd.Series(D[col].cumsum().values, index=range(len(D)))
        ax_be.plot(xe, yy.values, color=c[slot], lw=1.8,
                   solid_capstyle='round', label=lab, zorder=3)
        ends.append((xe[-1], float(yy.iloc[-1]), lab, c[slot]))
    # right margin so end labels have somewhere to go, and a bigger minimum
    # gap: 4 series in a quarter-width axes collide at the default 0.075.
    x0, x1 = ax_be.get_xlim()
    ax_be.set_xlim(x0, x1 + (x1 - x0) * 0.16)
    _label_ends(ax_be, ends, min_gap_frac=0.13)
    leg = ax_be.legend(loc='upper left', frameon=False, ncol=2, fontsize=7)
    for t in leg.get_texts():
        t.set_color(c['ink2'])
    ax_be.set_title('Cumulative EDGE, base ccy\n'
                    'realised - implied; NOT equity',
                    loc='left', color=c['ink'], fontsize=8.5)
    _finish(ax_be, c)
    ax_be.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
    ax_be.tick_params(labelsize=7)
    _date_axis(ax_be, xe, n=3)

    # ---- daily P&L distribution --------------------------------------- #
    pnl = D['pnl']
    q5 = pnl.quantile(0.05)
    cv = pnl[pnl <= q5].mean()
    ax_hist.hist(pnl, bins=45, color=c['s1'], lw=0)
    for v, ls in ((q5, (0, (5, 3))), (cv, (0, (2, 2))), (pnl.min(), '-')):
        ax_hist.axvline(v, color=c['s2'], lw=1.4, ls=ls)
    ax_hist.set_title(f'Daily P&L, base ccy\n'
                      f'sd {_k(pnl.std())}  skew {pnl.skew():+.2f}  '
                      f'(y = days)',
                      loc='left', color=c['ink'], fontsize=8.5)
    _finish(ax_hist, c)
    ax_hist.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
    ax_hist.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_hist.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_hist.tick_params(labelsize=7)
    # The rules stay; the numbers move to one corner block. Rotated on-line
    # labels collided with each other and with the y-axis label.
    ax_hist.annotate('\n'.join([f'VaR95  {_k(q5)}',
                                f'CVaR5  {_k(cv)}',
                                f'worst  {_k(pnl.min())}']),
                     xy=(0.02, 0.97), xycoords='axes fraction',
                     va='top', ha='left', fontsize=7, color=c['s2'],
                     family='monospace')

    # ---- the trades ---------------------------------------------------- #
    _trades_panel(ax_trade, res, roller, c, mode=trades_mode)
    ax_trade.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
    ax_trade.tick_params(labelsize=7)
    _date_axis(ax_trade, pd.to_datetime(res.trades['date'])
               if len(res.trades) else xe, n=3)

    # ---- spot-hedge delta residual ------------------------------------ #
    _delta_panel(ax_delta, res, c)
    ax_delta.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _k(v)))
    ax_delta.tick_params(labelsize=7)
    if len(res.hedges):
        _date_axis(ax_delta, pd.to_datetime(res.hedges['date']), n=3)

    fig.suptitle(f'{pair} {tenor}   greek-target book   units={units}'
                 f'   horizon={horizon:g}d',
                 x=0.035, ha='left', color=c['ink'], fontsize=12,
                 fontweight='semibold')
    fig._frame = d
    if path:
        fig.savefig(path, bbox_inches='tight')
        print(f'[dashboard] wrote {path}')
    return fig
