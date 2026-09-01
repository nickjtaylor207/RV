"""
report.py -- read a RunResult without fooling yourself.

WHY THIS IS NOT JUST print(res.daily)
-------------------------------------
`res.daily` has 30-odd columns holding TWO DIFFERENT decompositions of the same
P&L, on TWO DIFFERENT time conventions, in TWO DIFFERENT unit systems. Every
mistake worth making is a mistake of mixing them:

  1. TWO DECOMPOSITIONS, only one of which is additive.

         option_pnl = delta + gamma + theta + vega + vanna + volga + recon_resid

     is exact (identity #4). The `_be` view is NOT a decomposition of it:

         gamma_pnl_be = gamma_pnl + theta_pnl                 <- a REGROUPING
         vanna_pnl_be = vanna_pnl - S*sigma^2*rho*nu*dt       <- a SURPRISE
         volga_pnl_be = volga_pnl - sigma^2*nu^2*dt           <- a SURPRISE

     There is no vega_pnl_be at all. So summing the three `_be` buckets
     double-counts theta, drops vega entirely, and omits the implied premium you
     were paid. A stacked chart of `_be` buckets is an EDGE attribution, never a
     P&L attribution. `pnl_blocks()` prints both, separately labelled, and
     reconciles them so the gap is visible rather than assumed.

  2. TWO TIME CONVENTIONS. `net_*` and every per-position greek are
     START-of-period: row D is the book at the CLOSE of D-1. A roll on D shows
     up on row D+1. `n_open`, by contrast, is read after trading, so it is an
     END-of-day count. Comparing them on the same row is off by a day.

  3. TWO UNIT SYSTEMS. `net_volga_1vp` is a raw Taylor coefficient; the target
     was set in normalised one-sigma units. They differ by a factor that moves
     EVERY DAY with sigma and nu. `risk_tracking()` does the conversion with the
     shift(1) the first two points require.

WHAT TO READ, IN ORDER
----------------------
    tier 0  did the machine do what I asked?   <- ALWAYS first
    tier 1  is the risk on target, in the units I set it in?
    tier 2  do the books balance?
    tier 3  where did the P&L come from?
    tier 4  is the SHAPE right?
    tier 5  the distribution, not the mean     <- under-read, and where a short
                                                  convexity book actually dies
    tier 6  what to sweep before believing any of it

`report()` prints all seven. The two things numbers cannot show are the SHAPE of
the risk path and the DRIFT of the cumulative edge, so those get charts:
`chart_risk()` and `chart_attribution()`.


A NOTE ON A FREE COST MODEL
---------------------------
With `OptionCostModel(scale=0.0)`, `charge()` short-circuits to `cost=0.0`.
`Book.open` only logs a cost row `if pos.cost_paid:` -- zero is falsy -- while
`Book.close` logs unconditionally. So `res.costs` comes back holding ONLY
`reason='close'` rows, all at zero, and `option_tc` is flat zero. That is not a
bug, but it means identities #3 and #7 are trivially true, and any conclusion
about an EXIT RULE is invalid: the exit's entire cost side is switched off.
`report()` detects this and says so rather than printing a table of zeros.


COLOR
-----
The two charts use the documented reference palette in fixed slot order
(blue / orange / aqua / yellow) on the ADJACENT pairlist, which is the case that
palette validates in both light and dark. Slots are never cycled: a fifth series
would fold into "other" or become a small multiple, not a new hue. Slot 4
(yellow) is sub-3:1 on the light surface, so every series carries a direct end
label as well as a legend -- identity is never colour-alone, and the text report
is the table view.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from market.snapshot import MarketSnapshot
from core.calendar import add_tenor
from core.conventions import fx_calendar
from core.greeks import GreekVector, SPOT_MOVE
from strategy.sizer import sigma_scales


# --------------------------------------------------------------------- #
# Palette -- the reference instance, fixed slot order, light + dark.
# --------------------------------------------------------------------- #
PALETTE = {
    'light': dict(
        surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781',
        grid='#e1e0d9', axis='#c3c2b7', band='#eeede8',
        s1='#2a78d6', s2='#eb6834', s3='#1baf7a', s4='#eda100'),
    'dark': dict(
        surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781',
        grid='#2c2c2a', axis='#383835', band='#26261f',
        s1='#3987e5', s2='#d95926', s3='#199e70', s4='#c98500'),
}

GREEK_COL = {'volga': 'net_volga_1vp', 'vega': 'net_vega_1vp',
             'vanna': 'net_vanna_1pct_1vp', 'gamma': 'net_gamma_1pct',
             'theta': 'net_theta_1d', 'spot': 'net_spot_1pct'}


# ===================================================================== #
# Unit conversion -- the one place raw greeks become normalised ones
# ===================================================================== #
def scale_frame(ds, pair: str, tenor: str, horizon: float,
                dts: Sequence) -> pd.DataFrame:
    """
    Per-date `sigma_scales`, indexed 0..n-1 to align with a reset_index()'d
    daily frame.

    A FRESH tenor is resolved every day -- add_tenor(s.date, tenor) -- not the
    aging one. That is deliberate and it matches the sizer: `solve` reads sigma
    and nu at the MENU tenor's days-to-expiry, and `bucket_book` maps an aged
    position into that same bucket. Measuring against the aging tenor would
    compare the book to a yardstick the sizer never used.
    """
    fxc, rows = fx_calendar(pair), []
    for x in dts:
        s = MarketSnapshot.at(ds, pair, pd.Timestamp(x))
        e = add_tenor(s.date, tenor, fxc)
        rows.append(sigma_scales(s, (e - s.date).days, horizon))
    return pd.DataFrame(rows)


def risk_tracking(res, ds, pair: str, tenor: str, horizon: float,
                  greeks: Sequence[str] = ('volga', 'vega', 'vanna', 'gamma')
                  ) -> pd.DataFrame:
    """
    `res.daily`, reset, with a `norm_<greek>` column per requested greek.

    THE shift(1) IS LOAD-BEARING. `net_*` are START-of-period levels, so row D
    holds the book at the close of D-1 and must be scaled by D-1's factor --
    sigma and nu both moved overnight. Without the shift you multiply
    yesterday's greeks by today's yardstick, and the error is largest exactly
    when vol moved, which is when you care.
    """
    d  = res.daily.reset_index()
    sc = scale_frame(ds, pair, tenor, horizon, d['date'])
    for g in greeks:
        col = GREEK_COL[g]
        if col in d.columns:
            d['norm_' + g] = d[col] * sc[g].shift(1)
    return d


# ===================================================================== #
# Small helpers
# ===================================================================== #
def _hr(title: str, width: int = 78) -> None:
    print(f'\n{"=" * width}\n{title}\n{"=" * width}')


def _sub(title: str) -> None:
    print(f'\n--- {title} ' + '-' * max(0, 72 - len(title)))


def _costs_are_off(res) -> bool:
    """True when the cost model was disabled -- see the module docstring."""
    return float(res.daily['option_tc'].sum()) == 0.0


def _target_from_log(roller, greek: str) -> Optional[float]:
    """Recover the target the roller was actually asking for, from its log."""
    if roller is None:
        return None
    df = roller.frame()
    if not len(df):
        return None
    hits = [c for c in df.columns if c.startswith(f'want_{greek}@')]
    if not hits:
        return None
    v = df[hits[0]].dropna()
    return float(v.iloc[0]) if len(v) else None


def _roll_dates(roller) -> set:
    if roller is None:
        return set()
    df = roller.frame()
    if not len(df):
        return set()
    return set(df.loc[df['action'] == 'traded', 'date'])


def _exit_dates(exiter) -> set:
    if exiter is None:
        return set()
    df = exiter.frame()
    return set(df['date']) if len(df) else set()


# ===================================================================== #
# TIER 0 -- did the machine do what I asked?
# ===================================================================== #
def tier0_execution(res, roller=None, exiter=None,
                    leverage_limit: float = 10.0,
                    max_rows: int = 14) -> None:
    _hr('TIER 0  --  DID THE MACHINE DO WHAT I ASKED?   (read before any P&L)')
    if roller is not None:
        print(roller.report())
    if exiter is not None:
        print(exiter.report())

    if roller is not None and len(roller.frame()):
        log = roller.frame()
        _sub('roll log')
        cols = [c for c in ('date', 'action', 'legs', 'gross', 'cost',
                            'leverage', 'cond') if c in log.columns]
        cols += [c for c in log.columns if c.startswith(('want_', 'book_'))]
        cols += ['note'] if 'note' in log.columns else []
        shown = log[cols]
        if len(shown) > max_rows:
            h = max_rows // 2
            print(shown.head(h).to_string(index=False,
                                          float_format=lambda v: f'{v:,.1f}'))
            print(f'  ... {len(shown) - 2 * h} rows elided '
                  f'(max_rows={max_rows}) ...')
            print(shown.tail(h).to_string(index=False, header=False,
                                          float_format=lambda v: f'{v:,.1f}'))
        else:
            print(shown.to_string(index=False,
                                  float_format=lambda v: f'{v:,.1f}'))
        # Aggregate the log so a truncated view still cannot hide a bad roll.
        acts = log['action'].value_counts().to_dict()
        print(f'  actions {acts}   legs/roll mean '
              f'{log["legs"].mean():.1f}   gross total {log["gross"].sum():,.0f}')

        # A square system (columns == constraints) has a UNIQUE solution, so
        # the LP has nothing to minimise and cost_model in solve_kw is inert.
        # We cannot see the column count from here, but leverage is the tell:
        # near the limit means the target is met by large offsetting positions
        # in near-duplicate columns rather than by legs that span it.
        if 'leverage' in log.columns:
            lev = log['leverage'].dropna()
            if len(lev):
                hot = (lev > 0.7 * leverage_limit).mean()
                print(f'  leverage: mean {lev.mean():,.1f}  max {lev.max():,.1f}'
                      f'  ({hot:.0%} of rolls above 70% of the {leverage_limit:g} '
                      f'limit)')
                if lev.max() > 0.7 * leverage_limit:
                    print('  !! near the guard. A narrow menu (columns == '
                          'constraints) makes the solve\n     square: unique '
                          'answer, no cost minimisation, and offsetting size is\n'
                          '     the only way to hit the target. Widen '
                          'allow_deltas before trusting cost.')

        # want_ != book_ is the ONLY place the LP going infeasible and falling
        # back to least squares surfaces.
        for w in [c for c in log.columns if c.startswith('want_')]:
            b = 'book_' + w[len('want_'):]
            if b in log.columns:
                gap = (log[w] - log[b]).abs()
                if gap.max() > 1e-6:
                    print(f'  !! {w} != {b} on {int((gap > 1e-6).sum())} rolls '
                          f'(max {gap.max():,.1f}) -- the LP went infeasible and '
                          f'fell back to lstsq.')

    # --- the composition test nothing else performs -------------------- #
    rd, xd = _roll_dates(roller), _exit_dates(exiter)
    if xd:
        _sub('does wake= actually fire?')
        both = rd & xd
        print(f'  exit dates {len(xd)}   roll dates {len(rd)}   '
              f'same-day both {len(both)}  ({len(both) / len(xd):.0%} of exits)')
        print('  wake=[roller] nulls the cadence counter so top_up refills the '
              'hole the SAME\n  day. If that overlap is ~0 the wake is not '
              'firing and the book runs light\n  between rolls -- which is a '
              'legitimate choice, but make it deliberately.')

    _sub('what got traded')
    if len(res.trades):
        t = res.trades
        print(f'  legs opened {len(t)}   sleeves {list(t["sleeve"].unique())}'
              f'   tenors {list(t["tenor_label"].unique())}')
        dl = sorted(float(v) for v in t['target_delta'].round(2).dropna().unique())
        print(f'  deltas traded {dl}')
    reasons: Dict[str, int] = {}
    for p in res.book.positions.values():
        if not p.is_open:
            reasons[p.exit_reason] = reasons.get(p.exit_reason, 0) + 1
    print(f'  exit reasons {reasons}')
    if len(res.positions):
        print(f'  min t_days any leg reached '
              f'{res.positions.groupby("pos_id")["t_days"].min().min():.0f}'
              f'   (0 = died at expiry; ~7 = an exit rule caught it)')


# ===================================================================== #
# TIER 1 -- is the risk on target, in the units I set it in?
# ===================================================================== #
def tier1_risk(res, ds, pair: str, tenor: str, horizon: float,
               budget: Optional[float] = None, roller=None, exiter=None,
               greek: str = 'volga', every: int = 15) -> pd.DataFrame:
    _hr('TIER 1  --  IS THE RISK ON TARGET, IN THE UNITS I SET IT IN?')
    d = risk_tracking(res, ds, pair, tenor, horizon)
    if budget is None:
        budget = _target_from_log(roller, greek)
    live = d[(d[GREEK_COL[greek]].abs() > 1e-9) & d['norm_' + greek].notna()]
    if not len(live):
        print('  !! the book never carried risk. Check TIER 0 before anything '
              'else.')
        return d

    print(f'  live days {len(live)} of {len(d)}   '
          f'horizon={horizon:g}d   tenor={tenor}   '
          + (f'target {greek}={budget:,.0f}' if budget is not None else ''))

    _sub('tracking, normalised (the units the target was written in)')
    print(f'  {"greek":<10}{"mean":>12}{"mean|err|":>12}{"% of budget":>13}'
          f'{"p90|err|":>12}{"peak|err|":>12}')
    for g in ('volga', 'vega', 'vanna'):
        c = 'norm_' + g
        if c not in live.columns:
            continue
        # A PINNED greek's target is zero; only the sized greek has a budget.
        tgt = budget if (g == greek and budget is not None) else 0.0
        err = (live[c] - tgt).abs()
        den = abs(budget) if budget else np.nan
        print(f'  {g:<10}{live[c].mean():>12,.0f}{err.mean():>12,.0f}'
              f'{err.mean() / den if den else np.nan:>13.1%}'
              f'{err.quantile(0.9):>12,.0f}{err.max():>12,.0f}')
    print('  A pinned greek is targeted at ZERO, so its whole level is drift. '
          'Errors are\n  divided by |budget| -- a fixed, non-zero denominator. '
          'Never divide by the net\n  greek itself: it wanders through zero and '
          'the ratio explodes.')

    # --- the sawtooth: pinned on roll days, accumulating between -------- #
    rd = _roll_dates(roller)
    if rd:
        _sub('the sawtooth (top_up pins on roll days only)')
        # POST-roll rows, not roll rows. net_* are START-of-period, so a roll on
        # date D is visible on row D+1; filtering on `date in rd` reads the
        # PRE-roll book and reports the sawtooth upside down.
        prev_was_roll = d['date'].shift(1).isin(rd)
        post = live[prev_was_roll.reindex(live.index).fillna(False)]
        rest = live[~prev_was_roll.reindex(live.index).fillna(False)]
        print(f'  {"":<14}{"day AFTER a roll":>20}{"all other days":>18}'
              f'   (n {len(post)} / {len(rest)})')
        for g in ('vega', 'vanna'):
            c = 'norm_' + g
            if c in live.columns and len(post) and len(rest):
                print(f'  {("|" + g + "|"):<14}{post[c].abs().mean():>20,.0f}'
                      f'{rest[c].abs().mean():>18,.0f}')
        c = 'norm_' + greek
        if budget is not None and len(post) and len(rest):
            print(f'  {("|" + greek + " err|"):<14}'
                  f'{(post[c] - budget).abs().mean():>20,.0f}'
                  f'{(rest[c] - budget).abs().mean():>18,.0f}')
        print('  The left column should be ~0 for a PINNED greek: that is '
              'top_up doing its\n  job. The gap between the columns IS the '
              'strategy\'s real risk profile, and a\n  mean over both hides it. '
              'If the left column is NOT smaller, either the solve is\n  not '
              'hitting its pins or you are reading the pre-roll book.')

    _sub('position count')
    print(f'  n_open  med {res.daily["n_open"].median():.0f}  '
          f'max {res.daily["n_open"].max():.0f}')
    if roller is not None:
        log = roller.frame()
        traded = log[log['action'] == 'traded'] if len(log) else log
        if len(traded) > 1 and traded['legs'].sum():
            lpr = traded['legs'].mean()
            mnl = getattr(exiter, 'min_life_days', None)
            life = 30.0 - (mnl or 0.0)          # 1M ~ 30 calendar days
            # OBSERVED cadence, not roller.roll_days. wake= nulls the counter on
            # every exit date, so the effective cadence can be far tighter than
            # configured -- predicting off roll_days then reads as a failure
            # when the book is behaving exactly as specified.
            gaps = pd.Series(pd.to_datetime(traded['date'])).diff().dt.days
            cal = float(gaps.mean()) if gaps.notna().any() else np.nan
            cfg_cal = roller.roll_days * 7.0 / 5.0
            print(f'  predicted steady state ~ ({life:.0f} calendar days of '
                  f'life / {cal:.1f} observed cadence)'
                  f' x {lpr:.1f} legs/roll = {life / cal * lpr:.0f}')
            print(f'  observed cadence {cal:.1f} cal days vs configured '
                  f'{cfg_cal:.1f} ({len(traded)} trades)'
                  + ('   <- wake= is tightening it, which is the point'
                     if cal < 0.8 * cfg_cal else ''))
            print('  A mismatch AFTER accounting for the observed cadence means '
                  'vintages are not\n  expiring, or the roller is not firing.')

    _sub(f'path, every {every}th live day')
    cols = ['date', 'n_open'] + [c for c in ('norm_volga', 'norm_vega',
                                             'norm_vanna') if c in live.columns]
    print(live[cols].iloc[::every].to_string(
        index=False, float_format=lambda v: f'{v:,.0f}'))
    return d


# ===================================================================== #
# TIER 2 -- do the books balance?
# ===================================================================== #
def tier2_identities(res, roller=None, tol: float = 1e-6) -> bool:
    _hr('TIER 2  --  DO THE BOOKS BALANCE?')
    d, p = res.daily, res.positions
    free = _costs_are_off(res)
    ok = True

    def chk(name: str, val: float, note: str = '') -> None:
        nonlocal ok
        good = abs(val) < tol
        ok &= good
        print(f'  [{"OK " if good else "FAIL"}] {name:<52}{val:>12.2e}'
              + (f'  {note}' if note else ''))

    chk('1 pnl == option + hedge + carry - hedge_tc - option_tc',
        (d['pnl'] - (d['option_pnl'] + d['hedge_pnl'] + d['hedge_carry']
                     - d['hedge_tc'] - d['option_tc'])).abs().max())
    chk('2 equity is the cumsum of pnl',
        abs(d['equity'].iloc[-1] - d['pnl'].sum()))
    chk('3 daily option_tc == the cost ledger',
        abs(d['option_tc'].sum() - (res.costs['cost'].sum()
                                    if len(res.costs) else 0.0)),
        'TRIVIAL: costs are off' if free else '')
    tay = p[['delta_pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl',
             'vanna_pnl', 'volga_pnl', 'recon_resid']].sum(axis=1)
    chk('4 every position\'s Taylor expansion closes',
        (p['option_pnl'] - tay).abs().max())
    chk('5 positions roll up to the daily frame',
        abs(p.groupby('date')['option_pnl'].sum().sum() - d['option_pnl'].sum()))
    chk('6 delta P&L is cancelled by the hedge',
        (d['delta_pnl'] + d['hedge_pnl']).abs().max(),
        'only exact at hedge_fraction=1.0')
    if roller is not None and len(roller.frame()):
        chk('7 the sizer\'s cost estimate == what the Book charged',
            abs(roller.frame()['cost'].sum() - d['option_tc'].sum()),
            'TRIVIAL: costs are off' if free else
            'non-zero == priced-but-deadbanded rolls')
    if free:
        print('  NOTE cost_model is disabled (option_tc == 0 everywhere), so #3 '
              'and #7 are\n       0 == 0 and prove nothing.')
    return ok


# ===================================================================== #
# TIER 3 -- where did the P&L come from?
# ===================================================================== #
def pnl_blocks(res) -> None:
    _hr('TIER 3  --  WHERE DID THE P&L COME FROM?')
    d = res.daily

    _sub('P&L VIEW -- additive, sums exactly to pnl')
    add = ['gamma_pnl', 'theta_pnl', 'vega_pnl', 'vanna_pnl', 'volga_pnl',
           'recon_resid']
    tot = 0.0
    for c in add:
        v = d[c].sum()
        tot += v
        print(f'  {c:<16}{v:>16,.0f}')
    for c in ('delta_pnl', 'hedge_pnl', 'hedge_carry'):
        v = d[c].sum()
        tot += v
        print(f'  {c:<16}{v:>16,.0f}')
    for c in ('option_tc', 'hedge_tc'):
        v = d[c].sum()
        tot -= v
        print(f'  {("- " + c):<16}{-v:>16,.0f}')
    print(f'  {"=" * 32}')
    print(f'  {"sum":<16}{tot:>16,.0f}   vs equity '
          f'{d["equity"].iloc[-1]:>16,.0f}   '
          f'diff {tot - d["equity"].iloc[-1]:.2e}')
    print('  At hedge_fraction=1.0 delta_pnl and hedge_pnl cancel, so what '
          'actually drives\n  a delta-hedged book is the five remaining greek '
          'buckets and the cost lines.')

    _sub('EDGE VIEW -- realised vs implied. NOT a decomposition of pnl')
    for c, what in (('gamma_pnl_be', 'ATM variance risk premium'),
                    ('vanna_pnl_be', 'SKEW premium'),
                    ('volga_pnl_be', 'CONVEXITY / vol-of-vol premium')):
        print(f'  {c:<16}{d[c].sum():>16,.0f}   {what}')
    print(f'  {"vega_pnl":<16}{d["vega_pnl"].sum():>16,.0f}   '
          f'(no _be counterpart: E[dsigma] ~ 0)')
    be = d[['gamma_pnl_be', 'vanna_pnl_be', 'volga_pnl_be']].sum().sum()
    plain = d[['gamma_pnl', 'theta_pnl', 'vanna_pnl', 'volga_pnl']].sum().sum()
    print(f'\n  sum(_be) {be:>16,.0f}   sum(gamma+theta+vanna+volga) '
          f'{plain:>14,.0f}')
    print(f'  gap      {be - plain:>16,.0f}   = MINUS the premium the smile '
          f'implied,\n                              i.e. what you were paid to '
          f'carry the position.')
    print('  gamma_pnl_be is gamma+theta REGROUPED, so adding theta again '
          'double-counts.\n  vanna/volga_be are SURPRISES, not flows. Never '
          'stack these as an attribution\n  of equity -- read the DRIFT of the '
          'cumulative, which is the objective.')


# ===================================================================== #
# TIER 4 -- is the shape right?
# ===================================================================== #
def tier4_shape(res) -> None:
    _hr('TIER 4  --  IS THE SHAPE RIGHT?')
    p, t = res.positions, res.trades
    if 'tag' in p.columns:
        _sub('mean greek signature by tag')
        print(p.groupby('tag')[['vega_1vp', 'volga_1vp', 'vanna_1pct_1vp',
                                'theta_1d']].mean().round(1).to_string())
    if len(t):
        _sub('notional traded by type / delta / direction')
        g = (t.groupby(['option_type', 'target_delta', 'direction'],
                       dropna=False)['notional'].sum().round(0))
        print(g.to_string())
        # a volga short is roughly SYMMETRIC; a vanna short is LOPSIDED
        sgn = t.assign(signed=t['notional'] * t['direction'])
        by = sgn.groupby('option_type')['signed'].sum()
        c, pu = by.get('call', 0.0), by.get('put', 0.0)
        den = abs(c) + abs(pu)
        # call-vs-put IMBALANCE: 0 when the two sides match, 1 when it is all
        # on one side. NOT |c+p|/(|c|+|p|), which is 1 whenever both sides
        # share a sign -- i.e. always, for a strangle.
        print(f'\n  signed call {c:,.0f}   signed put {pu:,.0f}   '
              f'imbalance {abs(c - pu) / den if den else np.nan:.1%}'
              f'   (0% = symmetric, 100% = one-sided)')
        print('  A volga (wing-convexity) short should be roughly SYMMETRIC. A '
              'vanna (skew)\n  short should be LOPSIDED. If the shape does not '
              'match the target you set,\n  the pins are not binding the way '
              'you think.')
    _sub('sign checks on a short-convexity book')
    d = res.daily
    print(f'  theta_pnl {d["theta_pnl"].sum():>14,.0f}   '
          f'(positive => net short options, collecting decay)')
    lv = res.positions.groupby('date')[['volga_1vp', 'vega_1vp']].sum()
    print(f'  volga_1vp < 0 on {(lv["volga_1vp"] < 0).mean():.0%} of days   '
          f'vega_1vp < 0 on {(lv["vega_1vp"] < 0).mean():.0%} of days')


# ===================================================================== #
# TIER 5 -- the distribution, not the mean
# ===================================================================== #
def tier5_tail(res, worst: int = 5) -> None:
    _hr('TIER 5  --  THE DISTRIBUTION, NOT THE MEAN')
    d = res.daily
    pnl = d['pnl']
    dd = d['equity'] - d['equity'].cummax()

    _sub('daily P&L')
    print(f'  mean {pnl.mean():>12,.0f}   sd {pnl.std():>12,.0f}   '
          f'skew {pnl.skew():>7.2f}   min {pnl.min():>12,.0f}   '
          f'max {pnl.max():>12,.0f}')
    print(f'  max drawdown {dd.min():>12,.0f}   '
          f'on {dd.idxmin()}   final equity {d["equity"].iloc[-1]:>12,.0f}')
    sk = pnl.skew()
    print('  You are SHORT convexity, so the mean is not the risk -- the left '
          'tail is.')
    if sk < -0.5:
        print(f'    -> skew {sk:+.2f}: NEGATIVE, the expected shape. Small gains '
              f'most days,\n       occasional large losses. Judge this on the '
              f'tail, never on the mean.')
    elif sk > 0.5:
        print(f'    -> skew {sk:+.2f}: POSITIVE, which is NOT the expected shape '
              f'for a short-\n       convexity book. Before believing it, check '
              f'(a) costs -- a free run\n       removes the steady drag that '
              f'creates the negative arm, and (b) the\n       structure: long '
              f'ATM / short wings is long gamma near the money, so it\n       '
              f'is not a simple short-options profile. Read the tag table in '
              f'TIER 4.')
    else:
        print(f'    -> skew {sk:+.2f}: roughly symmetric. Over 195 days that '
              f'usually means the\n       tail event simply has not happened '
              f'in this window, not that it cannot.')

    _sub(f'worst {worst} days, split by bucket')
    cols = ['pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl', 'vanna_pnl',
            'volga_pnl', 'recon_resid']
    print(d.nsmallest(worst, 'pnl')[cols].to_string(
        float_format=lambda v: f'{v:,.0f}'))

    _sub('recon_resid -- gap risk, NOT an error to drive to zero')
    r = res.positions['recon_resid']
    print(f'  sum {r.sum():>12,.0f}   mean |r| {r.abs().mean():>10,.0f}   '
          f'p99 |r| {r.abs().quantile(0.99):>10,.0f}   '
          f'peak |r| {r.abs().max():>10,.0f}')
    print(f'  as % of |option_pnl|: '
          f'{r.abs().sum() / max(res.positions["option_pnl"].abs().sum(), 1):.2%}')
    w = res.positions.reindex(
        r.abs().sort_values(ascending=False).index).head(worst)
    keep = [c for c in ('date', 'pos_id', 'tag', 'dt_days', 'dS', 'dsigma',
                        'option_pnl', 'gamma_pnl', 'recon_resid')
            if c in w.columns]
    print(w[keep].to_string(index=False, float_format=lambda v: f'{v:,.2f}'))
    print('  A large |recon_resid| is a discrete jump or a Taylor breakdown. '
          'For a short\n  wings book those ARE the days that hurt -- track the '
          'distribution as a risk\n  metric, not as a check that passed.')


# ===================================================================== #
# TIER 6 -- what to sweep before believing any of it
# ===================================================================== #
def tier6_robustness(res, roller=None, exiter=None) -> None:
    _hr('TIER 6  --  WHAT TO SWEEP BEFORE BELIEVING ANY OF IT')
    if _costs_are_off(res):
        print('  1. COST SCALE, first and above everything else. This run had '
              'cost_model\n     disabled, so you do not yet know whether the '
              'result survives spread --\n     and spread is the binding '
              'constraint on a wing strategy. Re-run at\n     '
              'OptionCostModel(scale=1.0) and 2.0 before reading anything above '
              'as a\n     result rather than as mechanics.')
    else:
        print('  1. COST SCALE over (0.0, 1.0, 2.0, 4.0) -- run/breakeven_study '
              'does this.')
    if exiter is not None and getattr(exiter, 'min_life_days', None) is not None:
        print('  2. min_life_days over (None, 3, 5, 7, 10, 14). A result that '
              'only works at\n     exactly 7 is a result about 7. Rebuild BOTH '
              'roller and exiter each pass:\n     both carry _last and log '
              'across runs.')
    if roller is not None:
        print(f'  3. roll_days around {roller.roll_days} -- cadence trades '
              'tracking error against\n     spread, and rd=1 is a free daily '
              'vega controller with no hedge sleeve.')
    print('  4. allow_deltas. A menu with as many columns as constraints makes '
          'the solve\n     square: unique answer, no cost minimisation. Widen '
          'it and the LP starts\n     choosing again.')
    print('  5. WINDOW. Re-run on sub-windows and compare pnl_per_unit_greek, '
          'never raw\n     P&L -- different configs carry different risk and '
          'raw P&L has no common\n     denominator.')


# ===================================================================== #
# The one call
# ===================================================================== #
def report(res, ds, pair: str, tenor: str, horizon: float,
           budget: Optional[float] = None, roller=None, exiter=None,
           greek: str = 'volga', tiers: str = '0123456') -> pd.DataFrame:
    """
    Print the whole read-out. Returns the normalised daily frame so the charts
    (and any follow-up of your own) do not have to rebuild the snapshots.

    budget : the target, in normalised units. None recovers it from the
             roller's own log, which is safer than retyping it.
    tiers  : subset, e.g. '01' for execution + risk only.
    """
    d = None
    if '0' in tiers:
        tier0_execution(res, roller, exiter)
    if '1' in tiers:
        d = tier1_risk(res, ds, pair, tenor, horizon, budget, roller, exiter,
                       greek)
    if '2' in tiers:
        tier2_identities(res, roller)
    if '3' in tiers:
        pnl_blocks(res)
    if '4' in tiers:
        tier4_shape(res)
    if '5' in tiers:
        tier5_tail(res)
    if '6' in tiers:
        tier6_robustness(res, roller, exiter)
    if d is None:
        d = risk_tracking(res, ds, pair, tenor, horizon)
    return d


# ===================================================================== #
# CHARTS -- the two things the numbers above cannot show
# ===================================================================== #
def _style(dark: bool):
    import matplotlib.pyplot as plt
    c = PALETTE['dark' if dark else 'light']
    plt.rcParams.update({
        'figure.facecolor': c['surface'], 'axes.facecolor': c['surface'],
        'savefig.facecolor': c['surface'],
        'text.color': c['ink'], 'axes.labelcolor': c['ink2'],
        'xtick.color': c['muted'], 'ytick.color': c['muted'],
        'axes.edgecolor': c['axis'], 'axes.linewidth': 0.8,
        'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': c['grid'], 'grid.linewidth': 0.6,
        'figure.dpi': 130,
    })
    return plt, c


def _finish(ax, c, ylab: str = '') -> None:
    ax.grid(axis='y', alpha=1.0)
    ax.set_axisbelow(True)
    if ylab:
        ax.set_ylabel(ylab)


def _label_ends(ax, items, min_gap_frac: float = 0.075) -> None:
    """
    Direct end labels for every series on an axes, nudged apart so they cannot
    overlap. Selective direct labels are what keep identity off colour alone --
    which matters most for the yellow slot, sub-3:1 on the light surface.

    `items` is [(x_last, y_last, text, colour), ...]. Labels are laid out in
    value order with a minimum vertical separation, so two series ending at the
    same level read as two labels rather than one smear.
    """
    items = [(xv, yv, t, col) for xv, yv, t, col in items
             if yv is not None and np.isfinite(yv)]
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items.sort(key=lambda r: r[1])
    placed = []
    for xv, yv, t, col in items:
        pos = yv if not placed else max(yv, placed[-1] + gap)
        placed.append(pos)
        ax.annotate(f' {t}', xy=(xv, pos), xytext=(5, 0),
                    textcoords='offset points', va='center', ha='left',
                    fontsize=8, color=col, fontweight='medium', clip_on=False)


def _end_of(x, y, text, colour):
    """(x_last, y_last, text, colour) for one series; y_last None if empty."""
    yy = y.dropna()
    if not len(yy):
        return (None, None, text, colour)
    i = yy.index[-1]
    return (x.loc[i], float(y.loc[i]), text, colour)

def chart_risk(d: pd.DataFrame, budget: float, roller=None, exiter=None,
               greek: str = 'volga', band_frac: float = 0.20,
               dark: bool = False, path: Optional[str] = None, title: str = ''):
    """
    THE diagnostic chart: is the book where I said it should be, and what
    shape is the miss?

    Three rows, ONE x-axis, never two y-scales. The sized greek and the pinned
    greeks live on different rows precisely because they live on different
    scales -- small multiples, not a dual axis.

      row 1  the sized greek in normalised units, against its target
      row 2  the PINNED greeks as a fraction of |budget|, against zero, with
             the +/- band_frac deadband shaded. That normalisation is what makes
             two greeks with different magnitudes readable on one scale, and it
             is the unit 10.5's drift table is written in.
      row 3  an event rug -- roll dates and exit dates. Vertical rules for 39
             rolls would be noise; a rug keeps the timing legible.
    """
    plt, c = _style(dark)
    from matplotlib.ticker import FuncFormatter

    x = d['date']
    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(11, 7.2), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 2.4, 0.55], hspace=0.18))

    # ---- row 1: the sized greek vs its target ------------------------- #
    col = 'norm_' + greek
    a1.axhline(budget, color=c['muted'], lw=1.2, ls=(0, (5, 3)), zorder=1)
    a1.plot(x, d[col], color=c['s1'], lw=2.0, solid_capstyle='round', zorder=3)
    # target label parked in the RIGHT margin, clear of the series
    a1.annotate(f' target {budget:,.0f}', xy=(1.0, budget),
                xycoords=('axes fraction', 'data'),
                xytext=(5, -10), textcoords='offset points',
                fontsize=8, color=c['muted'], clip_on=False)
    _label_ends(a1, [_end_of(x, d[col], greek, c['s1'])])
    a1.set_title(f'{title}Normalised {greek} vs target  '
                 f'(one-sigma base-ccy P&L)', loc='left', color=c['ink'])
    _finish(a1, c, f'{greek}, normalised')
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))

    # ---- row 2: the pinned greeks, as a fraction of |budget| ---------- #
    den = abs(budget) if budget else 1.0
    a2.axhspan(-band_frac, band_frac, color=c['band'], zorder=0)
    a2.axhline(0.0, color=c['axis'], lw=1.0, zorder=1)
    pinned = [(g, k) for g, k in (('vega', 's2'), ('vanna', 's3'))
              if 'norm_' + g in d.columns and g != greek]
    ends, worst = [], 0.0
    for g, slot in pinned:
        y = d['norm_' + g] / den
        worst = max(worst, float(y.abs().max() or 0.0))
        a2.plot(x, y, color=c[slot], lw=2.0, solid_capstyle='round',
                label=g, zorder=3)
        ends.append(_end_of(x, y, g, c[slot]))
    # The shaded span vanishes when the drift dwarfs the band -- which is
    # exactly when you most need to see where the band was. Draw its edges too.
    for sgn in (-1, 1):
        a2.axhline(sgn * band_frac, color=c['muted'], lw=1.0,
                   ls=(0, (2, 3)), zorder=2)
    a2.annotate(f' +/-{band_frac:.2f}', xy=(1.0, band_frac),
                xycoords=('axes fraction', 'data'),
                xytext=(5, 0), textcoords='offset points',
                fontsize=8, color=c['muted'], clip_on=False)
    _label_ends(a2, ends)
    if len(pinned) >= 2:
        leg = a2.legend(loc='upper left', frameon=False, ncol=len(pinned))
        for t in leg.get_texts():
            t.set_color(c['ink2'])
    over = f'  --  drift peaks at {worst:.1f}x budget' if worst > 1.5 else ''
    a2.set_title(f'Pinned greeks, as a fraction of |budget|  '
                 f'(dashed = +/-{band_frac:.2f} deadband){over}',
                 loc='left', color=c['ink'])
    _finish(a2, c, 'x |budget|')

    # ---- row 3: event rug -------------------------------------------- #
    rd, xd = _roll_dates(roller), _exit_dates(exiter)
    for yv, dates, lab, slot in ((1.0, rd, 'roll', 's1'),
                                 (0.0, xd, 'exit', 's2')):
        if not dates:
            continue
        xs = [t for t in x if t in dates]
        a3.plot(xs, [yv] * len(xs), marker='|', ms=8, mew=1.4,
                ls='none', color=c[slot])
        a3.annotate(f'{lab} ({len(dates)})', xy=(0, yv),
                    xycoords=('axes fraction', 'data'),
                    xytext=(-2, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=8, color=c[slot])
    a3.set_ylim(-0.6, 1.6)
    a3.set_yticks([])
    a3.grid(False)
    for s in ('left', 'bottom'):
        a3.spines[s].set_visible(False)

    fig.align_ylabels()
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches='tight')
        print(f'[chart] wrote {path}')
    return fig


def chart_attribution(res, dark: bool = False, path: Optional[str] = None,
                      title: str = ''):
    """
    Does the edge actually drift, and does equity follow?

    row 1  cumulative EDGE buckets, as LINES not a stack -- they do not sum to
           equity (see pnl_blocks) and stacking them would assert that they do.
           vega_pnl rides alongside as the contamination measure: if vega is
           pinned it should be small next to volga_pnl_be, and when it is not,
           the sleeve is not isolating what you think.
    row 2  equity, the single headline series, on its own row because it is a
           different quantity from the buckets above -- not because of scale.
    """
    plt, c = _style(dark)
    from matplotlib.ticker import FuncFormatter

    d = res.daily
    x = pd.Series(pd.to_datetime(d.index), index=d.index)

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(11, 6.8), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 2], hspace=0.16))

    series = [('volga_pnl_be', 'volga_be (convexity)', 's1'),
              ('gamma_pnl_be', 'gamma_be (ATM VRP)',   's2'),
              ('vanna_pnl_be', 'vanna_be (skew)',      's3'),
              ('vega_pnl',     'vega_pnl (leakage)',   's4')]
    a1.axhline(0.0, color=c['axis'], lw=1.0, zorder=1)
    ends = []
    for col, lab, slot in series:
        if col not in d.columns:
            continue
        y = d[col].cumsum()
        a1.plot(x, y, color=c[slot], lw=2.0, solid_capstyle='round',
                label=lab, zorder=3)
        ends.append(_end_of(x, y, lab.split()[0], c[slot]))
    _label_ends(a1, ends)
    leg = a1.legend(loc='upper left', frameon=False, ncol=2)
    for t in leg.get_texts():
        t.set_color(c['ink2'])
    a1.set_title(f'{title}Cumulative EDGE (realised minus implied) -- not a '
                 f'decomposition of equity', loc='left', color=c['ink'])
    _finish(a1, c, 'cumulative, base ccy')
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))

    a2.axhline(0.0, color=c['axis'], lw=1.0, zorder=1)
    eq = d['equity']
    a2.plot(x, eq, color=c['s1'], lw=2.0, solid_capstyle='round', zorder=3)
    a2.fill_between(x, 0.0, eq, color=c['s1'], alpha=0.10, lw=0, zorder=2)
    _label_ends(a2, [_end_of(x, eq, 'equity', c['s1'])])
    dd = eq - eq.cummax()
    a2.set_title(f'Equity   (final {eq.iloc[-1]:,.0f}   '
                 f'max drawdown {dd.min():,.0f})', loc='left', color=c['ink'])
    _finish(a2, c, 'base ccy')
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))

    fig.align_ylabels()
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches='tight')
        print(f'[chart] wrote {path}')
    return fig
