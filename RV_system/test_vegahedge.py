"""
test_vegahedge.py -- learning VegaHedge, alone and with the roller.

THE QUESTION THIS FILE ANSWERS
------------------------------
`VegaHedge` and `GreekTargetRoller` both flatten vega, by different mechanisms,
and it is genuinely unobvious which is doing the work in a run that has both.

    roller, mode='top_up'   pins vega to EXACTLY zero, but only on roll days.
                            Between rolls it does nothing at all.
    VegaHedge               pins vega to WITHIN A BAND, every day.

So there are four configurations worth understanding, and this file runs them in
the order that makes each one interpretable:

    STAGE A   one static volga position, NO hedge
              -> how fast does vega actually drift off a struck-neutral book?
                 This is the number the hedge exists to fix. Read it first.

    STAGE B   one static volga position + VegaHedge
              -> the hedge in isolation. No roller, so nothing else is touching
                 vega and every trade in the hedge sleeve is attributable.

    STAGE C   the same, swept over `band`
              -> what does the band actually buy, and what does it cost?

    STAGE D   rolling top_up + VegaHedge, against two controls
              -> the interaction. And the control that matters is NOT the
                 unhedged run, it is roll_days=1 (see below).

RUN IT AS:  python test_vegahedge.py
Or set STAGES = 'AB' etc. at the top to run a subset. Bloomberg is hit once;
FXVolDataset.build is memoised on (pairs, days).


TWO THINGS THAT ARE NOT OBVIOUS FROM THE SOURCE
-----------------------------------------------
1. THERE IS NO ONE-SHOT STRATEGY CLASS, AND YOU DO NOT NEED ONE.
   `GreekTargetRoller.on_date` skips the cadence gate when `self._last is
   None`, so it always fires on the first date it sees. Give it a `roll_days`
   larger than the window and it fires EXACTLY ONCE and never again:

       GreekTargetRoller(..., roll_days=10**6)   # == "open at target, hold"

   `mode` is irrelevant on that first fire -- the book is flat, so 'top_up',
   'stack' and 'replace' all size the full target. Keep 'top_up' anyway so the
   line reads the same as the rolling version.

   CAVEAT: `_last` is set even when the attempt fails (`skipped_guard` /
   `deadband`), so a one-shot that trips a guard on day one never retries and
   you get an empty run. ALWAYS check `roll.report()` says `traded=1`.

2. VegaHedge NEUTRALISES BY ADDING LEGS, NOT BY CLOSING THEM.
   The only `book.close` call in `VegaHedge.on_date` is the re-strike path. A
   long-vega residual is flattened by SELLING a fresh straddle, not by closing
   the long one it sold earlier. So with `restrike_move=None`, or with spot
   sitting still, the hedge sleeve ACCUMULATES offsetting straddles -- each
   paying entry spread, each decaying, all of them visible to the next check
   because `hedge_sleeve` is auto-appended to `read_sleeves`.

   That matters most in exactly the STAGE A/B setup: as the wing legs age their
   vega shrinks, the hedge becomes over-hedged in the other direction, and it
   layers again. Watch `n_open` and `res.costs` by sleeve, not just the residual.


WHERE THE TWO RESIDUAL NUMBERS COME FROM, AND WHY BOTH
------------------------------------------------------
    vh.frame()['resid_vega']      PRE-trade. What the hedge SAW when it decided.
                                  Recorded after any re-strike unwind, before
                                  the new straddle. This is the drift.

    daily['net_vega_1vp'] * w     POST-trade. net_* are START-of-period levels,
                                  so row D is the book at the close of D-1 --
                                  which already includes D-1's hedge trade.
                                  This is what the hedge LEFT BEHIND.

Judging the hedge on `resid_vega` alone flatters it: tighten the band and the
hedge fires more often, so later checks start from a flatter book and the
pre-trade series falls even though you have not measured the thing you care
about. `drift_stats()` below reads the POST-trade series.
"""

import warnings
import numpy as np
import pandas as pd

from market.dataset import FXVolDataset
from market.snapshot import MarketSnapshot, business_dates
from core.calendar import add_tenor
from core.conventions import fx_calendar
from book.costs import OptionCostModel
from strategy.sizer import sigma_scales, GreekTarget
from strategy.roller import GreekTargetRoller
from engine.hedging import VegaHedge
from engine.loop import EngineConfig, run

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 60)
warnings.filterwarnings('ignore')


# ========================================================================= #
#                              CONFIGURATION
# ========================================================================= #
STAGES = 'A' #'ABCD'            # which stages to run, e.g. 'AB' or 'D'

PAIR    = 'USDCAD'
TENOR   = '1M'
DAYS    = 400              # history to pull
HORIZON = 7                # defines "one sigma". SAME everywhere or nothing compares.
MENU    = [25, 10, 'ATM']  # 5 columns: 25c 25p 10c 10p ATM-straddle
BUDGET  = -3_000.0         # normalised volga. Short wing convexity.

# --- STAGE A/B/C: one static vintage. The window must be about the OPTION'S
#     LIFE, not 200 days. A 1M leg is dead after ~22 business days, and a
#     hedge with no underlying risk to hedge reads `in_band` forever and
#     teaches you nothing. +4 days past expiry so you see the book go flat.
HOLD_BDAYS = 26

# --- STAGE D: the rolling case needs a long window to reach steady state.
ROLL_WINDOW = 200
ROLL_DAYS   = 5

# --- hedge policy
BAND_FRAC = 0.20           # band as a FRACTION of |BUDGET|. 0.20 -> 600.
RESTRIKE  = 0.01           # re-strike a hedge leg on a 1% move off ITS entry spot
HEDGE_MIN = 250_000        # deadband on hedge gross notional
BAND_GRID = (0.05, 0.10, 0.20, 0.50, 1.00)

ONE_SHOT = 10 ** 6         # roll_days large enough to fire exactly once


# ========================================================================= #
#                              THE WORLD
# ========================================================================= #
cm    = OptionCostModel(scale=1.0)
ds    = FXVolDataset.build(pairs=[PAIR], days=DAYS)
dates = business_dates(ds, PAIR)
fxc   = fx_calendar(PAIR)

print(f"[data] {len(dates)} business dates, "
      f"{dates[0].date()} -> {dates[-1].date()}")


def target():
    """
    The risk. Rebuilt per call so no two runs can share a mutated object.

    Three constraints in one bucket: volga at the budget, vega and vanna
    actively PINNED at zero. The pins are what make it a pure convexity
    position rather than a directional vol view -- and pinning vega at entry is
    precisely why this book is struck neutral and then drifts, which is the
    whole subject of this file.
    """
    return GreekTarget(by_tenor={TENOR: dict(volga=BUDGET, vega=0.0, vanna=0.0)},
                       horizon_days=HORIZON,
                       units='normalised')


def cfg(start_i, end_i=-1):
    """
    EngineConfig over dates[start_i:end_i]. A factory, not a constant --
    each run gets its own, so nothing can leak between stages.

    hedge_fraction=1.0 is what makes identity #6 (delta_pnl + hedge_pnl == 0)
    hold exactly; it is asserted in every stage below as a live test that the
    hedge is working at all.

    NOTE flatten_at_end defaults True, so the LAST date unwinds the spot hedge
    at full spot_tc. Over a 26-day window that terminal charge is a much bigger
    slice of the total than it is over 200 days -- do not read STAGE B's cost
    line as if it were annualised.
    """
    return EngineConfig(
        pairs          = [PAIR],
        start          = dates[start_i],
        end            = dates[end_i],
        cost_model     = cm,          # THE SAME OBJECT the sizer optimises on
        hedge_fraction = 1.0,
        spot_tc        = 0.0001,
        verbose        = False,
    )


def build_roller(roll_days, sleeve='wing_convex'):
    """A roller. roll_days=ONE_SHOT makes it fire once; see the module docstring.

    Rebuilt for every run without exception -- a roller carries `_last` (the
    cadence counter) and `log` across runs, so reusing one gives you a stale
    grid and two runs' logs concatenated.
    """
    return GreekTargetRoller(
        PAIR, TENOR, lambda _s: target(),
        roll_days = roll_days,
        sleeve    = sleeve,
        mode      = 'top_up',
        min_trade = 1_000_000,
        solve_kw  = dict(allow_deltas=MENU, cost_model=cm))


def build_hedge(band_frac, restrike=RESTRIKE, read=('wing_convex',)):
    """
    A VegaHedge.

    band          : NORMALISED units, so passing it as a fraction of |BUDGET|
                    is what makes the number mean the same thing as the drift
                    it is a band on.
    horizon_days  : MUST equal the roller's, or `band` is a fraction of a
                    different sigma from the budget it is a fraction of.
    read_sleeves  : whose vega to flatten. 'hedge' is appended automatically
                    inside __init__ -- the hedge's own decaying vega has to be
                    visible to the next check or the book drifts unnoticed.
    check_days=1  : every day. The DEADBAND, not the cadence, is the thing
                    that should be stopping trades.
    """
    return VegaHedge(
        PAIR, TENOR, list(read),
        band          = band_frac * abs(BUDGET),
        horizon_days  = HORIZON,
        hedge_sleeve  = 'hedge',
        restrike_move = restrike,
        min_trade     = HEDGE_MIN,
        cost_model    = cm,
        check_days    = 1)


class Composite:
    """
    engine.run drives ONE object. Order is the semantics:
    roller first (its solve pins vega to zero on roll days), hedge LAST (it
    cleans up what the roller left, and must not read a book about to change).
    """
    def __init__(self, *strategies):
        self.strategies = strategies

    def on_date(self, ctx):
        for s in self.strategies:
            s.on_date(ctx)


# ========================================================================= #
#                            MEASUREMENT
# ========================================================================= #
def scale_frame(dts):
    """
    Per-date sigma_scales, as a frame indexed 0..n-1 to line up with a
    reset_index()'d daily frame.

    A FRESH TENOR each day -- add_tenor(s.date, TENOR) -- not the aging one.
    That is deliberate and matches the sizer: `solve` reads sigma and nu at the
    menu tenor's own days-to-expiry, so the yardstick you compare against has
    to be read at the same point of the term structure.
    """
    rows = []
    for x in dts:
        s = MarketSnapshot.at(ds, PAIR, pd.Timestamp(x))
        e = add_tenor(s.date, TENOR, fxc)
        rows.append(sigma_scales(s, (e - s.date).days, HORIZON))
    return pd.DataFrame(rows)


def normalised(res):
    """
    daily frame with net_vega / net_volga converted into the units the target
    was set in. This is the POST-trade book -- see the module docstring.

    THE shift(1) IS NOT COSMETIC. net_* are START-of-period levels: row D holds
    the book at the close of D-1. So row D's greeks must be multiplied by
    D-1's scale, because sigma and nu both moved overnight. Drop the shift and
    you multiply yesterday's greeks by today's yardstick.
    """
    d  = res.daily.reset_index()
    sc = scale_frame(d['date'])
    d['norm_vega']  = d['net_vega_1vp']  * sc['vega'].shift(1)
    d['norm_volga'] = d['net_volga_1vp'] * sc['volga'].shift(1)
    return d


def drift_stats(res, band_frac=BAND_FRAC):
    """
    How well was vega actually controlled, and was volga held at target?

    Restricted to days the book carried risk -- a run whose legs have expired
    reports a perfect residual of zero and means nothing by it.
    """
    d    = normalised(res)
    live = d[(d['net_volga_1vp'].abs() > 1e-9) & d['norm_vega'].notna()]
    if not len(live):
        return None
    a    = live['norm_vega'].abs()
    band = band_frac * abs(BUDGET)
    return {
        'live_days':  len(live),
        'mean':       a.mean(),
        'median':     a.median(),
        'p90':        a.quantile(0.90),
        'peak':       a.max(),
        'signed':     live['norm_vega'].mean(),
        'mean/budget': a.mean() / abs(BUDGET),
        'pct>band':   float((a > band).mean()),
        'volga_te':   (live['norm_volga'] - BUDGET).abs().mean(),
    }


def show(name, res, roll, vh=None, band_frac=BAND_FRAC):
    """Everything worth reading off one run."""
    d = res.daily
    print(f'\n{"=" * 74}\n{name}\n{"=" * 74}')
    print(roll.report())
    if vh is not None:
        print(vh.report())

    st = drift_stats(res, band_frac)
    if st is None:
        print('  !! the book never carried risk -- check roller.report() above')
        return None

    print(f'\n  --- POST-TRADE vega, normalised (what the hedge LEFT) ---')
    print(f'  live days {st["live_days"]:>3}   '
          f'mean {st["mean"]:>7,.0f}   median {st["median"]:>7,.0f}   '
          f'p90 {st["p90"]:>7,.0f}   peak {st["peak"]:>8,.0f}')
    print(f'  mean/|budget| {st["mean/budget"]:>6.3f}   '
          f'out-of-band {st["pct>band"]:>6.1%}   '
          f'signed mean {st["signed"]:>8,.0f}'
          f'  ({abs(st["signed"]) / st["mean"]:.0%} of mean abs)')
    # 10.5's "the residual is ZERO-MEAN, so a hedge buys variance not return"
    # is a statement about the ROLLING book, where it averages over ~40 roll
    # dates and ~9 vintages. It does NOT survive on a single static vintage:
    # measured at signed -4,083 against a mean absolute of 4,222 -- 97%
    # directional. So print the verdict rather than asserting one.
    frac = abs(st['signed']) / st['mean'] if st['mean'] else 0.0
    print('    -> ' + ('DIRECTIONAL drift: hedging it changes EXPECTED P&L, '
                       'not just variance.'
                       if frac > 0.5 else
                       'near zero-mean: hedging it buys VARIANCE, not return.'))
    print(f'  volga tracking error {st["volga_te"]:>8,.0f} '
          f'({st["volga_te"] / abs(BUDGET):.1%} of budget)')

    # exit reasons: the only place 'min_life' / 'restrike' / 'expiry' surface.
    # res.costs['reason'] holds ONLY 'open'/'close' -- Book.close hardcodes it.
    reasons = {}
    for p in res.book.positions.values():
        if not p.is_open:
            reasons[p.exit_reason] = reasons.get(p.exit_reason, 0) + 1
    print(f'\n  legs opened {len(res.trades):>4}   '
          f'n_open med/max {d["n_open"].median():.0f}/{d["n_open"].max():.0f}   '
          f'exit reasons {reasons}')

    if len(res.costs):
        by = res.costs.groupby(['sleeve', 'reason'])['cost'].sum().round(0)
        print(f'  spread paid by sleeve/reason:\n{by.to_string()}')
    print(f'  option_tc {d["option_tc"].sum():>10,.0f}   '
          f'hedge_tc {d["hedge_tc"].sum():>10,.0f}   '
          f'equity {d["equity"].iloc[-1]:>12,.0f}')

    # identity #6 -- a live test of the delta hedge, not a formality
    print(f'  |delta_pnl + hedge_pnl| max  '
          f'{(d["delta_pnl"] + d["hedge_pnl"]).abs().max():.2e}   '
          f'(must be ~0 at hedge_fraction=1.0)')
    return st


def hedge_sleeve_volga(res):
    """
    CONTAMINATION. An ATM-forward straddle has volga/vega ~ 0.0000 -- until
    spot moves off its strike, at which point it starts injecting volga into
    the sleeve the roller is holding at target. The roller CANNOT see it: its
    current= is scoped to its own sleeve. restrike_move is the only control.
    """
    if 'sleeve' not in res.positions.columns:
        return None
    lvl = (res.positions.groupby(['date', 'sleeve'])[['vega_1vp', 'volga_1vp']]
                        .sum().unstack())
    hv = lvl['volga_1vp'].get('hedge')
    wv = lvl['volga_1vp'].get('wing_convex')
    if hv is None or wv is None:
        return None
    # DO NOT lead with hv/wv. On a static vintage the wing volga decays
    # THROUGH ZERO (measured: -3,000 -> +505 by mid-life), so the ratio
    # explodes on a vanishing denominator -- 2,276% peak, which is an artifact
    # of the denominator, not contamination. |BUDGET| is the fixed non-zero
    # denominator; same argument as the tier-2 "divide by |TARGET_VOLGA|, not
    # by net volga" note in test.py.
    w_raw = scale_frame([res.daily.index[-1]])['volga'].iloc[0]
    bud_raw = abs(BUDGET) / max(w_raw, 1e-12)     # budget in RAW volga_1vp
    r_bud = (hv.abs() / bud_raw).dropna()
    r_wv  = (hv.abs() / wv.abs()).dropna()
    return {'mean': r_bud.mean(), 'peak': r_bud.max(),
            'mean_vs_wing': r_wv.mean(), 'peak_vs_wing': r_wv.max()}


def ppg_sleeve(res, sleeve='wing_convex',
               pnl='volga_pnl_be', greek='volga_1vp'):
    """
    Premium per unit of the risk that earned it, RESTRICTED TO ONE SLEEVE.

    res.pnl_per_unit_greek is book-wide, so on a hedged run its denominator
    includes the hedge sleeve's volga and it is not apples-to-apples against an
    unhedged run. volga_pnl_be (not volga_pnl) is the point: realised dsigma^2
    netted against the sigma^2 nu^2 dt the smile implied.

    !! AND THAT IS EXACTLY WHY IT CANNOT JUDGE THE HEDGE. The roller's
    `current=` is scoped to sleeve='wing_convex' and the hedge trades into
    sleeve='hedge', so the two never see each other: the wing sleeve's legs,
    P&L and carried volga are IDENTICAL hedged or not. ppg_sleeve on the wing
    sleeve is therefore INVARIANT to `band` and to `restrike_move` -- a
    constant, carrying no information about the hedge at all. It moves only
    when something changes the ROLLER (roll_days, menu, budget).

    Measured, not asserted: STAGE C prints it across five bands and it is the
    same number to four decimals every time. Judge the hedge on `equity`
    (the spread bill), on `vega_pnl` (the gate), and on sd(pnl) (the variance
    claim it actually makes) -- see hedge_metrics().
    """
    p = res.positions
    if 'sleeve' not in p.columns:
        return float('nan')
    p = p[p['sleeve'] == sleeve]
    carried = p.groupby('date')[greek].sum().abs().mean()
    return p[pnl].sum() / carried if carried else float('nan')


def hedge_metrics(res):
    """
    The figures a vega hedge can actually move.

    sd_pnl  : std of daily book P&L. THE variance-reduction claim. 10.5 found
              the vega residual is zero-mean, so the hedge is not supposed to
              add return -- it is supposed to remove this.
    vega_pnl: THE GATE. Should collapse toward noise with the hedge on.
    equity  : should get WORSE by roughly the spread bill. If it improves, be
              suspicious before being pleased.
    """
    d = res.daily
    return {
        'sd_pnl':   d['pnl'].std(),
        'vega_pnl': d['vega_pnl'].sum(),
        'volga_be': d['volga_pnl_be'].sum(),
        'opt_tc':   d['option_tc'].sum(),
        'equity':   d['equity'].iloc[-1],
        'hedge$':   (res.costs[res.costs['sleeve'] == 'hedge']['cost'].sum()
                     if len(res.costs) and 'sleeve' in res.costs.columns else 0.0),
    }


def wing_pnl(res):
    """Wing-sleeve volga_pnl_be. Used to PROVE the sleeve scoping is airtight:
    it must be bit-identical between a hedged and an unhedged run."""
    p = res.positions
    if 'sleeve' not in p.columns:
        return float('nan')
    return p[p['sleeve'] == 'wing_convex']['volga_pnl_be'].sum()


# ========================================================================= #
# STAGE A -- ONE STATIC VOLGA POSITION, NO HEDGE
#
# The baseline, and the only run here that tells you whether a hedge is even
# needed. Struck vega-neutral on day one by the pins in `target()`, then left
# entirely alone for the life of the option. Every basis point of `norm_vega`
# after day one is pure drift.
#
# READ THIS BEFORE ANYTHING ELSE. If mean/|budget| comes back at 0.05 there is
# nothing here to fix and STAGE B is a study of a spread bill.
# ========================================================================= #
if 'A' in STAGES:
    roll_A = build_roller(ONE_SHOT)
    res_A  = run(roll_A, ds, cfg(-HOLD_BDAYS))
    st_A   = show('A.  ONE STATIC VINTAGE, NO HEDGE  (the drift you are fixing)',
                  res_A, roll_A)

    # The shape of the drift is the interesting part, not just its mean: with
    # no roller re-pinning it, this should be a fairly smooth walk away from
    # zero rather than the sawtooth STAGE D produces.
    dA = normalised(res_A)
    print('\n  vega path, normalised (every other day):')
    print(dA[['date', 'n_open', 'norm_vega', 'norm_volga']]
          .dropna().iloc[::2].to_string(index=False,
                                        float_format=lambda v: f'{v:,.0f}'))


# ========================================================================= #
# STAGE B -- THE SAME POSITION + VegaHedge.  THE HEDGE IN ISOLATION.
#
# Nothing else touches vega, so every leg in the 'hedge' sleeve and every
# dollar of its spread is attributable to this one policy. That is what makes
# this the right place to learn the knobs -- in STAGE D the roller is also
# pinning vega and the two are confounded.
#
# WATCH FOR: `n_open` climbing through the run. The hedge neutralises by
# ADDING an offsetting straddle, not by closing the one it already has (only
# the re-strike path closes anything). As the wing legs decay their vega
# shrinks, the hedge goes over-hedged the other way, and it layers again.
# ========================================================================= #
if 'B' in STAGES:
    roll_B = build_roller(ONE_SHOT)
    vh_B   = build_hedge(BAND_FRAC)
    res_B  = run(Composite(roll_B, vh_B), ds, cfg(-HOLD_BDAYS))
    st_B   = show(f'B.  ONE STATIC VINTAGE + VegaHedge  '
                  f'(band={BAND_FRAC:.2f}x budget, restrike={RESTRIKE})',
                  res_B, roll_B, vh_B)

    print('\n  hedge decision log:')
    print(vh_B.frame()[['date', 'action', 'resid_vega', 'resid_raw',
                        'restruck', 'gross', 'cost']]
          .to_string(index=False, float_format=lambda v: f'{v:,.0f}'))

    cont = hedge_sleeve_volga(res_B)
    if cont:
        print(f'\n  hedge volga as % of BUDGET (fixed denominator -- trust '
              f'this one):\n    mean {cont["mean"]:.1%}  '
              f'peak {cont["peak"]:.1%}   <- what restrike_move controls')
        print(f'  vs % of LIVE wing volga: mean {cont["mean_vs_wing"]:.1%}  '
              f'peak {cont["peak_vs_wing"]:.1%}\n'
              f'    (a static vintage wing volga decays THROUGH zero, so this '
              f'ratio blows up\n     on a vanishing denominator -- artifact, '
              f'not signal)')

    if 'A' in STAGES and st_A and st_B:
        print(f'\n  {"":<22}{"unhedged":>12}{"hedged":>12}')
        for k in ('mean', 'p90', 'peak', 'pct>band'):
            f = '.1%' if k == 'pct>band' else ',.0f'
            print(f'  {k:<22}{st_A[k]:>12{f}}{st_B[k]:>12{f}}')
        print(f'  {"option_tc":<22}{res_A.daily["option_tc"].sum():>12,.0f}'
              f'{res_B.daily["option_tc"].sum():>12,.0f}')
        print(f'  {"equity":<22}{res_A.daily["equity"].iloc[-1]:>12,.0f}'
              f'{res_B.daily["equity"].iloc[-1]:>12,.0f}')
        print(f'  {"sd(pnl)":<22}{res_A.daily["pnl"].std():>12,.0f}'
              f'{res_B.daily["pnl"].std():>12,.0f}   <- the variance claim')
        print('  10.5 says equity should get worse by roughly the spread bill '
              'BECAUSE the\n  residual is zero-mean. Check that premise on THIS '
              'run before invoking it --\n  the STAGE A signed-mean line decides '
              'whether it holds here, and on a single\n  static vintage it does '
              'not.')


# ========================================================================= #
# STAGE C -- SWEEP THE BAND ON THE STATIC POSITION
#
# band and restrike_move are free parameters, and a result that only works at
# one setting is a result about that setting. Both objects rebuilt every pass:
# VegaHedge carries _last and log exactly like the roller does.
#
# Read the POST-trade columns. The pre-trade `resid_vega` in vh.frame() falls
# as you tighten the band partly because the hedge fired yesterday, which is
# circular -- see the module docstring.
# ========================================================================= #
if 'C' in STAGES:
    print(f'\n{"=" * 74}\nC.  BAND SWEEP on one static vintage\n{"=" * 74}')
    print(f'{"band":>6}{"band$":>9}{"checks":>8}{"hedged":>8}{"restruck":>10}'
          f'{"mean|v|":>10}{"p90":>9}{"peak":>10}{"opt_tc":>11}'
          f'{"hedge$":>10}{"sd(pnl)":>10}{"equity":>11}{"ppg(wng)":>10}')

    for bf in (None,) + BAND_GRID:
        r = build_roller(ONE_SHOT)
        if bf is None:
            rr, v, tag = run(r, ds, cfg(-HOLD_BDAYS)), None, 'off'
        else:
            v  = build_hedge(bf)
            rr = run(Composite(r, v), ds, cfg(-HOLD_BDAYS))
            tag = f'{bf:.2f}'
        st = drift_stats(rr, bf if bf is not None else BAND_FRAC)
        if st is None:
            print(f'{tag:>6}  !! no risk carried -- {r.report().splitlines()[0]}')
            continue
        vf   = v.frame() if v is not None else None
        acts = vf['action'].value_counts().to_dict() if vf is not None else {}
        hcost = (rr.costs[rr.costs['sleeve'] == 'hedge']['cost'].sum()
                 if len(rr.costs) else 0.0)
        m = hedge_metrics(rr)
        print(f'{tag:>6}{(bf * abs(BUDGET) if bf else 0):>9,.0f}'
              f'{(len(vf) if vf is not None else 0):>8}'
              f'{acts.get("hedged", 0):>8}'
              f'{(int(vf["restruck"].sum()) if vf is not None else 0):>10}'
              f'{st["mean"]:>10,.0f}{st["p90"]:>9,.0f}{st["peak"]:>10,.0f}'
              f'{m["opt_tc"]:>11,.0f}{hcost:>10,.0f}'
              f'{m["sd_pnl"]:>10,.0f}{m["equity"]:>11,.0f}'
              f'{ppg_sleeve(rr):>10.4f}')

    print('\n  A tighter band buys a lower mean residual for more spread. The '
          'question is\n  whether the variance reduction is worth the bill -- '
          'and the bill is bounded\n  below by STAGE D control B, not by zero.')
    print('  ppg(wng) is printed ONLY to show it is CONSTANT: the roller cannot '
          'see the\n  hedge sleeve, so the wing sleeve is the same book at every '
          'band. Read sd(pnl)\n  and equity instead -- see ppg_sleeve.__doc__.')
    print('  If the tightest bands give IDENTICAL rows then `band` is not the '
          'binding\n  constraint there -- restrike_move (which is the only thing '
          'that CLOSES a hedge\n  leg) or min_trade is. Compare `restruck` '
          'against `checks`.')


# ========================================================================= #
# STAGE D -- ROLLING top_up + VegaHedge, AND THE CONTROL THAT MATTERS
#
# Now the roller fires every 5 business days and pins vega to EXACTLY zero
# each time, so the drift is a sawtooth: zero on roll days, accumulating in the
# four days between. EXPLANATION.md 10.5 measured it on this exact config --
# mean 0.676 of budget, p90 1.380, 67.5% of days beyond 0.20.
#
# THE CONTROL IS NOT THE UNHEDGED RUN. `target()` already pins vega=0, so
# roll_days=1 is a daily vega controller with NO new code and no hedge sleeve:
# 10.5 has it at mean 0.027 of budget for +48k of spread over rd=5. So the
# question VegaHedge has to answer is not "does it neutralise vega" -- it is
# "can an ATM straddle buy rd=1 quality at closer to rd=5 cost".
#
#   A  rd=5, unhedged          the drift
#   B  rd=1, unhedged          the zero-code benchmark to BEAT
#   D  rd=5 + VegaHedge        the thing under test
#
# KNOWN INEFFICIENCY, and it inflates D's bill: on a roll day the roller pins
# the WING sleeve's vega to zero, but its current= is scoped to
# sleeve='wing_convex' so it cannot see the hedge sleeve. Total book vega is
# therefore 0 + whatever the hedge still carries, and vh (which reads both) has
# to trade to unwind its own now-redundant straddle. The hedge pays to go on
# and come off around every roll. Measure it before deciding it matters:
#   res_D.costs[res_D.costs['sleeve']=='hedge'].groupby('reason')['cost'].sum()
# split against roll_D.frame()['date'].
# ========================================================================= #
if 'D' in STAGES:
    roll_D5 = build_roller(ROLL_DAYS)
    res_D5  = run(roll_D5, ds, cfg(-ROLL_WINDOW, -5))
    st_D5   = show('D-A.  ROLLING rd=5, UNHEDGED  (the sawtooth drift)',
                   res_D5, roll_D5)

    roll_D1 = build_roller(1)
    res_D1  = run(roll_D1, ds, cfg(-ROLL_WINDOW, -5))
    st_D1   = show('D-B.  ROLLING rd=1, UNHEDGED  (zero-code benchmark to BEAT)',
                   res_D1, roll_D1)

    roll_D  = build_roller(ROLL_DAYS)
    vh_D    = build_hedge(BAND_FRAC)
    res_D   = run(Composite(roll_D, vh_D), ds, cfg(-ROLL_WINDOW, -5))
    st_D    = show(f'D-C.  ROLLING rd=5 + VegaHedge  (band={BAND_FRAC:.2f})',
                   res_D, roll_D, vh_D)

    cont = hedge_sleeve_volga(res_D)
    if cont:
        print(f'\n  hedge volga as % of BUDGET: mean {cont["mean"]:.1%}  '
              f'peak {cont["peak"]:.1%}   (vs LIVE wing volga: '
              f'{cont["mean_vs_wing"]:.1%} / {cont["peak_vs_wing"]:.1%})')

    # the churn the known inefficiency predicts
    if len(res_D.costs):
        hc = res_D.costs[res_D.costs['sleeve'] == 'hedge']
        print(f'  hedge spread by reason: '
              f'{hc.groupby("reason")["cost"].sum().round(0).to_dict()}')
        roll_dates = set(roll_D.frame()['date'])
        on_roll = hc[hc['date'].isin(roll_dates)]['cost'].sum()
        print(f'  of which on ROLL dates: {on_roll:,.0f} '
              f'({on_roll / hc["cost"].sum():.1%})  '
              f'<- the roller/hedge sleeve-scoping churn')

    # ---- POSITIVE TEST of the sleeve scoping -----------------------------
    # The roller's current= is scoped to 'wing_convex' and the hedge tags
    # 'hedge', so the two are structurally blind to each other: D-A and D-C
    # must hold the IDENTICAL wing book. If these differ, sleeve scoping has
    # leaked somewhere and nothing below is interpretable.
    wa, wd = wing_pnl(res_D5), wing_pnl(res_D)
    print(f'\n  wing-sleeve volga_pnl_be  unhedged {wa:>14,.2f}'
          f'   hedged {wd:>14,.2f}')
    print(f'  |difference| {abs(wa - wd):.2e}  <- MUST be ~0: the roller cannot '
          f'see the hedge sleeve,\n  which is also why ppg(wing) below is the '
          f'same number for A and D and cannot\n  be used to judge the hedge.')

    print(f'\n{"=" * 74}\nTHE COMPARISON THAT DECIDES IT\n{"=" * 74}')
    print(f'{"config":<24}{"mean|v|":>9}{"/budg":>7}{"p90":>8}'
          f'{"%>band":>8}{"opt_tc":>10}{"hedge$":>9}{"vega_pnl":>10}'
          f'{"sd(pnl)":>9}{"equity":>11}{"ppg(wng)":>10}')
    for nm, rr, st in (('A  rd=5 unhedged',      res_D5, st_D5),
                       ('B  rd=1 unhedged',      res_D1, st_D1),
                       ('D  rd=5 + vega hedge',  res_D,  st_D)):
        if st is None:
            continue
        m = hedge_metrics(rr)
        print(f'{nm:<24}{st["mean"]:>9,.0f}{st["mean/budget"]:>7.3f}'
              f'{st["p90"]:>8,.0f}{st["pct>band"]:>8.1%}'
              f'{m["opt_tc"]:>10,.0f}{m["hedge$"]:>9,.0f}'
              f'{m["vega_pnl"]:>10,.0f}{m["sd_pnl"]:>9,.0f}'
              f'{m["equity"]:>11,.0f}{ppg_sleeve(rr):>10.4f}')
    print('  sd(pnl) is the variance-reduction claim -- the only column that '
          'can vindicate\n  the hedge sleeve on its own terms. ppg(wng) is '
          'invariant between A and D by\n  construction; it differs for B only '
          'because rd=1 changes the ROLLER.')

    inc = (res_D1.daily['option_tc'].sum() - res_D5.daily['option_tc'].sum())
    hb  = (res_D.costs[res_D.costs['sleeve'] == 'hedge']['cost'].sum()
           if len(res_D.costs) else 0.0)
    print(f'\n  hedge spread bill      {hb:>12,.0f}')
    print(f'  control B increment    {inc:>12,.0f}   '
          f'(rd=1 option_tc - rd=5 option_tc)')
    print(f'  verdict: the hedge {"BEATS" if hb < inc else "LOSES TO"} '
          f'the zero-code rd=1 route on cost.')
    print('  If it loses, set roll_days=1 and delete the sleeve -- that is the '
          'hedge\'s\n  only claim, and it is the comparison to make before '
          'reading any P&L.')

    # THE GATE, from PROJECT_STATE.md: with the hedge on, vega_pnl should
    # collapse toward noise and volga_pnl_be should dominate the sleeve.
    print(f'\n  --- P&L by bucket (whole book) ---')
    print(f'  {"bucket":<16}{"rd=5 unhedged":>16}{"rd=5 + hedge":>16}')
    for c in ('theta_pnl', 'vega_pnl', 'volga_pnl_be',
              'vanna_pnl_be', 'gamma_pnl_be', 'recon_resid'):
        print(f'  {c:<16}{res_D5.daily[c].sum():>16,.0f}'
              f'{res_D.daily[c].sum():>16,.0f}')
    print('  THE GATE: vega_pnl should collapse toward noise and volga_pnl_be '
          'should be\n  the dominant bucket. If it is not, the sleeve is not '
          'isolating what you think.')

    # WHY THE GATE ONLY PARTLY FIRES, and how to check it. Measured on this
    # config: vega_pnl went -56,658 -> -47,437, a 16% reduction, NOT a
    # collapse -- even though post-trade net vega sits at 0.033 of budget.
    #
    # Net vega_1vp ~ 0 does NOT null vega_pnl. attribution.py prices every leg
    # off ITS OWN sigma change, so a short-wing/long-body fly whose per-leg
    # vegas cancel still earns vega_pnl whenever the SMILE RESHAPES rather than
    # shifting in parallel. An ATM straddle hedges the parallel shift only, so
    # no choice of `band` will close this gap -- it is the wrong instrument for
    # the residual that is left.
    print('\n  --- vega_pnl by sleeve: is the residual PARALLEL or SMILE risk? ---')
    for nm, rr in (('rd=5 unhedged', res_D5), ('rd=5 + hedge', res_D)):
        if 'sleeve' in rr.positions.columns:
            by = rr.positions.groupby('sleeve')['vega_pnl'].sum().round(0)
            print(f'  {nm:<16}{by.to_dict()}')
    print('  A net-vega-flat book with large vega_pnl is carrying SMILE '
          'RESHAPING risk,\n  which an ATM straddle cannot hedge. Confirm on '
          'tags:\n    res_D.positions.groupby("tag")[["vega_1vp","vega_pnl"]].sum()')
