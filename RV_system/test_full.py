"""
================================================================================
 SHORT WINGS, SYSTEMATICALLY -- an end-to-end walkthrough of the whole stack
================================================================================

WHAT THIS IS
    One linear script, no functions, that runs a real strategy: sell FX
    convexity in the ATM / 10d / 25d part of the surface, size it by RISK
    rather than by notional, and maintain that risk on a rolling basis.

    Every section prints something and explains what it is looking at, so you
    can read it top to bottom and see each layer of the stack do its job.

THE STRATEGY, IN ONE SENTENCE
    Hold a constant amount of SHORT VOLGA (short convexity) with vega and
    vanna pinned at zero, rebalanced weekly, paying real bid/offer, and
    delta-hedged daily.

WHY SHORT VOLGA
    The butterfly quote is the market's price for convexity. If implied
    vol-of-vol persistently exceeds realised vol-of-vol -- which is what
    SECTION 3 measures -- then being short that convexity earns a premium.
    Whether the premium survives the spread is the question the whole stack
    exists to answer honestly.

HOW TO READ IT
    Run it once, top to bottom. Roughly 60-90 seconds on a warm dataset cache.
    Every SECTION banner marks a layer:

        1  data          pull spot / surface / curves
        2  snapshot      one point-in-time view, no look-ahead
        3  units         the vol-of-vol yardstick and what "normalised" means
        4  costs         what you actually pay to cross
        5  sizer         solve ONE structure to a risk target
        6  reading it    residual, conditioning, leverage
        7  roller        turn the target into a maintained position
        8  engine        walk the calendar
        9  roll log      including the rolls that did nothing
       10  the book      what risk was actually carried, day by day
       11  attribution   where the money came from
       12  diagnostics   is this doing what it claims

    Nothing here is wrapped in a function. Every variable stays live in the
    session afterwards, so you can poke at `res`, `roller`, `sized`, `snap`.
================================================================================
"""

# ==============================================================================
# SECTION 0 -- IMPORTS
# ==============================================================================
# Nothing subtle here, but note WHERE things come from -- the import list is a
# map of the stack. market/ reads data, core/ prices, book/ holds positions,
# strategy/ decides size, engine/ walks the calendar.

import warnings
import pandas as pd

from market.dataset import FXVolDataset
from market.snapshot import MarketSnapshot, business_dates
from core.calendar import add_tenor
from core.conventions import fx_calendar
from book.costs import OptionCostModel
from strategy.sizer import solve, sigma_scales, GreekTarget
from strategy.roller import GreekTargetRoller
from engine.loop import EngineConfig, run

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 60)
warnings.filterwarnings('ignore')

print()
print('#' * 80)
print('#  SHORT WINGS, SYSTEMATICALLY')
print('#' * 80)


# ==============================================================================
# SECTION 1 -- THE DATA
# ==============================================================================
# FXVolDataset.build pulls spot, the quoted vol surface (ATM / RR / BF at each
# delta pillar and tenor), SOFR OIS, and forward-implied yields. It is MEMOISED
# on (pairs, days), so calling it again in the same session is free -- which is
# why every script in this repo builds it at the top without worrying.
#
# The surface is stored in RAW PERCENT (8.5 means 8.5%) and converted to decimal
# at the point of use. Do not read vol_surface values as decimals.

PAIR      = 'USDCAD'
DATA_DAYS = 400          # how much history to pull
TENOR     = '1M'         # the wing tenor we sell
WINGS     = [25, 10, 'ATM']   # <-- the menu this strategy is allowed to trade





# ------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------



print()
print('=' * 80)
print('SECTION 1 -- THE DATA')
print('=' * 80)

ds = FXVolDataset.build(pairs=[PAIR], days=DATA_DAYS)

# business_dates drives the engine loop off OBSERVED spot dates, not a synthetic
# calendar. Weekends and holidays drop out naturally, and the loop can never ask
# for a date the data does not have.
dates = business_dates(ds, PAIR)

print(f'  pair                 {PAIR}')
print(f'  business dates       {len(dates)}   {dates[0].date()} -> {dates[-1].date()}')
print(f'  spot today           {ds.get_spot(PAIR, dates[-1]):,.4f}')
print(f'  surface columns      {len([c for c in ds.vol_surface.columns if c[0] == PAIR])}'
      f'   (tenor x field for this pair)')

# The quoted surface on the last date, so you can see what the sizer is reading.
last_row = ds.vol_surface.loc[:dates[-1]].iloc[-1]
print()
print('  QUOTED SURFACE on the last date (vol points):')
print(f"    {'tenor':>6}{'ATM':>9}{'RR25':>9}{'BF25':>9}{'RR10':>9}{'BF10':>9}")
for _t in ('1W', '1M', '2M', '3M', '6M', '1Y'):
    _vals = []
    for _f in ('ATM', 'RR25', 'BF25', 'RR10', 'BF10'):
        _v = last_row.get((PAIR, _t, _f))
        _vals.append('    --  ' if _v is None or pd.isna(_v) else f'{float(_v):>9.3f}')
    print(f'    {_t:>6}' + ''.join(_vals))

# READ THIS: BF is what you are SELLING. It is positive, it is bigger in the
# wings and at short tenors, and it is the market's price for convexity.


# ==============================================================================
# SECTION 2 -- ONE SNAPSHOT
# ==============================================================================
# A MarketSnapshot is a point-in-time view of one pair. Everything it can see is
# truncated at `as_of` STRUCTURALLY -- there is no way to accidentally read
# tomorrow's data through it. That is the single most important property in the
# whole stack, because a look-ahead bug does not announce itself.

print()
print('=' * 80)
print('SECTION 2 -- ONE SNAPSHOT (point-in-time, no look-ahead)')
print('=' * 80)

AS_OF_BACK = 30                      # business days back from the end
snap = MarketSnapshot.at(ds, PAIR, dates[-AS_OF_BACK])

# Resolve the tenor to a real expiry through the pair's FX calendar. This is the
# same call the sizer and the engine make, so the dates always agree.
fxc    = fx_calendar(PAIR)
expiry = add_tenor(snap.date, TENOR, fxc)
t_days = (expiry - snap.date).days

r_d, r_f = snap.rates(t_days)

print(f'  as_of                {snap.date}')
print(f'  spot                 {snap.spot:,.4f}')
print(f'  {TENOR} expiry           {expiry}   ({t_days} calendar days)')
print(f'  {TENOR} ATM vol          {snap.atm_vol(t_days) * 100:,.3f} %')
print(f'  forward              {snap.forward(t_days):,.4f}')
print(f'  r_d (JPY) / r_f (USD){r_d * 100:>8.3f} % /{r_f * 100:>8.3f} %')

# Resolve each wing to an actual strike. solve_strike_and_vol closes the
# circularity strike -> smile vol -> delta -> strike by iterating to a fixed
# point. This is what the sizer prices its candidate legs on.
print()
print('  THE MENU THIS STRATEGY MAY TRADE -- strikes resolved on this snapshot:')
print(f"    {'pillar':>8}{'type':>7}{'strike':>12}{'smile vol':>12}")
for _p in WINGS:
    if _p == 'ATM':
        from core.option import atm_forward_strike
        _K = atm_forward_strike(snap.spot, r_d, r_f, expiry, snap.date, pair=PAIR)
        print(f"    {'ATM':>8}{'strad':>7}{_K:>12.4f}{snap.smile_vol(_K, t_days) * 100:>12.3f}")
    else:
        for _ot, _d in (('put', -_p / 100.0), ('call', +_p / 100.0)):
            _K, _sig = snap.solve_strike_and_vol(_d, _ot, expiry)
            print(f'    {str(_p) + "d":>8}{_ot:>7}{_K:>12.4f}{_sig * 100:>12.3f}')

# READ THIS: the wings trade at a HIGHER vol than ATM. That gap is the butterfly.
# Selling it is the trade. ATM enters as a STRADDLE, not a single option, because
# a naked ATM leg is a large delta the solver has no reason to want.


# ==============================================================================
# SECTION 3 -- THE YARDSTICK: nu, AND WHAT "NORMALISED" MEANS
# ==============================================================================
# This is the part that makes risk targets comparable across tenors and pairs.
#
# Every greek in the stack is already money-per-standard-move (1% spot, 1 vol
# point). But "1 vol point" is a routine Tuesday in 1W and a real event in 6M,
# so a raw volga number does not tell you how much risk you hold.
#
# The fix: express the target as base-ccy P&L for a ONE-SIGMA move over
# `horizon_days`. That needs a vol-of-vol, nu:
#
#     u = sigma * sqrt(dt) / SPOT_MOVE        one-sigma SPOT move, in "1%" units
#     w = sigma * nu * sqrt(dt) / VOL_MOVE    one-sigma VOL  move, in "vol pts"
#
#     vega  -> vega_1vp       * w             (first order)
#     volga -> volga_1vp      * w^2           (second order: move SQUARED)
#     vanna -> vanna_1pct_1vp * u * w         (cross term)
#     gamma -> gamma_1pct     * u^2
#
# WHERE nu COMES FROM. It is the vol-of-vol of the SABR fit that builds the
# marks -- the same calibration get_smile_vol performs, read for its shape
# parameters instead of evaluated at a strike. The alternative (a closed form
# off BF25) tracks it well but runs ~1.28x high, and its rho has the wrong term
# structure. Both are still reachable via ds.nu_rho_source.

print()
print('=' * 80)
print('SECTION 3 -- THE YARDSTICK (nu) AND NORMALISED UNITS')
print('=' * 80)

print(f'  nu source in use     {ds.NU_RHO_SOURCE!r}   '
      f'(set ds.nu_rho_source to override)')
print()
print('  IMPLIED vol-of-vol from the calibrated surface, by tenor:')
print(f"    {'tenor':>6}{'days':>7}{'sigma':>9}{'nu':>9}{'rho':>9}"
      f"{'1sig spot':>11}{'1sig vol':>10}")
for _t in ('1W', '1M', '3M', '6M', '1Y'):
    _e  = add_tenor(snap.date, _t, fxc)
    _td = (_e - snap.date).days
    _nu, _rho = snap.nu_rho(_td)
    _sc = sigma_scales(snap, _td, 7)
    print(f'    {_t:>6}{_td:>7}{snap.atm_vol(_td) * 100:>8.2f}%'
          f'{_nu:>9.3f}{_rho:>+9.3f}'
          f'{_sc["spot"]:>10.3f}%{_sc["vega"]:>9.3f}vp')

# The scales for OUR tenor. These are the multipliers that turn a raw greek into
# the normalised number you set as a target.
HORIZON_DAYS = 7                     # one week -- matches a weekly roll
scales = sigma_scales(snap, t_days, HORIZON_DAYS)

print()
print(f'  MULTIPLIERS at {TENOR}, horizon {HORIZON_DAYS} calendar days:')
for _g in ('spot', 'gamma', 'vega', 'vanna', 'volga', 'theta'):
    print(f'    {_g:>7}  x {scales[_g]:>10.5f}')
print()
print(f'  So one sigma over a week is a {scales["spot"]:.2f}% spot move and a '
      f'{scales["vega"]:.2f} vol point move in {TENOR} vol.')

# NOW PICK THE RISK BUDGET.
# This is the number that defines the strategy's size. Read it as:
#   "on a one-sigma weekly move in vol, the volga term costs me this much".
TARGET_VOLGA = -3_000.0              # base ccy (USD for USDJPY). NEGATIVE = short convexity.

print()
print(f'  RISK BUDGET  volga = {TARGET_VOLGA:,.0f}  normalised')
print(f'    -> in raw units that is {TARGET_VOLGA / scales["volga"]:,.1f} of volga_1vp')
print( '    -> raw volga is what res.positions / res.daily report; the normalised')
print( '       number is what you SET. Divide by the multiplier to convert.')

# CAVEAT WORTH KNOWING: nu here is IMPLIED. Realised vol-of-vol on this sample is
# several times smaller, so "one sigma" is an implied sigma, not a typical week.
# That gap is the premium the strategy is trying to harvest -- it is the point,
# not an error. Just do not read -3,000 as "what a normal week costs me".


# ==============================================================================
# SECTION 4 -- THE COST MODEL
# ==============================================================================
# Phase 1 charged nothing to trade an option. For a wing-selling strategy that
# is not a small omission -- the wing spread is the dominant cost and can eat
# the entire premium. The Book OWNS the cost model, so a strategy physically
# cannot open a position without paying.
#
# scale=1.0 is a considered guess at G10 interbank levels, NOT a measurement.
# Treat every P&L number that depends on it as a sensitivity.

print()
print('=' * 80)
print('SECTION 4 -- WHAT IT COSTS TO CROSS')
print('=' * 80)

cost_model = OptionCostModel(scale=1.0)
print(cost_model.describe(PAIR))

print('  READ THIS: wings cost more than ATM per vol point, and short tenors')
print('  cost more than long. But min_premium_frac (3% of premium) BINDS at ATM')
print('  and not at 5d, because premium is largest ATM -- so ATM can end up the')
print('  expensive leg per unit notional even though its vol spread is tightest.')


# ==============================================================================
# SECTION 5 -- SIZE ONE STRUCTURE
# ==============================================================================
# The sizer inverts the specification. Instead of "sell 10mm of 25d strangle"
# you say "hold this much volga, with vega and vanna at zero", and it returns the
# LegRequests that produce exactly that, at minimum spread paid.
#
# THREE CONSTRAINTS, so at most THREE non-zero legs -- that is a property of the
# linear program (an LP basic solution has at most (#constraints) non-zero
# variables), not of any pruning step.
#
# Note the three distinct states a greek can be in:
#     'vega': 0.0   -> actively PINNED at zero (costs a constraint)
#     key absent    -> free, whatever the structure implies
#     'volga': -3000-> hit this number

print()
print('=' * 80)
print('SECTION 5 -- SIZE ONE SHORT-WING STRUCTURE')
print('=' * 80)

target = GreekTarget(
    by_tenor     = {TENOR: dict(volga=TARGET_VOLGA, vega=0.0, vanna=0.0)},
    horizon_days = HORIZON_DAYS,
    units        = 'normalised',
)

sized = solve(
    snap,
    tenors       = TENOR,
    target       = target,
    allow_deltas = WINGS,        # <-- restricts the menu to ATM / 10d / 25d
    sleeve       = 'wings',
    cost_model   = cost_model,
)

print(sized)


# ==============================================================================
# SECTION 6 -- READING THE RESULT
# ==============================================================================
# The three honesty checks, and what "bad" looks like for each.

print()
print('=' * 80)
print('SECTION 6 -- READING THE SIZER OUTPUT')
print('=' * 80)

print(f'  residual (max |.|)   {max(abs(x) for x in sized.residual):.2e}')
print( '     ~1e-12 means the menu spanned the target exactly. Anything larger')
print( '     means it could not, and the LP fell back to least squares.')
print()
print(f'  cond(A)              {sized.condition:,.2f}')
print( '     tests whether the CONSTRAINT ROWS are independent. Fires when you')
print( '     ask for two things that are secretly the same thing. Limit 1e4.')
print()
print(f'  leverage             {sized.leverage:,.2f}')
print( '     gross greek deployed / net delivered. Tests whether the COLUMNS are')
print( '     independent. 2-6 is healthy; tens means the answer is two adjacent')
print( '     strikes taken in vast offsetting size. Limit 10. This is the guard')
print( '     that actually fires in practice.')
print()
print(f'  gross notional       {sized.gross_notional:>16,.0f}')
print(f'  entry spread         {sized.cost:>16,.0f}')
print(f'  spot delta to hedge  {sized.realised.delta_hedge:>16,.0f}  ({PAIR[:3]})')
print()
print('  RAW greeks this structure carries (base ccy per standard move):')
print(f'    {sized.realised}')
print()
print('  Same greeks, NORMALISED (one-sigma week):')
for _g in ('vega', 'vanna', 'volga', 'gamma'):
    print(f'    {_g:>6}  {sized.greek_normalised(_g, TENOR):>12,.1f}')

print()
print('  THE LEGS, as the book will receive them:')
print(f"    {'type':>6}{'dir':>5}{'notional':>16}{'delta':>8}{'tag':>10}")
for _leg in sized.legs:
    _dl = 'ATM' if _leg.atm else f'{_leg.target_delta:+.2f}'
    print(f'    {_leg.option_type:>6}{_leg.direction:>5}{_leg.notional:>16,.0f}'
          f'{_dl:>8}{_leg.tag:>10}')

# READ THIS: short the wings, long the body (or the reverse if you flip the
# sign). That IS short convexity. The exact strike mix is a COST question, not a
# risk question -- change the cost model and the mix moves.


# ==============================================================================
# SECTION 7 -- FROM A CALCULATOR TO A STRATEGY
# ==============================================================================
# A greek target has to be MAINTAINED, not re-issued. If you simply re-open the
# structure every 5 days and let vintages overlap, you asked for -3,000 of volga
# and after four rolls you are carrying -12,000. The number stops describing
# your risk.
#
#   'top_up'   read what the book carries, trade only the DIFFERENCE   <- correct
#   'replace'  close the sleeve and re-strike (pays exit spread every roll)
#   'stack'    re-issue the full target every roll (the broken behaviour)
#
# The deadband matters: with top_up the increments get small, and paying spread
# to trade 400k of notional is worse than being slightly off target.

print()
print('=' * 80)
print('SECTION 7 -- THE ROLLER')
print('=' * 80)

ROLL_DAYS = 5            # business days between attempts (~weekly)
MIN_TRADE = 1_000_000    # deadband: skip increments smaller than this
MODE      = 'top_up'

# The ONE callable in this script. The roller calls it fresh every roll, which is
# the seam where a signal eventually goes: today it returns a constant target,
# in Phase 5 it returns a function of a richness z-score and nothing else in the
# class changes.
target_fn = lambda _snap: target

roller = GreekTargetRoller(
    PAIR, TENOR, target_fn,
    roll_days = ROLL_DAYS,
    sleeve    = 'wings',
    mode      = MODE,
    min_trade = MIN_TRADE,
    solve_kw  = dict(allow_deltas=WINGS, cost_model=cost_model),
)

print(f'  mode        {MODE}')
print(f'  tenor       {TENOR}      roll every {ROLL_DAYS} business days')
print(f'  menu        {WINGS}')
print(f'  deadband    {MIN_TRADE:,.0f} gross notional')
print(f'  target      volga {TARGET_VOLGA:,.0f} normalised, vega and vanna pinned at 0')


# ==============================================================================
# SECTION 8 -- RUN THE ENGINE
# ==============================================================================
# The daily sequence, and why it is in this order:
#
#   1 SNAPSHOT   point-in-time view per pair
#   2 MARK       attribute yesterday->today on the book AS IT WAS
#   3 HEDGE P&L  P&L and carry on the spot hedge carried IN
#   4 TRADE      the strategy acts   <-- the roller runs here
#   5 REHEDGE    ONE net spot trade per pair against the NEW book
#   6 RECORD
#
# Swapping 2 and 4, or 3 and 5, produces a backtest that looks fine and is wrong.

print()
print('=' * 80)
print('SECTION 8 -- RUN')
print('=' * 80)

WINDOW = 200        # business days to run over

cfg = EngineConfig(
    pairs          = [PAIR],
    start          = dates[-WINDOW],
    end            = dates[-5],
    cost_model     = cost_model,
    hedge_fraction = 1.0,        # full daily delta hedge
    spot_tc        = 0.0001,     # 1bp on spot notional traded
    verbose        = True,
)

res = run(roller, ds, cfg)

print()
print(f'  legs opened          {len(res.trades)}')
print(f'  position-days        {len(res.positions):,}')
print(f'  peak legs open       {int(res.daily["n_open"].max())}')


# ==============================================================================
# SECTION 9 -- THE ROLL LOG (INCLUDING THE ROLLS THAT DID NOTHING)
# ==============================================================================
# A greek-target strategy has two new ways to quietly do nothing: the guard trips
# or the increment falls inside the deadband. Neither shows up in a P&L curve.
# A strategy that skipped 40% of its rolls looks fine and is not the strategy you
# specified.

print()
print('=' * 80)
print('SECTION 9 -- ROLL LOG')
print('=' * 80)

print(roller.report())
print()

log = roller.frame()
log_cols = [c for c in ['date', 'action', 'legs', 'gross', 'cost', 'closed',
                        'leverage', 'cond',
                        f'want_volga@{TENOR}', f'book_volga@{TENOR}', 'note']
            if c in log.columns]
print(log[log_cols].to_string(index=False))

print()
print('  READ THIS CAREFULLY:')
print( '   * `gross` does NOT collapse to nothing here, and that is correct for')
print(f'     this config. A {TENOR} tenor rolled every {ROLL_DAYS} business days only ever has')
print( '     ~4-5 vintages alive at once, so a big slice of the book expires every')
print( '     few rolls and has to be replaced. Run the same thing at 3M and the')
print( '     increments shrink dramatically -- 13 overlapping vintages decay much')
print( '     more gently. Shorter tenor = more turnover = more spread, and that')
print( '     trade-off is the whole of Section 12(b).')
print( '   * `legs` alternates 3 and 4. There are always at most THREE non-zero')
print( '     COLUMNS (one per constraint), but the ATM column is a straddle, so it')
print( '     becomes two physical legs whenever the solver uses it.')
print(f'   * `book_volga@{TENOR}` reads {TARGET_VOLGA:,.0f} on every row and that proves')
print( '     NOTHING. It is achieved = carried + traded with a ~1e-12 residual, so')
print( '     it is true by arithmetic whether or not top_up is reading the book.')
print( '     It IS useful for spotting an infeasible solve: want != book means the')
print( '     LP failed and fell back to least squares.')
print( '   * `action` of "deadband" or "skipped_guard" are the interesting rows.')


# ==============================================================================
# SECTION 10 -- WHAT THE BOOK ACTUALLY CARRIED
# ==============================================================================
# CRITICAL GOTCHA. res.daily's net_* columns are START-of-period greeks --
# mark_position writes prev.greeks into the row, because that is what the day's
# P&L was earned on. So the row dated D holds the book as of the CLOSE OF D-1,
# and a roll on D shows up in the row for D+1. Shift for it or you will read the
# pre-trade book and conclude the target is not being held.

print()
print('=' * 80)
print('SECTION 10 -- THE BOOK, DAY BY DAY')
print('=' * 80)

daily = res.daily.reset_index()

# Convert the raw carried volga into normalised units so it is comparable with
# the target. The scale must come from the date the GREEKS belong to, i.e. D-1,
# hence the shift.
sc_volga, sc_vega, sc_vanna = [], [], []
for _d in daily['date']:
    _s  = MarketSnapshot.at(ds, PAIR, pd.Timestamp(_d))
    _e  = add_tenor(_s.date, TENOR, fxc)
    _sc = sigma_scales(_s, (_e - _s.date).days, HORIZON_DAYS)
    sc_volga.append(_sc['volga'])
    sc_vega.append(_sc['vega'])
    sc_vanna.append(_sc['vanna'])

daily['sc_volga'], daily['sc_vega'], daily['sc_vanna'] = sc_volga, sc_vega, sc_vanna
daily['norm_volga'] = daily['net_volga_1vp']       * daily['sc_volga'].shift(1)
daily['norm_vega']  = daily['net_vega_1vp']        * daily['sc_vega'].shift(1)
daily['norm_vanna'] = daily['net_vanna_1pct_1vp']  * daily['sc_vanna'].shift(1)

print('  every 10th day (raw greeks are point-in-time levels, never cumsum them):')
print(daily[['date', 'n_open', 'net_volga_1vp', 'net_vega_1vp',
             'net_vanna_1pct_1vp', 'norm_volga']]
      .iloc[2::10].to_string(index=False,
                             float_format=lambda v: f'{v:,.1f}'))

_live = daily['norm_volga'].iloc[2:]
print()
print(f'  TARGET TRACKING   target {TARGET_VOLGA:,.0f} normalised volga')
print(f'    mean {_live.mean():>10,.0f}   sd {_live.std():>8,.0f}'
      f'   min {_live.min():>10,.0f}   max {_live.max():>10,.0f}')
print(f'    mean |error| {(_live - TARGET_VOLGA).abs().mean():>9,.0f}'
      f'   ({(_live - TARGET_VOLGA).abs().mean() / abs(TARGET_VOLGA):.1%} of target)')
print( '    On roll days the solve pins this EXACTLY -- that is arithmetic. What')
print( '    matters is the four days in between, which is what this spread shows.')


# ==============================================================================
# SECTION 11 -- WHERE THE MONEY CAME FROM
# ==============================================================================
# `pnl` = option mark-to-market + hedge P&L + hedge carry - hedge TC - option TC.
# Premium flows are NOT added separately: option_pnl is a change in mark, which
# already embeds the premium from the day the leg was struck.

print()
print('=' * 80)
print('SECTION 11 -- ATTRIBUTION')
print('=' * 80)

print('  THE MONEY (base ccy, over the whole window):')
for _c in ('option_pnl', 'hedge_pnl', 'hedge_carry', 'hedge_tc', 'option_tc'):
    _sign = -1.0 if _c in ('hedge_tc', 'option_tc') else 1.0
    print(f'    {_c:<14}{_sign * daily[_c].sum():>16,.0f}')
print(f'    {"NET equity":<14}{daily["equity"].iloc[-1]:>16,.0f}')
print(f'    {"pre option TC":<14}{daily["equity"].iloc[-1] + daily["option_tc"].sum():>16,.0f}')

print()
print('  THE TAYLOR DECOMPOSITION of option P&L (should roughly reconcile):')
for _c in ('delta_pnl', 'gamma_pnl', 'theta_pnl', 'vega_pnl', 'vanna_pnl',
           'volga_pnl', 'recon_resid'):
    print(f'    {_c:<14}{daily[_c].sum():>16,.0f}')
print( '    recon_resid is option_pnl minus the Taylor sum -- a measure of how')
print( '    well the expansion held, NOT an error to drive to zero. Large means')
print( '    a discrete jump happened.')

print()
print('  THE BREAKEVEN BUCKETS (realised move MINUS what the smile implied):')
for _c in ('gamma_pnl_be', 'vanna_pnl_be', 'volga_pnl_be'):
    print(f'    {_c:<14}{daily[_c].sum():>16,.0f}')
print( '    volga_pnl_be positive = realised vol-of-vol came in BELOW implied and')
print( '    you kept the premium. These are counterfactual decompositions -- they')
print( '    do NOT sum to equity, and a large gap between them and the money is')
print( '    something to explain, not to celebrate.')

print()
print('  BY SLEEVE (cumulative volga breakeven):')
print(res.by_sleeve('volga_pnl_be').tail(3).to_string(float_format=lambda v: f'{v:,.0f}'))


# ==============================================================================
# SECTION 12 -- IS IT DOING WHAT IT CLAIMS?
# ==============================================================================
# Three diagnostics. Each one can fail while the equity curve looks fine.

print()
print('=' * 80)
print('SECTION 12 -- DIAGNOSTICS')
print('=' * 80)

# ---- (a) NEUTRALITY DRIFT ----------------------------------------------------
# A vega-neutral structure is vega-neutral on the day it is struck -- that is
# arithmetic and cannot fail. The question is how fast it decays, because nothing
# in the stack currently re-neutralises it between rolls.
# MEASURE IT AGAINST THE BUDGET, NOT AGAINST THE REALISED VOLGA. The obvious
# ratio |vega| / |volga| is unusable: net volga wanders through zero on the odd
# day, and dividing by ~0 produces a peak in the thousands that says nothing.
# Both legs are put in NORMALISED units first -- they are then the same currency
# and directly comparable -- and divided by the volga budget, which is a fixed,
# non-zero denominator.
_after = daily.iloc[2:]
_vv = (_after['norm_vega'] / abs(TARGET_VOLGA)).abs().dropna()
_na = (_after['norm_vanna'] / abs(TARGET_VOLGA)).abs().dropna()

print('  (a) NEUTRALITY DRIFT -- residual greek as a fraction of the volga budget')
print( '      (all in normalised one-sigma units, so these are like-for-like)')
print(f'      |vega| / |budget|   mean {_vv.mean():.3f}  median {_vv.median():.3f}'
      f'  p90 {_vv.quantile(0.9):.3f}  peak {_vv.max():.3f}')
print(f'      |vanna|/ |budget|   mean {_na.mean():.3f}  median {_na.median():.3f}'
      f'  p90 {_na.quantile(0.9):.3f}  peak {_na.max():.3f}')
print(f'      days above 0.20 : vega {int((_vv > 0.2).sum())}/{len(_vv)}'
      f'  ({(_vv > 0.2).mean():.0%})'
      f'   vanna {int((_na > 0.2).sum())}/{len(_na)}  ({(_na > 0.2).mean():.0%})')
print()
print( '      Both are pinned to EXACTLY zero on every roll day -- that is what')
print( '      the vega=0 / vanna=0 constraints do, and it cannot fail. Everything')
print( '      above is what accumulates in the four days in between, when nothing')
print( '      is re-neutralising them.')
print( '      Above ~0.2 the sleeve is not isolating volga; it is part vega trade.')
print( '      That fraction is what sizes a vega re-hedge and tells you how often')
print( '      it would have to fire to be worth the ATM spread.')

# ---- (b) COST vs PREMIUM -----------------------------------------------------
print()
print('  (b) COST INTENSITY')
_gross = log['gross'].sum()
_tc    = daily['option_tc'].sum()
print(f'      gross notional traded   {_gross:>16,.0f}')
print(f'      option spread paid      {_tc:>16,.0f}'
      f'   ({_tc / max(_gross, 1) * 1e4:,.1f} bp of notional)')
print(f'      spot hedge TC           {daily["hedge_tc"].sum():>16,.0f}')
print(f'      hedge carry             {daily["hedge_carry"].sum():>16,.0f}')
print( '      Compare the spread bill against pre-cost P&L. If costs dominate,')
print( '      this is a cost-execution problem before it is an alpha problem, and')
print( '      roll cadence and strike choice matter more than any signal.')

# ---- (c) P&L PER UNIT OF RISK CARRIED ---------------------------------------
# THE metric for this strategy. Raw P&L is not comparable across configurations
# because they carry different amounts of risk; this is.
print()
print('  (c) P&L PER UNIT OF RISK CARRIED')
_mean_volga = daily['net_volga_1vp'].iloc[2:].abs().mean()
print(f'      mean |raw volga| carried  {_mean_volga:>14,.1f}')
print(f'      equity per unit volga     {daily["equity"].iloc[-1] / _mean_volga:>14,.2f}')
print(f'      volga_be per unit volga   '
      f'{res.pnl_per_unit_greek("volga_pnl_be", "volga_1vp"):>14,.4f}')
print( '      Use this, not raw P&L, to rank wings / tenors / roll cadences.')


# ==============================================================================
# WHAT TO CHANGE, AND WHAT IT WILL TELL YOU
# ==============================================================================
print()
print('=' * 80)
print('KNOBS')
print('=' * 80)
print("""
  WINGS = [25, 10, 'ATM']   which strikes the solver may use. Try [25,'ATM'] vs
                            [10,'ATM'] vs all three. Less notional in the far
                            wing does NOT mean cheaper -- compare entry spread.

  TENOR = '1M'              try '3M'. Longer tenor = less rolling = less spread,
                            but slower decay and more vega drift between rolls.

  TARGET_VOLGA              scale the book. Structure is unchanged; for a
                            single-tenor target every leg scales by the same
                            factor, so cost scales linearly too.

  ROLL_DAYS = 5             tighter tracking vs more spread. Section 10's mean
                            |error| and Section 12(b)'s bill are the trade-off.

  MIN_TRADE                 raise it to stop paying spread on noise. Watch the
                            'deadband' count in the roll log go up and the
                            tracking error follow.

  MODE                      'replace' pays an exit spread on every roll but keeps
                            a clean single vintage. 'stack' is the broken one --
                            run it once to see the risk ramp away from target.

  cost_model scale          0.0 reproduces the free-options world. Sweep it to
                            find the level at which the premium disappears; that
                            break-even is more trustworthy than any single P&L
                            number, because the absolute spreads are assumptions.

  ds.nu_rho_source          'sabr' (default) or 'closed_form'. Changes what a
                            normalised target MEANS, so it resizes the book --
                            it does not change any mark.
""")
print('=' * 80)
print('END')
print('=' * 80)
