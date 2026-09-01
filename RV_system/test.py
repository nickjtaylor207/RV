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


# ========================================================================= #
# OPTIONAL STRATEGY WRAPPERS -- an exit rule, and a way to compose it

# The engine has no exit knob. Legs die exactly one way: Book.mark_all() sees
# days_to_expiry <= 0 and calls close(reason='expiry'), which is deliberately
# NOT charged a spread (an expiring option settles to intrinsic, it is not
# traded out of). "Exit with N days of life left" is therefore a STRATEGY
# behaviour, not an EngineConfig flag -- so it lives here.

# Switch it on/off with MIN_LIFE_DAYS below. None == hold to expiry.
# ========================================================================= #





class MinLifeExit:
    """
    Close every open position in `sleeves` once it has <= `min_life_days`
    CALENDAR days of life left.  min_life_days=7 is "exit at 1W".

    Fires on EVERY date, not on the roll grid. An exit rule that only fires on
    roll days is not a 1W rule, it is a "1W, or up to a roll later" rule.

    Passing snap= is what charges the exit spread. Do NOT pass
    reason='expiry' -- Book.close reads that exact string as "settled, don't
    charge", and you would be exiting early for free. Early exits show up in
    res.costs with reason='close'.

    wake : rollers whose cadence counter is nulled on any date this fires, so
           mode='top_up' refills the hole the SAME day rather than leaving the
           book under target until the next scheduled roll. Pass wake=() to
           keep the strict roll grid and let the book run light in between --
           that is a strategy choice, not a detail.
    """

    def __init__(self, pair, sleeves, min_life_days=7, wake=()):
        self.pair          = pair
        self.sleeves       = [sleeves] if isinstance(sleeves, str) else list(sleeves)
        self.min_life_days = min_life_days
        self.wake          = list(wake)
        self.log           = []

    def on_date(self, ctx) -> None:
        if self.min_life_days is None or self.pair not in ctx.snaps:
            return
        snap = ctx.snaps[self.pair]
        hit  = []
        for sl in self.sleeves:
            for pos in list(ctx.book.open_positions(pair=self.pair, sleeve=sl)):
                if pos.days_to_expiry(snap.date) <= self.min_life_days:
                    ctx.book.close(pos.pos_id, snap.date,
                                   reason='min_life', snap=snap)
                    hit.append(pos)
        if not hit:
            return
        self.log.append({
            'date':     snap.date,
            'closed':   len(hit),
            'notional': sum(p.notional for p in hit),
            'sleeves':  ','.join(sorted({p.sleeve for p in hit})),
            'min_life': min(p.days_to_expiry(snap.date) for p in hit),
        })
        for r in self.wake:                    # let top_up refill today
            r._last = None

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    def report(self) -> str:
        if self.min_life_days is None:
            return "[min_life=off] holding every leg to expiry"
        if not self.log:
            return f"[min_life={self.min_life_days}d] no early exits fired"
        df = self.frame()
        return (f"[min_life={self.min_life_days}d] {len(df)} exit dates, "
                f"{int(df['closed'].sum())} legs closed early, "
                f"{df['notional'].sum():,.0f} gross unwound "
                f"(sleeves: {sorted(set(df['sleeves']))})")


class Composite:
    """
    Run several strategies in order on each date, since engine.run() takes one.

    THE ORDER IS THE SEMANTICS. Put the exit BEFORE the roller: mode='top_up'
    passes current=ctx.book into the sizer, so the exit-first ordering means
    the solve already sees the hole and refills it in the same trade step.
    Reverse it and you top up, then close what you just paid to put on.
    """
    def __init__(self, *strategies):
        self.strategies = strategies

    def on_date(self, ctx) -> None:
        for s in self.strategies:
            s.on_date(ctx)





















"""
-------------------------------------- Structure Solving ------------------------------------------------

"""

# pair = 'USDJPY'
# P3_DAYS = 400
# back = 1

# ds    = FXVolDataset.build(pairs=[pair], days=P3_DAYS)
# dates = business_dates(ds, pair)
# _, snap, _ = ds, MarketSnapshot.at(ds, pair, dates[-back]), dates


# res = solve(snap,
#             tenors = '3M',
#             target = GreekTarget(
#                                 by_tenor={'3M': dict(volga=0.0, vega=+20_000)},
#                                 horizon_days=7),
#             allow_deltas=[25, 'ATM'],                    
#             sleeve = 'convexity',)

# print(res)



"""
-------------------------------------------------------------------------------------------------------

"""







# PAIR      = 'USDCAD'
# DATA_DAYS = 400               # how much history to pull
# TENOR     = '1M'              # the wing tenor we sell


# WINGS     = [25, 10, 'ATM']   # <-- the menu this strategy is allowed to trade
# HORIZON_DAYS = 7              # Rolling days back sefining 
# TARGET_VOLGA = -3_000.0       # Target normalized Volga Risk Exposure
# AS_OF_BACK = 30               # business days back from the end

# ROLL_DAYS = 5                 # business days between attempts (~weekly)
# MODE      = 'top_up'

# MIN_LIFE_DAYS = None          # CALENDAR days of life at which a leg is exited.
#                               #   7    -> exit at 1W left  (effective hold ~23d)
#                               #   None -> hold to expiry (original behaviour)
# WAKE_ROLLER   = True          # let top_up refill on the exit date itself



# # ------------------------------------------------------------------------


# # ----- The world's Data -----
# cost_model = OptionCostModel(scale=1.0)
# ds = FXVolDataset.build(pairs=[PAIR], days=DATA_DAYS)
# dates = business_dates(ds, PAIR)


# # ----- Risk to take -----
# target = GreekTarget(
#     by_tenor     = {TENOR: dict(volga=TARGET_VOLGA, 
#                                 vega=0.0, 
#                                 vanna=0.0)},
#     horizon_days = HORIZON_DAYS,
#     units        = 'normalised',)



# # ----- Strategy Objective -----
# target_fn = lambda _snap: target
# roller = GreekTargetRoller(
#     PAIR, TENOR, target_fn,
#     roll_days = ROLL_DAYS,
#     sleeve    = 'wing_convex',
#     mode      = MODE,
#     min_trade = 1_000_000 ,
#     solve_kw  = dict(allow_deltas=WINGS, cost_model=cost_model),
# )


# # ----- Exit rule (optional) -----
# # Same sleeve as the roller, or it closes nothing. Exit runs FIRST each day.
# exiter = MinLifeExit(
#     PAIR,
#     sleeves       = 'wing_convex',
#     min_life_days = MIN_LIFE_DAYS,
#     wake          = [roller] if WAKE_ROLLER else (),
# )
# strategy = roller if MIN_LIFE_DAYS is None else Composite(exiter, roller)




# # ----- The Run Engine Config -----
# WINDOW = 200        # business days to run over
# cfg = EngineConfig(
#     pairs          = [PAIR],
#     start          = dates[-WINDOW],
#     end            = dates[-5],
#     cost_model     = cost_model,
#     hedge_fraction = 1.0,        # full daily delta hedge
#     spot_tc        = 0.0001,     # 1bp on spot notional traded
#     verbose        = True,
# )

# res = run(strategy, ds, cfg)




# ------------------------------------------------------------------------
# ---------------------------- Analysis ----------------------------------
# ------------------------------------------------------------------------



## ----- 1)  Trade what I asked  -----
# print(roller.report())
# print(exiter.report())
# log = roller.frame()
# print(log[['date','action','legs','gross','cost','leverage','cond', f'want_volga@{TENOR}', f'book_volga@{TENOR}','note']].to_string(index=False))
# print(sorted(res.trades['target_delta'].round(2).dropna().unique()))
# print(res.trades['sleeve'].unique(), res.trades['tenor_label'].unique())
# print(res.daily['n_open'].describe())




## ----- 2)  Is this the risk I want  -----
# d = res.daily.reset_index()
# # net_* columns are START-of-period greeks: row D is the book at close of D-1.
# # A roll on D shows up on D+1. Shift, or you'll read the pre-trade book.
# print(d[['date','n_open','net_volga_1vp','net_vega_1vp','net_vanna_1pct_1vp']].iloc[2::10])

# from strategy.sizer import sigma_scales
# from core.calendar import add_tenor
# from core.conventions import fx_calendar
# fxc = fx_calendar(PAIR)
# sc = []
# for x in d['date']:
#     s = MarketSnapshot.at(ds, PAIR, pd.Timestamp(x))
#     e = add_tenor(s.date, TENOR, fxc)
#     sc.append(sigma_scales(s, (e - s.date).days, HORIZON_DAYS))
# d['norm_volga'] = d['net_volga_1vp'] * pd.Series([x['volga'] for x in sc]).shift(1)
# d['norm_vega']  = d['net_vega_1vp']  * pd.Series([x['vega']  for x in sc]).shift(1)

# print((d['norm_volga'] - TARGET_VOLGA).abs().mean())







## ----- 3)  Do the books balance  -----

# d, p, log = res.daily, res.positions, roller.frame()

# # 1  the P&L definition
# print((d['pnl'] - (d['option_pnl'] + d['hedge_pnl'] + d['hedge_carry']
#              - d['hedge_tc'] - d['option_tc'])).abs().max())                

# # 2  equity is just the cumsum
# print(abs(d['equity'].iloc[-1] - d['pnl'].sum()))      

# # 3  the daily TC column agrees with the cost ledger
# print(abs(d['option_tc'].sum() - res.costs['cost'].sum()))    

# # 4  every position's Taylor expansion closes
# tay = p[['delta_pnl','gamma_pnl','theta_pnl','vega_pnl',
#          'vanna_pnl','volga_pnl','recon_resid']].sum(axis=1)
# print((p['option_pnl'] - tay).abs().max())  

# # 5  positions roll up to the daily frame
# print(abs(p.groupby('date')['option_pnl'].sum().sum() - d['option_pnl'].sum()))  

# # 6  at hedge_fraction=1.0, delta P&L is exactly cancelled by the hedge
# print((d['delta_pnl'] + d['hedge_pnl']).abs().max())  

# # 7  the sizer's estimate equals what the Book charged
# print(abs(log['cost'].sum() - d['option_tc'].sum())) 





# ----- 4)  audit one position end to end  -----


# pid = int(res.positions['pos_id'].iloc[0])
# one = res.positions[res.positions['pos_id'] == pid]
# print(one[['date','t_days','spot','sigma','notional','option_pnl', 'theta_pnl','vega_pnl','volga_pnl','expired']].to_string(index=False))
# print(res.trades[res.trades['pos_id'] == pid].T)











# ========================================================================= #
# ========================================================================= #
#                                                                           #
#   THREE FULL EXAMPLE RUNS                                                 #
#   Short WING convexity (volga) and short SKEW convexity (vanna),          #
#   with and without the 1W min-life exit.                                  #
#                                                                           #
#   Uncomment a block and run it. `ds` is memoised on (pairs, days), so
#   rebuilding the same dataset is free -- but run() calls reset_position_ids()
#   every time, so pos_id is NOT comparable across runs. Compare on
#   pnl_per_unit_greek, never on raw P&L: these configs carry different
#   amounts of risk, and raw P&L has no common denominator.
#
#   What "short convexity" means in each case, one line each:
#     WING convexity = volga. Short it -> sell the 25d/10d wings, buy the ATM
#                      body. Earns when realised vol-of-vol < implied.
#                      Symmetric in spot.
#     SKEW convexity = vanna. Short it -> a risk reversal (sell one wing, buy
#                      the other) with vega AND volga pinned flat. Earns when
#                      realised spot/vol co-movement < the smile's implied
#                      slope. Directional in the spot-vol correlation.
#
#   Both targets PIN the greeks they are not trying to own -- three
#   constraints, so the menu must span at least three independent columns.
#   allow_deltas=[25, 10, 'ATM'] gives five (25c, 25p, 10c, 10p, ATM
#   straddle), which is why the same menu serves both sleeves.
# ========================================================================= #

EX_PAIR    = 'USDCAD'                       # Single Pair Traded
EX_TENOR   = '3M'                           # Tenor 
EX_DAYS    = int(365 * 5)                             # Historic Data Pulled
EX_WINDOW  = int(252 * 5)                             # Business Days of Backtesting
EX_HORIZON = 7                              # Risk Normalization Days Forward for GreekTarget
EX_MENU    = [10, 'ATM']                    # Smile Pillars Available to Use

ex_cm    = OptionCostModel(scale=0.0)                               # Vol Spread Across the Surface
ex_ds    = FXVolDataset.build(pairs=[EX_PAIR], days=EX_DAYS)        #   
ex_dates = business_dates(ex_ds, EX_PAIR)                           # 


def ex_cfg():
    """Fresh config per run. Identical across all three, deliberately."""
    return EngineConfig(
        pairs          = [EX_PAIR],
        start          = ex_dates[-EX_WINDOW],
        end            = ex_dates[-5],
        cost_model     = ex_cm,        # SAME object the sizer optimises against
        hedge_fraction = 1.0,
        spot_tc        = 0.0001,
        verbose        = True,)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

"""
-------------------- SHORT WING CONVEXITY, EXITED AT 1W LEFT --------------------
"""


tgt_2 = GreekTarget(
    by_tenor     = {EX_TENOR: dict(volga=-30_000.0, vega=0.0, vanna=0.0)},
    horizon_days = EX_HORIZON,
    units        = 'normalised')

roll_2 = GreekTargetRoller(
    EX_PAIR, EX_TENOR, lambda _s: tgt_2,
    roll_days = 5,
    sleeve    = 'wing_convex',
    mode      = 'top_up',
    min_trade = 1_000_000,
    solve_kw  = dict(allow_deltas=EX_MENU, cost_model=ex_cm))

exit_2 = MinLifeExit(EX_PAIR, 'wing_convex',
                     min_life_days = 7,          # <-- 1W of life left
                     wake          = [roll_2])   # refill the hole same day

res_2 = run(Composite(exit_2, roll_2), ex_ds, ex_cfg())   # EXIT FIRST



from run.report import report
from run.dashboard import dashboard

d = report(res_2, ex_ds, EX_PAIR, EX_TENOR, EX_HORIZON,
           roller=roll_2, exiter=exit_2)              # deep-dive tiers

dashboard(res_2, ex_ds, EX_PAIR, EX_TENOR, EX_HORIZON, budget=-30_000.0,
          roller=roll_2, exiter=exit_2, units='both', d=d,
          path='dash.png')  






# from run.report import report, chart_risk, chart_attribution

# d = report(res_2, ex_ds, EX_PAIR, EX_TENOR, EX_HORIZON,
#            roller=roll_2, exiter=exit_2)          # budget auto-read from the roll log
# chart_risk(d, -30_000.0, roller=roll_2, exiter=exit_2, path='risk.png')
# chart_attribution(res_2, path='attrib.png')


















# ========================================================================= #
# EXAMPLE 1 -- SHORT WING CONVEXITY, HELD TO EXPIRY        (the baseline)
#
# Short volga with vega and vanna pinned flat: sell the wings, buy the ATM
# body, sized to sit at -3,000 of normalised volga. Nothing exits early --
# every leg dies at expiry and pays NO exit spread, because Book.close only
# skips the charge for reason='expiry'. This is the run the exit rule has to
# beat, and the reason to run it first.
# ========================================================================= #


# tgt_1 = GreekTarget(
#     by_tenor     = {EX_TENOR: dict(volga=-3_000.0, 
#                                    vega=0.0, 
#                                    vanna=0.0)},
#     horizon_days = EX_HORIZON,
#     units        = 'normalised')


# roll_1 = GreekTargetRoller(
#     EX_PAIR, EX_TENOR, 
#     lambda _s: tgt_1,
#     roll_days = 5,
#     sleeve    = 'wing_convex',
#     mode      = 'top_up',
#     min_trade = 1_000_000,
#     solve_kw  = dict(allow_deltas=EX_MENU, cost_model=ex_cm))


# exit_1 = MinLifeExit(EX_PAIR, 'wing_convex', min_life_days=None)   # OFF


# res_1 = run(roll_1, ex_ds, ex_cfg())      # no Composite needed when it is off



# Expect: exit reasons {'expiry': N} and nothing else; min t_days == 0;
# option TC entirely reason 'open'; n_open median ~16 = (30/7) x ~3.7 legs
# per roll. If n_open does not match that arithmetic, either vintages are
# not expiring or the roller is not firing -- check report() before P&L.








# # ========================================================================= #
# # EXAMPLE 2 -- SHORT WING CONVEXITY, EXITED AT 1W LEFT
# #
# # Identical target, cadence, menu, window and cost model. The ONLY change is
# # the exit rule, so the change in pnl_per_unit_greek is attributable to it.
# #
# # Two effects pull opposite ways and this run measures the NET:
# #   (+) you drop the final week, where a short-wing book carries the most
# #       gamma and pin risk per unit of premium still left to collect
# #   (-) you now pay the CLOSE spread on every single leg -- roughly doubling
# #       per-leg cost on a strategy where cost is already the binding
# #       constraint. book/book.py:131 calls this asymmetry out explicitly as a
# #       genuine reason to prefer holding to expiry.
# #
# # Do not assume the sign of the net. That is the experiment.
# # ========================================================================= #

 

# tgt_2 = GreekTarget(
#     by_tenor     = {EX_TENOR: dict(volga=-30_000.0, vega=0.0, vanna=0.0)},
#     horizon_days = EX_HORIZON,
#     units        = 'normalised')

# roll_2 = GreekTargetRoller(
#     EX_PAIR, EX_TENOR, lambda _s: tgt_2,
#     roll_days = 5,
#     sleeve    = 'wing_convex',
#     mode      = 'top_up',
#     min_trade = 1_000_000,
#     solve_kw  = dict(allow_deltas=EX_MENU, cost_model=ex_cm))

# exit_2 = MinLifeExit(EX_PAIR, 'wing_convex',
#                      min_life_days = 7,          # <-- 1W of life left
#                      wake          = [roll_2])   # refill the hole same day

# res_2 = run(Composite(exit_2, roll_2), ex_ds, ex_cfg())   # EXIT FIRST



# print((res_2.daily).tail(5).T)

# print((res_2.positions).tail(12).T)




# closes = res_2.costs[res_2.costs['reason'] == 'close']['cost'].sum()
# print(f'\nexit rule, volga_be per unit |volga|: '
#       f'{ppg_1:.4f} -> {ppg_2:.4f}  ({(ppg_2 / ppg_1 - 1) * 100:+.1f}%)')
# print(f'exit spread paid that Example 1 did not pay: {closes:,.0f} '
#       f'({closes / res_1.daily["option_tc"].sum():.1%} of baseline option TC)')
# print(exit_2.frame().to_string(index=False))





# # Expect: exit reasons dominated by 'min_life' with few or no 'expiry';
# # min t_days ~7 and never 0; n_open median ~12 = ((30-7)/7) x ~3.7; and a
# # visibly larger option_tc than Example 1.
#
# # The threshold is a free parameter, and a result that only works at exactly
# # 7 days is a result about 7, not about the strategy. Sweep it before you
# # believe either number -- rebuild BOTH roller and exiter inside the loop,
# # since a roller carries state (_last, log) across runs:
# #
# # for mnl in (None, 3, 5, 7, 10, 14):
# #     r = GreekTargetRoller(EX_PAIR, EX_TENOR, lambda _s: tgt_2, roll_days=5,
# #                           sleeve='wing_convex', mode='top_up',
# #                           min_trade=1_000_000,
# #                           solve_kw=dict(allow_deltas=EX_MENU, cost_model=ex_cm))
# #     x = MinLifeExit(EX_PAIR, 'wing_convex', min_life_days=mnl, wake=[r])
# #     rr = run(r if mnl is None else Composite(x, r), ex_ds, ex_cfg())
# #     print(mnl, round(rr.pnl_per_unit_greek('volga_pnl_be', 'volga_1vp'), 4),
# #           round(rr.daily['option_tc'].sum()), rr.daily['n_open'].median())






# # ========================================================================= #
# # EXAMPLE 3 -- SHORT SKEW CONVEXITY (vanna), EXITED AT 1W LEFT
# #
# # A different risk entirely. vanna is the target; vega AND volga are pinned
# # flat, so the solve cannot pay for its vanna with wing convexity -- it has
# # to find a risk reversal that is neutral to both level and vol-of-vol. That
# # is a harder LP than Example 1, the columns net a small number out of two
# # large opposing wings, and the leverage guard trips more often. Read
# # roll_3.report() for the skipped_guard tally BEFORE reading any P&L: a run
# # that skipped half its rolls is not the strategy you specified.
# #
# # The separate `sleeve` is not cosmetic. sleeve scopes both what top_up reads
# # as existing inventory and what the exiter is allowed to close. Share a
# # sleeve name with the wing book and each roller would size against the
# # other's positions.
# # ========================================================================= #

# TARGET_VANNA = -3_000.0

# tgt_3 = GreekTarget(
#     by_tenor     = {EX_TENOR: dict(vanna=TARGET_VANNA, vega=0.0, volga=0.0)},
#     horizon_days = EX_HORIZON,
#     units        = 'normalised')

# roll_3 = GreekTargetRoller(
#     EX_PAIR, EX_TENOR, lambda _s: tgt_3,
#     roll_days = 5,
#     sleeve    = 'skew_convex',            # <-- NOT 'wing_convex'
#     mode      = 'top_up',
#     min_trade = 1_000_000,
#     solve_kw  = dict(
#         allow_deltas     = EX_MENU,
#         cost_model       = ex_cm,
#         # Gross/net runs hotter for a vanna-only target than for volga. If
#         # report() comes back mostly skipped_guard, raise this DELIBERATELY
#         # and read the leverage column -- do not switch the guard off.
#         require_leverage = 15.0,
#     ))

# exit_3 = MinLifeExit(EX_PAIR, 'skew_convex',
#                      min_life_days = 7,
#                      wake          = [roll_3])

# res_3 = run(Composite(exit_3, roll_3), ex_ds, ex_cfg())


# # ppg_3 is NOT comparable to ppg_1 / ppg_2 -- different greek, different
# # denominator. To judge the exit rule here, build this sleeve's own
# # hold-to-expiry twin (min_life_days=None) and compare against that.

# # Shape check: a skew short should be LOPSIDED in the deltas it trades,
# # where the wing short is roughly symmetric. If both look symmetric, the
# # target is not doing what you think.
# print(res_3.trades.groupby(['option_type', 'target_delta'])['notional']
#                   .sum().round(0))
# print(res_3.positions.groupby('tag')[['vega_1vp', 'volga_1vp',
#                                       'vanna_1pct_1vp', 'theta_1d']].mean())
#
# # Tracking error, converted back into the units the target was set in. The
# # shift(1) matters: net_* columns are START-of-period greeks, so row D is
# # the book at the close of D-1.
# fxc = fx_calendar(EX_PAIR)
# dd  = res_3.daily.reset_index()
# sc  = []
# for x in dd['date']:
#     s = MarketSnapshot.at(ex_ds, EX_PAIR, pd.Timestamp(x))
#     e = add_tenor(s.date, EX_TENOR, fxc)
#     sc.append(sigma_scales(s, (e - s.date).days, EX_HORIZON))
# dd['norm_vanna'] = (dd['net_vanna_1pct_1vp']
#                     * pd.Series([q['vanna'] for q in sc]).shift(1))
# te = (dd['norm_vanna'] - TARGET_VANNA).abs().mean()
# print(f'mean |norm_vanna - target| = {te:,.0f}  '
#       f'({te / abs(TARGET_VANNA):.1%} of target)')


# # ========================================================================= #
# # BONUS -- both sleeves in ONE book, which is what Composite is for
# #
# # One netted delta hedge across both sleeves instead of two, and
# # res.by_sleeve() splits the attribution back apart afterwards. Order still
# # holds: ALL exits, then all rollers.
# #
# # Note this re-runs roll_2/roll_3, which already carry state from Examples 2
# # and 3 -- their .log and ._last are stale. Rebuild them fresh for a clean
# # read; the line below is the wiring, not a result.
# # ========================================================================= #
#
# res_both = run(Composite(exit_2, exit_3, roll_2, roll_3), ex_ds, ex_cfg())
# print(res_both.by_sleeve('volga_pnl_be').tail(1))
# print(res_both.by_sleeve('vanna_pnl_be').tail(1))
# print(f'spot TC netted across sleeves: {res_both.daily["hedge_tc"].sum():,.0f}'
#       f'   vs run separately: '
#       f'{res_2.daily["hedge_tc"].sum() + res_3.daily["hedge_tc"].sum():,.0f}')






# ========================================================================= 
# ========================================================================= 
#                                                                           
#   EXAMPLE 4 -- SHORT VOLGA, HELD TO MATURITY, VEGA HEDGED                 
#   PHASE 4. Needs `engine/hedging.py`.                                     
#                                                                           
#   Same short-wing-convexity book as Example 1 -- volga at -3,000, vega and
#   vanna pinned, every leg held to natural expiry so nothing pays an exit
#   spread -- with one thing added: a rolled ATM straddle that flattens the
#   residual vega that accumulates BETWEEN rolls.
#
#   WHY THIS RUN EXISTS. A vega-neutral fly is neutral only on the day it is
#   struck. EXPLANATION.md 10.5 measured the drift on exactly this config:
#
#       1M, roll_days=5:  mean 0.676  median 0.466  p90 1.380  peak 8.62
#                         67.5% of days |vega|/budget > 0.20
#
#   pinned to EXACTLY zero on every roll day, with all of that accumulating in
#   the four days in between. So "short volga" is, two thirds of the time, a
#   short-volga-plus-two-thirds-of-a-budget-of-vega position.
#
#   WHAT TO EXPECT BEFORE YOU RUN IT. 10.5 also found the residual is
#   essentially ZERO-MEAN -- signed mean 209 against a mean absolute of 2,029.
#   So this buys attribution purity and variance reduction, NOT return. Final
#   equity should get WORSE by roughly the hedge's spread bill. If it improves,
#   be suspicious before being pleased.
#
#   Requires the Example-4 plumbing (ex_cm / ex_ds / ex_cfg / ex_summary) from
#   the shared block above, so uncomment that first.
# ========================================================================= #


# from engine.hedging import VegaHedge
# # engine/hedging.py also ships a `Composite` identical to the one defined at
# # the top of this file. Either works; using the local one to avoid shadowing.

# EX_BUDGET  = -30_000.0                # normalised volga. The hedge band is a
#                                      # FRACTION of this, so they must agree.
# VEGA_BAND  = 0.20 * abs(EX_BUDGET)   # = 600. The 0.20 threshold in 10.5.
# RESTRIKE   = 0.01                    # re-strike the straddle on a 1% spot move

# tgt_4 = GreekTarget(
#     by_tenor     = {EX_TENOR: dict(volga=EX_BUDGET, vega=0.0, vanna=0.0)},
#     horizon_days = EX_HORIZON,
#     units        = 'normalised')

# roll_4 = GreekTargetRoller(
#     EX_PAIR, EX_TENOR, lambda _s: tgt_4,
#     roll_days = 5,
#     sleeve    = 'wing_convex',
#     mode      = 'top_up',
#     min_trade = 1_000_000,
#     solve_kw  = dict(allow_deltas=EX_MENU, cost_model=ex_cm))

# vh_4 = VegaHedge(
#     EX_PAIR, EX_TENOR,
#     read_sleeves  = ['wing_convex'],   # 'hedge' is appended automatically --
#                                        # the hedge's OWN decaying vega has to
#                                        # be visible to the next check
#     band          = VEGA_BAND,
#     horizon_days  = EX_HORIZON,        # MUST match the roller's, or the band
#                                        # is a fraction of a different sigma
#     hedge_sleeve  = 'hedge',           # keep it separate or the gate below
#                                        # cannot be read
#     restrike_move = RESTRIKE,
#     min_trade     = 250_000,           # smaller than the roller's: the whole
#                                        # point is to trade small increments
#     cost_model    = ex_cm,             # the SAME object, as always
#     check_days    = 1)                 # every day. The deadband, not the
#                                        # cadence, is what stops trades.



# # HELD TO MATURITY: no exiter in the Composite at all. A MinLifeExit with
# # min_life_days=None is passed to ex_summary only so the report line records
# # the choice explicitly rather than leaving it implied.


# exit_4 = MinLifeExit(EX_PAIR, ['wing_convex', 'hedge'], 
#                      min_life_days = 7,
#                      wake          =[roll_4])


# # KNOWN INEFFICIENCY, and the sweep below is what measures it. On a roll day
# # the roller pins the WING sleeve's vega to zero -- but its current= is scoped
# # to sleeve='wing_convex', so it cannot see the hedge sleeve. Total book vega
# # is therefore 0 + whatever the hedge is still carrying, and vh_4 (which does
# # read both) has to trade to unwind its own now-redundant straddle. The hedge
# # pays to go on and come off around every roll. That churn is real spread and
# # it inflates the hedge bill against control B below.
# #
# # The alternative is to let the ROLLER read both sleeves, so it sizes the
# # wings against the hedge's vega and the hedge stays put. That needs a roller
# # subclass -- solve()'s sleeve= does double duty, so there is no argument for
# # "read these sleeves, tag as that one". Measure the churn before writing it:
# #   res_4.costs[res_4.costs['sleeve'] == 'hedge'].groupby('reason')['cost'].sum()
# # split against the roll dates in roll_4.frame()['date'].

# res_4 = run(Composite(roll_4, vh_4), ex_ds, ex_cfg())   # roller THEN hedge

# print(vh_4.report())


# # ----- THE GATE -----------------------------------------------------------
# # PROJECT_STATE.md: "vega_pnl should collapse toward noise and volga_pnl_be
# # should become the dominant bucket in the convexity sleeve. If it does not,
# # the sleeve is not isolating what you think it is."
#
# print('\n--- P&L by bucket, whole book ---')
# for c in ('theta_pnl', 'vega_pnl', 'volga_pnl_be',
#           'vanna_pnl_be', 'gamma_pnl_be', 'recon_resid'):
#     print(f'  {c:<14} {res_4.daily[c].sum():>14,.0f}')
# print(f'  {"option_tc":<14} {res_4.daily["option_tc"].sum():>14,.0f}'
#       f'   {"hedge_tc":<10} {res_4.daily["hedge_tc"].sum():>12,.0f}')
#
# # Did the hedge pay for itself out of the sleeve it was protecting?
# print('\n--- cumulative volga_pnl_be by sleeve ---')
# print(res_4.by_sleeve('volga_pnl_be').tail(1))
# print('\n--- who paid what spread ---')
# print(res_4.costs.groupby(['sleeve', 'reason'])['cost'].sum().round(0))


# # ----- CONTAMINATION: is the hedge sleeve injecting volga? ----------------
# # A straddle struck at the ATM forward has volga_1vp/vega_1vp ~ 0.0000. An
# # off-strike one does not: at a 1% spot move it is 0.027, at 2% it is 0.110.
# # The roller CANNOT see this -- its current= is scoped to sleeve
# # 'wing_convex' -- so the book's true volga drifts off target by whatever the
# # hedge sleeve carries, invisibly. RESTRIKE is the control; this measures
# # whether it is tight enough. Expect the hedge sleeve to run at a few percent
# # of the wing sleeve, and no more than ~10% at the peak.
#
# lvl = (res_4.positions.groupby(['date', 'sleeve'])[['vega_1vp', 'volga_1vp']]
#                       .sum().unstack())
# print('\n--- mean |greek level| by sleeve ---')
# print(lvl.abs().mean().round(1))
# hv = lvl['volga_1vp'].get('hedge')
# wv = lvl['volga_1vp'].get('wing_convex')
# if hv is not None and wv is not None:
#     print(f'hedge volga as % of wing volga:  mean '
#           f'{(hv.abs() / wv.abs()).mean():.1%}   peak '
#           f'{(hv.abs() / wv.abs()).max():.1%}')


# # ----- THE COMPARISON THAT DECIDES THE PHASE ------------------------------
# # ppg from ex_summary is BOOK-wide, so a hedged run's denominator includes
# # the hedge sleeve's volga and is NOT apples-to-apples against Example 1.
# # Restrict to the wing sleeve for the honest comparison.
#
# def ppg_sleeve(res, sleeve='wing_convex',
#                pnl='volga_pnl_be', greek='volga_1vp'):
#     p = res.positions
#     p = p[p['sleeve'] == sleeve]
#     carried = p.groupby('date')[greek].sum().abs().mean()
#     return p[pnl].sum() / carried if carried else float('nan')
#
# # CONTROL A -- the same book with no hedge at all. Rebuild the roller: it
# # carries _last and log across runs.
# roll_4a = GreekTargetRoller(
#     EX_PAIR, EX_TENOR, lambda _s: tgt_4, roll_days=5, sleeve='wing_convex',
#     mode='top_up', min_trade=1_000_000,
#     solve_kw=dict(allow_deltas=EX_MENU, cost_model=ex_cm))
# res_4a = run(roll_4a, ex_ds, ex_cfg())
#
# # CONTROL B -- the ZERO-CODE version of Phase 4. tgt_4 already pins vega=0,
# # so roll_days=1 re-neutralises vega every single day with no hedge sleeve at
# # all. 10.5 measured it: mean|vega|/budget 0.027, p90 0.000, 5.7% of days out
# # of band, at option_tc 116,404 against 68,291 for rd=5 -- i.e. Phase 4's
# # goal, reached for +48k of spread. THIS is what the ATM straddle has to beat
# # on cost, not the unhedged run.
# roll_4b = GreekTargetRoller(
#     EX_PAIR, EX_TENOR, lambda _s: tgt_4, roll_days=1, sleeve='wing_convex',
#     mode='top_up', min_trade=1_000_000,
#     solve_kw=dict(allow_deltas=EX_MENU, cost_model=ex_cm))
# res_4b = run(roll_4b, ex_ds, ex_cfg())
#
# print(f'\n{"config":<34}{"option_tc":>12}{"vega_pnl":>12}'
#       f'{"volga_be":>12}{"ppg(wing)":>12}{"equity":>14}')
# for nm, r in (('A  unhedged, rd=5',         res_4a),
#               ('B  rd=1  (zero-code Ph4)',  res_4b),
#               ('4  rd=5 + ATM vega hedge',  res_4)):
#     print(f'{nm:<34}{r.daily["option_tc"].sum():>12,.0f}'
#           f'{r.daily["vega_pnl"].sum():>12,.0f}'
#           f'{r.daily["volga_pnl_be"].sum():>12,.0f}'
#           f'{ppg_sleeve(r):>12.4f}{r.daily["equity"].iloc[-1]:>14,.0f}')
#
# # A PREDICTION TO CHECK IT AGAINST, from 10.5 rather than from this run:
# # a 1M ATM straddle costs ~$0.25 of spread per unit of vega_1vp neutralised.
# # Mean residual raw vega is 2,029/w = 848 (w = sqrt(5.730) at 1M), so ~$212
# # per full flattening. At a 0.20 band it fires on roughly 68% of ~196 days
# # ~= 130 times -> ~28k. Control B's increment is +48k. If the hedge lands
# # materially above 48k it has failed its only claim and you should just set
# # roll_days=1 and delete the sleeve.
# print(f'\nhedge spread bill {res_4.costs[res_4.costs["sleeve"] == "hedge"]["cost"].sum():,.0f}'
#       f'   vs control B increment '
#       f'{res_4b.daily["option_tc"].sum() - res_4a.daily["option_tc"].sum():,.0f}')


# # ----- THE POLICY SWEEP --------------------------------------------------
# # band and restrike_move are free parameters and a result that only works at
# # one setting is a result about that setting. Rebuild BOTH objects each pass;
# # rollers and hedgers are stateful.
#
# print(f'\n{"band/budget":>12}{"restrike":>10}{"option_tc":>12}'
#       f'{"mean|resid|":>13}{"p90":>9}{"%>band":>9}{"ppg(wing)":>12}')
# for band_frac in (None, 0.10, 0.20, 0.50, 1.00):
#     r = GreekTargetRoller(
#         EX_PAIR, EX_TENOR, lambda _s: tgt_4, roll_days=5,
#         sleeve='wing_convex', mode='top_up', min_trade=1_000_000,
#         solve_kw=dict(allow_deltas=EX_MENU, cost_model=ex_cm))
#     if band_frac is None:
#         rr, tag = run(r, ex_ds, ex_cfg()), 'off'
#         resid = None
#     else:
#         v = VegaHedge(EX_PAIR, EX_TENOR, ['wing_convex'],
#                       band=band_frac * abs(EX_BUDGET),
#                       horizon_days=EX_HORIZON, restrike_move=RESTRIKE,
#                       min_trade=250_000, cost_model=ex_cm, check_days=1)
#         rr, tag = run(Composite(r, v), ex_ds, ex_cfg()), f'{band_frac:.2f}'
#         resid = v.frame()['resid_vega'].abs()
#     print(f'{tag:>12}{RESTRIKE:>10.3f}{rr.daily["option_tc"].sum():>12,.0f}'
#           f'{(resid.mean() if resid is not None else float("nan")):>13,.0f}'
#           f'{(resid.quantile(0.9) if resid is not None else float("nan")):>9,.0f}'
#           f'{((resid > band_frac * abs(EX_BUDGET)).mean() if resid is not None else float("nan")):>9.1%}'
#           f'{ppg_sleeve(rr):>12.4f}')
#
# # Then the same sweep over restrike_move in (None, 0.005, 0.01, 0.02) at a
# # fixed band, reading the contamination block's hedge-volga-as-%-of-wing
# # figure rather than ppg -- that is the number restrike_move controls.
