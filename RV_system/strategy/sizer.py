"""
sizer.py -- the greek-target sizer (Phase 3).

WHAT THIS REPLACES
------------------
Up to now a strategy was specified in INSTRUMENT space:

    LegRequest('call', -1, 10_000_000, '1M', target_delta=+0.25)

That 10mm is arbitrary. It means a different amount of risk at 1M than at 3M,
and a different amount again in a 5-vol pair than a 12-vol pair. Every number
downstream -- P&L per unit greek, sleeve attribution, cross-pair comparison --
inherits that arbitrariness.

This module inverts the specification. You state the RISK you want:

    GreekTarget(by_tenor={'3M': dict(volga=-2_500, vega=0, vanna=0)})

and `solve()` returns the LegRequests that produce it, choosing among a fixed
menu of quoted strikes and minimising the spread you pay to get there.


THE THREE IDEAS
---------------

1. GREEKS ARE LINEAR IN NOTIONAL, SO SIZING IS A LINEAR SOLVE.
   core/greeks.py builds every GreekVector as `q = notional * direction` times
   a per-unit partial. So if `g_i` is candidate leg i's greek vector at unit
   notional, the book's greek is `sum_i n_i * g_i` for signed notionals n.
   Stack the g_i as columns of A, and hitting a target t is `A n = t`.

2. THE MENU IS STRIKES, NOT OPTIONS.
   By put-call parity a same-strike call and put differ by a forward, and a
   forward has no vol sensitivity. Their vega / vanna / volga are IDENTICAL --
   they are one column of A, not two. So the menu is one leg per strike, taken
   OTM (which is also what quotes tight). `otm_only=False` overrides.

   ATM is the exception: it enters as a STRADDLE (call + put at the ATM-forward
   strike, equal notional), because a single ATM leg is a large naked delta the
   solve has no reason to want. `atm_as_straddle=False` splits it into two
   independent columns.

3. UNDER-DETERMINED IS THE NORMAL CASE, SO THE OBJECTIVE MATTERS.
   Nine strikes per tenor against three or four constraints leaves an infinite
   family of solutions. We pick the one minimising expected SPREAD PAID, using
   the same OptionCostModel the engine charges. That turns "which structure"
   from a taste question into a priced one.


BUCKETS, AND THE TERM-STRUCTURE TRAP
------------------------------------
A flat greek vector sums across expiries, and that summation hides things.
Sell 1M vega, buy 3M vega, size for zero NET vega: your GreekVector reads
vega-neutral, and you are holding a pure term-structure position. Front vol and
back vol do not move together.

This is the FX analogue of a bond book with zero net duration and enormous
curve exposure, and the fix is the same one rates desks use -- bucket the
exposure by maturity instead of summing it:

    GreekTarget(
        by_tenor = {'1W': dict(vega=0), '3M': dict(vega=0)},   # each flat
        net      = {'gamma': +280_000},                        # sum only
    )

RULE OF THUMB: put anything you want FLAT in `by_tenor`. Use `net` only when
you deliberately want one bucket to offset another.


NORMALISED UNITS
----------------
`units='normalised'` (the default) denominates targets in BASE-CCY P&L PER ONE
SIGMA MOVE over `horizon_days`, which is a number you can reason about instead
of an index. The conversion is exactly the Taylor coefficient from
core/greeks.taylor_pnl, with the move set to one sigma:

    u = (sigma * sqrt(dt)) / SPOT_MOVE          spot 1-sigma, in "1%" units
    w = (sigma * nu * sqrt(dt)) / VOL_MOVE      vol  1-sigma, in "1vp" units

    vega  ->  vega_1vp       * w
    volga ->  volga_1vp      * w^2              (second order: move squared)
    vanna ->  vanna_1pct_1vp * u * w
    gamma ->  gamma_1pct     * u^2
    theta ->  theta_1d       * horizon_days

sigma is the ATM vol at that tenor and nu is the SABR vol-of-vol from
`snap.nu_rho`, so the factors are per-tenor and per-pair. That is what makes
"one unit of volga" mean the same amount of risk in 1W USDJPY as in 6M EURCHF.
This is the old stack's trap #6 ("constant notional across tenors is not
constant risk"), which gets materially worse in second-order greeks because
volga picks up the move SQUARED.

`units='raw'` skips all of it and targets the GreekVector fields directly.


THE TWO GUARDS
--------------
Some targets cannot be met honestly by the menu you supplied, and the failure
is silent rather than loud: the solver returns huge offsetting notionals whose
residual difference happens to equal the target. It satisfies the constraints
on paper and the cost line tells you six months later.

Two different diagnostics are needed, because they catch different things.

`require_cond` (default 1e4) tests whether the CONSTRAINT ROWS are
independent. It fires when you ask for two things that are the same thing.

`require_leverage` (default 10) tests whether the COLUMNS are -- gross greek
magnitude deployed divided by net delivered. cond(A) is blind to this: the
constraints can be perfectly independent while the answer is two adjacent
strikes taken in billions of offsetting notional. This is the guard that
actually fires in practice.

THE CANONICAL CASE, AND A CORRECTION WORTH KNOWING. Gamma and vega inside one
expiry are usually described as exactly collinear -- `vega = gamma * S^2 *
sigma * T`, with a strike-independent ratio. In THIS stack that is not quite
true, because greeks are premium-adjusted base-ccy P&L: `d2V/dS2 = G/S -
2*D_pa/S^2`, and the D_pa term varies strongly across strikes. So the columns
are ~10% independent rather than 0%, and a single-tenor gamma-vs-vega target is
technically feasible.

It is just ruinous. Measured on a 3M USDJPY menu, the same target:

    one tenor  : 9.3bn gross notional, $1.69m to trade, leverage 14
    two tenors : 156mm gross notional,   $26k to trade, leverage 1.9

65x. The fix is a second, well-separated tenor. `require_leverage=None` sizes
it anyway if you want to see the number.


WHAT THIS MODULE DOES NOT DO
----------------------------
It does not SEARCH for strikes. You give it a menu (default: the quoted
pillars) and it sizes within it. Automatic strike selection is a discrete
optimisation that would overfit whatever the cost model assumes, and it
properly belongs after Phase 5, when a richness signal can say which parts of
the surface are actually cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.calendar import add_tenor
from core.conventions import fx_calendar
from core.greeks import GreekVector, SPOT_MOVE, VOL_MOVE
from core.option import FXOption, atm_forward_strike
from book.position import LegRequest
from book.costs import OptionCostModel


# --------------------------------------------------------------------- #
# Menu and naming
# --------------------------------------------------------------------- #

# The quoted pillars. These are exactly core.conventions.DELTA_POINTS minus the
# 15d, plus ATM -- i.e. every one of them is a directly quoted RR/BF point, so
# nothing here relies on SABR extrapolation between knots.
DEFAULT_PILLARS: Tuple[Union[str, int], ...] = ('ATM', 35, 25, 10, 5)

# User-facing greek names -> GreekVector field names. Targets are written in
# the short names; everything internal uses the fields.
GREEK_FIELD: Dict[str, str] = {
    'vega':  'vega_1vp',
    'vanna': 'vanna_1pct_1vp',
    'volga': 'volga_1vp',
    'gamma': 'gamma_1pct',
    'theta': 'theta_1d',
    'spot':  'spot_1pct',
}

# Conditioning above which the solve is refused. 1e4 is generous -- a healthy
# multi-tenor solve lands around 5-50, a collinear one blows past 1e5.
DEFAULT_REQUIRE_COND = 1.0e4

# Gross-risk-to-net-risk ratio above which the solve is refused. Measured on
# real structures: a vega-neutral fly runs 2.4, a delta-hedged risk reversal
# 1.7, a front/back gamma calendar 1.9. Single-tenor gamma-vs-vega -- the
# canonical near-cancellation -- runs 14. So 10 sits with comfortable headroom
# above every legitimate structure and below the pathology.
DEFAULT_REQUIRE_LEVERAGE = 10.0

# Legs smaller than this get pruned and the solve re-run without them. A
# 40k tail leg is untradeable and only pollutes the cost line.
DEFAULT_MIN_NOTIONAL = 1_000_000.0


# --------------------------------------------------------------------- #
# The target
# --------------------------------------------------------------------- #
@dataclass
class GreekTarget:
    """
    What risk you want, and in what units.

    by_tenor : {tenor: {greek: value}} -- constrains ONLY that tenor's legs.
               This is where anything you want flat belongs.
    net      : {greek: value}          -- constrains the sum across all tenors.
               Use deliberately; a net-zero can hide a bucket offset.

    Only the greeks you name are constrained. There are three distinct states
    and the difference matters:

        'vega': 0.0     -> actively pinned at zero
        key absent      -> free, whatever the structure implies
        'volga': -2500  -> hit this number

    "Zero" and "don't care" are not the same instruction.

    horizon_days : CALENDAR days over which the one-sigma move is measured.
                   7 = one week, which pairs with a weekly roll.
    units        : 'normalised' (base-ccy P&L per 1-sigma move -- comparable
                   across pairs and tenors) or 'raw' (GreekVector fields
                   directly).
    """
    by_tenor:     Dict[str, Dict[str, float]] = field(default_factory=dict)
    net:          Dict[str, float]            = field(default_factory=dict)
    horizon_days: float                       = 7.0
    units:        str                         = 'normalised'

    def __post_init__(self):
        if self.units not in ('normalised', 'raw'):
            raise ValueError("units must be 'normalised' or 'raw'")
        for name in self._all_greek_names():
            if name not in GREEK_FIELD:
                raise ValueError(
                    f"unknown greek '{name}'. Known: {sorted(GREEK_FIELD)}")
        if not self.by_tenor and not self.net:
            raise ValueError("empty target -- nothing to solve for")

    def _all_greek_names(self):
        for d in self.by_tenor.values():
            yield from d.keys()
        yield from self.net.keys()

    def tenors_mentioned(self) -> List[str]:
        return list(self.by_tenor.keys())

    def rows(self) -> List[Tuple[Optional[str], str, float]]:
        """Flatten to [(bucket_or_None, greek_name, value), ...]."""
        out: List[Tuple[Optional[str], str, float]] = []
        for tenor, d in self.by_tenor.items():
            for g, v in d.items():
                out.append((tenor, g, float(v)))
        for g, v in self.net.items():
            out.append((None, g, float(v)))
        return out


# --------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------- #
@dataclass
class _SubLeg:
    """One physical option inside a candidate unit."""
    option_type:  str
    target_delta: float      # signed; NaN when atm
    atm:          bool
    weight:       float      # notional multiplier within the unit
    K:            float
    sigma:        float
    tag:          str


@dataclass
class Candidate:
    """
    One column of A: a unit structure at unit notional.

    Most candidates are a single option. ATM is a straddle (two sub-legs at
    equal notional) unless atm_as_straddle=False. `greek` is the summed
    GreekVector at notional=1, direction=+1, so a signed solve variable n_i
    scales it directly.
    """
    label:      str
    tenor:      str
    expiry:     date
    pillar:     Union[str, int]
    sub_legs:   List[_SubLeg]
    greek:      GreekVector
    cost_unit:  float          # base-ccy spread cost per unit of notional

    def to_leg_requests(self, n: float, sleeve: str,
                        tag_prefix: str = '') -> List[LegRequest]:
        """Turn a signed solved notional into LegRequests."""
        direction = 1 if n >= 0 else -1
        mag = abs(n)
        out = []
        for sl in self.sub_legs:
            notional = mag * sl.weight
            if notional <= 0:
                continue
            out.append(LegRequest(
                option_type  = sl.option_type,
                direction    = direction,
                notional     = notional,
                tenor        = self.tenor,
                target_delta = sl.target_delta,
                sleeve       = sleeve,
                atm          = sl.atm,
                tag          = f"{tag_prefix}{sl.tag}",
            ))
        return out


# --------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------- #
def sigma_scales(snap, t_days: float, horizon_days: float) -> Dict[str, float]:
    """
    Per-greek multipliers converting GreekVector fields into base-ccy P&L for a
    one-sigma move over `horizon_days`.

    These ARE the Taylor coefficients from core.greeks.taylor_pnl evaluated at a
    one-sigma move, which is what makes the normalised numbers interpretable:
    "volga = -2500" means "this book loses $2,500 to the volga term when vol
    moves one sigma".

    nu is the SABR vol-of-vol from snap.nu_rho -- see the PHASE 5 WARNING in
    market/snapshot.py before promoting it from a scale factor to a signal.
    """
    dt    = max(float(horizon_days), 1e-9) / 365.0
    sigma = float(snap.atm_vol(t_days))
    nu, _ = snap.nu_rho(t_days)
    nu    = float(nu)

    u = (sigma * np.sqrt(dt)) / SPOT_MOVE            # spot 1-sigma, "1%" units
    w = (sigma * nu * np.sqrt(dt)) / VOL_MOVE        # vol  1-sigma, "1vp" units

    return {
        'vega':  w,
        'volga': w * w,
        'vanna': u * w,
        'gamma': u * u,
        'theta': float(horizon_days),
        'spot':  u,
    }


# --------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------- #
@dataclass
class SizerResult:
    """
    What the sizer returns. Deliberately more than just notionals -- `residual`
    and `condition` are the honesty checks, and `cost` is what makes two
    candidate structures comparable.
    """
    pair:        str
    as_of:       date
    legs:        List[LegRequest]
    notionals:   Dict[str, float]              # label -> signed notional
    expiries:    Dict[str, date]               # tenor -> expiry
    realised:    GreekVector                   # THIS TRADE, raw units
    realised_by_tenor: Dict[str, GreekVector]  # this trade, per bucket
    carried:     Dict[str, GreekVector]        # already on risk, per bucket
    book_after:  GreekVector                   # carried + this trade
    bucket_map:  List[str]                     # how aged positions were bucketed
    target_rows: List[Tuple[Optional[str], str, float]]
    achieved:    List[float]                   # BOOK after trade, same order
    residual:    List[float]
    condition:   float
    method:      str                           # 'lp' or 'lstsq' fallback
    leverage:    float                         # gross risk / net risk
    cost:        float                         # base-ccy entry spread
    gross_notional: float
    units:       str
    horizon_days: float
    scales:      Dict[str, Dict[str, float]]   # tenor -> greek -> multiplier
    dropped:     List[str]                     # candidates pruned or unbuildable
    candidates:  List[Candidate] = field(default_factory=list, repr=False)

    # ----------------------------------------------------------------- #
    def open_into(self, book, snap, cost_model=None) -> list:
        """
        Open every leg with the expiry the sizer actually priced.

        Use this rather than opening the legs yourself -- it is the one place
        the tenor->expiry resolution is guaranteed to match the greeks that
        were solved on.
        """
        out = []
        for leg in self.legs:
            out.append(book.open(leg, snap, self.expiries[leg.tenor],
                                 cost_model=cost_model))
        return out

    def greek(self, name: str, tenor: Optional[str] = None) -> float:
        """Realised value of one greek, raw units, optionally in one bucket."""
        gv = self.realised if tenor is None else self.realised_by_tenor[tenor]
        return getattr(gv, GREEK_FIELD[name])

    def greek_normalised(self, name: str, tenor: str) -> float:
        """Realised value of one greek in that tenor, in 1-sigma units."""
        return self.greek(name, tenor) * self.scales[tenor][name]

    # ----------------------------------------------------------------- #
    def __str__(self) -> str:
        L = []
        L.append(f"SIZER  {self.pair}  {self.as_of}  "
                 f"tenors={sorted(self.expiries)}  units={self.units}"
                 + (f"  horizon={self.horizon_days:g}d"
                    if self.units == 'normalised' else ''))
        L.append('-' * 78)
        L.append(f"  {'leg':<16}{'K':>10}{'vol':>8}{'dir':>6}"
                 f"{'notional':>16}{'cost':>12}")
        for c in self.candidates:
            n = self.notionals.get(c.label, 0.0)
            if abs(n) < 1e-6:
                continue
            d = '+1' if n >= 0 else '-1'
            for i, sl in enumerate(c.sub_legs):
                name = c.label if len(c.sub_legs) == 1 else f"{c.label}.{sl.tag}"
                L.append(f"  {name:<16}{sl.K:>10.4f}{sl.sigma * 100:>8.2f}"
                         f"{d:>6}{abs(n) * sl.weight:>16,.0f}"
                         + (f"{abs(n) * c.cost_unit:>12,.0f}" if i == 0 else f"{'':>12}"))
        L.append('-' * 78)
        any_carried = any(abs(gv.vega_1vp) + abs(gv.volga_1vp) > 1e-9
                          for gv in self.carried.values())
        if any_carried:
            L.append(f"  {'target':<22}{'asked':>13}{'carried':>13}"
                     f"{'traded':>13}{'book after':>13}{'residual':>12}")
        else:
            L.append(f"  {'target':<22}{'asked':>14}{'realised':>14}"
                     f"{'residual':>14}")
        for i, ((bucket, g, want), got, res) in enumerate(
                zip(self.target_rows, self.achieved, self.residual)):
            nm = f"{g}@{bucket}" if bucket else f"{g}@net"
            if any_carried:
                keys = (self.carried.keys() if bucket is None else [bucket])
                cur = sum(getattr(self.carried[t], GREEK_FIELD[g])
                          * self.scales[t][g] for t in keys)
                L.append(f"  {nm:<22}{want:>13,.1f}{cur:>13,.1f}"
                         f"{got - cur:>13,.1f}{got:>13,.1f}{res:>12,.1e}")
            else:
                L.append(f"  {nm:<22}{want:>14,.1f}{got:>14,.1f}{res:>14,.2e}")
        L.append('-' * 78)
        L.append(f"  cond(A) {self.condition:>8,.1f}"
                 f"   leverage {self.leverage:>8,.1f}"
                 f"   gross {self.gross_notional:>16,.0f}"
                 f"   cost {self.cost:>12,.0f}"
                 f"   legs {len(self.legs):>3}"
                 + ('' if self.method == 'lp' else '   [lstsq fallback]'))
        if not np.isnan(self.realised.delta_hedge):
            L.append(f"  spot delta hedge {self.realised.delta_hedge:>14,.0f} "
                     f"({self.pair[:3]})")
        if self.bucket_map:
            L.append(f"  carried: {'; '.join(self.bucket_map[:4])}"
                     + ('' if len(self.bucket_map) <= 4
                        else f" (+{len(self.bucket_map) - 4} more)"))
        if self.dropped:
            head = self.dropped[:3]
            more = ('' if len(self.dropped) <= 3
                    else f" (+{len(self.dropped) - 3} more)")
            L.append(f"  dropped: {'; '.join(head)}{more}")
        return '\n'.join(L)


class SizerError(RuntimeError):
    """Raised when the menu cannot span the target, or the target is empty."""


# --------------------------------------------------------------------- #
# Building the menu
# --------------------------------------------------------------------- #
def build_candidates(snap,
                     tenors:          Sequence[str],
                     pillars:         Sequence[Union[str, int]] = DEFAULT_PILLARS,
                     otm_only:        bool = True,
                     atm_as_straddle: bool = True,
                     cost_model:      Optional[OptionCostModel] = None
                     ) -> Tuple[List[Candidate], Dict[str, date], List[str]]:
    """
    Price every (tenor, pillar) on the menu and return them as columns.

    Strikes are resolved through `snap.solve_strike_and_vol` -- the SAME fixed
    point `open_position` uses -- so the greeks solved on are the greeks you
    will be filled at, not an approximation of them.

    A pillar that cannot be struck (brentq's [0.85S, 1.20S] bracket does not
    reach a 5-delta wing in a high-vol pair) is dropped and named, rather than
    silently degrading the menu.
    """
    cm  = cost_model if cost_model is not None else OptionCostModel()
    fxc = fx_calendar(snap.pair)

    cands: List[Candidate] = []
    expiries: Dict[str, date] = {}
    dropped: List[str] = []

    for tenor in tenors:
        expiry = add_tenor(snap.date, tenor, fxc)
        expiries[tenor] = expiry
        t_days = max((expiry - snap.date).days, 1e-6)
        r_d, r_f = snap.rates(t_days)
        S = snap.spot

        for pillar in pillars:
            if str(pillar).upper() == 'ATM':
                K = atm_forward_strike(S, r_d, r_f, expiry, snap.date,
                                       pair=snap.pair)
                sigma = snap.smile_vol(K, t_days)
                groups = ([('call', 'put')] if atm_as_straddle
                          else [('call',), ('put',)])
                for grp in groups:
                    label = (f"{tenor} ATM straddle" if atm_as_straddle
                             else f"{tenor} ATM {grp[0]}")
                    subs, gv, cost = [], GreekVector.zero(S), 0.0
                    for ot in grp:
                        opt = FXOption(pair=snap.pair, K=K, expiry=expiry,
                                       option_type=ot)
                        gv = gv + opt.greeks(S, sigma, r_d, r_f, snap.date,
                                             notional=1.0, direction=+1)
                        cost += cm.charge(opt, 1.0, snap, sigma=sigma,
                                          abs_delta=0.5, reason='size')['cost']
                        subs.append(_SubLeg(ot, np.nan, True, 1.0, K, sigma,
                                            f"atm_{ot[0]}"))
                    cands.append(Candidate(label, tenor, expiry, 'ATM',
                                           subs, gv, cost))
                continue

            d = float(pillar) / 100.0
            # OTM convention: put below the forward, call above. Their vol
            # greeks are identical by put-call parity, so this loses nothing
            # and picks the side that quotes tight.
            sides = [('put', -d, f"p{int(pillar)}"),
                     ('call', +d, f"c{int(pillar)}")]
            if not otm_only:
                sides += [('call', -d, f"c{int(pillar)}itm"),
                          ('put', +d, f"p{int(pillar)}itm")]

            for ot, tgt, tag in sides:
                label = f"{tenor} {int(pillar)}d {ot}"
                try:
                    K, sigma = snap.solve_strike_and_vol(tgt, ot, expiry)
                except Exception as exc:                      # brentq bracket
                    dropped.append(f"{label} (unstrikeable: "
                                   f"{type(exc).__name__})")
                    continue
                opt = FXOption(pair=snap.pair, K=K, expiry=expiry,
                               option_type=ot)
                gv = opt.greeks(S, sigma, r_d, r_f, snap.date,
                                notional=1.0, direction=+1)
                cost = cm.charge(opt, 1.0, snap, sigma=sigma,
                                 abs_delta=abs(tgt), reason='size')['cost']
                cands.append(Candidate(label, tenor, expiry, int(pillar),
                                       [_SubLeg(ot, tgt, False, 1.0, K, sigma, tag)],
                                       gv, cost))

    return cands, expiries, dropped


# --------------------------------------------------------------------- #
# The solve
# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #
# What the book is already carrying
# --------------------------------------------------------------------- #
def bucket_book(book, snap, expiries: Dict[str, date],
                sleeve: Optional[str] = None
                ) -> Tuple[Dict[str, GreekVector], List[str]]:
    """
    Bucket a pair's open positions onto the menu tenors, so an incremental
    solve knows what it is topping up.

    THE JUDGEMENT CALL IN HERE. Aged positions do not sit on your menu: a 3M
    struck five days ago is a 2M-and-change today, and it has to count against
    SOME bucket or the incremental target is wrong. The rule used is nearest
    bucket in LOG remaining-days -- log because tenor is multiplicative, so a
    45-day position is much nearer 3M(91d) than 1W(7d) even though it is
    numerically closer to 7 than to 91 on a linear scale would suggest the
    opposite for shorter cases.

    The mapping is returned as text alongside the numbers so it appears in the
    result rather than being assumed. If you dislike the rule, this is the one
    function to change; nothing downstream depends on how the assignment was
    made.

    Greeks are recomputed off `snap` via `Position.greeks`, NOT read from
    `book.marks`, so they are on the same surface read as the candidate legs.
    Using cached marks here would silently mix two dates.
    """
    out = {t: GreekVector.zero(snap.spot) for t in expiries}
    notes: List[str] = []
    if book is None:
        return out, notes

    menu_days = {t: max((e - snap.date).days, 1) for t, e in expiries.items()}
    for pos in book.open_positions(pair=snap.pair, sleeve=sleeve):
        rem = max((pos.expiry - snap.date).days, 1)
        tenor = min(menu_days, key=lambda t: abs(np.log(rem / menu_days[t])))
        out[tenor] = out[tenor] + pos.greeks(snap)
        notes.append(f"pos {pos.pos_id} ({pos.tenor_label}, {rem}d left) "
                     f"-> {tenor} bucket")
    return out, notes


def _current_buckets(current, snap, expiries: Dict[str, date],
                     sleeve: Optional[str]
                     ) -> Tuple[Dict[str, GreekVector], List[str]]:
    """Normalise the `current=` argument into per-bucket GreekVectors."""
    zero = {t: GreekVector.zero(snap.spot) for t in expiries}
    if current is None:
        return zero, []
    if isinstance(current, GreekVector):
        if len(expiries) != 1:
            raise SizerError(
                "current= was given as a bare GreekVector but the menu has "
                f"{len(expiries)} tenors, so there is no way to know which "
                "bucket it belongs to. Pass the Book instead, or size one "
                "tenor at a time.")
        t = next(iter(expiries))
        return {t: current}, [f"supplied GreekVector -> {t} bucket"]
    if isinstance(current, dict):
        out = dict(zero)
        for t, gv in current.items():
            if t not in out:
                raise SizerError(f"current= has a '{t}' entry but '{t}' is "
                                 f"not on the menu {sorted(expiries)}")
            out[t] = gv
        return out, [f"supplied dict over {sorted(current)}"]
    return bucket_book(current, snap, expiries, sleeve=sleeve)   # a Book


def _assemble(cands: List[Candidate],
              target: GreekTarget,
              scales: Dict[str, Dict[str, float]]
              ) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """
    Build the constraint matrix A (rows = constraints, cols = candidates) and
    the rhs t, both in the target's units.
    """
    rows = target.rows()
    A = np.zeros((len(rows), len(cands)))
    t = np.zeros(len(rows))

    for r, (bucket, gname, value) in enumerate(rows):
        t[r] = value
        for j, c in enumerate(cands):
            if bucket is not None and c.tenor != bucket:
                continue                                  # bucket row: this
                                                          # tenor's legs only
            raw = getattr(c.greek, GREEK_FIELD[gname])
            A[r, j] = raw * scales[c.tenor][gname]
    return A, t, rows


def _row_normalise(A: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale each constraint row to unit length.

    Without this a gamma row (~1e5) and a volga row (~1e3) are implicitly
    weighted by three orders of magnitude of unit accident rather than by
    anything you chose, and cond(A) reports that accident instead of real
    collinearity. For the under-determined case (the normal one) the solution
    is unchanged; when over-determined it weights every constraint equally,
    which is the sane default.
    """
    s = np.linalg.norm(A, axis=1)
    s[s == 0] = 1.0
    return A / s[:, None], t / s


def intrinsic_condition(A: np.ndarray) -> float:
    """
    Conditioning of the constraint system with BOTH rows and columns
    normalised, so it measures genuine collinearity between constraints rather
    than a units mismatch or the cost weighting. This is the number the guard
    tests.
    """
    if A.size == 0:
        return np.inf
    An, _ = _row_normalise(A, np.zeros(A.shape[0]))
    cn = np.linalg.norm(An, axis=0)
    cn[cn == 0] = 1.0
    M = An / cn[None, :]
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 0]
    return float(sv[0] / sv[-1]) if len(sv) else np.inf


def _min_cost_solve(A: np.ndarray, t: np.ndarray,
                    cost: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """
    Minimise SPREAD PAID, sum_i cost_i * |n_i|, subject to A n = t.

    WHY L1, AND WHY AN LP
    ---------------------
    The objective is literally the money crossing the spread, so it is an L1
    norm, and L1 is also what makes the answer TRADEABLE: a linear program's
    basic solution has at most (number of constraints) non-zero variables. Three
    constraints gives at most three legs. Sparsity is not a post-processing step
    here, it is a property of the optimum.

    Two earlier attempts, and why they failed -- both worth knowing about
    because both look reasonable:

      * Cost-weighted L2 (one lstsq call). Optimises the wrong thing, and not
        merely imprecisely: L2 actively PREFERS to spread notional across many
        columns, because splitting a position in two lowers the sum of squares
        while leaving the sum of absolutes unchanged. It returned an eight-leg
        smear across every strike on the menu.

      * IRLS towards L1 (reweight by 1/sqrt|n|, iterate). Right objective,
        but it stalls. On the gamma calendar it locked onto a near-duplicate
        35-delta put/call pair -- effectively a synthetic forward -- at 3.4bn
        a side, and reweighting could not climb back out. A local trap.

    The LP has neither problem: it is convex, solved exactly, and cheap at this
    size (a few dozen columns).

    Formulation: split n into positive and negative parts, n = p - m with
    p, m >= 0. Then |n_i| = p_i + m_i and the whole thing is linear:

        minimise   sum_i cost_i * (p_i + m_i)
        subject to A(p - m) = t,   p, m >= 0

    Rows are normalised first (see _row_normalise).

    Returns (n, cond, method). `method` is 'lp' when the LP solved and
    'lstsq' when it was infeasible and we fell back to minimum-norm least
    squares -- which happens when the target genuinely cannot be met, and the
    caller reports the residual.
    """
    An, tn = _row_normalise(A, t)
    m_rows, n_cols = An.shape
    c = np.maximum(np.asarray(cost, dtype=float), 1e-12)

    try:
        from scipy.optimize import linprog
        A_eq = np.hstack([An, -An])
        res = linprog(c=np.concatenate([c, c]), A_eq=A_eq, b_eq=tn,
                      bounds=[(0, None)] * (2 * n_cols), method='highs')
        if res.success:
            x = res.x
            return x[:n_cols] - x[n_cols:], intrinsic_condition(A), 'lp'
    except Exception:
        pass

    # Infeasible or scipy unavailable: minimum-(cost-weighted)-norm least
    # squares. Sparsity is lost, so the caller's pruning does the work.
    B = An / c[None, :]
    y, *_ = np.linalg.lstsq(B, tn, rcond=None)
    return y / c, intrinsic_condition(A), 'lstsq'


def leverage(A: np.ndarray, t: np.ndarray, n: np.ndarray) -> float:
    """
    Gross risk deployed divided by net risk delivered.

    THE FAILURE MODE THIS CATCHES, which conditioning does not.

    cond(A) measures whether the CONSTRAINTS are independent. It says nothing
    about whether the COLUMNS are, and the expensive pathology lives there: two
    adjacent strikes have nearly identical greek signatures, so the solver can
    hit any target by taking a vast position in one and a vast offsetting
    position in the other and keeping the residual. The constraint matrix is
    perfectly well conditioned while the answer is billions of notional held to
    net out a few thousand of risk.

    So measure it directly: how much greek magnitude is being deployed against
    how much is being delivered.

        leverage = sum_j |n_j| * ||A[:,j]||  /  ||t||

    A clean structure runs 2-6 -- a vega-neutral fly deploys wing vega and body
    vega of the same order to net them, so a small multiple is normal and
    healthy. Tens or hundreds means the answer is a near-cancellation and the
    spread bill will say so.
    """
    An, tn = _row_normalise(A, t)
    denom = float(np.linalg.norm(tn))
    if denom <= 0:
        return np.inf if np.any(np.abs(n) > 0) else 0.0
    gross = float(np.sum(np.abs(n) * np.linalg.norm(An, axis=0)))
    return gross / denom


def _collinear_pairs(A: np.ndarray, rows, thresh: float = 0.999) -> List[str]:
    """Name the constraint rows that are nearly parallel, for the error text."""
    out = []
    norms = np.linalg.norm(A, axis=1)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if norms[i] == 0 or norms[j] == 0:
                continue
            c = abs(float(A[i] @ A[j]) / (norms[i] * norms[j]))
            if c > thresh:
                ni = f"{rows[i][1]}@{rows[i][0] or 'net'}"
                nj = f"{rows[j][1]}@{rows[j][0] or 'net'}"
                out.append(f"'{ni}' and '{nj}' (|cos| = {c:.5f})")
    return out


def solve(snap,
          tenors:          Union[str, Sequence[str]],
          target:          GreekTarget,
          current=None,
          sleeve:          str = 'unclassified',
          pillars:         Sequence[Union[str, int]] = DEFAULT_PILLARS,
          allow_deltas:    Optional[Sequence[Union[str, int]]] = None,
          otm_only:        bool = True,
          atm_as_straddle: bool = True,
          cost_model:      Optional[OptionCostModel] = None,
          min_notional:    float = DEFAULT_MIN_NOTIONAL,
          max_legs:        Optional[int] = None,
          require_cond:    Optional[float] = DEFAULT_REQUIRE_COND,
          require_leverage: Optional[float] = DEFAULT_REQUIRE_LEVERAGE,
          tag_prefix:      str = '') -> SizerResult:
    """
    Solve for the notionals that hit `target`, at minimum spread paid.

    Parameters
    ----------
    snap         : MarketSnapshot. Everything -- strikes, vols, nu, costs -- is
                   read at or before its as_of, so the sizer inherits the same
                   look-ahead discipline as the rest of the stack.
    tenors       : one tenor or several. Supplying several WIDENS the search;
                   it does not force a calendar. If a single-tenor answer is
                   cheapest the other tenor comes back at zero notional.
    target       : GreekTarget. Read as a BOOK-level statement.
    current      : what is already on risk, so the solve trades only the
                   INCREMENT needed to reach the target. Accepts a Book (the
                   normal case -- its open positions in this pair are bucketed
                   onto the menu by remaining days), a bare GreekVector
                   (single-tenor menus only), or a {tenor: GreekVector} dict.
                   None means "assume a flat book", which is what you want on
                   the first trade and wrong on every subsequent one.
    allow_deltas : shorthand for `pillars`, e.g. [10, 'ATM'] to force the
                   structure into the 10-delta wings for a comparison.
    min_notional : legs below this are pruned one at a time and the solve
                   re-run, which recovers the sparsity the L2 objective does
                   not give directly.
    max_legs     : cap the number of candidate units used. Never prunes below
                   the number of constraints.
    require_cond : refuse the solve above this conditioning. None to override.
    require_leverage : refuse the solve above this gross-risk-to-net-risk
                   ratio. This is the guard that catches near-duplicate
                   COLUMNS (two adjacent strikes taken in vast offsetting
                   size), which conditioning cannot see. None to override.

    Returns
    -------
    SizerResult -- print it.
    """
    if isinstance(tenors, str):
        tenors = [tenors]
    tenors = list(dict.fromkeys(tenors))          # de-dup, keep order

    missing = [t for t in target.tenors_mentioned() if t not in tenors]
    if missing:
        raise SizerError(
            f"target has by_tenor entries for {missing} but tenors={tenors}. "
            f"A bucket constraint on a tenor with no legs can never be met.")

    if allow_deltas is not None:
        pillars = allow_deltas

    cands, expiries, dropped = build_candidates(
        snap, tenors, pillars=pillars, otm_only=otm_only,
        atm_as_straddle=atm_as_straddle, cost_model=cost_model)

    if not cands:
        raise SizerError("no candidate legs could be struck on this menu")

    # Per-tenor normalisation factors.
    scales: Dict[str, Dict[str, float]] = {}
    for tenor, expiry in expiries.items():
        t_days = max((expiry - snap.date).days, 1e-6)
        if target.units == 'raw':
            scales[tenor] = {g: 1.0 for g in GREEK_FIELD}
        else:
            scales[tenor] = sigma_scales(snap, t_days, target.horizon_days)

    A_full, t_book, rows = _assemble(cands, target, scales)
    n_rows = len(rows)

    # ---- what the book already carries ------------------------------- #
    # The target is a BOOK-level statement, so what this trade has to deliver
    # is the target MINUS what is already on risk. With current=None that is
    # the target itself and nothing changes; with current=book the sizer stops
    # being a trade generator and becomes a book-level controller. That is the
    # same mechanism Phase 4's vega re-hedge needs.
    cur_buckets, bucket_map = _current_buckets(current, snap, expiries, sleeve)
    carried = np.zeros(n_rows)
    for r, (bucket, gname, _) in enumerate(rows):
        keys = expiries.keys() if bucket is None else [bucket]
        carried[r] = sum(getattr(cur_buckets[t], GREEK_FIELD[gname])
                         * scales[t][gname] for t in keys)
    t_vec = t_book - carried

    if len(cands) < n_rows:
        raise SizerError(
            f"{n_rows} constraints but only {len(cands)} candidate legs. "
            f"Widen the menu or drop a constraint.")

    # ---- solve, prune, re-solve -------------------------------------- #
    active = list(range(len(cands)))
    n_active = np.zeros(len(active))
    cond, method = np.inf, 'lp'

    for _ in range(len(cands) + 1):
        A = A_full[:, active]
        cost = np.array([cands[j].cost_unit for j in active])
        n_active, cond, method = _min_cost_solve(A, t_vec, cost)

        if len(active) <= n_rows:
            break

        # prune the smallest offending leg, one per pass
        mags = np.array([abs(n_active[k]) * max(
            sl.weight for sl in cands[active[k]].sub_legs)
            for k in range(len(active))])
        over = (max_legs is not None
                and int(np.sum(mags >= min_notional)) > max_legs)
        small = np.where(mags < min_notional)[0]

        if len(small):
            k = int(small[np.argmin(mags[small])])
        elif over:
            k = int(np.argmin(mags))
        else:
            break

        dropped.append(f"{cands[active[k]].label} "
                       f"({mags[k]:,.0f} < min_notional)")
        active.pop(k)

    # ---- guards -------------------------------------------------------- #
    lev = leverage(A_full[:, active], t_vec, n_active)

    if require_leverage is not None and lev > require_leverage:
        sized = ", ".join(f"{cands[j].label} {n_active[k]:,.0f}"
                          for k, j in enumerate(active)
                          if abs(n_active[k]) > 1e-6)
        raise SizerError("\n".join([
            f"near-cancelling solve -- leverage = {lev:,.1f} "
            f"(limit {require_leverage:,.1f}).",
            "  The target is being met by large offsetting positions in legs "
            "with nearly identical",
            "  greek signatures, not by legs that genuinely span it. "
            f"cond(A) = {cond:,.1f} looks fine",
            "  because the CONSTRAINTS are independent; it is the COLUMNS "
            "that are near-duplicates.",
            f"  Solved notionals: {sized}",
            "  Fix: separate the tenors further, raise min_notional, or drop "
            "a constraint.",
            "  Override with require_leverage=None to size it anyway.",
        ]))

    if require_cond is not None and cond > require_cond:
        pairs = _collinear_pairs(A_full[:, active], rows)
        hint = ("\n  Near-parallel constraints:\n    "
                + "\n    ".join(pairs)) if pairs else ""
        one_tenor = len(tenors) == 1
        fix = ("\n  Fix: add a second, well-separated tenor, e.g. "
               "tenors=['1W','3M'].\n"
               "       Within one expiry vega ~ gamma * S^2 * sigma * T, and "
               "that ratio is\n"
               "       strike-independent, so no choice of strikes separates "
               "them."
               if one_tenor else
               "\n  Fix: separate the tenors further, or drop a constraint.")
        raise SizerError(
            f"ill-conditioned solve -- cond(A) = {cond:.3g} "
            f"(limit {require_cond:.3g})."
            f"{hint}{fix}\n"
            f"  Solved notionals would be: "
            + ", ".join(f"{cands[j].label} {n_active[k]:,.0f}"
                        for k, j in enumerate(active))
            + "\n  Override with require_cond=None to solve anyway.")

    # ---- assemble the answer ----------------------------------------- #
    notionals: Dict[str, float] = {}
    legs: List[LegRequest] = []
    realised = GreekVector.zero(snap.spot)
    by_tenor: Dict[str, GreekVector] = {t: GreekVector.zero(snap.spot)
                                        for t in tenors}
    total_cost = 0.0
    used: List[Candidate] = []

    for k, j in enumerate(active):
        c, n = cands[j], float(n_active[k])
        notionals[c.label] = n
        if abs(n) < 1e-6:
            continue
        used.append(c)
        legs.extend(c.to_leg_requests(n, sleeve, tag_prefix))
        realised = realised + (c.greek * n)
        by_tenor[c.tenor] = by_tenor[c.tenor] + (c.greek * n)
        total_cost += abs(n) * c.cost_unit

    # `achieved` is the BOOK after this trade, not the trade on its own --
    # the target was a book-level statement, so the residual has to be too.
    traded   = A_full[:, active] @ n_active
    achieved = list(carried + traded)
    residual = [a - w for a, (_, _, w) in zip(achieved, rows)]

    book_after = realised
    for gv in cur_buckets.values():
        book_after = book_after + gv

    return SizerResult(
        pair              = snap.pair,
        as_of             = snap.date,
        legs              = legs,
        notionals         = notionals,
        expiries          = expiries,
        realised          = realised,
        realised_by_tenor = by_tenor,
        carried           = cur_buckets,
        book_after        = book_after,
        bucket_map        = bucket_map,
        target_rows       = rows,
        achieved          = achieved,
        residual          = residual,
        condition         = cond,
        method            = method,
        leverage          = lev,
        cost              = total_cost,
        gross_notional    = sum(l.notional for l in legs),
        units             = target.units,
        horizon_days      = target.horizon_days,
        scales            = scales,
        dropped           = dropped,
        candidates        = used,
    )


# --------------------------------------------------------------------- #
# Structure builders -- presets over solve()
# --------------------------------------------------------------------- #
def vega_neutral_fly(snap, tenor: str, volga: float,
                     sleeve: str = 'convexity', **kw) -> SizerResult:
    """
    The convexity atom. Volga to target, vega and vanna pinned at zero.

    Negative `volga` = short convexity = sell the wings, buy the body, which is
    the side this strategy is trying to harvest.
    """
    tkw = _target_kw(kw)
    tgt = GreekTarget(by_tenor={tenor: dict(volga=volga, vega=0.0, vanna=0.0)},
                      **tkw)
    return solve(snap, [tenor], tgt, sleeve=sleeve, **kw)


def risk_reversal(snap, tenor: str, vanna: float,
                  sleeve: str = 'skew', **kw) -> SizerResult:
    """
    The skew atom. Vanna to target, vega and volga pinned at zero.

    Note the solver will generally NOT return a clean two-leg risk reversal:
    the two wings are not exactly volga-symmetric, so it bolts on a small ATM
    straddle to null the residual convexity. Without that leg a "pure vanna"
    sleeve is quietly part volga, which defeats the point of separating them.
    """
    tkw = _target_kw(kw)
    tgt = GreekTarget(by_tenor={tenor: dict(vanna=vanna, vega=0.0, volga=0.0)},
                      **tkw)
    return solve(snap, [tenor], tgt, sleeve=sleeve, **kw)


def vega_sleeve(snap, tenor: str, vega: float,
                sleeve: str = 'atm', **kw) -> SizerResult:
    """Plain long/short vol: vega to target, vanna and volga flat."""
    tkw = _target_kw(kw)
    tgt = GreekTarget(by_tenor={tenor: dict(vega=vega, vanna=0.0, volga=0.0)},
                      **tkw)
    return solve(snap, [tenor], tgt, sleeve=sleeve, **kw)


def gamma_sleeve(snap, front: str, back: str, gamma: float,
                 sleeve: str = 'gamma', **kw) -> SizerResult:
    """
    Long (or short) gamma with NET vega flat, out of a front/back calendar.

    READ THIS BEFORE USING IT -- the obvious specification does not work.

    You might write `by_tenor={front: vega 0, back: vega 0}` so each bucket is
    flat on its own. That target is INFEASIBLE in any useful sense, and the
    reason is the collinearity from the module docstring: inside one expiry
    vega ~ gamma * S^2 * sigma * T with a strike-independent ratio, so a bucket
    with zero vega has essentially zero gamma too. Asking for both forces the
    solver into a near-cancellation -- two adjacent strikes in billions of
    offsetting notional, exploiting the tiny premium-adjustment term that keeps
    the columns from being exactly parallel. The leverage guard catches it and
    refuses; that refusal is correct, and the target is what is wrong.

    So gamma and vega separate ACROSS tenors, not within them:

        net = {'gamma': target, 'vega': 0}

    Long the front, short the back, net vega zero. Which means you are
    deliberately running front-vs-back vega -- the term-structure exposure the
    module docstring warns about is not a bug here, it is the unavoidable price
    of holding gamma without vega. Look at `realised_by_tenor` to see how much
    of it you took on.
    """
    tkw = _target_kw(kw)
    tgt = GreekTarget(net={'gamma': gamma, 'vega': 0.0}, **tkw)
    return solve(snap, [front, back], tgt, sleeve=sleeve, **kw)


def _target_kw(kw: dict) -> dict:
    """Pull GreekTarget's own kwargs out of a builder's **kw."""
    return {k: kw.pop(k) for k in ('horizon_days', 'units') if k in kw}
