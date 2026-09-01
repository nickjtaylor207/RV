# Systematic_RV — Project State

**Last updated:** Phase 3 built — `strategy/sizer.py` (the greek-target sizer)
and `strategy/roller.py` (maintaining a greek target across rolls). See §3.10
and §3.11.

**Two gates are outstanding and both are unrun on real data:** Phase 2b's
break-even study, and Phase 3's own drift gate. Phase 3 was built ahead of
Phase 2b's gate deliberately — the sizer's machinery does not depend on the
answer, only the choice of strike menu does — but nothing downstream should be
treated as validated until both have run. See Part II.

Prior state: Phase 2 complete (cost model built and wired), plus a
pricing/rates hardening pass — the USD leg reads a real SOFR OIS curve, and the
pricer runs on the option's three actual time clocks rather than one. See §3.2
and §3.3.

This document has two parts:

- **PART I — WHAT HAS BEEN BUILT.** Enough detail that someone who has never
  seen this codebase can understand what exists, why each piece exists, and
  what decisions are already locked in.
- **PART II — WHAT COMES NEXT.** The remaining phases, each with a concrete
  deliverable, an acceptance gate, and the decisions that are still open.

---
---

# PART I — WHAT HAS BEEN BUILT

## 0. Why this project exists

The predecessor, `../Delta_Hedged`, is a working FX delta-hedged options
backtester. It can price a multi-leg European FX structure, walk it forward
day by day, delta-hedge it, and attribute the P&L into greek buckets. It has a
signal layer, a regime-gate layer, and a grid sweeper on top.

The strategy goal changed. The intent is now a **systematic FX volatility
relative-value strategy** whose edge is the risk premium embedded in the
**wings** (convexity / vol-of-vol) and the **skew** of FX vol surfaces, sized
and risk-managed in **greek space** rather than notional space.

The old architecture could not express that, for one structural reason
described in §1. Rather than bolt onto it, `Systematic_RV` is a rewrite that
**reuses the expensive, correct parts verbatim** (pricer, SABR surface,
calendar, Bloomberg layer) and replaces the parts that were shaped by the old
loop ordering.

### What the strategy is actually trying to harvest

This is the intellectual core, and it determines everything downstream.

| Premium | Instrument atom | Greek you are short | How it is measured here |
|---|---|---|---|
| ATM variance premium | delta-hedged straddle | gamma (vs theta) | `gamma_pnl_be` |
| **vol-of-vol** (butterfly rich) | **vega-neutral butterfly** | **volga** | `volga_pnl_be` |
| **Skew** (risk-reversal rich) | **delta-hedged risk reversal** | **vanna** | `vanna_pnl_be` |

A distinction worth keeping sharp: *"the wings are rich"* is not the same as
*"the smile is convex-rich."* A rich 10-delta put is partly a convexity bid
(volga, symmetric) and partly a crash bid (vanna, one-sided). Selling a
strangle harvests both plus ATM VRP, all commingled. A **vega-neutral fly**
isolates volga; a **delta-hedged risk reversal** isolates vanna. Those two are
the atoms of the strategy as conceived.

### The biggest single reframe

The old stack already computed `gamma_pnl_be`, `vanna_pnl_be` and
`volga_pnl_be` — the "breakeven" buckets from Ravagli (2024), which net the
realised move against what the market's own implied dynamics predicted. It
treated them as a **diagnostic**.

**Here they are the objective function.** `volga_pnl_be` *is* the realised
convexity premium. `vanna_pnl_be` *is* the realised skew premium. The
measurement apparatus for this strategy already existed; it was just labelled
wrongly.

Corollary: the metric to rank strategies on is not calmar or total P&L. It is
**premium harvested per unit of the greek that harvested it** —
`RunResult.pnl_per_unit_greek('volga_pnl_be', 'volga_1vp')`.

---

## 1. The architectural change

Everything else follows from one line:

```
OLD:  for trade in trades:   for date in dates:   ...
NEW:  for date  in dates:    for position in book: ...
```

In the old stack, `run_backtest_multi_leg` took a static position and walked it
forward; `run_signal_backtest` called that once per entry date; the "book" was
reassembled afterwards by `reporting.build_daily_book`. Consequences, all
structural rather than financial:

- **One expiry per call.** Calendar spreads were impossible.
- **Static positions.** No rolling, no resizing.
- **Per-trade hedging.** Eight overlapping trades ran eight partly-offsetting
  hedges and paid transaction cost on *gross* delta where a desk pays on *net*.
- **Wide `*_legN` columns.** Cannot survive a variable position count.

Inverting the loop unlocked all four at once. The book became the primary
object that you steer, rather than an accounting artifact produced at the end.

---

## 2. Layout and dependency order

Each layer imports only from layers below it. `core/` never imports `market/`,
which is why the old `data.py` was split.

```
core/pricer.py        Black-76 closed forms on three time clocks    REWRITTEN
core/vol_surface.py   SABR smile, weekend-weighted ATM interp       LIFTED VERBATIM
core/calendar.py      FX spot lag / expiry conventions (QuantLib)   LIFTED VERBATIM
core/conventions.py   tenor/delta pillars, calendars, rate tickers  split from old data.py
core/greeks.py        GreekVector — the single risk unit            NEW
core/option.py        FXOption -> base-ccy partials -> GreekVector  ADAPTED

market/feeds.py       Bloomberg pulls (only xbbg/pdblp dependency)  split from old data.py
market/dataset.py     surface / spot store + the rate curves        REWORKED
market/snapshot.py    point-in-time view + strike/vol fixed point   NEW

book/position.py      LegRequest (intent) -> Position (fill)        NEW
book/costs.py         option spread in VOL POINTS                   NEW
book/attribution.py   one position, one day -> one record row       NEW
book/book.py          all positions + netted hedge + cost routing   NEW

engine/loop.py        the date loop, strategies, config, results    NEW

strategy/sizer.py     greek target -> LegRequests (LP over a menu)  NEW  Phase 3
strategy/roller.py    maintain a greek target across rolls          NEW  Phase 3

recon/reconcile.py    Phase 1 gate: diff against the old engine     NEW
recon/smoke_test.py   offline synthetic-market test, no Bloomberg   NEW
run/breakeven_study.py Phase 2 gate: cost feasibility               NEW
test.py               Phase 3 test suite — 7 tests, none yet run    NEW

risk/                 EMPTY — Phase 7
```

~8,600 lines total, of which `strategy/` is ~1,300. The SABR surface and the calendar engine are still verbatim
from the old stack; `core/pricer.py` no longer is (§3.2) and the rate half of
`market/dataset.py` no longer is (§3.3).

---

## 3. Module by module

### 3.1 `core/greeks.py` — one risk unit for the whole stack

**The problem it solves.** The old `FXOption.greeks_foreign()` returned a dict
mixing five scaling conventions: delta premium-adjusted, gamma 1%-scaled, vega
`×0.01/S`, **volga completely raw**, vanna premium-adjusted but unscaled. Each
consumer then re-scaled by hand — `* 0.01 * 0.01 / prev_spot` appears at three
separate call sites in `backtest_MLeg.py`. Survivable when greeks are a display
item; fatal when you size positions on volga.

**The fix.** Scaling happens **once**, at construction. Every field of a
`GreekVector` is:

> base-currency P&L, position-level (notional and direction already applied),
> for one standardised move — 1% of spot, 1 vol point, or 1 calendar day.

```python
spot_1pct       P&L for +1% spot            (1st order)
gamma_1pct      P&L for +1% spot            (2nd order, includes the ½)
vega_1vp        P&L for +1 vol point        (1st order)
volga_1vp       P&L for +1 vol point        (2nd order, includes the ½)
vanna_1pct_1vp  P&L for +1% spot AND +1 vol point together
theta_1d        P&L for one calendar day
delta_hedge     base-ccy SPOT notional to sell to flatten  (a QUANTITY, not P&L)
```

**Why it matters.** Because every field is money-for-a-standard-move,
GreekVectors are additive across legs, tenors, pairs and sleeves with no
further thought. That additivity is what makes a book-level risk budget and a
greek-target sizer possible at all.

`delta_hedge` is carried separately and **deliberately becomes NaN when you
aggregate across pairs**, because summing spot hedge notionals across different
pairs is meaningless. Failing loudly beats returning a plausible wrong number.

**The payoff appears in `taylor_pnl`,** which contains no scaling constants:

```
u = (dS / prev_spot) / 1%        spot move in standard units
w = dsigma / 1vp                 vol move in standard units

theta = theta_1d       * dt_days
gamma = gamma_1pct     * u²
vega  = vega_1vp       * w
volga = volga_1vp      * w²
vanna = vanna_1pct_1vp * u * w
```

`as_trader_units()` returns the conventional desk numbers (delta in notional,
gamma as delta-per-1%, vanna as delta-per-vol-point, volga as
vega-change-per-vol-point). **That view is for reading, never for arithmetic** —
gamma and vanna come out in notional units while vega, volga and theta come out
in money, so they cannot share a risk budget.

> **DEFECT, UNFIXED — `as_trader_units()` gamma and vanna are wrong by ~1/S.**
> Both divide by `SPOT_MOVE * S` where the derivation gives `SPOT_MOVE` alone.
> `delta_hedge` is a NOTIONAL, so its rate of change must also be a notional;
> the shipped numbers come out as notional-per-unit-of-spot. The error scales
> with spot exactly as that diagnosis predicts — 99.3% off on USDJPY (S≈159),
> 13% on EURUSD (S≈1.167). `delta`, `vega_1vp`, `volga_1vp` and `theta_1d` in
> that view are correct.
>
> ```
>                as_trader_units()      bumped truth
> gamma_1pct             -8,698.88     -1,357,150.54
> vanna_1vp               1,560.31        247,980.65
> ```
>
> Corrected forms:
> ```
> gamma_1pct = 2*self.gamma_1pct/SPOT_MOVE + SPOT_MOVE*self.delta_hedge
> vanna_1vp  = self.vanna_1pct_1vp / SPOT_MOVE
> ```
> **Nothing in `engine/` or `book/` reads this view**, so no backtest P&L is
> affected — it is display-only. But it is precisely the view you would hold up
> against an OVML ticket, which is where it will bite. Left unfixed
> deliberately; fix it before anyone uses the trader view for sizing.

### 3.2 `core/option.py` — the base-currency derivation, on three clocks

#### The forward is the state variable; spot is only an input to it

`core/pricer.py` is written in Black-76 form:

```
F  = S · exp((r_d − r_f) · τ_fwd)
DF = exp(−r_d · τ_disc)
d₁ = (ln(F/K) + σ²·τ_var/2) / (σ·√τ_var)
price = DF · [F·N(d₁) − K·N(d₂)]
```

Spot appears **nowhere except inside `F`**. That is not cosmetic — it is the
economically correct statement. A one-month USDJPY option is a claim on the
one-month *forward*, because that is the rate at which it delivers. Spot matters
only as today's observable that, combined with the rate differential, pins the
forward.

Practical consequence: **moneyness is `ln(K/F)`, never `ln(K/S)`.** A 25-delta
strike is a statement about where K sits relative to F, so any error in the
forward moves the strike.

#### The three clocks

An FX option runs on three time windows. The old code collapsed them into one
`T = (expiry − today)/365`:

| window | span | what it drives |
|---|---|---|
| `τ_var` | today → **expiry** | how much variance accumulates. σ enters **only** here. |
| `τ_fwd` | **spot date** → **delivery** | the rate differential accrues here. `r_d − r_f` enters **only** here. This is carry. |
| `τ_disc` | today → **delivery** | the payoff arrives here. Pure discounting, no risk content. |

They differ by the T+2 settlement lag at each end. Usually the two lags cancel
to within a day; a holiday cluster at either end pulls them apart. USDJPY 1M
struck 21-Aug-2026 runs **28 / 31 / 35 days**, because Tokyo's Silver Week
(21, 22 and 23 Sep 2026 are all JP holidays, a five-day closure) stretches
expiry→delivery to seven calendar days.

`FXOption.time_basis(date)` derives all three from the pair's settlement
calendar and returns a `pricer.OptionTime`. It is **cached per date**, because
`strike_from_delta`'s root-find reprices the same option dozens of times on one
date and each call would otherwise walk the calendar.

Every pricer function takes `T` as **either** a float (all three windows equal —
the old behaviour) **or** an `OptionTime`. The float path was regression-tested
against the original formulas on 4,000 random tickets × both option types: max
relative error **1.4e-12**. Nothing that passes a float changed.

#### Why it mattered

Model forward against live market outrights, worst error across USDJPY /
EURUSD / GBPUSD / AUDUSD at 1M / 3M / 6M / 1Y: **0.05 pips**. Before this pass
it was 52 pips at 1Y and ~4 pips at 1M. Solved 25-delta strikes now come back at
25.000000; before, a "25 delta" USDJPY 1M put was really a **25.31** delta.

The error was **jitter, not bias** — the median `τ_fwd − τ_var` gap across all of
2026 is **zero**, with tails of ±3–5 days. That is worse than a bias, because it
does not cancel in daily differences. Holding spot and vol fixed, the artifact
injected ~180–330 USD/day of standard deviation into the mark of a single 10mm
leg, against theta of ~1,540 USD/day.

Expiry dates themselves were checked against Bloomberg `OPT_EXPIRE_DT` for both
USDJPY and EURUSD across all ten tenor pillars: **20 for 20**, and every
delivery matches `SETTLE_DT`. `core/calendar.py` is faithful to Clark §1.4–1.5;
the Silver Week result above is correct, not a bug.

#### Three deltas, three different hedges

USDJPY 1M 25Δ put, 10mm, at S = 159.04, F = 158.65220:

| | per 1 USD | on 10mm |
|---|---|---|
| **spot delta** `∂P/∂S` = `DF·(F/S)·N(d₁)` | −0.250078 | −2,500,777 |
| **forward delta** `∂P/∂F` = `Δ_S · S/F` | −0.250689 | −2,506,890 |
| **premium-adjusted spot delta** (what the engine trades) | −0.253691 | −2,536,912 |
| premium embedded in base ccy | −0.003614 | −36,136 |

- **Spot delta** is what you trade in spot — but a spot hedge settles T+2 while
  the option delivers 35 days out, so it must be rolled via FX swaps. The roll
  cost **is** the carry, and it is not in this number.
- **Forward delta** is the fully-matched hedge: a forward to the option's own
  delivery date. No rolling, no carry leakage. `Δ_F = Δ_S · S/F` — here a
  0.244% / $6,113 difference.
- **Premium-adjusted spot delta** nets out the FX exposure in the premium
  itself, since a USD-notional USDJPY option is paid for in USD. This is what
  `delta_hedge` returns and what `book/book.py` nets into one spot trade.

`delta_hedge` is unchanged in meaning — still base-ccy **spot** notional. What
changed is that it is now computed off the right forward.

#### Theta now decomposes, and it jitters

All three clocks shrink as today advances, so

```
theta = −(∂P/∂τ_var + ∂P/∂τ_fwd + ∂P/∂τ_disc)
      =   variance decay  +  carry  +  discounting
```

which is the first time those three are separable. But the **spot date advances
in discrete business-day jumps**, so `τ_fwd` is flat on some calendar days and
drops by three across a weekend. Analytic theta is the *smooth* derivative;
realised next-day decay jitters around it by up to ~12% of daily carry:

```
val date      τ_var  τ_fwd   analytic   realised   jitter
2026-08-21       28     31      1,551      1,361     -190
2026-08-24       25     30      1,579      1,583       +3
2026-08-28       21     24      1,592      1,431     -160
2026-09-11        7     10      1,221      1,113     -109
```

That is genuine settlement-calendar granularity, not a modelling error, and it
belongs in `recon_resid` rather than in theta. Practical read: **on a Friday
mark, realised decay undershoots the greek and catches up on Monday.**

#### Other changes in this module

- **`FXOption.delivery`** — the spot date of the expiry, on the pair's
  settlement calendar. Fixed at construction, computed lazily.
- **`FXOption.forward(S, r_d, r_f, date)`** — the outright to *this option's*
  delivery date. Use this rather than `forward_price(...)` with a hand-rolled `T`.
- **`call_delta_equivalent` parity fix.** The put→call conversion is
  `d_c − d_p = DF·F/S`, which collapses to `exp(−r_f·T)` only when the clocks
  agree; it used the collapsed form. Parity now holds to 10 decimals.
- **`atm_forward_strike(..., pair=...)`** — pass `pair` to accrue over the true
  spot→delivery window. Without it, it falls back to today→expiry.

#### The base-currency identities are untouched

The engine accounts in **base (foreign) currency**, so one unit of notional is
worth `V = P / S` where `P` is the domestic price. Differentiating `V` rather
than `P` gives a clean rule:

> Derivatives with respect to **spot** pick up a premium-adjustment correction,
> because `S` sits in the denominator of `V`. Pure **vol** and **time**
> derivatives just divide by `S`.

```
dV/dS      = D_pa / S
d²V/dS²    = G_raw / S − 2·D_pa / S²
dV/dσ      = vega_raw / S
d²V/dσ²    = volga_raw / S
d²V/dSdσ   = (vanna_raw − vega_raw/S) / S
dV/dt      = theta_raw / S
```

Derivations of the two non-obvious ones are in the module docstring. The test
block **verifies all five against numerical differentiation of `value_base`**
and matches to ~1e-6 relative error. That test is the real proof the premium
adjustment is right.

Every one of these identities follows from `V = P/S` **alone**, so none of them
changed when `P` moved to three clocks — only the raw greeks feeding them did.
That is why the three-clock change touched `pricer.py` and `option.py` but not
`greeks.py`, `book/` or `engine/`.

**Trap recorded:** `pricer.gamma_premium_adjusted` is Wystup's Γ_pa, which is a
*different quantity* from `d²V/dS²`. Do not substitute one for the other. This
module uses the identity, not that function.

`greeks_foreign()` was deliberately **not** ported. Use
`.greeks(...).as_trader_units()` instead.

### 3.3 `market/dataset.py` — the rate curves

Rates used to be an afterthought: a **flat overnight SOFR fixing** for the USD
leg, Bloomberg forward-implied yields for everything else, interpolated on
**static** tenor-day integers. Three things were wrong, and they compounded.

**1. Flat USD broke covered interest parity.** Bloomberg's `XXXI<tenor>`
forward-implied yields are implied *off the USD SOFR OIS curve*. Backing the USD
rate out of market FX forward points plus `XXXI` reproduces `USOSFR` to within
**0.03 bp** on every G10 pair and tenor tested. Pairing those yields with a flat
overnight rate therefore mis-states the forward by construction — the curve runs
3.65% at 1W to 4.02% at 1Y against an overnight fixing of 3.63%, so ~5 bp at 1M
and ~38 bp at 1Y.

The USD leg now reads that curve (`USD_OIS_TICKER` in `core/conventions.py`:
`A`–`K` = 1M–11M, plain digit = whole years, `1Z/2Z/3Z` = 1W/2W/3W). Out to 1Y
the annual fixed leg has a single payment, so **par == zero and no bootstrap is
needed**. Past 1Y it would be, and the map says so.

**2. Both families are SIMPLE rates, and the basis is per-currency.** They were
being handed to the pricer as if continuous. At 3.68% over 31 days that alone is
a 4.5 bp error. Conversion is now

```
r_cont = ln(1 + r_simple · d_accrual / basis) · 365 / d_accrual
```

with `basis` from `MM_BASIS` — ACT/360 for USD/EUR/JPY/CHF/NOK/SEK/BRL/MXN,
**ACT/365 for GBP/AUD/NZD/CAD**. Assuming 360 everywhere leaves GBP 5.25 bp, AUD
6.09 bp, NZD 3.91 bp and CAD 2.85 bp adrift at 3M; with the map, all ten
currencies close to under 0.15 bp.

`d_accrual` is **spot → delivery**, not today → expiry. It is the window the
quoted rate actually accrues over; conflating the two converts to the wrong
continuous rate. `_compute_accrual_days` supplies it alongside
`_compute_pillar_days`, both cached in one pass per `(pair, date)`.

**3. Nodes sat on static `TENOR_DAYS` integers.** The vol surface had already
moved to real expiry-derived day counts; the rate curve had not, so the two
lived on different x-axes. Worse, it broke the property that matters most —
**at a pillar, interpolation must be the identity**. On 2026-08-21 the real
USDJPY 1M expiry is 28 days out while `TENOR_DAYS` says 30, so asking for the
actual 1M option's rate returned a blend of the 3W and 1M quotes instead of the
1M quote. EURUSD 1M the same week is 33 days, erring the other way. Both curves
now use `_compute_pillar_days` and the pillar identity holds to 0.00 bp.

**Node calendar is the TRADED pair, for both legs.** What drives the forward is
the differential `r_d − r_f`, so both legs must be read at the same x or the
differential is contaminated: at each pillar both currencies must return their
own quoted rate, not one quote and one interpolation. Putting each currency on
its own `XXXUSD` calendar instead would reintroduce that error on every cross.

**Two dead tickers were being silently swallowed.** `FWD_YIELD_TENORS` contained
`3W` and `1Y`. `XXXI3W` does not exist for any currency, and the 1Y point is
quoted `XXXI12M`, not `XXXI1Y`. An `except KeyError: continue` hid both, so **the
rate curve topped out at 9M and clamped flat above it** — a 1Y option read the 9M
rate. `3W` is now dropped from the rate grid (it stays in `TENOR_DAYS`, which is
the vol grid) and `FWD_YIELD_SUFFIX` maps 1Y → `12M`.

**Result.** USDJPY model forward against live market outrights, in pips:

| | 1M | 3M | 6M | 1Y |
|---|---|---|---|---|
| before | +4.72 | +7.01 | +17.85 | **+52.27** |
| rates fixed | +3.74 | +1.22 | −1.23 | −2.34 |
| + three clocks (§3.2) | −0.00 | −0.02 | +0.04 | **+0.05** |

Note the middle row barely helps at 1M: once the rates are right, the entire
remaining error is the clock mismatch. The two fixes are complements, not
alternatives.

**The standing check.** The falsifiable test is CIP against the market: for each
pair and tenor, `S·(1 + r_d·d/basis)/(1 + r_f·d/basis)` must reproduce the
outright implied by `PAIRnM BGN Curncy` forward points to under a pip. That one
test found all three problems above, and it tests against observable market data
rather than internal consistency. Re-run it after any rates change.

### 3.4 `market/snapshot.py` — point-in-time discipline, made structural

**The problem.** Trap #29 in the old guide reads: *"Point-in-time discipline is
yours. Nothing checks that your signal was built without look-ahead."* That is a
standing invitation to a bug that is invisible until it is expensive.

**The fix.** A `MarketSnapshot` is constructed for one `(pair, as_of)` and
exposes only market state at or before it. Trailing-window accessors
(`spot_history`, `quote_history`, `atm_history`) hard-truncate at `as_of`.

> **Rule for everything added from here on: if a function needs market data it
> takes a `MarketSnapshot`, never an `FXVolDataset`.**

**Second job — the strike/vol fixed point.** Solving a strike from a target
delta needs a vol; the smile vol needs a strike. `solve_strike_and_vol`
resolves that circularity in one place, and the pair it returns is
**self-consistent**: `sigma == smile_vol(K)` exactly. (The old engine's was not
— see §5.1.)

Interface: `spot`, `rates(t)`, `atm_vol(t)`, `forward(t)`, `smile_vol(K,t)`,
`nu_rho(t)`, `price_state(K, expiry)`, `solve_strike_and_vol(...)`,
`spot_history(n)`, `quote_history(tenor, field, n)`.

`forward(t)` returns the outright to the **delivery** date of an option expiring
`t` days out, not `S·exp((r_d−r_f)·t/365)` — see §3.2. `smile_vol` uses that same
forward, so the SABR strike↔delta conversion and the pricer now agree on what F
is. `rates(t)` returns **continuous ACT/365** zero rates off the curves in §3.3.

### 3.5 `book/position.py` — intent versus fill

Two objects, deliberately separate:

- **`LegRequest`** — what you ASK for. `('put', -1, 10mm, '1M',
  target_delta=-0.25, sleeve='skew')`. Delta space, notional space, **no market
  data in it**, so a strategy can express intent without touching a snapshot.
- **`Position`** — what you GOT. A struck strike, the smile vol it was struck
  at, the premium, an id, and a mutable notional. Created by `open_position`,
  which is the only place market data enters.

That mirrors order → fill, and it is the seam where Phase 3's sizer plugs in:
the sizer emits `LegRequest`s and the engine fills them. **That seam held** —
`strategy/sizer.py` was added without a single change to `book/` or `engine/`,
which is the main evidence that the intent/fill split was the right shape.

**`sleeve`** is a free-text attribution tag (`'convexity'`, `'skew'`, `'atm'`,
`'hedge'`). Tag honestly at creation — it cannot be reconstructed later, and it
is what turns "did the volga sleeve make money?" into a groupby rather than a
re-run.

**Convention inherited from the old stack and still enforced:** direction lives
in `direction` (+1/−1), never in a negative notional. `notional` is always a
positive magnitude. `LegRequest.__post_init__` raises if you violate it.

**`value_base` is signed** — a short position has negative value because it is a
liability — so differences in `value_base` *are* P&L with no sign flips anywhere
downstream.

### 3.6 `book/costs.py` — option transaction costs (Phase 2)

**Why this is the gating item.** Neither the old stack nor Phase 1 charged
anything to trade an option; `tc_fraction` only ever multiplied the *spot*
hedge. For a delta-hedged ATM strategy that is survivable. For a strategy whose
entire edge is a couple of vol points from the **wings** it is not, for three
compounding reasons: wing spreads are multiples of ATM spreads; a rolled book
pays repeatedly; and the premium is small in absolute terms so the cost is a
large fraction of it.

**Model** (parametric — a deliberate choice, see §6.2):

```
spread_vp(pair, t_days, |delta|) = base_1M_ATM[pair]
                                 × tenor_mult(t_days)     ~ sqrt(30/t), clipped [0.6, 3.0]
                                 × wing_mult(|delta|)     50d 1.00 → 25d 1.40 → 10d 2.20 → 5d 3.50
                                 × scale
```

Assumed USDJPY surface, full bid-ask in vol points:

```
  tenor |   50d    35d    25d    15d    10d     5d
     1W |  0.518  0.621  0.725  0.906  1.139  1.811
     1M |  0.250  0.300  0.350  0.438  0.550  0.875
     3M |  0.150  0.180  0.210  0.263  0.330  0.525
```

**Two modelling decisions worth understanding:**

1. **Cost is charged as vega × vol points**, not as a fraction of notional.
   That is how FX options actually trade, and it matters: a 5-delta option has
   almost no vega, so a flat vega-based cost would make deep wings look *cheap*.
   They are not — their vol spread is 3.5× ATM. The two effects partly offset,
   which is exactly why they must be modelled separately.
2. **There is a premium floor** (`min_premium_frac`, default 3%). At 5 delta and
   short tenors vega collapses toward zero and a pure vega×spread cost would go
   to zero with it. Real market makers hold a minimum spread as a fraction of
   premium. The charged cost is `max(vega_cost, floor_cost)`, and every cost
   record reports which one bound.

**`crossing_fraction`** (default 0.5) is how much of the quoted width you
actually pay — 0.5 means dealing at bid or ask from mid. Set 1.0 as a stress
case for a systematic taker.

**`scale`** is the global knob the break-even study sweeps. `scale=0`
reproduces Phase 1 (free options) exactly.

**Expiry settles free; early closes pay.** An expiring option is not traded out
of. That asymmetry is real and is a genuine argument for holding to expiry over
rolling early in a wing strategy.

### 3.7 `book/attribution.py` — the daily mark

Given a position, its state at the START of the period, and a snapshot at the
END, produce one record row.

- **All attribution is priced off START-of-period greeks.** The greeks written
  into a row describe the position going *into* that day, not coming out of it.
  Same convention as the old engine; mixing the two is the classic way to get an
  attribution that looks right and drifts.
- **`dt_days` is the ACTUAL calendar gap** — 3 over a weekend, not 1. Theta and
  carry scale by it. Gamma and vol P&L over a gap are realised as one lump move
  off Friday's greeks (standard daily-bar limitation).
- **Natural expiry is special-cased:** the option settles to intrinsic, all vol
  buckets are hard-zeroed, but gamma and theta are still computed from
  start-of-day greeks because the realised spot move over that final period was
  real.
- **`recon_resid`** is the gap between the exact reprice and the Taylor
  expansion. Unlike the old engine's version it is per-position and hedge-free
  (it includes the first-order delta term instead of netting the hedge in), so
  it is a clean "how good was the expansion" number. It is a **diagnostic and a
  risk metric**, not an error to drive to zero — see §7.2.
- Hedging is **not** here. A position only reports its own delta; netting and
  the single hedge live in `book/book.py`.

### 3.8 `book/book.py` — the primary object

Holds `positions`, cached `marks` (one per open position, rolled forward daily
so the surface is hit once per position per day), the per-pair spot `hedge`, the
`cost_model`, and a `cost_log`.

**Netted hedging** is the substantive gain over the old stack. `net_delta(pair)`
sums premium-adjusted deltas across *every* open position in that pair, and
`rebalance_hedge` places **one** spot trade against it. In the offline test, on
the busiest day of a rolled book: gross |delta| 18.9mm across legs versus **net
0.54mm** — 97% less to trade. The old engine paid transaction cost on gross.

**`residual_delta(pair)` = option delta + hedge carried.** That, not gross option
delta, is what a band-hedging rule must test. (The old stack's `DeltaBandHedge`
tested gross and therefore either never rehedged or degenerated into a daily
hedge — a bug that was already known and fixed there.)

**The Book owns the cost model.** Every option trade routes through
`Book.open` / `Book.close` / `Position.resize`, so a strategy physically cannot
trade an option without paying the spread. A forgotten cost is exactly the kind
of bug that makes a backtest look good.

### 3.9 `engine/loop.py` — the date loop

The six daily steps, **in an order that is not negotiable**:

| # | Step | Why here |
|---|---|---|
| 1 | **Snapshot** | point-in-time view per pair |
| 2 | **Mark** | attribute yesterday→today on the book AS IT WAS. Before trading, or a day's P&L is attributed to a size that was not on risk for it. |
| 3 | **Hedge P&L** | on the hedge CARRIED IN. Before rebalancing, or the hedge gets a one-day look-ahead and every result is flattered. |
| 4 | **Trade** | the strategy opens / closes / resizes. The only step that changes composition. |
| 5 | **Rehedge** | one net spot trade per pair against the NEW book. |
| 6 | **Record** | drain option costs, roll up the daily row. |

Swapping 2↔4 or 3↔5 produces a backtest that looks fine and is wrong.

**Strategies** are any object with `on_date(ctx)`. Two ship:

- **`HoldStatic`** — one vintage to expiry. Exists solely because it is the
  configuration the old engine also produces, i.e. the only thing that can be
  reconciled. Use it to validate, then never again.
- **`RollingStructure`** — re-opens every `roll_days`, previous vintages run to
  expiry alongside. The Phase 2 baseline and the first config where netted
  hedging actually differs from per-trade hedging.

**Date grid** is the union of every requested pair's observed spot dates. A pair
missing on a date is skipped, not forward-filled — inventing a mark is worse
than not having one.

### 3.10 `strategy/sizer.py` — the greek-target sizer (Phase 3)

**The inversion.** Up to here a strategy was specified in *instrument* space —
`LegRequest('call', -1, 10_000_000, '1M', target_delta=+0.25)`. That 10mm is
arbitrary: it is a different amount of risk at 1M than at 3M, and different
again in a 5-vol pair than a 12-vol pair. Every downstream number inherits the
arbitrariness. This module lets you specify *risk* and solves for the notionals:

```python
solve(snap, tenors='3M',
      target=GreekTarget(by_tenor={'3M': dict(volga=-2_500, vega=0, vanna=0)}),
      sleeve='convexity')                      # -> SizerResult, with .legs
```

#### The three ideas it rests on

**1. Greeks are linear in notional, so sizing is a linear solve.**
`GreekVector.from_partials` multiplies everything by `q = notional × direction`,
so if `gᵢ` is candidate leg *i*'s greek vector at unit notional, the book's
greek is `Σ nᵢ·gᵢ`. Stack the `gᵢ` as columns of `A` and hitting a target `t` is
`A n = t`. This is the whole payoff of §3.1's single-risk-unit decision.

**2. The menu is strikes, not options.** By put-call parity a same-strike call
and put differ by a forward, and a forward has no vol sensitivity — their vega,
vanna and volga are *identical*. They are one column of `A`, not two. So the
menu takes one leg per strike, **OTM** (puts below the forward, calls above),
which is also the side that quotes tight. `otm_only=False` overrides.

ATM is the exception: it enters as a **straddle** (call + put at ATM-forward,
equal notional), because a single ATM leg is a large naked delta the solve has
no reason to want. `atm_as_straddle=False` splits it into two columns.

Default pillars are `('ATM', 35, 25, 10, 5)` — every one a directly quoted
RR/BF point from `DELTA_POINTS`, so nothing relies on SABR extrapolation
between knots. Nine strikes per tenor.

**3. Under-determined is the normal case, so the objective matters.** Nine
columns against three constraints leaves an infinite family of solutions. The
one chosen minimises **expected spread paid**, using the same
`OptionCostModel` the engine charges (§3.6). That turns "which structure" from
a taste question into a priced one.

#### The objective is a linear program, and two earlier attempts failed

Recorded because both wrong versions looked reasonable.

- **Cost-weighted L2** (one `lstsq` call). Optimises the wrong thing, and not
  merely imprecisely: L2 actively *prefers* to spread notional across columns,
  because splitting a position in two lowers the sum of squares while leaving
  the sum of absolutes unchanged. It returned an eight-leg smear across every
  strike on the menu where the answer was a four-leg fly.
- **IRLS towards L1** (reweight by `1/√|n|`, iterate). Right objective, but it
  stalls. On the gamma calendar it locked onto a near-duplicate 35Δ put/call
  pair — effectively a synthetic forward — at 3.4bn a side and could not climb
  back out.

The shipped solver is an **LP**: split `n = p − m` with `p, m ≥ 0`, minimise
`Σ costᵢ(pᵢ + mᵢ)` subject to `A(p − m) = t`, via `scipy.optimize.linprog`
(HiGHS). Convex, solved exactly, cheap at a few dozen columns — and it carries a
property worth relying on:

> **An LP basic solution has at most (number of constraints) non-zero
> variables.** Three constraints → at most three legs. Sparsity is a property
> of the optimum, not a post-processing step.

Rows are normalised to unit length before solving. Without that, a gamma row
(~1e5) and a volga row (~1e3) are weighted by three orders of magnitude of unit
accident rather than by anything chosen. There is an `lstsq` fallback for
infeasible systems; `SizerResult.method` reports which path ran.

#### Buckets, and the term-structure trap

A flat greek vector sums across expiries, and that summation hides things. Sell
1M vega, buy 3M vega, size for zero *net* vega: the `GreekVector` reads
vega-neutral and the book is a pure term-structure position, because front vol
and back vol do not move together. This is the FX analogue of a bond book with
zero net duration and large curve exposure, and the fix is the one rates desks
use — bucket by maturity instead of summing:

```python
GreekTarget(
    by_tenor = {'1W': dict(vega=0), '3M': dict(vega=0)},   # each flat on its own
    net      = {'gamma': +280_000},                        # sum only
)
```

**Rule: anything you want FLAT goes in `by_tenor`. Use `net` only when you
deliberately want one bucket to offset another.**

Three states per greek, and the difference is load-bearing: `'vega': 0.0` is
*actively pinned at zero*; a **missing key** is *free, don't care*;
`'volga': -2500` is *hit this number*. "Zero" and "don't care" are not the same
instruction.

#### Normalised units

`units='normalised'` (default) denominates targets in **base-ccy P&L per
one-sigma move over `horizon_days`** — a number you can reason about rather
than an index. The multipliers are exactly `taylor_pnl`'s coefficients (§3.1)
evaluated at a one-sigma move:

```
u = (σ·√dt) / SPOT_MOVE            spot 1σ, in "1%" units
w = (σ·ν·√dt) / VOL_MOVE           vol  1σ, in "1vp" units

vega  -> vega_1vp       × w
volga -> volga_1vp      × w²       second order: the move enters SQUARED
vanna -> vanna_1pct_1vp × u·w
gamma -> gamma_1pct     × u²
theta -> theta_1d       × horizon_days
```

σ is the ATM vol at that tenor and ν is the SABR vol-of-vol from `snap.nu_rho`,
so the factors are per-tenor and per-pair. That is what makes one unit of volga
mean the same risk in 1W USDJPY as in 6M EURCHF — the old stack's trap #6,
which gets materially worse in second-order greeks because volga picks up the
move squared. `units='raw'` targets the `GreekVector` fields directly.

`SizerResult.scales[tenor]` reports the multipliers actually used, which is
where to look first if a normalised number looks two orders of magnitude off.

#### The two guards

A target the menu cannot meet honestly fails *silently*: the solver returns
huge offsetting notionals whose residual difference equals the target. It
satisfies the constraints on paper and the cost line says so six months later.
Two diagnostics, because they catch different things:

| Guard | Default | What it tests |
|---|---|---|
| `require_cond` | `1e4` | whether the **constraint rows** are independent — fires when you ask for two things that are the same thing |
| `require_leverage` | `10` | whether the **columns** are — gross greek magnitude deployed ÷ net delivered |

`cond(A)` is blind to the second failure: the constraints can be perfectly
independent while the answer is two adjacent strikes in billions of offsetting
notional. **`require_leverage` is the guard that fires in practice.** Measured
on real structures: vega-neutral fly 2.4, delta-hedged risk reversal 1.7,
front/back gamma calendar 1.9, single-tenor gamma-vs-vega 14.0. Hence the
threshold at 10.

**A textbook claim that is false in this stack, and it matters.** Gamma and vega
inside one expiry are usually described as exactly collinear —
`vega = gamma·S²·σ·T`, strike-independent ratio. Here they are not, because
greeks are premium-adjusted base-ccy P&L: `d²V/dS² = G/S − 2·D_pa/S²`, and the
`D_pa` term varies strongly across strikes. The columns are ~10% independent
rather than 0%, so a single-tenor gamma-vs-vega target is *feasible*. It is just
ruinous — measured on a 3M USDJPY menu, the same target:

| | gross notional | entry cost | leverage |
|---|---|---|---|
| one tenor (3M) | 9.3bn | $1,694,766 | 14.0 |
| two tenors (1W/3M) | 156mm | $26,240 | 1.9 |

65×. The fix is always a second, well-separated tenor.
`require_leverage=None` sizes it anyway.

#### What comes back

`SizerResult` is deliberately more than notionals — `residual` and the two
guard numbers are the honesty checks, and `cost` is what makes two candidate
structures comparable. `print(res)` gives the leg table, the
asked/carried/traded/book-after block, and the footer. Key fields:

```
legs / notionals / expiries       what to trade, and the expiry it was priced on
realised / realised_by_tenor      THIS TRADE's greeks, raw units, per bucket
carried / book_after              what was already on risk; the sum
achieved / residual               the BOOK after the trade, vs what was asked
condition / leverage / method     the guards, and which solver path ran
cost / gross_notional             spread to put it on, total notional
scales / bucket_map / dropped     the normalisation, the bucketing, what was pruned
```

`res.open_into(book, snap)` is the intended way to trade it — it is the one
place the tenor→expiry resolution is guaranteed to match the greeks the solve
was built on.

#### Structure builders

Presets over `solve()`, not separate code paths: `vega_neutral_fly` (the volga
atom), `risk_reversal` (the vanna atom), `vega_sleeve`, `gamma_sleeve`.

Two findings worth keeping from building them:

- **The vanna sleeve is not a clean risk reversal.** The two wings are not
  exactly volga-symmetric, so an RR sized purely for vanna leaves residual
  convexity. The solver bolts on a small ATM straddle — a few percent of the
  wing notional — to null it. Without that leg a "pure vanna" sleeve is quietly
  part volga, which defeats the point of separating the sleeves at all. This is
  the clearest single case of the sizer beating a hand-rolled structure.
- **`gamma_sleeve` uses `net`, not `by_tenor`.** The obvious specification —
  vega flat in *each* bucket plus net gamma — is infeasible for the collinearity
  reason above: a bucket with zero vega has essentially zero gamma. So gamma and
  vega separate *across* tenors, which means a gamma sleeve necessarily runs
  front-versus-back vega. That is not sloppiness; it is the unavoidable price of
  holding gamma without vega, and `realised_by_tenor` is where you read how much
  of it you took.

#### What this module does NOT do

It does not **search** for strikes. You give it a menu and it sizes within it.
Automatic strike selection is a discrete optimisation that would overfit
whatever the cost model assumes, and it belongs after Phase 5, when a richness
signal can say which parts of the surface are actually cheap.

It also does not carry the **portfolio-level budget** from the original Phase 3
spec (cap total book volga/vanna, allocate across pairs against a covariance of
the *signals* rather than of the P&L). That needs signals to have a covariance
of, so it has moved to Phase 5/6. See Part II.

### 3.11 `strategy/roller.py` — maintaining a greek target across rolls

**The incoherence this fixes.** `RollingStructure` (§3.9) re-opens a fixed
structure every N days and lets vintages overlap. Coherent for a fixed
notional: you asked for 10mm every 5 days, you get a stack of 10mm vintages.
**Incoherent for a greek target.** Ask for −2,500 of volga, roll three times
with overlap, and the book carries −7,500 — so the number no longer describes
the risk held, and the multiple depends on roll cadence and tenor. Every
downstream figure inherits the error.

A greek target has to be **maintained, not re-issued**. Three modes:

| mode | behaviour | verdict |
|---|---|---|
| `top_up` | read what the book carries, trade only the difference | **default, correct** |
| `replace` | close the sleeve, re-strike at target | correct, but pays the exit spread every roll |
| `stack` | re-issue the full target, let vintages accumulate | broken — exists only so the accumulation can be measured against the other two |

`top_up` is also the mechanism **Phase 4 needs**: a vega re-hedge is exactly
"the book has drifted to +X vega, trade the increment that returns it to zero",
which is this same call with a different target. Building `replace`-only would
have meant rewriting it next phase.

#### `solve(..., current=...)` — the sizer as a book-level controller

The target is read as a **book-level statement**. `current=` supplies what is
already on risk and the solve returns only the increment:

```python
solve(snap, '3M', target, current=ctx.book)    # trade the difference
solve(snap, '3M', target)                      # assume a flat book
```

Accepts a `Book` (normal case), a bare `GreekVector` (single-tenor menus only),
or a `{tenor: GreekVector}` dict. Verified on a stub: pass back 60% of a solved
structure and the increment is exactly 40% of the gross and 40% of the cost;
pass back 100% and it trades nothing.

**The judgement call, in `bucket_book()`.** Aged positions do not sit on the
menu — a 3M struck five days ago is a 2M-and-change today, and it must count
against *some* bucket or the incremental target is wrong. The rule is **nearest
bucket in log remaining-days** (log because tenor is multiplicative). The
mapping is returned in `SizerResult.bucket_map` and printed, so it is visible
rather than assumed. If the rule is wrong for your purposes, that one function
is the only thing to change; nothing downstream depends on how the assignment
was made.

Greeks for carried positions are recomputed off the passed snapshot via
`Position.greeks`, **not** read from `book.marks` — using cached marks would
silently mix two dates.

#### The log is part of the deliverable

A greek-target strategy has two new ways to quietly do nothing — the guard
trips, or the increment falls inside the deadband — and **neither shows up in a
P&L curve.** A strategy that skipped 40% of its rolls looks fine and is not the
strategy specified. So every attempt is recorded in `roller.log`, including the
no-trades, with `action ∈ {traded, deadband, skipped_guard}`, the gross and cost
traded, the guard numbers, and the book level reached on each constrained greek.
`roller.report()` prints the tally; `roller.frame()` returns it as a DataFrame.

`min_trade` is the deadband. Raise it and you trade less often but let the book
drift further from target between rolls; lower it and you pay spread to chase
noise. There is a real optimum and it depends on the drift rate, which is what
the Phase 3 gate measures — the two are meant to be read together.

#### The seam Phase 5 plugs into

`target_fn` takes a snapshot and returns a `GreekTarget`. Constant today:

```python
lambda snap: GreekTarget(by_tenor={'3M': dict(volga=-2_500, vega=0, vanna=0)})
```

In Phase 5 it becomes `volga = -2_500 × richness_z(snap)` and **nothing else in
the class changes.** That was the point of the whole design.

---

## 4. The data model

The single most consequential design choice after the loop inversion:
**one record row per `(date, position)`** — long format, not the old stack's wide
`*_legN` columns, which cannot survive a variable position count.

Every downstream question becomes a `groupby`: by sleeve, by pair, by
`tenor_label`, by `target_delta` bucket.

### `RunResult.positions` — one row per (date, position)

```
- date, pos_id, pair, sleeve, tag, tenor_label, target_delta, option_type, strike, 
  expiry, direction, notional

- dt_days, spot, prev_spot, dS, sigma, prev_sigma, dsigma, t_days, nu_be, rho_be, 
  expired

- spot_1pct, gamma_1pct, vega_1vp, volga_1vp, vanna_1pct_1vp, 
  theta_1d, delta_hedge   ← START-of-day greeks

- option_pnl, delta_pnl, gamma_pnl, theta_pnl, vega_pnl, vanna_pnl, volga_pnl,
  gamma_pnl_be, vanna_pnl_be, volga_pnl_be, recon_resid
```

### `RunResult.daily` — the book roll-up

```
- n_open

- option_pnl, delta_pnl, gamma_pnl, theta_pnl, vega_pnl, vanna_pnl volga_pnl 
  gamma_pnl_be, vanna_pnl_be, volga_pnl_be, recon_resid          ← FLOW

- net_spot_1pct, net_gamma_1pct, net_vega_1vp, net_volga_1vp, net_vanna_1pct_1vp, 
  net_theta_1d, net_delta_hedge             ← EXPO

- option_tc, hedge_pnl, hedge_carry, hedge_tc, net_hedge, pnl equity

- gamma_pnl_be_cum vanna_pnl_be_cum volga_pnl_be_cum
```

```
pnl = option_pnl + hedge_pnl + hedge_carry − hedge_tc − option_tc
```

**FLOW versus EXPO is the whole design of the roll-up.** Flow columns are summed
then cumsummed into equity. Exposure columns are summed and **never** cumsummed
— they are point-in-time levels. Conflating them is the easiest way to produce a
nonsense equity curve.

### `RunResult.hedges` / `.costs` / `.trades`

```
hedges : date, pair, hedge_pnl, hedge_carry, hedge_carried, dt_days, spot, dS,
         hedge_before, hedge_target, hedge_gap, hedge_traded, hedge_after,
         hedge_tc, rehedged

costs  : date, pair, pos_id, sleeve, tag, reason, cost, spread_vp, paid_vp, bound_by

trades : date, pos_id, pair, sleeve, tag, tenor_label, option_type, target_delta,
         strike, direction, notional, entry_date, expiry, entry_spot, entry_vol,
         entry_atm_vol, entry_premium, action
```

`costs.reason` is `open` / `close` / `resize`; `bound_by` is `vega` or
`premium_floor`, so every charge is auditable.

### Convenience readers

```python
res.pnl_per_unit_greek('volga_pnl_be', 'volga_1vp')    # THE ranking metric
res.by_sleeve('volga_pnl_be')                          # cumulative, split by sleeve
res.greek_carried('volga_1vp')                         # daily net exposure series
```

---

## 5. Bugs found in the old engine during reconciliation

Recorded because they affect how prior results should be read.

### 5.1 The strike bootstrap was off by half a step

`backtest_MLeg.py:336-353`:

```python
for _ in range(5):
    K_seed = find_strike_from_delta(..., sigma=sigma)
    sigma  = get_smile_vol(K_seed)          # vol at K_seed
K = find_strike_from_delta(..., sigma=sigma)  # one EXTRA solve
return K, sigma
```

The returned `sigma` is the smile vol at the **previous iterate's** strike, not
at the `K` returned. So options were struck at K but priced and risk-managed at
a vol belonging to a slightly different strike.

Harmless at 25 delta where the smile is shallow. In the reconciliation the
relative error in `option_pnl` was **8×10⁻⁶ at 25 delta but 5.7×10⁻⁴ at 10
delta** — 70× worse — because dσ/dK is steep in the wings, which is precisely
where this strategy operates. It will be worse again at 5 delta.

Reproducible via `legacy_strike_halfstep=True`.

### 5.2 The `_be` buckets used end-of-period ν/ρ — a look-ahead

`backtest_MLeg.py:630`:

```python
nu_BE, rho_BE = dataset.get_smile_nu_rho(pair, current_dt, t_remaining)
```

`current_dt` is **today's** date and **today's** remaining tenor, then applied to
yesterday's greeks. The `_be` buckets ask *"what did the market, at the start of
the period, imply this period's dS·dσ and dσ² would be?"* — that expectation must
be formed from start-of-period information.

**This is the most consequential of the three,** because `volga_pnl_be` and
`vanna_pnl_be` are the objective function of the new strategy. Observed
divergence: **6% on a 1M strangle, 10% on 3M**, with single-day gaps over 2,600.

It also explains the *shape* of the errors — large, sporadic, concentrated on
particular dates rather than evenly spread. `get_smile_nu_rho` snaps to the
**nearest tenor pillar**, so as `t_remaining` crosses a pillar midpoint ν jumps
discontinuously, and start-of-day versus end-of-day can land on opposite sides.

Reproducible via `legacy_nu_rho_at_end=True`.

### 5.3 Hedge carry used the option's shrinking remaining tenor

`backtest_MLeg.py:479` takes `prev_r_f − prev_r_d` at the *option's* remaining
tenor, which shrinks daily — an odd rate for an overnight spot hedge, and it
drifts as the trade ages. Worth ~2% of carry over a 3M trade, concentrated in the
final weeks. Reproducible via `carry_tenor_days=None`.

### 5.4 Deliberate divergence: transaction cost at expiry

The old engine charged no transaction cost on the terminal spot-hedge unwind at
natural expiry but full cost on an early exit — an asymmetry baked into every
hold-to-expiry versus exit-early comparison it produced. The new default charges
it. Reproducible via `charge_tc_on_expiry_unwind=False`.

---

## 6. Test infrastructure and gate status

### 6.1 `recon/reconcile.py` — the Phase 1 gate — **PASSED, now superseded**

> **Status note.** This gate passed and its conclusion stands: the rewrite was
> arithmetically identical to the old engine under the old engine's own rules.
> But the rates and pricing pass (§3.2, §3.3) introduced divergences that
> `legacy=True` has **no flag to switch off** — there is no toggle to restore
> single-clock pricing or a flat SOFR leg. **Re-running it today will fail, and
> that is expected, not a regression.** Either add `legacy_single_clock` and
> `legacy_flat_sofr` flags if you want the bit-identical gate back, or treat
> §3.2/§3.3's market-forward check (worst error 0.05 pips against live BBG
> outrights) as the replacement gate. The latter is the better test anyway: it
> validates against observable market prices rather than against a previous
> implementation.

Runs the same strangle through both engines and diffs it day by day, bucket by
bucket. Two passes:

- **Pass 1, `legacy=True`** — all four old conventions turned on. The two engines
  should agree to floating point. **This is the gate, and it passes.** It proves
  the rewrite is arithmetically identical to the old engine under the old
  engine's own rules.
- **Pass 2, `legacy=False`** — the new engine as intended. Every diff is a
  deliberate improvement and its size answers "how much was that convention
  costing me?"

Tolerance is **relative** (`rtol`) with an absolute floor (`atol`). An absolute
1e-6 on numbers of order 1e5 flags floating-point reassociation as failure,
which is useless — this was an early mistake in the harness.

The harness also prints an explicit **strike diff** against the old engine's
`leg_summaries`, in basis points, per leg. If the strikes differ, nothing below
them is meaningful.

Cases covered: 25Δ short, 10Δ short (deep wings), 25Δ **long** (every sign
flips), 3M (longer tenor), EURUSD (USD as quote rather than base).

### 6.2 `recon/smoke_test.py` — offline, no Bloomberg — **30 checks, all passing**

A synthetic market (GBM spot; mean-reverting ATM vol negatively correlated with
spot returns; a quadratic-in-standardised-log-moneyness smile with real skew and
convexity; a deliberate −4.5% spot jump) driving the full engine in about a
second.

It cannot catch a mistake in the SABR interpolation or the pillar arithmetic,
because it does not use them. What it does catch is **every wiring error**:
ordering, sign conventions, netting, expiry handling, aggregation, cost flow.

Covers: point-in-time truncation; strike/vol fixed-point convergence and
self-consistency; short-strangle sign sanity; Taylor sum + residual == exact
reprice identically; hedge offsetting delta P&L; multi-vintage multi-expiry
coexistence; gross-vs-net delta saving; `_be` buckets populated; sleeve
groupby; cost monotonicity in `scale`; cost reducing P&L one-for-one;
`scale=0` reproducing Phase 1; expiry settling free.

Run it after any refactor.

### 6.3 `run/breakeven_study.py` — the Phase 2 gate — **WRITTEN, NOT YET RUN**

See Part II §1.

### 6.4 `test.py` — the Phase 3 suite — **WRITTEN, NOT YET RUN**

Seven tests at the bottom of `test.py`, calls commented out, one at a time.
None has been run against Bloomberg. The sizer's linear algebra and all four
structure builders **were** verified end-to-end against an offline stub
snapshot (flat rates, synthetic skewed smile), which is what caught the L2 and
IRLS solver failures in §3.10 — but a stub cannot validate the SABR fit or the
pillar arithmetic.

| test | what it establishes |
|---|---|
| `p3_test1_volga` | the convexity atom on real data; ≤3 legs, residual ~1e-12, leverage ~2 |
| `p3_test2_wing_comparison` | same target across 35/25/10/5Δ, compared on **cost**. Phase 2's wing question asked per-structure. **Blocked on the premium-floor calibration** (§8) — its entire output is the cost column |
| `p3_test3_vanna` | the skew atom, and whether the solver adds the small ATM leg that de-contaminates it |
| `p3_test4_gamma_calendar` | multi-tenor works; and (b) that the bucketed-vega version correctly refuses |
| `p3_test5_guard` | the leverage guard firing, the answer it refused, and the two-tenor version's cost — side by side |
| `p3_test6_gate` | **THE PHASE 3 GATE.** Hold a sized fly and track `vega/volga` and `vanna/volga` drift |
| `p3_test7_roller` / `p3_test7b_compare_modes` | the roller; and the three modes compared on `peak_net_volga` and `option_tc` |

**Run order:** `test5` first (fast, no engine, validates the plumbing), then
`test7b` (validates the roller design decision), then `test6` (the gate).
`test2` last, after the premium-floor question is settled.

---

## 7. Conventions and invariants

### 7.1 The eleven rules

Anyone extending this must not break these:

1. **Mark → trade → hedge.** Swapping 2↔4 or 3↔5 in the daily loop gives a
   backtest that looks fine and is wrong.
2. **Direction lives in `direction`, never in a negative notional.**
3. **Market data comes from a `MarketSnapshot`, never a dataset.** That is the
   look-ahead guard.
4. **Greeks are P&L-per-standard-move — add those.** Read `as_trader_units()`;
   never sum it.
5. **`dt_days` is the actual calendar gap**, 3 over a weekend.
6. **Options trade through the Book**, so nothing skips the cost model.
7. **Pricing runs on three clocks.** Get them from `FXOption.time_basis()`.
   Never hand-roll `(expiry − today)/365` and pass it as `T` — that silently
   reintroduces the single-clock approximation for the forward and the
   discounting. Moneyness is `ln(K/F)`, never `ln(K/S)`.
8. **Rate curves are quoted SIMPLE on a per-currency basis.** Anything read
   straight out of `rate_curves` must go through `_simple_to_continuous` with
   that currency's `MM_BASIS` and its spot→delivery accrual window before it
   touches the pricer.
9. **A greek target is a BOOK-level statement, not a trade instruction.** Pass
   `current=` on every solve after the first. Omitting it re-issues the full
   target and the book accumulates a multiple of the risk you asked for, scaled
   by roll cadence and tenor (§3.11).
10. **Anything you want FLAT goes in `by_tenor`, never `net`.** A net-zero
    across expiries is satisfied by a large front-versus-back position that
    reports as neutral and is not (§3.10).
11. **Trade a `SizerResult` through `open_into`**, not by opening its legs
    yourself. It is the only place the tenor→expiry resolution is guaranteed to
    match the greeks the solve was built on.

### 7.2 Two things that are easy to misread

**`recon_resid` is a risk metric, not just a check.** It is the gap between the
exact reprice and the Taylor expansion, so it spikes on discrete jumps and where
the expansion breaks down. For a short-wings book **those are the days that
hurt**. In the smoke test the deliberate jump day shows a residual of 331k,
dwarfing everything else. Track its distribution.

**A single day of any `_be` bucket is noise.** The **drift of the cumulative** is
the signal — Ravagli's own framing. Judge on the drift per unit of the
corresponding greek carried.

---

## 8. Assumptions and inherited caveats

| Item | Status |
|---|---|
| **Cost levels are ASSUMPTIONS** | Parametric model, no bid/ask calibration. Treat any P&L derived from them as a *sensitivity*. The break-even **scale** is the robust output. |
| **`min_premium_frac` binds at ATM, not in the wings — OPEN** | The 3% premium floor exists because vega collapses at 5Δ (§3.6). But the floor is a fraction of **premium**, and premium is largest ATM, so it binds hardest at exactly the strike it was not aimed at. Measured on 3M USDJPY: ATM `cost/unit = 0.000558` bound by the floor against `0.000147` from the vega mechanism — **3.8×** — while 5Δ is vega-bound. Equivalent to charging 0.58 vp full width on a 3M ATM the model itself quotes at 0.25. This biases every sizer structure choice toward the wings. **Decide before reading any cost comparison** (`p3_test2_wing_comparison`'s entire output is the cost column). Options: leave it and read the tilt, drop to ~0.01, or make it delta-dependent so it only bites where vega actually collapses. Check the sensitivity by re-running with `min_premium_frac=0.0`. |
| **Gamma and vega are NOT exactly collinear here** | The textbook `vega = gamma·S²σT` identity does not hold for premium-adjusted base-ccy greeks; the `D_pa` term makes the columns ~10% independent. So a single-tenor gamma-vs-vega target is feasible and 65× more expensive than the two-tenor version (§3.10). Documented rather than fixed — the guard catches it. |
| **Aged positions are bucketed by nearest log remaining-days** | An incremental solve has to assign carried positions to menu tenors, and a 3M struck five days ago is not 3M. The rule is a choice, not a derivation (§3.11). `SizerResult.bucket_map` reports it on every solve. |
| **Guard thresholds are empirical** | `require_leverage=10` and `require_cond=1e4` are calibrated off four measured structures (fly 2.4, RR 1.7, calendar 1.9, the pathology 14.0). Comfortable headroom on that sample; not validated across pairs, tenors or vol regimes. A legitimate structure tripping the guard is possible — read the error, do not reflexively pass `require_leverage=None`. |
| **`as_trader_units()` gamma/vanna are wrong** | Off by ~1/S. Display-only, nothing in `engine/` or `book/` reads it, so no backtest P&L is affected. **Unfixed** — see the box in §3.1 for the corrected forms. |
| Rate curves stop at 1Y | `USOSFR` par == zero only while the annual fixed leg has a single payment. Extending `TENOR_DAYS` past 1Y requires bootstrapping the >1Y points first. |
| `r_d` discounts over `τ_disc` | The zero read at the *expiry* pillar is applied over the today→delivery window. A few days of extrapolation on a curve node; immaterial at G10 rate levels, less so for a high-rate quote ccy (MXN, BRL). |
| Theta is the smooth derivative | The spot date rolls in discrete business-day jumps, so realised decay jitters ±~12% of daily carry around the greek (§3.2). Expected behaviour, lands in `recon_resid`. |
| `NU_BE_C=4.0` / `RHO_BE_D=2.5` | Ravagli's G10-fitted constants. Harmless as a diagnostic; **load-bearing** once ν/ρ drive the signal (Phase 5 — see the mitigation there). |
| SABR β=0.5 fit to Malz pillars | Fine for 25Δ; strains at 10Δ/5Δ where this strategy lives. Fit residuals at the quoted pillars have not been checked. |
| Effective breadth | 7 USD pairs is ~2–3 independent bets — all share a dollar factor, AUD/NZD are near-duplicates. |
| No USD conversion yet | Cross-pair money sums are not legitimate. Single-pair only. |
| Bloomberg dependency | Results are only as good as the pull. |

---
---

# PART II — WHAT COMES NEXT

Phases are ordered by dependency. Each has a **deliverable**, an **acceptance
gate**, and any **open decisions**. Do not start a phase until the previous
gate has passed — the whole point of the gates is that a wrong answer early
invalidates everything built on top of it.

---

## PHASE 2 (remaining) — run the break-even study

**Status:** the machinery is built and offline-tested. It has not been run
against real data. **This is the immediate next action.**

### Why it matters

The strategy premise is that FX vol surfaces embed a harvestable premium in the
wings and the skew. Phase 1 can measure that premium. But a rolled wing-selling
book pays the wing spread over and over, so the premium is only real if:

```
gross premium harvested  >  cost of repeatedly putting the trade on
```

### What to run

```bash
python run/breakeven_study.py        # uncomment the test block first
```

Four studies, in order of importance:

1. **Headline sweep.** 25Δ 1M strangle rolled every 5 days, sweeping
   `OptionCostModel.scale` from 0 to 3. Find where net P&L crosses zero.
2. **Wing comparison** (35Δ / 25Δ / 15Δ / 10Δ). The central design question:
   the 10Δ is richer in convexity terms but costs multiples more to trade.
3. **Roll cadence** (5 / 10 / 21 days). Turnover is the other half of the cost
   equation.
4. **Hold-to-expiry reference.** One entry cost, no roll cost, expiry settles
   free — the cheapest possible version of the trade.

Build the dataset **once** and pass it via `dataset=` to every sweep, or each
call re-pulls Bloomberg.

### The gate — how to read the answer

| Break-even cost scale | Verdict |
|---|---|
| **< 1.0** | Below realistic costs. The premise does not survive execution as structured. **Do not build Phases 3–7.** Try wider wings, longer tenors, slower rolls, hold-to-expiry. |
| **1.0 – 2.0** | Marginal. This is a cost-execution problem before it is an alpha problem; roll cadence (Phase 6) and sizing efficiency (Phase 3) will dominate any signal. |
| **> 2.0** | Comfortable headroom. Proceed to Phase 3. |
| **NaN** | Negative even with free options — the problem is not cost, this structure did not harvest a premium over that window at all. Check whether `volga_be` is negative (short convexity into a vol-of-vol expansion). |

The break-even **scale** is deliberately the output rather than a P&L number,
because it is robust to the fact that the absolute spreads are assumptions. It
says "costs must be below X", and X can be judged against real execution
without trusting the defaults.

### What the wing comparison decides

**If break-even falls as you go further out the wing, the extra convexity
premium is not paying for the extra spread, and the sweet spot is nearer 25Δ
than 10Δ.** That single result sets the strike range the sizer should solve
over.

The sizer (§3.10) was built before this ran, so it asks the same question a
second way: `p3_test2_wing_comparison` solves one fixed risk target against
each wing slice and compares them on **cost**. The two are complementary — the
break-even study asks "does a rolled book clear its costs over a real window",
the sizer asks "which strikes deliver this risk most cheaply today". Run both;
if they disagree, the disagreement is informative. Note the sizer's version is
**blocked on the premium-floor decision** (§8), since its whole output is the
cost column.

### Also in Phase 2, but secondary

- **`reporting/` — USD conversion and the scorecard.** Needed before any
  cross-pair number is legitimate. Port the metric *definitions* from the old
  `reporting.py`; replace the wide-column plumbing with groupbys over
  `RunResult.positions`. Add the per-greek ratios as first-class metrics.
- **Record Pass 2 of the reconciliation.** It quantifies how far the old `_be`
  figures were off once the ν/ρ look-ahead is removed. Since those buckets are
  the objective function, that delta determines whether any wing/skew richness
  conclusion already drawn from the old stack still holds.

---

## PHASE 3 — the greek-target sizer — **BUILT, GATE NOT RUN**

**Built:** `strategy/sizer.py` (§3.10) and `strategy/roller.py` (§3.11).
Verified end-to-end against an offline stub snapshot. **Not yet run against
Bloomberg, and the drift gate below has not been evaluated.**

Built ahead of Phase 2b's gate deliberately: the sizer's machinery does not
depend on the break-even answer, only the choice of strike menu does. Pin the
menu with `allow_deltas` until Phase 2b and the premium-floor question (§8) are
settled, so the structure is your choice rather than the cost model's.

### What was delivered against the original four deliverables

1. **The solve** — delivered, but as a **linear program** rather than least
   squares. L1 (minimum spread paid) is the right objective and it makes the
   answer sparse for free: an LP basic solution has at most (number of
   constraints) non-zero legs. Two earlier attempts and why they failed are
   recorded in §3.10 — worth reading before touching the solver.
2. **Risk normalisation** — delivered, denominated in **base-ccy P&L per
   one-sigma move over `horizon_days`** rather than an abstract unit index, so
   a target is a number you can reason about. Formulas in §3.10.
3. **Structure builders** — delivered: `vega_neutral_fly`, `risk_reversal`,
   `vega_sleeve`, `gamma_sleeve`. Two findings came out of building them (the
   vanna sleeve needs an extra ATM leg to de-contaminate; `gamma_sleeve` cannot
   use bucketed vega) — see §3.10.
4. **Portfolio-level budget** — **NOT delivered. Moved to Phase 5/6.** It
   allocates across pairs against a covariance of the *signals*, and there are
   no signals yet to have a covariance of. Nothing in the sizer blocks it; it is
   a layer above `solve()`.

### Added beyond the original spec

- **Bucketed targets** (`by_tenor` vs `net`). Not in the original spec and it
  should have been: a flat greek vector summed across expiries hides a
  term-structure position behind a net zero. See §3.10.
- **Two guards** (`require_cond`, `require_leverage`). The leverage guard is the
  one that fires, and it catches a failure mode conditioning cannot see.
- **`current=` and the roller** (§3.11), which turns the sizer from a trade
  generator into a book-level controller. This is what Phase 4 plugs into.

### The gate — still outstanding

Ask the sizer for a **pure-volga target** and confirm that realised book vega
and vanna stay near zero over a full run, not just on day one. `p3_test6_gate`
in `test.py` does exactly this: hold a sized fly, then report `net_vega_1vp /
net_volga_1vp` and `net_vanna_1pct_1vp / net_volga_1vp` day by day as
**ratios**, so the number is scale-free.

The result is not a pass/fail checkbox — it is **Phase 4's specification**:

| peak \|vega/volga\| | Read |
|---|---|
| **< ~0.1** | the fly is more self-neutralising than expected; Phase 4's re-hedge can be infrequent and cheap |
| **> ~0.2 quickly** | Phase 4 is the binding problem. The *drift rate* sets the re-hedge frequency, which sets the spread bill, which feeds back into Phase 2's break-even |

Write the number down either way.

### Open decisions

**Separate volga and vanna sleeves, or one combined book?** Current lean
unchanged: **separate**. Building the sizer strengthened the argument — the
vanna sleeve needs a specific extra ATM leg to null residual volga (§3.10),
which is only diagnosable if the sleeves are apart. Execution can always be
netted afterwards; a combined book cannot be taken apart. Costs nothing
structurally, since `solve()` emits `LegRequest`s that a later netting layer can
merge before they hit the Book.

**`top_up` or `replace` for rolling?** `top_up` is the default and is cheaper
(it only ever trades the increment). `replace` is easier to reason about and
pays the exit spread every roll. `p3_test7b_compare_modes` decides it on
measured `option_tc` and `peak_net_volga`. Unresolved until that runs.

**The deadband (`min_trade`).** Trades off spread paid against drift from
target between rolls. Its optimum depends on the drift rate from the gate above,
so the two tests are meant to be read together.

---

## PHASE 4 — the vega hedge

**Depends on:** Phase 3's gate, which is what *specifies* this phase — the drift
rate from `p3_test6_gate` sets how often the re-hedge must fire.

**Most of the mechanism already exists.** A vega re-hedge is exactly
`solve(snap, tenors, target_with_vega_zero, current=book)` — read the drift,
trade the increment that flattens it. That is §3.11's `top_up` path with a
different target, which is why it was built that way rather than as
close-and-restrike. What remains is the *policy*: which instrument, what
deadband, what cadence.

### Goal

Hold **pure volga**. A vega-neutral butterfly is neutral only on the day it is
struck; it drifts immediately as spot moves and the smile reshapes. Nothing in
the stack currently re-neutralises it.

### Deliverables

`engine/hedging.py` — generalise the hedge from a scalar delta fraction to a
**greek-vector hedge with instruments**:

- spot → hedges delta (exists)
- rolled ATM straddle → hedges vega (new)

The sizing half is done; what is new is the hedge *policy* — a deadband on
residual vega (by analogy with `residual_delta`, §3.8, and with the roller's
`min_trade`), and a decision on whether the hedge instrument is the ATM straddle
or whether the re-hedge is folded into the sleeve's own roll. Folding it in is
cheaper (one trade, not two) and muddier to attribute; the `sleeve='hedge'` tag
exists so the separate version stays diagnosable.

The cost side is **already done**: `book/costs.py` exposes one `charge`
interface and the Book routes every option trade through it, so the vega hedge
gets charged automatically with no changes to the cost model. That was a
deliberate Phase 2 decision.

### Gate

With the vega hedge on, `vega_pnl` should collapse toward noise and
`volga_pnl_be` should become the dominant bucket in the convexity sleeve. If it
does not, the sleeve is not isolating what you think it is.

### The thing to watch

The vega hedge is an **option** hedge, so it pays the vol-point spread on every
rebalance, and it is exposed to exactly the wing-versus-ATM relative move the
strategy is trying to isolate. Re-run `run/breakeven_study.py` with it enabled —
it may materially move the break-even.

---

## PHASE 5 — features and signals

**Depends on:** nothing in Phases 3–4 strictly, so it can run in parallel. It
depends on Phase 2 only in the sense that a failed cost gate makes it moot.

### Goal

Extend the validated IV−RV machinery one moment up, to the second and third
moments of the surface.

### The core idea

The old stack's signals were time-series percentiles of one scalar (IV, IV−RV,
cross-currency spread). The natural extension to the wings is the same idea
applied to convexity and skew:

| Level | Signal |
|---|---|
| ATM VRP | `IV − RV` (exists) |
| **Convexity** | **ν_implied − ν_realised** |
| **Skew** | **ρ_implied − ρ_realised** |

**The implied side already exists in closed form** — `get_smile_nu_rho` reads it
straight off quoted BF25/RR25/ATM. The realised side is cheap to build from data
already pulled:

- realised ν = stdev of log-changes in the trailing constant-maturity ATM vol
  series (`snapshot.atm_history`)
- realised ρ = `corr(dS/S, dσ)` over the trailing window

### Deliverables

`strategy/features.py`, `strategy/fair_value.py`, `strategy/signals.py`:

- Realised ν and ρ estimators, taking a snapshot and a lookback (so
  point-in-time discipline is structural).
- Implied-minus-realised z-scores per pair × tenor.
- A **cross-sectional demean across pairs** to strip the common dollar factor.

### Critical mitigation

**Z-score `BF25/(σ√τ)` and `RR25/σ` directly rather than `nu_BE`/`rho_BE`.** Then
the ad-hoc constants `NU_BE_C=4.0` and `RHO_BE_D=2.5` are a pure scale factor
and **cancel out of the z-score**. Those constants are harmless as a diagnostic
but become load-bearing the moment they drive a signal, and they are G10-fitted
with no validation on crosses.

### Gate

**Signals only, no trading.** Check that the richness series is stationary, not
autocorrelated to death, and that implied-minus-realised ν has the positive mean
being claimed as the premium. **If the premium is not visible in the signal
series, no amount of trading machinery will find it.**

### Also here — the portfolio budget, inherited from Phase 3

Deliverable 4 of the original Phase 3 spec lands here, because it needs signals
to exist: **cap total book volga and vanna, and allocate across pairs against a
covariance of the SIGNALS, not of the P&L.** Realised P&L across FX pairs is
dominated by the common dollar factor, so a P&L covariance says every pair is
the same trade and concentrates the book into one. Signal covariance asks the
question that matters — is EURUSD wing-richness telling you the same thing as
USDJPY wing-richness?

Nothing in `strategy/sizer.py` blocks this; it is a layer above `solve()` that
scales each pair's target before the solve runs.

**Add crosses** (EURGBP, AUDNZD, EURJPY, CADJPY). Cross-sectional surface RV
lives or dies on breadth, and 7 USD pairs is effectively 2–3 independent bets.
This forces the `fx_usd` conversion path, which has never been exercised because
every current pair has a USD leg — so the Phase 2 reporting work is a
prerequisite.

---

## PHASE 6 — continuous target sizing

**Depends on:** Phases 3 and 5.

### Goal

Replace binary 0/1 entry with a continuous target risk level. This is the
mechanical change that breaks the old Layer 3/4 contract: a `pd.Series` of 0/1
dates cannot express *"I want 0.4× my maximum volga in EURUSD 3M today."*

### Deliverables

`strategy/target.py` and `engine/rebalance.py`:

- richness z-score → target greek vector, clipped and risk-budgeted
- a roll policy
- a rebalance rule: how far from target before you trade (a deadband, because
  every adjustment pays the spread)

**Two of the three already exist in skeleton.** `GreekTargetRoller.target_fn`
(§3.11) takes a snapshot and returns a `GreekTarget`, so "z-score → target" is a
one-line substitution into a class that does not otherwise change. The roll
policy is `roll_days`, and the deadband is `min_trade`. What Phase 6 adds is the
z-score itself (from Phase 5), the clipping and budgeting, and calibration of
the deadband against measured turnover.

### Gate

**Turnover and cost per unit of carried greek.** This is where a good-looking
strategy dies. A continuously-resized book can churn its entire edge into the
bid-offer.

### Open decision

**Fixed-calendar rolling or signal-driven?** Fixed roll is predictable and
cost-boundable; signal-driven is more selective but turnover becomes an emergent
property that cannot be reasoned about in advance. Current lean: **fixed roll,
continuous size.** That separates "when do I trade" (mechanical) from "how much
risk do I want" (the signal), which is what makes the results interpretable.

---

## PHASE 7 — risk

**Depends on:** Phase 6, though the scenario engine can be built any time.

### Goal

Greek-native risk management, replacing single-path drawdown statistics.

### Deliverables

`risk/scenarios.py`:

- Reprice the whole book across a grid of spot × ATM level × skew shift ×
  convexity shift. The pricer and surface builder already exist, so this is a
  small engine.
- This is the greek-native replacement for max drawdown, which rests on a single
  order statistic from one realised path (the old stack's trap #11) and is
  nearly useless for comparing sleeves.

`risk/limits.py`:

- Greek caps as **constraints inside the sizer**, not just reported columns.

`reporting/` additions:

- Per-greek risk-adjusted metrics as the primary scorecard.
- `recon_resid` distribution as an explicit gap-risk metric.
- CVaR plus explicit jump scenarios — short wings has a fat left tail that daily
  volatility statistics will not show.

---

## Summary of the critical path

```
[DONE]  Phase 1   engine rewrite            gate: reconcile vs old engine      PASSED
[DONE]  Phase 2a  option cost model         gate: 30 offline checks            PASSED
[DONE]  --------  rates + pricing hardening gate: fwd vs market < 0.1 pip      PASSED
[BUILT] Phase 3   greek-target sizer        gate: pure-volga target holds      ← test6, UNRUN
[BUILT] Phase 3b  greek-target roller       gate: target held, not accrued     ← test7b, UNRUN
[NEXT]  Phase 2b  break-even study          gate: break-even scale > 1.0       ← RUN THIS
        Phase 2c  reporting / USD           gate: cross-pair sums legitimate
        Phase 4   vega hedge                gate: vega_pnl -> noise
        Phase 5   ν/ρ signals               gate: premium visible in the signal
                  + portfolio budget        (moved from Phase 3)
        Phase 6   continuous sizing         gate: turnover cost acceptable
        Phase 7   scenario risk             gate: —
```

Phase 3 sits above Phase 2b in the list because it was built out of dependency
order — deliberately, since the sizer's machinery does not depend on the
break-even answer. That does **not** discharge the gate.

### The three numbers to get, in this order

1. **Break-even cost scale** (`run/breakeven_study.py`). The one number that
   gates everything. Below 1.0 across wings and tenors and the correct next move
   is not Phase 4 — it is restructuring the trade.
2. **The premium-floor decision** (§8). It biases every structure the sizer
   chooses and every cost comparison it prints, including the wing comparison
   that is supposed to inform decision 1.
3. **Peak `|vega/volga|` drift** (`p3_test6_gate`). Phase 3's gate, and Phase
   4's specification.

Nothing above the line "trust these results" should be claimed until all three
exist.
