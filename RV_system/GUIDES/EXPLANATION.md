# Systematic_RV — How It Actually Works

A bottom-up mechanical walkthrough of every part of the codebase, in dependency
order. Read it beside the source.

**This is not `PROJECT_STATE.md`.** That document explains *what was built and
why those design decisions were made*, plus the roadmap. This one explains
*what the code does, line by line, layer by layer* — so you can open any file
and know exactly what it is for, what it reads, what it returns, and what would
break if you changed it.

---

## Contents

| Part | What it covers |
|---|---|
| **0** | Orientation — the one idea, the dependency map |
| **1** | `core/` — conventions, calendar, pricer, option, greeks, vol surface |
| **2** | `market/` — feeds, dataset, snapshot |
| **3** | `book/` — position, costs, attribution, book |
| **4** | `engine/` — the date loop |
| **5** | `strategy/sizer.py` — risk-space sizing |
| **6** | `strategy/roller.py` — maintaining a risk target |
| **7** | `run/` and `recon/` — studies and validation harnesses |
| **8** | One day in the life — full data flow |
| **9** | The traps, consolidated |
| **10** | Measured facts worth remembering |

---
---

# PART 0 — ORIENTATION

## 0.1 The one idea

Everything in this repo exists to answer one question honestly:

> **FX vol surfaces price convexity (the butterfly) and skew (the risk
> reversal). Is the price persistently above what gets realised, and does the
> gap survive the bid-offer you pay to harvest it?**

That question forces three structural commitments, and if you understand these
three you understand the architecture:

**1. The loop is inverted.**

```
OLD:  for trade in trades:  for date in dates:   ...
NEW:  for date in dates:    for position in book: ...
```

The old ordering meant every trade lived in its own universe: one expiry, its
own delta hedge, no way to roll. The new ordering unlocks multi-expiry books,
rolling, netted hedging, continuous resizing, and cross-sleeve attribution.
None of those were missing *finance* — they were blocked by loop ordering.

**2. Risk is one unit, applied once.**

Every greek in the system is **base-currency P&L for one standardised move**,
with notional and direction already applied. That means greeks add across legs,
tenors, pairs and sleeves without any further thought — which is what makes a
book-level risk budget and a greek-target sizer possible at all.

**3. You specify RISK, not notional.**

`10mm of a 25-delta strangle` means a different amount of risk at 1M than at
3M, and different again in a 5-vol pair than a 12-vol pair. So instead you say
*"hold −3,000 of volga, vega and vanna flat"* and a linear program returns the
legs that produce it at minimum spread.

## 0.2 The dependency map

Arrows point *downward* — nothing ever imports upward. `core/` never imports
`market/`, which is why `core/` has no Bloomberg dependency and can be unit
tested offline.

```
                    test.py  /  test_full.py          <- you drive from here
                              |
              +---------------+---------------+
              |                               |
      strategy/roller.py               run/breakeven_study.py
              |                               |
      strategy/sizer.py                       |
              |                               |
              +---------------+---------------+
                              |
                        engine/loop.py                <- THE date loop
                              |
              +---------------+---------------+
              |               |               |
        book/book.py   book/attribution   book/costs
              |               |
        book/position.py -----+
                              |
                       market/snapshot.py              <- point-in-time gate
                              |
                       market/dataset.py               <- all the tables
                              |
                       market/feeds.py                 <- ONLY Bloomberg file
                              |
    +---------+---------+-----+-----+---------+
    |         |         |           |         |
 core/     core/      core/       core/     core/
convent-  calendar   pricer      option    greeks
 ions                              |         |
    +---------+---------+----------+---------+
                              |
                     core/vol_surface.py
```

## 0.3 File-by-file, one line each

| File | Lines | One-line purpose |
|---|---|---|
| `core/conventions.py` | 225 | Tenor/delta pillars, per-currency calendars, day-count bases, ticker maps |
| `core/calendar.py` | 276 | FX date arithmetic — spot date, expiry from tenor, business-day rolling |
| `core/pricer.py` | 341 | Garman-Kohlhagen closed forms with three separate time clocks |
| `core/option.py` | 381 | `FXOption` — one contract, repriceable; strike solvers |
| `core/greeks.py` | 389 | `GreekVector` — THE risk unit; Taylor P&L decomposition |
| `core/vol_surface.py` | 484 | Quote→smile grid, SABR fit, term interpolation, arb removal |
| `market/feeds.py` | 170 | The only module that talks to Bloomberg |
| `market/dataset.py` | 778 | All tables + every read: rates, ATM vol, smile vol, ν/ρ |
| `market/snapshot.py` | 395 | Point-in-time view of one pair. The look-ahead firewall |
| `book/position.py` | 401 | `LegRequest` (what you ask) vs `Position` (what you got) |
| `book/costs.py` | 310 | Parametric bid-offer. Every option trade routes through it |
| `book/attribution.py` | 403 | Daily marking, Taylor buckets, the `_be` premium buckets |
| `book/book.py` | 401 | Holds positions + marks + the netted spot hedge |
| `engine/loop.py` | 477 | The date loop and its six-step ordering |
| `strategy/sizer.py` | 1112 | Solve legs from a greek target at minimum spread |
| `strategy/roller.py` | 181 | Maintain that target across rolls |
| `run/breakeven_study.py` | 294 | Sweep cost scale; find where the premium dies |
| `recon/reconcile.py` | 429 | Phase 1 gate — bit-compare against the old engine |
| `recon/smoke_test.py` | 355 | Synthetic end-to-end, no Bloomberg needed |
| `recon/ovml_lineup.py` | 422 | Describe trades in plain English, check vs Bloomberg OVML |

---
---

# PART 1 — `core/`: THE FOUNDATIONS

No market data, no I/O. Pure functions and value objects.

## 1.1 `core/conventions.py` — the constants everything agrees on

This file exists so that `core/` never has to import `market/`. It holds the
facts about FX markets that are not opinions.

**Tenor and delta grids**

```python
TENOR_DAYS   = {'ON':1,'1W':7,'2W':14,'3W':21,'1M':30,'2M':60,
                '3M':91,'6M':182,'9M':273,'1Y':365}
DELTA_POINTS = [35, 25, 15, 10, 5]
```

`TENOR_DAYS` values are *nominal* — used only for parsing and ordering. The
actual day counts come from `dataset._compute_pillar_days`, which walks the real
calendar, so a 1M pillar in February is 28 days and in March is 31.

**`FWD_YIELD_TENORS` is deliberately a different list** — no `'ON'`, no `'3W'`:

- `ON` is excluded because giving one currency an overnight node the other lacks
  reintroduces a covered-interest-parity error on sub-1W options. Both legs clamp
  to their 1W quote below that, which keeps the *differential* exact.
- `3W` is excluded because `USOSFR3Z` exists but `XXXI3W` does not, for any
  currency. It used to be in the list and was silently swallowed by an
  `except KeyError: continue`.

**Day-count bases** — `MM_BASIS` maps currency → 360 or 365. GBP/AUD/NZD/CAD are
365, everything else 360. This matters: assuming 360 everywhere leaves those four
adrift by 3–6bp at 3M. With the map, every currency closes to under 0.15bp on a
CIP check.

**Ticker maps** — `USD_OIS_TICKER`, `CCY_FWD_YIELD_PREFIX`, `FWD_YIELD_SUFFIX`.
Note the 1Y point is quoted `XXXI12M`, not `XXXI1Y`; that is why the suffix map
exists separately from the tenor string.

**`FXCalendar` and `fx_calendar(pair)`** — the piece most people skip and then
get wrong. A pair has *three* relevant calendars:

| field | meaning |
|---|---|
| `cal_trade` | good for BOTH legs — picks the trade/horizon date |
| `cal_interim` | good for the NON-USD legs only — the T+2 interim hop |
| `cal_settle` | ccy1 + ccy2 + USD — final spot date and delivery validity |

Plus `spot_days` (T+1 for CAD, T+2 otherwise) and `special_latam`. `fx_calendar`
is cached, so calling it in a tight loop is free.

## 1.2 `core/calendar.py` — FX date arithmetic

Two functions matter; the rest support them.

**`spot_from_horizon(horizon, fxc)`** — Clark §1.4. For T+2:
1. Advance one good business day to the *interim* date, skipping only the
   non-USD legs' holidays (USD holidays are allowed on the interim — unless a
   special LatAm leg forces them to be skipped too).
2. Advance one more good business day to the final spot date, now skipping
   ccy1, ccy2 **and** USD.

**`add_tenor(horizon, tenor, fxc)`** — the option **expiry** for a tenor. For
months and years it routes `today → spot → delivery → expiry`; for days and
weeks it counts calendar days directly. This is the function that makes `'1M'`
mean a real date. Both the sizer and the engine call it, which is why their
expiries always agree.

Supporting: `is_business_day`, `preceding_business_day`, `next_business_day`,
`modified_following`, `_parse_tenor` (accepts `ND`/`NW`/`NM`/`NY`).

## 1.3 `core/pricer.py` — Garman-Kohlhagen, with three clocks

The unusual part, and it is correct rather than pedantic: **an FX option runs on
three different time windows.**

```
tau_var   today      -> expiry      how much variance accumulates
tau_fwd   spot date  -> delivery    over which the rate differential accrues
tau_disc  today      -> delivery    over which the payoff is discounted
```

They differ by the T+2 settlement lag at each end. Usually the two lags cancel
and all three land within a day of each other. They come apart whenever a
holiday cluster stretches one lag — a USDJPY 1M struck 21-Aug-2026 runs
**28 / 31 / 35 days**, because Tokyo's Silver Week pushes the expiry→delivery lag
out to seven calendar days.

Every function takes `T`, which may be a plain float (all three equal — the old
behaviour, unchanged) **or** an `OptionTime(var, fwd, disc)`. So float call sites
keep returning exactly what they always did.

Written in forward terms:

```
F  = S * exp((r_d - r_f) * tau_fwd)
DF = exp(-r_d * tau_disc)
d1 = (ln(F/K) + sigma^2 * tau_var / 2) / (sigma * sqrt(tau_var))
d2 = d1 - sigma * sqrt(tau_var)
call = DF * (F * N(d1) - K * N(d2))
```

When the three coincide, `DF*F` collapses to `S*exp(-r_f*T)` and every formula
reduces algebraically to textbook spot-based GK.

Exports: `bs_price`, `delta`, `delta_premium_adjusted`, `gamma`, `gamma_trader`,
`gamma_premium_adjusted`, `vega`, `volga`, `vanna`, `vanna_premium_adjusted`,
`theta`, `rho_d`, `rho_f`, `rho`, `fwd_price`, `df_domestic`.

> ⚠️ `gamma_premium_adjusted` is Wystup's `Gamma_pa`, which is a **different
> quantity** from `d²V/dS²`. `core/greeks.py` uses the identity, not that
> function. Do not substitute one for the other.

## 1.4 `core/option.py` — one contract, repriced forever

```python
@dataclass
class FXOption:
    pair: str; K: float; expiry: date; option_type: str = 'call'
```

**Rates and vols are never stored.** They are passed at call time, so one
`FXOption` can be repriced freely at any point in a backtest. Only the contract
terms are fixed at construction. (`S0`, `r_d`, `r_f`, `sigma0` exist as inception
snapshots for reporting; nothing in the pricing path reads them.)

Key methods:

| method | returns |
|---|---|
| `delivery()` | the settlement date, via the pair's calendar |
| `time_basis(current_date)` | the `OptionTime` triple |
| `price_domestic(...)` | `P` — the GK price in quote ccy |
| `value_base(...)` | **`V = P / S`** — the base-ccy value of one unit of notional |
| `intrinsic_base(S)` | settlement value at expiry |
| `spot_delta`, `call_delta_equivalent` | delta conventions |
| `base_ccy_partials(...)` | the six raw partials of `V` |
| `greeks(...)` | a `GreekVector` |

**`value_base` is the pivot of the whole accounting scheme.** The engine accounts
in base (foreign) currency, so the value of one unit of notional is `V = P/S`.
Differentiating `V` rather than `P` gives the clean rule in §1.5.

Module-level solvers:

- `atm_forward_strike(S, r_d, r_f, expiry, today, pair)` — the ATMF reference.
- `strike_from_delta(...)` — `brentq` root-find, bracketed to roughly
  `[0.85S, 1.20S]`. **A 5-delta wing in a high-vol pair can fall outside that
  bracket and fail to strike** — the sizer catches this and *names* the dropped
  candidate rather than silently degrading the menu.

## 1.5 `core/greeks.py` — the single risk unit

**This is the most important file in the repo.** In the predecessor stack,
`greeks_foreign()` returned a dict with *five different scaling conventions*
mixed together, and every consumer re-scaled by hand — `* 0.01 * 0.01 /
prev_spot` appeared at three separate call sites. Survivable when greeks are a
display item; fatal when you size positions on volga.

Here the scaling is applied **once, at construction**:

```python
SPOT_MOVE = 0.01     # 1% of spot
VOL_MOVE  = 0.01     # 1 vol point, decimal
TIME_MOVE = 1.0      # 1 calendar day
```

### The fields

| field | meaning |
|---|---|
| `spot_1pct` | \$ for a +1% spot move (1st order) |
| `gamma_1pct` | \$ for a +1% spot move (2nd order, **includes the ½**) |
| `vega_1vp` | \$ per +1 vol point (1st order) |
| `volga_1vp` | \$ per +1 vol point (2nd order, **includes the ½**) |
| `vanna_1pct_1vp` | \$ for +1% spot **and** +1 vol pt together |
| `theta_1d` | \$ for one calendar day |
| `delta_hedge` | base-ccy spot NOTIONAL to trade to flatten — a **quantity**, not a P&L |
| `spot` | the spot it was struck at (carried to detect cross-pair aggregation) |

### The base-currency derivatives

Differentiating `V = P/S`:

```
dV/dS      =  D_pa / S                       (D_pa = premium-adjusted delta)
d2V/dS2    =  G_raw / S  -  2 * D_pa / S^2
dV/dsig    =  vega_raw / S
d2V/dsig2  =  volga_raw / S
d2V/dSdsig =  vanna_pa / S
dV/dt      =  theta_raw / S
```

Derivatives w.r.t. **spot** pick up a premium-adjustment correction because `S`
sits in the denominator. Pure **vol** and **time** derivatives just divide by `S`.

That `−2·D_pa/S²` term is not cosmetic. It is why gamma and vega inside one
expiry are only ~90% collinear in this stack rather than exactly collinear — see
§5.8.

### `from_partials` — where standardisation happens

```python
q  = notional * direction
ds = SPOT_MOVE * spot        # absolute spot move equal to +1%
dv = VOL_MOVE                # absolute vol move equal to +1 vol pt

spot_1pct      = q * dV_dS      * ds
gamma_1pct     = q * 0.5 * d2V_dS2 * ds * ds
vega_1vp       = q * dV_dsig    * dv
volga_1vp      = q * 0.5 * d2V_dsig2 * dv * dv
vanna_1pct_1vp = q * d2V_dSdsig * ds * dv
theta_1d       = q * dV_dt      * TIME_MOVE / 365.0
delta_hedge    = q * dV_dS      * spot
```

### Arithmetic

`__add__`, `__mul__`, `__neg__`, `__sub__`, `GreekVector.total()`. The subtlety:
**when two vectors have different spots you are aggregating across pairs, so
`delta_hedge` is blanked to `NaN`** rather than silently returning a meaningless
sum. The P&L fields still add (they are all money), but see the caveat in
`Book.greeks` — they are each in their *own* pair's base currency, so a
multi-pair sum needs USD conversion first.

### `as_trader_units()` — for READING ONLY

Conventional desk numbers: delta in notional, gamma as delta-per-1%, vanna as
delta-per-vol-point, volga as vega-change-per-vol-point. **Gamma and vanna come
out in notional while vega/volga/theta come out in money** — that mixture is
precisely the trap this module removes from the P&L view. Never sum these.

### `taylor_pnl(g, dS, dsigma, dt_days, prev_spot)`

Decompose a realised move. Because every greek is already
money-per-standard-move, each bucket is the greek times the move in **standard
units**, raised to the order of the term. No scaling constants appear anywhere:

```
u = (dS / prev_spot) / SPOT_MOVE     spot move in "1% units"
w = dsigma / VOL_MOVE                vol move in "vol point units"

theta = theta_1d       * dt_days
gamma = gamma_1pct     * u^2
vega  = vega_1vp       * w
volga = volga_1vp      * w^2
vanna = vanna_1pct_1vp * u * w
```

**Remember `u` and `w`. They come back in §5.4 as the definition of "one
sigma".**

## 1.6 `core/vol_surface.py` — quotes to a usable smile

### Step 1 — `build_vol_grid(atm, rr, bf)`

Market convention:
```
RR_d = vol(d-delta call) - vol(d-delta put)
BF_d = 0.5*(vol(d call) + vol(d put)) - ATM
```
Solved:
```
call_vol = ATM + BF_d + 0.5 * RR_d
put_vol  = ATM + BF_d - 0.5 * RR_d
```
Returns a Series keyed `'ATM'`, `'C25'`, `'P25'`, …

### Step 2 — `_fit_sabr_to_grid(vol_grid, F, T, r_f)`

1. **Analytically invert each pillar delta to a strike:**
   `d1 = N⁻¹(dce · e^{r_f T})`, then `K = F · exp(−d1·σ·√T + ½σ²T)`.
2. Sort ascending (QuantLib requires it).
3. Calibrate `alpha / nu / rho` with **β = 0.5 fixed** via `ql.SABRInterpolation`.

Returns `(sabr, ks, vs, atm_vol)`; `sabr` is `None` on failure or if fewer than
three pillars exist.

The `DELTA_GRID_MAP` puts everything on a **call-delta axis** (0 = deep OTM call,
0.50 = ATM, 1.0 = deep ITM call) so the smile is monotonic and continuous through
the money.

### Step 3 — `get_sabr_vol_at_K(...)`

Evaluates the fitted smile at a fixed `K`. **No delta iteration** — `K` is passed
directly, so there is no circularity. Extrapolation is clipped to 25% beyond the
outermost pillar to stop blow-ups.

### Step 3b — `get_sabr_params(...)` *(added recently)*

Returns `(nu, rho)` from the **same** fit instead of evaluating at a strike. See
§2.3 for why this replaced a closed-form approximation, and §10 for the measured
comparison.

### Step 4 — ATM term interpolation, weekend-weighted

`interpolate_atm_vol` implements Wystup §1.3.4. Standard variance interpolation
treats every calendar day equally, which overstates variance accrued over
weekends. So:

- weekend day weight `0.15`, holiday weight `0.25`
- business-day weight `α` solved so total weighted variance between pillars
  *exactly* reproduces the pillar vols
- forward variance between pillars: `σ_f² = (σ²(t2)·t2 − σ²(t1)·t1)/(t2−t1)`

### Step 5 — calendar-arbitrage removal (PAVA)

`_enforce_monotonic_variance` projects cumulative variance (`days · vol²`) onto
the non-decreasing cone via Pool-Adjacent-Violators (isotonic regression, O(n)).
This removes calendar-spread arbitrage from an inverted quoted term structure.
**Pillars not involved in a violation are untouched** — it is the minimal
least-squares adjustment, not a smoothing.

### Step 6 — smile spread interpolation

`interpolate_spread_sqrt_t` — the *spread* (smile vol minus ATM) interpolates in
**√t**, while the *level* interpolates in **variance (t)**. Jump and skew risk
scale with √t; the vol level scales with t. Mixing the two spaces is the
industry-standard decomposition, not an inconsistency.

---
---

# PART 2 — `market/`: DATA AND THE LOOK-AHEAD FIREWALL

## 2.1 `market/feeds.py`

The **only** module that imports `xbbg`/`pdblp`. Five pulls:
`pull_dailyCCY_close`, `pull_fx_vol_surface`, `pull_sofr`,
`pull_fwd_implied_yields`, `pull_usd_ois_curve`.

Everything downstream consumes DataFrames, so the whole stack is testable
without a terminal (see `recon/smoke_test.py`).

## 2.2 `market/dataset.py` — the tables and every read

### The tables

| attribute | contents |
|---|---|
| `spot` | daily close per pair |
| `vol_surface` | MultiIndex `(pair, tenor, field)` — **raw percent**, e.g. `8.5` |
| `sofr` | overnight fixing |
| `fwd_yields` | forward-implied yields per non-USD ccy |
| `usd_ois` | SOFR OIS par swap rates |
| `rate_curves` | `fwd_yields` + `usd_ois` concatenated — **one table, one code path** |

> ⚠️ `vol_surface` is stored in **percent**. Converted to decimal at the point of
> use. Reading it as decimal is a 100× error that will look like a working
> backtest.

`FXVolDataset.build(pairs, days)` is **memoised on `(pairs, days)`** in a
module-level `_DATASET_CACHE`. Two reasons beyond speed: four pulls seconds apart
could return slightly different data, and the instance carries its own caches
which are then shared across a sweep. `clear_dataset_cache()` after a refresh.

### The caches

| cache | key | why |
|---|---|---|
| `_pillar_cache` | `(pair, date)` | node days (as_of→expiry) **and** accrual days (spot→delivery) |
| `_rate_cache` | `(pair, date, t_days)` | measured 25,398 calls / 1,238 distinct = **95% duplicate** |
| `_smile_grid_cache` | `(pair, date)` | 8,647 calls / 121 distinct = **98.6% duplicate** |
| `_sabr_param_cache` | `(pair, date, tenor)` | one SABR fit per pillar per day |

All four are pure functions of their keys, so caching cannot change a number —
only how many times it is computed. Verified bit-identical; see §10.

**Why the duplication is structural, not accidental:** `snapshot.price_state`
asks for the same tenor's rates *three times* — directly, inside `smile_vol`, and
again inside `forward` — and every open position on a date shares a handful of
distinct expiries.

### Node days vs accrual days

Two different day counts, and conflating them converts a simple rate to the wrong
continuous rate:

- **node** — `as_of → expiry`. The interpolation x-axis, shared with the vol surface.
- **accrual** — pair's spot date → tenor's delivery date. The window a quoted
  money-market rate actually accrues over.

### `get_rates_for_tenor` → `_zero_rate`

Returns `(r_d, r_f)` as **continuous ACT/365** zero rates. Convention: for
`XXXYYY`, `r_d` is YYY's rate and `r_f` is XXX's. So for USDJPY, `r_d` = JPY and
`r_f` = USD.

Inside `_zero_rate`:
1. Read each pillar's quote, `.dropna()` **before** `.iloc[-1]` — the last *row*
   at or before `as_of` is not necessarily the last *quote*, because a currency's
   own holiday blanks its column while the pair still trades.
2. Convert simple → continuous:
   `r_cont = ln(1 + r·d/basis) · 365/d`. Feeding a simple rate in as if it were
   continuous is a 4.5bp error at 3.68% over 31 days.
3. Build a `ql.DiscountCurve` on the real node days and read the zero rate.

Both legs are read on the **traded pair's** calendar. What drives the forward is
the *differential* `r_d − r_f`, so both legs must sit at the same x or the
differential is contaminated.

### `get_smile_vol(pair, as_of, t_days, K, F, r_f)`

```
1. ATM level  -> weekend-weighted variance across pillars      (Wystup 1.3.4)
2. Vol spread -> SABR vol at K minus that pillar's ATM,
                 interpolated in sqrt(t) between the two
                 surrounding pillars                            (Wystup 1.3.8)
   final vol = ATM(tr) + spread(tr)
```

One documented approximation: `spread_at` passes the **target tenor's** `F` into
every pillar's fit, on the argument that the error partly cancels in the
ATM-relative spread. Fine for a spread; that is exactly why ν/ρ use a **separate,
own-forward** fit (below).

### ν and ρ — `get_smile_nu_rho`

Dispatches on `NU_RHO_SOURCE`:

**`'sabr'` (default).** `_sabr_pillar_params` fits each pillar with **its own
forward** — which is what makes it cacheable — and `_nu_rho_sabr` interpolates
across pillars:

- **ρ** in √t directly, same scheme as the smile spread.
- **ν** as `ν·√days` in √t, because ν's term structure is close to `ν ∝ 1/√τ` and
  `ν·√τ` is nearly flat (measured **0.587–0.629** across the whole USDJPY curve).
  Interpolating ν itself would cut a corner off a 1/√τ curve.
- Outside the pillar range, the nearest pillar's values are used **unchanged**,
  matching how `interpolate_atm_vol` and `get_smile_vol` clamp. Extrapolating
  inwards would reintroduce the τ→0 blow-up.

**`'closed_form'`.** The Ravagli (2024) formula:
```
nu_BE  = NU_BE_C  * sqrt(Fly25 / (tau * sigma_ATM))     NU_BE_C  = 4.0
rho_BE = RHO_BE_D * (RR25 / sigma_ATM)                  RHO_BE_D = 2.5
```
from the pillar **nearest** `t_days` (not interpolated), clipped at
`NU_BE_MAX = 5.0`. Kept as the reconciliation reference and as a fit-free
fallback when a calibration fails. `MIN_PILLAR_DAYS = 7` excludes `'ON'` on both
paths.

Set per instance: `ds.nu_rho_source = 'closed_form'`.

## 2.3 `market/snapshot.py` — the point-in-time firewall

```python
snap = MarketSnapshot.at(ds, 'USDJPY', some_date)
```

**Everything a snapshot can see is truncated at `as_of` structurally.** There is
no way to accidentally read tomorrow's data through it. This is the single most
important property in the stack, because a look-ahead bug does not announce
itself — it just makes results better.

| method | what it gives |
|---|---|
| `spot` | spot on that date |
| `rates(t_days)` | `(r_d, r_f)` |
| `atm_vol(t_days)` | weekend-weighted interpolated ATM |
| `forward(t_days)` | outright forward to **delivery**, not expiry |
| `smile_vol(K, t_days)` | SABR vol at a **fixed** strike |
| `nu_rho(t_days)` | vol-of-vol and spot/vol correlation |
| `solve_strike_and_vol(delta, type, expiry)` | the fixed point |
| `price_state(K, expiry)` | `{S, sigma, r_d, r_f, t_days}` in one call |
| `spot_history(n)` / `quote_history(tenor, field, n)` / `atm_history(tenor, n)` | trailing series, hard-truncated |

### The strike/vol fixed point

`solve_strike_and_vol` resolves the circularity **strike → smile vol → delta →
strike**: start from the ATM vol, solve the strike giving `target_delta` under it,
read the smile vol at that strike, re-solve, repeat. Converges in 2–3 passes for
G10; `STRIKE_SOLVE_ITERS = 5` matches the old engine so Phase 1 reconciles
exactly. `legacy_strike_halfstep` reproduces the old bootstrap.

`smile_vol` does **not** have this circularity — strikes are locked at trade
entry, so `K` is passed directly.

### `quote_history` vs `atm_history`

Both are **constant-maturity pillar series**, which is what a realised-ν or
realised-ρ estimator wants: a single option's own vol shortens every day and so
mixes term-structure roll into the measurement; a pillar series does not.

### `business_dates(dataset, pair, start, end)`

The date grid the engine walks: every date the pair has an observed spot,
optionally clipped. Driving off observed spot rather than a synthetic calendar
means the loop can never ask for a date the data lacks, and weekends/holidays
drop out naturally. **Gaps are still gaps** — the engine must scale theta and
carry by the actual day count between consecutive rows, not by 1.

---
---

# PART 3 — `book/`: POSITIONS, COSTS, MARKING

## 3.1 `book/position.py` — two objects, deliberately separate

```
LegRequest : what you ASK for -- "a 25-delta put, 1M, short, 10mm".
             Delta-space, notional-space. No market data in it.

Position   : what you GOT -- a struck strike, the smile vol you were
             struck at, the premium, and an id.
```

That mirrors **order → fill**. A strategy can express intent without touching a
snapshot; the sizer emits `LegRequest`s and the engine fills them.

`open_position(req, snap, expiry, cost_model)` is **the only place market data
enters** a position. It resolves the strike (or uses ATM-forward if
`req.atm`), reads the smile vol, builds the `FXOption`, computes the entry
premium, and charges the spread.

A `Position` is long-lived, with an identity (`pos_id`), carried across the daily
loop and closed by expiry, roll, or risk decision. That is the whole unlock over
the old `LegSpec`, which was consumed once inside a single call.

Key methods: `value_base(snap)`, `settle_base(spot)`, `greeks(snap)`,
`greeks_from_state(st, on)`, `resize(new_notional, snap, cost_model)`,
`close(on, reason)`, `describe()`.

**Sleeves.** Every position carries `sleeve` — `'convexity'`, `'skew'`, `'atm'`,
`'wings'`, `'hedge'`. This is what turns "did the volga sleeve make money" into a
groupby instead of a re-run. **Tag honestly at creation — it cannot be
reconstructed later.**

> **INVARIANT:** `notional` is mutable and greeks are computed off the *current*
> notional. Therefore the loop must **MARK BEFORE IT TRADES**.

## 3.2 `book/costs.py` — the spread you actually pay

Neither the old stack nor Phase 1 charged anything to trade an option. For a
wing-selling strategy that is not a small omission — the wing spread is the
dominant cost and can consume the entire premium.

### The spread surface

```
spread_vp = base_vp[pair] * tenor_mult(t_days) * wing_mult(|delta|) * scale
```

- `base_vp` — the pair's 1M ATM full bid-ask in vol points.
- `tenor_mult` — `sqrt(TENOR_REF_DAYS / t_days)`, clipped floor/cap. Short tenors
  cost more.
- `wing_mult` — interpolated on `WING_DELTA_KNOTS`. Wings cost more.
- `scale` — **THE knob** `run/breakeven_study.py` sweeps. `1.0` = the defaults,
  `0.0` = free (Phase 1 behaviour).

You pay `spread_vp * crossing_fraction` (default `0.5` = deal at bid or ask from
mid; `1.0` is a reasonable stress case for a systematic taker).

### `charge(...)` — the one interface, two mechanisms

```python
vega_cost  = notional * |vega_1vp| * paid_vp        # mechanism 1
floor_cost = premium  * min_premium_frac            # mechanism 2 (default 3%)
cost       = max(vega_cost, floor_cost)
```

The floor exists because vega → 0 in the deep wings, so a pure vega×spread model
charges almost nothing for a 5-delta option that nobody will actually quote you
tightly. Returns a **record dict**, not a bare float, so you can always see which
spread applied and which mechanism bound (`bound_by`).

> **A consequence worth internalising:** `min_premium_frac` **binds at ATM and
> at 25d, and does not bind at 5d**, because premium is largest ATM. That makes
> ATM ~3.8× more expensive per unit notional than its vega spread implies, and it
> pushes the sizer toward the wings. Whether that floor is calibrated the way you
> intended is a real open question.

**The Book owns the cost model**, so a strategy physically cannot trade an option
without paying. A forgotten cost is the kind of bug that makes a backtest look
good.

### Cost asymmetry on exit

`Book.close` charges the full spread on an **early** close and **nothing** at
expiry — an expiring option settles to intrinsic, it is not traded out of. That
asymmetry is real, and it is a genuine reason to prefer holding to expiry over
rolling early in a wing strategy.

## 3.3 `book/attribution.py` — marking and the premium buckets

### `PositionMark`

The position's state at a point in time, cached by the Book, rolled forward
daily: `on, spot, sigma, r_d, r_f, t_days, value_base, greeks, nu, rho, notional`.
Caching it means the surface is hit **once per position per day, never twice**.

### `take_mark(pos, snap)`

Snapshot the current state. The result becomes tomorrow's START-of-period state.

### `mark_position(pos, prev, snap, nu_rho_at_end=False)`

Two branches:

- **Natural expiry** — settles to intrinsic. All vol buckets are hard-zeroed
  (there is no vol sensitivity left), but gamma and theta ARE still computed from
  start-of-day greeks, because the realised spot move over that final period was
  real.
- **Ordinary day** — exact reprice for the P&L, Taylor buckets off `prev` greeks
  for the attribution, and the gap recorded as `recon_resid`.

**The one-period convention:** all attribution is priced off **START-of-period
greeks**. `dt_days` is the ACTUAL calendar gap — 3 over a weekend, not 1.

> ⚠️ **THE GOTCHA THAT WILL BITE YOU.** The row written for date `D` contains
> `prev.greeks` — the book as of the **close of D−1**. That is correct (it is
> what the day's P&L was earned on) but it means `res.daily`'s `net_*` columns
> lag one business day. **A trade on D shows up in the row for D+1.**

### The `_be` buckets — the objective function

Each nets the realised move against what the market's own implied dynamics
predicted, leaving only the surprise:

| bucket | formula | premium it measures |
|---|---|---|
| `gamma_pnl_be` | `gamma_pnl + theta_pnl` | ATM variance risk premium |
| `vanna_pnl_be` | `vanna × (dS·dσ − S·σ²·ρ·ν·dt)` | **skew** premium |
| `volga_pnl_be` | `volga × (dσ² − σ²·ν²·dt)` | **convexity / vol-of-vol** premium |

`gamma_pnl_be` works because BS theta is to leading order `−½S²Γσ²`, so adding it
back cancels the variance the position was priced for.

`nu_rho_at_end=False` (the default and correct) forms the expectation from
**start-of-period** information. `True` reproduces the old engine's mild
look-ahead and is for reconciliation only.

**A single day of any of these is mostly noise. The DRIFT of the cumulative is
the signal.** Rank strategies on that drift per unit of the corresponding greek
carried — not on calmar.

### `recon_resid`

`option_pnl − taylor_total`. A measure of how well the expansion held, **not an
error to drive to zero**. Large means a discrete jump (gap risk) or that the
expansion broke down. Unlike the old engine's version, this one is per-position
and hedge-free — it includes the first-order delta term, so it is a clean "how
good was the Taylor expansion" number rather than a quantity entangled with the
hedge.

### `hedge_period_pnl(...)`

```
hedge_pnl   = hedge_notional * dS / prev_spot
hedge_carry = hedge_notional * (r_f - r_d) * dt_days / 365
```

`hedge_notional` is the position held **coming INTO** the period. Getting that
wrong gives the hedge a one-day look-ahead and quietly flatters every result.

## 3.4 `book/book.py` — the container

```python
positions: Dict[int, Position]      # open AND recently closed
marks:     Dict[int, PositionMark]  # one per OPEN position
hedge:     Dict[str, float]         # pair -> outstanding spot hedge, base ccy
cost_log:  List[dict]               # drained daily by the engine
cost_model: OptionCostModel
```

In the old stack the "book" was an accounting artifact assembled *after* the fact
by superimposing trades that had each already been run and hedged in isolation.
It could describe the portfolio but could not steer it.

### The substantive change: netted hedging

```python
net_delta(pair)       # sum of premium-adjusted deltas across open positions
residual_delta(pair)  # net_delta + the hedge already carried  <- test THIS
rebalance_hedge(pair, fraction, tc_fraction)
```

**Netting means transaction cost is paid on NET delta, the way a desk pays it,
instead of on GROSS delta across overlapping trades.** For a strategy earning a
couple of vol points on the wings, that is not a rounding error.

`residual_delta` — not gross option delta — is what a band-hedging rule must
test. The old stack's `DeltaBandHedge` tested gross and therefore either never
rehedged or degenerated into a daily hedge.

`rebalance_hedge` moves toward flat: `target = −net_delta`, `traded = fraction ×
(target − current)`. Note that skipping or partially hedging only ever changes
hedge P&L, carry and cost — **it never changes `option_pnl` or any greek
bucket**, which come purely from the option's own risk and the realised move.

### `mark_all(snaps)`

Advances every open position, returns one record row each, and **auto-closes
positions that settled at expiry** with `exit_reason='expiry'` (their final row
is still returned). Calling this is what makes `Book.greeks()` current.

---
---

# PART 4 — `engine/loop.py`: THE DATE LOOP

## 4.1 The six steps, and why that order

```
1. SNAPSHOT   build a point-in-time view per pair
2. MARK       attribute yesterday -> today on the book AS IT WAS
3. HEDGE P&L  P&L and carry on the spot hedge CARRIED IN
4. TRADE      the strategy opens / closes / resizes  <- the ONLY mutation
5. REHEDGE    ONE net spot trade per pair against the NEW book
6. RECORD     append the rows
```

**Swapping 2 and 4, or 3 and 5, produces a backtest that looks fine and is
wrong.** Marking after trading attributes a day's P&L to a position size that was
not on risk for it. Hedging before computing hedge P&L gives the hedge a one-day
look-ahead.

The date grid is the **union** of every requested pair's observed spot dates. A
pair missing on a date is skipped, not forward-filled — inventing a mark is worse
than not having one.

## 4.2 `DayContext` — what a strategy receives

```python
date, index, snaps: Dict[str, MarketSnapshot], book: Book, is_first, is_last
```

`index` is the **0-based position in the date grid**, which is why `roll_days`
counts *business* days.

## 4.3 `Strategy` — the interface

One method: `on_date(ctx) -> None`. Mutate the book directly. That is the entire
contract, and it does not change across phases.

Two built-ins:
- **`HoldStatic`** — open one structure on day one, hold to expiry. Exists for
  *one* reason: it is the configuration the old engine also produces, so it is
  the only strategy against which the new engine can be reconciled bit-for-bit.
- **`RollingStructure`** — re-open a fixed structure every N days, vintages
  overlap. Coherent for fixed notional; **incoherent for a greek target** — see
  §6.

## 4.4 `EngineConfig`

| field | meaning |
|---|---|
| `pairs`, `start`, `end` | scope |
| `hedge_fraction` | 1.0 = full daily delta hedge |
| `spot_tc` | cost per unit of spot notional (default 1bp) |
| `cost_model` | `None` = options are FREE (Phase 1 behaviour) |
| `flatten_at_end` | unwind the hedge on the last date |
| `carry_tenor_days` | which curve point the hedge carry differential comes from |
| `legacy_strike_halfstep`, `legacy_nu_rho_at_end` | reconciliation only |
| `charge_tc_on_expiry_unwind` | the old engine charged nothing here; new default charges |

On `carry_tenor_days`: the hedge is an **overnight** spot position, so a short
tenor is the defensible choice (default 30d). `None` reproduces the old engine's
convention of using the option's own remaining tenor, which shrinks every day —
its carry therefore drifted as the trade aged (~2% over a 3M trade).

## 4.5 `RunResult` — three tidy frames

| frame | grain |
|---|---|
| `positions` | one row per **(date, position)** — the long format |
| `hedges` | one row per (date, pair) |
| `daily` | book-level roll-up, indexed by date. `pnl` and `equity` live here |
| `.trades` | one row per leg opened |
| `.costs` | every option cost event |

The long format for `positions` is the design choice that makes everything
downstream a `groupby` — by sleeve, pair, tenor bucket, delta bucket — instead of
the old stack's wide `*_legN` columns, which cannot survive a variable position
count.

Helpers: `by_sleeve(col)`, `greek_carried(greek)`, and
**`pnl_per_unit_greek(pnl_col, greek)`** — *the* metric for this strategy:
premium harvested per unit of the risk that harvested it. Ranks configurations
far better than calmar, which rests on a single order statistic from one path.

## 4.6 `_roll_up` — FLOW vs EXPO

```python
_FLOW = ['option_pnl','delta_pnl','gamma_pnl','theta_pnl','vega_pnl',
         'vanna_pnl','volga_pnl','gamma_pnl_be','vanna_pnl_be',
         'volga_pnl_be','recon_resid']
_EXPO = ['spot_1pct','gamma_1pct','vega_1vp','volga_1vp',
         'vanna_1pct_1vp','theta_1d','delta_hedge']
```

**FLOW columns are summed and then cumsummed into equity. EXPO columns are
summed and NEVER cumsummed — they are point-in-time levels.** Conflating the two
is the single easiest way to produce a nonsense equity curve.

```python
pnl = option_pnl + hedge_pnl + hedge_carry - hedge_tc - option_tc
```

Premium flows are **not** added separately: `option_pnl` is a mark-to-market
change, which already embeds the premium from the day the leg was struck.

---
---

# PART 5 — `strategy/sizer.py`: SIZING IN RISK SPACE

## 5.1 The three ideas

**1. Greeks are linear in notional, so sizing is a linear solve.** Every
`GreekVector` is `notional × direction` times a per-unit partial. If `g_i` is
candidate `i`'s greek vector at unit notional, the book's greek is `Σ n_i·g_i`.
Stack the `g_i` as columns of `A` and hitting a target `t` is `A n = t`.

**2. The menu is STRIKES, not options.** By put-call parity a same-strike call
and put differ by a forward, and a forward has no vol sensitivity. Their vega /
vanna / volga are **identical** — they are one column of `A`, not two. So the
menu is one leg per strike, taken OTM (which is also what quotes tight).

**ATM is the exception:** it enters as a **straddle** (call + put, equal
notional), because a single ATM leg is a large naked delta the solve has no
reason to want.

**3. Under-determined is the normal case, so the objective matters.** Nine
strikes against three constraints leaves an infinite family. We pick the one
minimising **expected spread paid**, using the same `OptionCostModel` the engine
charges. That turns "which structure" from a taste question into a priced one.

## 5.2 `GreekTarget`

```python
GreekTarget(
    by_tenor = {'3M': dict(volga=-2500, vega=0.0, vanna=0.0)},
    net      = {'gamma': 280_000},
    horizon_days = 7,
    units    = 'normalised',
)
```

- **`by_tenor`** constrains only that tenor's legs. **Anything you want flat
  belongs here.**
- **`net`** constrains the sum across tenors. Use deliberately — a net-zero can
  hide a bucket offset.

**Three distinct states, and the difference matters:**

```
'vega': 0.0     -> actively PINNED at zero (consumes a constraint)
key absent      -> free, whatever the structure implies
'volga': -2500  -> hit this number
```

*"Zero"* and *"don't care"* are not the same instruction.

### The term-structure trap

A flat greek vector sums across expiries, and that summation hides things. Sell
1M vega, buy 3M vega, size for zero net vega: your `GreekVector` reads
vega-neutral and you are holding a pure term-structure position. Front vol and
back vol do not move together.

This is the FX analogue of a bond book with zero net duration and enormous curve
exposure, and the fix is the one rates desks use — **bucket by maturity instead
of summing**.

## 5.3 Building the menu — `build_candidates`

`DEFAULT_PILLARS = ('ATM', 35, 25, 10, 5)` — exactly `DELTA_POINTS` minus the
15d, plus ATM. Every one is a directly quoted RR/BF point, so nothing relies on
SABR extrapolation between knots.

Strikes resolve through `snap.solve_strike_and_vol` — **the same fixed point
`open_position` uses** — so the greeks solved on are the greeks you will be
filled at, not an approximation of them.

A pillar that cannot be struck (brentq's bracket does not reach a 5-delta wing in
a high-vol pair) is **dropped and named** in `res.dropped`, rather than silently
degrading the menu.

Each `Candidate` carries: `label, tenor, expiry, pillar, sub_legs, greek`
(summed `GreekVector` at notional=1) and `cost_unit` (base-ccy spread per unit
notional).

## 5.4 Normalised units — `sigma_scales`

`units='normalised'` (default) denominates targets in **base-ccy P&L per one-sigma
move over `horizon_days`**. The conversion is exactly the Taylor coefficient from
`taylor_pnl`, with the move set to one sigma:

```python
dt    = horizon_days / 365
sigma = snap.atm_vol(t_days)        # ATM vol at THAT tenor
nu, _ = snap.nu_rho(t_days)         # vol-of-vol at THAT tenor

u = (sigma * sqrt(dt)) / SPOT_MOVE       # 1-sigma SPOT move, in "1%" units
w = (sigma * nu * sqrt(dt)) / VOL_MOVE   # 1-sigma VOL  move, in "vol points"
```

| target | field | multiplier |
|---|---|---|
| `vega` | `vega_1vp` | `w` |
| `volga` | `volga_1vp` | `w²` ← second order |
| `vanna` | `vanna_1pct_1vp` | `u·w` ← cross term |
| `gamma` | `gamma_1pct` | `u²` |
| `spot` | `spot_1pct` | `u` |
| `theta` | `theta_1d` | `horizon_days` (deterministic) |

So `volga = −2,500` reads: *"if vol moves one sigma over a week — either
direction — the volga term of my P&L is −\$2,500."*

**`u` and `w` are the whole idea.** They are the same `u` and `w` from
`taylor_pnl`, except with a one-sigma move plugged in instead of a realised one.

### Scaling laws when you change `horizon_days`

Since `u, w ∝ √dt`:
- `vega` target ∝ `√horizon` → raw position ∝ `1/√horizon`
- `volga`, `gamma`, `vanna`, `theta` ∝ `horizon` → raw position ∝ `1/horizon`

Going from 7 to 28 days **quarters** the raw volga position for the same stated
number. Two runs are not comparable unless the horizon matches.

### Three caveats

1. **The scales are today's market, so a static target is a floating notional.**
   Intended — constant *risk*, not constant notional — but it means a book that
   was exactly on target yesterday can read off-target today purely because
   σ or ν moved, and `top_up` will trade the difference.
2. **"One sigma" is an IMPLIED sigma.** ν comes from the surface. Realised
   vol-of-vol is several times smaller (see §10), so a normalised target measures
   P&L on ~3–4 realised sigmas. That gap *is* the premium — but do not read
   `−2,500` as "what a normal week costs me".
3. `units='raw'` skips all of it and targets the `GreekVector` fields directly.

## 5.5 Assembling and conditioning

`_assemble` builds `A` (rows = constraints, cols = candidates) and `t`, both in
the target's units. A `by_tenor` row zeroes out columns from other tenors.

`_row_normalise` scales each constraint row to unit length. Without it a gamma
row (~1e5) and a volga row (~1e3) are implicitly weighted by three orders of
magnitude of *unit accident* rather than by anything you chose. **Purely
numerical — it does not change the underdetermined solution.**

`intrinsic_condition(A)` normalises **both** rows and columns before taking the
SVD ratio, so it measures genuine collinearity rather than a units mismatch.

## 5.6 The solve — `_min_cost_solve`

Minimise `Σ cost_i·|n_i|` subject to `A n = t`.

**Why L1, and why an LP.** The objective is literally the money crossing the
spread, so it is an L1 norm — and L1 is also what makes the answer **tradeable**:
an LP basic solution has at most (number of constraints) non-zero variables.
Three constraints → at most three legs. **Sparsity is a property of the optimum,
not a post-processing step.**

Formulation: split `n = p − m` with `p, m ≥ 0`, so `|n_i| = p_i + m_i` and
everything is linear. Solved by `scipy.optimize.linprog(method='highs')`.

Two earlier attempts, and why they failed — both worth knowing because both look
reasonable:

- **Cost-weighted L2 (one lstsq call).** Optimises the wrong thing, and not
  merely imprecisely: L2 actively *prefers* to spread notional across many
  columns, because splitting a position in two lowers the sum of squares while
  leaving the sum of absolutes unchanged. It returned an eight-leg smear.
- **IRLS towards L1.** Right objective, but it stalls. On the gamma calendar it
  locked onto a near-duplicate 35-delta put/call pair — effectively a synthetic
  forward — at 3.4bn a side, and reweighting could not climb back out.

If the LP is infeasible it falls back to minimum-norm least squares and reports
`method='lstsq'`; the caller reports the residual.

## 5.7 Pruning

Legs below `min_notional` (default 1mm) are dropped **one at a time** and the
solve re-run. A 40k tail leg is untradeable and only pollutes the cost line.
`max_legs` caps the count but never prunes below the number of constraints.

## 5.8 The two guards

Some targets cannot be met honestly, and the failure is **silent**: the solver
returns huge offsetting notionals whose residual difference happens to equal the
target. It satisfies the constraints on paper and the cost line tells you six
months later.

**`require_cond` (default 1e4)** tests whether the **constraint ROWS** are
independent. It fires when you ask for two things that are the same thing.

**`require_leverage` (default 10)** tests whether the **COLUMNS** are:

```
leverage = sum_j |n_j| * ||A[:,j]||  /  ||t||
```

`cond(A)` is blind to this — the constraints can be perfectly independent while
the answer is two adjacent strikes taken in billions of offsetting notional.
**This is the guard that actually fires in practice.** Calibration: a
vega-neutral fly runs 2.4, a delta-hedged risk reversal 1.7, a front/back gamma
calendar 1.9; single-tenor gamma-vs-vega runs 14.

### The canonical case, and a correction worth knowing

Gamma and vega inside one expiry are usually described as exactly collinear —
`vega = gamma·S²·σ·T`, strike-independent ratio. **In this stack that is not
quite true**, because greeks are premium-adjusted base-ccy P&L:
`d²V/dS² = G/S − 2·D_pa/S²`, and the `D_pa` term varies strongly across strikes.
So the columns are ~10% independent rather than 0%, and a single-tenor
gamma-vs-vega target is technically **feasible**.

It is just ruinous. Measured on a 3M USDJPY menu, the same target:

```
one tenor  : 9.3bn gross notional, $1.69m to trade, leverage 14
two tenors :  156mm gross notional,   $26k to trade, leverage 1.9
```

**65×.** The fix is a second, well-separated tenor. `require_leverage=None` sizes
it anyway if you want to see the number.

## 5.9 `current=` — from calculator to controller

The target is a **book-level** statement, so what this trade must deliver is the
target **minus what is already on risk**:

```python
carried[r] = sum(raw_carried_greek * scales[t][gname] for t in keys)
t_vec      = t_book - carried
achieved   = carried + traded
```

`current` accepts a `Book` (normal case), a bare `GreekVector` (single-tenor
menus only), or a `{tenor: GreekVector}` dict. `None` means "assume a flat book",
which is right on the first trade and **wrong on every subsequent one**.

### `bucket_book` — the judgement call

Aged positions do not sit on your menu: a 3M struck five days ago is a
2M-and-change today, and it must count against *some* bucket. The rule is
**nearest bucket in LOG remaining-days** — log because tenor is multiplicative.

Greeks are recomputed off `snap` via `Position.greeks`, **not** read from
`book.marks`, so they are on the same surface read as the candidate legs. Using
cached marks would silently mix two dates.

The mapping is returned as text in `res.bucket_map` so it appears in the result
rather than being assumed.

> With a **single-tenor** menu the log-distance rule is vacuous — everything goes
> in the one bucket, scaled by *today's* multiplier regardless of its own age.
> The arithmetic still nets correctly at book level, but the bucket label carries
> no information.

## 5.10 `SizerResult`

| field | meaning |
|---|---|
| `legs` | the `LegRequest`s to open |
| `notionals` | label → signed notional |
| `realised`, `realised_by_tenor` | **this trade's** greeks, raw |
| `carried`, `book_after` | already on risk / after the trade |
| `target_rows`, `achieved`, `residual` | the honesty check |
| `condition`, `leverage`, `method` | diagnostics |
| `cost`, `gross_notional` | what it costs |
| `scales` | tenor → greek → multiplier |
| `bucket_map`, `dropped` | what got assumed / discarded |

`open_into(book, snap)` opens every leg **with the expiry the sizer actually
priced** — the one place tenor→expiry resolution is guaranteed to match the
greeks that were solved on. Use it rather than opening legs yourself.

`greek(name, tenor)` → raw. `greek_normalised(name, tenor)` → one-sigma units.

## 5.11 Preset builders

| builder | target |
|---|---|
| `vega_neutral_fly(snap, tenor, volga)` | volga to target, vega and vanna at 0 — **the convexity atom** |
| `risk_reversal(snap, tenor, vanna)` | vanna to target, vega and volga at 0 — **the skew atom** |
| `vega_sleeve(snap, tenor, vega)` | plain long/short vol |
| `gamma_sleeve(snap, front, back, gamma)` | `net={'gamma': X, 'vega': 0}` across a calendar |

`risk_reversal` generally will **not** return a clean two-leg RR: the two wings
are not exactly volga-symmetric, so it bolts on a small ATM straddle to null the
residual convexity. Without that leg a "pure vanna" sleeve is quietly part volga.

`gamma_sleeve` — read its docstring before using it. The obvious specification
(`by_tenor={front: vega 0, back: vega 0}`) is **infeasible in any useful sense**:
a bucket with zero vega has essentially zero gamma too. **Gamma and vega separate
ACROSS tenors, not within them.** The resulting book deliberately runs front-vs-
back vega: net zero, buckets not. That is the unavoidable price of holding gamma
without vega, and `realised_by_tenor` is where you see how much you took.

## 5.12 What the sizer does NOT do

It does not **search** for strikes. You give it a menu and it sizes within it.
Automatic strike selection is a discrete optimisation that would overfit whatever
the cost model assumes, and it properly belongs after a richness signal exists.

---
---

# PART 6 — `strategy/roller.py`: MAINTAINING A TARGET

## 6.1 Why this is not `RollingStructure` with a sizer bolted on

`RollingStructure` re-opens a fixed structure every N days and lets old vintages
run alongside. With fixed notional that is coherent. **With a greek target it is
incoherent.** You asked for −2,500 of volga; roll three times with overlap and
the book carries −7,500. The number you set no longer describes the risk you
hold, and the multiple depends on your cadence and tenor.

## 6.2 The three modes

| mode | closes? | `current` | result |
|---|---|---|---|
| `'top_up'` | no | the Book | trades only the **difference** — DEFAULT, correct |
| `'replace'` | yes, whole sleeve | `None` | fresh structure, pays exit spread every roll |
| `'stack'` | no | `None` | full target every roll — **the broken behaviour**, kept only so it can be measured |

`replace` and `stack` differ *only* by whether positions are closed first.

## 6.3 `on_date`, step by step

| # | gate | what happens |
|---|---|---|
| 1 | `pair not in ctx.snaps or ctx.is_last` | return — no data, or last date |
| 2 | `ctx.index - self._last < roll_days` | return — counts **business** days |
| 3 | `mode == 'replace'` | close every open position in this **pair + sleeve** |
| 4 | `current = ctx.book if top_up else None` | **this one line is the three modes** |
| 5 | `solve(snap, tenors, target_fn(snap), current=..., sleeve=...)` | target rebuilt fresh each roll |
| 6 | `SizerError` | → `'skipped_guard'`, logged, **`_last` still set** |
| 7 | record `leverage`, `cond`, `want_*`/`book_*` | |
| 8 | `res.gross_notional < min_trade` | → `'deadband'`, no trade |
| 9 | `res.open_into(ctx.book, snap)` | opens legs; Book charges the spread |

`_last = ctx.index` fires on **all four** exit paths, so any attempt resets the
clock. A transient guard trip therefore costs you a full roll period.

**`target_fn(snap)` is the Phase-5 seam.** Today it returns a constant; make it a
function of a richness z-score and nothing else in the class changes.

## 6.4 The log is part of the deliverable

A greek-target strategy has two new ways to quietly do nothing — the guard trips,
or the increment falls inside the deadband — and **neither shows up in a P&L
curve**. A strategy that skipped 40% of its rolls looks fine and is not the
strategy you specified. `roller.report()` prints the tally; `roller.frame()` is
the full log.

## 6.5 Reading the log correctly

> ⚠️ **`book_volga@3M` reads exactly the target on every row and that proves
> NOTHING.** `achieved = carried + traded` and the LP drives the residual to
> ~1e-13, so it is true by arithmetic whether or not `top_up` is reading the
> book. If `carried` were wrongly zero, the solve would trade the full amount and
> still report the target.

The genuine diagnostics:

- **`gross` collapsing after roll 0** → top-up is reading the book.
- **`want_*` ≠ `book_*`** → the LP went infeasible and fell back to `lstsq`. This
  is the *only* place that surfaces, since `res.method` is not logged.
- **`res.daily['net_volga_1vp']`** → what you actually carry (remember the
  one-day lag).

## 6.6 A caution about `min_trade`

The deadband tests **gross notional summed over legs**, not the size of the greek
increment. A three-leg solve with large offsetting legs has a big gross even if
the risk delta is small, so a heavily-offsetting increment can slip past. If that
matters, tighten `min_notional` in `solve_kw` instead.

---
---

# PART 7 — STUDIES AND VALIDATION HARNESSES

## 7.1 `run/breakeven_study.py` — the Phase 2 gate

**Not a backtest. A feasibility test.** The question:

```
gross premium harvested  >  cost of repeatedly putting the trade on ?
```

`sweep(pair, tenor, wing, roll_days, dataset)` runs the same rolled strangle at
`cost_model.scale` from 0 (free) up to several multiples of assumed interbank
levels. `breakeven_scale(sw)` finds where net P&L crosses zero.

**Why it sweeps a scale rather than quoting one number:** the absolute spreads in
`book/costs.py` are *assumptions*, not measurements. Any single P&L figure
inherits that uncertainty. A break-even **level** does not — it says "the strategy
needs costs below X", and you can judge X against your own execution without
trusting the defaults at all.

```
break-even scale >> 1   the premium survives realistic costs. Proceed.
break-even scale ~  1   marginal. A cost-execution problem before an alpha one.
break-even scale <  1   the premise does not survive. Stop and rethink.
```

## 7.2 `recon/reconcile.py` — the Phase 1 gate

Runs the same trade through the old `Delta_Hedged` engine and the new one, day by
day, bucket by bucket. The old stack is the only trusted reference that exists,
and **once netted hedging, option costs and multi-expiry rolling exist the two
engines are no longer computing the same thing** — so this is the one window in
which a silent arithmetic error can be caught cheaply.

The `legacy_*` flags in `EngineConfig` exist to serve this: `legacy_strike_
halfstep`, `legacy_nu_rho_at_end`, `charge_tc_on_expiry_unwind=False`, and
`ds.nu_rho_source='closed_form'`.

## 7.3 `recon/smoke_test.py`

End-to-end on a **synthetic** dataset, no Bloomberg needed. Proves the plumbing:
greeks wire through, costs charge, `_be` buckets populate, `scale=0` reproduces
free options exactly. It explicitly does **not** validate any P&L number.

## 7.4 `recon/ovml_lineup.py`

Describe trades the way you would say them out loud —
`"USDJPY 1M 25d call 20mm"` — and check strike/vol/premium/greeks against
Bloomberg OVML. Read-only.

## 7.5 `test.py` and `test_full.py`

- **`test.py`** — the Phase 3 test suite: `p3_test1_volga` (the convexity atom)
  through `p3_test7_roller` / `p3_test7b_compare_modes`, plus
  **`p3_test7_trace`**, which intercepts the `SizerResult` the roller discards and
  prints every leg of every roll, the increment decomposed into *reprice* vs
  *drift*, expiry events, and target tracking.
- **`test_full.py`** — the linear, function-free walkthrough: a systematic
  short-wing strategy in ATM/10d/25d space, twelve sections from data to
  diagnostics.

---
---

# PART 8 — ONE DAY IN THE LIFE

Following a single date through the whole stack:

```
engine/loop.run, iteration i, date D
│
├─ 1 SNAPSHOT
│    MarketSnapshot.at(ds, 'USDJPY', D)
│      └─ holds a reference to the dataset + as_of. Nothing is read yet.
│
├─ 2 MARK  (book.mark_all)
│    for each open Position:
│      prev = book.marks[pos_id]                    (state at close of D-1)
│      snap.price_state(K, expiry)
│        ├─ rates(t_days)          -> dataset._rate_cache
│        └─ smile_vol(K, t_days)   -> dataset._smile_grid_cache -> SABR fit
│      mark_position(pos, prev, snap)
│        ├─ option_pnl = v_new - prev.value_base    (the truth)
│        ├─ Taylor buckets off PREV greeks          (the decomposition)
│        ├─ _be buckets using prev.nu / prev.rho    (the premium)
│        ├─ recon_resid = option_pnl - taylor_total
│        └─ row includes **prev.greeks**            <- the one-day lag
│      book.marks[pos_id] = new_mark
│    positions that expired are auto-closed (free)
│
├─ 3 HEDGE P&L  (book.hedge_pnl)
│    on the hedge notional carried IN from D-1
│      hedge_pnl   = h * dS / prev_spot
│      hedge_carry = h * (r_f - r_d) * dt / 365
│
├─ 4 TRADE  (strategy.on_date(ctx))       <-- GreekTargetRoller
│    if not a roll date: return
│    solve(snap, tenors, target_fn(snap), current=ctx.book, sleeve=...)
│      ├─ build_candidates      -> strikes via solve_strike_and_vol
│      ├─ sigma_scales          -> u, w from atm_vol and nu_rho
│      ├─ bucket_book(current)  -> what is already on risk
│      ├─ _assemble  -> A, t    -> t_vec = t - carried
│      ├─ _min_cost_solve       -> LP, minimum spread
│      ├─ prune below min_notional, re-solve
│      └─ guards: leverage, cond
│    res.open_into(book, snap)
│      └─ book.open -> open_position -> cost_model.charge -> book.cost_log
│
├─ 5 REHEDGE  (book.rebalance_hedge)
│    ONE net spot trade per pair against the NEW book
│      target = -net_delta(pair);  traded = fraction * (target - current)
│      hedge_tc = |traded| * spot_tc
│
└─ 6 RECORD
     day_cost_rows = book.drain_costs()
     _roll_up(...)  -> FLOW summed, EXPO summed (never cumsummed)
     pnl = option_pnl + hedge_pnl + hedge_carry - hedge_tc - option_tc
     prev_state[pair] = {spot, r_d, r_f, on}     for tomorrow's hedge P&L
```

---
---

# PART 9 — THE TRAPS, CONSOLIDATED

Every one of these produces a result that looks plausible and is wrong.

| # | Trap | Where |
|---|---|---|
| 1 | **Mark after trading** → a day's P&L attributed to a size that was not on risk | `engine/loop.py` step order |
| 2 | **Hedge before computing hedge P&L** → one-day look-ahead, flatters everything | steps 3 vs 5 |
| 3 | **`res.daily`'s `net_*` columns lag one business day** — a trade on D appears on D+1 | `attribution.mark_position` writes `prev.greeks` |
| 4 | **Cumsumming an EXPO column** → nonsense equity curve | `_roll_up`'s FLOW/EXPO split |
| 5 | **Reading `vol_surface` as decimal** — it is stored in percent | `market/dataset.py` |
| 6 | **Constant notional is not constant risk** across tenors — and it is worse in second-order greeks, where the move enters squared | `sigma_scales` |
| 7 | **`'vega': 0.0` ≠ omitting the key.** Zero pins and consumes a constraint | `GreekTarget` |
| 8 | **Net-zero hides a bucket offset.** Put anything you want flat in `by_tenor` | `GreekTarget` |
| 9 | **A "vega-neutral" structure is neutral only on the day it is struck** | no re-hedge exists yet |
| 10 | **`book_volga@X` in the roll log is tautological** — it cannot detect a broken book read | `roller.on_date` |
| 11 | **Re-issuing a greek target every roll stacks the risk** — use `top_up` | `roller` modes |
| 12 | **Gamma vs vega in one expiry is technically feasible and ruinous** — 65× the cost | leverage guard |
| 13 | **`min_premium_frac` binds at ATM, not in the far wings** — ATM is the expensive leg per unit notional | `costs.charge` |
| 14 | **`nu` is IMPLIED.** A "one-sigma" normalised target is ~3–4 realised sigmas | `sigma_scales` |
| 15 | **The deadband tests gross notional, not the greek increment** | `roller.min_trade` |
| 16 | **A guard trip costs a full roll period** — `_last` is set even on a skip | `roller.on_date` |
| 17 | **`delta_hedge` does not sum across pairs** — `GreekVector` blanks it to NaN | `core/greeks.py` |
| 18 | **P&L fields are each in their own pair's base ccy** — multi-pair sums need USD conversion first | `Book.greeks` |
| 19 | **`dt_days` is 3 over a weekend, not 1** — theta and carry scale by it | `attribution` |
| 20 | **`recon_resid` is a diagnostic, not an error to zero out** | `attribution` |

---
---

# PART 10 — MEASURED FACTS WORTH REMEMBERING

All measured on USDJPY, ~400 days of history. One pair, one sample — treat cost
and structural numbers as reliable, P&L numbers as noisy.

## 10.1 The ν calibration

Closed form vs the SABR fit that builds the marks, **2,583 (date, tenor) fits**:

```
tenor   nu_SABR   nu_BE_raw   ratio    corr
1W        4.724       6.100   1.29     0.992
1M        2.072       2.664   1.285    0.989
3M        1.178       1.494   1.269    0.983
6M        0.830       1.060   1.277    0.974
1Y        0.590       0.787   1.334    0.922

pooled ratio: median 1.277, sd 0.040
NU_BE_C that would match SABR:  4.0 / 1.277 = 3.13
```

The closed form **tracks beautifully and overshoots by a near-constant 1.28×**.
Its ρ is worse than a scale error — calibrated ρ decays to ~0 and **flips sign by
1Y** (`+0.017`) while `rho_BE` stays at `−0.111`. No constant fixes that.

`NU_BE_MAX = 5.0` binds on **8.7%** of fits overall — and **60% at 1W**.

`ν·√τ ≈ 0.587–0.629` across the whole curve. This near-invariance is what
justifies interpolating `ν·√days` rather than ν itself.

## 10.2 Implied vs realised vol-of-vol — the premium

Measured on 3,318 position-days from a real run:

```
realised nu (option-level, incl. roll-down)    1.131
mean nu_be used by the attribution             2.721      2.41x

mean realised dsigma^2      1.020e-04
mean var_expected           5.577e-04           5.46x
bracket (dsigma^2 - var_exp) negative on       98.5% of days
```

**This is the premium, not an error.** The closed form was only 1.28× off as a
reading of *implied* ν; the remaining gap is genuinely implied-over-realised.

But it means `volga_pnl_be` is a **near-deterministic positive drip** for a
short-volga book, so it cannot currently be used as a cross-sectional predictor.
Reconciling that against realised money is the open question.

## 10.3 Performance

120-day, single-pair run, **before caching: 208 s under profile / 83.9 s clean.**

```
get_rates_for_tenor -> _zero_rate    116.9 s   56%   25,398 calls
get_smile_vol                         74.4 s   36%    8,647 calls
  ├─ _fit_sabr_to_grid                18.5 s    9%   17,225 calls
  │    └─ QuantLib SABR calibration    2.7 s  1.3%
  └─ build_vol_grid                   10.6 s    5%   86,470 calls
pandas MultiIndex.get_loc             18.9 s (self)  2.1M calls
```

**The SABR calibration was 1.3% of runtime.** The bottleneck was pandas
MultiIndex lookup. After caching: **32.8 s (2.6×)**, verified bit-identical
(max diff 4.657e-10, which is exactly the CSV round-trip error of the reference
against itself).

## 10.4 Switching ν source resizes the book

Because `sigma_scales` reads ν, the *same* normalised target sizes differently:

```
1M, 2026-07-15:  w^2 closed 10.361 -> sabr 5.730     x1.808
3M, same date:   w^2 closed  2.608 -> sabr 1.521     x1.715
```

Every leg, gross notional and cost scale by **exactly** that factor — the LP
solution's shape is preserved because all columns in a single-tenor row share the
same multiplier. **The conversion is tenor-dependent, so there is no single
global rescale.**

## 10.5 Neutrality drift — what sizes the vega hedge

Residual vega per unit of volga carried, normalised:

```
3M, roll_days=5:  mean 0.392  median 0.296  p90 0.859  peak 2.03   59% of days > 0.2
1M, roll_days=5:  mean 0.676  median 0.466  p90 1.380  peak 8.62   68% of days > 0.2
```

Both are pinned to **exactly zero** on every roll day. Everything above
accumulates in the four days between.

Cadence sweep (1M, ATM/10d/25d, 200 business days):

```
roll_days  option_tc  mean|vega|/budget    p90   %>0.20  volga_err
        5     68,291              0.676  1.380     67.5%       564
        3     93,796              0.447  1.055     51.0%       411
        2     88,469              0.209  0.527     34.0%       195
        1    116,404              0.027  0.000      5.7%        11
```

**The residual vega is essentially zero-mean** (signed mean 209 against a mean
absolute of 2,029 at rd=5), so a vega hedge buys **attribution purity and
variance reduction, not expected return**. A 1M ATM straddle costs
**\$0.25 of spread per unit of vega_1vp** neutralised.

## 10.6 Roller behaviour

3M, `top_up`, `roll_days=5`, 24 attempts:

```
first roll gross 13.1mm  ->  subsequent 1-8mm
total cost 16,325   positions closed 0   steady state 39-42 legs (~13 vintages x 3)
sizer's estimated cost == engine's charged option_tc, to the cent
post-trade raw vega == 0.0 exactly, across 42 aged legs
post-trade raw volga == -2500 / w^2, floating 2x over the window as vol fell
sum|reprice| 2,134  vs  sum|drift| 2,466   <- HALF the turnover is nu remeasurement
27% of leg-selections reverse the prior roll's sign on the same pillar
```

---

## Where to go next

`GUIDES/PROJECT_STATE.md` Part II holds the phase roadmap and acceptance gates.
The short version of what is outstanding:

1. **Version control.** There is no `.git` here.
2. **Bridge the `_be` buckets to realised money** — §10.2 is the open question.
3. **Phase 4, the vega re-hedge** — §10.5 sizes it; note it is a purity purchase,
   not a P&L improvement.
4. **Re-run the Phase 2 gate** through the sizer and roller, now that it is fast
   enough to sweep.
5. **Phase 5, the signal** — `target_fn` is already the seam, and implied-vs-
   realised ν (§10.2) is the natural first candidate.
