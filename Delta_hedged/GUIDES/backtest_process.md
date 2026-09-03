# FX Delta-Hedged Options Backtest — Full Codebase Guide

This document walks the entire codebase bottom-up, from the option pricer to the
`RUN()` function you actually type. It describes CURRENT behaviour — for what changed
when, and which recorded results a change invalidates, see
[CHANGELOG.md](CHANGELOG.md). Every layer gets: what it does, its exact
inputs/outputs, the formulas it uses, a runnable example, and the traps in it.

**How to read it.** The stack is strictly layered — each layer only knows about the
one below it. Read §0 for the map, then go in order. If you only want to understand
what `RUN()` is doing to your data, read §0, §18, §19.

---

# PART 0 — ORIENTATION

## 0.1 The stack

```
                        ┌────────────────────────────────────────────┐
  YOU TYPE THIS ───────▶│  §18  TEST.py :: RUN()                     │
                        │       one call = one whole experiment      │
                        └───────────────────┬────────────────────────┘
                                            │  builds ComboSpec list
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │ §16 grid_eval     │   │ §16 grid_eval     │   │ §17 gate_sweep    │
        │ pair × tenor      │   │ pair × signal     │   │ pair × gate       │
        │ (gates=None,      │   │ (signals={...})   │   │ (gates=[...])     │
        │  signals=None)    │   │                   │   │                   │
        └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
                  └───────────────────────┬┴───────────────────────┘
                                          ▼
                        ┌────────────────────────────────────────────┐
                        │  §15  reporting.py :: evaluate()           │
                        │       trade_log + daily book → scorecard   │
                        └───────────────────┬────────────────────────┘
                                            ▼
                        ┌────────────────────────────────────────────┐
                        │  §10  backtest_signal.py                   │
                        │       run_signal_backtest()                │
                        │       signal series → many trades          │
                        └──────┬──────────────────────┬──────────────┘
                               │                      │
              ┌────────────────▼─────────┐   ┌────────▼─────────────────┐
              │ §11-§12 Signal_Gen/      │   │ §13 regime_filter.py     │
              │  WHEN to enter           │   │  WHEN NOT to enter       │
              │  (0/1 series)            │   │  (0/1 veto gate)         │
              └──────────────────────────┘   └──────────────────────────┘
                               │
                               ▼
                        ┌────────────────────────────────────────────┐
                        │  §2-§7  backtest_MLeg.py                   │
                        │         run_backtest_multi_leg()           │
                        │         ONE trade, priced + hedged daily   │
                        └──────┬──────────────────────┬──────────────┘
                               │                      │
              ┌────────────────▼─────────┐   ┌────────▼─────────────────┐
              │ §8-§9 exit_hedge_logic   │   │ §1 dataset / vol_surface │
              │  ExitRule / HedgeRule    │   │    option / pricer       │
              │                          │   │    trading_calendar      │
              └──────────────────────────┘   └──────────────────────────┘
                                                          │
                                                          ▼
                                                    Bloomberg (xbbg)
```

## 0.2 File map

| File | Layer | Responsibility |
|---|---|---|
| `pricer.py` | 0 | Raw Garman-Kohlhagen formulas — price, delta, gamma, vega, vanna, volga, theta |
| `option.py` | 0 | `FXOption` wrapper + `find_strike_from_delta` root-finder |
| `vol_surface.py` | 0 | SABR smile fit; weekend-weighted ATM term-structure interpolation |
| `dataset.py` | 0 | `FXVolDataset` — spot/rates/ATM/smile lookups, closed-form `nu_BE`/`rho_BE` |
| `trading_calendar.py` | 0 | FX date conventions — spot lag, tenor rolls, holidays (QuantLib) |
| `backtest_MLeg.py` | 1 | `LegSpec`, `run_backtest_multi_leg` — **one** trade, daily reprice + hedge + attribution |
| `exit_hedge_logic.py` | 2 | Pluggable `ExitRule` / `HedgeRule` classes |
| `backtest_signal.py` | 3 | `run_signal_backtest` — loop the engine over a signal series |
| `Signal_Gen/Implied_Realized.py` | 4 | IV/RV and cross-tenor vol-spread entry signals; always-on; date-list |
| `Signal_Gen/xCCY_Spread.py` | 4 | Cross-currency ATM vol-spread entry signal |
| `Signal_Gen/regime_filter.py` | 5 | Regime **gates** — 0/1 vetoes ANDed onto a signal |
| `reporting.py` | 6 | Daily book construction, USD conversion, the metric definitions |
| `grid_eval.py` | 7 | `ComboSpec` / `run_grid` / `print_grid` — sweep many configs into one table |
| `gate_sweep.py` | 8 | `run_gate_sweep` — score many gates off one backtest, plus attribution |
| `TEST.py` | 9 | `RUN()` — the single front door you actually call |

## 0.3 One sentence each

- **Layer 0** turns a date and a pair into a price and a set of greeks.
- **Layer 1** takes a set of legs, prices them at entry, then walks forward day by day
  repricing, delta-hedging, and attributing P&L into buckets. **One trade.**
- **Layer 2** decides, each day, whether to close the trade and how much of the delta
  gap to actually hedge.
- **Layer 3** takes a 0/1 series of dates and runs Layer 1 once per qualifying date,
  respecting a concurrency cap.
- **Layer 4** produces that 0/1 series from market data (vol rich/cheap percentiles).
- **Layer 5** produces a second 0/1 series that can *veto* Layer 4's entries.
- **Layer 6** stitches every trade's daily rows into one book equity curve and computes
  the ~50 metrics you compare on.
- **Layer 7** runs Layers 3–6 once per (pair, tenor, label) cell and prints a ranked table.
- **Layer 8** exploits the fact that a gate can only *remove* entries to score dozens of
  gates from a single backtest.
- **Layer 9** is one function with every dial as a keyword argument.

## 0.4 Two conventions that run through everything

**Direction lives in `direction`, never in a negative notional.** `notional` is always
a positive magnitude; `direction=-1` makes you short. This holds at every layer.

**Base (foreign) currency is the native accounting unit.** For `USDJPY`, base = `USD`.
Every P&L number out of Layers 1–3 is in base currency. USD conversion happens once, at
Layer 6, and covers **both** lenses — the daily book and the per-trade log — off one shared
factor. See §15.4.

---

# PART I — FOUNDATIONS (LAYER 0)

## §1 Market data and pricing

You never call these directly, but four of their behaviours leak upward.

### 1.1 `FXVolDataset` (`dataset.py`)

`FXVolDataset.build(pairs=[pair], days=N)` issues the Bloomberg pulls (`blp.bdh`,
daily 5pm closes) and holds:

- **spot** — one column per pair
- **vol surface** — ATM plus risk-reversal/butterfly quotes at `{35, 25, 15, 10, 5}` delta
- **rates** — flat SOFR for USD; forward-implied yield curves for non-USD, interpolated
  log-linearly on discount factors across tenor pillars via `ql.DiscountCurve`
- **`get_smile_vol(date, tenor, K)`** — SABR smile vol at a specific strike
- **`get_smile_nu_rho(date, t_remaining)`** — closed-form vol-of-vol / spot-vol
  correlation for the breakeven P&L lens (§5.7)

Forward price is Garman-Kohlhagen: `F = S · exp((r_d − r_f) · T)`.

### 1.2 Rate convention

For a pair `XXXYYY`: **`r_d` is `YYY`'s rate, `r_f` is `XXX`'s rate.** So for `USDJPY`,
`r_d` = JPY, `r_f` = USD.

### 1.3 Smile construction (`vol_surface.py`)

SABR with **β = 0.5 fixed**, fit to Malz-style ATM/RR/BF pillar quotes at each available
tenor. Two separate interpolations:

- **ATM level across tenors** uses weekend/holiday-weighted *variance* interpolation —
  weekends carry **15%** of a normal day's variance, holidays **25%** — so a
  Friday→Monday vol node isn't overstated.
- **Smile spread** (strike vol − ATM vol) is interpolated between tenor pillars in
  `√t` space and added back on top.

### 1.4 Calendar (`trading_calendar.py`)

`add_tenor()` applies real FX market conventions: T+1/T+2 spot lag,
modified-following, month-end stickiness. Follows *Foreign Exchange Option Pricing*
§1.4 (Spot Settlement Rules) and §1.5 (Expiry and Delivery Rules). Holidays come from
the QuantLib settlement calendar.

### 1.5 Greeks are base-currency greeks

All greeks come from `FXOption.greeks_foreign()`, **not** raw domestic-currency greeks:

| Greek | Definition in this codebase |
|---|---|
| `delta` | **Premium-adjusted** spot delta — nets out the FX risk of the premium itself. This is the delta that sizes the spot hedge. |
| `gamma` | Trader's 1%-scaled gamma (`Γ_raw · S · 0.01`), **not** premium-adjusted. A separate `get_gamma_pa` exists for Taylor diagnostics but isn't what the engine reports. |
| `vega` | Scaled to a 1-vol-point move, converted to base ccy (`/S`). |
| `vanna` | Premium-adjusted (`vanna_raw − vega/S`), consistent with the PA delta. |
| `theta` | Per year, converted to base ccy (`/S`). The engine divides by 365 for daily accrual. |

---

# PART II — ONE TRADE (LAYER 1)

## §2 What `run_backtest_multi_leg()` does

Prices a **static, multi-leg European FX option position** — all legs entered the same
day, all sharing one expiry — then walks forward day by day repricing every leg off real
historical spot/rates/vol, running a **net delta hedge** on the combined position, and
attributing P&L into theta / vega / vanna / volga / gamma / hedge-carry / transaction-cost
buckets. Gamma, vanna and volga each also get a breakeven (`_be`) variant netted against
the market's own implied dynamics (§5.7).

It simulates **one trade inception** and follows it to expiry — or to today if the trade
hasn't expired, in which case the output is a live mark-to-market.

### 2.1 Example

```python
from backtest_MLeg import LegSpec, run_backtest_multi_leg, print_pnl, print_exposures
from exit_hedge_logic import HoldToExpiry, DailyHedge

legs = [
    LegSpec('put',  -0.10, +1, 75_000_000),   # long 10d put
    LegSpec('put',  -0.30, -1, 75_000_000),   # short 30d put
]

df_agg, leg_dfs, summary, leg_summaries = run_backtest_multi_leg(
    legs,
    pair='USDJPY',
    tenor='1M',
    verbose=True,
    hedge_rule=DailyHedge(),
    exit_rule=HoldToExpiry(),
)

print_pnl(df_agg, leg_summaries=leg_summaries, summary=summary)
print_exposures(df_agg, leg_summaries=leg_summaries, summary=summary)
```

### 2.2 Convenience wrappers

Six `run_backtest_*` wrappers exist. They build a `List[LegSpec]` and call the engine —
pure convenience, no extra logic.

```python
# ATMF straddle
run_backtest_straddle(pair, tenor, notional, direction,
                      entry_days_back, history_days, exit_rule, hedge_rule)

run_backtest_strangle(pair, tenor, call_delta, put_delta, notional, direction, ...)

# dir=+1: long call / short put   |   dir=-1: short call / long put
run_backtest_risk_reversal(pair, tenor, call_delta, put_delta, notional, direction, ...)

# dir=+1: long low-strike / short high-strike
run_backtest_call_spread(pair, tenor, long_delta, short_delta, notional, direction, ...)

# dir=+1: long high-strike / short low-strike
run_backtest_put_spread(pair, tenor, long_delta, short_delta, notional, direction, ...)

# dir=+1: short 50d straddle / long wings. Straddle notional is SOLVED so the
#         structure's initial net vega = 0.
run_backtest_vega_neutral_butterfly(pair, tenor, wing_delta, direction,
                                    wing_notional, ...)
```

## §3 `LegSpec` — defining a position

```python
LegSpec(option_type, target_delta, direction, notional)
```

| Field | Meaning |
|---|---|
| `option_type` | `'call'` or `'put'` |
| `target_delta` | **Signed spot delta** used to solve for the strike at entry — positive for calls (`+0.25` = 25∆ call), negative for puts (`-0.25` = 25∆ put). ATM ≈ `±0.50`. |
| `direction` | `+1` = long the leg, `-1` = short the leg |
| `notional` | Base-currency notional for **this leg**, always a positive magnitude |

Each leg is priced independently (own strike, own smile vol) but all legs in one call
share **one spot** and **one expiry**. It **cannot express calendar spreads.**

### 3.1 Structure builders

`build_straddle`, `build_strangle`, `build_risk_reversal`, `build_call_spread`,
`build_put_spread`, `build_vega_neutral_butterfly` return a `List[LegSpec]`.

`vega_neutral_butterfly_factory` is different in kind: the vega-neutral fly needs to
know entry-date market data to solve the straddle notional, so it returns a *factory*
that the engine calls once it has that data. This is why Layer 7 has both a `legs_fn`
and a `legs_factory_fn` slot (§16.3).

## §4 Parameter reference

```python
run_backtest_multi_leg(
    legs,                      # List[LegSpec], required
    pair='USDJPY',
    tenor='1M',                # tenor string ('1D','1W','1M','3M','1Y',...) or int days
    entry_days_back=None,      # int: force entry N calendar days before today
    history_days=365,          # minimum days of market data to pull (auto-adjusts up)
    tc_fraction=0.0001,        # hedge transaction cost, fraction of hedge notional traded
    verbose=True,              # print entry/summary report
    exit_rule=None,            # ExitRule instance; defaults to HoldToExpiry()
    hedge_rule=None,           # HedgeRule instance; defaults to DailyHedge()
    dataset=None,              # pre-built FXVolDataset override (used by Layer 3)
    entry_date=None,           # explicit entry date override (used by Layer 3)
)
```

**How the entry date is chosen:**

- `entry_days_back` given → `entry_date ≈ today − entry_days_back`, rolled back to the
  preceding business day on the pair's FX calendar.
- Neither given → `entry_date ≈ today − tenor`. The default therefore backtests **the
  most recent trade of that tenor that would be maturing around now.**

`expiry` follows from `add_tenor(entry_date, tenor)` (§1.4). **If `expiry > today` the
trade is live**: the loop still runs on whatever history exists, and the output is a
live mark-to-market. `summary['live_trade']` flags it.

Returns `(df_agg, leg_dfs, summary, leg_summaries)` — see §6.

## §5 Inside a run — step by step

### 5.1 Date + calendar resolution

`entry_date`, `expiry`, and `option_tenor_days = (expiry − entry_date).days` are computed
once. `exit_rule` and `hedge_rule` each get `bind(entry_date, fxc, option_tenor_days)`,
which is what lets `ExitAfterNDays('1W')` / `ExitAtDaysRemaining('1W')` resolve a tenor
string into a day count against the real calendar.

### 5.2 Market data snapshot

`FXVolDataset.build(pairs=[pair], days=dataset_days)` covers at least `entry_date`
through `max(expiry, today)` plus a 10-day buffer. `S0`, `r_d`, `r_f`, `sigma_atm0` are
read as of `entry_date`.

Two sanity asserts fire here: `0 < sigma_atm0 < 2.0` and `abs(r_d), abs(r_f) < 1` — they
catch a percent/decimal unit mixup (passing `8.5` instead of `0.085`).

### 5.3 Per-leg bootstrap: target delta → strike → smile vol

A "25∆ put" isn't a strike — it's a delta, and delta depends on the strike *through the
smile vol at that strike*. The engine breaks the circularity with a fixed-point loop,
per leg:

1. Seed with ATM vol; solve for the strike `K` giving exactly `target_delta`
   (`find_strike_from_delta`, a `brentq` root-find bounded to `[0.85·S0, 1.20·S0]`).
2. Look up the SABR smile vol *at that strike* (`dataset.get_smile_vol`).
3. Re-solve for `K` with the new vol. Repeat up to **5** times or until the vol change
   is `< 1e-8`.

The final `(K, sigma)` is self-consistent. An `FXOption` is built from that strike;
entry price / greeks / premium are stored per leg.

**Caveats:**
- No explicit convergence check after the 5th iteration — if it hasn't converged (only
  plausible in extreme skew / very short dated) it silently uses the 5th pass.
- The `brentq` bracket is fixed at `S0 · [0.85, 1.20]`. A very deep delta (5∆) with high
  vol or a long tenor can put the true strike outside it, which **flags** rather than
  degrading gracefully.

### 5.4 Entry portfolio quantities

- **Net delta / initial hedge** — `net_delta_entry = Σ delta_pa[i] · notional[i] · direction[i]`;
  `initial_hedge = −net_delta_entry`. Carried in the same units as leg notional
  (base-ccy notional of a spot position), sign-negated to offset the book's delta.
- **Net premium** — `net_premium = Σ premium[i] · direction[i]`.
  **Convention: positive `net_premium` = premium owed (a debit).** The verbose printout
  displays `−net_premium` as "cashflow" so a positive display reads as cash received.
  Per leg, `leg_summaries[i]['premium'] = premium · direction`.
- **Vega-weighted average entry vol** (`avg_entry_sigma`) — reference level for the
  realized-vs-implied comparison in the summary, weighted by `abs(vega · notional)`
  rather than notional, so a deep-OTM leg with negligible vega doesn't drag the
  reference. Falls back to notional-weighting in the all-zero-vega edge case.

### 5.5 The daily loop

Iterates `dataset.spot.loc[entry_date:series_end, pair]`, skipping the entry date. Per date:

**1. Start-of-period ("prev") greeks** for every leg, off *yesterday's*
spot/vol/rates/date. All of that day's attribution is priced off these. Consequence:
**the greeks in each output row describe the position going *into* that day, not coming
out of it.**

**2. Hedge P&L and carry** — portfolio-level, computed off `current_hedge`, i.e. the
hedge carried in from yesterday, *not* today's just-computed target:

```
hedge_pnl   = current_hedge · dS / prev_spot
hedge_carry = current_hedge · (r_f − r_d) · dt_days / 365
```

`dt_days` is the actual calendar gap since the last row (3 over a weekend), not a fixed 1.

**3. Per-leg theta P&L** — `theta[i] · notional[i] / 365 · dt_days · direction[i]`, summed.

**4. Gamma P&L** — a direct analytic Taylor term per leg:

```
gamma_pnl_leg = 0.5 · notional · dS² · (Γ_raw/S − 2·Δ_pa/S²)
```

the exact 2nd derivative of the base-ccy option price w.r.t. spot (the `Δ_pa` term folds
in the correction for the premium itself being base-ccy denominated). Closed form from
start-of-day greeks and realized `dS` — it does **not** depend on `hedge_pnl` or any
other bucket.

**5. Repricing / vol P&L** (skipped on the natural-expiry day, §5.6):
- Each leg is **fully repriced** at the new spot/vol/rates → `option_pnl`. Exact, not Taylor.
- The vol buckets are Taylor approximations off start-of-day greeks:
  ```
  vega_pnl  = vega  · notional · (dsigma/0.01)               · direction
  vanna_pnl = vanna · dS · dsigma / prev_spot · notional     · direction
  volga_pnl = 0.5 · volga · dsigma² / prev_spot · notional   · direction
  ```
- **`recon_resid`** (also exposed as `hedge_drift_pnl`, same value) is a diagnostic: the
  gap between the *exact*-reprice-implied convexity P&L
  (`net_option_pnl + hedge_pnl − theta_pnl − vega_pnl − vanna_pnl − volga_pnl`, computed
  internally, not exposed as a column) and the analytic `gamma_pnl`. A large value means
  either a big discrete jump (gap risk) or that the Taylor approximations broke down.

**6. Exit-rule check** — builds an `ExitContext` (today's date, days in trade, days
remaining, spot, entry spot, vega-weighted vol today and at entry, and `running_net_pnl`
= cumulative P&L including today, pre-exit-tc) and calls `exit_rule.check(ctx)`.

- **`True`** → `target_hedge = 0.0`, `hedge_fraction = 1.0`. The whole hedge is
  force-unwound today (no partial unwind), position closed at today's mark. The loop
  records the row and **breaks**, `exit_reason` = the rule's `.name`.
- **`False`** → net delta is recomputed off today's fresh greeks, and
  `hedge_rule.decide(ctx)` returns a fraction in `[0,1]` of the gap between
  `current_hedge` and `target_hedge` to trade today.

  **Important:** skipping or partial-hedging only changes `hedge_pnl` / `hedge_carry` /
  `tc` going forward. It never changes `option_pnl`, `theta_pnl`, `vega_pnl`, etc., which
  come purely from the option's own greeks and the realized market move.

**7. Transaction cost & net P&L**

```
hedge_gap   = target_hedge − current_hedge      (pre-trade)
hedge_trade = hedge_fraction · hedge_gap
tc          = |hedge_trade| · tc_fraction
net_pnl     = option_pnl + hedge_pnl + hedge_carry − tc
```

TC is charged **only** on hedge rebalancing. Entering or exiting the option legs carries
no bid/offer cost in this model.

`rehedged` (`|hedge_trade| > 1e-9`) and `days_since_hedge` (calendar days since
`hedge_position` last changed, reset to 0 on a rehedge) are recorded purely as
diagnostics — they don't feed back into any P&L.

**8. State roll-forward** — `current_hedge += hedge_trade`, stored in the row as
`hedge_position` (outstanding spot hedge notional carried into tomorrow; negative =
short). `prev_*` roll to today's values.

### 5.6 Natural expiry settlement (special-cased)

If `t_remaining <= 0`, the loop takes a different branch:

- Each leg settles to **intrinsic value** — `max(S−K, 0)` for calls, `max(K−S, 0)` for
  puts, divided by `S_new` for base-ccy terms.
- `vega_pnl` / `vanna_pnl` / `volga_pnl` / `vanna_pnl_be` / `volga_pnl_be` are hard-set
  to `0.0` (no vol sensitivity left). `nu_be` / `rho_be` also `0.0`.
- `gamma_pnl` and `gamma_pnl_be` are computed exactly as on any other day — they never
  depended on the vol greeks.
- Loop **breaks** with `exit_reason = 'expiry'`.

**No hedge trade or transaction cost is charged on this final unwind.** The terminal
hedge is netted into `hedge_pnl` for the last interval, but there's no simulated
close-out cost. **This is an asymmetry versus an early exit (step 6), which does pay a
full-cost unwind** — it matters whenever you compare hold-to-expiry against an early exit.

If the trade is still live, the loop simply runs out of spot data and ends on an ordinary
row with no `exit_reason`.

### 5.7 Breakeven (`_be`) P&L — realized-vs-implied attribution

A second attribution layer on top of the plain Taylor buckets, following Ravagli,
*"Harvesting the FX skew premium"* (Risk.net, June 2024). Each nets the realized move
against what the market's own implied dynamics predicted, leaving only the surprise:

- **`gamma_pnl_be = gamma_pnl + theta_pnl`** — realized vs implied variance scaled by
  gamma; algebraically `0.5·S²·Γ·((dS/S)² − σ²·dt)`. This works because BS theta is, to
  leading order, `−0.5·S²·Γ·σ²`, so adding it back cancels the *expected* variance the
  position was priced for.
- **`vanna_pnl_be` / `volga_pnl_be`** subtract the expected value of `dS·dsigma` /
  `dsigma²` implied by a stochastic-vol model with known vol-of-vol and spot-vol
  correlation:
  ```
  vanna_pnl_be = vanna · (dS·dsigma − prev_spot·σ0²·ρ_BE·ν_BE·dt_years)/prev_spot · notional · direction
  volga_pnl_be = 0.5 · volga · (dsigma² − σ0²·ν_BE²·dt_years)/prev_spot · notional · direction
  ```
  where `σ0` is that leg's own start-of-day **smile** vol, not ATM vol.

**`nu_BE` / `rho_BE` sourcing** (`FXVolDataset.get_smile_nu_rho`) is a **closed-form read
of quoted market data**, not a numerical fit:

```
nu_BE  = NU_BE_C  * sqrt(Fly25 / (tau * sigma_ATM))     # NU_BE_C  = 4.0
rho_BE = RHO_BE_D * (RR25 / sigma_ATM)                  # RHO_BE_D = 2.5
```

using 25-delta butterfly / risk-reversal / ATM vol at the tenor pillar **nearest**
`t_remaining` — nearest-pillar, not `√t`-interpolated the way the smile level/spread is,
because nu/rho describe the smile's whole shape at that pillar rather than a
strike-specific value, and feed a diagnostic bucket rather than the option's price.

Guards: `MIN_PILLAR_DAYS` = 7 calendar days (excludes the 'ON' pillar) floors the pillar
search, since `tau → 0` makes `nu_BE` blow up algebraically; `NU_BE_MAX` = 5.0 is a
defensive clip. Falls back to `(0.0, 0.0)` — collapsing the `_be` buckets to the plain
ones — if the pillar's ATM/RR25/BF25 quotes are missing.

This replaced a per-date SABR-fitted nu/rho + EWMA-smoothing approach: the closed form
reads straight off quoted RR/BF/ATM with no calibration, so there's no
convergence/instability to smooth away, and it's orders of magnitude cheaper
(microseconds vs tens of milliseconds per date).

## §6 Reading Layer 1's outputs

Returns `(df_agg, leg_dfs, summary, leg_summaries)`.

**`df_agg`** — one row per backtest day, indexed by date, portfolio-level plus every
`*_legN` column:

| Group | Columns |
|---|---|
| Market | `spot`, `dS`, `sigma` (vega-weighted, off the same start-of-day snapshot as that day's attribution) |
| Greeks (all start-of-day, signed/netted) | `delta`, `gamma_1pct`, `vega_1vp`, `vanna_1vp`, `volga_1vp`, `theta_daily` |
| P&L buckets | `option_pnl`, `hedge_pnl`, `hedge_carry`, `gamma_pnl`, `gamma_pnl_be`, `theta_pnl`, `vega_pnl`, `vanna_pnl`, `volga_pnl`, `vanna_pnl_be`, `volga_pnl_be`, `tc`, `net_pnl` |
| BE inputs | `nu_be`, `rho_be` |
| Diagnostic | `recon_resid` / `hedge_drift_pnl` (same value, two names) |
| Hedge mechanics | `hedge_gap`, `hedge_trade`, `hedge_fraction`, `hedge_position`, `days_since_hedge`, `rehedged` |
| Variance check | `realised_var`, `implied_var`, `var_spread` |

**`leg_dfs`** — `leg_dfs[i]` is the same information sliced to leg `i` (suffix stripped),
plus shared `t_remaining` / `spot` / `dS`.

**`summary`** — a `pd.Series`: entry/exit dates, `live_trade`, `exit_reason`,
`days_held`, the full P&L breakdown summed over the trade's life, plus `realised_vol` vs
`avg_entry_sigma` (annualized). Also carries `vol_spread` and `atm_entry_vol`, which
Layer 6 uses for the mechanism metrics (§15.3).

**`leg_summaries`** — one `pd.Series` per leg: strike, entry vol, signed premium, that
leg's own summed attribution.

`print_pnl()` / `print_exposures()` (and `build_pnl_views()` / `build_exposure_views()`)
are formatted re-slices — they compute nothing new. **Note the split:** `hedge_pnl`,
`hedge_carry` and `net_pnl` are **portfolio-level only** (they depend on the single shared
hedge) and are not broken out per leg. Everything else is attributable per leg and
appears in both views.

## §7 Layer 1 assumptions & limitations

- **One expiry, one hedge, per call.** All legs share a single netted spot hedge. No
  per-leg hedge, no calendar spreads.
- **Static position.** Entered once, only ever fully closed. No adding/rolling/resizing.
- **No cost on the terminal unwind at natural expiry** (§5.6) but full cost on an
  `exit_rule` close. Baked in, not a bug to route around.
- **No cost on option entry/exit at all** — `tc_fraction` only ever multiplies
  `|hedge_trade|`.
- **Gaps are bridged, not rehedged through.** `dt_days` scales theta and carry correctly,
  but gamma/vega P&L over a Friday→Monday gap is realized as one lump move using Friday's
  start-of-day greeks. Standard daily-bar limitation; worth remembering when a weekend
  shows an outsized `gamma_pnl`.
- **The P&L buckets do not sum to the exact reprice.**
  `theta + vega + vanna + volga + gamma` will generally **not** equal
  `option_pnl + hedge_pnl`. The gap is `recon_resid`. This mirrors Ravagli 2024's own
  framing — "Greek P&L" there is an independent reconstruction checked against actual P&L
  by R² (never 100%, even with all eight greeks), not an exact decomposition. Use
  `recon_resid` as a sanity check, not an error to drive to zero.
- **`NU_BE_C` / `RHO_BE_D` (4.0 / 2.5) are validated on G10, not EM/LatAm.** Worth
  checking against a full SABR fit before trusting the `_be` buckets on EM crosses.
- **Strike bootstrap is a 5-iteration fixed point** bounded to `S0 · [0.85, 1.20]`.
- **Default entry-date logic tests "the trade about to mature now"** — not "a
  representative past 1M trade." To study many windows you need Layer 3.
- **Live trades are mark-to-market, not a forecast.**
- **Bloomberg dependency** — results are only as good as the pull, and connectivity is
  required.

---

# PART III — RULES (LAYER 2)

## §8 Exit rules (`exit_hedge_logic.py`)

All subclass `ExitRule`, implement `check(ctx: ExitContext) -> bool`, optionally override
`bind()` for calendar-aware setup.

| Rule | Behaviour |
|---|---|
| `HoldToExpiry()` | Default — never exits early. |
| `ExitAfterNDays(n)` | Close once `days_in_trade >= n`. `n` is an int (calendar days) or a tenor string (`'1W'`), resolved via the pair's FX calendar in `bind()`. **Must resolve to fewer days than the trade's own tenor.** |
| `ExitAtDaysRemaining(n)` | Mirror image — close once `t_remaining <= n`. Same int/string handling and the same assertion. |
| `TakeProfitStopLoss(take_profit=, stop_loss=)` | Close once cumulative `net_pnl` crosses either bound. Either side optional; `stop_loss` is a positive magnitude. |

**Trap.** `ExitAtDaysRemaining('1W')` asserts the threshold is shorter than the trade's
own tenor, so it raises on `tenors=['1W']`. In a grid this surfaces as `status='error'`
rows for those cells rather than a crash (§16.4).

## §9 Hedge rules (`exit_hedge_logic.py`)

All subclass `HedgeRule`, implement `decide(ctx: HedgeContext) -> float` — the fraction of
today's hedge gap to actually trade.

| Rule | Behaviour |
|---|---|
| `DailyHedge()` | Default — always `1.0`. Full rehedge every day. |
| `DeltaBandHedge(band)` | Rehedge fully once `\|net_delta\| / total_notional > band`. |
| `SpotMoveHedge(bps)` | Rehedge fully once spot has moved more than `bps` from the level at the **last actual hedge trade**. Stateful — tracks its own last-hedge spot, seeded from entry spot. |
| `GammaScaledHedge([(gamma_frac, interval_days), ...])` | Schedule keyed off `\|net_gamma\| / total_notional` — higher gamma, shorter interval. Uses whichever threshold is exceeded; falls back to the lowest-threshold interval. |
| `PartialHedge(fraction)` | Always trades a fixed `fraction` of the gap, every day — a permanent partial hedge rather than an on/off trigger. |

**Statefulness is why every layer above takes factories, not instances.**
`SpotMoveHedge._last_hedge_spot` and `GammaScaledHedge._days_since_hedge` are set in
`__init__` and **not** reset by `bind()`. Sharing one instance across two trades leaks
trade 1's state into trade 2 — silently wrong, not an error.

---

# PART IV — MANY TRADES (LAYER 3)

## §10 `run_signal_backtest()` (`backtest_signal.py`)

A thin orchestration layer. It doesn't reprice or hedge anything — it decides *when* to
call `run_backtest_multi_leg()` and stitches the per-trade outputs into one trade log.

### 10.1 Example

```python
import pandas as pd
from backtest_MLeg import LegSpec, print_pnl
from exit_hedge_logic import HoldToExpiry, DailyHedge
from backtest_signal import run_signal_backtest, print_trade_log

signal = pd.Series(...)   # 0/1, indexed by date — 1 = enter a new trade

legs = [                                      # long 25d strangle, 10mm per leg
    LegSpec('put',  -0.25, +1, 10_000_000),
    LegSpec('call', +0.25, +1, 10_000_000),
]

trade_log, trade_dfs, trade_leg_sums = run_signal_backtest(
    signal, legs,
    pair='USDJPY', tenor='1M',
    hedge_rule_factory=lambda: DailyHedge(),
    exit_rule_factory=lambda: HoldToExpiry(),
    max_concurrent=1,
    verbose=False,
)

print_trade_log(trade_log)

# Drill into any one trade exactly like a standalone engine call —
# trade_dfs[i] / trade_leg_sums[i] / trade_log.iloc[i] line up by position.
print_pnl(trade_dfs[0], leg_summaries=trade_leg_sums[0], summary=trade_log.iloc[0])
```

### 10.2 Parameter reference

```python
run_signal_backtest(
    signal,                     # pd.Series, 0/1 (or truthy), indexed by date
    legs,                       # List[LegSpec] — reused unmutated for every trade
    pair='USDJPY',
    tenor='1M',
    hedge_rule_factory=DailyHedge,   # zero-arg callable -> FRESH HedgeRule per trade
    exit_rule_factory=HoldToExpiry,  # zero-arg callable -> FRESH ExitRule per trade
    tc_fraction=0.0001,
    history_days=365,
    max_concurrent=1,           # None = unlimited, 1 = non-overlapping, N = cap
    verbose=False,
)
```

`legs` is a plain fixed list shared across every trade — `LegSpec` is an immutable
dataclass the engine only reads, so that's safe. **The rules are the opposite** — see §9.

### 10.3 Inside a run

**One shared `FXVolDataset`, built once.** Re-pulling per trade would mean dozens of
redundant overlapping Bloomberg pulls. Instead it estimates a tenor-length day count from
the *earliest* signal date (`add_tenor(earliest_entry, tenor, fxc)` — the same calendar
logic the engine uses), builds one dataset covering the whole window, and passes it into
every engine call via the `dataset=` override. Each trade also passes its own resolved
`entry_date` directly, bypassing the from-today derivation in §4.

**The entry scan is a single day-by-day pass gated by `max_concurrent`.** It walks
`signal` in date order maintaining `active_exits` — the exit dates of every open trade.
On each date it first drops any trade whose `exit_date` is **strictly before** today (a
trade's own exit day still occupies its slot; it frees the day after), then, if `signal`
is truthy and a slot is free, opens a trade and adds its exit date.

- **`max_concurrent=1`** — non-overlapping. A signal while a trade is open is ignored
  entirely; the next entry is the day after the open trade's `exit_date`.
- **`max_concurrent=N`** — up to N open at once; signals beyond that are ignored until a
  slot frees.
- **`max_concurrent=None`** — unlimited. A new trade on *every* `signal==1` date, pure
  stacking.

Because each entry requires actually running the engine to learn its `exit_date` before
the scan can know when a slot frees, this is **inherently sequential** — you cannot
precompute the entry dates without executing every trade in order.

**Concurrent trades are simulated fully independently.** Each is its own engine call with
its own strike bootstrap, its own hedge, its own P&L. There is **no netting of deltas or
hedges across overlapping trades** — two open trades are two parallel spot hedges, not one
netted portfolio hedge. A single netted hedge across overlapping positions would be a
materially different engine, not something `max_concurrent > 1` gives you.

### 10.4 Outputs

`(trade_log, trade_df_aggs, trade_leg_sums)`:

- **`trade_log`** — one row per trade: every field from the engine's `summary` (§6), plus
  **`concurrent_at_entry`**, how many trades were already open when this one entered
  (before it was added to `active_exits`). A direct check that `max_concurrent` is being
  respected. Empty `DataFrame` if the signal never fires.
- **`trade_df_aggs`** — list of each trade's `df_agg`, in `trade_log` row order.
- **`trade_leg_sums`** — list of each trade's `leg_summaries`, same ordering.

`print_trade_log(trade_log)` is a formatted view plus headline stats (total/average net
P&L, win rate, average days held).

### 10.5 Layer 3 limitations

- **The signal only decides entries, never exits.** Each trade runs to its own exit rule
  regardless of what the signal does afterward. No built-in "exit when the signal flips
  to 0" — that needs a custom `ExitRule` reading the signal inside `check()`.
- **`max_concurrent` controls trade *count*, not risk.** N concurrent full-size trades
  carry N× the risk. There is no position-sizing layer.
- **No portfolio-level daily P&L series here.** That's Layer 6's job (§14).
- **Point-in-time discipline is the caller's responsibility.** The engine only looks at
  data as of each trade's `entry_date`, but nothing checks that your *signal* was built
  without look-ahead.
- **One Bloomberg pull per call, not per trade** — but repeated calls (across pairs, or a
  parameter sweep) each re-pull. No cross-call dataset caching.

---

# PART V — SIGNALS (LAYER 4)

Everything in `Signal_Gen/` exists to produce one object: a **0/1 `pd.Series` indexed by
date**, where 1 = "consider entering here." It carries **no direction information** —
long vs short comes from `LegSpec.direction`, entirely separately.

## §11 `Signal_Gen/Implied_Realized.py`

### 11.1 The data primitives

**`get_ImplRealVol(currency_list, tenors, days_back)`** — one batched `blp.bdh` call
building every ticker up front:

```
ticker  f"{ccy}{V|H}{tenor} Curncy"     e.g. 'USDJPYV1M Curncy',  'USDJPYH1W Curncy'
column  f"{ccy}_{V|H}{tenor}"           e.g. 'USDJPY_V1M',        'USDJPY_H1W'
```

`V` = implied vol, `H` = Bloomberg's **already-annualized realized** vol over that
trailing window. Because `H` is pre-annualized, nothing downstream ever does per-day
return math.

**`get_Allvol_df(ccys, tenors, days_back)`** — matched-tenor view. For each (ccy, tenor)
it emits up to three columns:

```
{ccy}_{tenor}_VDiff   = V{tenor} − H{tenor}     (implied minus realized, same tenor)
{ccy}_{tenor}_IV      = V{tenor}
{ccy}_{tenor}_HV      = H{tenor}
```

`VDiff` requires both legs; the pair is `dropna()`'d to common dates first.

**`build_vol_spread(ccys, leg_a, leg_b, days_back, label='VDiff')`** — the general
two-leg spread, where each leg is a `(vol_type, tenor)` tuple and **the tenors need not
match**. This is what `get_Allvol_df` cannot express.

```python
build_vol_spread(ccys, ('V','1M'), ('H','1W'))   # 1M implied − 1W realized
build_vol_spread(ccys, ('V','1M'), ('V','1W'))   # IV term structure 1M − 1W
build_vol_spread(ccys, ('V','1W'), ('H','1W'))   # matched VDiff
```

Output column name: `f"{ccy}_{IV|RV}{ta}-{IV|RV}{tb}_{label}"`, e.g.
`'USDJPY_IV1M-RV1W_VDiff'`. The trailing `_{label}` is load-bearing — the squeeze logic
in `get_vol_signal` keys off that suffix.

### 11.2 Percentile → signal

**`compute_days_needed(pct_lookback, days_back, holiday_buffer_pct=0.15)`** — converts a
trading-day lookback into the calendar days of history to pull:

```
int(pct_lookback · 7/5 · 1.15) + days_back
```

**`rolling_percentile(df, lookback_days=252, days_back=180)`** — the general primitive:

```python
pct = df.rolling(lookback_days, min_periods=20).rank(pct=True) * 100
```

Row-based window, **current point included** → point-in-time, no look-ahead. Then trimmed
to the last `days_back` calendar days. Unlike `compute_percentile_timeseries_fast` it does
**not** reconstruct column names from a ccy/tenor/metric convention — it ranks whatever
you hand it and passes names through, which is what makes it reusable for cross-tenor
spreads.

**`compute_signals(pct_ts, buy_pct=20, sell_pct=80)`** — a raw three-state series per column:

```
-1  where pct > sell_pct     (rich  → sell vol)
+1  where pct < buy_pct      (cheap → buy vol)
 0  otherwise
<NA> where pct is NaN        (nullable Int64, so warm-up stays distinguishable from 0)
```

**`_entry_series_from_signal(raw, side)`** — collapses that to 0/1:

| `side` | fires on |
|---|---|
| `'buy'` | `raw == +1` — percentile below `buy_pct` |
| `'sell'` | `raw == −1` — percentile above `sell_pct` |
| `'both'` | `raw != 0` — either extreme |

### 11.3 The three one-call signal builders

**`get_cross_vol_signal(ccys, leg_a, leg_b, ...)`** — the one your `_ivrv` uses.

Pipeline: `compute_days_needed` → `build_vol_spread` → `rolling_percentile` →
`compute_signals` → squeeze. Returns `(pct_metric, signal_metric, signal_series)`.
**Asserts exactly one column**, so pass exactly one ccy.

```python
pct, sig, signal_series = get_cross_vol_signal(
    ['USDJPY'], ('V','1M'), ('H','1W'),
    pct_lookback=252, days_back=180, sell_pct=80, side='sell')

print(signal_series.tail(6))
# 2026-08-03    0
# 2026-08-04    1
# 2026-08-05    1
# 2026-08-06    1
# 2026-08-07    0
# 2026-08-10    0
# Name: USDJPY_IV1M-RV1W_VDiff, dtype: int64
```

**`get_vol_signal(ccys, tenors, metric, ...)`** — the matched-tenor equivalent, via
`get_Allvol_df` → `compute_percentile_timeseries_fast`.

**Trap:** the docstring says `metric` accepts `'IV'`, `'RV'`, or `'IV-RV'`, but the code
asserts `metric in {'IV', 'RV', 'VD'}` and maps `'VD' → 'VDiff'`. **Pass `'VD'`, not
`'IV-RV'`** — the latter raises. Also asserts exactly one matching column.

**`get_always_on_signal(days_back, pair, tenor)`** — all-ones on the pair's **own
Bloomberg business dates**, read off the `f"{pair}_V{tenor}"` column's non-NaN index, then
trimmed to `<= today`.

Two things matter here. It uses the traded pair's own calendar, so the entry dates line up
with the pair actually being traded — an earlier hard-coded EURUSD calendar silently
mismatched USDJPY backtests. And the trailing trim exists because an entry within one
tenor of today has no post-inception market data, which makes the engine build an empty
frame and raise `KeyError("None of ['date'] are in the columns")`.

**`get_date_signal(dates, pair, tenor)`** — fires only on the given dates
(`'%d%b%y'` format, e.g. `'24Jul26'`), 0 elsewhere. History window derived automatically from the
earliest date. Requested dates are unioned into the calendar even if outside the pulled
history, since Layer 3 resolves each signal date to the preceding business day anyway.

## §12 `Signal_Gen/xCCY_Spread.py`

### 12.1 `get_ATM_XCCY_Spread_TimeSeries(ccy_int, tenor_int, ccys, pct_lookback, days_back)`

Asks: **is this pair's ATM vol rich relative to a basket of its peers?** Six steps:

1. **Pull** `f"{ccy}V{tenor} BGN Curncy"` for every ccy in the basket (one `bdh` per
   ticker, wrapped in `try/except` so a missing ticker is skipped), then
   `pd.DataFrame(data_dict).dropna()`.
2. **Spreads** — `spreads[other] = df_all[ccy_int] − df_all[other]` for every other ccy.
3. **Rolling percentile** of each spread within a trailing **time-based** window
   `f'{pct_lookback}D'`, via `percentileofscore(x, x[-1])`.
4. **Rolling correlation** of daily *changes* (`df_all.diff()`) between `ccy_int` and each
   other ccy, same window.
5. **Combine** — `weights = corr.clip(lower=0)`, then
   `val = Σ(pct · w) / Σ w`. So peers that co-move more with `ccy_int` count more, and
   negatively-correlated peers are dropped rather than inverted.
6. **Trim** to `iloc[-days_back:]`.

Output: a single `pd.Series` named `f"{ccy_int}_{tenor_int}_XCCY_Pct"`, already a 0–100
percentile.

**Three traps specific to this module:**

- **The rolling window is time-based (`'365D'`), not row-based**, unlike
  `rolling_percentile`'s 252-*observation* window in §11.2. Default `pct_lookback=365`
  here vs `252` there — they are not the same quantity, and 365 calendar days ≈ 252
  trading days, so the defaults happen to be roughly comparable. Don't assume that if you
  change one.
- **`days_back` trims ROWS, not calendar days** — `iloc[-days_back:]`. Passing
  `days_back=1825` (your 5 years) returns *every* available row, since there are only
  ~1250 trading days. Harmless, but it means the trim is a no-op at your settings, and it
  is not the same trim `rolling_percentile` applies.
- **`df_all.dropna()` is an inner join across the whole basket.** One ticker with a short
  or gappy history truncates the entire series for every pair. Worth checking basket data
  coverage before trusting a short output.

### 12.2 `get_xccy_spread_signal(...)`

`get_ATM_XCCY_Spread_TimeSeries` → `compute_signals` → `_entry_series_from_signal`. Note
`buy_pct`/`sell_pct` are applied **directly** to the percentile that function already
returns — there is no second `rolling_percentile` step, because the percentile is computed
internally.

```python
pct, sig, signal_series = get_xccy_spread_signal(
    'USDJPY', '1M',
    ccys=['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','NZDUSD','USDCAD'],
    days_back=180, sell_pct=80, side='sell')
```

Raises `ValueError` if the underlying series comes back empty.

---

# PART VI — REGIME GATES (LAYER 5)

## §13 `Signal_Gen/regime_filter.py`

### 13.1 The problem it solves

Layer 4 decides *when the carry looks good*. None of those signals know *what regime vol
is in*, so a "sell" can fire exactly as vol is breaking higher. A **gate** is a second
0/1 series:

```
1 = regime is OK, let the entry through
0 = veto this entry (vol is high and/or rising — stand aside)
```

It composes by multiplication, because Layer 3 only cares whether the signal is truthy:
`filtered = apply_regime_gate(signal, gate)`.

### 13.2 Three layers, so you can work at whichever you need

| Layer | Object | Use |
|---|---|---|
| 1 | `check_*(series, ...) -> bool Series` | Pure, testable, plottable on its own |
| 2 | `build_check_panel(...) -> DataFrame` | Every selected check as one boolean column; built **once** per (pair, tenor, params) and cached |
| 3 | `GateSpec` + `gated(signal_fn, spec)` | A named, reusable gate config. This is the layer the grid uses |

**True always means "OK to sell vol."** Every check uses only trailing windows. During
warm-up the default is to **ALLOW** — absence of evidence is not evidence of a bad regime,
and vetoing a third of the sample silently would be worse. `_allow_on_warmup(ok, valid)`
enforces that.

### 13.3 The six registered checks

| Name | Asks | Default params | Warm-up |
|---|---|---|---|
| `trend` | Is front IV ≤ its own trailing MA? | `ma_window=20` | ~20d |
| `momentum` | Has front IV *not* risen over the last n days? | `change_days=5` | ~6d |
| `termstructure` | Is the curve *not* backwardated (front ≤ back + buffer)? | `buffer=0.0` | 2d |
| `level` | Is IV's level percentile ≤ cap? | `lookback=252, cap_pct=90.0` | ~252d |
| `spike` | No > z-score IV jump in the last `cooloff` days? | `z_window=60, z_thresh=2.0, cooloff=5` | ~65d |
| `har` | Is the HAR RV forecast *not* expanding vs recent realized? | `bbg_tenors=('1W','1M','3M'), train_window=252, refit_every=5, rising_ratio=1.0` | ~274d |

Details worth knowing:

- **`trend`** — the registry default is `ma_window=20` even though the pure function
  `check_trend` defaults to 5. `min_periods` is derived as `max(5, ma_window//2)` and is
  **not** settable through `GateSpec`.
- **`momentum`** — a raw n-day difference. Sharper and more reactive than `trend`, and
  largely redundant with it since both read front-IV direction.
- **`termstructure`** — cheapest warm-up, one parameter, hardest to overfit. Needs two
  *different* tenors; it raises if front == back. The back tenor defaults from
  `_NEXT_TENOR` (`1D/24H→1W, 1W→1M, 2W→1M, 3W→2M, 1M→3M, 2M→3M, 3M→6M, 4M→6M, 6M→1Y,
  9M→1Y, 1Y→2Y, 2Y→3Y`); anything else maps to itself and therefore raises.
- **`spike`** — the **only** check with no warm-up allowance; it returns `~recent_spike`
  directly, so it's active from day one. `recent_spike` uses a *backward* rolling max, so
  a day is vetoed only by spikes at or before it.
- **`har`** — the only *forecast* check. Rolling Corsi HAR regression in log-vol space on
  Bloomberg `H` realized vols at three horizons, refit every `refit_every` days and
  forward-filled between. Vetoes when `forecast > rising_ratio × recent_realized`, where
  "recent realized" is the **middle** `bbg_tenors` entry — so changing `bbg_tenors` changes
  the comparison, not just the features.

### 13.4 Combining checks

`gate_from_panel(panel, combine, k)`:

| `combine` | Meaning |
|---|---|
| `'all'` (default) | OK only if **every** check says OK — veto if any vetoes. Strictest; starves the sample fastest. |
| `'any'` | OK if **any** check says OK — veto only on unanimity. |
| `'k'` | Veto once `>= k` checks veto. The usable middle for wide combinations. `k=1` is identical to `'all'`. |

An empty panel returns all-ones — a gate that vetoes nothing.

### 13.5 `GateSpec` — every field

```python
GateSpec(
    checks     = (),        # tuple[str] from list_checks(); a bare 'trend' also works
    combine    = 'all',     # 'all' | 'any' | 'k'
    k          = None,      # required when combine='k'; must be in 1..len(checks)
    params     = {},        # {check_name: {param: value}}
    tenor      = None,      # IV tenor the regime is read on; None = the TRADED tenor
    back_tenor = None,      # termstructure comparison leg; None -> _NEXT_TENOR[tenor]
    on_missing = 'allow',   # 'allow' | 'veto' for dates the gate doesn't cover
    name       = None,      # grid-axis label; auto-derived if omitted
)

NO_GATE = GateSpec(name='none')     # the ungated baseline
```

`__post_init__` enforces: unknown check names raise; `combine='k'` without `k` raises;
`k` outside range raises; `params` keyed on an unregistered check raises. Unknown *param*
keys raise later, in `_check_series`, against each check's `defaults`.

Auto-naming: `'trend+har'`, `'any:trend+momentum'`, `'2of3:trend+level+spike'`, with `*`
appended if `params` were overridden and `@1W` appended if `tenor` was pinned.

### 13.6 The caching trick that makes wide sweeps cheap

Two module-level caches:

```
_SERIES_CACHE : (kind, pair, tenor, hist_days)                      -> vol series
_CHECK_CACHE  : (check, pair, tenor, back_tenor, hist_days, params) -> bool series
```

The important detail is `_hist_budget`: `hist_days` is sized off the longest warm-up among
**all registered checks**, not just the selected ones. That keeps `hist_days` — and
therefore every cache key — constant across every *subset* of checks, so a
22-gate sweep shares **one** Bloomberg pull per (pair, tenor) instead of one per subset.
Costs one slightly longer `bdh` call; saves dozens.

Consequence: retuning a check's params changes the `_CHECK_CACHE` key but **not**
`hist_days`, so a parameter sweep recomputes the check without re-pulling.

`clear_regime_cache()` drops both; `regime_cache_info()` reports sizes.

### 13.7 Adding a check — the whole recipe

```python
# a) the pure check — series in, bool series out (True = OK to sell)
def check_myfilter(iv, thresh=1.5):
    vv = iv.rolling(10).std()
    return _allow_on_warmup(vv <= thresh, vv.notna())

# b) register an adapter
@register_check('myfilter', defaults={'thresh': 1.5}, lookback=lambda p: 10)
def _reg_myfilter(ctx, thresh=1.5):
    return check_myfilter(ctx.iv, thresh=thresh)
```

That's it. `'myfilter'` is now valid in `GateSpec(checks=...)`, picked up automatically by
`enumerate_gate_specs()`, param-overridable, and cached. `ctx` exposes `ctx.iv`,
`ctx.iv_back`, `ctx.iv_at(tenor)`, `ctx.rv_at(tenor)`, `ctx.rv_comp(tenors)` plus
`ctx.pair / tenor / back_tenor / days_back` — all cached, so a new check never adds a
round-trip it doesn't need.

### 13.8 Usage

```python
from Signal_Gen.regime_filter import (describe_checks, build_check_panel, build_gate,
                                      GateSpec, enumerate_gate_specs, gated)

describe_checks()                       # print the live registry

# every check as one boolean column, ONE pull
panel = build_check_panel('EURUSD', tenor='1M', days_back=1825, verbose=True)
# [panel] EURUSD 1M/3M | 1247 days | hist 1478d
#     trend          blocks  512 day(s) (41%)
#     momentum       blocks  548 day(s) (44%)
#     termstructure  blocks  173 day(s) (14%)
#     level          blocks  118 day(s)  (9%)
#     spike          blocks  201 day(s) (16%)
#     har            blocks  409 day(s) (33%)

# slice it into gates — all cache hits after the first
gate = build_gate(GateSpec(('trend','har')), 'EURUSD', '1M', 1825, verbose=True)
# [gate] EURUSD 1M trend+har | 1247 days, 703 vetoed (56%)

# wrap a signal for the plain-grid route (honours max_concurrent)
gated_sig = gated(_ivrv(80), GateSpec(('har',)))
```

`enumerate_gate_specs(checks=None, sizes=(1,), combine='all', k=None, ...)` generates
every combination at each size, prepending `NO_GATE` by default. `sizes=(1,)` gives 1+6=7
specs; `sizes=(1,2)` gives 1+6+15=22. Size-1 combos are skipped for `combine='k'/'any'`
since both degenerate to `'all'` on one check. Results are de-duped by config, not name.

`apply_regime_gate(signal, gate, on_missing='allow', warn=True)` does the AND. **The
`warn` flag matters** — it prints a warning when the gate covers < 95% of the days the
signal actually *fires* on. Without it, a gate built over a shorter window than the signal
silently no-ops across most of the sample and reads as "the regime filter did nothing,"
which is the easiest way to fool yourself here.

---

# PART VII — BOOK AND METRICS (LAYER 6)

## §14 `reporting.py` — building the daily book

Layer 3 gives you a *trade-level* view. Almost every risk question (drawdown, Sharpe,
peak exposure) is a *time-level* question, and under stacking two trades can be open on
the same day. Layer 6 stitches them.

### 14.1 The column contract

```python
_FLOW_COLS = ['net_pnl','option_pnl','hedge_pnl','hedge_carry','gamma_pnl','theta_pnl',
              'vega_pnl','vanna_pnl','volga_pnl','tc',
              'gamma_pnl_be','vanna_pnl_be','volga_pnl_be']     # daily increments

_BE_FLOW_COLS = ['gamma_pnl_be','vanna_pnl_be','volga_pnl_be']  # also get a _cum total

_EXPO_COLS = ['delta','gamma_1pct','vega_1vp','vanna_1vp','volga_1vp',
              'theta_daily','hedge_position']                    # point-in-time levels
```

The distinction is the whole design: **FLOW** columns are 0 off-trade, summed across
trades, then cumsummed into equity. **EXPO** columns are 0 off-trade, summed across trades,
and **never** cumsummed.

### 14.2 `build_daily_book(trade_dfs, trade_log=None)`

Unions every trade's date index, then for each trade reindexes onto it and accumulates:

```python
flow   += f[_FLOW_COLS].fillna(0.0)            # additive daily P&L
expo   += f[_EXPO_COLS].fillna(0.0)            # net level exposure
n_open += f['net_pnl'].notna().astype(int)     # trades open that day
spot    = spot.combine_first(f['spot'])        # market context
```

Output columns:

| Column | Meaning |
|---|---|
| `pnl` | `flow['net_pnl']` — the book's daily P&L |
| `equity` | `pnl.cumsum()` |
| `option_pnl` … `tc` | daily attribution, base ccy |
| `gamma_pnl_be`, `vanna_pnl_be`, `volga_pnl_be` | daily BE flow |
| `*_be_cum` | running total of each BE bucket |
| `net_delta`, `net_gamma_1pct`, `net_vega_1vp`, `net_vanna_1vp`, `net_volga_1vp`, `net_theta_daily`, `net_hedge` | point-in-time net book greeks |
| `net_delta_approx` | `net_delta + net_hedge` |
| `n_open` | concurrent trades that day |
| `spot` | market context |

Two caveats carried in the docstring:

- **`net_delta_approx` is approximate.** `delta` is a **start**-of-day greek and
  `hedge_position` is the **end**-of-day hedge, so their sum has a one-step timing
  mismatch. Use it as "is the book roughly delta-neutral," not as an exact residual.
- **The `_be_cum` columns are NOT part of `pnl`/`equity`.** The plain buckets are the real
  P&L; `_be` is a diagnostic lens layered on top. A single day's BE value is mostly noise —
  the drift of the cumulative is the signal (Ravagli's own framing).

`build_pnl_book()` and `build_exposure_book()` are exact column subsets for time-series
work; `print_daily_book(book, tail=20, ccy=...)` is the formatted terminal view (attribution
totals, max drawdown, worst/best day, peak concurrency, peak net vega/gamma, last N days).

### 14.3 `to_usd_book(book, pair, fx_usd=None)`

Converts a base-ccy book into USD so the same strategy is comparable across pairs.

```
base == 'USD'   ->  factor = 1.0
quote == 'USD'  ->  factor = book['spot']          (e.g. EURUSD: USD per EUR)
neither         ->  requires fx_usd (USD-per-base, indexed by date), else ValueError
```

Every money column is multiplied by **that day's** rate (daily-rate convention).
`equity` and the `_cum` columns are **excluded** from the scaling and then **re-derived**
from the converted daily flows — scaling a cumulative total by a single day's FX rate
would be wrong. `n_open` and `spot` are also left alone.

All seven pairs in `grid_pairs` have USD on one side, so `fx_usd` never needs to be passed.

The factor itself comes from `_usd_factor(index, pair, spot, fx_usd)` — one helper shared
with `to_usd_trade_log` so the book and per-trade lenses cannot drift apart. It ffills then
bfills gaps in the pair's own spot; a missing rate would otherwise silently NaN out that
day's converted P&L instead of failing.

### 14.3b `to_usd_trade_log(trade_log, trade_dfs, pair, fx_usd=None)`

The per-trade counterpart. Each money field on `trade_log` is a plain sum of that trade's own
daily flows, so each is **re-summed** as `sum_days(factor(day) × flow(day))` rather than
scaled by one rate — see §15.4 for why that matters. `net_premium` converts at the
**entry-date** rate (it is a point cashflow, not a flow). Vol-point fields and flags pass
through. Requires positional alignment with `trade_dfs`; raises if the lengths differ.

**Reading a converted money figure.** `usd_total / base_total` is a **P&L-weighted** average
rate, `sum(pnl_d × s_d) / sum(pnl_d)` — *not* the average spot. When gains and losses partly
cancel, the denominator is small and that ratio can land well outside the spot range. A real
5y EURUSD short strangle: `net_pnl` −41,453 EUR → −30,356 USD, a ratio of 0.73 against an
average spot of 1.10 and a sample low near 0.95. Nothing is wrong — losses simply landed on
stronger-EUR days than gains did. Level-based metrics (`max_drawdown`, `cvar_95`, the greek
exposures) have no sign cancellation and do scale inside the spot range (~1.10 in that run).

## §15 `reporting.py` — the metric set

Two lenses, each returning a labelled `pd.Series` so comparing configs is just
`pd.concat`. `_ANN = 252`.

### 15.1 `trade_metrics(trade_log, settled_only=True)` — the per-trade lens

`settled_only=True` drops live trades, whose `net_pnl` is only a partial mark-to-market
and would bias every per-trade statistic. `n_live_excluded` reports how many were dropped.

| Key | Definition |
|---|---|
| `n_trades` | count **after** the live filter |
| `n_live_excluded` | how many were dropped |
| `total_pnl`, `expectancy`, `median_pnl` | `pnl.sum()`, `.mean()`, `.median()` |
| `win_rate` | `(pnl > 0).mean()` |
| `payoff_ratio` | `wins.mean() / abs(losses.mean())` — **1.0 is the neutral point** |
| `profit_factor` | `wins.sum() / abs(losses.sum())` |
| `pnl_skew` | `pnl.skew()`, needs n ≥ 3 |
| `cvar_5pct` | mean of trades at or below the 5th percentile |
| `worst_trade` | `pnl.min()` |
| `t_stat` | `mean / (std / √n)` — **n = n_trades**, so this is the *trade*-sample significance |
| `return_on_premium` | `pnl.sum() / abs(net_premium).sum()` |
| `avg_days_held` | mean `days_held` |
| `pct_hold_to_expiry` | `(exit_reason == 'expiry').mean()` |
| `theta_carry_pnl` | `theta_pnl + hedge_carry_pnl` |
| `gamma_pnl` | summed |
| `vol_pnl` | `vega_pnl + vanna_pnl + volga_pnl` |
| `spot_tc` | summed transaction costs |
| `theta_carry_share` / `gamma_share` / `vol_share` | each bucket over `gross = \|theta_carry\| + \|gamma\| + \|vol\| + \|tc\|` |
| `avg_vol_spread` | mean of `trade_log['vol_spread']`, which the engine defines as `realised_vol − avg_entry_sigma` — realized minus the **vega-weighted** entry implied across legs |
| `real_VRP_ave` | mean of `realised_vol − atm_entry_vol` — realized minus the **pure ATM** vol at inception, *not* vega-weighted |
| `worst_trade_*` | the worst trade's entry/exit dates plus its own `gamma_pnl_be`, `vega_pnl`, `vanna_pnl_be`, `volga_pnl_be` |

### 15.1a The two realized-vs-implied columns — what they are and are not

Both are **ex-post**: measured over each trade's life, then averaged across trades. They
differ only in *which implied vol* they measure against.

| | Reference implied | Source |
|---|---|---|
| `avg_vol_spread` | `avg_entry_sigma` — vega-weighted across legs, i.e. **the vol you actually sold** | `backtest_MLeg.py:903` |
| `real_VRP_ave` | `atm_entry_vol` — pure ATM at inception, ignores where on the smile you traded | `reporting.py:376` |

**For a short book (`direction=-1`) you want both NEGATIVE** — realized came in below
implied, so the premium you sold was rich. That is the opposite sign convention from every
other "higher is better" metric in the table, and it flips at `direction=+1`.

`avg_vol_spread` is the more directly P&L-relevant of the two, since it measures against
the vol you actually transacted. The **gap** between them is informative in its own right:
for a short 25d strangle you sold the wings, which sit above ATM by the butterfly, so
`avg_entry_sigma > atm_entry_vol` and therefore `avg_vol_spread < real_VRP_ave`. The
difference is roughly the smile premium you picked up by trading wings instead of ATM.

**Neither is an ex-ante read.** Nothing in `trade_log` records the signal's own entry
condition, so **"did the signal actually select rich implied vol?" cannot be answered from
the grid at all.** For that you need the signal's own percentile series — the *first*
element of the `(pct_metric, signal_metric, signal_series)` triple that every
`get_*_signal` returns (§11.3), which the builders in §18.2 discard with `[2]`.

What the grid *can* tell you is the thing that matters more for P&L: whether the trades a
signal chose realized below the vol they were sold at.

### 15.2 `book_metrics(book, active_only=False)` — the time-level lens

| Key | Definition |
|---|---|
| `n_days`, `active_days` | rows, and rows with `n_open > 0` |
| `coverage` | `active_days / n_days` |
| `book_total_pnl` | `pnl.sum()` |
| `sharpe_ann` | `mean/std · √252` on the daily series |
| `sortino_ann` | `mean/downside · √252`, where `downside = √(mean(min(x,0)²))` |
| `max_drawdown` | `(equity − equity.cummax()).min()`, signed ≤ 0 |
| `max_drawdown_days` | **peak-to-trough** days — *not* recovery time to a new high |
| `calmar` | `ann_pnl / abs(max_drawdown)`, where `ann_pnl = total / (n_days/252)` |
| `daily_pnl_skew` | `.skew()` |
| `cvar_5pct_daily` | **mean** of days at or below the 5th percentile |
| `var_95_daily`, `var_99_daily` | the 5th / 1st percentile **levels** themselves |
| `worst_day`, `best_day` | min / max daily P&L |
| `peak_concurrency` | `n_open.max()` |
| `avg_net_vega`, `avg_net_gamma` | mean over **active** days only |
| `peak_net_vega`, `peak_net_gamma` | signed value of largest magnitude (works for a net-short book) |
| `worst_day_*` | the worst day's date plus its `gamma_pnl_be`, `vega_pnl`, `vanna_pnl_be`, `volga_pnl_be` |
| `net_gamma_pnl_BE`, `net_vanna_pnl_BE`, `net_volga_pnl_BE` | last value of each `_be_cum` |

**`active_only=True` judges per-deployed-day instead** — drawdown always uses the full
equity curve regardless. This parameter is **not** plumbed through `evaluate()`, so
`run_combo` cannot reach it. That has a consequence, §15.5.

### 15.3 `scorecard()` and `evaluate()`

```python
scorecard(book, trade_log, settled_only=True, ...)
  = pd.concat([trade_metrics(trade_log, settled_only), book_metrics(book)])
    + sc.attrs['worst_days']   (DataFrame)
    + sc.attrs['worst_trades'] (DataFrame)

evaluate(trade_dfs, trade_log, pair=None, to_usd=False, settled_only=False,
         fx_usd=None, ...)
  = build_daily_book(trade_dfs, trade_log)
    -> to_usd_book(book, pair, fx_usd)                    if to_usd
    -> to_usd_trade_log(trade_log, trade_dfs, pair, fx_usd) if to_usd
    -> scorecard(book, trade_log, ...)
```

`evaluate` is the single source of truth for every metric. Nothing downstream recomputes
anything — Layer 7 only *selects* from this output.

### 15.4 UNITS — `to_usd=True` converts BOTH lenses

The engine works natively in the pair's **BASE** currency (`pair[:3]`): option value is
`(intrinsic / S) x notional`, and `intrinsic/S` is dimensionless, so the result carries
`LegSpec.notional`'s unit. That is why the USD factor is USD-per-**base**.

`to_usd=True` converts the daily book (`to_usd_book`) **and** the per-trade log
(`to_usd_trade_log`), both off the same shared factor helper `_usd_factor`. So with
`to_usd=True` every money figure in the scorecard is USD — `expectancy`, `median_pnl`,
`worst_trade`, `cvar_5pct`, `theta_carry_pnl`, `gamma_pnl`, `vol_pnl`, `spot_tc` included —
and all of them are cross-pair comparable.

**Why the per-trade side is re-summed rather than scaled.** Every money field on
`trade_log` is a plain sum of that trade's own daily flows
([backtest_MLeg.py:898-924](Delta_Hedged/backtest_MLeg.py#L898-L924)), so `to_usd_trade_log`
rebuilds each one as `sum_days(factor(day) x flow(day))` — the same daily-rate convention
as the book. Two things depend on it:

- a trade spanning a large FX move is not converted at a rate it never earned its P&L at;
- the two lenses **reconcile**: `sum(trade net_pnl) == book_total_pnl`, because both are the
  same double sum over the same days. Scaling the trade total by one rate would break that.

`net_premium` is the single exception — an entry-date cashflow, not a flow over the trade's
life, so it converts at the **entry-date** rate. `return_on_premium` is therefore USD P&L
over USD premium.

**One consequence to know:** `return_on_premium` was unit-free before and still is, but it
is **not invariant** under the conversion — numerator at daily rates, denominator at the
entry rate, so it now embeds the FX translation the base-ccy ratio hid. USD-over-USD is the
honest read for a USD book, but a `ret_on_prem` figure from before this change will not
match one from after.

`trade_dfs` must stay **positionally aligned** with `trade_log`'s rows.
`run_signal_backtest` appends the two in lockstep
([backtest_signal.py:199-200](Delta_Hedged/backtest_signal.py#L199-L200)) and
`gate_sweep` preserves it (`trade_log.iloc[idx]` beside `[trade_dfs[i] for i in idx]`);
`to_usd_trade_log` raises if the lengths disagree.

**Unit-free is not the same as invariant.** A ratio is unchanged only under a *common*
multiplier, and the daily-rate convention gives each trade its own effective rate. So:

| Exactly invariant | Unit-free but **shifts** |
|---|---|
| `win_rate` (sign-based), `n_trades`, `coverage`, `avg_days_held`, `pct_hold_to_expiry`, `avg_vol_spread`, `real_VRP_ave` (vol points, not money) | `payoff_ratio`, `profit_factor`, `t_stat`, `pnl_skew`, the `*_share`s — each is a ratio of sums or a mean/std over trades converted at *different* rates. Measured `payoff_ratio` 0.6557 → 0.6681 |

The book-sourced ratios — `sharpe`, `sortino`, `calmar` — are untouched by the trade-log
conversion, but note they were never invariant to `to_usd` itself either: `to_usd_book`
reshapes the daily P&L series (each day at its own rate) rather than scaling it, so they
already differed between `to_usd=False` and `True`. Measured on the same run: `calmar`
−0.0259 → −0.0170, `sharpe` −0.0656 → −0.0438.

Two places still base-ccy **by design**:

| Where | Why it's fine |
|---|---|
| `to_usd=False` (the default on `evaluate`; `run_grid`/`run_combo` default to **True**) | single-pair base-ccy read |
| `gate_sweep`'s `ATTR_COLS` (§17.6) | reads the raw log it holds, not `evaluate`'s converted copy; only ever compared kept-vs-removed within a row, where the factor cancels |

For the record, the base-ccy exposure across your seven pairs — the distortion this used to
put on `expectancy`:

| pairs | base-ccy money unit |
|---|---|
| USDJPY, USDCHF, USDCAD | **already USD** (base == USD -> factor 1.0) |
| EURUSD, GBPUSD, AUDUSD, NZDUSD | EUR / GBP / AUD / NZD (~0.6-1.3 x USD) |

### 15.5 THE DEPLOYMENT TRAP — read this before comparing signals or gates

**The daily book has no flat rows at all.** `build_daily_book` indexes on the *union* of
the trades' own date indices ([reporting.py:69](Delta_Hedged/reporting.py#L69)), so a day
with no trade open is **absent**, not zero. Two consequences, and they pull in opposite
directions:

**`sharpe_ann` and `sortino_ann` are already per-deployed-day.** With no idle rows to
dilute the mean, they measure *quality per day in the market* — effectively what
`active_only=True` would give. There is **no √coverage penalty to undo**, and they are
directly comparable across signals. (An earlier version of this guide claimed otherwise;
that was wrong.)

**`calmar` was inflated, and `coverage` could only ever read 100%.** Both were computed
off `len(book)` — the *deployed*-day count:

- `ann_pnl = total / (len(book)/252)` annualized a selective signal's P&L over its
  deployed span, overstating `calmar` by `calendar / deployed`. A signal trading a third
  of the time got ~3× the calmar it earned.
- `coverage = active_days / len(book)` is identically `1.0`, because every row in the union
  belongs to at least one trade.

**Both are now fixed** ([reporting.py:419-446](Delta_Hedged/reporting.py#L419)) off the
book's own first and last date: `ann_pnl` annualizes over the **calendar span**, and
`coverage = active_days / span_bdays` where `span_bdays` is business days from first row to
last. `span_bdays` is also exposed as its own scorecard key.

Residual limitation: the span still starts at the *first trade*, not at the start of your
requested window — so if a signal doesn't fire for the first six months, that period isn't
counted against it. And `np.busday_count` ignores FX holidays, making `coverage` a mild
under-estimate. Both are far better than a constant 1.0. A full fix would reindex the book
onto the signal's own calendar inside `evaluate`.

**What to rank on anyway.** `ret_on_prem` remains the right primary for signals — it's a
ratio, so it's cross-pair safe (§15.4) and independent of deployment. `win_rate`,
`payoff_ratio`, `avg_vol_spread` and `real_VRP_ave` are likewise deployment-invariant.
`sharpe` is now legitimately usable as a secondary read. `net_pnl` still scales with how
often you traded, so it answers "does this matter in money," not "is this good."

Read `coverage` as a diagnostic of *what kind of thing you built*: **high coverage means
the signal is a timing dial** (it shifts entry dates but you're exposed anyway); **low
coverage means it's a genuine filter.** With 1M holds and a signal firing ~20% of days,
expect ~80%.

---

# PART VIII — THE GRID (LAYER 7)

## §16 `grid_eval.py`

One `ComboSpec` = one fully-specified backtest. `run_grid` runs a list of them and returns
a tidy frame; `print_grid` renders it.

### 16.1 The `Metric` contract

`CLEAN_METRICS` is the **only** definition of the comparison column set. Each entry maps
a display name to a scorecard key `evaluate()` already returns, plus formatting and colour:

```python
Metric(name, key, fmt, color, center=0.0)

fmt    : 'ratio' (2dp) | 'usd' (compact k/M) | 'pct' (×100, 0dp) | 'count' | 'greek'
color  : 'div'  diverging around `center` (blue = high/good, red = low/bad)
         'seq'  sequential magnitude
         'loss' pure-loss scale (values ≤ 0; more negative = darker red)
center : neutral point for 'div'
```

The 19 current entries:

| Group | Metrics |
|---|---|
| Performance / edge | `calmar`, `sortino`, `sharpe`, `net_pnl`, `expectancy`¹, `ret_on_prem`, `win_rate` (center 0.5), `payoff_ratio` (**center 1.0**) |
| Tail / drawdown | `max_drawdown`, `cvar_95`, `var_95`, `var_99` — all stored signed ≤ 0, so higher = better |
| Exposure | `avg_net_vega`, `avg_net_gamma` — colour encodes **sign**, not quality; a short-vol book sits on the red side by construction |
| Sample-size context | `n_trades`, `t_stat`, `coverage` |
| Realized-vs-implied (vol points) | `vol_spread`, `real_vrp` |

¹ `expectancy` is base-currency (§15.4) — it leads `SIGNAL_TABLE_COLS` for single-pair
work, but read it down a column (one pair) and never across pairs.

**`vol_spread` and `real_vrp` both have an inverted sign** relative to every other `div`
metric: both are realized *minus* implied, so for a short book more negative is better.
Both are ex-post — see §15.1a.

### 16.2 The two table presets

```python
DEFAULT_TABLE_COLS = ['calmar','sortino','sharpe','net_pnl','n_trades',
                      'win_rate','var_95','max_drawdown','t_stat','coverage']

SIGNAL_TABLE_COLS  = ['expectancy','max_drawdown','ret_on_prem','net_pnl',
                      'calmar','sharpe',
                      'n_trades','coverage',
                      'win_rate','payoff_ratio',
                      'cvar_95',
                      'real_vrp','avg_net_vega','avg_net_gamma']
```

`SIGNAL_TABLE_COLS` exists because of §15.5. It leads with the per-trade / worst-case size
pair (`expectancy`, `max_drawdown`), then the coverage-invariant edge metric, then
`calmar`/`sharpe` as the "what does this do to my book" read with `coverage` right beside
them to correct by, then distribution shape, the tail, the ex-post vol read, and the greek
exposures the P&L came from.

It drops `sortino` (empirically 1.11–1.22× `sharpe` row for row — zero incremental
information), `var_95`/`var_99` (superseded by `cvar_95`), `t_stat` (selectivity deflates it
mechanically — §15.5; `n_trades` already carries the sample-size warning), and `vol_spread`
(within the smile premium of `real_vrp`, so one ex-post vol read suffices).

Every money column in the preset is USD when the grid is built with `to_usd=True` (the
`run_grid` default), `expectancy` and the two greeks included — §15.4. The one read-direction
caveat left: `avg_net_vega`/`avg_net_gamma` encode **sign**, not quality, so a short-vol book
is red by construction.

`DEFAULT_PANELS` is the separate 3×3 heatmap selection.

### 16.3 `ComboSpec`

```python
@dataclass
class ComboSpec:
    pair:  str
    tenor: str
    signal_fn:          Callable[[ComboSpec], pd.Series]      # required
    legs_fn:            Optional[Callable[[ComboSpec], list]]     = None
    legs_factory_fn:    Optional[Callable[[ComboSpec], Callable]] = None
    hedge_rule_factory: Callable[[], Any] = None
    exit_rule_factory:  Callable[[], Any] = None
    days_back:      int   = 94
    notional:       float = 10_000_000
    direction:      int   = -1
    tc_fraction:    float = 0.0001
    max_concurrent: Optional[int] = None
    label:          Optional[str] = None

    def cell(self): return self.label if self.label is not None else self.tenor
    def key(self):  return (self.pair, self.tenor, self.cell())
```

**Every callable takes the spec itself.** That's deliberate: a pair-aware signal ("same
signal per pair") becomes honest rather than accidental — `signal_fn(s)` can read
`s.pair`, `s.tenor`, `s.days_back`.

**`legs_fn` vs `legs_factory_fn`** — exactly one. Use the factory when the structure needs
entry-date market data to build itself (the vega-neutral fly, §3.1).

**`label` is the grid's column axis.** It defaults to `tenor`, which is why a plain
pair × tenor sweep behaves as it always did. Set it to name a third variation — a gate, a
signal, an exit rule — and the table and heatmaps switch their column axis automatically.

### 16.4 `run_combo` — never raises

One spec → one flat dict of clean metrics. It calls `signal_fn`, `legs_fn`/`legs_factory_fn`,
then `run_signal_backtest`, then `evaluate`, then selects `CLEAN_METRICS`.

Three outcomes, all with the same schema:

| `status` | When |
|---|---|
| `'ok'` | normal |
| `'no_trades'` | signal never fired, or every candidate was blocked — metrics all NaN |
| `'error'` | any exception; the message lands in the `error` column — metrics all NaN |

**This is why a bad cell degrades instead of killing the sweep.** `ExitAtDaysRemaining('1W')`
against `tenors=['1W']` produces seven `status='error'` rows, not a traceback.

`settled_only=True` is the default here — matching `scorecard()`'s own default rather than
`evaluate()`'s — which keeps the grid consistent with the single-combo tearsheet. Note it
affects only the trade-level metrics; book-level ones are unaffected.

### 16.5 `run_grid` and its progress output

```python
grid = run_grid(specs, to_usd=True, settled_only=True,
                verbose=True, combo_progress=False)
```

Returns a frame indexed by `(pair, tenor, label)` with the `CLEAN_METRICS` columns in
contract order, then `days_back, direction, notional, status, error`.
`grid.attrs['to_usd']` records the unit.

`verbose=True` gives grid-level progress; `combo_progress=True` additionally turns on
`run_signal_backtest`'s per-trade heartbeat inside each combo (use when one combo is slow
and you want to see inside it).

```
[1/21] EURUSD 1M [none] ...
      -> 56 trades, 41.2s | sweep 41.2s elapsed, ~824.0s left
[2/21] EURUSD 1M [ivrv80] ...
      -> 48 trades, 38.7s | sweep 79.9s elapsed, ~758.7s left
...
[grid] done: 21 combos in 812.4s (38.7s/combo avg) | 21 ok, 0 empty/error
```

### 16.6 `print_grid` and `plot_grid_heatmaps`

```python
print_grid(grid, sort_by='calmar', ascending=False, cols=None, title=None)
```

`cols` defaults to `DEFAULT_TABLE_COLS`. Sorting is `kind='stable'` with
`na_position='last'`, and `sort_by` is checked against `view.columns` **after**
`reset_index()` — so `'pair'`, `'tenor'` and `'label'` are all valid sort keys, not just
metrics. Sorting on a key column **groups** the table; sorting on a metric **ranks** it.

A `label` key column appears whenever `_resolve_col` decides `label` carries information
beyond the tenor — i.e. when labels vary *and* aren't just the tenors.

`plot_grid_heatmaps(grid, panels=None, col='auto', ...)` renders a pair × `col` colour
matrix per metric. `_matrix` **raises** if there are multiple rows per `(pair, col)` —
which happens if you vary tenor *and* label — with the fix in the message: put the tenor
into the label, e.g. `label='1M|trend'`. The table has no such restriction.

---

# PART IX — GATE SWEEPS (LAYER 8)

## §17 `gate_sweep.py`

### 17.1 Why this exists instead of just calling `run_grid` with gated signals

A regime gate is a pure **entry filter** — it can only ever *remove* entry days. And when
`max_concurrent=None`, every signal day opens its own trade and concurrent trades are
simulated fully independently (§10.3). So **removing entry days cannot change any trade
that remains.**

Therefore, for a fixed (pair, tenor, signal, structure, hedge, exit):

```
run the UNGATED backtest once
  -> filter the resulting trade_log / trade_dfs by each gate
  -> re-evaluate each subset
```

is **exactly** equivalent to re-running the backtest once per gate. 20 gate variants
become 1 backtest + 20 `evaluate()` calls.

### 17.2 The caveat that forces `max_concurrent=None`

**The equivalence requires unlimited stacking, which this module hardcodes and asserts on.**
With `max_concurrent=1` or `N`, removing an entry **frees a slot** and admits a later trade
the ungated run had skipped — so the surviving trades genuinely differ and subsetting the
ungated log would give the *wrong* answer, not an approximation.

`run_signal_backtest(..., max_concurrent=None)` is passed literally, with a
`# REQUIRED — see module docstring` comment. `RUN()` asserts on it too, so you can't
silently think you got `max_concurrent=1`.

**If you need both a gate and a concurrency cap**, use the plain-grid route with
`gated(signal_fn, spec)` (§13.8) — one backtest per gate, and you lose attribution.

### 17.3 The second thing you get free: veto attribution

Because both subsets come from the same run, you can see **the P&L of the trades each gate
removed.** That's the question that actually matters. Comparing gated vs ungated Calmar
alone conflates "the gate found a better sub-sample" with "the gate shrank the sample."

A gate that removes *profitable* trades is destroying edge no matter what happened to the
ratio. A gate that earns its keep usually does it by truncating the left tail — visible in
`max_drawdown` / `var_95` — not by raising the mean.

### 17.4 `run_gate_sweep`

```python
grid, attr = run_gate_sweep(
    pairs, tenors, signal_fn, gate_specs,
    legs_fn=None, legs_factory_fn=None,
    hedge_rule_factory=lambda: DailyHedge(),
    exit_rule_factory=lambda: HoldToExpiry(),
    days_back=94, notional=10_000_000, direction=-1, tc_fraction=0.0001,
    to_usd=True, settled_only=True,
    verbose=True, progress=False, gate_verbose=False)
```

- `signal_fn` is the **ungated** signal. Do **not** pre-wrap it with `gated(...)` — the
  gates come from `gate_specs`.
- **Gate labels must be unique** (they are the grid's column axis) — asserted up front.
- Include `NO_GATE` so every table is self-comparing. `enumerate_gate_specs()` does this
  by default.
- `days_back` is used both for `signal_fn` and to build every gate, so the gate and signal
  windows always agree.

Per (pair, tenor): one base run, then per gate a `_keep_mask`, an attribution row, and an
`evaluate()` on the surviving subset. A gate that keeps nothing is recorded as
`status='no_trades'`.

Returns `grid` with the **same schema as `run_grid`** — so `print_grid` and
`plot_grid_heatmaps` work unchanged, laid out pair × gate — plus `attr`.

```
[1/7] EURUSD 1M: one base run + 7 gate(s) ...
      -> 1247 base trades, 7 gates scored in 63.4s | sweep 63.4s elapsed, ~380.4s left
...
[gate_sweep] done: 49 (cell x gate) rows from 7 backtest(s) in 448.1s | 49 ok, 0 empty/error
```

### 17.5 `_keep_mask` and the coverage warning

```python
gate = build_gate(spec, pair, tenor, days_back)
g    = gate.reindex(entry_ts)
cov  = g.notna().mean()
if cov < 0.95:  print("[gate_sweep] WARNING: ... covers only {cov:.0%} of entry dates ...")
```

Uncovered dates fall back to `spec.on_missing`. A low number means the gate window doesn't
reach the whole sample and most trades are effectively ungated — the failure mode that
makes a filter look inert. It warns rather than guessing.

### 17.6 The attribution table

```python
ATTR_COLS = ['n_all','n_kept','n_removed','veto_rate',
             'kept_mean_pnl','removed_mean_pnl','removed_total_pnl',
             'worst_kept','worst_removed','gate_cov']
```

**P&L here is in the pair's BASE currency** — read straight off the raw `trade_log` this
module holds. `evaluate(to_usd=True)` does convert the per-trade log, but it converts its
own copy, so this attribution is untouched. USD for USDJPY/USDCHF/USDCAD, foreign for the
other four.
That's deliberate and safe: the comparison that matters is kept-vs-removed *within* one
row, where the FX factor cancels. The USD-comparable money metrics live in the grid frame.

`settled_only=True` mirrors `trade_metrics` — live trades carry only a partial
mark-to-market and would bias the per-trade comparison.

```python
print_gate_attribution(attr, sort_by='removed_total_pnl', ascending=True,
                       title=None, pair=None)
```

**The default sort puts the most NEGATIVE `removed_total_pnl` first** — gates that
stripped out the biggest cumulative losers, i.e. those plausibly earning their veto. **A
gate with POSITIVE `removed_total_pnl` threw away money.** No Calmar improvement redeems
that; it just means the survivors were smoother.

```
=========================================================================================================
  GATE VETO ATTRIBUTION (base ccy)  |  har gate   (ranked by removed_total_pnl)
=========================================================================================================
    pair  tenor    gate  n_all  n_kept  n_removed  veto_rate  kept_mean_pnl  removed_mean_pnl  removed_total_pnl  worst_kept  worst_removed  gate_cov
  USDCHF     1M     har   1247     838        409        33%           -412            -2,180           -891,620     -48,200        -96,400      100%
  EURUSD     1M     har   1247     851        396        32%            118            -1,004           -397,584     -31,700        -74,900      100%
  USDCAD     1M     har   1247     902        345        28%            241               186             64,170     -22,100        -19,800      100%
  USDCAD     1M    none   1247    1247          0         0%            226               nan                nan     -22,100            nan      100%
=========================================================================================================
```
*(illustrative shape and formatting; numbers are not from a real run)*

Read that as: the `har` gate earned its veto on USDCHF and EURUSD (removed trades averaged
big losses) but **destroyed** edge on USDCAD (`removed_total_pnl` positive — it threw away
+64k of profitable trades).

### 17.7 The consistency table

```python
gate_consistency(grid, metric='calmar', baseline='none') -> DataFrame
print_gate_consistency(grid, metric='calmar', title=None,
                       baseline='none', label_header='gate')
```

Pivots `metric` to `(pair, tenor) × label`, differences every column against `baseline`,
and reports mean/median, mean delta, pairs improved, and hit rate.

**Why this is the read that matters.** Enumerating gate combinations means many looks at
one dataset, so the best single cell is close to meaningless. A gate that improves 5 of 6
pairs modestly is a regime effect; one that transforms a single pair is noise.

**Nothing here is gate-specific** — it works on any grid whose `label` axis varies and
contains a baseline column, which is why a *signal* sweep reuses it unchanged with
`label_header='signal'`. For signals it matters even more than for gates: a selective
signal shrinks `n_trades`, which deflates `t_stat` even when per-trade edge improves
(§15.5), so the cross-pair hit rate has to carry the significance argument. Pass
`metric='ret_on_prem'` there.

```
==============================================================================
  SIGNAL CONSISTENCY on ret_on_prem across pairs  |  signals | 1M
==============================================================================
  signal                         ret_on_prem    median  d_vs_none   better    hit
  ivrv80                                0.16      0.14       0.04    5/7      71%
  xccy80                                0.13      0.11       0.01    4/7      57%
  none                                  0.12      0.12       0.00    0/7       0%
==============================================================================
```
*(illustrative)*

**One calibration on the hit rate.** Every pair in `grid_pairs` has USD on one side, and
AUD/NZD are near-duplicates, so seven pairs is maybe 2–3 independent tests. "6 of 7" is
much weaker evidence than it reads.

---

# PART X — THE FRONT DOOR (LAYER 9)

## §18 `TEST.py` — `RUN()`

Everything above is library code. `TEST.py` is your experiment file, and `RUN()` is one
function with every dial as a keyword argument.

### 18.1 File layout

```
imports                       every layer, flat
── Signal Builders ──         _always_on, _ivrv(sell_pct), _xccy(sell_pct)
                              + _sig_ivrv / _sig_xccy back-compat aliases at sell_pct=80
── Constant Leg Structures ── _straddle, _strangle(cd,pd), _risk_reversal(cd,pd), _custom
RUN(...)                      the three-branch dispatcher
config block                  grid_pairs, grid_tenors, grid_notional, DAYS_BACK
experiments                    your actual RUN() calls, mostly commented out
```

### 18.2 The signal builders

```python
def _always_on(s):
    return get_always_on_signal(s.days_back, pair=s.pair, tenor=s.tenor)

def _ivrv(sell_pct=80):
    """Enter when implied (trade tenor) is rich vs 1W realized."""
    def sig(s):
        return get_cross_vol_signal(
            ccys=[s.pair], leg_a=('V', s.tenor), leg_b=('H', '1W'),
            days_back=s.days_back, sell_pct=sell_pct, side='sell')[2]
    sig.__name__ = f'ivrv{sell_pct:.0f}'
    return sig

def _xccy(sell_pct=80):
    """Enter when the pair's ATM vol is rich vs a cross-ccy basket."""
    basket = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','NZDUSD','USDCAD']
    def sig(s):
        return get_xccy_spread_signal(
            ccy_int=s.pair, tenor_int=s.tenor, ccys=basket,
            days_back=s.days_back, sell_pct=sell_pct, side='sell')[2]
    sig.__name__ = f'xccy{sell_pct:.0f}'
    return sig

_sig_ivrv = _ivrv(80)      # back-compat: the old bare fn(s) at the old default
_sig_xccy = _xccy(80)
```

`_always_on` is a bare `fn(s)`. `_ivrv` / `_xccy` are **parameterized factories** —
call them. Both read `s.pair`, `s.tenor`, `s.days_back` off the spec, so one definition
serves every cell of a grid. `[2]` picks `signal_series` out of the
`(pct_metric, signal_metric, signal_series)` triple.

`sell_pct` is the **selectivity dial**, and sweeping it is the strongest test of a signal
available to you — see §19.4.

### 18.3 The leg structures

```python
def _straddle(s):                    return build_straddle(s.notional, s.direction)
def _strangle(cd=0.25, pd=0.25):     ...returns legs(s)
def _risk_reversal(cd=0.25, pd=0.25):...returns legs(s)
def _custom(s):                      # hand-rolled 4-leg condor
    return [LegSpec('call', +0.10, +1, s.notional),
            LegSpec('call', +0.25, -1, s.notional),
            LegSpec('put',  -0.25, -1, s.notional),
            LegSpec('put',  -0.10, +1, s.notional)]
```

**`_straddle` and `_custom` go in bare** (they *are* `fn(s)`). **`_strangle(...)` and
`_risk_reversal(...)` get called first**, because they're parameterized and return the
`fn(s)`. Same distinction as the signal builders.

### 18.4 `RUN()` — full signature

```python
def RUN(title, *,
        signal_fn         = _always_on,      # single signal (ignored when signals= is set)
        signals           = None,            # {label: signal_fn} -> pair x signal grid
        gates             = None,            # [GateSpec, ...]    -> pair x gate sweep
        legs_fn           = _strangle(0.25, 0.25),
        legs_factory_fn   = None,            # exactly one of legs_fn / legs_factory_fn
        direction         = -1,              # -1 = short the structure
        exit_rule_factory = lambda: HoldToExpiry(),
        hedge_rule_factory= lambda: DailyHedge(),
        pairs             = None,            # None -> grid_pairs
        tenors            = None,            # None -> grid_tenors
        days_back         = None,            # None -> DAYS_BACK
        notional          = 10_000_000,
        tc_fraction       = 0.0001,
        max_concurrent    = None,            # None=stack, 1=non-overlapping, N=cap
        sort_by           = None,            # None -> 'ret_on_prem' (signals) / 'calmar'
        ascending         = None,            # None -> A->Z on a key column, best-first on a metric
        cols              = None,            # None -> SIGNAL_TABLE_COLS (signals) / default
        show              = False):          # True -> also draw the heatmap figure
```

**The `lambda:` on the rule factories is not optional** — Layer 3 needs a fresh rule per
trade, or stateful rules leak state between trades (§9).

### 18.5 The three branches

| Branch | Trigger | Route | Returns | `max_concurrent` |
|---|---|---|---|---|
| **pair × tenor** | `signals=None, gates=None` | `run_grid` | `grid` | honoured |
| **pair × signal** | `signals={...}` | `run_grid` with `label=` set | `grid` | honoured |
| **pair × gate** | `gates=[...]` | `run_gate_sweep` | `(grid, attr)` | **must be `None`** |

Three guards:

```python
# signals branch
assert gates is None,   "signals= and gates= are separate sweeps — run them one at a time"
assert 'none' in signals, "signals= needs an unfiltered baseline keyed 'none' ..."

# gates branch
assert max_concurrent is None, "max_concurrent is not supported with gates= ..."
```

The `'none'` requirement is the same discipline as always including `NO_GATE`: every
column is scored as a delta against the baseline, and without one there's nothing to
compare.

**Spec construction order in the signals branch:**

```python
for p in P for t in T for lbl, fn in signals.items()
```

pair-major, signal innermost. Combined with `print_grid`'s stable sort, `sort_by='pair'`
puts each pair's baseline row directly above its own alternatives. Dict insertion order is
preserved, so putting `'none'` first in the dict puts it first in every block.

**`ascending` resolution:**

```python
if ascending is None:
    ascending = sort_by in ('pair', 'tenor', 'label')
```

`print_grid` defaults to `ascending=False`, which would reverse-alphabetise a grouped
table — so sorting on a key column flips to A→Z automatically, while sorting on a metric
stays best-first.

**The signals branch also calls `print_gate_consistency(..., metric='ret_on_prem',
label_header='signal')`** — `ret_on_prem` and not `calmar`, for the reason in §15.5.

### 18.6 Combining a gate with a concurrency cap

`gates=` can't do it (§17.2). Use `gated()` on the plain-grid route:

```python
from Signal_Gen.regime_filter import gated, GateSpec

g = RUN('1M hold | max 1 | HAR gate',
        signal_fn=gated(_always_on, GateSpec(('har',))),
        max_concurrent=1,
        exit_rule_factory=lambda: ExitAfterNDays('1M'))
```

Exact, but one backtest per gate instead of one per pair, and no attribution or
consistency tables. The intended workflow is: **screen with `gates=` under stacking to
find which checks earn their veto, then re-run the two or three survivors through
`gated()` with your real `max_concurrent`.**

---

# PART XI — WORKED EXAMPLES

## §19 End to end

The shared config block:

```python
grid_pairs  = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','NZDUSD','USDCAD']
grid_tenors = ['1M']
grid_notional  = 10_000_000
grid_direction = -1
DAYS_BACK = (365 * 5)
```

### 19.1 Baseline — pair × tenor

```python
g = RUN('base | short 25d strangle | always-on | Hold Till Expiry | Max 1 Trade',
        tenors=['1W','1M','3M'],
        max_concurrent=1)
```

Real output (5 years, `DailyHedge`, `tc_fraction=0.0001`):

```
=========================================================================================================================
  GRID SCORECARD (USD)  |  base | short 25d strangle | always-on | Hold Till Expiry | Max 1 Trade   (ranked by calmar)
=========================================================================================================================
    pair  tenor  calmar  sortino  sharpe  net_pnl  n_trades  win_rate  var_95  max_drawdown  t_stat  coverage  status
  USDCAD     3M    0.71     1.06    0.85     270k        19       74%     -7k          -75k    2.92      100%      ok
  USDCAD     1M    0.21     0.48    0.40     179k        56       61%     -9k         -174k    0.94      100%      ok
  EURUSD     1W    0.17     0.29    0.25     232k       218       62%    -22k         -308k    0.49      100%      ok
  ...
  USDCHF     1M   -0.16    -0.75   -0.67    -464k        56       43%    -14k         -574k   -1.50      100%      ok
=========================================================================================================================
```

**How to read it.** `n_trades` confirms the mechanics: 5 years ÷ tenor with
`max_concurrent=1` and `HoldToExpiry` gives ~218 weekly / ~56 monthly / ~19 quarterly.
`coverage` is 100% everywhere because always-on plus back-to-back entries means the book is
never flat. `t_stat` is the per-trade significance (§15.1) while `sharpe` is annualized on
~1250 daily observations (§15.2) — that's why they disagree most on the 19-trade 3M cells.

### 19.2 Signals — pair × signal

```python
g = RUN('signals | short 1M 25d strangle | Hold to Expiry | Max 1 Trade',
        tenors=['1M'],
        max_concurrent=1,
        signals={'none':   _always_on,
                 'ivrv80': _ivrv(80),
                 'xccy80': _xccy(80)},
        sort_by='pair')
```

21 backtests. Output shape (numbers illustrative):

```
====================================================================================================================================================================
  GRID SCORECARD (USD)  |  signals | short 1M 25d strangle | Hold to Expiry | Max 1 Trade   (ranked by pair)
====================================================================================================================================================================
    pair  tenor   label  ret_on_prem  net_pnl  calmar  sharpe  n_trades  coverage  t_stat  win_rate  payoff_ratio  max_drawdown  cvar_95  vol_spread  real_vrp  status
  AUDUSD     1M    none         0.09     110k    0.09    0.18        56      100%    0.39       61%          0.72         -244k     -19k       -0.55     -0.30      ok
  AUDUSD     1M  ivrv80         0.14      92k    0.11    0.19        47       82%    0.35       64%          0.81         -201k     -17k       -1.02     -0.74      ok
  AUDUSD     1M  xccy80         0.07      61k    0.06    0.12        44       78%    0.21       59%          0.68         -218k     -18k       -0.48     -0.22      ok
  EURUSD     1M    none        -0.08    -159k   -0.08   -0.23        56      100%   -0.61       55%          0.64         -408k     -21k       -0.11      0.14      ok
  EURUSD     1M  ivrv80         0.02      18k    0.02    0.05        46       81%    0.08       57%          0.71         -333k     -19k       -0.68     -0.41      ok
  ...
====================================================================================================================================================================

==============================================================================
  SIGNAL CONSISTENCY on ret_on_prem across pairs  |  signals | ...
==============================================================================
  signal                         ret_on_prem    median  d_vs_none   better    hit
  ivrv80                                0.11      0.12       0.05    6/7      86%
  xccy80                                0.07      0.07      -0.01    3/7      43%
  none                                  0.06      0.05       0.00    0/7       0%
==============================================================================
```

**How to read it, in order:**

1. **`coverage` first.** ~80% means these signals are timing dials, not filters (§15.5) —
   so the `calmar`/`sharpe` penalty is ~10% and those columns remain roughly comparable.
2. **`ret_on_prem`, not `calmar` or `net_pnl`.** It's the only coverage-invariant,
   cross-pair-safe edge metric.
3. **`vol_spread` — did the trades the signal chose actually realize below the vol they
   were sold at?** More negative is better (§15.1a). `ivrv80` nearly doubles the captured
   premium (−0.55 → −1.02 on AUDUSD, −0.11 → −0.68 on EURUSD). `xccy80` barely moves it
   (−0.55 → −0.48, i.e. slightly *worse*), so it isn't finding cheaper-to-own vol — grounds
   to drop it regardless of what its P&L did.
4. **`real_vrp` — the same question against the ATM reference.** It should move with
   `vol_spread` and sit ~0.25–0.30 vol points above it (the 25d butterfly you collected by
   selling wings instead of ATM). `EURUSD/none` is the instructive row: `real_vrp = +0.14`
   means realized *exceeded* ATM implied, and only the smile premium (`vol_spread = −0.11`)
   kept it marginally on the right side — which is exactly why that cell loses money.
5. **The consistency table is the verdict.** 6/7 on `ivrv80` versus 3/7 on `xccy80`.
6. **Ignore `t_stat` for this comparison.** It falls with `n_trades` even when per-trade
   edge improves, *and* it overstates significance more for a clustered signal than for
   always-on (§15.5). Ambiguous in both directions.

**What this table cannot tell you** is whether the signal *selected* rich implied vol at
entry — both vol columns are ex-post (§15.1a). If `ivrv80` improves `vol_spread`, you know
its trades realized better; you do not know from here whether that came from picking richer
implied or from avoiding high-realized regimes. To separate those, look at the signal's own
`pct_metric` output directly.

### 19.3 Gates — pair × gate

```python
g, a = RUN('base | 1M hold | singles',
           tenors=['1M'],
           exit_rule_factory=lambda: ExitAfterNDays('1M'),
           gates=enumerate_gate_specs(sizes=(1,)))     # NO_GATE + 6 singles
```

Note there is **no `max_concurrent`** — the gates branch asserts it's `None`. Returns
`(grid, attr)` and prints three tables: `print_grid` (pair × gate),
`print_gate_attribution` (§17.6), `print_gate_consistency` (§17.7).

Screening order, from §17.3:

1. **Attribution.** `removed_total_pnl` must be **negative**. Positive means the gate threw
   away money; stop there regardless of Calmar.
2. **Consistency.** Hit rate across pairs, discounted for the fact that 7 USD pairs is
   maybe 2–3 independent tests.
3. **`veto_rate` sanity.** A gate vetoing 50%+ isn't a filter, it's a different strategy on
   half the sample. Fine — but judge it as such.

Then confirm survivors under your real convention via `gated()` (§18.6).

### 19.4 The threshold frontier — the strongest single test

```python
g = RUN('ivrv frontier | short 1M 25d strangle | Hold to Expiry | Max 1 Trade',
        tenors=['1M'],
        max_concurrent=1,
        signals={'none': _always_on,
                 **{f'ivrv{p}': _ivrv(p) for p in (50, 60, 70, 80, 90)}},
        sort_by='pair')
```

Read `coverage` and `ret_on_prem` **together, down each pair's block**. A real signal
traces a monotone frontier — tighter threshold, lower coverage, better per-trade edge:

```
    pair  tenor   label  ret_on_prem  ...  n_trades  coverage  ...  vol_spread  real_vrp
  EURUSD     1M    none         0.06  ...        56      100%  ...       -0.11      0.14
  EURUSD     1M  ivrv50         0.07  ...        54       96%  ...       -0.19      0.09
  EURUSD     1M  ivrv60         0.09  ...        52       92%  ...       -0.30     -0.02
  EURUSD     1M  ivrv70         0.12  ...        49       87%  ...       -0.52     -0.24
  EURUSD     1M  ivrv80         0.16  ...        46       81%  ...       -0.69     -0.41
  EURUSD     1M  ivrv90         0.21  ...        38       67%  ...       -0.96     -0.68
```
*(illustrative — this is what "passing" looks like)*

Monotone in all four columns is a real effect: `ret_on_prem` rising, `coverage` falling, and
**both** vol columns getting steadily *more negative* (more premium captured), with the
~0.28 gap between them holding constant as the butterfly you collect doesn't change.

A **peak** at one threshold with worse results either side is noise, and you can see it
without any significance test. This is worth more than the always-on comparison because it
uses the signal's own internal structure as the control instead of a single alternative
config.

### 19.5 Every other dial

```python
# exits
g = RUN('exit +1W',  exit_rule_factory=lambda: ExitAfterNDays('1W'))
g = RUN('1W left',   exit_rule_factory=lambda: ExitAtDaysRemaining('1W'))
g = RUN('exit +10d', exit_rule_factory=lambda: ExitAfterNDays(10))        # int = calendar days
g = RUN('TP/SL',     exit_rule_factory=lambda: TakeProfitStopLoss(take_profit=100_000,
                                                                  stop_loss=150_000))
# hedges
g = RUN('2% band', hedge_rule_factory=lambda: DeltaBandHedge(0.02))
g = RUN('25bp',    hedge_rule_factory=lambda: SpotMoveHedge(0.0025))
g = RUN('half',    hedge_rule_factory=lambda: PartialHedge(0.5))
g = RUN('gamma',   hedge_rule_factory=lambda: GammaScaledHedge(
                       [(0.0020, 1), (0.0010, 2), (0.0, 5)]))   # daily >0.20%, else 2d, else 5d
# structures
g = RUN('25d strangle', legs_fn=_strangle(0.25, 0.25))    # note the parens
g = RUN('10d strangle', legs_fn=_strangle(0.10, 0.10))
g = RUN('broken wing',  legs_fn=_strangle(0.25, 0.10))
g = RUN('ATM straddle', legs_fn=_straddle)                # bare — already fn(s)
g = RUN('condor',       legs_fn=_custom)
# universe / sizing / costs
g = RUN('LONG',      direction=+1)
g = RUN('50m',       notional=50_000_000)
g = RUN('5bp costs', tc_fraction=0.0005)
g = RUN('zero cost', tc_fraction=0.0)                     # diagnostic, not tradeable
g = RUN('G3 only',   pairs=['EURUSD','USDJPY','GBPUSD'])
g = RUN('term',      tenors=['1W','1M','3M'])
```

---

# PART XII — TRAPS INDEX

## §20 Everything that can silently mislead you

Ordered by how much damage it does.

### 20.1 Unit and comparability traps

| # | Trap | Where |
|---|---|---|
| 1 | **`to_usd=True` now converts both lenses** — book *and* per-trade log, off one shared daily factor — so `expectancy` and friends are USD and cross-pair comparable. Residual traps: `to_usd=False` (the `evaluate` default, though `run_grid` defaults True) leaves everything base-ccy, and `ret_on_prem` is **not invariant** under conversion (numerator at daily rates, `net_premium` at the entry rate), so pre- and post-change figures won't match. | §15.4 |
| 2 | **The book has no flat rows** — idle days are absent, not zero. `sharpe`/`sortino` are therefore already per-deployed-day (fine), but `calmar` used to be annualized over deployed days (inflated ~3× for a selective signal) and `coverage` was identically 100%. **Both fixed**; `net_pnl` still scales with deployment. | §15.5 |
| 3 | **`vol_spread` and `real_vrp` both have inverted signs** — both are realized *minus* implied, so for a short book more negative is better. Opposite to every other metric in the table. | §15.1a |
| 3b | **Both vol columns are EX-POST, not ex-ante.** `vol_spread` is `realised_vol − avg_entry_sigma`, *not* the implied spread the signal saw at entry. Nothing in `trade_log` records the signal's entry condition, so the grid cannot answer "did the signal select rich implied?" | §15.1a |
| 4 | **`payoff_ratio` is neutral at 1.0, not 0.** A 0-centred colour scale paints every cell "good". | §16.1 |
| 5 | **Attribution P&L is BASE currency by design.** Only compare kept-vs-removed *within* a row, where the FX factor cancels. | §17.6 |
| 6 | **Constant notional across tenors ≠ constant risk.** A 3M strangle carries far more vega than a 1W at the same notional, so tenor *magnitudes* aren't comparable without normalizing. | §19.1 |

### 20.2 Statistical traps

| # | Trap | Where |
|---|---|---|
| 7 | **`t_stat` is per-trade, `sharpe` is per-day.** They legitimately disagree, most on low-`n_trades` cells. | §15.1 / §15.2 |
| 8 | **A selective signal shows a *lower* `t_stat` even when it's better** — halving `n_trades` needs a 41% better per-trade edge to print the same value. | §15.5 |
| 9 | **Percentile signals cluster**, so `t_stat` overstates significance *more* for a signal than for always-on. Combined with #8, don't use it to decide. | §15.5 |
| 10 | **7 USD pairs is not 7 independent tests.** All share a dollar factor; AUD/NZD are near-duplicates. Effective breadth ~2–3. | §17.7 |
| 11 | **`calmar` rests on one order statistic** from one realized path. Noisiest of the three risk-adjusted reads on small samples. | §15.2 |
| 12 | **The best cell of a wide sweep is usually the least-sampled one.** Expected max \|t\| under a pure-noise null across ~7 independent looks is already ~2.0–2.2. | §19.1 |

### 20.3 Mechanical traps

| # | Trap | Where |
|---|---|---|
| 12b | ~~`live_trade = expiry > today`~~ — **FIXED** to `exit_reason is None`. The old test mislabelled every exit-rule-closed trade entered within one tenor of today as "live", so `trade_metrics(settled_only=True)` silently dropped real settled trades — always the most recent ones. Only `HoldToExpiry` made it coincidentally correct. | §5.6 |
| 12c | ~~`DeltaBandHedge` tested the legs' GROSS delta~~ — **FIXED** to the unhedged residual (`target_hedge - current_hedge`). The old test ignored the hedge already on, so the rule either never rehedged or degenerated into `DailyHedge`. | §9 |
| 13 | **`max_concurrent` is asserted `None` with `gates=`.** Not a limitation but a correctness requirement — removing an entry frees a slot, so the surviving trades genuinely differ. Use `gated()` if you need both. | §17.2 / §18.6 |
| 14 | **Signals with `max_concurrent=1` are NOT a subset of always-on.** Entry dates shift, so no kept-vs-removed decomposition exists for signals. | §15.5 / §17.2 |
| 15 | **Stateful hedge/exit rules must come from factories.** `SpotMoveHedge`/`GammaScaledHedge` state is set in `__init__` and not reset by `bind()`. A shared instance leaks silently. | §9 / §10.2 |
| 16 | **`ExitAtDaysRemaining('1W')` raises on `tenors=['1W']`** — surfaces as `status='error'` rows, so a 14-cell grid quietly replaces a 21-cell one. Check the `status` column. | §8 / §16.4 |
| 17 | **No transaction cost on the natural-expiry unwind**, but full cost on an early exit. Baked into any hold-to-expiry vs exit-early comparison. | §5.6 |
| 18 | **P&L buckets don't sum to the exact reprice.** `recon_resid` is the by-design gap, not an error to eliminate. | §7 |
| 19 | **`net_delta_approx` mixes a start-of-day greek with an end-of-day hedge.** Directional check only. | §14.2 |
| 20 | **Heatmaps raise if you vary tenor AND label.** Put the tenor into the label (`'1M\|trend'`). Tables are fine. | §16.6 |

### 20.4 Signal-generation traps

| # | Trap | Where |
|---|---|---|
| 21 | **`get_vol_signal(metric=...)` accepts only `'IV'`, `'RV'`, `'VD'`** — `'IV-RV'` raises. Docstring corrected. | §11.3 |
| 22 | **xCCY uses a *time*-based window (`'365D'`), Implied_Realized a *row*-based one (252 obs.)** Not the same quantity. | §12.1 |
| 23 | ~~xCCY's `days_back` trimmed ROWS, not calendar days~~ — **FIXED.** It was a no-op at `days_back=1825`, so the xccy signal silently spanned ~380 calendar days MORE than an always-on/ivrv signal built with the same `days_back`, and the grid compared different samples. Now a calendar cutoff, and warm-up `min_periods` raised 2→20. | §12.1 |
| 24 | **xCCY's `df_all.dropna()` is an inner join over the whole basket.** One gappy ticker truncates every pair's series. | §12.1 |
| 25 | **A gate built over a shorter window than the signal silently no-ops.** `apply_regime_gate(warn=True)` and `_keep_mask`'s <95% warning exist for this; don't suppress them. | §13.8 / §17.5 |
| 26 | **Warm-up ALLOWS by default.** `level` and `har` need ~252+ days before they can veto at all. Flip `on_missing='veto'` only deliberately. | §13.2 |
| 27 | **`spike` is the one check with no warm-up allowance** — active from day one. | §13.3 |
| 28 | **`termstructure` raises if front == back tenor.** Only tenors in `_NEXT_TENOR` get a sensible default. | §13.3 |
| 29 | **Point-in-time discipline is yours.** Nothing checks that your signal was built without look-ahead. | §10.5 |
| 30 | **One Bloomberg pull per `run_signal_backtest` call, not per trade — but no cross-call caching.** A 21-cell grid is 21 pulls. (The *gate* caches are the exception, §13.6.) | §10.5 |

---

## Reference

- Ravagli, *"Harvesting the FX skew premium"*, Risk.net, June 2024 — the `_be` attribution
  framework and the `NU_BE_C`/`RHO_BE_D` constants (§5.7).
- Clark, *Foreign Exchange Option Pricing* — §1.4 Spot Settlement Rules, §1.5 Expiry and
  Delivery Rules (§1.4 of this guide).
- Corsi (2009), HAR — the `har` regime check (§13.3).
