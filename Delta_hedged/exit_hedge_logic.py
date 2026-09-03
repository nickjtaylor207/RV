from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple, Union

from trading_calendar import add_tenor


# ═══════════════════════════════════════════════════════════════════════════════
# Exit Rules — decide whether to close the position before natural expiry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExitContext:
    """Snapshot of trade state handed to ExitRule.check() once per backtest day."""
    date:            date
    day_index:       int     # 1-based count of backtest days elapsed (records so far + 1)
    days_in_trade:   int     # calendar days since entry
    t_remaining:     int     # calendar days remaining to original expiry
    spot:            float
    entry_spot:      float
    sigma_avg:       float   # today's vega-weighted smile vol
    entry_sigma_avg: float   # entry vega-weighted smile vol
    running_net_pnl: float   # cumulative net P&L through today (pre exit-tc)


class ExitRule:
    """
    Subclass and override check(). Returning True closes the position at
    today's mark-to-market value (hedge fully unwound) and stops the backtest
    before the natural expiry settlement.
    """
    name = "exit_rule"

    def bind(self, entry_date: date, fxc, option_tenor_days: int) -> None:
        """
        Optional hook, called once by the engine before the backtest loop
        starts. Gives rules that need calendar-aware setup (e.g. resolving a
        tenor string like '1W' into a day count via the same FX convention
        used for the option's own expiry) access to entry_date / fxc /
        option_tenor_days. No-op by default.
        """
        pass

    def check(self, ctx: ExitContext) -> bool:
        raise NotImplementedError


class HoldToExpiry(ExitRule):
    """Baseline — never exits early; natural t_remaining<=0 settlement ends the trade."""
    name = "hold_to_expiry"

    def check(self, ctx: ExitContext) -> bool:
        return False


class ExitAfterNDays(ExitRule):
    """
    Close once a hold period has elapsed since entry. hold_for is either:
      - an int  : calendar days, e.g. ExitAfterNDays(10)
      - a tenor : 'ND', 'NW', 'NM', 'NY' (any N), e.g. ExitAfterNDays('1W')
                  resolved to a day count via the same add_tenor() FX
                  convention used for the option's own expiry.
    The resolved day count must be less than the trade's own tenor.
    """
    name = "exit_after_n_days"

    def __init__(self, hold_for: Union[int, str]):
        assert isinstance(hold_for, (int, str)), \
            "hold_for must be an int (calendar days) or a tenor string (e.g. '1W', '1M')"
        if isinstance(hold_for, int):
            assert hold_for > 0, "hold_for must be positive"
        self.hold_for = hold_for
        self.n_days   = hold_for if isinstance(hold_for, int) else None

    def bind(self, entry_date: date, fxc, option_tenor_days: int) -> None:
        if isinstance(self.hold_for, str):
            exit_date    = add_tenor(entry_date, self.hold_for, fxc)
            self.n_days  = (exit_date - entry_date).days
            assert self.n_days > 0, \
                f"tenor '{self.hold_for}' resolved to a non-positive day count"
        assert self.n_days < option_tenor_days, (
            f"ExitAfterNDays hold period ({self.n_days}d"
            f"{f' from {self.hold_for!r}' if isinstance(self.hold_for, str) else ''}) "
            f"must be less than the option's own tenor ({option_tenor_days}d)"
        )

    def check(self, ctx: ExitContext) -> bool:
        return ctx.days_in_trade >= self.n_days


class ExitAtDaysRemaining(ExitRule):
    """
    The mirror image of ExitAfterNDays — close once the time remaining to the
    original expiry falls to (or below) a threshold, rather than once a hold
    period has elapsed since entry. remaining_for is either:
      - an int  : calendar days, e.g. ExitAtDaysRemaining(7)
      - a tenor : 'ND', 'NW', 'NM', 'NY' (any N), e.g. ExitAtDaysRemaining('1W')
                  resolved to a day count via the same add_tenor() FX
                  convention used for the option's own expiry.
    e.g. on a 1M trade, ExitAtDaysRemaining('1W') holds for ~3W then exits
    with 1W left to expiry. The resolved day count must be less than the
    trade's own tenor.
    """
    name = "exit_at_days_remaining"

    def __init__(self, remaining_for: Union[int, str]):
        assert isinstance(remaining_for, (int, str)), \
            "remaining_for must be an int (calendar days) or a tenor string (e.g. '1W', '1M')"
        if isinstance(remaining_for, int):
            assert remaining_for > 0, "remaining_for must be positive"
        self.remaining_for = remaining_for
        self.n_days        = remaining_for if isinstance(remaining_for, int) else None

    def bind(self, entry_date: date, fxc, option_tenor_days: int) -> None:
        if isinstance(self.remaining_for, str):
            tenor_date  = add_tenor(entry_date, self.remaining_for, fxc)
            self.n_days = (tenor_date - entry_date).days
            assert self.n_days > 0, \
                f"tenor '{self.remaining_for}' resolved to a non-positive day count"
        assert self.n_days < option_tenor_days, (
            f"ExitAtDaysRemaining threshold ({self.n_days}d"
            f"{f' from {self.remaining_for!r}' if isinstance(self.remaining_for, str) else ''}) "
            f"must be less than the option's own tenor ({option_tenor_days}d)")
    def check(self, ctx: ExitContext) -> bool:
        return ctx.t_remaining <= self.n_days


class TakeProfitStopLoss(ExitRule):
    """
    Close once cumulative net P&L crosses a profit target or loss limit.
    Pass None to disable either side. stop_loss is a positive magnitude.
        - TakeProfitStopLoss(take_profit='X', stop_loss='Y')
    """
    name = "take_profit_stop_loss"

    def __init__(self, take_profit: Optional[float] = None,
                 stop_loss: Optional[float] = None):
        assert take_profit is not None or stop_loss is not None, \
            "provide at least one of take_profit / stop_loss"
        assert stop_loss is None or stop_loss > 0, "stop_loss must be a positive magnitude"
        self.take_profit = take_profit
        self.stop_loss   = stop_loss

    def check(self, ctx: ExitContext) -> bool:
        if self.take_profit is not None and ctx.running_net_pnl >= self.take_profit:
            return True
        if self.stop_loss is not None and ctx.running_net_pnl <= -self.stop_loss:
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Hedge Rules — decide whether/how much to re-hedge delta each day
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HedgeContext:
    """Snapshot of trade state handed to HedgeRule.decide() once per backtest day."""
    date:               date
    day_index:          int     # 1-based count of backtest days elapsed (records so far + 1)
    days_in_trade:      int     # calendar days since entry
    t_remaining:        int     # calendar days remaining to original expiry
    spot:               float
    prev_spot:          float
    dS:                 float
    entry_spot:         float
    net_delta_pretrade: float   # today's start-of-day (pre-hedge) signed net delta
    net_gamma_pretrade: float   # today's start-of-day (pre-hedge) signed net gamma (1% bump)
    total_notional:     float
    current_hedge:      float   # spot units currently held, before today's decision
    target_hedge:       float   # spot units that would fully offset net_delta_pretrade


class HedgeRule:
    """
    Subclass and override decide(). Return the fraction in [0, 1] of the gap
    between current_hedge and target_hedge to trade today: 1.0 = full daily
    rehedge (the baseline), 0.0 = skip entirely, 0.5 = trade half the gap, etc.

    Greek P&L attribution (theta/vega/vanna/volga/option repricing) is always
    computed independently of this decision — it comes from the option's own
    Greeks and today's realized market move, not from the hedge. Skipping or
    partially hedging only changes hedge_pnl/hedge_carry (computed off
    whatever hedge is actually carried forward) and the transaction cost paid.
    """
    name = "hedge_rule"

    def bind(self, entry_date: date, fxc, option_tenor_days: int) -> None:
        """
        Optional hook, called once by the engine before the backtest loop
        starts. Gives rules that need calendar-aware setup access to
        entry_date / fxc / option_tenor_days. No-op by default.
        """
        pass

    def decide(self, ctx: HedgeContext) -> float:
        raise NotImplementedError


class DailyHedge(HedgeRule):
    """Baseline — fully rehedge to target delta every day."""
    name = "daily_hedge"
    def decide(self, ctx: HedgeContext) -> float:
        return 1.0


class DeltaBandHedge(HedgeRule):
    """
    Only rehedge (fully, to target) once the book's UNHEDGED DELTA RESIDUAL
    exceeds a band, expressed as a fraction of total notional.
      e.g. DeltaBandHedge(0.02) hedges only once the residual > 2% of notional.

    The residual is `target_hedge - current_hedge` (== the engine's `hedge_gap`):
    today's option delta net of the hedge actually being carried. That is the
    exposure you are running, and the quantity a delta band is meant to cap.

    NOTE this previously tested `abs(ctx.net_delta_pretrade)` — the option legs'
    own GROSS delta, ignoring the existing hedge entirely — which made the rule
    behave nothing like a band:
      * a short strangle sits near delta-flat at entry, so with a hedge already
        on, a gross delta below the band meant the rule NEVER rehedged for the
        whole trade, holding the day-1 hedge to expiry;
      * conversely, once gross delta sat persistently above the band it rehedged
        every single day, i.e. it silently degenerated into DailyHedge.
    Either way the band threshold was not controlling hedge error. It also mixed
    timing: net_delta_pretrade is a START-of-day greek while target_hedge is
    computed off today's, so the two were a day apart.
    """
    name = "delta_band_hedge"
    def __init__(self, band: float):
        assert band > 0, "band must be positive"
        self.band = band
    def decide(self, ctx: HedgeContext) -> float:
        if ctx.total_notional == 0:
            return 1.0
        residual = abs(ctx.target_hedge - ctx.current_hedge)
        return 1.0 if residual / ctx.total_notional > self.band else 0.0


class SpotMoveHedge(HedgeRule):
    """
    Only rehedge (fully) once spot has moved more than bps_threshold (decimal,
    e.g. 0.0025 = 25bps) away from the spot level as of the last rehedge.
    Stateful — tracks the spot level at the last hedge, seeded from entry spot.
    """
    name = "spot_move_hedge"

    def __init__(self, bps_threshold: float):
        assert bps_threshold > 0, "bps_threshold must be positive"
        self.bps_threshold    = bps_threshold
        self._last_hedge_spot = None

    def decide(self, ctx: HedgeContext) -> float:
        if self._last_hedge_spot is None:
            self._last_hedge_spot = ctx.entry_spot
        if abs(ctx.spot / self._last_hedge_spot - 1) > self.bps_threshold:
            self._last_hedge_spot = ctx.spot
            return 1.0
        return 0.0


class GammaScaledHedge(HedgeRule):
    """
    Increase hedging frequency as gamma grows. gamma_schedule is a list of
    (gamma_frac_threshold, interval_days) pairs. Each day, the highest
    threshold that |net_gamma_pretrade|/total_notional exceeds sets the hedge
    interval (in backtest days since the last hedge); if none are exceeded,
    the lowest-threshold entry's interval is used as the default.

    e.g. GammaScaledHedge([(0.0020, 1), (0.0010, 2), (0.0, 5)])
    -> hedge every day once gamma/notional > 0.20%, every 2 days above 0.10%,
       otherwise every 5 days.

    `interval_days` counts BACKTEST days (decide() calls), not calendar days, so
    an interval of 5 spans a business week — 7 calendar days — not 5. This is
    deliberate (you can only act on days the market printed) but it means the
    rule's own counter and the engine's `days_since_hedge` diagnostic column
    (which IS calendar days) will not agree over weekends and holidays.
    """
    name = "gamma_scaled_hedge"

    def __init__(self, gamma_schedule: List[Tuple[float, int]]):
        assert gamma_schedule, "provide at least one (gamma_frac_threshold, interval_days) pair"
        for _, interval in gamma_schedule:
            assert interval > 0, "interval_days must be positive"
        self.gamma_schedule    = sorted(gamma_schedule, key=lambda pair: -pair[0])
        self._days_since_hedge = 0

    def _interval_for(self, gamma_frac: float) -> int:
        for threshold, interval in self.gamma_schedule:
            if gamma_frac >= threshold:
                return interval
        return self.gamma_schedule[-1][1]

    def decide(self, ctx: HedgeContext) -> float:
        self._days_since_hedge += 1
        gamma_frac = abs(ctx.net_gamma_pretrade) / ctx.total_notional if ctx.total_notional else 0.0
        if self._days_since_hedge >= self._interval_for(gamma_frac):
            self._days_since_hedge = 0
            return 1.0
        return 0.0


class PartialHedge(HedgeRule):
    """Always trade a fixed fraction of the gap to full rehedge, every day."""
    name = "partial_hedge"

    def __init__(self, fraction: float):
        assert 0 < fraction <= 1, "fraction must be in (0, 1]"
        self.fraction = fraction

    def decide(self, ctx: HedgeContext) -> float:
        return self.fraction










# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------








# # ── Exit rules ─────────────────────────────────────────────────────────────────

# exit_rules = {
#     'hold_to_expiry': HoldToExpiry(),
#     'exit_after_5d':  ExitAfterNDays(5),
#     'exit_after_10d': ExitAfterNDays(10),
#     'tp_10k':         TakeProfitStopLoss(take_profit=10_000),
#     'sl_10k':         TakeProfitStopLoss(stop_loss=10_000),
#     'tp10k_sl10k':    TakeProfitStopLoss(take_profit=10_000, stop_loss=10_000),
# }

# # ── Hedging rules ─────────────────────────────────────────────────────────────────

# hedge_rules = {
#     'daily_hedge':      DailyHedge(),
#     'delta_band_5pct':  DeltaBandHedge(0.05),
#     'spot_move_30bps':  SpotMoveHedge(0.003),
#     'gamma_scaled':     GammaScaledHedge([(0.0020, 1), (0.0010, 2), (0.0, 5)]),
#     'partial_50pct':    PartialHedge(0.5),
# }


