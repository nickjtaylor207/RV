"""
Volga / vanna signals — producer contract, the carry producer, and the decision layer.

    nu   prices VOLGA   (curvature / butterfly)      — symmetric
    rho  prices VANNA   (slope / risk reversal)      — directional

Sign convention, fixed once and used everywhere
-----------------------------------------------
    gap   = pct(implied) - pct(realized)        rich  => gap > 0
    score = -gap, rank-normalised to [-1, +1]

`score` IS the target position in that Greek: positive = LONG, negative = SHORT.

    volga rich (gap>0) -> score<0 -> short volga -> sell the fly
    rho far below realized (gap<0) -> score>0 -> long vanna -> buy calls/sell puts

One rule, both risks, no per-risk special case to forget.

Layering
--------
    Layer 1  producers   -> compute() returns {'signals', 'context'}   (this file)
    Layer 2  combiner    -> NOT BUILT. With one producer it is the identity
                            function; building it now would bake in the wrong
                            assumptions. Add it when a second producer exists.
    Layer 3  decide()    -> dead band, gates, hysteresis, target exposure

Every producer emits FULL HISTORY, not a snapshot: the score's own distribution
is what self-calibrates the thresholds, and forward-IC validation needs nothing
else.

Burn-in
-------
    longest realized window (60bd) + MIN_PCT_OBS (126) + MIN_SCORE_OBS (252)
    ~= 440bd ~= 1.75y before the first score exists.
With a 5y pull that leaves ~3.2y of scored history. Lower MIN_SCORE_OBS to trade
sample length against threshold stability, or pull more history.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from dataset import FXVolDataset
from SABR import (REALIZED_WINDOWS, _realized_rho_nu, calibrate_sabr_history,
                  tenor_to_calendar_days)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — the contract
# ══════════════════════════════════════════════════════════════════════════════

RISKS = ('volga', 'vanna')

# nu prices volga, rho prices vanna.
RISK_PARAM = {'volga': 'nu', 'vanna': 'rho'}

SIGNAL_COLS = ['date', 'pair', 'tenor', 'risk', 'name',
               'raw', 'score', 'confidence', 'horizon_bd']

# Percentile horizons, in BUSINESS days. rolling(W, min_periods<W) is expanding
# until the window fills, so '5Y' means "up to 5y, at least MIN_PCT_OBS" rather
# than a strict 5y window — with only 5y of data a strict one would exist on a
# single date. To get a true rolling 5y percentile you need ~10y of history.
PCT_WINDOWS = {'1Y': 252, '5Y': 1260}

MIN_PCT_OBS   = 126   # before a percentile is emitted at all
MIN_SCORE_OBS = 252   # gaps needed before the gap's own rank is meaningful

# Risks whose implied and realized estimates share units, so their LEVELS can be
# compared directly. SABR rho and corr(dlnS, dlnATM) are both correlations in
# [-1, +1]. Implied and realized NU are not comparable in level -- implied nu
# absorbs smile curvature that stdev(dlnv) cannot produce -- which is the whole
# reason the score is a rank-of-rank.
LEVEL_COMPARABLE_RISKS = ('vanna',)

# Where levels ARE comparable, |impl - real| measured against the sampling error
# of the realized correlation is extra evidence the percentile gap discards.
# It scales CONFIDENCE rather than vetoing: the percentile signal claims each
# series is unusual within its own history, which is a different (and still
# valid) claim from "the levels differ". So an insignificant level gap should
# shrink the position, not zero it.
LEVEL_SUPPORT_Z     = 1.5    # |gap| at this many SE earns full weight
LEVEL_SUPPORT_FLOOR = 0.40   # never de-rate below this


@dataclass(frozen=True)
class SignalSpec:
    """Static description of a producer. One per signal, immutable."""
    name: str
    risks: tuple
    description: str


def validate_signal_frame(df: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """
    Enforce the producer contract. Cheap to run, and it catches the failures that
    are otherwise invisible until they silently corrupt a combined score:
    unnormalised `score`, a stray risk name, duplicate keys.
    """
    missing = [c for c in SIGNAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{spec.name}: signal frame missing columns {missing}.")

    bad_risk = set(df['risk'].unique()) - set(RISKS)
    if bad_risk:
        raise ValueError(f"{spec.name}: unknown risk(s) {bad_risk}; expected {RISKS}.")

    dup = df.duplicated(['date', 'pair', 'tenor', 'risk']).sum()
    if dup:
        raise ValueError(f"{spec.name}: {dup} duplicate (date,pair,tenor,risk) rows.")

    for col, lo, hi in (('score', -1.0, 1.0), ('confidence', 0.0, 1.0)):
        s = df[col].dropna()
        if len(s) and (s.min() < lo - 1e-9 or s.max() > hi + 1e-9):
            raise ValueError(
                f"{spec.name}: `{col}` out of [{lo}, {hi}] "
                f"(got [{s.min():.3f}, {s.max():.3f}]). Producers must normalise "
                f"before emitting -- otherwise the loudest signal wins the average.")

    return df[SIGNAL_COLS].sort_values(['pair', 'tenor', 'risk', 'date'])


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — the carry producer (implied vs realized, from the smile panel)
# ══════════════════════════════════════════════════════════════════════════════

CARRY_SPEC = SignalSpec(
    name='vv_carry',
    risks=RISKS,
    description='Implied-vs-realized percentile gap on SABR nu (volga) and '
                'rho (vanna), two realized windows per tenor, 1Y/5Y horizons.',
)


def _roll_pct(s: pd.Series, W: int) -> pd.Series:
    """
    Point-in-time percentile: rank of s_t within the trailing W observations,
    in percent. Matches (win <= cur).mean()*100 but vectorised.
    """
    return s.rolling(W, min_periods=min(MIN_PCT_OBS, W)).rank(pct=True) * 100.0


def _tenor_horizon_bd(pair: str, tenor: str) -> int:
    """The option tenor's own horizon in business days (for the contract field)."""
    t_days, _, _ = tenor_to_calendar_days(pair, tenor)
    return max(1, int(round(t_days * 252.0 / 365.0)))


def _tenor_reliability(long_window_bd: int) -> float:
    """
    How much to trust a tenor's realized estimate, from its long window length.
    A 10-point correlation earns roughly half the weight of a 40-point one; this
    is what handles the 1W noise problem structurally instead of by a hard floor.
    """
    return float(np.clip(np.sqrt(long_window_bd / 40.0), 0.30, 1.0))


class VolgaVannaCarry:
    """
    The producer built on your smile panel.

    Per (date, pair, tenor, risk):
      level      = percentile of realized on the LONG window
      momentum   = pct(short) - pct(long)      -> context, feeds a gate, NOT the score
      gap_h      = pct(implied) - pct(realized_long)     for h in 1Y, 5Y
      concord    = |gap|_min / |gap|_max if signs agree else 0
      gap        = mean(gap_1Y, gap_5Y) x (0.5 + 0.5*concord)
                   -- full weight only when the horizons agree in sign AND in
                   magnitude. Sign agreement alone passes (-4, -44) unhaircut,
                   where one horizon carries everything (the 'already
                   mean-reverted' trap in its subtler form).
      score      = -(2*expanding_rank(gap) - 1)          in [-1, +1]
      confidence = window_agreement x horizon_agreement
                   x tenor_reliability x sample_adequacy
                   x level_support     (vanna only -- see LEVEL_COMPARABLE_RISKS)

    Note on rank-normalising the gap: score 0 means "as rich as usual for this
    pair", NOT "fairly priced". That is deliberate — it is what makes the score
    comparable across pairs and immune to the structural level offset between
    SABR nu and the realized proxy. It also means a persistently rich pair reads
    flat, which is correct for a relative-value signal and wrong if you wanted an
    absolute one.
    """

    spec = CARRY_SPEC

    def __init__(self,
                 windows: Optional[dict] = None,
                 pct_windows: Optional[dict] = None,
                 min_score_obs: int = MIN_SCORE_OBS,
                 delta_pts: Sequence[int] = (35, 25, 10),
                 beta: float = 0.5):
        self.windows       = dict(REALIZED_WINDOWS if windows is None else windows)
        self.pct_windows   = dict(PCT_WINDOWS if pct_windows is None else pct_windows)
        self.min_score_obs = int(min_score_obs)
        self.delta_pts     = tuple(delta_pts)
        self.beta          = float(beta)

    # ── per-tenor assembly ───────────────────────────────────────────────────
    def _tenor_frame(self, pair: str, tenor: str, dataset: FXVolDataset,
                     imp: pd.DataFrame, spot_r: pd.Series) -> Optional[pd.DataFrame]:
        """Aligned implied + short/long realized series for one tenor."""
        col = (pair, tenor, 'ATM')
        if col not in dataset.vol_surface.columns:
            warnings.warn(f"No ATM series for {pair} {tenor}; tenor skipped.")
            return None

        wins = sorted(int(w) for w in self.windows[tenor])
        if len(wins) < 2:
            raise ValueError(f"{tenor}: need at least two realized windows, got {wins}.")
        n_short, n_long = wins[0], wins[-1]

        atm = (dataset.vol_surface[col] / 100.0).dropna()
        rea_s = _realized_rho_nu(spot_r, atm, n_short)
        rea_l = _realized_rho_nu(spot_r, atm, n_long)
        if rea_s.empty or rea_l.empty:
            warnings.warn(f"No realized data for {pair} {tenor}; tenor skipped.")
            return None

        f = pd.concat({
            'nu_impl':   imp[(tenor, 'nu')],
            'rho_impl':  imp[(tenor, 'rho')],
            'nu_real_s': rea_s['nu'],  'rho_real_s': rea_s['rho'],
            'nu_real_l': rea_l['nu'],  'rho_real_l': rea_l['rho'],
        }, axis=1).dropna()
        if f.empty:
            return None

        # context-only columns: present where available, never gate the join
        f['atm']     = atm.reindex(f.index)
        f['rmse_vp'] = imp[(tenor, 'rmse_vp')].reindex(f.index)
        f.attrs['n_short'], f.attrs['n_long'] = n_short, n_long
        return f

    # ── the score, for one tenor and one risk ────────────────────────────────
    def _score_one(self, f: pd.DataFrame, risk: str) -> pd.DataFrame:
        p       = RISK_PARAM[risk]
        n_long  = f.attrs['n_long']
        out     = pd.DataFrame(index=f.index)

        # rolling percentiles of implied and of both realized windows
        pct = {}
        for h, W in self.pct_windows.items():
            pct[('impl',   h)] = _roll_pct(f[f'{p}_impl'],   W)
            pct[('real_l', h)] = _roll_pct(f[f'{p}_real_l'], W)
            pct[('real_s', h)] = _roll_pct(f[f'{p}_real_s'], W)

        # gap per horizon: implied vs the LONG realized window (the level)
        gaps = {h: pct[('impl', h)] - pct[('real_l', h)] for h in self.pct_windows}
        gap_df = pd.DataFrame(gaps)

        # Blend horizons. A sign-only agreement test is too weak: gaps of
        # (-4, -44) agree in sign, yet one horizon carries the entire signal and
        # the mean of -24 overstates the other. So grade CONCORDANCE
        # continuously on the magnitude ratio, and keep the 0.5 floor so a
        # one-horizon signal is weakened rather than discarded.
        agree     = np.sign(gap_df).nunique(axis=1).le(1) | gap_df.isna().any(axis=1)
        mag_ratio = (gap_df.abs().min(axis=1) /
                     gap_df.abs().max(axis=1).replace(0.0, np.nan))
        concord   = pd.Series(np.where(agree, mag_ratio.fillna(1.0).clip(0.0, 1.0),
                                       0.0), index=f.index)
        gap = gap_df.mean(axis=1) * (0.5 + 0.5 * concord)
        gap = pd.Series(gap, index=f.index)

        # score: rank the gap within its OWN history, centre, flip sign so that
        # score = target position in the Greek
        rank  = gap.expanding(min_periods=self.min_score_obs).rank(pct=True)
        score = -(2.0 * rank - 1.0)

        # ── confidence ───────────────────────────────────────────────────────
        # short vs long window agreement on the 1Y horizon
        h0        = '1Y' if '1Y' in self.pct_windows else list(self.pct_windows)[0]
        win_agree = 1.0 - (pct[('real_s', h0)] - pct[('real_l', h0)]).abs() / 100.0
        hor_agree = 0.5 + 0.5 * concord
        reliab    = _tenor_reliability(n_long)
        adequacy  = (gap.expanding().count() / float(self.min_score_obs)).clip(0, 1)

        # Level support: only where implied and realized share units (vanna).
        # se of a sample correlation ~ (1 - r^2)/sqrt(N-1); a level gap inside
        # that is a coin flip, and the rank transform will happily turn it into
        # a large percentile gap. De-rate rather than veto.
        if risk in LEVEL_COMPARABLE_RISKS:
            se      = ((1.0 - f[f'{p}_real_l'] ** 2)
                       / np.sqrt(max(2.0, float(n_long) - 1.0)))
            lvl_z   = (f[f'{p}_impl'] - f[f'{p}_real_l']).abs() / se.replace(0.0, np.nan)
            lvl_sup = (lvl_z / LEVEL_SUPPORT_Z).clip(LEVEL_SUPPORT_FLOOR, 1.0)
            lvl_sup = lvl_sup.fillna(1.0)
        else:
            lvl_z   = pd.Series(np.nan, index=f.index)
            lvl_sup = pd.Series(1.0, index=f.index)

        conf = (win_agree.clip(0, 1) * hor_agree * reliab
                * adequacy * lvl_sup).clip(0.05, 1.0)
        conf = conf.where(score.notna())

        out['raw']        = gap
        out['score']      = score
        out['confidence'] = conf

        # ── context: everything a gate or a reason string needs ──────────────
        for h in self.pct_windows:
            out[f'gap_{h}']       = gap_df[h]
            out[f'pct_impl_{h}']  = pct[('impl', h)]
            out[f'pct_real_{h}']  = pct[('real_l', h)]
        h_last                    = list(self.pct_windows)[-1]
        out['pct_real_short']     = pct[('real_s', h0)]
        out['realized_momentum']  = pct[('real_s', h0)] - pct[('real_l', h0)]
        out['real_level_pct']     = pct[('real_l', h_last)]
        # Short-window level, on the SAME horizon as real_level_pct. The floor
        # gate reads the long window, which misses a short window sitting at an
        # all-time low — the sharpest form of the run-over risk it exists for.
        out['real_level_pct_short'] = pct[('real_s', h_last)]
        out['impl_level']         = f[f'{p}_impl']
        out['real_level']         = f[f'{p}_real_l']
        out['horizons_agree']     = agree
        out['horizon_concord']    = concord
        out['window_agreement']   = win_agree.clip(0, 1)
        out['tenor_reliability']  = reliab
        out['level_gap']          = (f[f'{p}_impl'] - f[f'{p}_real_l']
                                     if risk in LEVEL_COMPARABLE_RISKS else np.nan)
        out['level_gap_z']        = lvl_z
        out['level_support']      = lvl_sup
        # Realized-window lengths, so a gate can size the sampling error of the
        # realized estimate without re-deriving it from a windows dict.
        out['n_short']            = f.attrs['n_short']
        out['n_long']             = n_long
        # rmse_vp / atm live on `f` and were previously dropped here, which left
        # gate_fit_quality reading NaN and passing unconditionally.
        out['rmse_vp']            = f['rmse_vp']
        out['atm']                = f['atm']
        return out

    # ── public entry point ───────────────────────────────────────────────────
    def compute(self, pair: str, tenors: Sequence[str],
                dataset: FXVolDataset, verbose: bool = False) -> dict:
        """
        Returns {'signals': DataFrame[SIGNAL_COLS], 'context': DataFrame}.

        `context` shares the (date, pair, tenor, risk) key and carries the
        diagnostics the gates and reason strings read. It is kept separate so the
        signal frame stays a clean contract surface.
        """
        if isinstance(tenors, (str, int)):
            tenors = [tenors]
        missing = [t for t in tenors if t not in self.windows]
        if missing:
            raise KeyError(f"No realized windows defined for {missing}.")

        imp = calibrate_sabr_history(pair, list(tenors), dataset,
                                     self.delta_pts, self.beta, verbose=verbose)
        if imp.empty:
            return {'signals': pd.DataFrame(columns=SIGNAL_COLS),
                    'context': pd.DataFrame()}
        imp_tenors = set(imp.columns.get_level_values('tenor'))

        spot   = dataset.spot[pair].dropna()
        spot_r = np.log(spot / spot.shift(1))

        blocks = []
        for tenor in tenors:
            if tenor not in imp_tenors:
                warnings.warn(f"No implied history for {pair} {tenor}; skipped.")
                continue
            f = self._tenor_frame(pair, tenor, dataset, imp, spot_r)
            if f is None:
                continue
            for risk in self.spec.risks:
                b = self._score_one(f, risk).dropna(subset=['score'])
                if b.empty:
                    continue
                b = b.reset_index(names='date')
                b['pair']       = pair
                b['tenor']      = tenor
                b['risk']       = risk
                b['name']       = self.spec.name
                b['horizon_bd'] = _tenor_horizon_bd(pair, tenor)
                blocks.append(b)

        if not blocks:
            return {'signals': pd.DataFrame(columns=SIGNAL_COLS),
                    'context': pd.DataFrame()}

        allrows = pd.concat(blocks, ignore_index=True)
        signals = validate_signal_frame(allrows, self.spec)

        key = ['date', 'pair', 'tenor', 'risk']
        ctx_cols = [c for c in allrows.columns
                    if c not in SIGNAL_COLS or c in key]
        context = (allrows[ctx_cols]
                   .sort_values(['pair', 'tenor', 'risk', 'date'])
                   .reset_index(drop=True))

        if verbose:
            print(f"\n  {self.spec.name} -- {pair}: {len(signals)} signal rows, "
                  f"{signals['date'].min().date()} -> {signals['date'].max().date()}")
            for risk in self.spec.risks:
                sub = signals[signals['risk'] == risk]
                print(f"    {risk:>5}: {len(sub)} rows across "
                      f"{sub['tenor'].nunique()} tenor(s)")

        return {'signals': signals, 'context': context}


# ══════════════════════════════════════════════════════════════════════════════
# Tenor aggregation — per (date, pair, risk)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_tenors(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the tenor dimension: reliability-and-confidence-weighted mean score
    per (date, pair, risk), plus DISPERSION across tenors.

    Dispersion is not decoration. It tells you which instrument to trade:
        high |score|, low dispersion  -> genuine regime, trade it outright
        high |score|, high dispersion -> one tenor is the outlier, trade a calendar
        low  |score|, high dispersion -> no directional view, RV only
    """
    if signals.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        w = g['confidence'].to_numpy(dtype=float)
        s = g['score'].to_numpy(dtype=float)
        tot = w.sum()
        return pd.Series({
            'score':       float(np.average(s, weights=w)) if tot > 0 else np.nan,
            'confidence':  float(np.average(w, weights=w)) if tot > 0 else np.nan,
            'dispersion':  float(np.std(s)),
            'n_tenors':    int(len(g)),
            'sign_agree':  float(max((s > 0).mean(), (s < 0).mean())),
            'max_abs_tenor': g.loc[g['score'].abs().idxmax(), 'tenor'],
        })

    out = (signals.groupby(['date', 'pair', 'risk'], sort=True)
                  .apply(_agg, include_groups=False)
                  .reset_index())
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Orthogonality — are volga and vanna actually two signals, or one in two hats?
# ══════════════════════════════════════════════════════════════════════════════

ORTHO_WARN = 0.50


def orthogonality_report(signals: pd.DataFrame,
                         context: Optional[pd.DataFrame] = None,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Volga and vanna can be targeted structurally as independent exposures — but
    that is a statement about EXECUTION. It does not make them independent
    signals, because nu and rho come out of ONE least-squares fit to the SAME
    seven pillars and are therefore jointly determined. A bad or unstable smile
    fit corrupts both together, and short-dated fits tend to pair an extreme rho
    with an inflated nu.

    So measure it rather than assume it:

    impl_corr   corr(nu_impl, rho_impl)        parameter correlation IN THE FIT
    real_corr   corr(nu_real, rho_real)        same for the realized twins
    score_corr  corr(volga_score, vanna_score) what actually matters for sizing

    Read `score_corr`. Above ~0.5 in absolute value you have one signal wearing
    two hats, and doubling up is levering a single bet. The fix is to
    orthogonalise — regress the vanna score on the volga score and trade the
    residual — not to widen the caps.

    Returns a DataFrame indexed by (pair, tenor) with an ('*', 'AGG') row
    carrying the same correlation on the tenor-aggregated scores.
    """
    if signals.empty:
        return pd.DataFrame()

    def _pair_corr(df: pd.DataFrame, value: str, idx) -> pd.Series:
        w = df.pivot_table(index=idx, columns='risk', values=value)
        if not {'volga', 'vanna'}.issubset(w.columns):
            return pd.Series(dtype=float)
        w = w[['volga', 'vanna']].dropna()
        return pd.Series({'corr': w['volga'].corr(w['vanna']), 'n': len(w)})

    rows = {}
    for (pair, tenor), g in signals.groupby(['pair', 'tenor'], sort=True):
        rec = {}
        sc = _pair_corr(g, 'score', 'date')
        rec['score_corr'] = sc.get('corr', np.nan)
        rec['n_obs']      = int(sc.get('n', 0))

        if context is not None and not context.empty:
            c = context[(context['pair'] == pair) & (context['tenor'] == tenor)]
            for col, name in (('impl_level', 'impl_corr'), ('real_level', 'real_corr')):
                if col in c.columns:
                    rec[name] = _pair_corr(c, col, 'date').get('corr', np.nan)
        rows[(pair, tenor)] = rec

    # same measurement on the tenor-aggregated scores — the number you size on
    agg = aggregate_tenors(signals)
    if not agg.empty:
        for pair, g in agg.groupby('pair', sort=True):
            sc = _pair_corr(g, 'score', 'date')
            rows[(pair, 'AGG')] = {'score_corr': sc.get('corr', np.nan),
                                   'n_obs': int(sc.get('n', 0))}

    out = pd.DataFrame.from_dict(rows, orient='index')
    out.index = pd.MultiIndex.from_tuples(out.index, names=['pair', 'tenor'])
    cols = [c for c in ('impl_corr', 'real_corr', 'score_corr', 'n_obs')
            if c in out.columns]
    out = out[cols]

    if verbose:
        print('\n=== orthogonality: volga vs vanna ===')
        print(out.round(3).to_string())
        worst = out['score_corr'].abs().max() if 'score_corr' in out else np.nan
        if pd.notna(worst) and worst > ORTHO_WARN:
            print(f'  WARNING: |score_corr| up to {worst:.2f} > {ORTHO_WARN} - the two '
                  f'scores are largely the same bet. Separable targeting does not '
                  f'fix this; orthogonalise the scores before sizing both.')
        else:
            print(f'  score_corr within +/-{ORTHO_WARN} - treating them as two '
                  f'independent decisions is defensible.')
        print('  (Separable targeting still leaves the two P&Ls correlated in the '
              'left tail - keep a joint risk budget above the per-risk caps.)')

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Vanna quality — is rho trustworthy on this pair at all?
# ══════════════════════════════════════════════════════════════════════════════

def vanna_quality_report(signals: pd.DataFrame,
                         context: pd.DataFrame,
                         entry_band: float = 0.40,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Per-tenor screen on whether the vanna leg carries signal for this pair.

    The score itself needs no per-pair sign handling: score is a monotone
    transform of rho, so a pair whose risk reversal lives above zero (USDCAD,
    ~93% of days) and one whose RR never crosses it (AUDUSD, ~0%) both work
    unmodified. What DOES vary by pair is whether rho is measurable at all.

    Columns, and what disqualifies a tenor:

    frac_rho_small    |implied rho| < 0.10. A near-symmetric smile carries no
                      slope information, so the fit determines rho weakly.
                      RMSE stays LOW here, so gate_fit_quality is blind to it.
                      Above ~0.25 the tenor is mostly noise.

    frac_impl_pos     Read WITH n_crossings. Near 0 or 1 means rho sits on one
                      side and pokes across briefly -- flicker, and the
                      percentile history is homogeneous. In [0.2, 0.8] the pair
                      genuinely lives on both sides and the percentile history
                      mixes two structurally different skew regimes, which the
                      expanding score rank then bakes in permanently.

    frac_level_sig    fraction with |level gap| > LEVEL_SUPPORT_Z standard
                      errors. Low means the levels are unresolvable and the
                      percentile gap is amplifying noise.

    disagree_traded   sign(level gap) != sign(percentile gap), on days the
                      position is live. Do NOT gate on this directly: implied
                      rho carries a skew risk premium, so the level gap is
                      biased negative and a sign test vetoes asymmetrically.

    conviction_ratio  disagree_traded / disagree_all. THE column to rank on.
                      Below 1.0, high-conviction days are the ones where the
                      level gap concurs. At or above 1.0, conviction is
                      uncorrelated or anti-correlated with agreement -- the rank
                      transform is manufacturing it.

    real_share        |corr(gap, pct_real)| / (|corr(gap, pct_impl)| + same).
                      Near 1.0 means the implied side contributes nothing and
                      the 'relative value' signal is realized-correlation
                      reversal in disguise -- a legitimate signal, but not the
                      one you think you are trading, and not one that belongs at
                      the same sizing.
    """
    if signals.empty or context.empty:
        return pd.DataFrame()

    key = ['date', 'pair', 'tenor', 'risk']
    v = (signals[signals['risk'] == 'vanna'][key + ['raw', 'score']]
         .merge(context[context['risk'] == 'vanna'], on=key, how='left'))
    if v.empty or 'level_gap' not in v.columns:
        return pd.DataFrame()

    v['traded']   = v['score'].abs() >= entry_band
    v['disagree'] = np.sign(v['level_gap']) != np.sign(v['raw'])

    rows = {}
    for (pair, tenor), g in v.groupby(['pair', 'tenor'], sort=True):
        t  = g[g['traded']]
        sg = np.sign(g['impl_level'].dropna())
        ci = abs(g['raw'].corr(g[[c for c in g if c.startswith('pct_impl_')][0]]))
        cr = abs(g['raw'].corr(g[[c for c in g if c.startswith('pct_real_')][0]]))
        d_all = g['disagree'].mean()
        rows[(pair, tenor)] = {
            'frac_rho_small':   float((g['impl_level'].abs() < 0.10).mean()),
            'frac_impl_pos':    float((g['impl_level'] > 0).mean()),
            'n_crossings':      int((sg.diff() != 0).sum() - 1) if len(sg) else 0,
            'frac_level_sig':   float((g['level_gap_z'] >= LEVEL_SUPPORT_Z).mean()),
            'med_level_z':      float(g['level_gap_z'].median()),
            'disagree_traded':  float(t['disagree'].mean()) if len(t) else np.nan,
            'conviction_ratio': float(t['disagree'].mean() / d_all)
                                if len(t) and d_all > 0 else np.nan,
            'real_share':       float(cr / (ci + cr)) if (ci + cr) > 0 else np.nan,
        }

    out = pd.DataFrame.from_dict(rows, orient='index')
    out.index = pd.MultiIndex.from_tuples(out.index, names=['pair', 'tenor'])

    if verbose:
        print('\n=== vanna quality (is rho measurable on this pair?) ===')
        print(out.round(3).to_string())
        bad = out[(out['frac_rho_small'] > 0.25) |
                  (out['conviction_ratio'] >= 1.0)]
        for (pair, tenor), r in bad.iterrows():
            why = []
            if r['frac_rho_small'] > 0.25:
                why.append(f"rho inside +/-0.10 on {r['frac_rho_small']:.0%} of days")
            if r['conviction_ratio'] >= 1.0:
                why.append(f"conviction ratio {r['conviction_ratio']:.2f} >= 1.0")
            print(f'  DROP {pair} {tenor}: ' + '; '.join(why))
        mixed = out[out['frac_impl_pos'].between(0.20, 0.80)]
        for (pair, tenor), r in mixed.iterrows():
            print(f'  REGIME {pair} {tenor}: rho positive {r["frac_impl_pos"]:.0%} of '
                  f'days with {r["n_crossings"]} crossings - the percentile '
                  f'history spans two skew regimes')
        if bad.empty and mixed.empty:
            print('  all tenors pass: rho is well identified and single-regime')

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — gates
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str = ''


# A gate sees the score and the context row. It returns pass/fail + a reason.
# Gates are direction-aware, and some are RISK-aware. That distinction matters:
#
#   nu  (volga) is NON-NEGATIVE by construction. Only its lower tail is the
#       "can only revert upward" danger, so a floor-only veto is correct.
#   rho (vanna) is SIGNED and bounded [-1, +1]. BOTH tails are dangerous, and a
#       floor-only veto leaves every long position unguarded.
#
# The original gates were written for volga and applied unchanged to vanna. The
# `signed_risks` parameter is what repairs that: the extra veto arms only for
# risks whose underlying quantity can sit at a dangerous upper extreme too.
Gate = Callable[[float, pd.Series], GateResult]

SIGNED_RISKS = ('vanna',)


def gate_realized_momentum(threshold: float = 20.0,
                           signed_risks: Sequence[str] = SIGNED_RISKS) -> Gate:
    """
    Block a SHORT when realized is accelerating: short-window percentile well
    above long-window. Never sell the thing that is picking up, however rich.

    For a SIGNED risk, mirror it: block a LONG when realized is decelerating
    that hard. Long vanna needs realized correlation to hold up; buying it while
    it collapses is the same mistake in the opposite direction.
    """
    def g(score: float, ctx: pd.Series) -> GateResult:
        m = ctx.get('realized_momentum', np.nan)
        if pd.isna(m):
            return GateResult(True)
        if score < 0 and m > threshold:
            return GateResult(False, f'realized accelerating (+{m:.0f}pct)')
        if score > 0 and m < -threshold and ctx.get('risk') in signed_risks:
            return GateResult(False, f'realized decelerating ({m:.0f}pct)')
        return GateResult(True)
    return g


def gate_absolute_floor(pct_floor: float = 5.0,
                        pct_ceiling: float = 95.0,
                        check_short_window: bool = True,
                        signed_risks: Sequence[str] = SIGNED_RISKS) -> Gate:
    """
    Block a SHORT when the realized level is at a multi-year absolute low.
    Rich-and-calm is not rich-and-stable; this is the run-over trade.

    Two repairs on the original:

    `check_short_window` — the floor is now tested on the SHORT realized window
        as well as the long one. Reading the long window alone lets through the
        sharpest version of exactly this risk: AUDCAD 1M volga sits at 5Y pct 24
        on its 20d window while its 10d window is at pct 1.6, so the veto misses
        a short position taken into record-low realized vol-of-vol.

    `pct_ceiling` — for a SIGNED risk, block a LONG at the opposite extreme. Long
        vanna with realized correlation at a 5Y high needs it to stay pinned
        near its historical maximum, which is the mirror of the floor trade.
    """
    def g(score: float, ctx: pd.Series) -> GateResult:
        lvls = {'long window': ctx.get('real_level_pct', np.nan)}
        if check_short_window:
            lvls['short window'] = ctx.get('real_level_pct_short', np.nan)
        signed = ctx.get('risk') in signed_risks

        for which, lvl in lvls.items():
            if pd.isna(lvl):
                continue
            if score < 0 and lvl < pct_floor:
                return GateResult(False, f'realized {which} at pct {lvl:.0f} floor')
            if score > 0 and signed and lvl > pct_ceiling:
                return GateResult(False, f'realized {which} at pct {lvl:.0f} ceiling')
        return GateResult(True)
    return g


def gate_param_identifiable(min_abs_rho: float = 0.08) -> Gate:
    """
    Block VANNA when implied rho sits near zero.

    A near-symmetric smile carries almost no slope information, so the
    least-squares fit determines rho only weakly — it trades off against alpha
    and nu. Critically, `gate_fit_quality` cannot catch this: a flat smile fits
    beautifully, so RMSE stays low while rho is noise. RMSE measures fit
    quality; this measures identifiability, and the two come apart precisely
    here.

    Two-sided — a poorly determined parameter supports no view either way.

    Only meaningful for vanna. nu is a magnitude and is identified by the
    butterfly regardless of the skew's sign.
    """
    def g(score: float, ctx: pd.Series) -> GateResult:
        if ctx.get('risk') != 'vanna':
            return GateResult(True)
        r = ctx.get('impl_level', np.nan)
        if pd.notna(r) and abs(r) < min_abs_rho:
            return GateResult(False, f'rho {r:+.3f} near zero, weakly identified')
        return GateResult(True)
    return g


def gate_level_significant(k_se: float = 1.5) -> Gate:
    """
    Block VANNA when |implied rho - realized rho| falls inside the sampling
    error of the realized correlation estimate.

    Why this is available for vanna but NOT for volga: SABR rho and realized
    corr(dlnS, dlnATM) are both correlations bounded [-1, +1] — the same object,
    so their difference has meaning. Implied and realized NU are not comparable
    in level (implied nu absorbs smile curvature that stdev(dlnv) cannot
    produce), which is the whole reason the score is a rank-of-rank.

    So for vanna there is a levels check available that the percentile gap
    throws away, and it matters because the rank transform will happily
    manufacture a large gap out of a level difference that is inside the noise:

        AUDCAD 3M   impl -0.208, real_60d -0.354  ->  level gap +0.146
                    se = (1 - 0.354^2)/sqrt(59)   =   0.114   ->  1.3 SE
                    percentile gap -24, score +0.55, position LONG

    Tests MAGNITUDE, not sign. A raw sign-agreement test would inherit the skew
    risk premium's bias — implied rho runs structurally more negative than
    realized because downside protection is persistently bid, so the level gap
    is biased negative and a sign test vetoes asymmetrically, blocking longs and
    waving shorts through. Comparing against the standard error is neutral.
    """
    def g(score: float, ctx: pd.Series) -> GateResult:
        if ctx.get('risk') != 'vanna':
            return GateResult(True)
        impl, real, n = (ctx.get('impl_level', np.nan),
                         ctx.get('real_level', np.nan),
                         ctx.get('n_long', np.nan))
        if any(pd.isna(x) for x in (impl, real, n)):
            return GateResult(True)
        se = (1.0 - float(real) ** 2) / np.sqrt(max(2.0, float(n) - 1.0))
        lg = float(impl) - float(real)
        if abs(lg) < k_se * se:
            return GateResult(False,
                              f'level gap {lg:+.3f} inside {k_se:g}x SE ({se:.3f})')
        return GateResult(True)
    return g


def gate_fit_quality(max_rmse_vp: float = 0.25) -> Gate:
    """Block in BOTH directions on a bad smile fit — the parameters are noise."""
    def g(score: float, ctx: pd.Series) -> GateResult:
        r = ctx.get('rmse_vp', np.nan)
        if pd.notna(r) and r > max_rmse_vp:
            return GateResult(False, f'smile fit rmse {r:.2f}vp')
        return GateResult(True)
    return g


def gate_window_disagreement(min_agreement: float = 0.35) -> Gate:
    """
    Block when the short and long realized windows disagree badly — at that
    point you are reading sampling error, not a level.
    """
    def g(score: float, ctx: pd.Series) -> GateResult:
        a = ctx.get('window_agreement', np.nan)
        if pd.notna(a) and a < min_agreement:
            return GateResult(False, f'realized windows disagree ({a:.2f})')
        return GateResult(True)
    return g


def gate_event_window(event_dates: Sequence, lookahead_bd: int = 3) -> Gate:
    """
    Block a SHORT when a scheduled event lands inside the next `lookahead_bd`
    days. Pass the dates in (RBA/BoC/CPI); no calendar is inferred for you.
    """
    ev = pd.DatetimeIndex(pd.to_datetime(list(event_dates))).sort_values()

    def g(score: float, ctx: pd.Series) -> GateResult:
        if score >= 0 or len(ev) == 0:
            return GateResult(True)
        d  = pd.Timestamp(ctx['date'])
        hi = d + pd.tseries.offsets.BDay(lookahead_bd)
        hit = ev[(ev >= d) & (ev <= hi)]
        if len(hit):
            return GateResult(False, f'event {hit[0].date()} in window')
        return GateResult(True)
    return g


DEFAULT_GATES: Dict[str, Gate] = {
    'momentum':     gate_realized_momentum(),   # two-sided on signed risks
    'level':        gate_absolute_floor(),      # floor + ceiling, both windows
    'fit':          gate_fit_quality(),         # live now that rmse_vp reaches ctx
    'window_agree': gate_window_disagreement(),
    'rho_ident':    gate_param_identifiable(),  # vanna only
}

# `gate_level_significant` is deliberately NOT a default. Level support is
# already priced into `confidence` via LEVEL_SUPPORT_Z, which de-rates size
# instead of forcing flat. As a hard veto it gated ~58% of AUDCAD vanna days on
# its own — it was overriding the signal's thesis rather than qualifying it.
# Opt in only if you want the strict levels-must-resolve version:
#
#   gates = {**DEFAULT_GATES, 'level_sig': gate_level_significant(k_se=1.5)}

# `gate_event_window` is deliberately absent: it needs a calendar of scheduled
# events (RBA/BoC/CPI) that nothing here can infer. Until you supply one, the
# book sells vol into central bank meetings. Wire it up as:
#
#   gates = {**DEFAULT_GATES, 'event': gate_event_window(my_dates)}
#   decide(sig, ctx, config=cfg, gates=gates)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — decide
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DecisionConfig:
    """
    entry_band / exit_band implement hysteresis: enter at |score| >= entry,
    hold until |score| < exit. Without the gap you flip the book on noise around
    zero. exit_band MUST be below entry_band.

    cap : max target exposure. Either a single number (both risks share it), or
          PER-RISK in the Greek's own units, which is what you want once vanna
          and volga are targeted structurally:

              cap = {'volga': <ccy per vol-pt^2>,
                     'vanna': <ccy per vol-pt per spot %>}

          Set the two STRESS-EQUIVALENT, not equal-notional and not
          equal-expected-P&L. Pick a reference joint spot/vol shock and size each
          cap so a full position loses the same amount under it. Equal notional
          is meaningless across two different Greeks; equal expected P&L
          understates the volga leg's tail.

          Note that separable TARGETING does not make the two P&Ls independent —
          both lose in the same stress events — so a joint portfolio risk budget
          still sits above this.
    """
    entry_band: float = 0.40
    exit_band: float = 0.20
    cap: Union[float, Dict[str, float]] = 1.0
    size_by_confidence: bool = True
    min_confidence: float = 0.20     # below this, no position at all

    def __post_init__(self):
        if not 0 <= self.exit_band < self.entry_band <= 1:
            raise ValueError('need 0 <= exit_band < entry_band <= 1.')
        if isinstance(self.cap, dict):
            bad = set(self.cap) - set(RISKS)
            if bad:
                raise ValueError(f'cap has unknown risk key(s) {bad}; expected {RISKS}.')
            neg = {k: v for k, v in self.cap.items() if not v > 0}
            if neg:
                raise ValueError(f'cap must be positive, got {neg}.')
        elif not self.cap > 0:
            raise ValueError(f'cap must be positive, got {self.cap}.')

    def cap_for(self, risk: str) -> float:
        """Cap for one risk. Raises rather than defaulting — a silent fallback to
        1.0 would size a real book in the wrong units."""
        if isinstance(self.cap, dict):
            if risk not in self.cap:
                raise KeyError(
                    f"No cap defined for risk {risk!r}. Provide it in "
                    f"DecisionConfig(cap={{...}}) or pass a single float.")
            return float(self.cap[risk])
        return float(self.cap)


DECISION_COLS = ['date', 'pair', 'tenor', 'risk', 'score', 'confidence',
                 'target', 'cap', 'state', 'gates_failed', 'reason']


def decide(signals: pd.DataFrame,
           context: Optional[pd.DataFrame] = None,
           config: Optional[DecisionConfig] = None,
           gates: Optional[Dict[str, Gate]] = None) -> pd.DataFrame:
    """
    Turn scores into target exposures.

    Order of operations, and it matters: gates are HARD and evaluated first, so a
    gated row is flat regardless of score and cannot be re-entered by hysteresis.
    Then the dead band, then hysteresis, then sizing.

    Returns one row per (date, pair, tenor, risk) with `target` in the Greek's
    OWN units (positive = long, negative = short) and the `cap` used alongside
    it. With per-risk caps the two risks' targets are no longer on a common
    scale, so read `target` against its own `cap` — that is why the cap is
    emitted per row rather than left in the config.

    `reason` is an audit trail rather than a bare number.
    """
    cfg   = config or DecisionConfig()
    gates = DEFAULT_GATES if gates is None else gates
    if signals.empty:
        return pd.DataFrame(columns=DECISION_COLS)

    key = ['date', 'pair', 'tenor', 'risk']
    df  = signals.copy()
    if context is not None and not context.empty:
        df = df.merge(context, on=key, how='left', suffixes=('', '_ctx'))
    df = df.sort_values(['pair', 'tenor', 'risk', 'date'])

    rows = []
    for (pair, tenor, risk), g in df.groupby(['pair', 'tenor', 'risk'], sort=True):
        cap   = cfg.cap_for(risk)   # raises early if a risk has no cap defined
        state = 0          # -1 short, 0 flat, +1 long  (carried across dates)
        for _, r in g.iterrows():
            score = float(r['score'])
            conf  = float(r['confidence']) if pd.notna(r['confidence']) else 0.0

            # 1. gates — hard, and they force flat. Evaluate each ONCE: a gate
            # is arbitrary user code and may be expensive.
            verdicts = {nm: gate(score, r) for nm, gate in gates.items()}
            failed   = [nm for nm, v in verdicts.items() if not v.passed]
            reasons  = [verdicts[nm].reason for nm in failed]
            if failed:
                state = 0
                rows.append((r['date'], pair, tenor, risk, score, conf, 0.0,
                             cap, 'gated', ','.join(failed),
                             f"gated: {'; '.join(x for x in reasons if x)}"))
                continue

            # 2. confidence floor
            if conf < cfg.min_confidence:
                state = 0
                rows.append((r['date'], pair, tenor, risk, score, conf, 0.0,
                             cap, 'flat', '', f'confidence {conf:.2f} below floor'))
                continue

            # 3. dead band + hysteresis
            a = abs(score)
            if state == 0:
                if a >= cfg.entry_band:
                    state = int(np.sign(score))
                    note  = f'enter {"long" if state > 0 else "short"}'
                else:
                    note = f'|score| {a:.2f} inside entry band'
            else:
                if np.sign(score) != state and a >= cfg.entry_band:
                    state = int(np.sign(score))
                    note  = f'flip to {"long" if state > 0 else "short"}'
                elif a < cfg.exit_band:
                    state, note = 0, f'|score| {a:.2f} below exit band'
                else:
                    note = 'hold'

            # 4. size
            if state == 0:
                target = 0.0
            else:
                mag    = (a - cfg.exit_band) / max(1e-9, 1.0 - cfg.exit_band)
                mag    = float(np.clip(mag, 0.0, 1.0))
                target = state * mag * cap
                if cfg.size_by_confidence:
                    target *= conf

            side  = 'long' if state > 0 else ('short' if state < 0 else 'flat')
            gap_s = f"gap {r['raw']:+.0f}" if pd.notna(r.get('raw')) else ''
            reason = (f'{side} {risk} {tenor}: score {score:+.2f}, {gap_s}, '
                      f'conf {conf:.2f} | {note}')
            rows.append((r['date'], pair, tenor, risk, score, conf, target,
                         cap, side, '', reason))

    return pd.DataFrame(rows, columns=DECISION_COLS)


def latest_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per (pair, tenor, risk) — the actionable slice."""
    if decisions.empty:
        return decisions
    return (decisions.sort_values('date')
                     .groupby(['pair', 'tenor', 'risk'], as_index=False)
                     .tail(1)
                     .sort_values(['pair', 'risk', 'tenor'])
                     .reset_index(drop=True))
