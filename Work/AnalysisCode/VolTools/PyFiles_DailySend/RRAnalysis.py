import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from pandas.plotting import table
from PIL import Image

from xbbg import blp
from datetime import datetime, timedelta

from scipy.stats import percentileofscore








# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------- BASE VOL ADJUSTED RR ----------------------------------------------------------
# Gets ATM Vols, 25D RR Vols, and base adjusts RR (= 25DRR(Tenor) / ATM Vol(Tenor))
def BaseVolAdjRRDataDaily(ccy, tenor, timeHist, delta):
    day = timeHist
    start_date = (datetime.today() - timedelta(days=(day))).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    ticker_IV = f"{ccy}V{tenor} BGN Curncy"
    field = "PX_LAST"
    data_IV = blp.bdh(
        tickers=ticker_IV,
        flds=field,
        start_date=start_date,
        end_date=end_date)
    ticker_RR = f"{ccy}{delta}R{tenor} BGN Curncy" # !!!!!!!!!!! ADJUST CCY for match
    field = "PX_LAST"
    data_RR = blp.bdh(
        tickers=ticker_RR,
        flds=field,
        start_date=start_date,
        end_date=end_date)
    combined_data = pd.concat([data_IV, data_RR], axis=1)
    combined_data.columns = [f'V{tenor}', f'{delta}r{tenor}']
    combined_data[f'{tenor}_VolAdjRR'] = combined_data[f'{delta}r{tenor}'] / combined_data[f'V{tenor}']
    return combined_data



def _z_score(series, x):
    mu    = series.mean()
    sigma = series.std(ddof=0)           # population st.dev.
    return np.nan if sigma == 0 else (x - mu) / sigma


def BaseVolAdjRREval(ccy, tenors, timeHist, delta):
    # ── 1. Pull historical data for all tenors ────────────────────────
    df_all = pd.DataFrame()
    for tenor in tenors:
        df_tenor = BaseVolAdjRRDataDaily(ccy, tenor, timeHist, delta)
        df_all   = df_tenor if df_all.empty else df_all.join(df_tenor, how="outer")

    df_all.index = pd.to_datetime(df_all.index)

    # ── 2. Latest values we want to display in the output table ───────
    latest_rows = []
    for tenor in tenors:
        latest_rows.append({
            "Tenor"            : tenor,
            f"{delta}D RR"     : df_all[f"{delta}r{tenor}"].iloc[-1],
            "ATM Vol"          : df_all[f"V{tenor}"].iloc[-1],
            "VolAdjRR"         : df_all[f"{tenor}_VolAdjRR"].iloc[-1],
        })
    df_latest = pd.DataFrame(latest_rows)

    # ── 3. Window cut-offs ────────────────────────────────────────────
    end   = df_all.index[-1]
    cut_3m  = end - pd.DateOffset(months=3)
    cut_1y  = end - pd.DateOffset(years=1)
    cut_5y  = end - pd.DateOffset(years=5)
    cut_10y = end - pd.DateOffset(years=10)

    # ── 4. Compute z-scores for each tenor and window ─────────────────
    score_rows = []
    for tenor in df_latest["Tenor"]:
        col = f"{tenor}_VolAdjRR"
        current = df_latest.loc[df_latest["Tenor"] == tenor, "VolAdjRR"].values[0]

        score_rows.append({
            "Tenor"          : tenor,
            "3M Z-Score"     : _z_score(df_all.loc[df_all.index >= cut_3m,  col], current),
            "1Y Z-Score"     : _z_score(df_all.loc[df_all.index >= cut_1y,  col], current),
            "5Y Z-Score"     : _z_score(df_all.loc[df_all.index >= cut_5y,  col], current),
            "10Y Z-Score"    : _z_score(df_all.loc[df_all.index >= cut_10y, col], current),
        })

    df_scores = pd.DataFrame(score_rows)

    # ── 5. Merge, tidy column names, rounding, and set index ──────────
    out = (
        pd.merge(df_latest, df_scores, on="Tenor")
          .rename(columns={
              f"{delta}D RR"        : "25D Risk Reversal",
              "ATM Vol"             : "At-The-Money Volatility",
              "VolAdjRR"            : "Volatility Adjusted RR",
          })
          .round({"3M Z-Score": 2, "1Y Z-Score": 2,
                  "5Y Z-Score": 2, "10Y Z-Score": 2})
          .set_index("Tenor")
    )


    return out



def multipleCcyAdjRRSorted(currency_pairs, tenors, timeHist, delta):
    """
    Build a cross-sectional table of Vol-Adj RR diagnostics for many CCYs
    and rank by the absolute 10-year Z-score (largest deviation first).
    """
    frames = []

    for ccy in currency_pairs:
        # New BaseVolAdjRREval returns Z-scores (not percentiles)
        df_ccy = BaseVolAdjRREval(ccy, tenors, timeHist, delta)
        df_ccy["Currency Pair"] = ccy              # tag the pair
        frames.append(df_ccy)

    # ------------------------------------------------------------------
    combined = (
        pd.concat(frames)           # index = Tenor -> row index
          .reset_index()            # Tenor back to a column
          .loc[:, ["Currency Pair", "Tenor",
                   "25D Risk Reversal", 
                   "At-The-Money Volatility",
                   "Volatility Adjusted RR",
                   "3M Z-Score", "1Y Z-Score",
                   "5Y Z-Score", "10Y Z-Score"]]
    )

    # sort by |10-year z| (use abs() in case the values are signed)
    combined = (
        combined.assign(abs10y=lambda d: d["1Y Z-Score"].abs())
                .sort_values("abs10y", ascending=False)
                .drop(columns="abs10y")
                .reset_index(drop=True)
    )

    combined.columns = [
        "Pair", "Tenor", "25D RR", "ATM Vol", "Adj RR",
        "Z-3M", "Z-1Y", "Z-5Y", "Z-10Y"
    ]

    return combined






# ---------------------------------------------------------------------------------------------------------------------------------------------------------



# currency_pairs = [ 
#     'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD',
#     'USDMXN', 'USDBRL', 'USDCNH', 'EURSEK', 'EURNOK', 'EURGBP', 'EURCHF'
# ]

# currency_pairs = [ 
#     'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD',
#     'USDMXN', 'USDBRL', 'USDCNH'
# ]

currency_pairs = ['USDJPY']

tenors = ['1W', '2W', '1M', '2M']
timeHist = 365 * 10
delta = '25'

df = multipleCcyAdjRRSorted(currency_pairs, tenors, timeHist, delta)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)


print(df)







