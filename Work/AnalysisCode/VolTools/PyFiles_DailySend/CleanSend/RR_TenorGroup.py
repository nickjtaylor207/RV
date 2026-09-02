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

import plotly.graph_objects as go
from plotly.subplots import make_subplots





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
    sigma = series.std(ddof=0)       
    return np.nan if sigma == 0 else (x - mu) / sigma

def BaseVolAdjRREval(ccy, tenors, timeHist, delta):
    df_all = pd.DataFrame()
    for tenor in tenors:
        df_tenor = BaseVolAdjRRDataDaily(ccy, tenor, timeHist, delta)
        df_all   = df_tenor if df_all.empty else df_all.join(df_tenor, how="outer")
    df_all.index = pd.to_datetime(df_all.index)
    latest_rows = []
    for tenor in tenors:
        latest_rows.append({
            "Tenor"            : tenor,
            f"{delta}D RR"     : df_all[f"{delta}r{tenor}"].iloc[-1],
            "ATM Vol"          : df_all[f"V{tenor}"].iloc[-1],
            "VolAdjRR"         : df_all[f"{tenor}_VolAdjRR"].iloc[-1],})
    df_latest = pd.DataFrame(latest_rows)
    end   = df_all.index[-1]
    cut_3m  = end - pd.DateOffset(months=3)
    cut_1y  = end - pd.DateOffset(years=1)
    cut_5y  = end - pd.DateOffset(years=5)
    cut_10y = end - pd.DateOffset(years=10)
    score_rows = []
    for tenor in df_latest["Tenor"]:
        col = f"{tenor}_VolAdjRR"
        current = df_latest.loc[df_latest["Tenor"] == tenor, "VolAdjRR"].values[0]
        score_rows.append({
            "Tenor"          : tenor,
            "3M Z-Score"     : _z_score(df_all.loc[df_all.index >= cut_3m,  col], current),
            "1Y Z-Score"     : _z_score(df_all.loc[df_all.index >= cut_1y,  col], current),
            "5Y Z-Score"     : _z_score(df_all.loc[df_all.index >= cut_5y,  col], current),
            "10Y Z-Score"    : _z_score(df_all.loc[df_all.index >= cut_10y, col], current),})
    df_scores = pd.DataFrame(score_rows)
    out = (
        pd.merge(df_latest, df_scores, on="Tenor")
          .rename(columns={
              f"{delta}D RR"        : "25D Risk Reversal",
              "ATM Vol"             : "At-The-Money Volatility",
              "VolAdjRR"            : "Volatility Adjusted RR",})
          .round({"3M Z-Score": 2, "1Y Z-Score": 2,
                  "5Y Z-Score": 2, "10Y Z-Score": 2})
          .set_index("Tenor"))
    return out

def multipleCcyAdjRRSorted(currency_pairs, tenors, timeHist, delta):
    frames = []
    for ccy in currency_pairs:
        df_ccy = BaseVolAdjRREval(ccy, tenors, timeHist, delta)
        df_ccy["Currency Pair"] = ccy              # tag the pair
        frames.append(df_ccy)
    combined = (
        pd.concat(frames)           # index = Tenor -> row index
          .reset_index()            # Tenor back to a column
          .loc[:, ["Currency Pair", "Tenor",
                   "25D Risk Reversal", 
                   "At-The-Money Volatility",
                   "Volatility Adjusted RR",
                   "3M Z-Score", "1Y Z-Score",
                   "5Y Z-Score", "10Y Z-Score"]])
    combined = (
        combined.assign(abs10y=lambda d: d["1Y Z-Score"].abs())
                .sort_values("abs10y", ascending=False)
                .drop(columns="abs10y")
                .reset_index(drop=True))
    combined.columns = [
        "Pair", "Tenor", "25D RR", "ATM Vol", "Adj RR",
        "Z-3M", "Z-1Y", "Z-5Y", "Z-10Y"]
    return combined


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

def output_RR_Tenor(currency_pairs, tenors, timeHist, delta):
    df = multipleCcyAdjRRSorted(currency_pairs, tenors, timeHist, delta)
    def get_zscore_color(value):
        """Return color based on z-score: red (extreme negative) to yellow (neutral) to green (extreme positive)"""
        abs_val = abs(value)
        if abs_val >= 2.0:  
            if value > 0:
                return f'rgba(0, 150, 0, {0.5 + min(abs_val/4, 0.5)})'  
            else:
                return f'rgba(255, 0, 0, {0.5 + min(abs_val/4, 0.5)})'  
        elif abs_val >= 1.0:  
            if value > 0:
                return f'rgba(144, 238, 144, 0.6)'  
            else:
                return f'rgba(255, 165, 0, 0.6)'  
        else:  
            return f'rgba(255, 255, 0, 0.4)' 
    tenors_list = tenors
    n_tenors = len(tenors_list)
    n_currencies = len(currency_pairs)
    cell_height = 35
    header_height = 35
    table_height = header_height + (cell_height * n_currencies)
    total_height = (table_height + 100) * n_tenors
    fig = make_subplots(
        rows=n_tenors, 
        cols=1,
        subplot_titles=[f'<b>{tenor} Tenor</b>' for tenor in tenors_list],
        vertical_spacing=0.01,
        specs=[[{"type": "table"}] for _ in range(n_tenors)])
    for idx, tenor in enumerate(tenors_list, 1):
        df_tenor = df[df['Tenor'] == tenor].copy()
        df_tenor = df_tenor.assign(abs_z1y=lambda d: d['Z-1Y'].abs())
        df_tenor = df_tenor.sort_values('abs_z1y', ascending=False).drop(columns='abs_z1y')
        colors_z1y = [get_zscore_color(val) for val in df_tenor['Z-1Y']]
        fill_colors = [
            ['white'] * len(df_tenor),      
            ['darkgrey'] * len(df_tenor),   
            ['lightgrey'] * len(df_tenor),                
            colors_z1y,                      
            ['lightgrey'] * len(df_tenor),               
            ['lightgrey'] * len(df_tenor)]
        bold_z1y_values = [f'<b>{val:.2f}</b>' for val in df_tenor['Z-1Y']]
        vertical_line_widths = [0, 0, 0, 0, 0, 0]
        table = go.Table(
            header=dict(
                values=['<b>Currency Pair</b>', '<b>Adj RR</b>', 
                    '<b>Z-3M</b>', '<b>Z-1Y</b>', '<b>Z-5Y</b>', '<b>Z-10Y</b>'],
                fill_color='rgb(100, 150, 200)',
                align='center',
                font=dict(size=13, color='white', family='Arial Black'),
                height=header_height,
                line=dict(
                    color='black',
                    width=vertical_line_widths)),
            cells=dict(
                values=[
                    df_tenor['Pair'],
                    (df_tenor['Adj RR'] * 100).round(2),  # Convert to percentage
                    df_tenor['Z-3M'].round(2),
                    bold_z1y_values,  # Bold 1Y Z-Score
                    df_tenor['Z-5Y'].round(2),
                    df_tenor['Z-10Y'].round(2)],
                fill_color=fill_colors,
                align='center',
                font=dict(size=12, color='black', family='Arial'),
                height=cell_height,
                line=dict(
                    color='black',
                    width=vertical_line_widths)))
        fig.add_trace(table, row=idx, col=1)
    fig.update_layout(
        title={
            'text': '<b>Vol Adjusted 25D Risky - Tenor Grouped</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'darkblue'}},
        height=total_height,
        width=1400,
        showlegend=False,
        paper_bgcolor='rgb(250, 250, 250)')
    fig.show()



# currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
#                     'USDMXN', 'USDBRL', 'USDCNH']

currency_pairs = ['EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD', 'EURJPY']


tenors = ['1W', '2W', '1M', '3M', '6M', '1Y']
# tenors = ['1M', '2M']
timeHist = 365 * 10
delta = '25'

output_RR_Tenor(currency_pairs, tenors, timeHist, delta)


# Add timeseries plot of zscore/percentile for most standout RR and BF