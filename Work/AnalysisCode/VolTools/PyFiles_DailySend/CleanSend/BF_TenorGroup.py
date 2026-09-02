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


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots











# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------- BASE VOL ADJUSTED RR ----------------------------------------------------------
# Gets ATM Vols, 25D RR Vols, and base adjusts RR (= 25DRR(Tenor) / ATM Vol(Tenor))
def BaseVolAdjBFDataDaily(ccy, tenor, timeHist, delta):
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
    data_IV = data_IV.tail(day)
    ticker_BF = f"{ccy}{delta}B{tenor} BGN Curncy" # 
    field = "PX_LAST"
    data_BF = blp.bdh(
        tickers=ticker_BF,
        flds=field,
        start_date=start_date,
        end_date=end_date)
    combined_data = pd.concat([data_IV, data_BF], axis=1)
    combined_data.columns = [f'V{tenor}', f'{delta}b{tenor}']
    combined_data[f'{tenor}_VolAdjBF'] = combined_data[f'{delta}b{tenor}'] / combined_data[f'V{tenor}']
    return combined_data

def BaseVolAdjBFEval(ccy, tenors, timeHist, delta):
    df_allVolAdjBFs = pd.DataFrame()
    for tenor in tenors:
        df_tenor = BaseVolAdjBFDataDaily(ccy, tenor, timeHist, delta)
        if df_allVolAdjBFs.empty:
            df_allVolAdjBFs = df_tenor
        else:
            df_allVolAdjBFs = df_allVolAdjBFs.join(df_tenor, how='outer')
    tenor_data = []
    for tenor in tenors:
        tenor_row = {
            'Tenor': tenor,
            f'{delta}D BF': df_allVolAdjBFs[f'{delta}b{tenor}'].iloc[-1],  # Latest 25D RBF
            'ATM Vol': df_allVolAdjBFs[f'V{tenor}'].iloc[-1],   # Latest ATM Vol
            'VolAdjBF': df_allVolAdjBFs[f'{tenor}_VolAdjBF'].iloc[-1]}
        tenor_data.append(tenor_row)
    df_reshaped = pd.DataFrame(tenor_data)
    df_allVolAdjBFs.index = pd.to_datetime(df_allVolAdjBFs.index)
    three_months_ago = pd.Timestamp(df_allVolAdjBFs.index[-1]) - pd.DateOffset(months=3)
    one_year_ago = pd.Timestamp(df_allVolAdjBFs.index[-1]) - pd.DateOffset(years=1)
    three_year_ago = pd.Timestamp(df_allVolAdjBFs.index[-1]) - pd.DateOffset(years=3)
    five_year_ago = pd.Timestamp(df_allVolAdjBFs.index[-1]) - pd.DateOffset(years=5)
    percentile_data = []
    for tenor in df_reshaped['Tenor']:
        vol_adj_bf_col = f'{tenor}_VolAdjBF'
        past_3m_data = df_allVolAdjBFs.loc[df_allVolAdjBFs.index >= three_months_ago, vol_adj_bf_col]
        past_1y_data = df_allVolAdjBFs.loc[df_allVolAdjBFs.index >= one_year_ago, vol_adj_bf_col]
        past_3y_data = df_allVolAdjBFs.loc[df_allVolAdjBFs.index >= three_year_ago, vol_adj_bf_col]
        past_5y_data = df_allVolAdjBFs.loc[df_allVolAdjBFs.index >= five_year_ago, vol_adj_bf_col]
        current_value = df_reshaped.loc[df_reshaped['Tenor'] == tenor, 'VolAdjBF'].values[0]
        three_months_percentile = np.sum(past_3m_data < current_value) / len(past_3m_data) * 100
        one_year_percentile = np.sum(past_1y_data < current_value) / len(past_1y_data) * 100
        three_year_percentile = np.sum(past_3y_data < current_value) / len(past_3y_data) * 100
        five_year_percentile = np.sum(past_5y_data < current_value) / len(past_5y_data) * 100
        percentile_data.append({
            'Tenor': tenor,
            '3M%_BF': round(three_months_percentile, 2),
            '1Y%_BF': round(one_year_percentile, 2),
            '3Y%_BF': round(three_year_percentile, 2),
            '5Y%_BF': round(five_year_percentile, 2)})
    df_percentiles = pd.DataFrame(percentile_data)
    df_reshaped = pd.merge(df_reshaped, df_percentiles, on='Tenor')
    return df_reshaped

def multipleCCYAdjBFSorted(currency_pairs, tenors, timeHist, delta):
    combined_df = pd.DataFrame()
    for ccy in currency_pairs:
        df_ccy = BaseVolAdjBFEval(ccy, tenors, timeHist, delta)
        df_ccy['Currency Pair'] = ccy
        combined_df = pd.concat([combined_df, df_ccy], ignore_index=False)
    combined_df.reset_index(inplace=True)
    combined_df = combined_df[['Currency Pair', 'Tenor', f'{delta}D BF', 
                                    'ATM Vol', 'VolAdjBF', 
                                    '3M%_BF', '1Y%_BF',
                                    '3Y%_BF', '5Y%_BF']]
    combined_df.sort_values(by='5Y%_BF', ascending=False, inplace=True)
    return combined_df




# -----------------------------------------------------------------------------------------------------

def output_BF_Tenor(currency_pairs, tenors, timeHist, delta):
    df_AdjBFOrdered = multipleCCYAdjBFSorted(currency_pairs, tenors, timeHist, delta)
    def get_color_scale(value):
        """Return color based on percentile: red (low) to yellow (mid) to green (high)"""
        if value >= 80:
            return f'rgba(0, 150, 0, {0.3 + (value-75)/100})'  # Green
        elif value >= 50:
            return f'rgba(255, 255, 0, {0.3 + (value-50)/100})'  # Yellow
        elif value >= 20:
            return f'rgba(255, 165, 0, {0.3 + (value-25)/100})'  # Orange
        else:
            return f'rgba(255, 0, 0, {0.3 + value/100})'  # Red
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
        df_tenor = df_AdjBFOrdered[df_AdjBFOrdered['Tenor'] == tenor].copy()
        df_tenor = df_tenor.sort_values('1Y%_BF', ascending=False)
        colors_1y = [get_color_scale(val) for val in df_tenor['1Y%_BF']]
        fill_colors = [
            ['white'] * len(df_tenor),  # Currency Pair
            ['darkgrey'] * len(df_tenor),  # VolAdjBF
            ['lightgrey'] * len(df_tenor),  # 3M%
            colors_1y,  # 1Y% - ONLY COLORED COLUMN
            ['lightgrey'] * len(df_tenor),  # 3Y%
            ['lightgrey'] * len(df_tenor)]   # 5Y%
        bold_1y_values = [f'<b>{val:.2f}</b>' for val in df_tenor['1Y%_BF']]
        vertical_line_widths = [0, 0, 0, 0, 0, 0, 0]
        table = go.Table(
            header=dict(
                values=['<b>Currency Pair</b>', '<b>VolAdjBF</b>', 
                    '<b>3M %</b>', '<b>1Y %</b>', '<b>3Y %</b>', '<b>5Y %</b>'],
                fill_color='rgb(100, 150, 200)',
                align='center',
                font=dict(size=13, color='white', family='Arial Black'),
                height=header_height,
                line=dict(
                    color='black',
                    width=vertical_line_widths)),
            cells=dict(
                values=[
                    df_tenor['Currency Pair'],
                    (df_tenor['VolAdjBF'] * 100).round(2),
                    df_tenor['3M%_BF'].round(2),
                    bold_1y_values,
                    df_tenor['3Y%_BF'].round(2),
                    df_tenor['5Y%_BF'].round(2)],
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
            'text': '<b>Vol Adjusted 10D Butterfly by Tenor</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'darkblue'}},
        height=total_height,
        width=1500,
        showlegend=False,
        paper_bgcolor='rgb(250, 250, 250)')
    fig.show()







currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
                    'USDMXN', 'USDBRL', 'USDCNH']

# currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCHF']

tenors = ['1W', '2W','3W', '1M', '3M', '6M', '1Y']


# currency_pairs = ['USDJPY']

# tenors = ['1W', '2W', '3W']


timeHist = 365 * 5
delta = '25'


output_BF_Tenor(currency_pairs, tenors, timeHist, delta)