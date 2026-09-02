import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset


spread_order = ['1W-2W', '1W-1M', 
                '1M-2M',  '1M-3M', '1M-6M', 
                '3M-6M', '3M-1Y',
                '6M-1Y']



def calculate_term_percentiles(ccys):
    tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '6M', '1Y', '2Y']
    spreads = [
            ('1W', '2W'),
            ('1W', '1M'),
            ('1M', '2M'),
            ('1M', '3M'),
            ('1M', '6M'),
            ('3M', '6M'),
            ('3M', '1Y'),
            ('6M', '1Y')]
    results = {}
    for ccy in ccys:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        df_vols = {}
        for tenor in tenors:
            ticker_IV = f"{ccy}V{tenor} Curncy"
            data_IV = blp.bdh(
                tickers=ticker_IV,
                flds="PX_LAST",
                start_date=start_date,
                end_date=end_date)
            if not data_IV.empty:
                data_IV.columns = [tenor]
                df_vols[tenor] = data_IV
            else:
                print(f"No data for {ticker_IV}, skipping.")
        df_vols_all = pd.concat(df_vols.values(), axis=1)
        df_spreads = pd.DataFrame(index=df_vols_all.index)
        for spread in spreads:
            first_tenor, second_tenor = spread
            spread_name = f"{first_tenor}-{second_tenor}"
            df_spreads[spread_name] = df_vols_all[first_tenor] - df_vols_all[second_tenor]
        three_months_ago = (df_spreads.index[-1] - DateOffset(months=3)).date()
        one_year_ago = (df_spreads.index[-1] - DateOffset(years=1)).date()
        three_year_ago = (df_spreads.index[-1] - DateOffset(years=3)).date()
        df_spreads_3m = df_spreads[df_spreads.index >= three_months_ago]
        df_spreads_1Y = df_spreads[df_spreads.index >= one_year_ago]
        df_spreads_3Y = df_spreads[df_spreads.index >= three_year_ago]
        current_spread = df_spreads.iloc[-1]
        spread_data = {}
        for column in df_spreads.columns:
            actual_value = current_spread[column]
            percentiles_3M = np.sum(df_spreads_3m[column] < actual_value) / df_spreads_3m[column].count() * 100
            percentiles_1Y = np.sum(df_spreads_1Y[column] < actual_value) / df_spreads_1Y[column].count() * 100
            percentiles_3Y = np.sum(df_spreads_3Y[column] < actual_value) / df_spreads_3Y[column].count() * 100
            percentiles_5Y = np.sum(df_spreads[column] < actual_value) / df_spreads[column].count() * 100

            spread_data[column] = {
                "Current Spread": round(actual_value, 4),
                "3M Percentile": round(percentiles_3M, 3),
                "1Y Percentile": round(percentiles_1Y, 3),
                "3Y Percentile": round(percentiles_3Y, 3),
                "5Y Percentile": round(percentiles_5Y, 3)}
        results[ccy] = pd.DataFrame.from_dict(spread_data, orient="index")
    combined_df = []
    for ccy, df_summary in results.items():
        df_summary['CCY'] = ccy
        combined_df.append(df_summary)
    final_df = pd.concat(combined_df, axis=0)
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Spread'}, inplace=True)
    df = final_df[['Spread', 'CCY', 'Current Spread', '3M Percentile', '1Y Percentile', '3Y Percentile', '5Y Percentile']]
    df.sort_values(by='5Y Percentile', ascending=False, inplace=True)
    return df










ccys = [ 
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD',
    'USDMXN', 'USDBRL']

df = calculate_term_percentiles(ccys)

spread_order = ['1W-2W', '1W-1M', 
                '1M-2M',  '1M-3M', '1M-6M', 
                '3M-6M', '3M-1Y',
                '6M-1Y']


# spread_order = ['1W-1M']




def get_spread_color(spread_value, percentile_value):
    if spread_value < 0: 
        return f'rgba(255, 0, 0, {0.6 + abs(spread_value) * 0.1})'
    else: 
        if percentile_value >= 80:
            return f'rgba(0, 150, 0, {0.3 + (percentile_value-75)/100})' 
        elif percentile_value >= 50:
            return f'rgba(255, 255, 0, {0.3 + (percentile_value-50)/100})' 
        elif percentile_value >= 20:
            return f'rgba(255, 165, 0, {0.3 + (percentile_value-25)/100})'  
        else:
            return f'rgba(200, 200, 200, 0.4)'  
spreads_list = []
for spread in spread_order:
    if spread in df['Spread'].unique():
        df_temp = df[df['Spread'] == spread]
        if (df_temp['Current Spread'] > 0).any():  
            spreads_list.append(spread)
n_spreads = len(spreads_list)
cell_height = 30
header_height = 35
filtered_spreads = {}
for spread in spreads_list:
    df_spread = df[df['Spread'] == spread].copy()
    df_spread = df_spread.sort_values('1Y Percentile', ascending=False)
    mask = (
        (df_spread['Current Spread'] > 0) | 
        (df_spread['1Y Percentile'] > 90) | 
        (df_spread['1Y Percentile'] < 10) )
    df_spread = df_spread[mask]
    if len(df_spread) > 0: 
        filtered_spreads[spread] = df_spread
spreads_list = list(filtered_spreads.keys())
n_spreads = len(spreads_list)
total_height = sum([header_height + (cell_height * len(filtered_spreads[spread])) + 200  # Increased from 100 to 200
                    for spread in spreads_list])
fig = make_subplots(
    rows=n_spreads, 
    cols=1,
    subplot_titles=[f'<b>{spread} Spread</b>' for spread in spreads_list],
    vertical_spacing=0.03,  # Increased from 0.01 to 0.03 for better spacing
    specs=[[{"type": "table"}] for _ in range(n_spreads)])
for idx, spread in enumerate(spreads_list, 1):
    df_spread = filtered_spreads[spread]
    colors_current = [get_spread_color(spread_val, 50) 
                     for spread_val in df_spread['Current Spread']]
    colors_3m = [get_spread_color(df_spread.iloc[i]['Current Spread'], 
                                   df_spread.iloc[i]['3M Percentile']) 
                for i in range(len(df_spread))]
    colors_1y = [get_spread_color(df_spread.iloc[i]['Current Spread'], 
                                   df_spread.iloc[i]['1Y Percentile']) 
                for i in range(len(df_spread))]
    colors_3y = [get_spread_color(df_spread.iloc[i]['Current Spread'], 
                                   df_spread.iloc[i]['3Y Percentile']) 
                for i in range(len(df_spread))]
    colors_5y = [get_spread_color(df_spread.iloc[i]['Current Spread'], 
                                   df_spread.iloc[i]['5Y Percentile']) 
                for i in range(len(df_spread))]
    fill_colors = [
        ['white'] * len(df_spread),     # CCY
        ['darkgrey'] * len(df_spread),    # Current Spread
        ['lightgrey'] * len(df_spread), # 3M Percentile
        colors_1y,                       # 1Y Percentile (highlighted)
        ['lightgrey'] * len(df_spread), # 3Y Percentile
        ['lightgrey'] * len(df_spread)]
    bold_1y_values = [f'<b>{val:.2f}</b>' for val in df_spread['1Y Percentile']]
    current_spread_values = []
    for val in df_spread['Current Spread']:
        if val < 0:
            current_spread_values.append(f'<b style="color:darkred">{val:.4f}</b>')
        else:
            current_spread_values.append(f'{val:.4f}')
    vertical_line_widths = [0, 0, 0, 0, 0, 0]
    table = go.Table(
        header=dict(
            values=['<b>Currency</b>', '<b>Current Spread</b>', 
                   '<b>3M %</b>', '<b>1Y %</b>', '<b>3Y %</b>', '<b>5Y %</b>'],
            fill_color='rgb(100, 150, 200)',
            align='center',
            font=dict(size=13, color='white', family='Arial Black'),
            height=header_height,
            line=dict(color='black', width=vertical_line_widths)),
        cells=dict(
            values=[
                df_spread['CCY'],
                current_spread_values,  # Inversions in bold red
                df_spread['3M Percentile'].round(2),
                bold_1y_values,  # Bold 1Y Percentile
                df_spread['3Y Percentile'].round(2),
                df_spread['5Y Percentile'].round(2)],
            fill_color=fill_colors,
            align='center',
            font=dict(size=12, color='black', family='Arial'),
            height=cell_height,
            line=dict(color='black', width=vertical_line_widths)))
    fig.add_trace(table, row=idx, col=1)
fig.update_layout(
    title={
        'text': '<b>Term Structure Spread Percentiles </b>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': 'darkblue'}},
    height=total_height,
    width=1400,
    showlegend=False,
    paper_bgcolor='rgb(250, 250, 250)')
fig.show()


