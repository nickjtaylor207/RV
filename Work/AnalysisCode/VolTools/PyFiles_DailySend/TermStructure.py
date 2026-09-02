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





def calculate_term_percentiles(ccys, normalize=True):
    tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '6M', '1Y', '2Y']
    spreads = [
        ('1M', '3M'),
        ('3M', '6M'),
        ('3M', '1Y')]
    results = {}
    
    for ccy in ccys:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        df_vols = {}
        
        for tenor in tenors:
            ticker_IV = f"{ccy}V{tenor} BGN Curncy"
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
            
            if normalize:
                # Normalized spread: (short - long) / long
                # Expressed as percentage of the longer tenor vol
                df_spreads[spread_name] = (
                    (df_vols_all[first_tenor] - df_vols_all[second_tenor]) 
                    / df_vols_all[second_tenor]
                ) * 100  # multiply by 100 to express as percentage
            else:
                # Raw spread (original approach)
                df_spreads[spread_name] = df_vols_all[first_tenor] - df_vols_all[second_tenor]
        
        # Rest of your code stays the same...
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
                'Spread/Long Tenor (%)': round(actual_value, 2),
                "3M Percentile": round(percentiles_3M, 2),
                "1Y Percentile": round(percentiles_1Y, 2),
                "3Y Percentile": round(percentiles_3Y, 2),
                "5Y Percentile": round(percentiles_5Y, 2)}
        results[ccy] = pd.DataFrame.from_dict(spread_data, orient="index")
    # Combine results...
    combined_df = []
    for ccy, df_summary in results.items():
        df_summary['CCY'] = ccy
        combined_df.append(df_summary)
    final_df = pd.concat(combined_df, axis=0)
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Spread'}, inplace=True)
    df = final_df[['Spread', 'CCY', 'Spread/Long Tenor (%)', '3M Percentile', '1Y Percentile', '3Y Percentile', '5Y Percentile']]
    # Group by CCY (in original order) and sort by 1Y Percentile within each group
    df['CCY'] = pd.Categorical(df['CCY'], categories=ccys, ordered=True)
    df = df.sort_values(by=['CCY', '1Y Percentile'], ascending=[True, False])
    df.reset_index(drop=True, inplace=True)
    
    return df








# ccys = [
#     'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
#     'USDMXN', 'USDBRL']


ccys = ['NZDUSD']



df = calculate_term_percentiles(ccys)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

print(df)




