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





# --------------------------------------------------------------------------------






# Calculate individual Vol Adjusted Butterfly Implied Vol 
def BaseVolAdjBFEval(ccy, tenors, timeHist, delta):


    df_allVolAdjBFs = pd.DataFrame()

    # Loop through each tenor and calculate the data
    for tenor in tenors:
        df_tenor = BaseVolAdjBFDataDaily(ccy, tenor, timeHist, delta)
        if df_allVolAdjBFs.empty:
            df_allVolAdjBFs = df_tenor
        else:
            df_allVolAdjBFs = df_allVolAdjBFs.join(df_tenor, how='outer')



    tenor_data = []

    for tenor in tenors:
        # Extract columns for the current tenor
        tenor_row = {
            'Tenor': tenor,
            f'{delta}D BF': df_allVolAdjBFs[f'{delta}b{tenor}'].iloc[-1],  # Latest 25D RBF
            'ATM Vol': df_allVolAdjBFs[f'V{tenor}'].iloc[-1],   # Latest ATM Vol
            'VolAdjBF': df_allVolAdjBFs[f'{tenor}_VolAdjBF'].iloc[-1]  # Latest VolAdjBF
        }
        tenor_data.append(tenor_row)


    df_reshaped = pd.DataFrame(tenor_data)

    df_allVolAdjBFs.index = pd.to_datetime(df_allVolAdjBFs.index)

    three_months_ago = pd.Timestamp(df_allVolAdjBFs.index[-1]) - pd.DateOffset(months=3)
    one_year_ago = pd.Timestamp(df_allVolAdjBFs.index[-1]) - pd.DateOffset(years=1)
    three_year_ago = pd.Timestamp(df_allVolAdjBFs.index[-1]) - pd.DateOffset(years=3)
    five_year_ago = pd.Timestamp(df_allVolAdjBFs.index[-1]) - pd.DateOffset(years=5)



    # ----------------------------------------------------------------------------------


    percentile_data = []

    for tenor in df_reshaped['Tenor']:
        # Get the relevant column for the tenor
        vol_adj_bf_col = f'{tenor}_VolAdjBF'

        # Filter data for the past 3 months and 1 year
        past_3m_data = df_allVolAdjBFs.loc[df_allVolAdjBFs.index >= three_months_ago, vol_adj_bf_col]
        past_1y_data = df_allVolAdjBFs.loc[df_allVolAdjBFs.index >= one_year_ago, vol_adj_bf_col]
        past_3y_data = df_allVolAdjBFs.loc[df_allVolAdjBFs.index >= three_year_ago, vol_adj_bf_col]
        past_5y_data = df_allVolAdjBFs.loc[df_allVolAdjBFs.index >= five_year_ago, vol_adj_bf_col]


        # Get the current VolAdjRR value for the tenor
        current_value = df_reshaped.loc[df_reshaped['Tenor'] == tenor, 'VolAdjBF'].values[0]

        # Calculate percentiles using scipy.stats
        three_months_percentile = np.sum(past_3m_data < current_value) / len(past_3m_data) * 100
        one_year_percentile = np.sum(past_1y_data < current_value) / len(past_1y_data) * 100
        three_year_percentile = np.sum(past_3y_data < current_value) / len(past_3y_data) * 100
        five_year_percentile = np.sum(past_5y_data < current_value) / len(past_5y_data) * 100

        # Append the results
        percentile_data.append({
            'Tenor': tenor,
            '3M%_BF': round(three_months_percentile, 2),
            '1Y%_BF': round(one_year_percentile, 2),
            '3Y%_BF': round(three_year_percentile, 2),
            '5Y%_BF': round(five_year_percentile, 2)

        })



    df_percentiles = pd.DataFrame(percentile_data)
    df_reshaped = pd.merge(df_reshaped, df_percentiles, on='Tenor')


    return df_reshaped


# ----------------------------------------------------------------------------------------------------------------

# Adjust for multiple ccys and sort based on 
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







# -----------------------------------------------------------------------------------------------------



currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK']

tenors = ['1W', '1M', '3M']
timeHist = 365 * 5
delta = '10'


df_AdjBFOrdered_10d = multipleCCYAdjBFSorted(currency_pairs, tenors, timeHist, delta)





pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

print(df_AdjBFOrdered_10d)

