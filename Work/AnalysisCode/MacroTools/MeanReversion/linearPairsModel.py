import blpapi

import pdblp
from numpy import unique
import numpy as np
from xbbg import blp
from xbbg import blp, pipeline

from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

import os
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts



def plotSpotCompare(df_price, t_or_f):

    if t_or_f == True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Dual-Axis Line Plot ---
        # Plot EURUSD on the left y-axis
        ax1.plot(df_price.index, df_price['EURUSD'], color='blue', label='EURUSD')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('EURUSD', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        # Create a second y-axis for GBPUSD
        ax1_2 = ax1.twinx()
        ax1_2.plot(df_price.index, df_price['GBPUSD'], color='green', label='GBPUSD')
        ax1_2.set_ylabel('GBPUSD', color='green')
        ax1_2.tick_params(axis='y', labelcolor='green')
        ax1_2.legend(loc='upper right')

        # Add a title for the dual-axis plot
        ax1.set_title('EURUSD and GBPUSD Daily Prices')

        # --- Scatter Plot ---
        # Scatter plot of GBPUSD vs. EURUSD
        ax2.scatter(df_price['EURUSD'], df_price['GBPUSD'], color='purple', alpha=0.7)
        ax2.set_xlabel('EURUSD')
        ax2.set_ylabel('GBPUSD')
        ax2.set_title('GBPUSD vs. EURUSD')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

        return




# Get data for ccys in ccy_pairs
def getData(ccy_pairs):
    daily_prices = []
    for ccy in ccy_pairs:
            
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            ticker_IV = f"{ccy} BGN Curncy"
            data_spot = blp.bdh(
                tickers=ticker_IV,
                flds="PX_LAST",
                start_date=start_date,
                end_date=end_date
            )
            data_spot.columns = [ccy]
            daily_prices.append(data_spot)

    df_price = pd.concat(daily_prices, axis=1)
    return df_price




# Generate residual score - model fit returning 
def create_residuals(price_df):
    # Create OLS model
    Y = price_df['EURUSD']
    x = price_df['GBPUSD']
    x = sm.add_constant(x)
    model = sm.OLS(Y, x)
    res = model.fit()
    
    # Beta hedge ratio (coefficent from OLS)
    beta_hr = res.params[1]
    print(f'Beta Hedge Ratio: {beta_hr}')
    
    # Residuals
    price_df["Residuals"] = res.resid
    return price_df

















# -------------------------------------------------------------------------------------------------------------------------------------------------------------------




ccy_pairs = ['EURUSD', 'GBPUSD']


df_price = getData(ccy_pairs)

df_resid = create_residuals(df_price)


print(df_resid)