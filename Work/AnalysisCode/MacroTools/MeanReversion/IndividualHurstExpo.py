import blpapi

import pdblp
from numpy import unique
import numpy as np
from xbbg import blp
from xbbg import blp, pipeline

from datetime import datetime, timedelta
import pandas as pd


from numpy import cumsum, log, polyfit, sqrt, std, subtract




# Calculation of Hurst Exponential
def hurst(timeSeries):
    
    lags = range(2, 50)
    tau = [sqrt(std(subtract(timeSeries[lag:], timeSeries[:-lag]))) for lag in lags] 
    poly = polyfit(log(lags), log(tau), 1)

    return poly[0] * 2.0




# Compile all Hurst Expo Scores for currencies of interest 
def calculate_hurst_for_tickers(tickers, days_past):

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_past)).strftime("%Y-%m-%d")

    hurst_exponents = {}
    for ticker in tickers:
       
        data_spot = blp.bdh(
            tickers=f'{ticker} BGN Curncy',
            flds="PX_LAST",
            start_date=start_date,
            end_date=end_date
        )

       
        if data_spot.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue
        data_spot.columns = [ticker]



        array_prices = data_spot[ticker].dropna().to_numpy()
        hurst_exponent = hurst(array_prices)
        
        hurst_exponents[ticker] = hurst_exponent

    return hurst_exponents




tickers = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK',
    'USDMXN', 'USDBRL', 'USDCNH'
]

days_past = 365

hurst_results = calculate_hurst_for_tickers(tickers, days_past)

hurst_df = pd.DataFrame(list(hurst_results.items()), columns=['Currency', 'Hurst Exponent'])
hurst_df = hurst_df.sort_values(by='Hurst Exponent', ascending=True)

print(hurst_df)