import pdblp

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



# ------------------------------ Day of Interest Spot change (2 year refference) -----------------------------------------------------------------
# Get spot move realtive to past two years 
def getSpotMoveRef(ccy_list, date_interest):
    date_interest = datetime.strptime(date_interest, '%Y-%m-%d')
    current_date = datetime.today()
    weekdays_passed = np.busday_count(date_interest.date(), current_date.date())    
    results = []
    for ccy in ccy_list:
        start_date = (datetime.today() - timedelta(days= (2 * 365))).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        ticker_IV = f"{ccy} BGN Curncy"
        field = "PX_LAST"
        data_spot = blp.bdh(
            tickers=ticker_IV,
            flds=field,
            start_date=start_date,
            end_date=end_date)
        data_spot.columns = [ccy]
        data_spot['DayDiff'] = data_spot[ccy].pct_change()
        data_spot['AbsDayDiff'] = data_spot['DayDiff'] ** 2
        data_spot = data_spot.dropna()
        abs_day_diff = data_spot['AbsDayDiff']
        today_value = abs_day_diff.iloc[-(weekdays_passed + 1)]
        mean_abs_day_diff = abs_day_diff.mean()
        std_abs_day_diff = abs_day_diff.std()
        ZscoreValue = (today_value - mean_abs_day_diff) / std_abs_day_diff
        percentile = percentileofscore(abs_day_diff, today_value, kind='rank')
        results.append({
            'Currency': ccy,
            'Percentile': percentile,
            'Zscore': ZscoreValue})
    results_df = pd.DataFrame(results)
    results_df.set_index('Currency', inplace=True)
    return results_df




# ------------------------------ Implied Daily Spot Move From Implied {Tenor} Spot -----------------------------------------------------------------
def ImpSpotMoves(ccy_list):
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    data_rows = []
    for ccy in ccy_list:
        tickers = [f"{ccy} Curncy", f"{ccy}V1W Curncy", f"{ccy}V1M Curncy", f"{ccy}V3M Curncy", f"{ccy}V6M Curncy"]   
        iv_data = con.ref(tickers, ["PX_LAST"])
        spot = iv_data["value"].iloc[0]
        iv_1W = iv_data["value"].iloc[1] / 100
        iv_1M = iv_data["value"].iloc[2] / 100
        iv_3M = iv_data["value"].iloc[3] / 100
        iv_6M = iv_data["value"].iloc[4] / 100
        day_impSpotM_1W = spot * iv_1W * np.sqrt(1 / 5)
        day_impSpotM_1M = spot * iv_1M * np.sqrt(1 / 21)
        day_impSpotM_3M = spot * iv_3M * np.sqrt(1 / 63)
        day_impSpotM_6M = spot * iv_6M * np.sqrt(1 / 126)
        data_rows.append({
            "ticker": ccy,
            "Spot": spot,
            "1WV SpotMove": day_impSpotM_1W,
            "1MV SpotMove": day_impSpotM_1M,
            "3MV SpotMove": day_impSpotM_3M,
            "6MV SpotMove": day_impSpotM_6M,})
    df_spotMoves = pd.DataFrame(data_rows)
    return df_spotMoves






# ------------------------------ Yearly Spot % move Trend  (TO BE WORKED ON )-----------------------------------------------------------------
def spotYearlySeasonality(ccy):
    start_date = (datetime.today() - timedelta(days = 5 * 365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    ticker_IV = f"{ccy} BGN Curncy"
    field = "PX_LAST"
    data_spot = blp.bdh(
        tickers=ticker_IV,
        flds=field,
        start_date=start_date,
        end_date=end_date)
    data_spot.columns = [ccy]
    data_spot.index = pd.to_datetime(data_spot.index)
    grouped = data_spot.groupby(data_spot.index.year)
    first_week_pct_changes = []
    first_two_weeks_pct_changes = []
    for year, group in grouped:    
        group = group.sort_index()              
        first_day_spot = group.iloc[0][f'{ccy}'] # Takes first day of the year
        first_week_spot = group.iloc[min(4, len(group) - 1)][f'{ccy}']  # Takes 5th day of the year (first business week)
        first_two_weeks_spot = group.iloc[min(9, len(group) - 1)][f'{ccy}']  # Takes 10th day of the year (first 2 business weeks)
        first_week_pct_change = ((first_week_spot - first_day_spot) / first_day_spot) * 100
        first_two_weeks_pct_change = ((first_two_weeks_spot - first_day_spot) / first_day_spot) * 100
        first_week_pct_changes.append({'Year': year, 'First Week % Change': first_week_pct_change})
        first_two_weeks_pct_changes.append({'Year': year, 'First Two Weeks % Change': first_two_weeks_pct_change})
    df_first_week_pct = pd.DataFrame(first_week_pct_changes)
    df_first_two_weeks_pct = pd.DataFrame(first_two_weeks_pct_changes)
    df_pct_changes = pd.merge(df_first_week_pct, df_first_two_weeks_pct, on='Year')
    return df_pct_changes


















# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------







# ----------------------------Implied Vols --> Forward Implied Vol --> Implied Spot Move -----------------------------------

# con = pdblp.BCon(debug=False, port=8194, timeout=5000)
# con.start()

# ccy = 'EURUSD'

# iv_combined = []

# tickers = [f"{ccy} Curncy", f"{ccy}V1W Curncy", f"{ccy}V2W Curncy"]

# iv_data = con.ref(tickers, ["PX_LAST"])
# iv_spot = iv_data["value"].iloc[0]
# iv_1W = iv_data["value"].iloc[1]
# iv_2W = iv_data["value"].iloc[2]

# iv_combined.append([ccy, iv_spot, iv_1W, iv_2W])




# df_combined = pd.DataFrame(
#         iv_combined,
#         columns=["Currency Pair", "Spot", "1W_IV", "2W_IV"]
#     )




# # Calculate forward volatility
# def calculate_forward_vol(t1, t2, iv_1, iv_2):
   
#     iv_1 = iv_1 / 100
#     iv_2 = iv_2 / 100
    
#     forward_vol = np.sqrt((t2 * iv_2**2 - t1 * iv_1**2) / (t2 - t1))
    
#     return forward_vol * 100


# t1 = 5  # 1W = 5 tradeDs
# t2 = 10  # 2W = 10 tradeDs

# # Extract volatilities
# iv_1 = df_combined.loc[0, '1W_IV']  # 1-week IV
# iv_2 = df_combined.loc[0, '2W_IV']  # 2-week IV

# # Calculate forward volatility
# df_combined['Forward_Vol'] = calculate_forward_vol(t1, t2, iv_1, iv_2)





# def calculate_implied_move(spot_price, forward_vol, t1, t2, trading_days_per_year=252):
    
#     # Convert forward volatility from % to decimal
#     forward_vol = forward_vol / 100
    
#     # Time duration in trading days
#     time_duration = t2 - t1
    
#     # Calculate the implied spot move
#     implied_move = spot_price * forward_vol * np.sqrt(time_duration / trading_days_per_year)
    
#     return implied_move










# spot_price = df_combined['Spot']  
# forward_vol = df_combined.loc[0, 'Forward_Vol']  

# # Calculate implied spot move
# df_combined['Implied_Spot_Move'] = calculate_implied_move(spot_price, forward_vol, t1, t2)

# df_combined