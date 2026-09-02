from ImpliedvRealized import gammaScreenerPresent
from detailScreen import *

# create_combined_summary, create_fx_volatility_heatmap, rank_volatility_opportunities



# ----------------------------- HEAT MAP PLOTTING -----------------------------


""" Simple Implied v Realized Gamma Screen Heat Map for front end
    Inputs:
        - CCYs: List of ccys of interest

    Outputs:
        -       Realized Vol: Current 30min sampled, annulaized 
        -        Implied Vol: Current implied vol annulaized
        - Implied - Realized: Current Diferent 
"""
ccy_gamma1 = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK',
    'USDMXN', 'USDBRL', 'USDCNH', 'EURGBP', 'EURCHF']

""" 1W, 2W, 1M """
# gammaScreenerPresent(ccy_gamma1)

# --------------------------------------------------------------------------------------------------

""" Detailed Gamma Screen Heat Map:
    Inputs:
        - CCYs: List of ccys of interest
        - Tenors: List of tenors of Interest
        - days_back: Number of days to compare current states

    Outputs:
        -            Implied Vol: Current Value and Percentile
        - Implied - Realized Vol: Current Difference and Percentile 
        - Vol Relation to Market: The percentile ranking of CCY of interest relative to other ccys in list
"""

"""ASIA"""
# currency_list =  ['USDJPY', 'USDCNH', 'USDKRW', 'USDSGD', 'USDTWD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD']

# currency_list = ['USDJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'EURJPY', 'CHFJPY', 'CADJPY']

"""LATAM"""
# currency_list = ['USDBRL', 'USDMXN', 'USDCLP', 'USDCOP', 'USDPEN', 'EURUSD', 'USDJPY', 'AUDUSD']

"""EURO Block"""
# currency_list = ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF',  'EURHUF',  'EURNOK', 'EURSEK', 'EURCAD']

"""Majors"""
currency_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN', 'USDBRL'] 












# currency_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDCNH'] 

# currency_list = ['AUDUSD', 'EURAUD', 'AUDCHF', 'AUDJPY', 'AUDCAD', 'EURCAD']

# tenors = ['1M', '2M', '3M']

tenors = ['1W', '2W', '3W', '1M', '2M', '3M']

days_back = 365

create_fx_volatility_heatmap(currency_list, tenors, days_back)





# --------------------------------------------------------------------------------------------------

# currency_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN','USDBRL', 'USDCNH']
# tenors = ['1W', '2W', '3W', '1M']
# days_back = 365

# combined_summary, analysis_results = create_combined_summary(currency_list, tenors, days_back)

# weighted_results = rank_volatility_opportunities(
#     combined_summary, 
#     tenors, 
#     method='weighted_score',
#     iv_weight=1.0,      # Implied Vol Percentile
#     ir_weight=1.0,      # Relative Vol Premia Percentile
#     mrkt_weight=1.0,    # Relative Ranking to Rest of Market (Corr Weighted)
# )

# print("=== TOP 5 BUY OPPORTUNITIES (Simple Ranking) ===")
# print(weighted_results['buy_opportunities'])
# print()
# print(weighted_results['sell_opportunities'])





