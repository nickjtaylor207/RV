from temp_coint import *


# currencies = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCHF', 'USDMXN', 'USDBRL']
# tenor = '1M'


# top_pairs = Screen_MultiPair_Metrics.analyze_currencies(currencies, tenor=tenor)
# print(top_pairs.columns)






currencies = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 
                'USDCHF', 'USDCAD', 'USDMXN', 'USDBRL', 'USDZAR', 
                'EURGBP', 'USDCNH', 'USDNOK', 'USDSEK', ] 

tenors = ['1W', '2W', '1M', '2M', '3M', '6M', '1Y'] 
days=730
output_file='fx_cointegration_analysis.xlsx'

results = run_multi_tenor_analysis(currencies, tenors, days, output_file)


