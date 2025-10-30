"""
-------------------------------------------------------------------------------------------------
RUNNING THE SIGNAL ON PAST VOL DATA - Collecting Brief Summery and All Trade Info
-------------------------------------------------------------------------------------------------
"""

import pandas as pd
from xCCY_BackTestEval_EvenCorrCoint import *
















currency_list = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 
        'USDCAD', 'USDCHF', 'USDNOK', 'USDSEK', 'USDMXN',
        'USDBRL', 'USDCNH']
    

gamma_tenors = ['1W', '2W', '3W', '1M']
days_backtested= 365 * 5
percentile_threshold_low=15.0
percentile_threshold_high=85.0
frequency='d'
cointegration_excel_path_1st = r'C:\Users\Nick Taylor\Desktop\Work\AnalysisCode\VolTools\xCCyVol_ATM_GAMMA\fxVol_cointegrated.xlsx'  # Need to Rework the rankings

cointegration_excel_path_2nd = r'C:\Users\Nick Taylor\Desktop\Work\AnalysisCode\VolTools\xCCyVol_ATM_GAMMA\fx_cointegration_analysis_5Y.xlsx'


df_summary_EqualWeights, df_trades_EqualWeights = backtest_ImpliedRealizedDiff_EqualWeightAve_Results(currency_list, gamma_tenors, 
                                                    days_backtested, 
                                                    percentile_threshold_low, percentile_threshold_high, 
                                                    frequency)



df_summary_CorrWeights, df_trades_CorrWeights = backtest_ImpliedRealizedDiff_CorrWeightAve_Results(currency_list, gamma_tenors, 
                                                    days_backtested, 
                                                    percentile_threshold_low, percentile_threshold_high, 
                                                    frequency)



df_summary_CointWeights_1st, df_trades_CointWeights_1st = backtest_ImpliedRealizedDiff_CointWeightAve_Results(currency_list, gamma_tenors,
                                                        days_backtested,
                                                        percentile_threshold_low, percentile_threshold_high,
                                                        frequency,
                                                        cointegration_excel_path_1st)


df_summary_CointWeights_2nd, df_trades_CointWeights_2nd = backtest_ImpliedRealizedDiff_CointWeightAve_Results(currency_list, gamma_tenors,
                                                        days_backtested,
                                                        percentile_threshold_low, percentile_threshold_high,
                                                        frequency,
                                                        cointegration_excel_path_2nd)




pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

df_trades_EqualWeights = df_trades_EqualWeights.sort_values('signal_date', ascending=True)
df_trades_CorrWeights = df_trades_CorrWeights.sort_values('signal_date', ascending=True)
df_trades_CointWeights_1st = df_trades_CointWeights_1st.sort_values('signal_date', ascending=True)
df_trades_CointWeights_2nd = df_trades_CointWeights_2nd.sort_values('signal_date', ascending=True)


filename = 'backtest_resultsEqualCorrCoint_30Sep_5yBack_New.xlsx'

with pd.ExcelWriter(filename, engine='openpyxl') as writer:
    df_summary_EqualWeights.to_excel(writer, sheet_name='Summary_EqualWeights', index=True)
    df_trades_EqualWeights.to_excel(writer, sheet_name='Trades_EqualWeights', index=True)

    df_summary_CorrWeights.to_excel(writer, sheet_name='Summary_CorrWeights', index=True)
    df_trades_CorrWeights.to_excel(writer, sheet_name='Trades_CorrWeights', index=True)

    df_summary_CointWeights_1st.to_excel(writer, sheet_name='Summary_CointWeights_1st', index=True)
    df_trades_CointWeights_1st.to_excel(writer, sheet_name='Trades_CointWeights_1st', index=True)

    df_summary_CointWeights_2nd.to_excel(writer, sheet_name='Summary_CointWeights_2nd', index=True)
    df_trades_CointWeights_2nd.to_excel(writer, sheet_name='Trades_CointWeights_2nd', index=True)



print(f"All DataFrames saved to {filename}")





# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------









# # df_summary_betaWeights, df_trades_betaWeights = backtest_ImpliedRealizedDiff_BetaWeightAve_Results(currency_list, gamma_tenors, 
# #                                                     days_backtested, 
# #                                                     percentile_threshold_low, percentile_threshold_high, 
# #                                                     frequency)

