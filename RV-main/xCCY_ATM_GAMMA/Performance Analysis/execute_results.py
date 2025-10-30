from Results_Gen import VolatilityScreenerAnalyzer

import pandas as pd




"""Laptop"""
# path = r''

"""Desktop"""
path = r'C:\Users\ntaylor\Desktop\RV-main\xCCY_ATM_GAMMA\Data\backtest_resultsEqualCorrCoint_30Sep_5yBack_New.xlsx'


analyzer = VolatilityScreenerAnalyzer(path,
                                      'correlation',  # 'equal', 'correlation', 'cointegration_1'
                                      selected_tenors=['1W', '2W', '3W', '1M'])
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)




"""------------------------- Trade Identification -------------------------"""  
tenors =          ['1M']
currency_pair=    ['GBPUSD']
signal_type=      'expensive'
start_date=       '2020-01'
end_date=         '2025-12'

trades = analyzer.data_tenorPairSignal_trades(
    tenor=tenors, 
    currency_pair=currency_pair, 
    signal_type=signal_type, 
    start_date=start_date, 
    end_date=end_date)

print(trades)




"""------------------------- Performance Breakdown -------------------------"""

"""----- TENOR Performance -----""" 
# analyzer.print_TenorGeneralRiskRewards(
#     start_date=     '2025-01',
#     end_date=       '2025-09')



"""----- Weighting Mentrics Performance -----"""
# analyzer.print_GEN_WeightingCompare(
#             start_date='2025-01',
#             end_date='2025-09')

# comparison_df = analyzer.get_weighting_comparison_df(
#     start_date='2021-01-01',
#     end_date='2025-09-29'
# )



"""----- Currnecy Performance -----"""
# 
# analyzer.print_ALLcurrency_enhanced(
#             start_date='2021-01',
#             end_date='2021-02')


# analyzer.print_SingleccyTenoranalysis(
#     currency_pair='GBPUSD',
#     tenor='1M',
#     start_date='2022-01',
#     end_date='2025-09',
#     show_details=True,)



"""------------------------------"""
# comparison_df = analyzer.get_MultipleccyTenorComparison_df(
#     currency_pairs=['EURUSD'],
#     tenor='1W',
#     start_date='2025-01',
#     end_date='2025-10'
# )

# print(comparison_df)

# analyzer.analyze_seasonal_patterns()














# trades_df = analyzer.data_ccyTenor_trades(
#     currency_pair='USDCAD',
#     tenor='1W',
#     start_date='2025-01',
#     end_date='2025-09')
# print(trades_df)




# monthly_df = analyzer.data_monthlyBreakdown_GeneralPerformance(
#             start_date='2021-01',
#             end_date='2025-09')


# print(monthly_df)

# cols_a = ['Month','Total_Trades','Overall_Hit_Rate','Exp_Count',
#           'Exp_Hit_Rate','Exp_Avg_Vol_Diff','Exp_Std_Vol_Diff',
#           'Exp_Avg_Vol_Pct','Exp_Std_Vol_Pct','Exp_Max_Win','Exp_Max_Loss']

# cols_b = ['Month', 'Cheap_Count','Cheap_Hit_Rate','Cheap_Avg_Vol_Diff',
#           'Cheap_Std_Vol_Diff','Cheap_Avg_Vol_Pct','Cheap_Std_Vol_Pct',
#           'Cheap_Max_Win','Cheap_Max_Loss']

# print(monthly_df.loc[:, cols_a].to_string(index=True))
# print(monthly_df.loc[:, cols_b].to_string(index=True))

