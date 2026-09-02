from EventData import *
import re



manager = setup_economic_data_manager(blp)

# Test individual currency
# usd_schedule = manager.get_currency_schedule('USD', days=10)

# print(usd_schedule)

# # Get currency pair schedule  
# eurusd_schedule = manager.get_currency_pair_schedule('EURUSD', days=14)

# # Get all major pairs
# major_pairs = manager.get_major_pairs_schedule(days=30)




# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.expand_frame_repr', False)




# manager = setup_economic_data_manager(blp)


# dates = manager.get_pastData('CPI YOY Index', 100)

# print(dates)



# #Example 1: Get next week's data for all currencies
# print("=== NEXT WEEK (1w) - ALL CURRENCIES ===")
# next_week = manager.get_schedule_by_period('1w')
# print(next_week)

# # Example 2: Get next 2 weeks for USD only
# print("\n=== NEXT 2 WEEKS (2W) - USD ONLY ===")
# usd_2weeks = manager.get_currency_schedule_by_period('USD', '1W')
# print(usd_2weeks)


# # Example 3: Get next 3 weeks for EURUSD pair
# print("\n=== NEXT 3 WEEKS (3w) - EURUSD PAIR ===")
# eurusd_3weeks = manager.get_currency_pair_schedule_by_period('USDJPY', '2M')
# print(eurusd_3weeks)






# print("\n=== ENHANCED ALL CURRENCIES - NEXT WEEK (1w) WITH DETAILED FIELDS ===")
# enhanced_all = manager.get_enhanced_schedule_by_period('2w', include_detailed_fields=True)
# print(enhanced_all)



usdjpy_basic = manager.get_currency_pair_detailed_by_period('USDJPY', '2w', 'future', try_all_fields=True)
print(usdjpy_basic)