import pandas as pd
from datetime import datetime

# ============================================================================
# QUERY PARAMETERS - MODIFY THESE
# ============================================================================

FILE_PATH = 'News.csv'

# Currency to filter (e.g., 'EUR', 'USD', 'JPY', 'GBP') - Set to None for all
CURRENCY = 'JPY'

# Date range (format: 'YYYY-MM-DD') - Set to None to skip date filtering
START_DATE = '2026-01-28'
END_DATE = '2026-02-2'

# Importance levels to include - ['M', 'D'] for both, ['M'] for M only, etc.
IMPORTANCE = ['M', 'D']

# Display mode: 'full' for detailed, 'compact' for one-line summaries
DISPLAY_MODE = 'full'  # Options: 'full' or 'compact'

# Export options (set to None to skip export)
EXPORT_EXCEL = None  # e.g., 'output.xlsx' or None
EXPORT_CSV = None    # e.g., 'output.csv' or None

# ============================================================================
# QUERY EXECUTION - DON'T MODIFY BELOW
# ============================================================================

def query_news(file_path, currency=None, start_date=None, end_date=None, importance=['M', 'D']):
    """Query news from CSV file."""
    df = pd.read_csv(file_path)
    df.columns = ['Date', 'Importance', 'Country/Region', 'Detail']
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    result = df.copy()
    
    if start_date:
        result = result[result['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        result = result[result['Date'] <= pd.to_datetime(end_date)]
    if importance:
        result = result[result['Importance'].isin(importance)]
    if currency:
        result = result[result['Country/Region'].str.strip().str.upper() == currency.upper()]
    
    return result.sort_values('Date').reset_index(drop=True)


def display_results(results, mode='full'):
    """Display results."""
    if len(results) == 0:
        print("\n⚠️  No results found!\n")
        return
    
    print(f"\n{'='*120}")
    print(f"📰 Found {len(results)} news items")
    print(f"{'='*120}\n")
    
    if mode == 'compact':
        for _, row in results.iterrows():
            print(f"{row['Date'].strftime('%Y-%m-%d')} | {row['Importance']:>1} | {row['Country/Region']:>3} | {row['Detail']}")
    else:
        for _, row in results.iterrows():
            print(f"📅 {row['Date'].strftime('%Y-%m-%d')} ({row['Date'].strftime('%A')})")
            print(f"⚡ Importance: {row['Importance']}")
            print(f"💱 Currency: {row['Country/Region']}")
            print(f"📝 {row['Detail']}")
            print(f"{'-'*120}\n")


# Execute query
print("\n🔍 Querying news...")
print(f"   Currency: {CURRENCY if CURRENCY else 'ALL'}")
print(f"   Date Range: {START_DATE} to {END_DATE}")
print(f"   Importance: {IMPORTANCE}")

results = query_news(FILE_PATH, CURRENCY, START_DATE, END_DATE, IMPORTANCE)
display_results(results, DISPLAY_MODE)

# Export if requested
if EXPORT_EXCEL:
    results.to_excel(EXPORT_EXCEL, index=False)
    print(f"✅ Exported to Excel: {EXPORT_EXCEL}")

if EXPORT_CSV:
    results.to_csv(EXPORT_CSV, index=False)
    print(f"✅ Exported to CSV: {EXPORT_CSV}")

# Summary
if len(results) > 0:
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total items: {len(results)}")
    print(f"Date range: {results['Date'].min().strftime('%Y-%m-%d')} to {results['Date'].max().strftime('%Y-%m-%d')}")
    print(f"{'='*60}\n")