import pandas as pd
import numpy as np

from xbbg import blp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pandas.plotting import table
import re

from pandas.tseries.offsets import DateOffset
from scipy.stats import percentileofscore
from typing import List, Dict, Optional, Union






from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime, timedelta

class EconomicDataManager:
    DEFAULT_FIELDS = [
        "NAME",
        "TIME", 
        "ECO_RELEASE_TIME",
        "ECO_RELEASE_DT",
        "ACTUAL_RELEASE",
        "BN_SURVEY_MEDIAN",
        "BN_SURVEY_AVERAGE", 
        "BN_SURVEY_HIGH",
        "BN_SURVEY_LOW",
        "FORECAST_STANDARD_DEVIATION",
        "BN_SURVEY_NUMBER_OBSERVATIONS"]
    FUTURE_FIELDS = ["ECO_RELEASE_DT", "ECO_RELEASE_TIME", "NAME"]
    FIELD_RENAME_MAP = {
        "ECO_RELEASE_DT": "ReleaseDate",
        "ECO_RELEASE_TIME": "ReleaseTime",
        "ACTUAL_RELEASE": "Actual", 
        "BN_SURVEY_MEDIAN": "SMed",
        "BN_SURVEY_AVERAGE": "SAve",
        "BN_SURVEY_HIGH": "SHigh",
        "BN_SURVEY_LOW": "SLow",
        "FORECAST_STANDARD_DEVIATION": "ForecastSDTv",
        "BN_SURVEY_NUMBER_OBSERVATIONS": "NumbSurvey"}
    # -------------------------------------------------------------------------------------------
    # INITIALIZING Functions
    def __init__(self, bloomberg_api, indicators_dict: Optional[Dict] = None):
        self.blp = bloomberg_api
        self.indicators = indicators_dict or {}
    def add_indicators(self, currency_code: str, indicators: List[Dict]) -> None:
        self.indicators[currency_code] = indicators
    # -------------------------------------------------------------------------------------------
    # CORE DATA Retrieving Functions
    def get_futureData(self, ticker: str, days, 
                    custom_fields: Optional[List[str]] = None) -> pd.DataFrame:
        fields = custom_fields or self.FUTURE_FIELDS
        today = datetime.today().date()
        target_date = today + timedelta(days=days)
        try:
            data = self.blp.bdh(
                tickers=ticker,
                flds=fields,
                start_date=today.strftime('%Y-%m-%d'),
                end_date=target_date.strftime('%Y-%m-%d'))
            if data.empty:
                return pd.DataFrame()
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = data.columns.droplevel(0)
            data = data.rename(columns=self.FIELD_RENAME_MAP)
            if 'ReleaseDate' in data.columns:
                data['ReleaseDate'] = pd.to_datetime(
                    data['ReleaseDate'].astype(str).str.split('.').str[0],
                    format='%Y%m%d',
                    errors='coerce')
                data = data.dropna(subset=['ReleaseDate'])
                data = data[data['ReleaseDate'].dt.date < target_date]
                data = data.sort_values('ReleaseDate')
            return data
        except Exception as e:
            print(f"Error getting future data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_pastData(self, ticker: str, days: int, 
                    custom_fields: Optional[List[str]] = None) -> pd.DataFrame:
        # For past data, we can try to get all fields
        fields = custom_fields or self.DEFAULT_FIELDS
        today = datetime.today().date()
        cutoff_date = today - timedelta(days=days)
        
        try:
            data = self.blp.bdh(
                tickers=ticker,
                flds=fields,
                start_date=(today - timedelta(days=days*2)).strftime('%Y-%m-%d'),
                end_date=today.strftime('%Y-%m-%d'))
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle multi-level columns if they exist
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = data.columns.droplevel(0)
            
            # Rename columns using the map
            data = data.rename(columns=self.FIELD_RENAME_MAP)
            
            # Process release date if it exists
            if 'ReleaseDate' in data.columns:
                data['ReleaseDate'] = pd.to_datetime(
                    data['ReleaseDate'].astype(str).str.split('.').str[0],
                    format='%Y%m%d',
                    errors='coerce')
                data = data.dropna(subset=['ReleaseDate'])
                data = data[data['ReleaseDate'].dt.date > cutoff_date]
                data = data.sort_values('ReleaseDate')
            
            return data
            
        except Exception as e:
            print(f"Error getting past data for {ticker}: {str(e)}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------------------------
    # DATE OF EVENT Functions
    def get_nextRelease(self, ticker: str, days: int = 100) -> Optional[str]:
        try:
            future_df = self.get_futureData(ticker, days)
            if future_df.empty or 'ReleaseDate' not in future_df.columns:
                return None
            return str(future_df.iloc[0]['ReleaseDate'].date())
        except Exception as e:
            print(f"Error getting next release for {ticker}: {str(e)}")
            return None
  
    # -------------------------------------------------------------------------------------------
    # SCHEDULE OF EVENTS Functions
    def get_release_schedule(self, indicators: List[Dict], days: int = 100) -> pd.DataFrame:
        results = []
        failed_tickers = []
        
        for indicator in indicators:
            try:
                release_date = self.get_nextRelease(indicator["Ticker"], days)
                if release_date:  # Only add if we got a valid date
                    results.append({
                        "Country": indicator["Country"],
                        "Data": indicator["Data"], 
                        "Ticker": indicator["Ticker"],
                        "Release Date": release_date})
                else:
                    failed_tickers.append(indicator["Ticker"])
            except Exception as e:
                print(f"Warning: Could not get release date for {indicator['Ticker']}: {str(e)}")
                failed_tickers.append(indicator["Ticker"])
                continue
        
        # Report on failed tickers
        if failed_tickers:
            print(f"Could not get release dates for {len(failed_tickers)} tickers: {failed_tickers[:5]}{'...' if len(failed_tickers) > 5 else ''}")
        
        # Check if we have any results before creating DataFrame
        if not results:
            print("No release dates found for any indicators")
            return pd.DataFrame(columns=["Country", "Data", "Ticker", "Release Date", "Weekday", "Week"])
        
        df = pd.DataFrame(results)
        print(f"Found {len(df)} release dates out of {len(indicators)} indicators")
        
        # Convert Release Date column to datetime
        df["Release Date"] = pd.to_datetime(df["Release Date"], errors='coerce')
        df = df.dropna(subset=["Release Date"])
        
        # Check again after date conversion
        if df.empty:
            print("No valid release dates after datetime conversion")
            return pd.DataFrame(columns=["Country", "Data", "Ticker", "Release Date", "Weekday", "Week"])
        
        df = df.sort_values(by="Release Date")
        df.insert(
            loc=df.columns.get_loc("Release Date") + 1,
            column="Weekday", 
            value=df["Release Date"].dt.day_name().str[:3])
        df = self._add_week_grouping(df)
        return df
    
    # -------------------------------------------------------------------------------------------
    # UTILITY Functions
    def filter_by_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return df[(df['Release Date'] >= start) & (df['Release Date'] <= end)]
    
    def _add_week_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        today = pd.Timestamp.today().normalize()
        start_of_this_week = today - pd.Timedelta(days=today.weekday())
        df["Week"] = ((df["Release Date"] - start_of_this_week).dt.days // 7).clip(lower=0)
        return df
    
    def test_ticker_availability(self, tickers: List[str], days: int = 30) -> Dict[str, bool]:
        """Test which tickers actually return data"""
        results = {}
        for ticker in tickers:
            try:
                data = self.get_futureData(ticker, days)
                results[ticker] = not data.empty
            except:
                results[ticker] = False
        return results

    # -------------------------------------------------------------------------------------------
    # CURRENCY SPECIFIC Functions
    def get_currency_schedule(self, currency_code: str, days: int = 100) -> pd.DataFrame:
        if currency_code not in self.indicators:
            raise ValueError(f"Currency code '{currency_code}' not found in indicators")
        return self.get_release_schedule(self.indicators[currency_code], days)
    
    def get_all_currencies_schedule(self, days: int = 100) -> pd.DataFrame:
        all_indicators = []
        for currency_indicators in self.indicators.values():
            all_indicators.extend(currency_indicators)
        return self.get_release_schedule(all_indicators, days)

    def get_currency_pair_schedule(self, currency_pair: str, days: int = 100) -> pd.DataFrame:
        """
        Get combined release schedule for a currency pair (e.g., 'EURUSD', 'GBPJPY').
        """
        # Validate input length
        if len(currency_pair) != 6:
            raise ValueError(f"Currency pair '{currency_pair}' must be exactly 6 characters (e.g., 'EURUSD')")
        
        # Extract base and quote currencies
        base_currency = currency_pair[:3].upper()
        quote_currency = currency_pair[3:].upper()
        
        # Validate currencies exist in indicators
        missing_currencies = []
        if base_currency not in self.indicators:
            missing_currencies.append(base_currency)
        if quote_currency not in self.indicators:
            missing_currencies.append(quote_currency)
        
        if missing_currencies:
            available = list(self.indicators.keys())
            raise ValueError(
                f"Currency(ies) {missing_currencies} not found in indicators. "
                f"Available currencies: {available}"
            )
        
        # Combine indicators from both currencies
        combined_indicators = []
        combined_indicators.extend(self.indicators[base_currency])
        combined_indicators.extend(self.indicators[quote_currency])
        
        # Get the combined release schedule
        schedule_df = self.get_release_schedule(combined_indicators, days)
        
        # Add currency pair information
        if not schedule_df.empty:
            schedule_df.insert(0, 'Currency Pair', currency_pair)
            
            # Add base/quote currency classification
            def classify_currency(row):
                country = row['Country']
                # Check if this indicator belongs to base currency
                for indicator in self.indicators[base_currency]:
                    if indicator['Country'] == country and indicator['Data'] == row['Data']:
                        return base_currency
                return quote_currency
            
            schedule_df.insert(1, 'Currency', schedule_df.apply(classify_currency, axis=1))
            
            # Sort by release date for chronological view
            schedule_df = schedule_df.sort_values('Release Date').reset_index(drop=True)
        
        return schedule_df

    def get_currency_pair_data_detailed(self, currency_pair: str, days: int = 100, 
                                      data_type: str = 'future') -> pd.DataFrame:
        """
        Get detailed economic data (with all fields) for a currency pair.
        """
        # Validate input
        if len(currency_pair) != 6:
            raise ValueError(f"Currency pair '{currency_pair}' must be exactly 6 characters")
        
        if data_type not in ['future', 'past']:
            raise ValueError("data_type must be either 'future' or 'past'")
        
        base_currency = currency_pair[:3].upper()
        quote_currency = currency_pair[3:].upper()
        
        # Validate currencies exist
        missing_currencies = []
        if base_currency not in self.indicators:
            missing_currencies.append(base_currency)
        if quote_currency not in self.indicators:
            missing_currencies.append(quote_currency)
        
        if missing_currencies:
            available = list(self.indicators.keys())
            raise ValueError(
                f"Currency(ies) {missing_currencies} not found. Available: {available}"
            )
        
        # Combine indicators from both currencies
        combined_indicators = []
        combined_indicators.extend(self.indicators[base_currency])
        combined_indicators.extend(self.indicators[quote_currency])
        
        # Get detailed data for each indicator
        detailed_results = []
        
        for indicator in combined_indicators:
            try:
                # Choose the appropriate data retrieval method
                if data_type == 'future':
                    indicator_df = self.get_futureData(indicator["Ticker"], days)
                else:
                    indicator_df = self.get_pastData(indicator["Ticker"], days)
                
                if not indicator_df.empty:
                    # Add indicator metadata to each row
                    indicator_df['Currency_Pair'] = currency_pair
                    indicator_df['Country'] = indicator['Country']
                    indicator_df['Data_Type'] = indicator['Data']
                    indicator_df['Ticker'] = indicator['Ticker']
                    
                    # Determine which currency this indicator belongs to
                    if indicator in self.indicators[base_currency]:
                        indicator_df['Currency'] = base_currency
                    else:
                        indicator_df['Currency'] = quote_currency
                    
                    detailed_results.append(indicator_df)
                    
            except Exception as e:
                print(f"Warning: Could not retrieve data for {indicator['Ticker']}: {str(e)}")
                continue
        
        # Combine all results
        if detailed_results:
            final_df = pd.concat(detailed_results, ignore_index=True)
            
            # Reorder columns for better readability
            priority_cols = ['Currency_Pair', 'Currency', 'Country', 'Data_Type', 'Ticker']
            if 'ReleaseDate' in final_df.columns:
                priority_cols.append('ReleaseDate')
            other_cols = [col for col in final_df.columns if col not in priority_cols]
            final_df = final_df[priority_cols + other_cols]
            
            # Sort by release date if available
            if 'ReleaseDate' in final_df.columns:
                final_df = final_df.sort_values('ReleaseDate').reset_index(drop=True)
            
            return final_df
        else:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=['Currency_Pair', 'Currency', 'Country', 'Data_Type', 'Ticker', 'ReleaseDate'])

    def get_major_pairs_schedule(self, days: int = 100) -> pd.DataFrame:
        """
        Get release schedule for all major currency pairs.
        """
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'NZDUSD']
        available_pairs = []
        
        # Check which pairs can be created with available currencies
        for pair in major_pairs:
            base = pair[:3]
            quote = pair[3:]
            if base in self.indicators and quote in self.indicators:
                available_pairs.append(pair)
        
        if not available_pairs:
            print("No major pairs available with current currency data")
            return pd.DataFrame()
        
        print(f"Getting schedules for available major pairs: {available_pairs}")
        
        # Get schedules for all available major pairs
        all_schedules = []
        for pair in available_pairs:
            try:
                pair_schedule = self.get_currency_pair_schedule(pair, days)
                if not pair_schedule.empty:
                    all_schedules.append(pair_schedule)
            except Exception as e:
                print(f"Warning: Could not get schedule for {pair}: {str(e)}")
                continue
        
        if all_schedules:
            combined_df = pd.concat(all_schedules, ignore_index=True)
            return combined_df.sort_values('Release Date').reset_index(drop=True)
        else:
            return pd.DataFrame()
        


    def parse_time_period(self, period_str: str) -> int:
        """
        Parse time period strings like '1w', '2W', '3m', '1M' into number of days.
        
        Args:
            period_str (str): Time period string (e.g., '1w', '2W', '3m', '1M')
            
        Returns:
            int: Number of days
            
        Raises:
            ValueError: If the period string format is invalid
        """
        if not isinstance(period_str, str):
            raise ValueError("Period must be a string")
        
        # Remove any spaces and convert to lowercase for processing
        period_str = period_str.strip().lower()
        
        # Use regex to parse the period string
        match = re.match(r'^(\d+)([wm])$', period_str)
        
        if not match:
            raise ValueError(
                f"Invalid period format: '{period_str}'. "
                f"Use format like '1w', '2W', '3m', '1M' (w=weeks, m=months)"
            )
        
        number = int(match.group(1))
        unit = match.group(2)
        
        if unit == 'w':  # weeks
            return number * 7
        elif unit == 'm':  # months (approximate as 30 days)
            return number * 30
        else:
            raise ValueError(f"Unsupported time unit: {unit}")

    def get_schedule_by_period(self, period: str, currency_codes: list = None) -> pd.DataFrame:
        """
        Get economic release schedule for a specific time period.
        
        Args:
            period (str): Time period like '1w', '2W', '3m', '1M'
            currency_codes (list, optional): List of currency codes. If None, uses all currencies.
            
        Returns:
            pd.DataFrame: Schedule with columns [Country, Data, Ticker, Release Date, Weekday, Week]
        """
        days = self.parse_time_period(period)
        
        if currency_codes is None:
            # Get all currencies
            return self.get_all_currencies_schedule(days)
        else:
            # Get specific currencies
            all_indicators = []
            missing_currencies = []
            
            for currency_code in currency_codes:
                currency_code = currency_code.upper()
                if currency_code in self.indicators:
                    all_indicators.extend(self.indicators[currency_code])
                else:
                    missing_currencies.append(currency_code)
            
            if missing_currencies:
                available = list(self.indicators.keys())
                print(f"Warning: Currency(ies) {missing_currencies} not found. Available: {available}")
            
            if not all_indicators:
                return pd.DataFrame(columns=["Country", "Data", "Ticker", "Release Date", "Weekday", "Week"])
            
            return self.get_release_schedule(all_indicators, days)

    def get_currency_schedule_by_period(self, currency_code: str, period: str) -> pd.DataFrame:
        """
        Get economic release schedule for a specific currency and time period.
        
        Args:
            currency_code (str): Currency code (e.g., 'USD', 'EUR')
            period (str): Time period like '1w', '2W', '3m', '1M'
            
        Returns:
            pd.DataFrame: Schedule with columns [Country, Data, Ticker, Release Date, Weekday, Week]
        """
        days = self.parse_time_period(period)
        return self.get_currency_schedule(currency_code.upper(), days)

    def get_currency_pair_schedule_by_period(self, currency_pair: str, period: str) -> pd.DataFrame:
        """
        Get economic release schedule for a currency pair and time period.
        
        Args:
            currency_pair (str): Currency pair like 'EURUSD', 'GBPJPY'
            period (str): Time period like '1w', '2W', '3m', '1M'
            
        Returns:
            pd.DataFrame: Schedule with columns [Currency Pair, Currency, Country, Data, Ticker, Release Date, Weekday, Week]
        """
        days = self.parse_time_period(period)
        return self.get_currency_pair_schedule(currency_pair, days)

    def get_major_pairs_schedule_by_period(self, period: str) -> pd.DataFrame:
        """
        Get economic release schedule for all major currency pairs and time period.
        
        Args:
            period (str): Time period like '1w', '2W', '3m', '1M'
            
        Returns:
            pd.DataFrame: Schedule with columns [Currency Pair, Currency, Country, Data, Ticker, Release Date, Weekday, Week]
        """
        days = self.parse_time_period(period)
        return self.get_major_pairs_schedule(days)

    def get_filtered_schedule_by_period(self, period: str, 
                                    currencies: list = None,
                                    data_types: list = None,
                                    countries: list = None) -> pd.DataFrame:
        """
        Get filtered economic release schedule for a specific time period with multiple filter options.
        
        Args:
            period (str): Time period like '1w', '2W', '3m', '1M'
            currencies (list, optional): Filter by currency codes (e.g., ['USD', 'EUR'])
            data_types (list, optional): Filter by data types (e.g., ['CPI YoY', 'GDP QoQ'])
            countries (list, optional): Filter by countries (e.g., ['US', 'EU'])
            
        Returns:
            pd.DataFrame: Filtered schedule
        """
        # Get the base schedule
        schedule_df = self.get_schedule_by_period(period, currencies)
        
        if schedule_df.empty:
            return schedule_df
        
        # Apply filters
        if data_types:
            data_types = [dt.strip() for dt in data_types]
            schedule_df = schedule_df[schedule_df['Data'].isin(data_types)]
        
        if countries:
            countries = [c.strip() for c in countries]
            schedule_df = schedule_df[schedule_df['Country'].isin(countries)]
        
        return schedule_df.reset_index(drop=True)
    


    def get_currency_pair_detailed_by_period(self, currency_pair: str, period: str, 
                                        data_type: str = 'future',
                                        try_all_fields: bool = True,
                                        verbose: bool = False) -> pd.DataFrame:
        """
        Get detailed economic data for a currency pair and time period with all available fields.
        
        Args:
            currency_pair (str): Currency pair like 'EURUSD', 'GBPJPY'
            period (str): Time period like '1w', '2W', '3m', '1M'
            data_type (str): Either 'future' or 'past'
            try_all_fields (bool): Whether to request all Bloomberg fields (True) or just basic fields (False)
            verbose (bool): Whether to show debug information (False for clean output)
            
        Returns:
            pd.DataFrame: Detailed schedule with all Bloomberg fields when available
        """
        days = self.parse_time_period(period)
        
        # Validate input
        if len(currency_pair) != 6:
            raise ValueError(f"Currency pair '{currency_pair}' must be exactly 6 characters")
        
        if data_type not in ['future', 'past']:
            raise ValueError("data_type must be either 'future' or 'past'")
        
        base_currency = currency_pair[:3].upper()
        quote_currency = currency_pair[3:].upper()
        
        # Validate currencies exist
        missing_currencies = []
        if base_currency not in self.indicators:
            missing_currencies.append(base_currency)
        if quote_currency not in self.indicators:
            missing_currencies.append(quote_currency)
        
        if missing_currencies:
            available = list(self.indicators.keys())
            raise ValueError(
                f"Currency(ies) {missing_currencies} not found. Available: {available}"
            )
        
        # Combine indicators from both currencies
        combined_indicators = []
        combined_indicators.extend(self.indicators[base_currency])
        combined_indicators.extend(self.indicators[quote_currency])
        
        # Get detailed data for each indicator
        detailed_results = []
        
        # Determine which fields to request
        if try_all_fields:
            custom_fields = self.DEFAULT_FIELDS  # Request all Bloomberg fields
            if verbose:
                print(f"Requesting all Bloomberg fields: {self.DEFAULT_FIELDS}")
        else:
            custom_fields = None  # Use default fields for data type
            if verbose:
                print(f"Using default fields for {data_type} data")
        
        for indicator in combined_indicators:
            try:
                # Choose the appropriate data retrieval method
                if data_type == 'future':
                    indicator_df = self.get_futureData(indicator["Ticker"], days, custom_fields=custom_fields)
                else:
                    indicator_df = self.get_pastData(indicator["Ticker"], days, custom_fields=custom_fields)
                
                if not indicator_df.empty:
                    # Add indicator metadata to each row
                    indicator_df['Currency_Pair'] = currency_pair
                    indicator_df['Country'] = indicator['Country']
                    indicator_df['Data'] = indicator['Data']
                    indicator_df['Ticker'] = indicator['Ticker']
                    
                    # Determine which currency this indicator belongs to
                    if indicator in self.indicators[base_currency]:
                        indicator_df['Currency'] = base_currency
                    else:
                        indicator_df['Currency'] = quote_currency
                    
                    # Add weekday and week information if ReleaseDate exists
                    if 'ReleaseDate' in indicator_df.columns:
                        indicator_df['Weekday'] = indicator_df['ReleaseDate'].dt.day_name().str[:3]
                        
                        # Add week grouping
                        today = pd.Timestamp.today().normalize()
                        start_of_this_week = today - pd.Timedelta(days=today.weekday())
                        indicator_df["Week"] = ((indicator_df["ReleaseDate"] - start_of_this_week).dt.days // 7).clip(lower=0)
                    
                    detailed_results.append(indicator_df)
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not retrieve data for {indicator['Ticker']}: {str(e)}")
                continue
        
        # Combine all results
        if detailed_results:
            final_df = pd.concat(detailed_results, ignore_index=True)
            
            # Reorder columns to match your desired output format
            base_cols = ['Currency_Pair', 'Currency', 'Country', 'Data', 'Ticker']
            
            # Add ReleaseDate if it exists (renamed from ReleaseDate to Release Date for display)
            if 'ReleaseDate' in final_df.columns:
                final_df = final_df.rename(columns={'ReleaseDate': 'Release Date'})
                base_cols.append('Release Date')
            
            # Add Weekday and Week if they exist
            if 'Weekday' in final_df.columns:
                base_cols.append('Weekday')
            if 'Week' in final_df.columns:
                base_cols.append('Week')
            
            # Add all the additional Bloomberg fields in logical order
            bloomberg_fields = ['ReleaseTime', 'Actual', 'SMed', 'SAve', 'SHigh', 'SLow', 'ForecastSDTv', 'NumbSurvey']
            available_bloomberg_fields = [col for col in bloomberg_fields if col in final_df.columns]
            
            # Combine all columns
            final_cols = base_cols + available_bloomberg_fields
            
            # Add any remaining columns that weren't explicitly listed
            other_cols = [col for col in final_df.columns if col not in final_cols]
            final_cols.extend(other_cols)
            
            # Only include columns that actually exist in the DataFrame
            available_final_cols = [col for col in final_cols if col in final_df.columns]
            final_df = final_df[available_final_cols]
            
            # Sort by release date if available
            if 'Release Date' in final_df.columns:
                final_df = final_df.sort_values('Release Date').reset_index(drop=True)
            
            # Print comprehensive debug information only if verbose=True
            if verbose:
                print(f"\n=== DEBUG INFO for {currency_pair} ({period}, {data_type}) ===")
                print(f"Total events found: {len(final_df)}")
                print(f"Available columns: {final_df.columns.tolist()}")
                
                # Check data availability for each Bloomberg field
                if not final_df.empty:
                    print("\nBloomberg field data availability:")
                    for field in bloomberg_fields:
                        if field in final_df.columns:
                            non_null_count = final_df[field].count()
                            total_count = len(final_df)
                            percentage = (non_null_count / total_count * 100) if total_count > 0 else 0
                            print(f"  {field}: {non_null_count}/{total_count} rows ({percentage:.1f}%) with data")
                            
                            # Show sample values if any exist
                            if non_null_count > 0:
                                sample_values = final_df[field].dropna().unique()[:3]
                                print(f"    Sample values: {sample_values}")
                        else:
                            print(f"  {field}: Column not available")
                    
                    # Show date range
                    if 'Release Date' in final_df.columns:
                        min_date = final_df['Release Date'].min()
                        max_date = final_df['Release Date'].max()
                        print(f"\nDate range: {min_date.date()} to {max_date.date()}")
                
                print("=" * 50)
            
            return final_df
        else:
            # Return empty DataFrame with expected structure
            empty_cols = ['Currency_Pair', 'Currency', 'Country', 'Data', 'Ticker', 'Release Date', 'Weekday', 'Week']
            empty_cols.extend(['ReleaseTime', 'Actual', 'SMed', 'SAve', 'SHigh', 'SLow', 'ForecastSDTv', 'NumbSurvey'])
            if verbose:
                print(f"No data found for {currency_pair} in the {period} period")
            return pd.DataFrame(columns=empty_cols)



    def get_enhanced_schedule_by_period(self, period: str, 
                                    currency_codes: list = None,
                                    include_detailed_fields: bool = True) -> pd.DataFrame:
        """
        Get enhanced economic release schedule with optional detailed Bloomberg fields.
        
        Args:
            period (str): Time period like '1w', '2W', '3m', '1M'
            currency_codes (list, optional): List of currency codes. If None, uses all currencies.
            include_detailed_fields (bool): Whether to include detailed Bloomberg fields
            
        Returns:
            pd.DataFrame: Enhanced schedule with optional detailed fields
        """
        days = self.parse_time_period(period)
        
        if currency_codes is None:
            # Get all currencies
            all_indicators = []
            for currency_indicators in self.indicators.values():
                all_indicators.extend(currency_indicators)
        else:
            # Get specific currencies
            all_indicators = []
            missing_currencies = []
            
            for currency_code in currency_codes:
                currency_code = currency_code.upper()
                if currency_code in self.indicators:
                    all_indicators.extend(self.indicators[currency_code])
                else:
                    missing_currencies.append(currency_code)
            
            if missing_currencies:
                available = list(self.indicators.keys())
                print(f"Warning: Currency(ies) {missing_currencies} not found. Available: {available}")
        
        if not all_indicators:
            empty_cols = ["Country", "Data", "Ticker", "Release Date", "Weekday", "Week"]
            if include_detailed_fields:
                empty_cols.extend(['ReleaseTime', 'Actual', 'SMed', 'SAve', 'SHigh', 'SLow', 'ForecastSDTv', 'NumbSurvey'])
            return pd.DataFrame(columns=empty_cols)
        
        # Get detailed data for each indicator if requested
        if include_detailed_fields:
            detailed_results = []
            
            for indicator in all_indicators:
                try:
                    # Use all DEFAULT_FIELDS to try to get more data
                    indicator_df = self.get_futureData(indicator["Ticker"], days, custom_fields=self.DEFAULT_FIELDS)
                    
                    if not indicator_df.empty:
                        # Add indicator metadata
                        indicator_df['Country'] = indicator['Country']
                        indicator_df['Data'] = indicator['Data']
                        indicator_df['Ticker'] = indicator['Ticker']
                        
                        # Add weekday and week information
                        if 'ReleaseDate' in indicator_df.columns:
                            indicator_df = indicator_df.rename(columns={'ReleaseDate': 'Release Date'})
                            indicator_df['Weekday'] = indicator_df['Release Date'].dt.day_name().str[:3]
                            
                            today = pd.Timestamp.today().normalize()
                            start_of_this_week = today - pd.Timedelta(days=today.weekday())
                            indicator_df["Week"] = ((indicator_df["Release Date"] - start_of_this_week).dt.days // 7).clip(lower=0)
                        
                        detailed_results.append(indicator_df)
                        
                except Exception as e:
                    print(f"Warning: Could not retrieve detailed data for {indicator['Ticker']}: {str(e)}")
                    continue
            
            if detailed_results:
                final_df = pd.concat(detailed_results, ignore_index=True)
                
                # Reorder columns - prioritize the important ones first
                base_cols = ['Country', 'Data', 'Ticker', 'Release Date', 'Weekday', 'Week']
                
                # Add Bloomberg fields in logical order
                bloomberg_fields = ['ReleaseTime', 'Actual', 'SMed', 'SAve', 'SHigh', 'SLow', 'ForecastSDTv', 'NumbSurvey']
                available_bloomberg_fields = [col for col in bloomberg_fields if col in final_df.columns]
                
                # Combine all priority columns
                final_cols = base_cols + available_bloomberg_fields
                
                # Add any remaining columns that weren't explicitly listed
                other_cols = [col for col in final_df.columns if col not in final_cols]
                final_cols.extend(other_cols)
                
                # Reorder the DataFrame with available columns only
                available_final_cols = [col for col in final_cols if col in final_df.columns]
                final_df = final_df[available_final_cols]
                
                if 'Release Date' in final_df.columns:
                    final_df = final_df.sort_values('Release Date').reset_index(drop=True)
                
                # Print debug info to see what fields are actually available
                print(f"Available columns: {final_df.columns.tolist()}")
                if not final_df.empty:
                    # Check which Bloomberg fields have actual data
                    data_summary = {}
                    for field in bloomberg_fields:
                        if field in final_df.columns:
                            non_null_count = final_df[field].count()
                            data_summary[field] = f"{non_null_count}/{len(final_df)} rows with data"
                    
                    if data_summary:
                        print("Bloomberg field data availability:")
                        for field, summary in data_summary.items():
                            print(f"  {field}: {summary}")
                
                return final_df
            else:
                empty_cols = ["Country", "Data", "Ticker", "Release Date", "Weekday", "Week"]
                empty_cols.extend(['ReleaseTime', 'Actual', 'SMed', 'SAve', 'SHigh', 'SLow', 'ForecastSDTv', 'NumbSurvey'])
                return pd.DataFrame(columns=empty_cols)
        else:
            # Use the existing schedule method
            return self.get_release_schedule(all_indicators, days)









    def preview_time_periods(self) -> pd.DataFrame:
        """
        Show what date ranges different period strings represent.
        
        Returns:
            pd.DataFrame: Preview of different time periods
        """
        periods = ['1w', '2w', '3w', '1m', '2m', '3m']
        today = datetime.today().date()
        
        preview_data = []
        for period in periods:
            try:
                days = self.parse_time_period(period)
                end_date = today + timedelta(days=days)
                preview_data.append({
                    'Period': period,
                    'Days': days,
                    'Start Date': today,
                    'End Date': end_date,
                    'Description': f"Next {period.replace('w', ' week(s)').replace('m', ' month(s)')}"
                })
            except Exception as e:
                preview_data.append({
                    'Period': period,
                    'Days': 'Error',
                    'Start Date': 'Error',
                    'End Date': 'Error',
                    'Description': str(e)
                })
        
        return pd.DataFrame(preview_data)


# Factory function to set up EconomicDataManager with predefined indicators
def setup_economic_data_manager(bloomberg_api):
    USD_indicators = [
        {"Country": "US", "Data": "CPI YoY", "Ticker": "CPI YOY Index"},
        {"Country": "US", "Data": "CPI MoM", "Ticker": "CPI CHNG Index"},
        {"Country": "US", "Data": "Core CPI YoY", "Ticker": "CPURNSA Index"},
        {"Country": "US", "Data": "Core CPI MoM", "Ticker": "CPUPXCHG Index"},
        {"Country": "US", "Data": "PPI MoM", "Ticker": "FDIDFDMO Index"},
        {"Country": "US", "Data": "PPI YoY", "Ticker": "FDIUFDYO Index"},
        {"Country": "US", "Data": "Unemployment Rate", "Ticker": "USURTOT Index"},
        {"Country": "US", "Data": "Nonfarm Payrolls (NFPs)", "Ticker": "NFP TCH Index"},
        {"Country": "US", "Data": "Initial Jobless Claims", "Ticker": "INJCJC Index"},
        {"Country": "US", "Data": "Continuing Claims", "Ticker": "INJCSP   Index"},
        {"Country": "US", "Data": "Fed Rate Decision", "Ticker": "FDTR Index"},
        {"Country": "US", "Data": "Retail Sales MoM", "Ticker": "RSTAMOM Index"},
        {"Country": "US", "Data": "Retail Sales Ex Auto MoM", "Ticker": "RSTAXMOM Index"},
        {"Country": "US", "Data": "GDP QoQ Annualized", "Ticker": "GDP CQOQ Index"},
        {"Country": "US", "Data": "Core PCE YoY", "Ticker": "PCE CYOY Index"},
        {"Country": "US", "Data": "Core PCE MoM", "Ticker": "PCE CMOM Index"},
        {"Country": "US", "Data": "University of Michigan Sentiment", "Ticker": "CONSSENT Index"},
        {"Country": "US", "Data": "ISM Manufacturing", "Ticker": "NAPMPMI Index"},
        {"Country": "US", "Data": "ISM Services", "Ticker": "NAPMNMI Index"},
        {"Country": "US", "Data": "Industrial Production MoM", "Ticker": "IPMGCHNG Index"},
        {"Country": "US", "Data": "Durable Goods Orders MoM", "Ticker": "DGNOCHNG Index"},
        {"Country": "US", "Data": "Housing Starts", "Ticker": "NHCHST Index"},
        {"Country": "US", "Data": "Building Permits", "Ticker": "NHSPATOT Index"},
        ]

    EUR_indicators = [
        {"Country": "EU", "Data": "CPI YoY", "Ticker": "ECCPEMUY Index"},
        {"Country": "EU", "Data": "Core CPI YoY", "Ticker": "CPEXEMUY Index"},
        {"Country": "EU", "Data": "CPI MoM", "Ticker": "ECCPEMUM Index"},
        {"Country": "EU", "Data": "GDP QoQ", "Ticker": "ECCPEMUM Index"},
        {"Country": "EU", "Data": "GDP YoY", "Ticker": "EUGNEMUY Index"},
        {"Country": "EU", "Data": "PPI MoM", "Ticker": "EUPPEMUM Index"},
        {"Country": "EU", "Data": "PPI Finished Goods", "Ticker": "EUPPEMUY Index"},
        {"Country": "EU", "Data": "Unemployment Rate", "Ticker": "UMRTEMU Index"},
        {"Country": "EU", "Data": "Employment YoY", "Ticker": "EMEMULYY Index"},
        {"Country": "EU", "Data": "Employment QoQ", "Ticker": "EMEMULQQ Index"},
        {"Country": "EU", "Data": "ECB Rate Decision", "Ticker": "EURR002W Index"},
        {"Country": "Germany", "Data": "IFO Gern Business Climate", "Ticker": "GRIFPBUS Index"}, 
        {"Country": "Germany", "Data": "ZEW Germ Eco Growth Expectations", "Ticker": "GRZEWI Index"}, 
        {"Country": "EU", "Data": "Consumer Confidence", "Ticker": "EUCCEMU Index"},
        {"Country": "EU", "Data": "Economic Confidence", "Ticker": "EUESEMU  Index"},
        {"Country": "EU", "Data": "Manufacturing PMI", "Ticker": "MPMIEZMA Index"},
        {"Country": "EU", "Data": "Services PMI", "Ticker": "MPMIEZSA Index"},
        {"Country": "EU", "Data": "Composite PMI", "Ticker": "MPMIEZCA Index"},
        {"Country": "EU", "Data": "Industrial Production MoM", "Ticker": "EUITEMUM Index"},
        {"Country": "EU", "Data": "Retail Sales YoY", "Ticker": "RSWAEMUY Index"},
        {"Country": "EU", "Data": "Trade Balance", "Ticker": "XTTBEZ Index"}
        ]

    JPY_indicators = [
        {"Country": "Japan", "Data": "CPI YoY", "Ticker": "JNCPIYOY Index"},
        {"Country": "Japan", "Data": "CPI Ex Fresh Food YoY (Core)", "Ticker": "JNCPIXFF Index"},
        {"Country": "Japan", "Data": "CPI Ex Fresh Food & Energy YoY (Core-Core)", "Ticker": "JCPTEFFE Index"},
        {"Country": "Japan", "Data": "PPI YoY", "Ticker": "JNWSDYOY Index"},
        {"Country": "Japan", "Data": "Unemployment Rate", "Ticker": "JNUE Index"},
        {"Country": "Japan", "Data": "Job-To-Applicant Ratio", "Ticker": "JBTARATE Index"},
        {"Country": "Japan", "Data": "Industrial Production YoY", "Ticker": "JNIPYOY Index"},
        {"Country": "Japan", "Data": "Tankan Large mfg Index", "Ticker": "JNTSMFG Index"},
        {"Country": "Japan", "Data": "Tankan Large mfg Outlook", "Ticker": "JPTFLMFG Index"},
        {"Country": "Japan", "Data": "S&P JPN PMI", "Ticker": "JPTFLMFG Index"},
        {"Country": "Japan", "Data": "Retail Sales MoM", "Ticker": "JNRETMOM Index"},
        {"Country": "Japan", "Data": "Trade Balance", "Ticker": "JNTBAL Index"},
        {"Country": "Japan", "Data": "Exports YoY", "Ticker": "JNTBEXPY Index"},
        {"Country": "Japan", "Data": "Imports YoY", "Ticker": "JNTBIMPY Index"},
        {"Country": "Japan", "Data": "BOJ Rate Decision", "Ticker": "BOJDTR Index"},
        {"Country": "Japan", "Data": "GDP Annualized QoQ", "Ticker": "JGDPQGDP Index"},  
        ]

    GBP_indicators = [
        {"Country": "UK", "Data": "GDP QOQ", "Ticker": "UKGRABIQ Index"},
        {"Country": "UK", "Data": "CPI YoY", "Ticker": "UKRPCJYR Index"},
        {"Country": "UK", "Data": "CPI MoM", "Ticker": "UKRPCJMR Index"},
        {"Country": "UK", "Data": "Core CPI YoY", "Ticker": "UKHCA9IQ Index"},
        {"Country": "UK", "Data": "RPI YoY", "Ticker": "UKRPYOY  Index"},
        {"Country": "UK", "Data": "Unemployment Claims", "Ticker": "UKUEMOM Index"},
        {"Country": "UK", "Data": "Retail Sales MoM", "Ticker": "UKUEMOM Index"},
        {"Country": "UK", "Data": "Industrial Production YoY", "Ticker": "UKMPIYOY Index"},
        {"Country": "UK", "Data": "Gfk Consumer Confidence", "Ticker": "UKCCI Index"},
        {"Country": "UK", "Data": "Manufacturing PMI", "Ticker": "UKCCI Index"},
        {"Country": "UK", "Data": "Service PMI", "Ticker": "MPMIGBSA Index"},
        {"Country": "UK", "Data": "Composite PMI", "Ticker": "MPMIGBCA Index"},
        {"Country": "UK", "Data": "Trade Balance", "Ticker": "MPMIGBCA Index"},
        {"Country": "UK", "Data": "BoE Rate Decision", "Ticker": "UKBRBASE Index"},
    ]

    AUD_indicators = [
        {"Country": "AU", "Data": "CPI YoY", "Ticker": "AUCPIYOY Index"},
        {"Country": "AU", "Data": "CPI QoQ", "Ticker": "AUCPICHG Index"},
        {"Country": "AU", "Data": "Trimmed Mean CPI YoY", "Ticker": "RBCPTRIY Index"},
        {"Country": "AU", "Data": "Trimmed Mean CPI QoQ", "Ticker": "RBCPTRIQ Index"},
        {"Country": "AU", "Data": "Melbrn Infl Gauge QoQ", "Ticker": "TDMIMOM Index"},
        {"Country": "AU", "Data": "Melbrn Cnsmr Confi MoM", "Ticker": "TDMIMOM Index"},
        {"Country": "AU", "Data": "Unemployment Rate", "Ticker": "AULFUNEM Index"},
        {"Country": "AU", "Data": "Employment Change", "Ticker": "AULFEMPC Index"},
        {"Country": "AU", "Data": "Participation Rate", "Ticker": "AULFPART Index"},
        {"Country": "AU", "Data": "Industrial Production YoY", "Ticker": "AUIPTYOY Index"},
        {"Country": "AU", "Data": "Retail Sales MoM", "Ticker": "AURSTSA Index"},
        {"Country": "AU", "Data": "Trade Balance", "Ticker": "AUITGSB Index"},
        {"Country": "AU", "Data": "RBA CAsh Rate", "Ticker": "RBATCTR  Index"},
        {"Country": "AU", "Data": "RBA Cash Rate Target", "Ticker": "RBATCTR Index"},
        {"Country": "AU", "Data": "GDP QoQ", "Ticker": "AUNAGDPC Index"},
        {"Country": "AU", "Data": "GDP YoY", "Ticker": "AUNAGDPY Index"},
    ]

    NZD_indicators = [
        {"Country": "NZ", "Data": "CPI YoY", "Ticker": "NZCPICHG Index"},
        {"Country": "NZ", "Data": "CPI QoQ", "Ticker": "NZCPIYOY Index"},
        {"Country": "NZ", "Data": "Unemployment Rate", "Ticker": "NZLFUNER Index"},
        {"Country": "NZ", "Data": "Employment Change QoQ", "Ticker": "NZLFQOQ  Index"},
        {"Country": "NZ", "Data": "GDP QoQ", "Ticker": "NZNTGDPC Index"},
        {"Country": "NZ", "Data": "GDP YoY", "Ticker": "NZNTGDPY Index"},
        {"Country": "NZ", "Data": "Retail Sales Ex Inflation QoQ", "Ticker": "NZRREXIN Index"},
        {"Country": "NZ", "Data": "Trade Balance", "Ticker": "NZMTBAL Index"},
        {"Country": "NZ", "Data": "Exports", "Ticker": "NZMTEXP Index"},
        {"Country": "NZ", "Data": "Imports", "Ticker": "NZMTIMP Index"},
        {"Country": "NZ", "Data": "RBNZ Official Cash Rate", "Ticker": "NZOCR Index"},
        {"Country": "NZ", "Data": "Business Mnft PMI", "Ticker": "NZPMISA Index"},
        {"Country": "NZ", "Data": "Consumer Confidence", "Ticker": "NZANCCT Index"},
    ]

    CHF_indicators = [
        {"Country": "CHF", "Data": "CPI YoY", "Ticker": "SZCPIYOY Index"},
        {"Country": "CHF", "Data": "CPI MoM", "Ticker": "SZCPIMOM Index"},
        {"Country": "CHF", "Data": "Unemployment Rate", "Ticker": "SZUE Index"},
        {"Country": "CHF", "Data": "GDP QoQ", "Ticker": "SZGDPCQQ Index"},
        {"Country": "CHF", "Data": "GDP YoY", "Ticker": "SZGRGDPY Index"},
        {"Country": "CHF", "Data": "SNB Policy Rate", "Ticker": "SZLTDEP Index"},
        {"Country": "CHF", "Data": "CB Foreign Reserves", "Ticker": "SZRAFCRC Index"},
        {"Country": "CHF", "Data": "Retail Sales YoY", "Ticker": "SZRSRYOY Index"},
        {"Country": "CHF", "Data": "PMI Manufacturing", "Ticker": "SZPUI Index"},
    ]

    manager = EconomicDataManager(bloomberg_api)
    manager.add_indicators('USD', USD_indicators)
    manager.add_indicators('EUR', EUR_indicators)
    manager.add_indicators('JPY', JPY_indicators)
    manager.add_indicators('GBP', GBP_indicators)
    manager.add_indicators('AUD', AUD_indicators)
    manager.add_indicators('NZD', NZD_indicators)
    manager.add_indicators('CHF', CHF_indicators)

    return manager


# eco_manager = setup_economic_data_manager(blp)
# df_next_releases = eco_manager.get_currency_schedule('USD', days=10)
# print(df_next_releases)



















































