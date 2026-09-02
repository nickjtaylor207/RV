from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import percentileofscore
from xbbg import blp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pandas.tseries.offsets import DateOffset
import re

# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------- ECO DATA SETUP ---------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

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
    def __init__(self, bloomberg_api, indicators_dict: Optional[Dict] = None):
        self.blp = bloomberg_api
        self.indicators = indicators_dict or {}

    def add_indicators(self, currency_code: str, indicators: List[Dict]) -> None:
        self.indicators[currency_code] = indicators

    def parse_time_period(self, period_str: str) -> int:
        if not isinstance(period_str, str):
            raise ValueError("Period must be a string")
        period_str = period_str.strip().lower()
        match = re.match(r'^(\d+)([wm])$', period_str)
        if not match:
            raise ValueError(
                f"Invalid period format: '{period_str}'. "
                f"Use format like '1w', '2W', '3m', '1M' (w=weeks, m=months)")
        number = int(match.group(1))
        unit = match.group(2)
        if unit == 'w':  
            return number * 7
        elif unit == 'm':  
            return number * 30
        else:
            raise ValueError(f"Unsupported time unit: {unit}")
        
    def _add_week_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        today = pd.Timestamp.today().normalize()
        start_of_this_week = today - pd.Timedelta(days=today.weekday())
        df["Week"] = ((df["Release Date"] - start_of_this_week).dt.days // 7).clip(lower=0)
        return df
    
    # ------------------------------------------------------------------------------------------
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
        if failed_tickers:
            print(f"Could not get release dates for {len(failed_tickers)} tickers: {failed_tickers[:5]}{'...' if len(failed_tickers) > 5 else ''}")
        if not results:
            print("No release dates found for any indicators")
            return pd.DataFrame(columns=["Country", "Data", "Ticker", "Release Date", "Weekday", "Week"])
        df = pd.DataFrame(results)
        print(f"Found {len(df)} release dates out of {len(indicators)} indicators")
        df["Release Date"] = pd.to_datetime(df["Release Date"], errors='coerce')
        df = df.dropna(subset=["Release Date"])
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



    def get_all_currencies_schedule(self, days: int = 100) -> pd.DataFrame:
        all_indicators = []
        for currency_indicators in self.indicators.values():
            all_indicators.extend(currency_indicators)
        return self.get_release_schedule(all_indicators, days)


    def get_schedule_by_period(self, period: str, currency_codes: list = None) -> pd.DataFrame:
        days = self.parse_time_period(period)
        if currency_codes is None:
            return self.get_all_currencies_schedule(days)
        else:
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
        {"Country": "US", "Data": "Building Permits", "Ticker": "NHSPATOT Index"},]
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
        {"Country": "EU", "Data": "Trade Balance", "Ticker": "XTTBEZ Index"}]
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
        {"Country": "Japan", "Data": "GDP Annualized QoQ", "Ticker": "JGDPQGDP Index"},]
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
        {"Country": "UK", "Data": "BoE Rate Decision", "Ticker": "UKBRBASE Index"},]
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
        {"Country": "AU", "Data": "GDP YoY", "Ticker": "AUNAGDPY Index"},]
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
        {"Country": "NZ", "Data": "Consumer Confidence", "Ticker": "NZANCCT Index"},]
    CHF_indicators = [
        {"Country": "CHF", "Data": "CPI YoY", "Ticker": "SZCPIYOY Index"},
        {"Country": "CHF", "Data": "CPI MoM", "Ticker": "SZCPIMOM Index"},
        {"Country": "CHF", "Data": "Unemployment Rate", "Ticker": "SZUE Index"},
        {"Country": "CHF", "Data": "GDP QoQ", "Ticker": "SZGDPCQQ Index"},
        {"Country": "CHF", "Data": "GDP YoY", "Ticker": "SZGRGDPY Index"},
        {"Country": "CHF", "Data": "SNB Policy Rate", "Ticker": "SZLTDEP Index"},
        {"Country": "CHF", "Data": "CB Foreign Reserves", "Ticker": "SZRAFCRC Index"},
        {"Country": "CHF", "Data": "Retail Sales YoY", "Ticker": "SZRSRYOY Index"},
        {"Country": "CHF", "Data": "PMI Manufacturing", "Ticker": "SZPUI Index"},]

    manager = EconomicDataManager(bloomberg_api)
    manager.add_indicators('USD', USD_indicators)
    manager.add_indicators('EUR', EUR_indicators)
    manager.add_indicators('JPY', JPY_indicators)
    manager.add_indicators('GBP', GBP_indicators)
    manager.add_indicators('AUD', AUD_indicators)
    manager.add_indicators('NZD', NZD_indicators)
    manager.add_indicators('CHF', CHF_indicators)

    return manager















# # --------------------------------------------------------------------------------------------------------------------
# # ------------------------------------- EVENT CALANDER MANAGEMENT ----------------------------------------------------
# # --------------------------------------------------------------------------------------------------------------------


class FXEventCalendar:
    def __init__(self):
        self.events = {}
    def add_event(self, event_name: str, event_date: datetime, event_type: str = 'economic', 
                  affected_ccys: List[str] = None):
        self.events[event_name] = {
            'date': event_date,
            'type': event_type,
            'affected_ccys': affected_ccys or []}
    def get_events_in_range(self, start_date: datetime, end_date: datetime):
        return {
            name: info for name, info in self.events.items()
            if start_date <= info['date'] <= end_date}
    def get_days_to_events(self, base_date: datetime = None):
        if base_date is None:
            base_date = datetime.now()
        return {
            name: (info['date'] - base_date).days 
            for name, info in self.events.items()
            if info['date'] >= base_date}
    def get_events_by_currency(self, ccy: str):
        return {
            name: info for name, info in event_calendar.events.items()
            if ccy in info['affected_ccys'] or not info['affected_ccys']
        }





# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------- EVENT TYPE CLASSIFICATION ----------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class EventClassifier:
    """Classify economic indicators into event types and determine affected currencies"""
    CENTRAL_BANK_KEYWORDS = [
        'rate decision', 'rate', 'cash rate', 'policy rate', 'official cash rate',
        'fed', 'ecb', 'boj', 'rba', 'rbnz', 'boe', 'snb', 'fomc']
    
    HIGH_IMPACT_KEYWORDS = [
        'cpi', 'inflation', 'gdp', 'employment', 'nonfarm payrolls', 'nfp',
        'unemployment', 'retail sales', 'pmi', 'trade balance']
    
    CURRENCY_TO_PAIRS = {
        'USD': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF'],
        'EUR': ['EURUSD', 'EURGBP', 'EURJPY', 'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF'],
        'GBP': ['GBPUSD', 'EURGBP', 'GBPJPY', 'GBPAUD', 'GBPNZD', 'GBPCAD', 'GBPCHF'],
        'JPY': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY'],
        'AUD': ['AUDUSD', 'EURAUD', 'GBPAUD', 'AUDJPY', 'AUDNZD', 'AUDCAD', 'AUDCHF'],
        'NZD': ['NZDUSD', 'EURNZD', 'GBPNZD', 'NZDJPY', 'AUDNZD', 'NZDCAD', 'NZDCHF'],
        'CAD': ['USDCAD', 'EURCAD', 'GBPCAD', 'CADJPY', 'AUDCAD', 'NZDCAD'],
        'CHF': ['USDCHF', 'EURCHF', 'GBPCHF', 'CHFJPY', 'AUDCHF', 'NZDCHF']}
    
    @staticmethod
    def classify_event_type(indicator_name: str) -> str:
        """Classify an economic indicator into event type"""
        indicator_lower = indicator_name.lower()
        for keyword in EventClassifier.CENTRAL_BANK_KEYWORDS:
            if keyword in indicator_lower:
                return 'central_bank'
        for keyword in EventClassifier.HIGH_IMPACT_KEYWORDS:
            if keyword in indicator_lower:
                return 'high_impact'
        return 'economic'
    

    @staticmethod
    def get_affected_currency_pairs(currency_code: str, 
                                   available_pairs: Optional[List[str]] = None) -> List[str]:
        """Get list of currency pairs affected by a currency's economic data"""
        currency_code = currency_code.upper()
        if currency_code not in EventClassifier.CURRENCY_TO_PAIRS:
            return []
        affected = EventClassifier.CURRENCY_TO_PAIRS[currency_code]
        if available_pairs:
            affected = [pair for pair in affected if pair in available_pairs]
        return affected
    
    @staticmethod
    def get_event_importance(event_type: str, indicator_name: str) -> int:
        """Assign importance score (1-10) based on event type and name"""
        indicator_lower = indicator_name.lower()
        if event_type == 'central_bank':
            return 10
        tier1_keywords = ['nonfarm payrolls', 'nfp', 'cpi', 'gdp']
        if any(kw in indicator_lower for kw in tier1_keywords):
            return 9
        tier2_keywords = ['unemployment', 'retail sales', 'pmi', 'inflation']
        if any(kw in indicator_lower for kw in tier2_keywords):
            return 7
        tier3_keywords = ['trade balance', 'industrial production', 'housing']
        if any(kw in indicator_lower for kw in tier3_keywords):
            return 5
        return 3


def _map_country_to_currency(country: str) -> Optional[str]:
    """Map country name/code to currency code"""
    mapping = {
        'US': 'USD', 'USA': 'USD',
        'EU': 'EUR', 'EUR': 'EUR', 'EURO': 'EUR', 'Germany': 'EUR',
        'Japan': 'JPY', 'JP': 'JPY',
        'UK': 'GBP', 'GB': 'GBP', 'United Kingdom': 'GBP',
        'AU': 'AUD', 'Australia': 'AUD',
        'NZ': 'NZD', 'New Zealand': 'NZD',
        'CA': 'CAD', 'Canada': 'CAD',
        'CH': 'CHF', 'CHF': 'CHF', 'Switzerland': 'CHF'
    }
    return mapping.get(country.strip(), None)


# --------------------------------------------------------------------------------------------------------------------
# -------------------------------- BBG DATA GATHER -------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


def populate_calendar_from_bloomberg(economic_data_manager,
                                    event_calendar,
                                    period: str = '3m',
                                    currency_codes: Optional[List[str]] = None,
                                    available_pairs: Optional[List[str]] = None,
                                    importance_threshold: int = 0) -> Dict:
    """
    Populate FXEventCalendar with events from EconomicDataManager
    """
    stats = {
        'total_indicators_processed': 0,
        'events_added': 0,
        'events_skipped_no_date': 0,
        'events_skipped_low_importance': 0,
        'events_by_type': {},
        'events_by_currency': {}}
    try:
        if currency_codes:
            schedule_df = economic_data_manager.get_schedule_by_period(
                period=period,
                currency_codes=currency_codes)
        else:
            schedule_df = economic_data_manager.get_all_currencies_schedule(
                days=economic_data_manager.parse_time_period(period))
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return stats
    if schedule_df.empty:
        print("No events found in the specified period")
        return stats
    for idx, row in schedule_df.iterrows():
        stats['total_indicators_processed'] += 1
        try:
            event_date = pd.to_datetime(row['Release Date'])
            country = row['Country']
            indicator_name = row['Data']
            ticker = row['Ticker']
        except Exception as e:
            stats['events_skipped_no_date'] += 1
            continue
        if pd.isna(event_date):
            stats['events_skipped_no_date'] += 1
            continue
        event_type = EventClassifier.classify_event_type(indicator_name)
        importance = EventClassifier.get_event_importance(event_type, indicator_name)
        if importance < importance_threshold:
            stats['events_skipped_low_importance'] += 1
            continue
        currency_code = _map_country_to_currency(country)
        if not currency_code:
            continue
        affected_pairs = EventClassifier.get_affected_currency_pairs(
            currency_code, 
            available_pairs)
        event_name = f"{country} {indicator_name}"
        event_calendar.add_event(
            event_name=event_name,
            event_date=event_date,
            event_type=event_type,
            affected_ccys=affected_pairs)
        stats['events_added'] += 1
        stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
        stats['events_by_currency'][currency_code] = stats['events_by_currency'].get(currency_code, 0) + 1
    return stats


def get_granular_vol_data(currency_list: List[str], days_back: int = 90):
    """Fetch implied volatility across granular tenor structure"""
    tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '6M', '9M', '1Y']
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    data_dict = {}
    for tenor in tenors:
        for ccy in currency_list:
            ticker = f"{ccy}V{tenor} Curncy"
            try:
                df = blp.bdh(
                    tickers=ticker,
                    flds="PX_LAST",
                    start_date=start_date,
                    end_date=end_date)
                df.columns = df.columns.get_level_values(1)
                series = df["PX_LAST"].astype(float).dropna()
                data_dict[f"{ccy}_{tenor}"] = series
            except Exception as e:
                continue
    result = pd.DataFrame(data_dict)
    return result



# --------------------------------------------------------------------------------------------------------------------
# -------------------------------- FWD VOL CALCULATIONS --------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------



def tenor_to_days(tenor: str) -> int:
    """Convert tenor string to approximate days"""
    mapping = {
        'O/N': 1, 'T/N': 2,
        '1W': 7, '2W': 14, '3W': 21,
        '1M': 30, '2M': 60, '3M': 90,
        '6M': 180, '9M': 270, '1Y': 365}
    return mapping.get(tenor, 30)


def calculate_forward_vol(vol_short: float, tenor_short_days: int, 
                         vol_long: float, tenor_long_days: int) -> float:
    """Calculate forward starting volatility between two tenors"""
    if pd.isna(vol_short) or pd.isna(vol_long):
        return np.nan
    if tenor_long_days <= tenor_short_days:
        return np.nan
    T1 = tenor_short_days / 365
    T2 = tenor_long_days / 365
    var_short = (vol_short / 100) ** 2 * T1
    var_long = (vol_long / 100) ** 2 * T2
    forward_var = (var_long - var_short) / (T2 - T1)
    if forward_var <= 0:
        return np.nan
    forward_vol = np.sqrt(forward_var) * 100
    return forward_vol


def calculate_all_forward_vols(vol_data: pd.DataFrame, ccy: str, tenors: List[str]) -> pd.DataFrame:
    """Calculate all forward volatilities for a currency pair"""
    latest_date = vol_data.index[-1]
    forward_matrix = []
    for i, tenor1 in enumerate(tenors):
        row = []
        for j, tenor2 in enumerate(tenors):
            if j <= i:
                row.append(np.nan)
            else:
                col1 = f"{ccy}_{tenor1}"
                col2 = f"{ccy}_{tenor2}"
                if col1 in vol_data.columns and col2 in vol_data.columns:
                    vol1 = vol_data.loc[latest_date, col1] if latest_date in vol_data.index else np.nan
                    vol2 = vol_data.loc[latest_date, col2] if latest_date in vol_data.index else np.nan
                    days1 = tenor_to_days(tenor1)
                    days2 = tenor_to_days(tenor2)
                    fwd_vol = calculate_forward_vol(vol1, days1, vol2, days2)
                    row.append(fwd_vol)
                else:
                    row.append(np.nan)
        forward_matrix.append(row)
    df = pd.DataFrame(forward_matrix, index=tenors, columns=tenors)
    return df





def analyze_event_risk_premium(vol_data: pd.DataFrame, ccy: str, 
                               event_calendar: FXEventCalendar,
                               tenors: List[str]) -> pd.DataFrame:
    """Analyze which events have the highest implied vol premium"""
    latest_date = vol_data.index[-1]
    base_datetime = pd.to_datetime(latest_date)
    all_events = event_calendar.get_events_by_currency(ccy)
    future_events = {
        name: info for name, info in all_events.items()
        if info['date'] >= base_datetime}
    days_to_events = {
        name: (info['date'] - base_datetime).days
        for name, info in future_events.items()}
    avg_vols = {}
    current_vols = {}
    for tenor in tenors:
        col = f"{ccy}_{tenor}"
        if col in vol_data.columns:
            avg_vols[tenor] = vol_data[col].mean()
            current_vols[tenor] = vol_data.loc[latest_date, col] if latest_date in vol_data.index else np.nan
    event_analysis = []
    for event_name, days_out in days_to_events.items():
        tenor_days = {t: tenor_to_days(t) for t in tenors}
        before_tenor = None
        after_tenor = None
        for tenor, t_days in tenor_days.items():
            if t_days <= days_out:
                if before_tenor is None or t_days > tenor_to_days(before_tenor):
                    before_tenor = tenor
            if t_days >= days_out:
                if after_tenor is None or t_days < tenor_to_days(after_tenor):
                    after_tenor = tenor
        if before_tenor and after_tenor and before_tenor != after_tenor:
            vol_before = current_vols.get(before_tenor, np.nan)
            vol_after = current_vols.get(after_tenor, np.nan)
            avg_before = avg_vols.get(before_tenor, np.nan)
            avg_after = avg_vols.get(after_tenor, np.nan)
            fwd_vol = calculate_forward_vol(
                vol_before, tenor_to_days(before_tenor),
                vol_after, tenor_to_days(after_tenor))
            baseline = (avg_before + avg_after) / 2 if not (pd.isna(avg_before) or pd.isna(avg_after)) else np.nan
            premium = fwd_vol - baseline if not (pd.isna(fwd_vol) or pd.isna(baseline)) else np.nan
            premium_pct = (premium / baseline * 100) if not pd.isna(baseline) and baseline != 0 else np.nan
            event_analysis.append({
                'Event': event_name,
                'Days_Out': days_out,
                'Before_Tenor': before_tenor,
                'After_Tenor': after_tenor,
                'Forward_Vol': fwd_vol,
                'Historical_Avg': baseline,
                'Premium_Vol_Points': premium,
                'Premium_Pct': premium_pct,
                'Current_Vol_Before': vol_before,
                'Current_Vol_After': vol_after})
    df = pd.DataFrame(event_analysis)
    if not df.empty:
        df = df.sort_values('Days_Out')
    return df






def plot_vol_term_structure_with_events(vol_data: pd.DataFrame, ccy: str,
                                       event_calendar: FXEventCalendar,
                                       tenors: List[str],
                                       show_forward_vols: bool = True):
    """Plot volatility term structure with event markers and forward vols"""
    latest_date = vol_data.index[-1]
    base_datetime = pd.to_datetime(latest_date)
    
    current_vols = []
    tenor_days = []
    
    for tenor in tenors:
        col = f"{ccy}_{tenor}"
        if col in vol_data.columns:
            vol = vol_data.loc[latest_date, col] if latest_date in vol_data.index else np.nan
            if not pd.isna(vol):
                current_vols.append(vol)
                tenor_days.append(tenor_to_days(tenor))
    
    if not current_vols:
        print(f"No data available for {ccy}")
        return
    
    all_events = event_calendar.get_events_by_currency(ccy)
    future_events = {
        name: info for name, info in all_events.items()
        if info['date'] >= base_datetime
}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(tenor_days, current_vols, 'o-', linewidth=2.5, markersize=10, 
            color='#2E86AB', label=f'{ccy} ATM Implied Vol')
    avg_vols = []
    for tenor in tenors:
        col = f"{ccy}_{tenor}"
        if col in vol_data.columns:
            avg = vol_data[col].mean()
            if not pd.isna(avg):
                avg_vols.append(avg)
            else:
                avg_vols.append(None)
    valid_avg = [v for v in avg_vols if v is not None]
    if valid_avg:
        ax1.plot(tenor_days[:len(valid_avg)], valid_avg, '--', linewidth=1.5, 
                color='gray', alpha=0.6, label='Historical Average')
    colors_by_type = {
        'central_bank': '#E63946',
        'high_impact': '#F77F00',
        'economic': '#06A77D',
        'political': '#06FFA5',
        'other': '#9D4EDD'}
    max_vol = max(current_vols)
    min_vol = min(current_vols)
    vol_range = max_vol - min_vol
    for event_name, info in future_events.items():
        days_out = (info['date'] - base_datetime).days
        if days_out <= max(tenor_days):
            color = colors_by_type.get(info['type'], 'gray')
            ax1.axvline(days_out, color=color, linestyle='--', linewidth=2, alpha=0.7)
            ax1.text(days_out, max_vol + vol_range * 0.02, event_name,
                    rotation=45, va='bottom', ha='left', fontsize=9,
                    color=color, fontweight='bold')
    ax1.set_xlabel('Days to Expiry', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{ccy} Volatility Term Structure with Event Risk Premium', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='best', fontsize=10)
    if show_forward_vols:
        forward_vols = []
        forward_labels = []
        forward_days = []
        for i in range(len(tenors) - 1):
            tenor1 = tenors[i]
            tenor2 = tenors[i + 1]
            col1 = f"{ccy}_{tenor1}"
            col2 = f"{ccy}_{tenor2}"
            if col1 in vol_data.columns and col2 in vol_data.columns:
                vol1 = vol_data.loc[latest_date, col1] if latest_date in vol_data.index else np.nan
                vol2 = vol_data.loc[latest_date, col2] if latest_date in vol_data.index else np.nan
                days1 = tenor_to_days(tenor1)
                days2 = tenor_to_days(tenor2)
                fwd_vol = calculate_forward_vol(vol1, days1, vol2, days2)
                if not pd.isna(fwd_vol):
                    forward_vols.append(fwd_vol)
                    forward_labels.append(f"{tenor1}x{tenor2}")
                    forward_days.append((days1 + days2) / 2)
        if forward_vols:
            bars = ax2.bar(range(len(forward_vols)), forward_vols, 
                          color='#06A77D', alpha=0.7, edgecolor='black', linewidth=1.5)
            ax2.set_xticks(range(len(forward_vols)))
            ax2.set_xticklabels(forward_labels, rotation=45, ha='right')
            ax2.set_ylabel('Forward Vol (%)', fontsize=11, fontweight='bold')
            ax2.set_title('Forward Starting Volatilities', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
            for idx, (label, days) in enumerate(zip(forward_labels, forward_days)):
                for event_name, info in future_events.items():
                    event_days = (info['date'] - base_datetime).days
                    tenor1, tenor2 = label.split('x')
                    days1 = tenor_to_days(tenor1)
                    days2 = tenor_to_days(tenor2)
                    if days1 < event_days <= days2:
                        bars[idx].set_color('#E63946')
                        bars[idx].set_alpha(0.9)
    legend_elements = [
        mpatches.Patch(color=colors_by_type['central_bank'], label='Central Bank'),
        mpatches.Patch(color=colors_by_type['high_impact'], label='High Impact'),
        mpatches.Patch(color=colors_by_type['economic'], label='Economic Data')]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + legend_elements, 
              loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig






def compare_event_risk_across_currencies(currency_list: List[str],
                                        event_calendar: FXEventCalendar,
                                        days_back: int = 90):
    """Compare how different currencies price the same events"""
    tenors = ['1W', '2W', '3W', '1M', '2M', '3M']
    
    vol_data = get_granular_vol_data(currency_list, days_back)
    all_events = event_calendar.events
    event_comparison = {}
    for event_name, event_info in all_events.items():
        base_datetime = pd.to_datetime(vol_data.index[-1])
        days_out = (event_info['date'] - base_datetime).days
        if days_out < 0:
            continue
        ccy_premiums = {}
        for ccy in currency_list:
            event_risk_df = analyze_event_risk_premium(vol_data, ccy, event_calendar, tenors)
            if not event_risk_df.empty:
                event_row = event_risk_df[event_risk_df['Event'] == event_name]
                if not event_row.empty:
                    premium = event_row.iloc[0]['Premium_Pct']
                    fwd_vol = event_row.iloc[0]['Forward_Vol']
                    ccy_premiums[ccy] = {
                        'Premium_Pct': premium,
                        'Forward_Vol': fwd_vol}
        if ccy_premiums:
            event_comparison[event_name] = ccy_premiums
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    for event_name, ccy_data in event_comparison.items():
        ccys = list(ccy_data.keys())
        premiums = [ccy_data[ccy]['Premium_Pct'] for ccy in ccys]
        x_pos = np.arange(len(ccys))
        ax1.plot(x_pos, premiums, marker='o', linewidth=2, markersize=8, label=event_name)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ccys, rotation=45)
    ax1.set_ylabel('Vol Premium (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Event Risk Premium by Currency', fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.8)
    event_names = list(event_comparison.keys())
    all_ccys = currency_list
    heatmap_data = np.zeros((len(event_names), len(all_ccys)))
    for i, event in enumerate(event_names):
        for j, ccy in enumerate(all_ccys):
            if ccy in event_comparison[event]:
                heatmap_data[i, j] = event_comparison[event][ccy]['Premium_Pct']
            else:
                heatmap_data[i, j] = np.nan
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
    ax2.set_xticks(np.arange(len(all_ccys)))
    ax2.set_yticks(np.arange(len(event_names)))
    ax2.set_xticklabels(all_ccys, rotation=45, ha='right')
    ax2.set_yticklabels(event_names, fontsize=8)
    ax2.set_title('Event Vol Premium Heatmap (%)', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Premium %', fontsize=10)
    for i in range(len(event_names)):
        for j in range(len(all_ccys)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax2.text(j, i, f'{heatmap_data[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.show()
    return event_comparison







def generate_event_vol_report(currency_list: List[str], 
                             event_calendar: FXEventCalendar,
                             days_back: int = 90):
    """Generate comprehensive event volatility analysis across all currencies"""
    tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '6M']
    print("Fetching volatility data...")
    vol_data = get_granular_vol_data(currency_list, days_back)
    all_results = {}
    for ccy in currency_list:
        print(f"\nAnalyzing {ccy}...")
        event_risk_df = analyze_event_risk_premium(vol_data, ccy, event_calendar, tenors)
        forward_matrix = calculate_all_forward_vols(vol_data, ccy, tenors)
        all_results[ccy] = {
            'event_risk': event_risk_df,
            'forward_matrix': forward_matrix}
        if not event_risk_df.empty:
            print(f"\n{ccy} - Top Events by Vol Premium:")
            print(event_risk_df[['Event', 'Days_Out', 'Forward_Vol', 
                               'Historical_Avg', 'Premium_Pct']].head())
        plot_vol_term_structure_with_events(vol_data, ccy, event_calendar, tenors)
    return all_results, vol_data


def print_event_calendar_summary(event_calendar, currency_list: List[str]):
    """Print a formatted summary of the event calendar"""
    all_events = event_calendar.events
    if not all_events:
        print("   No events in calendar")
        return
    for ccy in currency_list:
        ccy_events = event_calendar.get_events_by_currency(ccy)
        if not ccy_events:
            continue
        print(f"\n   {ccy}:")
        sorted_events = sorted(
            ccy_events.items(),
            key=lambda x: x[1]['date'])
        for i, (name, info) in enumerate(sorted_events[:5]):
            days_out = (info['date'] - datetime.now()).days
            event_type_icon = {
                'central_bank': '🏦',
                'high_impact': '⚠️',
                'economic': '📊',
                'other': '📌'
            }.get(info['type'], '📌')
            print(f"      {event_type_icon} {name}: {info['date'].strftime('%Y-%m-%d')} ({days_out} days)")









tenors = ['1W', '2W', '3W', '1M']
ccy_int = ['EUR']
days_back=180


eco_manager = setup_economic_data_manager(blp)
event_calendar = FXEventCalendar()

# Populate for EUR only
populate_calendar_from_bloomberg(eco_manager, event_calendar, 
                    period='2w', 
                    currency_codes=ccy_int,
                    importance_threshold = 0)



# vol_data = get_granular_vol_data(ccy_int, days_back)
# # results = analyze_event_risk_premium(vol_data, ccy_int[0], event_calendar, tenors)
# # plot_vol_term_structure_with_events(vol_data, ccy_int[0],
# #                                        event_calendar,
# #                                        tenors,
# #                                        show_forward_vols = True)



# event_comparison = compare_event_risk_across_currencies(
#             currency_list=ccy_int,
#             event_calendar=event_calendar,
#             days_back=days_back
#         )