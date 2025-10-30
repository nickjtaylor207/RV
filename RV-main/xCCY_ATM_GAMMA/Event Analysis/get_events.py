import pandas as pd
import numpy as np
from xbbg import blp

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
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                data.columns = data.columns.droplevel(0)
            data = data.rename(columns=self.FIELD_RENAME_MAP)
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





