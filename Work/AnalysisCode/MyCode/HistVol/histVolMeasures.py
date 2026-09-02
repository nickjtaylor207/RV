import pdblp

import pandas as pd
import numpy as np
import pytz

import seaborn as sns
import matplotlib.pyplot as plt

from xbbg import blp
from datetime import datetime, timedelta

from pandas.tseries.offsets import DateOffset

# -------------------------------------------------------------------------------------------------------------------
# ---------------------------------------Get Spot Data --------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------


# Get Daily Spot Data
def getDailySpot(ticker, days):
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    ticker_IV = f"{ticker} BGN Curncy"
    flds=['high', 'low', 'last_price', 'open']
    data_IV = blp.bdh(
        tickers=ticker_IV,
        flds = flds,
        start_date=start_date,
        end_date=end_date)
    data_IV.columns = ["high", "low", "close", "open"]
    return data_IV



# Get Intraday Spot Data
def getintradayCCY(ticker, days, interv):
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    eastern = pytz.timezone("US/Eastern")
    end_time_et = eastern.localize(datetime.now())
    start_time_et = end_time_et - timedelta(days=days)
    end_time_gmt = end_time_et.astimezone(pytz.utc)
    start_time_gmt = start_time_et.astimezone(pytz.utc)
    df = con.bdib(
            ticker=f"{ticker} Curncy",     
            start_datetime=start_time_gmt,    
            end_datetime=end_time_gmt,        
            event_type="TRADE",           
            interval= interv)
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    df = df.reset_index()
    df['date'] = df['time'].dt.date  # Extract the date
    df['time_close'] = df['time'].dt.time  # Extract the time
    df = df.drop(columns=['time'])
    df = df[['date', 'time_close'] + [col for col in df.columns if col not in ['date', 'time_close']]]
    return df


# -------------------------------------------------------------------------------------------------------------------
# ----------------------------------------Close to Close Basis ------------------------------------------------------

# Get DAILY close to close EOD Hist vol
def getEOD_C2CVol(df):
    eodClose = df['close']
    logReturn = np.log(eodClose / eodClose.shift(1)).dropna() 
    n = len(logReturn) 
    realizedVol = np.sqrt(((np.sum(logReturn ** 2)) * 252)/ n)
    return realizedVol *100


# Get INTRADAY close to close for Interval Hist Vol
def getIntraDay_C2CVol(df, interval):
    F = 1440 / interval # Number of samples per day
    intervalClose = df['close'] # Get Close data
    logreturns = np.log(intervalClose / intervalClose.shift(1)).dropna()
    n = len(logreturns) # Amount of log return samples
    realizedvol = np.sqrt((np.sum(logreturns ** 2) * F * 252) / n)
    return realizedvol *100

# -------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- High/Low - Parkinson Volatitlity--------------------------------------------
# Get DAILY High/Low Hist Vol
def getEOD_HLVol(df):
    eodHigh = df['high'] # Get High price on day
    eodLow = df['low'] # Get Low price on day
    logReturnHL = np.log(eodHigh / eodLow)
    n = len(logReturnHL) # Amount of samples
    realizedVolHL = np.sqrt((np.sum(logReturnHL ** 2) * 252) / (n * 4*np.log(2)))
    return realizedVolHL * 100


#Get IntraDay High/Low Hist Vol
def getIntraDay_HLVol(df, interval):
    F = 1440 / interval # numb samples per day
    intraHigh = df['high'] # Get High price on interval period
    intraLow = df['low'] # Get Low price on interval period
    logReturnHL  = np.log(intraHigh/intraLow)
    n = len(logReturnHL) # Amount of Samples
    realizedVol = np.sqrt((np.sum(logReturnHL ** 2) * F * 252) / (n * 4*np.log(2)))
    return realizedVol * 100


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------- GK Volatility - High/Low/Open/Close (Grarman Klass) ------------------------------

# Get DAILY GK Volatility
def getEOD_GKVol(df):
    n = len(df) # Amount of samples
    eodHigh = df['high'] # Get High price on day
    eodLow = df['low'] # Get Low price on day
    eodOpen = df['open'] # Get Open price on day
    eodClose = df['close'] # Get Close price on day
    logReturn_HL = np.log(eodHigh / eodLow) # Log Return of High/Low Prices
    logReturn_CO = np.log(eodClose / eodOpen) # Log Return of Close/Open Price
    var1 = np.sum(logReturn_HL ** 2)/ 2  # sum of log(high / low)
    var2 = np.sum(logReturn_CO ** 2)  # sum of log(close / open)
    # Get relaized vol - GK Vol
    realizedVol = np.sqrt(252 / n) * (np.sqrt(var1 - (2* np.log(2) - 1) * var2)) 
    return realizedVol * 100


# Get IntraDay GK Volatility
def getIntraDay_GKVol(df, interval):
    F = 1440 / interval # numb samples per day
    n = len(df) # amount of samples
    intervalHigh = df['high'] # Get High price on day
    intervalLow = df['low'] # Get Low price on day
    intervalOpen = df['open'] # Get Open price on day
    intervalClose = df['close'] # Get Close price on day
    logReturn_HL = np.log(intervalHigh / intervalLow) # Log Return of High/Low Prices
    logReturn_CO = np.log(intervalClose / intervalOpen) # Log Return of Close/Open Price
    var1 = np.sum(logReturn_HL ** 2)/ 2  # sum of log(high / low)
    var2 = np.sum(logReturn_CO ** 2)  # sum of log(close / open)
    # Get relaized vol - GK Vol
    realizedVol = np.sqrt((F * 252 ) / n) * (np.sqrt(var1 - (2* np.log(2) - 1) * var2))
    return realizedVol * 100


# -------------------------------------------------------------------------------------------------------------------
# ---------------------- GKYZ Volatility - High/Low/Open/Close (Grarman Klass Yang Zhang) ---------------------------

# Get EOD GKYZ Realized Vol measure 
def getEOD_GKVol(df):

    df['log_OCL'] = np.log(df["open"] / df["close"].shift(1)) # Close-to-Open return (overnight jumps)
    df['log_HL'] = np.log(df["high"] / df["low"])  # High-Low return (intraday range)
    df['log_OC'] = np.log(df["close"] / df["open"])  # Open-to-Close return (drift component)

    df = df.dropna()
    n = len(df)

    # Compute GKYZ variance components
    sum_OCL= np.sum(df['log_OCL'] ** 2)  # Sum of overnight variances
    sum_HL = np.sum((df['log_HL'] ** 2) / 2)  # Intraday high-low variance
    sum_OC = np.sum((df['log_OC'] ** 2))  # Drift correction

    realizedVol = np.sqrt((252) / n) * (np.sqrt(sum_OCL + sum_HL - (2* np.log(2) - 1) * sum_OC))

    return realizedVol * 100


# Get Intraday GKYZ Realized Vol Measure
def getIntraDay_GKVol(df, interval):
    F = 1440 / interval  # Number of samples per day

    df['log_OCL'] = np.log(df["open"] / df["close"].shift(1)) # Close-to-Open return (overnight jumps)
    df['log_HL'] = np.log(df["high"] / df["low"])  # High-Low return (intraday range)
    df['log_OC'] = np.log(df["close"] / df["open"])  # Open-to-Close return (drift component)

    df = df.dropna()
    n = len(df)

    # Compute GKYZ variance components
    sum_OCL= np.sum(df['log_OCL'] ** 2)  # Sum of overnight variances
    sum_HL = np.sum((df['log_HL'] ** 2) / 2)  # Intraday high-low variance
    sum_OC = np.sum((df['log_OC'] ** 2))  # Drift correction


    realizedVol = np.sqrt((F * 252 ) / n) * (np.sqrt(sum_OCL + sum_HL - (2* np.log(2) - 1) * sum_OC))

    return realizedVol * 100

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------
#                                ROLLING CALCULATIONS 
# -------------------------------------------------------------------------------------------------------------------



# Get IntraDay rolling Close to Close Realized Vol
def getIntraDay_RollingC2CVol(df, interval, window):
    F = 1440/interval
    logRetrunC2C = np.log(df['close'] / df['close'].shift(1)).dropna()
    rollingVolC2C = logRetrunC2C.rolling(window=window).apply(
            lambda x: np.sqrt((np.sum(x ** 2) * F * 252) / len(x)), raw= True)
    return rollingVolC2C * 100 



# Get IntraDay rolling GK Realized Vol
def getIntraDay_RollingGKVol(df, interval, window):
    F = 1440 / interval
    logReturn_HL = np.log(df['high'] / df['low']).dropna()  # High-Low log return
    logReturn_CO = np.log(df['close'] / df['open']).dropna()  # Close-Open log return
    rollingVol = (
        logReturn_HL.rolling(window=window).apply(lambda x: np.sum(x ** 2) / 2, raw=True) - 
        (2 * np.log(2) - 1) * logReturn_CO.rolling(window=window).apply(lambda x: np.sum(x ** 2), raw=True))
    rollingVol = rollingVol.apply(lambda x: np.sqrt(np.abs(F * 252 * x / window)) * 100 )
    return rollingVol






ticker = 'USDJPY'
days = 30
df = getDailySpot(ticker, days)
print(getEOD_C2CVol(df))
print(getEOD_HLVol(df))