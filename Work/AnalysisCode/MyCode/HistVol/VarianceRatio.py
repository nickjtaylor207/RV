import pdblp

import pandas as pd
import numpy as np
import pytz

import seaborn as sns
import matplotlib.pyplot as plt

from xbbg import blp
from datetime import datetime, timedelta

from pandas.tseries.offsets import DateOffset


from histVolMeasures import *




ticker = 'USDMXN'
days = 60
interval = 30
window = int(7 * (1440 / interval))

df_ID = getintradayCCY(ticker, days, interval)



df_ID['IDRolHV_C2C'] = getIntraDay_RollingC2CVol(df_ID, interval, window)
df_ID['IDRollHV_GK'] = getIntraDay_RollingGKVol(df_ID, interval, window)

df_ID["GK_C2C_VR"] = (df_ID["IDRollHV_GK"] / df_ID["IDRolHV_C2C"])

df_ID = df_ID.dropna()

print(df_ID)
