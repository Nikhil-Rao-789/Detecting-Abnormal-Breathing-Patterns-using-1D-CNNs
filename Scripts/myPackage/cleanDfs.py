import pandas as pd
import numpy as np
from scipy.signal import *

def cleanDfSPO2(df):

    df = df.copy()
    df.sort_index(inplace=True)

    df["value"] = pd.to_numeric(df["value"], errors = "coerce")
    df.loc[ (df["value"] < 70) | (df["value"] > 100), "value"] = np.nan

    jumpVals = df["value"].diff().abs()
    df.loc[jumpVals > 5, "value"] = np.nan

    df["value"] = df["value"].interpolate(method = "time",limit = 20)

    df["value"] = df["value"].rolling(window = 8, center = True, min_periods = 1).median()
    df["value"] = df["value"].bfill().ffill()

    f = "31.25ms"

    df = df.resample(f).interpolate(method="time",limit = 20)

    return df


def cleanDfNasalFlow(df):
    
    df = df.copy()
    df.sort_index(inplace=True)

    df["value"] = pd.to_numeric(df["value"], errors = "coerce")
    df["value"] = df["value"].interpolate(limit=64).bfill().ffill()
    df["value"] = df["value"].clip(df["value"].quantile(0.01), df["value"].quantile(0.99))
    df["value"] = bpFilter(df["value"])

    f = "31.25ms"

    df = df.resample(f).mean().interpolate(method="time",limit = 64)

    return df


def cleanDfThoracic(df):
    
    df = df.copy()
    df.sort_index(inplace=True)

    df["value"] = pd.to_numeric(df["value"], errors = "coerce")

    mean = df["value"].mean()
    sigma = df["value"].std()
    zs = (df["value"] - mean) / sigma

    df.loc[zs.abs() > 5, "value"] = np.nan
    df["value"] = df["value"].interpolate(method = "linear", limit=64).bfill().ffill()

    df["value"] = bpFilter(df["value"])

    f = "31.25ms"

    df = df.resample(f).mean().interpolate(method="time",limit = 64)

    return df

def bpFilter(data):

    lw = 0.17
    hi = 0.4
    f = 32
    order = 4
    nyq = 0.5 * f
    low = lw / nyq
    high = hi / nyq

    b, a = butter(order, [low, high], btype='band')
    newData = filtfilt(b, a, data)

    return newData