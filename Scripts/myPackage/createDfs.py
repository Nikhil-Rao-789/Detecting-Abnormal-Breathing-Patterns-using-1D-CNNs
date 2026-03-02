import pandas as pd
import os

def createDfSPO2(participant):
    df = pd.read_csv(os.path.join(participant, 'SPO2.txt'), sep = ";", skiprows = 7, names = ['Time', 'value'])
    df["Time"] = pd.to_datetime(df["Time"], format = "%d.%m.%Y %H:%M:%S,%f")
    df.set_index("Time", inplace = True)
    return df

def createDfNasalFlow(participant):
    df = pd.read_csv(os.path.join(participant, 'Flow.txt'), sep = ";", skiprows = 7, names = ['Time', 'value'])
    df["Time"] = pd.to_datetime(df["Time"], format = "%d.%m.%Y %H:%M:%S,%f")
    df.set_index("Time", inplace = True)
    return df

def createDfThoracic(participant):
    df = pd.read_csv(os.path.join(participant, 'Thorac.txt'), sep = ";", skiprows = 7, names = ['Time', 'value'])
    df["Time"] = pd.to_datetime(df["Time"], format = "%d.%m.%Y %H:%M:%S,%f")
    df.set_index("Time", inplace = True)
    return df

def createDfFlowEvents(participant):
    df = pd.read_csv(os.path.join(participant, 'Flow Events.txt'), sep = ";", skiprows = 5, names = ["Range","Duration","Event","Stage"])
    df[["Start","End"]] = df["Range"].str.split("-", expand=True)
    df["Date"] = df["Start"].str.split().str[0]
    df["Start"] = pd.to_datetime(df["Start"],format="%d.%m.%Y %H:%M:%S,%f")
    df["End"] = pd.to_datetime(df["Date"] + " " + df["End"],format="%d.%m.%Y %H:%M:%S,%f")
    return df

def createDfSleepProfile(participant):
    df = pd.read_csv(os.path.join(participant, 'Sleep profile.txt'), sep = ";", skiprows = 7, names = ['Time', 'value'])
    df["Time"] = pd.to_datetime(df["Time"], format = "%d.%m.%Y %H:%M:%S,%f")
    df.set_index("Time", inplace = True)
    return df