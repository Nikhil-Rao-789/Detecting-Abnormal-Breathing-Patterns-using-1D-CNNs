"""
python3.12 Scripts/create_dataset.py -in_dir Data -out_dir Dataset
"""
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from myPackage.createDfs import *
from myPackage.cleanDfs import *

def getLabel(st, end, dfEvents):

    overlap = dfEvents[(dfEvents["Start"] < end) & (dfEvents["End"] > st)]
    
    if overlap.empty:
        return "Normal"

    duration = (end - st).total_seconds()

    for _, row in overlap.iterrows():
    
        overlapTime = (min(end, row["End"]) - max(st, row["Start"])).total_seconds()

        if (overlapTime / duration) > 0.5:
            return row["Event"]

    return "Normal"

def getStage(st, dfSleepProfile):
    
    past = dfSleepProfile[dfSleepProfile.index <= st]
    
    if past.empty:
        return "Unknown"
        
    return past["value"].iloc[-1]


parser = argparse.ArgumentParser()
parser.add_argument("-in_dir", required=True)
parser.add_argument("-out_dir", required=True)
args = parser.parse_args()

os.makedirs(os.path.join(os.getcwd(), args.out_dir), exist_ok=True)

inPath = os.path.join(os.getcwd(), args.in_dir)
participants = [d for d in os.listdir(inPath) if os.path.isdir(os.path.join(inPath, d))]
participants.sort()

window = 30
step = 15
f = 32

X = []
y = []
stages = []
groups = []

numSamples = window * f
stepSamples = step * f

for participant in participants:

    print(f"Processing ### {participant} ###")

    path = os.path.join(inPath, participant)

    dfSPO2 = cleanDfSPO2( createDfSPO2( path ) )
    dfFlow = cleanDfNasalFlow( createDfNasalFlow( path ) )
    dfThorac = cleanDfThoracic( createDfThoracic( path ) )
    dfEvents = createDfFlowEvents( path )
    dfSleepProfile = createDfSleepProfile( path )

    dfComb = pd.concat([dfFlow["value"], dfThorac["value"], dfSPO2["value"]], axis=1, join="inner")
    dfComb.columns = ["Flow", "Thorac", "SPO2"]

    dfComb.dropna(inplace=True)

    dataVals = dfComb.values
    timeIdx = dfComb.index

    for i in tqdm(range(0, len(dataVals) - numSamples + 1, stepSamples), desc="### Slicing Windows ###"):
        matrix = dataVals[i : i + numSamples]
        
        st = timeIdx[i]
        end = timeIdx[i + numSamples - 1]

        label = getLabel(st, end, dfEvents)
        stage = getStage(st, dfSleepProfile)

        X.append(matrix)
        y.append(label)
        stages.append(stage)
        groups.append(participant)


X = np.array(X)
y = np.array(y)

stages = np.array(stages)
groups = np.array(groups)

outPath = os.path.join(os.getcwd(), args.out_dir)

np.save(os.path.join(outPath, "X.npy"), X)
np.save(os.path.join(outPath, "y.npy"), y)
np.save(os.path.join(outPath, "stages.npy"), stages)
np.save(os.path.join(outPath, "groups.npy"), groups)

print("\nDataset successfully created and saved!")
print(f"X shape: {X.shape} (Windows, Steps, Channels)")
print(f"y shape: {y.shape}")
print(f"stages shape: {stages.shape}")