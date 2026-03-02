# python3.12 Scripts/vis.py -name Data/AP01
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from myPackage.createDfs import *
from myPackage.cleanDfs import *
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-name", required = True)
args = parser.parse_args()
participant = args.name

dfSPO2 =  cleanDfSPO2( createDfSPO2( participant ) )
dfFlow = cleanDfNasalFlow( createDfNasalFlow( participant ) )
dfThorac = cleanDfThoracic( createDfThoracic( participant ) )
dfFlowEvents = createDfFlowEvents( participant )

start = max( dfSPO2.index[0] , dfFlow.index[0] , dfThorac.index[0] )
end = min( dfSPO2.index[-1] , dfFlow.index[-1] , dfThorac.index[-1] )

path = os.path.join(os.getcwd(), f"Visualizations/{participant.split('/')[1]}_visualization.pdf")
output_dir = os.path.dirname(path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

window = pd.Timedelta( minutes=5 )
ranges = pd.date_range( start = start , end = end, freq = window )

print(f"Generating visualizations for ### {participant.split('/')[1]} ###")

with PdfPages(path) as pdf:
    for st in tqdm(ranges, desc="### Generating Pdf ###", unit="page"):

        end = st + window

        dfSP02_temp = dfSPO2.loc[st:end]
        dfFlow_temp = dfFlow.loc[st:end]
        dfThorac_temp = dfThorac.loc[st:end]
        dfFlowEvents_temp = dfFlowEvents[(dfFlowEvents["Start"] <= end) & (dfFlowEvents["End"] >= st)]
        fig, ax = plt.subplots(3, 1,figsize=(18,6),sharex=True,constrained_layout=True)

        fig.patch.set_facecolor("white")
        fig.patch.set_edgecolor("black")
        fig.patch.set_linewidth(1.2)

        ax[0].plot(dfFlow_temp.index, dfFlow_temp["value"],color='tab:blue',label = "Nasal Flow")
        ax[0].legend(loc = "upper right")
        ax[0].set_ylabel("Nasal Flow (L/min)")

        for _, ev in dfFlowEvents_temp.iterrows():
            st2 = max( ev["Start"] , st )
            end2   = min( ev["End"] , end )

            color = "turquoise"
            if ev["Event"] == "Obstructive Apnea":
                color = "firebrick"
            ax[0].axvspan( st2 , end2 , color=color , alpha=0.6 ,zorder=0 )

            y_pos = 0.85
            mid = st2 + (end2 - st2) / 2
            ax[0].text(mid, y_pos, ev["Event"], transform=ax[0].get_xaxis_transform(), ha="center", va="center", fontsize=9, color="black", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"))
        
        ax[1].plot(dfThorac_temp.index, dfThorac_temp["value"],color='tab:orange',label = "Thoracic/Abdominal Resp.")
        ax[1].set_ylabel("Resp. Amplitude")
        ax[1].legend(loc = "upper right")

        ax[2].plot(dfSP02_temp.index, dfSP02_temp["value"],color='grey',label = "SPO₂")
        ax[2].legend(loc = "upper right")
        ax[2].set_ylabel("SPO₂ (%)")

        ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M:%S"))
        ax[2].xaxis.set_major_locator(mdates.SecondLocator(interval=5))

        plt.setp(ax[2].get_xticklabels(), rotation=90)
        
        fig.supxlabel("Time")

        for a in ax:
            a.grid(True, color="lightgray")
            for spine in a.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.2)

        fig.suptitle(f"{participant.split('/')[1]} - {st:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}",fontsize=14)

        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved visualizations for ### {participant.split('/')[1]} ### to {path}")