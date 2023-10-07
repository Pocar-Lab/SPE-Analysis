#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10, 2023

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
import datetime as dt
from datetime import timedelta
import sys
import os
from os.path import exists
import ast
import h5py
plt.style.use('D:/Xe/AnalysisScripts/LXe May 2023/nexo_new.mplstyle')

RTDS = ['RTD0', 'RTD1', 'RTD2', 'RTD3', 'RTD4']
Pres = ['P1', 'P2', 'Flow', 'LXe', 'Heater']

def read_h5(p, filetype='RTD'):

    with h5py.File(p,'r') as hdf:
        TCdata = hdf.get('Data')[:]

    labels_TC = ['datetime'] + (RTDS if filetype == 'RTD' else Pres)
    df = pd.DataFrame(TCdata,columns = labels_TC)
    dates = []
    times = []
    for i in range(len(df)):
        ddtt = dt.datetime.fromtimestamp(int(df['datetime'][i]))
        if filetype == 'RTD':
            ddtt += timedelta(hours=5)
        dates.append(ddtt.strftime('%#m/%#d/%Y'))
        times.append(ddtt.strftime('%H:%M:%S'))

    df = df.assign(Date = dates)
    df = df.assign(Time = times)

    date = df['Date'] = pd.to_datetime(df['Date'])
    time = df['Time'] = pd.to_timedelta(df['Time'])

    # TODO replace with interplate value
    for RTD in RTDS: # Remove voltage spikes
        diff = df[RTD].diff().fillna(0)
        df = df[abs(diff) < 0.5]

    df['Timestamp'] = date + time

    print(f"loaded {p}")
    return df

df = read_h5('RTD_2023_05.h5')

# select region of interest
# start = dt.datetime(2023, 5, 18, 14)
# end   = dt.datetime(2023, 5, 19, 0)
start = dt.datetime(2023, 5, 18, 18, 30)
end   = dt.datetime(2023, 5, 18, 21, 15)
timestamp = df['Date'] + df['Time']
df = df[start <= timestamp]
df = df[timestamp <= end]

# x axis to be plotted
x = df['Date'] + df['Time']

# datestr = str(df['Date'].iloc[0])

fs = 20 # font size
# plt.figure(figsize=(14,12))
# fig, ax = plt.subplots()
plt.rc('xtick', labelsize = fs-10)
plt.rc('ytick', labelsize = fs-10)

# ax.xaxis.get_ticklocs(minor=True)
# ax.minorticks_on()
ax = plt.gca()
ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(n=6))
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(n=5))
ax.grid(visible=True, which='minor', color='lightgrey', linewidth=0.5)

#Highlight regions of data collection

# alpha_start_2 = dt.datetime(2023, 5, 18, 17, 55)
# alpha_end_2   = dt.datetime(2023, 5, 18, 19, 8)
# ax.axvspan(date2num(alpha_start_2), date2num(alpha_end_2), facecolor='cyan', edgecolor='none', alpha=.2)
# ax.text(alpha_start_2+(alpha_end_2-alpha_start_2)/2, 196, '100ns Alpha\nPulses',
#         verticalalignment='top', horizontalalignment='center', fontsize=12)

# alpha_start = dt.datetime(2023, 5, 18, 20, 55)
# alpha_end   = dt.datetime(2023, 5, 18, 22, 4)
# ax.axvspan(date2num(alpha_start), date2num(alpha_end), facecolor='blue', edgecolor='none', alpha=.2)
# ax.text(alpha_start+(alpha_end-alpha_start)/2, 196, '1us Alpha\nPulses',
#         verticalalignment='top', horizontalalignment='center', fontsize=12)
spe_start = dt.datetime(2023, 5, 18, 19, 10)
spe_end   = dt.datetime(2023, 5, 18, 20, 45)
ax.axvspan(date2num(spe_start), date2num(spe_end), facecolor='darkblue', edgecolor='none', alpha=.2)
ax.text(spe_start+timedelta(minutes=3), 196, 'SPE',
        verticalalignment='top', horizontalalignment='left', fontsize=12)

# Select region of data collection to determine stability
alpha_df = df
timestamp = alpha_df['Date'] + alpha_df['Time']
alpha_df = alpha_df[spe_start <= timestamp]
alpha_df = alpha_df[timestamp <= spe_end]

# Plot RTDs
for RTD in RTDS:
    # if RTD == 'RTD0':
    #     continue
    temp = alpha_df[RTD].mean()
    temp_err = alpha_df[RTD].std()
    print(RTD,'=',np.round(temp,3),'+/-',np.round(temp_err,3),'K')
    plt.plot(x, df[RTD], markersize=0.5, label =
              f'{RTD} = {temp:.4} $\pm$ {temp_err:.3} K\n')
    plt.ylabel('Temperature (K)', fontsize=fs)
    plt.xlabel('Time', fontsize=fs)

# plt.title('June 28th 2023 Liquefaction RTD Stability', fontsize=fs)
plt.grid()
plt.legend(markerscale=fs, fontsize=fs-12, loc='upper left') #loc='center left', bbox_to_anchor=(1.0, 0.5)
# plt.savefig('plots/may-6-std-%d-%d-%03d.png'%(longtime, shorttime, thres*1000))
plt.show()
