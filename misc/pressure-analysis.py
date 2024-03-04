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
    
    # for RTD in RTDS: # Remove voltage spikes
    #     diff = df[RTD].diff().fillna(0)
    #     df = df[abs(diff) < 0.5]

    df['Timestamp'] = date + time

    print(f"loaded {p}")
    return df

df_T = read_h5('D:/Xe/AnalysisScripts/LXe May 2023/RTD_2023_05.h5')
df = read_h5('D:/Xe/AnalysisScripts/LXe May 2023/Fl_Pres_2023_05.h5', filetype = 'Pres')
#%%
# select region of interest
start = dt.datetime(2023, 5, 18, 14)
end   = dt.datetime(2023, 5, 19, 0)
# start = dt.datetime(2023, 5, 18, 19, 0)
# end   = dt.datetime(2023, 5, 18, 21, 0)
timestamp = df['Date'] + df['Time']
df = df[start <= timestamp]
df = df[timestamp <= end]
df_T = df[start <= timestamp]
df_T = df[timestamp <= end]

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

alpha_start_2 = dt.datetime(2023, 5, 18, 17, 55)
alpha_end_2   = dt.datetime(2023, 5, 18, 19, 8)
ax.axvspan(date2num(alpha_start_2), date2num(alpha_end_2), facecolor='cyan', edgecolor='none', alpha=.2)
ax.text(alpha_start_2+(alpha_end_2-alpha_start_2)/2, 196, '100ns Alpha\nPulses',
        verticalalignment='top', horizontalalignment='center', fontsize=12)

alpha_start = dt.datetime(2023, 5, 18, 20, 55)
alpha_end   = dt.datetime(2023, 5, 18, 22, 4)
ax.axvspan(date2num(alpha_start), date2num(alpha_end), facecolor='blue', edgecolor='none', alpha=.2)
ax.text(alpha_start+(alpha_end-alpha_start)/2, 196, '1us Alpha\nPulses',
        verticalalignment='top', horizontalalignment='center', fontsize=12)

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

# Plot pressure
for p in Pres[:2]:
    pressure = alpha_df[p].mean()
    pressure_err = alpha_df[p].std()
    print(p,'=',np.round(pressure,3),'+/-',np.round(pressure_err,3),'Torr')
    plt.plot(x, df[p], markersize=0.5
               # , label = f'{p} = {pressure:.4} $\pm$ {pressure_err:.3} Torr\n'
                , label = f'{p}'
             )
    plt.ylabel('Pressure (Torr)', fontsize=fs)
    plt.xlabel('Time', fontsize=fs)

# plt.title('June 28th 2023 Liquefaction RTD Stability', fontsize=fs)
plt.grid()
plt.legend(markerscale=fs, fontsize=fs-12, loc='upper left') #loc='center left', bbox_to_anchor=(1.0, 0.5)
# plt.savefig('plots/may-6-std-%d-%d-%03d.png'%(longtime, shorttime, thres*1000))
plt.show()
#%%
# Plot P1 v T

# select region of interest
# start = dt.datetime(2023, 5, 18, 20, 55)
# end   = dt.datetime(2023, 5, 19, 22, 4)
start = dt.datetime(2023, 5, 18, 17, 30)
end   = dt.datetime(2023, 5, 18, 21, 30)
timestamp = df['Date'] + df['Time']
df = df[start <= timestamp]
df = df[timestamp <= end]
timestamp = df_T['Date'] + df_T['Time']
df_T = df_T[start <= timestamp]
df_T = df_T[timestamp <= end]
#%%
# x axis to be plotted
x = df_T['RTD4'][:-1]
x0 = df_T['RTD0'][:-1]
x1 = df_T['RTD1'][:-1]
x2 = df_T['RTD2'][:-1]
x3 = df_T['RTD3'][:-1]
fs = 20 # font size

plt.rc('xtick', labelsize = fs-10)
plt.rc('ytick', labelsize = fs-10)

ax = plt.gca()
ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(n=6))
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator(n=5))
ax.grid(visible=True, which='minor', color='lightgrey', linewidth=0.5)
plt.ylabel('Pressure (Torr)', fontsize=fs)
plt.xlabel('Temperature (K)', fontsize=fs)

plt.grid()
plt.show()
plt.scatter(x, df['P1'], s=0.5, label = 'RTD4')
plt.scatter(x0, df['P1'], s=0.5,  label = 'RTD0')
plt.scatter(x1, df['P1'], s=0.5, label = 'RTD1')
plt.scatter(x2, df['P1'], s=0.5,  label = 'RTD2')
plt.scatter(x3, df['P1'], s=0.5,  label = 'RTD3')
plt.legend(markerscale=fs, fontsize=fs-12, loc='upper left')