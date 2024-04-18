# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 2023

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

%load_ext autoreload
%autoreload 2
import sys
import numpy as np
from MeasurementInfo import MeasurementInfo
from RunInfo import RunInfo
import heapq
from scipy import signal
from scipy.optimize import curve_fit
import AnalyzePDE
from AnalyzePDE import SPE_data
from AnalyzePDE import Alpha_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import ProcessWaveforms_MultiGaussian
from ProcessWaveforms_MultiGaussian import WaveformProcessor as WaveformProcessor
import pickle
import dill
import pandas as pd

#%% ALPHA - 1us
#1us, no gain, no filter
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021

file_path = '/media/ed/My Passport/ed/aug-1-alpha/' # folder with H5 data files
runs_alpha_1us = []
files = [
'Run_1690911731.hdf5', 'Run_1690912805.hdf5', 'Run_1690913161.hdf5', 'Run_1690913708.hdf5',
'Run_1690914374.hdf5', 'Run_1690914711.hdf5', 'Run_1690915132.hdf5', 'Run_1690915713.hdf5',
'Run_1690915942.hdf5', 'Run_1690916527.hdf5', 'Run_1690917512.hdf5', 'Run_1690917804.hdf5',
'Run_1690918016.hdf5', 'Run_1690918246.hdf5', 'Run_1690918477.hdf5', 'Run_1690918747.hdf5', 'Run_1690919270.hdf5' ]
proms = [
    .1, .1, .1, .15,
    .005, .015, .015, .1,
    .015, .015, .2, .015,
    .1, .1, .015, .015, .005
]
upperlim = [
    1, 3, 3, 1.5,
    1, 10, .5, 10,
    .2, .2, 2, .4,
    .4, 1.4, .7, .1, 1
]
# ii = 3
# i = 4*ii; c = 4
# i=16
# files = files[i:i+c]
for file in range(len(files)):
    if file + i == 4:
        continue
    run_alpha_1us = RunInfo([file_path+files[file]], do_filter=False,
                            upper_limit=upperlim[file+i], baseline_correct=True,
                            prominence=proms[file+i], plot_waveforms=False)
    # run_alpha_1us.plot_hists('','')
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

runs_alpha_1us[0].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [
    39, 40, 40, 37,
        38, 38, 40,
    40, 38, 38, 40,
    40, 40, 40, 40, 39
]
for n in range(len(runs_alpha)):
# if n := 15:
    # if n < 2:
    #     continue
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.001
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[n].date
    info_alpha.temperature = 167
    info_alpha.bias = runs_alpha[n].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = bins[n]
    print(f'{bins[n]=}')
    # print(f"{info_alpha.bias=}")
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[n], baseline_correct = True,
                           no_solicit = True, cutoff=(0,10))
    wp.process(do_spe = False, do_alpha = True)
    # j, k = wp.get_alpha()
    wp.plot_alpha_histogram(peakcolor = 'blue', with_fit=True)
    campaign_alpha.append(wp)
    # break

#%%
storage_path = '/media/ed/My Passport/ed/'

with open(storage_path+'aug-10-campaign-alpha.pickle', 'wb') as f:
    dill.dump(campaign_alpha, f)

# with open('LED-SPE/SPE-June-28.pickle', 'rb') as f:
#     spe = dill.load(f)

p = dill.Unpickler(open("/media/ed/My Passport/ed/aug-01-campaign-alpha.pickle","rb"))
p.fast = True
# p.dump(campaign_alpha)
campaign_alpha = p.load()

p = dill.Unpickler(open(storage_path+"CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()



v_bd = 27.2
v_bd_err = 0.07
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

p = dill.Pickler(open("July-2023-alpha_data.pickle","wb"))
p.fast = True
p.dump(alpha_data)

alpha_data.analyze_alpha(out_file='aug-01-2023-lxe.csv')

#%%
alpha_data.plot_alpha(out_file='aug-01-2023_alpha_amp.csv')

alpha_data.plot_alpha()

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE_noCU_dif = .003353
PTE_noCU_spec = .001979
PTE_noCU_noref = .001979
alpha_data.plot_PDE(N*PTE_noCU_spec)


PTE_spec = .003353
PTE_dif = .001979
PTE_noref = .001360

plt.style.use('misc/nexo.mplstyle')

data = 'num'
df = pd.read_csv('aug-01-2023-lxe.csv').sort_values('ov')
data_x = df['ov']
data_x_err = df['ov error']
data_y = df[data]
data_y_err = df[data+' error']

df10 = pd.read_csv('aug-10-2023-lxe.csv').sort_values('ov')
d10_x = df10['ov']
d10_x_err = df10['ov error']
d10_y = df10[data]
d10_y_err = df10[data+' error']

ratio = data_y / d10_y

udata_y = np.array([ ufloat(v, e) for v, e in zip(data_y, data_y_err)])
udata10_y = np.array([ ufloat(v, e) for v, e in zip(d10_y, d10_y_err)])
ratio = udata_y / udata10_y

ration = [ r.n for r in ratio ]
ratios = [ r.s for r in ratio ]

color = 'tab:blue'
fig = plt.figure()
fig.tight_layout()
# plt.rc('font', size=22)
plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 8, fmt = '.',
             color = color, label='Silicon (Aug 1 2023)')
plt.errorbar(d10_x, d10_y, xerr = d10_x_err, yerr = d10_y_err, markersize = 8, fmt = '.',
             color = 'tab:purple', label='No Reflector (Aug 10 2023)')
# plt.errorbar(data_x, ration, xerr=data_x_err, yerr=ratios, markersize = 8, fmt = '.', color =
#              'tab:green', label='Ratio (Average: 2.238±.00058)')
plt.xlabel('Overvoltage [V]')
plt.ylabel('Number of Detected Photons')
# plt.ylabel('Alpha Amplitude [V]')
# textstr = f'Date: {self.campaign[0].info.date}\n'
textstr = f'Silicon Reflector\n'
textstr += f'Condition: LXe\n'
textstr += f'RTD4: 167 [K]'
props = dict(boxstyle='round', facecolor=color, alpha=0.4)
fig.text(.75, 0.25, textstr, fontsize=10,
        verticalalignment='top', bbox=props)
plt.legend()
plt.show()

plt.errorbar(data_x, ration, xerr=data_x_err, yerr=ratios, markersize = 8, fmt = '.', color = 'tab:green', label='Ratio')
plt.xlabel('Overvoltage [V]')
plt.ylabel('Alpha Pulse Ratio')
# textstr = f'Date: {self.campaign[0].info.date}\n'
textstr = f'Silicon Reflector\n'
textstr += f'Condition: LXe\n'
# textstr += f'Ratio Average: 18.228 ±\n'
textstr += f'RTD4: 167 [K]'
props = dict(boxstyle='round', facecolor=color, alpha=0.4)
fig.text(0.75, 0.2, textstr, fontsize=10,
        verticalalignment='top', bbox=props)
plt.legend()
plt.show()

