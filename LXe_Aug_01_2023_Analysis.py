# -*- coding: utf-8 -*-
"""
Alpha: Created on Fri Jul 13 2023
@author: Ed van Bruggen (evanbruggen@umass.edu)

SPE: Created on Mon Oct 16 18:41:24 2023
@author: Hannah
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

#%% SPE
#1us, 10x gain, filter off
invC =0.0132
invC_err =0.000089
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
run_spe_solicited = RunInfo(['D:/Xe/DAQ/Run_1690920592.hdf5'],  specifyAcquisition = False, do_filter = False, is_solicit = True, upper_limit = 0.2, baseline_correct = True)
files = ['Run_1690921033','Run_1690932970','Run_1690924626','Run_1690930429','Run_1690928479','Run_1690927092','Run_1690931305','Run_1690925548','Run_1690933780','Run_1690932183', 'Run_1690929508'] #31.5, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36, 36.5
upperlim = [0.12, 0.12, 0.2, 0.2, 0.2, 0.6, 0.6, 0.6, 0.6, 0.6, 1.2]
proms = [0.0048, 0.0048, 0.0048, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006]
runs = []
for file in range(len(files)):
    run_spe = RunInfo(['D:/Xe/DAQ/'+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = False, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], poly_correct = True)
    runs.append(run_spe)
biases = [run.bias for run in runs]
#%%
# RunInfo(['D:/Xe/DAQ/'+files[0]+'.hdf5'], specifyAcquisition = False, do_filter = False, upper_limit = 0.1, baseline_correct = True, prominence = 0.005, poly_correct = True, plot_waveforms = True)
# test=RunInfo(['D:/Xe/DAQ/'+files[2]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.12, baseline_correct = True, prominence = 0.0048, poly_correct = True)
# test.plot_hists('','')
# runs[-1] = test
#%%
cutoffs = []
centers_list=[]
centers_guesses = []

for run in runs:
    bins = int(round(np.sqrt(len(run.all_peak_data))))
    print(bins)
    count, edges = np.histogram(run.all_peak_data, bins=bins)
    centers = (edges[:-1] + edges[1:])/2
    peaks, props = signal.find_peaks(count, prominence=25, distance=3)
    fitrange = ((centers[peaks[3]] - centers[peaks[0]])/2)
    if run.bias < 32.5:
        range_low =  centers[peaks[0]]- 0.3*fitrange
        range_high = centers[peaks[3]]+ 0.42*fitrange
    if run.bias == 32.5:
        range_low =  centers[peaks[0]]- 0.31*fitrange
        range_high = centers[peaks[3]]+ 0.45*fitrange
    if run.bias > 32.5:
        range_low =  centers[peaks[0]]- 0.22*fitrange
        range_high = centers[peaks[3]]+ 0.32*fitrange
    if run.bias >= 34:
        range_low =  centers[peaks[0]]- 0.31*fitrange
        range_high = centers[peaks[3]]+ 0.37*fitrange 
    if run.bias == 33.5:
        range_low =  centers[peaks[0]]- 0.24*fitrange
        range_high = centers[peaks[3]]+ 0.31*fitrange
    if run.bias > 34.5:
        range_low =  centers[peaks[0]]- 0.3*fitrange
        range_high = centers[peaks[3]]+ 0.31*fitrange
    cutoffs.append((range_low, range_high))
    centers_list.append(centers[peaks[0]])
    peaks = peaks[0:]
    centers_guesses.append([centers[peak] for peak in peaks])
    
    # plt.figure()
    # plt.hist(run.all_peak_data, bins=bins)
    # for peak in peaks:
        # plt.scatter(centers[peak],count[peak])
    # plt.axvline(range_low, c='red')
    # plt.axvline(range_high, c='black')
    # plt.xlim([0,1])
    # plt.yscale('log')
#%% testing cell - plot with peak fit
# set conditions, temp
n=10
T = 169.5
con = 'LXe'
info_spe = MeasurementInfo()
info_spe.condition = con
info_spe.date = runs[n].date
info_spe.temperature = T
info_spe.bias = biases[n]
info_spe.baseline_numbins = 100
info_spe.peaks_numbins = 200
info_spe.data_type = 'h5'
wp = WaveformProcessor(info_spe, run_info_self = runs[n], run_info_solicit = run_spe_solicited, baseline_correct = True, cutoff = cutoffs[n], centers = centers_guesses[n], numpeaks = 4)
wp.process(do_spe = True, do_alpha = False)
wp.plot_peak_histograms(log_scale = False)
wp.plot_spe()
# wp.plot_both_histograms()
# wp.plot_baseline_histogram()
#%%
campaign_spe = []
for i in range(len(runs)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'LXe'
    info_spe.date = runs[i].date
    info_spe.temperature = 169.5
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 150
    info_spe.data_type = 'h5'
    num = 4
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True,  cutoff = cutoffs[i], centers = centers_guesses[i], no_solicit = False, numpeaks = num)
    wp_spe.process(do_spe = True, do_alpha = False)
    # wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/silicon_gauss_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/silicon_spe_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_both_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/siicon_baseline_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/silicon_gauss_'+str(info_spe.bias)+'.svg')
    # wp_spe.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/silicon_spe_'+str(info_spe.bias)+'.svg')
    # wp_spe.plot_both_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/silicon_baseline_'+str(info_spe.bias)+'.svg')
    # plt.close()
    campaign_spe.append(wp_spe)
    print(info_spe.bias)

#%% plot linear fit to the breakdown voltage
curr_campaign = campaign_spe
spe = SPE_data(curr_campaign, invC, invC_err, filtered = False)
spe.plot_spe(in_ov = False, absolute = False, out_file = 'D:/Xe/AnalysisScripts/LXe August 1 2023/vbd_bias.csv') #



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

