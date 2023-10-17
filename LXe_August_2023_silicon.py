# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:41:24 2023

@author: Hannah
"""

import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
sys.path.append('D:/Xe/AnalysisScripts/LXe May 2023/')   
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
import h5py
plt.style.use('D:/Xe/AnalysisScripts/LXe May 2023/nexo_new.mplstyle')
#%%
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
