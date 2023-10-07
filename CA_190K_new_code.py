# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:06:03 2022

@author: lab-341
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:34:31 2022

@author: lab-341
"""

#%%
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
run_spe_solicited = RunInfo(['D:/Xe/DAQ/Run_1648191265.hdf5'], do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
files = ['Run_1648176286','Run_1648179496', 'Run_1648181807', 'Run_1648184235', 'Run_1648186910'] #, 'Run_1648186910', 'Run_1648171846'
# upperlim = [1, 1.76, 4.4, 4.4, 4.4]
upperlim = [5, 5, 5, 5, 5, 5]
runs = []
for file in range(len(files)):
    run_spe = RunInfo(['D:/Xe/DAQ/' + files[file] + '.hdf5'], do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = 0.05)
    runs.append(run_spe)
biases = [run.bias for run in runs] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
#%%
#%%
cutoffs = []
centers_list=[]
centers_guesses = []

for run in runs:
    bins = int(round(np.sqrt(len(run.all_peak_data)))*1.5)
    print(bins)
    count, edges = np.histogram(run.all_peak_data, bins=bins)
    if run.bias == 36.8:
        peak_data = run.all_peak_data[run.all_peak_data > 0.08]
        bins = int(round(np.sqrt(len(peak_data)))*1.5)
        count, edges = np.histogram(peak_data, bins=bins)
    centers = (edges[:-1] + edges[1:])/2
    peaks, props = signal.find_peaks(count, prominence=10, distance=2)
    print(peaks)
    fitrange = ((centers[peaks[3]] - centers[peaks[0]])/2)
    range_low =  centers[peaks[0]]- 0.31*fitrange
    range_high = centers[peaks[3]]+ 0.35*fitrange
    
    cutoffs.append((range_low, range_high))
    centers_list.append(centers[peaks[0]])
    peaks = peaks[0:]
    centers_guesses.append([centers[peak] for peak in peaks])
    
    plt.figure()
    plt.hist(run.all_peak_data, bins=bins)
    for peak in peaks:
        plt.scatter(centers[peak],count[peak])
    plt.axvline(range_low, c='red')
    plt.axvline(range_high, c='black')
    plt.xlim([0,1])
    plt.yscale('log')
#%%

#%% make a list of ProcessWaveforms objects
campaign_spe = []
for i in range(len(runs)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'LXe'
    info_spe.date = runs[i].date
    info_spe.temperature = 170
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 150
    info_spe.data_type = 'h5'
    # if i < 3:
    #     num = 3
    # else:
    num = 4
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True,  cutoff = cutoffs[i], centers = centers_guesses[i], no_solicit = False, numpeaks = num)
    wp_spe.process(do_spe = True, do_alpha = False)
    wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/CA_190K_gauss_'+str(info_spe.bias)+'.png')
    wp_spe.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/May_170K_spe_'+str(info_spe.bias)+'.png')
    wp_spe.plot_both_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/CA_190K_baseline_'+str(info_spe.bias)+'.png')
    wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/CA_190K_gauss_'+str(info_spe.bias)+'.svg')
    wp_spe.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/May_170K_spe_'+str(info_spe.bias)+'.svg')
    wp_spe.plot_both_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/CA_190K_baseline_'+str(info_spe.bias)+'.svg')
    plt.close()
    campaign_spe.append(wp_spe)

#%% plots
for i in range(len(campaign_spe)):
    campaign_spe[i].plot_peak_histograms()
#%%
n = 0
for i in campaign_spe:
    n+=1
    i.plot_peak_histograms(log_scale=False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe May 2023/100ns_gauss_'+str(n)+'.png')
    i.plot_peak_histograms(log_scale=False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe May 2023/100ns_gauss_'+str(n)+'.svg')
    plt.grid(True)
    # plt.close()
#%%
invC_spe_filter = 0.19261918346201742
invC_spe_err_filter = 0.0021831140214106596
spe = SPE_data(campaign_spe, invC_spe_filter, invC_spe_err_filter, filtered = True)
spe.plot_spe(in_ov = False, absolute = False, out_file = None)
#%%
path = 'D:/Xe/AnalysisScripts/March 2022 HD3 Vacuum/CA_updated.csv'
spe.plot_CA()
