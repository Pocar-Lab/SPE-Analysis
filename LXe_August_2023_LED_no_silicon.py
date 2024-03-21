# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:41:24 2023

@author: Hannah
"""

import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
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
import csv
plt.style.use('D:/Xe/AnalysisScripts/LXe May 2023/nexo_new.mplstyle')
#%%
#1us, 10x gain, filter off
invC = 0.0132
invC_err =0.000089
invC_filter = 0.01171
invC_err_filter = 0.000048
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
run_spe_solicited = RunInfo(['D:/Xe/DAQ/Run_1691696340.hdf5'],  specifyAcquisition = False, do_filter = True, is_solicit = True, upper_limit = 0.2, baseline_correct = True)
files = ['Run_1691691406','Run_1691690935','Run_1691688275','Run_1691694839','Run_1691693056','Run_1691694534','Run_1691695317','Run_1691689385','Run_1691689812','Run_1691693423', 'Run_1691694026'] #31.5, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36, 36.5
upperlim = [0.062, 0.062, 0.1, 0.11, 0.11, 0.11, 0.11, 0.18, 0.2, 0.2, 0.5]
# upperlim = [0.062, 0.062, 0.1, 0.1, 0.1, 0.1, 0.1, 0.18, 0.3, 0.45, 0.45]
runs = []
for file in range(2,len(files)):
    run_spe = RunInfo(['D:/Xe/DAQ/'+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = 0.0048, is_led = True)
    runs.append(run_spe)
biases = [run.bias for run in runs]
#%%
n=-5
test = RunInfo(['D:/Xe/DAQ/'+files[n]+'.hdf5'], specifyAcquisition = False, do_filter = False, upper_limit = 2, baseline_correct = True, prominence = 0.005, is_led = True)
test.plot_hists('', '')
#%%
def get_subtract_hist_mean(bias, data1, data2, numbins = 2000, plot = False):
    if plot:
        plt.figure()
        (n, b, p) = plt.hist(data1, bins = numbins, density = False, label = 'LED-On', histtype='step')
        plt.axvline(x = np.mean(data1), color = 'blue')
        print('LED on hist: ' + str(np.mean(data1)))
        print('LED off hist: ' + str(np.mean(data2)))
        plt.axvline(x = np.mean(data2), color = 'orange')
        plt.hist(data2, bins = b, density = False, label = 'LED-Off', histtype='step')
    counts1, bins1 = np.histogram(data1, bins = numbins, density = False)
    counts2, bins2 = np.histogram(data2, bins = bins1, density = False)
    centers = (bins1[1:] + bins1[:-1])/2
    subtracted_counts = counts1 - counts2
    # subtracted_counts[subtracted_counts < 0] = 0
    if plot:
        plt.step(centers, subtracted_counts, label = 'subtracted hist')
        plt.legend()
        
    big_n = np.sum(subtracted_counts)
    norm_subtract_hist = subtracted_counts / big_n
    # weights = 1.0 / subtracted_counts / 
    mean_value = np.sum(centers * norm_subtract_hist)
    if plot:
        plt.title(f'Bias: {bias}V, mean_value: {mean_value}')
        plt.axvline(x = mean_value, color = 'green')  
    # mean_err = np.sum((centers/big_n) ** 2)(subtracted_counts) + (np.sum(subtracted_counts*centers)/(big_n)**2) ** 2 * (np.sum(subtracted_counts)) #overestimation
    a = np.sum(subtracted_counts * centers)
    mean_err = np.sqrt(np.sum( ((a - centers * big_n)/ big_n ** 2) ** 2 * (counts1 + counts2)))
    
    return (mean_value, mean_err)
    
#%%
for run in runs:
    get_subtract_hist_mean(run.bias,run.all_led_peak_data, run.all_dark_peak_data, plot = True)

#%% variable peaks
range_lows = [0.004126,0.00435,0.0045,0.0049,0.0057,0.0062,0.0063,0.0066,0.0074] 
centers = [0.0062,0.00680,0.00707,0.00750,0.00776,0.00846,0.0095,0.01,0.0105] #
range_highs = [0.0268,0.035,0.0305,0.04065,0.03537,0.038,0.049,0.0419,0.054] #only use from 33V and onwards
numpeaks = [4,5,4,5,4,4,5,4,5]

#%% testing cell - plot with peak fit
# set conditions, temp
n=8
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
wp = WaveformProcessor(info_spe, run_info_self = runs[n], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[n], range_high = range_highs[n], center = centers[n], peak_range = (1,numpeaks[n]))
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
    info_spe.temperature = 168
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True,  range_low = range_lows[i], range_high = range_highs[i], center = centers[i], peak_range = (1,numpeaks[i]))
    wp_spe.process(do_spe = True, do_alpha = False)
    # wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/plots_no_silicon/no_silicon_gauss_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/plots_no_silicon/no_silicon_spe_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_both_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/plots_no_silicon/no_silicon_baseline_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/plots_no_silicon/no_silicon_gauss_'+str(info_spe.bias)+'.svg')
    # wp_spe.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/plots_no_silicon/no_silicon_spe_'+str(info_spe.bias)+'.svg')
    # wp_spe.plot_both_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe August 1 2023/plots_no_silicon/no_silicon_baseline_'+str(info_spe.bias)+'.svg')
    # plt.close()
    campaign_spe.append(wp_spe)
    print(info_spe.bias)

#%%
for run in campaign_spe:
    run.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/No Source Run/August_no_silicon_gauss_'+str(run.info.bias)+'.png')
    run.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/No Source Run/August_no_silicon_'+str(run.info.bias)+'.png')
    plt.close()
#%% plot linear fit to the breakdown voltage
curr_campaign = campaign_spe
spe = SPE_data(curr_campaign, invC_filter, invC_err_filter, filtered = True)
spe.plot_spe(in_ov = False, absolute = False, out_file = 'D:/Raw Results/Breakdown Voltage/SPE Amplitudes/vbd_bias_no_silicon.csv') #
spe.plot_CA(out_file = 'D:/Raw Results/Correlated Avalanche Data/August10_CA_no_silicon_NoSubtraction.csv')
