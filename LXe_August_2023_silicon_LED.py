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
import os
plt.style.use('C:/Users/Hannah/Documents/GitHub/SPE-Analysis/nexo_new.mplstyle')
#%%
#1us, 10x gain, filter off
invC =0.0132
invC_err =0.000089
invC_filter = 0.01171
invC_err_filter = 0.000048
path ='C:/Users/Hannah/OneDrive - University of Massachusetts/Dataruns_2023-24/DAQ_folder_google_drive/20230801_SPE_LXe/'
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
dir_list = os.listdir(path)
run_spe_solicited = RunInfo([path+'Run_1690920592.hdf5'],  specifyAcquisition = False, do_filter = True, is_solicit = True, upper_limit = 0.2, baseline_correct = True)
files = dir_list[1:]
upperlim = [0.18, 0.38, 0.9, 0.9, 0.38, 1.8, 0.38,0.9,0.9,0.18,0.9]
proms = [0.0045, 0.0045, 0.0045, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
runs = []
for file in range(1,len(files)):
    run_spe = RunInfo([path+files[file]], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file]*0.5, baseline_correct = True, prominence = proms[file], is_led = True, poly_correct = True, poly_correct_degree=3)
    runs.append(run_spe)
biases = [run.bias for run in runs]
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
    subtracted_counts = -counts1 + counts2
    subtracted_counts[subtracted_counts < -70] = 0
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
    
#%% # unfortunately the subtraction results are no good for this run
for run in runs:
    get_subtract_hist_mean(run.bias,run.all_led_peak_data, run.all_dark_peak_data, plot = True)
    
#%% variable peaks
range_lows2 = [0.003,0.0046, 0.0055,0.00525,0.00435,0.0062,0.005,0.0054,0.0055,0.0047,0.006] 
centers2 = [0.00575,0.0058,0.0086,0.0075,0.00680,0.0101,0.0065,0.008,0.0096,0.00545,0.0095] #
range_highs2 = [0.032,0.0315,0.0549,0.041,0.037,0.056,0.0415,0.044,0.0522,0.0286,0.05] #only use from 33V and onwards
numpeaks = [3,5,6,5,5,5,6,5,5,5,5]

#%%
n=10#test the different biases
T = 169
con = 'LXe'
info_spe = MeasurementInfo()
info_spe.condition = con
info_spe.date = runs[n].date
info_spe.temperature = T
info_spe.bias = runs[n].bias
info_spe.baseline_numbins = 50
info_spe.peaks_numbins = 200 #unused
info_spe.data_type = 'h5'
wp = WaveformProcessor(info_spe, run_info_self = runs[n], run_info_solicit = run_spe_solicited, baseline_correct = True, cutoff = (range_lows2[n],range_highs2[n]), centers = [centers2[n]*a for a in range(1,numpeaks[n]+1)], peak_range = (1,numpeaks[n]))
wp.process(do_spe = True, do_alpha = False)
wp.plot_both_histograms(with_baseline_fit=True,with_fit=True)
# wp.plot_peak_histograms()
# wp.get_subtract_hist_mean(runs2[8].all_led_peak_data,runs2[8].all_dark_peak_data,plot = True)
wp.plot_spe()
#%%
campaign_spe = []
for i in range(1,len(runs)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'LXe'
    info_spe.date = runs[i].date
    info_spe.temperature = 169
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 150
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, cutoff = (range_lows2[i],range_highs2[i]), centers = [centers2[i]*a for a in range(1,numpeaks[i]+1)], peak_range = (1,numpeaks[i]))
    wp_spe.process(do_spe = True, do_alpha = False)
    wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/August 1/silicon_gauss_'+str(info_spe.bias)+'.png')
    wp_spe.plot_spe(savefig=True, path = 'C:/Users/Hannah/Downloads/August 1/silicon_spe_'+str(info_spe.bias)+'.png')
    wp_spe.plot_both_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/August 1/siicon_baseline_'+str(info_spe.bias)+'.png')
    wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/August 1/silicon_gauss_'+str(info_spe.bias)+'.svg')
    wp_spe.plot_spe(savefig=True, path = 'C:/Users/Hannah/Downloads/August 1/silicon_spe_'+str(info_spe.bias)+'.svg')
    wp_spe.plot_both_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/August 1/silicon_both_'+str(info_spe.bias)+'.svg')
    plt.close()
    campaign_spe.append(wp_spe)
    print(info_spe.bias)

#%% plot linear fit to the breakdown voltage
curr_campaign = campaign_spe
spe = SPE_data(curr_campaign, invC_filter, invC_err_filter, filtered = True)
spe.plot_spe(in_ov = False, absolute = False, out_file = 'C:/Users/Hannah/Downloads/August 1/August1_silicon_vbd.csv') #
spe.plot_CA(out_file = None) 
