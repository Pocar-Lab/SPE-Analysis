# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:13:35 2023

@author: heps1
"""
import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
# sys.path.append('/Users/albertwang/Desktop/nEXO')   
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
plt.style.use('nexo_new.mplstyle')

#%%
path = 'C:/Users/Hannah/OneDrive - University of Massachusetts/Dataruns_2023-24/DAQ_folder_google_drive/June 28th & July 11th 13th Data/July 11th (Dark)/july_11th_dark_data/CA data/405nm/'
invC_filter = 0.01171
invC_err_filter = 0.000048
# 405 nm 2.55V
run_spe_solicited_CA = RunInfo([path+'Run_1689175515.hdf5'], do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
files = ['Run_1689178348','Run_1689177955', 'Run_1689177434', 'Run_1689176952', 'Run_1689175759', 'Run_1689175049'] #, 'Run_1689173976'
runs_CA = []
for file in range(len(files)):
    # run_spe_CA = RunInfo(['D:/Xe/DAQ/' + files[file] + '.hdf5'], do_filter = True, upper_limit = 1, baseline_correct = True, prominence = 0.0052, is_led = True)
    run_spe_CA = RunInfo([path + files[file] + '.hdf5'], do_filter = True, upper_limit = 1, baseline_correct = True, prominence = 0.0052, is_led = True)
    runs_CA.append(run_spe_CA)
biases = [run.bias for run in runs_CA] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
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
    subtracted_counts[subtracted_counts < -0.5*max(subtracted_counts)] = 0
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
for run in runs_CA:
    get_subtract_hist_mean(run.bias,run.all_led_peak_data, run.all_dark_peak_data, plot = True)


#%%#%% variable peaks
range_lows2 = [0.0052,0.005,0.0056,0.0056,0.0052,0.005] 
centers2 = [0.0068,0.0071,0.0082,0.008,0.0065,0.0056] #
range_highs2 = [0.031,0.033,0.0535,0.0435,0.0275,0.0245]
numpeaks = [4,4,6,5,4,4]

#%%
i=5 #test the different biases
T = 169
peak_num = (1,numpeaks[i])
con = 'LXe'
info_spe = MeasurementInfo()
info_spe.condition = con
info_spe.date = runs_CA[i].date
info_spe.temperature = T
info_spe.bias = runs_CA[i].bias
info_spe.baseline_numbins = 50
info_spe.peaks_numbins = 200 #unused
info_spe.data_type = 'h5'
wp =WaveformProcessor(info_spe, run_info_self = runs_CA[i], run_info_solicit = run_spe_solicited_CA, baseline_correct = True, cutoff = (range_lows2[i],range_highs2[i]), centers = [centers2[i]*a for a in range(1,numpeaks[i]+1)], peak_range = peak_num)
wp.process(do_spe = True, do_alpha = False)
wp.plot_both_histograms(with_fit = True, with_baseline_fit = True)
wp.plot_peak_histograms()
# wp.get_subtract_hist_mean(runs2[8].all_led_peak_data,runs2[8].all_dark_peak_data,plot = True)
wp.plot_spe()


#%%
campaign_spe2 = []
for i in range(0,len(runs_CA)):
    peak_num = (1,numpeaks[i])
    T = 171
    con = 'LXe'
    info_spe = MeasurementInfo()
    info_spe.condition = con
    info_spe.date = runs_CA[i].date
    info_spe.temperature = T
    info_spe.bias = runs_CA[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs_CA[i], run_info_solicit = run_spe_solicited_CA, subtraction_method=True, baseline_correct = True, cutoff = (range_lows2[i],range_highs2[i]), centers = [centers2[i]*a for a in range(1,numpeaks[i]+1)], peak_range = peak_num)
    wp_spe.process(do_spe = True, do_alpha = False)
    campaign_spe2.append(wp_spe)

#%% plots
for i in range(len(campaign_spe2)):
    campaign_spe2[i].plot_peak_histograms()

#%%
path='C:/Users/Hannah/Documents/GitHub/SPE-Analysis/2023_July_GN_CA/'
for run in campaign_spe2:
    run.plot_peak_histograms(log_scale = False, savefig=True, path = path+'July_gauss_'+str(run.info.bias)+'.png')
    run.plot_spe(savefig=True, path = path+'July_spe_'+str(run.info.bias)+'.png')
    # plt.close()
#%%
curr_campaign = campaign_spe2

filtered_spe = SPE_data(curr_campaign, invC_filter, invC_err_filter, filtered = True)
filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = None)

print(filtered_spe.v_bd)
print(filtered_spe.v_bd_err)

# filtered_spe.get_CA_ov(8)
# spe.get_CA_ov(8)
#%%
path='C:/Users/Hannah/Documents/Raw Results/'
filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = path+'Breakdown Voltage/July13_171K_SPE_GN.csv')
filtered_spe.plot_CA(out_file = path+'Correlated Avalanche Data/July13_171K_CA_GN.csv')