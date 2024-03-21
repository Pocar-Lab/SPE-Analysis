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
invC_spe_filter2 = 0.01171
invC_spe_err_filter2 = 0.000048
# separate files for each bias voltage -> specifyAcquisition = False
path ='C:/Users/Hannah/OneDrive - University of Massachusetts/Dataruns_2023-24/DAQ_folder_google_drive/June 28th & July 11th 13th Data/June 28th/SPE/'
dir_list = os.listdir(path)
print(dir_list) # print out all the files in this directory
#%%
# run_spe_solicited = RunInfo([path+'Run_1666778594.hdf5'],  specifyAcquisition = True, acquisition ='Acquisition_1666781843', do_filter = True, is_solicit = True, upper_limit = .5, baseline_correct = True) #this is not the correct baseline data?? we don't seem to have any for this run.
#       SPE & CA                 SPE & CA                SPE & CA             CA only                CA only             SPE & CA                CA only
files_CA =['Run_1687986955.hdf5','Run_1687988853.hdf5','Run_1687989375.hdf5','Run_1687989654.hdf5','Run_1687989878.hdf5','Run_1687990206.hdf5','Run_1687988498.hdf5']
files =['Run_1687986955.hdf5','Run_1687988853.hdf5','Run_1687989375.hdf5','Run_1687990206.hdf5']
upperlim = [1.25, 1.29, 1.25, 1.25]
upperlim_CA = [1.25, 1.29, 1.25, 0.059, 0.059, 1.25, 0.059]
proms = [0.004, 0.004, 0.0035, 0.004]
runs = []
for file in range(len(files)):
    run_spe = RunInfo([path+files[file]], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], poly_correct = True, poly_correct_degree=3, is_led = True)
    runs.append(run_spe)
biases = [run.bias for run in runs]
run_spe_baseline = RunInfo([path+'Run_1687988498.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.0125, baseline_correct = True, do_peak_find = False, poly_correct = True, num_waveforms = 0)
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
    subtracted_counts[subtracted_counts < -0.33*max(subtracted_counts)] = 0
    if plot:
        plt.step(centers, subtracted_counts, label = 'subtracted hist')
        plt.legend()
        
    big_n = np.sum(subtracted_counts)
    norm_subtract_hist = subtracted_counts / big_n
    mean_value = np.sum(centers * norm_subtract_hist)
    if plot:
        plt.title(f'Bias: {bias}V, mean_value: {mean_value}')
        plt.axvline(x = mean_value, color = 'green')  
    # mean_err = np.sum((centers/big_n) ** 2)(subtracted_counts) + (np.sum(subtracted_counts*centers)/(big_n)**2) ** 2 * (np.sum(subtracted_counts)) #overestimation
    a = np.sum(subtracted_counts * centers)
    mean_err = np.sqrt(np.sum( ((a - centers * big_n)/ big_n ** 2) ** 2 * (counts1 + counts2)))
    
    return (mean_value, mean_err)
#%% create a separate set of RunInfo objects for CA (without peak finding)
runs_CA = []
for i in range(len(files_CA)):
    runs_CA.append(RunInfo([path+files_CA[i]], specifyAcquisition = False, do_filter = True, 
                           upper_limit = upperlim_CA[i], baseline_correct = True, do_peak_find = False, 
                           poly_correct = True, poly_correct_degree = 3, is_led = True, num_waveforms = 0, 
                           plot_waveforms = False))
biases_CA = [run.bias for run in runs_CA]
#%% save the mean values of the subtracted histogram
mean_values = []
mean_errs = []
for run in runs_CA:
    mean_val, mean_err = get_subtract_hist_mean(run.bias,run.all_led_peak_data, run.all_dark_peak_data, plot = False, numbins = 2000)
    mean_values.append(mean_val)
    mean_errs.append(mean_err)
# runs_spe[3].plot_hists()
#%%
import pandas as pd
data = {
        "mean": mean_values,
        "mean error": mean_errs,
        "Bias Voltage [V]": biases_CA,
        "Bias Voltage [V] error": [0.0025*V + 0.015 for V in biases_CA]
        }
df = pd.DataFrame(data)
df.to_csv('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/June2023_means_subtraction_method_all_amplitudes.csv')
#%% variable peaks
range_lows2 = [0.0037,0.005,0.0043,0.004] 
centers2 = [0.006,0.008,0.007,0.0051] #
range_highs2 = [0.029,0.048,0.0338,0.0195]
numpeaks = [4,5,4,3]
#%%
n=3 #test the different biases
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
wp = WaveformProcessor(info_spe, run_info_self = runs[n], run_info_solicit = run_spe_baseline, baseline_correct = True, cutoff = (range_lows2[n],range_highs2[n]), centers = [centers2[n]*a for a in range(1,numpeaks[n]+1)], peak_range = (1,numpeaks[n]), subtraction_method=True)
wp.process(do_spe = True, do_alpha = False)
wp.plot_both_histograms(with_fit=True,with_baseline_fit=True)
wp.plot_peak_histograms()
wp.plot_spe()
#%%
campaign_spe = []
for i in range(len(runs)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'LXe'
    info_spe.date = runs[i].date
    info_spe.temperature = 169
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 150
    info_spe.data_type = 'h5'
    num = numpeaks[i]
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_baseline, baseline_correct = True, cutoff = (range_lows2[i],range_highs2[i]), centers = [centers2[i]*a for a in range(1,numpeaks[i]+1)], peak_range = (1,numpeaks[i]), subtraction_method=True)
    wp_spe.process(do_spe = True, do_alpha = False)
    wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/June 28/06282023_gauss_'+str(info_spe.bias)+'.png')
    wp_spe.plot_spe(savefig=True, path = 'C:/Users/Hannah/Downloads/June 28/06282023_spe_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_both_histograms(with_fit = True, with_baseline_fit = True, log_scale = True, savefig=False, path = 'C:/Users/Hannah/Downloads/June 28/06282023_baseline_'+str(info_spe.bias)+'.png')
    wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/June 28/06282023_gauss_'+str(info_spe.bias)+'.svg')
    wp_spe.plot_spe(savefig=True, path = 'C:/Users/Hannah/Downloads/June 28/06282023_spe_'+str(info_spe.bias)+'.svg')
    # wp_spe.plot_both_histograms(with_fit = True, with_baseline_fit = True, log_scale = True, savefig=False, path = 'C:/Users/Hannah/Downloads/June 28/06282023_baseline_'+str(info_spe.bias)+'.svg')
    plt.close()
    campaign_spe.append(wp_spe)
    print(info_spe.bias)

#%%
# for run in campaign_spe:
#     run.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/No Source Run/svg/June_gauss_'+str(run.info.bias)+'.svg')
#     run.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/No Source Run/svg/June_spe_'+str(run.info.bias)+'.svg')
#     plt.close()
#%% plot linear fit to the breakdown voltage
curr_campaign = campaign_spe
spe = SPE_data(curr_campaign, invC_spe_filter2, invC_spe_err_filter2, filtered = True)
# spe.plot_spe(in_ov = False, absolute = False, out_file = None) 
spe.plot_spe(in_ov = False, absolute = False, out_file = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/LXe_June_28_2023_vbd_bias_include_32V.csv') 
# spe.plot_CA(out_file = None)
