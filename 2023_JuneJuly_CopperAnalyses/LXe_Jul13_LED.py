# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:13:35 2023

@author: heps1
"""
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
import csv
plt.style.use('C:/Users/Hannah/Documents/GitHub/SPE-Analysis/nexo_new.mplstyle')

#%%
path ='C:/Users/Hannah/Downloads/405nm/'
invC_filter = 0.01171
invC_err_filter = 0.000048
# 405 nm 2.55V
run_spe = RunInfo([path+'Run_1689326327.hdf5'], do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
upper_limit = [0.6, 0.6, 0.72, 1.5, 1.5, 1.6, 1.7, 1.5, 1.5, 1.5] #SPE
upper_limit_CA = [0.74, 0.74, 0.74, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75] #CA
proms = [0.0046, 0.0046, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.007, 0.0072, 0.0072] #SPE
proms_CA = [0.0046, 0.0046, 0.0046, 0.0046, 0.0046, 0.0046, 0.0046, 0.0046, 0.0046, 0.0046] #CA
files = ['Run_1689330664','Run_1689325599', 'Run_1689328582', 'Run_1689326523', 'Run_1689327971', 'Run_1689327247', 'Run_1689324090', 'Run_1689329925', 'Run_1689329252', 'Run_1689324841']
runs_spe = []
for file in range(len(files)):
    run = RunInfo([path + files[file] + '.hdf5'], do_filter = True, upper_limit = upper_limit[file], baseline_correct = True, prominence = proms[file], is_led = True, poly_correct = True, condition='LXe')
    runs_spe.append(run)
biases = [run.bias for run in runs_spe] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
#%%
runs_CA = []
for file in range(len(files)):
    run = RunInfo([path + files[file] + '.hdf5'], do_filter = True, upper_limit = upper_limit_CA[file], baseline_correct = True, prominence = proms_CA[file], is_led = True, poly_correct = True, condition='LXe')
    runs_CA.append(run)
biases_CA = [run.bias for run in runs_CA]
#%%
def get_subtract_hist_mean(bias, data1, data2, numbins = 1000, plot = False):
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
    subtracted_counts[subtracted_counts < -50] = 0
    if plot:
        plt.step(centers, subtracted_counts, label = 'subtracted hist')
        plt.legend()
        
    big_n = np.sum(subtracted_counts)
    norm_subtract_hist = subtracted_counts / big_n
    
    mean_value = np.sum(centers * norm_subtract_hist)
    a = np.sum(subtracted_counts * centers)
    mean_err = np.sqrt(np.sum( ((a - centers * big_n)/ big_n ** 2) ** 2 * (counts1 + counts2)))
    
    if plot:
        plt.title(f'Bias: {bias}V, mean_value: {mean_value:0.5} +/- {mean_err:0.1}')
        plt.axvline(x = mean_value, color = 'green')
    
    return (mean_value, mean_err)
    
#%%
mean_values = []
mean_errs = []
for run in runs_spe:
    mean_val, mean_err = get_subtract_hist_mean(run.bias,run.all_led_peak_data, run.all_dark_peak_data, plot = True, numbins = 4000)
    mean_values.append(mean_val)
    mean_errs.append(mean_err)
#%%
import pandas as pd
data = {
        "mean": mean_values,
        "mean error": mean_errs,
        "Bias Voltage [V]": biases,
        "Bias Voltage [V] error": [0.0025*V + 0.015 for V in biases]
        }
df = pd.DataFrame(data)
df.to_csv('July13_means_subtraction_method_for_CA.csv')
#%%#%% variable peaks
range_lows2 = [0.0046,0.005,0.0048,0.0048,0.0049,0.0054,0.0066,0.00635, 0.00615, 0.00615] 
centers2 = [0.005,0.00555,0.006,0.0068,0.0075,0.008,0.0086, 0.0092, 0.0099, 0.0101] #
range_highs2 = [0.026,0.0264,0.034,0.031,0.0337,0.035,0.039, 0.0419, 0.0435, 0.0465]
numpeaks = [4,4,5,4,4,4,4,4,4,4]

#%%
campaign_spe2 = []
for i in range(1,len(runs_spe)):
    peak_num = (1,numpeaks[i])
    T = 170
    con = 'LXe'
    info_spe = MeasurementInfo()
    info_spe.condition = con
    info_spe.date = runs_spe[i].date
    info_spe.temperature = T
    info_spe.bias = runs_spe[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs_spe[i], run_info_solicit = run_spe, baseline_correct = True, cutoff = (range_lows2[i],range_highs2[i]), centers = [centers2[i]*a for a in range(1,numpeaks[i]+1)], peak_range = (1,numpeaks[i]))
    wp_spe.process(do_spe = True, do_alpha = False)
    # wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/July 13/July13_LXe_gauss_'+str(run.bias)+'.png')
    # wp_spe.plot_spe(savefig=True, path = 'C:/Users/Hannah/Downloads/July 13/July13_LXe_spe_'+str(run.bias)+'.png')
    # wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/July 13/July13_LXe_gauss_'+str(run.bias)+'.svg')
    # wp_spe.plot_spe(savefig=True, path = 'C:/Users/Hannah/Downloads/July 13/July13_LXe_spe_'+str(run.bias)+'.svg')
    plt.close()
    campaign_spe2.append(wp_spe)
#%%
curr_campaign = campaign_spe2

filtered_spe = SPE_data(curr_campaign, invC_filter, invC_err_filter, filtered = True)
filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = None)

print(filtered_spe.v_bd)
print(filtered_spe.v_bd_err)
#%%