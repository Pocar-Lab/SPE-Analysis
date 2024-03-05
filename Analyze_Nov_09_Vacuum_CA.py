#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:45:31 2023

@author: albertwang
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
plt.style.use('D:/Xe/AnalysisScripts/LXe May 2023/nexo_new.mplstyle')

#%%
files2 = ['Run_1699562899.hdf5','Run_1699558774.hdf5','Run_1699561870.hdf5','Run_1699555259.hdf5','Run_1699562338.hdf5','Run_1699555782.hdf5','Run_1699556577.hdf5','Run_1699558155.hdf5','Run_1699554230.hdf5','Run_1699561179.hdf5','Run_1699557334.hdf5','Run_1699560648.hdf5'] #32,32.5V,33V,33.5V,34V,34.5V,35V,35.5V,36V,36.5V,37V,37.5V
proms2 = [0.0041,0.0042,0.0043,0.0043,0.0044,0.0045,0.0048,0.005,0.0051,0.0051,0.0051,0.0054,0.0056] 
upperlim2 = [0.079,0.079,0.399,0.399,0.399,0.399,0.399,0.99,0.99,0.99,1.99,1.99] 
runs2 = []
for file in range(10):
    run_spe2 = RunInfo(['D:/Xe/DAQ/' + files2[file]], specifyAcquisition = False, do_filter = True, upper_limit = upperlim2[file], baseline_correct = True, prominence = proms2[file], is_led = True)
    runs2.append(run_spe2)
biases2 = [run.bias for run in runs2]

invC_spe_filter2 = 0.01171
invC_spe_err_filter2 = 0.000048
#%% split into two so as not to overload memory
# files2 = ['Run_1699562899.hdf5','Run_1699558774.hdf5','Run_1699561870.hdf5','Run_1699555259.hdf5','Run_1699562338.hdf5','Run_1699555782.hdf5','Run_1699556577.hdf5','Run_1699558155.hdf5','Run_1699554230.hdf5','Run_1699561179.hdf5','Run_1699557334.hdf5','Run_1699560648.hdf5'] #32,32.5V,33V,33.5V,34V,34.5V,35V,35.5V,36V,36.5V,37V,37.5V
# proms2 = [0.0065,0.0065,0.007,0.007,0.0075,0.0075,0.0075,0.0075,0.0075,0.0075,0.008,0.008] 
# upperlim2 = [0.1,0.1,1,1,1,1,1,1,2,1,2,1] 
# runs2 = []
for file in range(10,12):
    run_spe2 = RunInfo(['D:/Xe/DAQ/' + files2[file]], specifyAcquisition = False, do_filter = True, upper_limit = upperlim2[file], baseline_correct = True, prominence = proms2[file], is_led = True)
    runs2.append(run_spe2)
biases2 = [run.bias for run in runs2]


invC_spe_filter2 = 0.01171
invC_spe_err_filter2 = 0.000048

#%%
runs2[2].plot_hists('','')
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
for run in runs2:
    get_subtract_hist_mean(run.bias,run.all_led_peak_data, run.all_dark_peak_data, plot = True)

#%%

for run in runs2:
    run.plot_led_dark_hists('169', '.')
    # if run != runs2[-1]:
    plt.figure()
    plt.savefig('D:/Images/No Source/Hists/'+str(run.bias)+'.png')
    plt.savefig('D:/Images/No Source/Hists/'+str(run.bias)+'.svg')


#%%
run_spe_solicited2 = RunInfo(['D:/Xe/DAQ/Run_1699564042.hdf5'], do_filter = True, is_solicit = True, upper_limit = .08, baseline_correct = True, plot_waveforms = False, fourier = False)
#%% assume 4 peaks
range_lows2 = [0.0037, 0.0041,0.004126,0.00536,0.0045,0.0049,0.0057,0.0065,0.0066,0.0073,0.0074,0.0076] 
centers2 = [0.0053,0.0057,0.0062,0.007,0.00707,0.00750,0.00776,0.00846,0.0095,0.01,0.0105,0.011] #
range_highs2 = [0.023,0.0255,0.0272,0.0304,0.0323,0.035,0.037,0.039,0.041,0.0437,0.0469,0.0476] #only use from 33V and onwards
#%% variable peaks
range_lows2 = [0.00399, 0.0041,0.004126,0.00435,0.0045,0.0049,0.0055,0.0065,0.0066,0.0073,0.0074,0.0076] 
centers2 = [0.0052,0.0057,0.0062,0.00680,0.00707,0.00750,0.00776,0.00846,0.0095,0.01,0.0105,0.011] #
range_highs2 = [0.0182,0.0255,0.0272,0.037,0.04,0.052,0.055,0.058,0.061,0.064,0.067,0.07] #only use from 33V and onwards
numpeaks = [3,4,4,5,5,6,6,6,6,6,6,6]
#%%

n=1 #test the different biases
T = 169
con = 'LXe'
info_spe = MeasurementInfo()
info_spe.condition = con
info_spe.date = runs2[n].date
info_spe.temperature = T
info_spe.bias = runs2[n].bias
info_spe.baseline_numbins = 50
info_spe.peaks_numbins = 200 #unused
info_spe.data_type = 'h5'
wp = WaveformProcessor(info_spe, run_info_self = runs2[n], run_info_solicit = run_spe_solicited2, baseline_correct = True,  range_low = range_lows2[n], range_high = range_highs2[n], center = centers2[n], peak_range = (1,numpeaks[n]), status=0)
wp.process(do_spe = True, do_alpha = False)
wp.plot_both_histograms()
wp.plot_peak_histograms()
# wp.get_subtract_hist_mean(runs2[8].all_led_peak_data,runs2[8].all_dark_peak_data,plot = True)
wp.plot_spe()


#%%
campaign_spe2 = []
for i in range(0,2):
    peak_num = (1,numpeaks[i])
    T = 169
    con = 'LXe'
    info_spe = MeasurementInfo()
    info_spe.condition = con
    info_spe.date = runs2[i].date
    info_spe.temperature = T
    info_spe.bias = runs2[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs2[i], run_info_solicit = run_spe_solicited2, baseline_correct = True, range_low = range_lows2[i], range_high = range_highs2[i], center = centers2[i], peak_range = peak_num, status = 0)
    wp_spe.process(do_spe = True, do_alpha = False)
    campaign_spe2.append(wp_spe)

#%%

#%%
for run in campaign_spe2:
    # run.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/No Source Run/Nov_169K_gauss_'+str(run.info.bias)+'.png')
    run.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/No Source Run/Nov_169K_spe_'+str(run.info.bias)+'.png')
    plt.close()
#%%
curr_campaign2 = campaign_spe2[:-2]

filtered_spe2 = SPE_data(curr_campaign2, invC_spe_filter2, invC_spe_err_filter2, filtered = True)

filtered_spe2.plot_spe(in_ov = False, absolute = False, out_file = 'D:/Raw Results/Breakdown Voltage/Nov9_169K_vbd_bias_NoSource_cut.csv')
# filtered_spe2.plot_CA(out_file = 'D:/Raw Results/Correlated Avalanche Data/Nov9_169K_CA_ov_NoSource.csv')
# filtered_spe2.plot_spe(in_ov = False, absolute = False, out_file = None)
filtered_spe2.plot_CA()

#%%
csv_name = 'Nov9_405nm_CA_NoSource.csv'
with open(csv_name, 'w') as csvfile:
    field_names = ['OV', 'OV error','num_CA','num_CA error']
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    for i in range(len(filtered_spe2.ov)):
        ov_values = str(filtered_spe2.ov[i])
        ov_values_error = str(filtered_spe2.ov_err[i])
        num_CA = str(filtered_spe2.CA_vals[i])
        num_CA_error = str(filtered_spe2.CA_err[i])
        writer.writerow({'OV':ov_values,'OV error':ov_values_error,'num_CA':num_CA,'num_CA error':num_CA_error})








