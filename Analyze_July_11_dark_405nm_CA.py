#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:37:16 2023

@author: albertwang
"""
import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
sys.path.append('/Users/albertwang/Desktop/nEXO')   
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

#%%
# def get_subtract_hist_mean(data1, data2, bias = '', numbins = 500, plot = False):
#     if plot:
#         plt.figure()
#         (n, b, p) = plt.hist(data1, bins = numbins, density = False, label = 'data 1', histtype='step')
#         plt.axvline(x = np.mean(data1), color = 'blue')
#         print('LED on hist: ' + str(np.mean(data1)))
#         print('LED off hist: ' + str(np.mean(data2)))
#         plt.axvline(x = np.mean(data2), color = 'orange')
#         plt.hist(data2, bins = b, density = False, label = 'data 2', histtype='step')
#     counts1, bins1 = np.histogram(data1, bins = numbins, density = False)
#     counts2, bins2 = np.histogram(data2, bins = bins1, density = False)
#     centers = (bins1[1:] + bins1[:-1])/2
#     subtracted_counts = counts1 - counts2
#     # subtracted_counts[subtracted_counts < 0] = 0
#     if plot:
#         plt.step(centers, subtracted_counts, label = 'subtracted hist')
#         plt.legend()
        
#     norm_subtract_hist = subtracted_counts / np.sum(subtracted_counts)
#     # weights = 1.0 / subtracted_counts / 
#     mean_value = np.sum(centers * norm_subtract_hist)
#     if plot:
#         plt.axvline(x = mean_value, color = 'green')
#         plt.title('The bias: ' + str(bias) + ', The mean value: ' + str(mean_value))
#     return mean_value

#%%  405nm CA analysis

# test_high_bias = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/july_11th_dark_data/CA data/405nm/Run_1689177434.hdf5'], do_filter = True, upper_limit = 1, baseline_correct = True, prominence = 0.007, plot_waveforms= False, is_led = True)
# bias_test = test_high_bias.run_meta_data['/Users/albertwang/Desktop/nEXO/HDF Data/july_11th_dark_data/CA data/405nm/Run_1689177434.hdf5']['RunNotes']
# get_subtract_hist_mean(test_high_bias.all_led_peak_data, test_high_bias.all_dark_peak_data, bias = bias_test ,plot = True)
# # test_high_bias.plot_hists('169', '.') #regular histogram
# # test_high_bias.plot_peak_waveform_hist() #2D plot
# # test_high_bias.plot_led_dark_hists('171', '.') #LED comparison plot

#%%

files2 = ['Run_1689176196.hdf5', 'Run_1689173976.hdf5','Run_1689175049.hdf5','Run_1689175759.hdf5','Run_1689178348.hdf5','Run_1689177955.hdf5','Run_1689176952.hdf5','Run_1689177434.hdf5'] #31.5V, 32V, 32.5, 33.0V, 33.5V, 34.0V, 34.5V, 35.0V 
proms2 = [0.0065, 0.0063,0.0065,0.0065,0.0065,0.0065,0.0065,0.007] 
upperlim2 = [1 for file in files2] 
runs2 = []
for file in range(len(files2)):
    run_spe2 = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/july_11th_dark_data/CA data/405nm/' + files2[file]], specifyAcquisition = False, do_filter = False, upper_limit = upperlim2[file], baseline_correct = True, prominence = proms2[file], is_led = True)
    runs2.append(run_spe2)
biases2 = [run.bias for run in runs2]




invC_spe_filter2 =  0.0132 
invC_spe_err_filter2 = 0.000089 

#%% take a look at waveforms (solicited)
run_spe_solicited2 = RunInfo(['/Users/albertwang/Desktop/nEXO/HDF Data/july_11th_dark_data/CA data/405nm/Run_1689175515.hdf5'], do_filter = False, is_solicit = True, upper_limit = .02, baseline_correct = True, plot_waveforms = False, fourier = False)
#%% take a look at finger plots 

# runs2[1].plot_led_dark_hists('169', '.')
for i in range(len(files2)):
    plt.figure()
    runs2[i].plot_led_dark_hists('171', '.')

#%% set the initial guesses and fitting range

range_lows2 = [12,0.00465,0.00495, 0.004984, 0.00540,0.00523, 0.00583, 0.00592] 
centers2 = [12,0.006174, 0.006249, 0.006636, 0.00704, 0.00723,0.00786, 0.00834] #
range_highs2 = [12,0.022,0.0246, 0.02671, 0.02947, 0.03110,0.0332, 0.03658] #only use from 33V and onwards

#%% plot with peak fit - test

# n= 6#test the different biases
# T = 171
# con = 'GN'
# info_spe = MeasurementInfo()
# info_spe.condition = con
# info_spe.date = runs2[n].date
# info_spe.temperature = T
# info_spe.bias = runs2[n].bias
# info_spe.baseline_numbins = 50
# info_spe.peaks_numbins = 200 #unused
# info_spe.data_type = 'h5'
# wp = WaveformProcessor(info_spe, run_info_self = runs2[n], run_info_solicit = run_spe_solicited2, baseline_correct = True,  range_low = range_lows2[n], range_high = range_highs2[n], center = centers2[n], peak_range = (1,4))
# wp.process(do_spe = True, do_alpha = False)
# wp.plot_both_histograms()
# wp.plot_peak_histograms()
# wp.plot_spe()

#%% make a list of ProcessWaveforms objects
campaign_spe2 = []

for i in range(3, len(runs2)):
    peak_num = ()
    if i == 0: #these are Gauss fitted with 3 peaks (32V and 32.5V)
        continue
    else: 
        peak_num = (1,4)
        T = 171
        con = 'GN'
        info_spe = MeasurementInfo()
        info_spe.condition = con
        info_spe.date = runs2[i].date
        info_spe.temperature = T
        info_spe.bias = runs2[i].bias
        info_spe.baseline_numbins = 50
        info_spe.peaks_numbins = 200
        info_spe.data_type = 'h5'
        wp_spe = WaveformProcessor(info_spe, run_info_self = runs2[i], run_info_solicit = run_spe_solicited2, baseline_correct = True, range_low = range_lows2[i], range_high = range_highs2[i], center = centers2[i], peak_range = peak_num)
        wp_spe.process(do_spe = True, do_alpha = False)
        campaign_spe2.append(wp_spe)
    
# campaign_spe2 = []

# for i in range(1, len(runs2)):
#     peak_num = ()
#     if i == 2: #these are Gauss fitted with 3 peaks (32V and 32.5V)
#         continue
#     else: 
#         peak_num = (1,4)
#     T = 171
#     con = 'GN'
#     info_spe = MeasurementInfo()
#     info_spe.condition = con
#     info_spe.date = runs2[i].date
#     info_spe.temperature = T
#     info_spe.bias = runs2[i].bias
#     info_spe.baseline_numbins = 50
#     info_spe.peaks_numbins = 200
#     info_spe.data_type = 'h5'
#     wp_spe = WaveformProcessor(info_spe, run_info_self = runs2[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i], peak_range = peak_num, status = 1)
#     wp_spe.process(do_spe = True, do_alpha = False)
#     campaign_spe2.append(wp_spe)
    
# campaign_spe3 = []

# for i in range(1,len(runs2)):
#     peak_num = ()
#     if i == 2: #these are Gauss fitted with 3 peaks (32V and 32.5V)
#         continue
#     else: 
#         peak_num = (1,4)
#     T = 171
#     con = 'GN'
#     info_spe = MeasurementInfo()
#     info_spe.condition = con
#     info_spe.date = runs2[i].date
#     info_spe.temperature = T
#     info_spe.bias = runs2[i].bias
#     info_spe.baseline_numbins = 50
#     info_spe.peaks_numbins = 200
#     info_spe.data_type = 'h5'
#     wp_spe = WaveformProcessor(info_spe, run_info_self = runs2[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i], peak_range = peak_num, status = 2)
#     wp_spe.process(do_spe = True, do_alpha = False)
#     campaign_spe3.append(wp_spe)

    
#%% plot linear fit to the breakdown voltage
    
# curr_campaign = campaign_spe

# filtered_spe = SPE_data(curr_campaign, invC_spe_filter, invC_spe_err_filter, filtered = False) #not filtered for this liquefaction
# filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = None)

# print(filtered_spe.v_bd)
# print(filtered_spe.v_bd_err)

#%%

for i in range(len(campaign_spe2)):
    campaign_spe2[i].plot_peak_histograms()
    campaign_spe2[i].plot_spe()


#%% plot linear fit to the breakdown voltage
    
curr_campaign2 = campaign_spe2

filtered_spe2 = SPE_data(curr_campaign2, invC_spe_filter2, invC_spe_err_filter2, filtered = False)

filtered_spe2.plot_CA()
# filtered_spe2.plot_spe(in_ov = False, absolute = False, out_file = None)
#%% plot linear fit to the breakdown voltage
    
# curr_campaign3 = campaign_spe3

# filtered_spe3 = SPE_data(curr_campaign3, invC_spe_filter, invC_spe_err_filter, filtered = False)
# filtered_spe3.plot_spe(in_ov = False, absolute = False, out_file = None)
#%% plot linear fit to the breakdown voltage

# breakdown_vals = []
# breakdown_err_vals= []
# conditions = ['GN', 'LXe', 'LED-on']
# fig = plt.figure()

# # filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = None)
# # breakdown_vals.append(filtered_spe.v_bd)
# # breakdown_err_vals.append(filtered_spe.v_bd_err)
# filtered_spe2.plot_spe(in_ov = False, absolute = False, out_file = None)
# breakdown_vals.append(filtered_spe2.v_bd)
# breakdown_err_vals.append(filtered_spe2.v_bd_err)
# # filtered_spe3.plot_spe(in_ov = False, absolute = False, out_file = None)
# breakdown_vals.append(filtered_spe3.v_bd)
# breakdown_err_vals.append(filtered_spe3.v_bd_err)

# textstr = f'Date: {filtered_spe.campaign[0].info.date}\n'
# textstr += f'Condition: {filtered_spe.campaign[0].info.condition}\n'
# textstr += f'RTD4: {filtered_spe.campaign[0].info.temperature} [K] (nominal) \n'
# textstr += f'LED: 405 [nm] \n'
# textstr += f'V$_L$$_E$$_D$: 2.5500 $\pm$ 0.0005 [V] \n'

# for i in range(3):
#     textstr += f'\n--\n'
#     textstr += rf'V$_b$$_d$ ({conditions[i]}): {breakdown_vals[i]:0.3} $\pm$ {breakdown_err_vals[i]:0.1} [V]'
    

# props = dict(boxstyle='round', facecolor= 'blue', alpha=0.2)
# fig.text(0.72, 0.48, textstr, fontsize=8,
#     verticalalignment='top', bbox=props)
