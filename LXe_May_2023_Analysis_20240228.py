# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:28:38 2023

@author: hpeltzsmalle
"""

import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
# sys.path.append('D:/Xe/AnalysisScripts/LXe May 2023/')   
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
plt.style.use('C:/Users/Hannah/Documents/GitHub/SPE-Analysis/nexo_new.mplstyle')

#%% SPE-VBD
#100ns, new shaper, 10x gain, filtered 
invC_spe_filter =  0.011959447603692185
invC_spe_err_filter = 3.881945391072933E-05
path ='C:/Users/Hannah/OneDrive - University of Massachusetts/Dataruns_2023-24/DAQ_folder_google_drive/20230518_SPE_Alpha_LXe/'
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
# run_spe_solicited = RunInfo(['D:/Xe/DAQ/Run_1686747967.hdf5'],  specifyAcquisition = False, do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
# run_spe_solicited = RunInfo(['D:/Xe/DAQ/Run_1691696340.hdf5'],  specifyAcquisition = False, do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True, poly_correct = True)
# run_spe_baseline = RunInfo(['D:/Xe/DAQ/'+files[0]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.032, baseline_correct = True, do_peak_find = False, poly_correct = True, num_waveforms = 9000)
# files = ['Run_1684433317','Run_1684434365','Run_1684434800','Run_1684435362','Run_1684435943','Run_1684436416','Run_1684436838','Run_1684437151','Run_1684437508'] #32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36
files = ['Run_1684433317','Run_1684434365','Run_1684434800','Run_1684435362','Run_1684435943','Run_1684436838','Run_1684437508'] #32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36
upperlim = [0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 1, 1.2, 1.2]
runs = []
for file in range(1,len(files)): #len(files)
    run_spe = RunInfo([path+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = 0.0032, poly_correct = True, width = [86,94], poly_correct_degree = 3)
    runs.append(run_spe)
    # runs[file] = run_spe
biases = [run.bias for run in runs]
run_spe_baseline = RunInfo([path+files[0]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.032, baseline_correct = True, do_peak_find = False, poly_correct = True, num_waveforms = 9700)
#%%
#%% CA?
#100ns, new shaper, 10x gain, filtered 
invC_spe_filter =  0.011959447603692185
invC_spe_err_filter = 3.881945391072933E-05
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
#run_spe_solicited = RunInfo(['D:/Xe/DAQ/Run_1686747967.hdf5'],  specifyAcquisition = False, do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
run_spe_solicited = RunInfo(['D:/Xe/DAQ/Run_1691696340.hdf5'],  specifyAcquisition = False, do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
files = ['Run_1684433317','Run_1684434365','Run_1684434800','Run_1684435362','Run_1684435943','Run_1684436416','Run_1684436838','Run_1684437151','Run_1684437508'] #32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36
upperlim = [0.18, 0.18, 0.5, 0.9, 1.25, 1.35, 1.4, 1.6, 1.5]
runs = []
# for file in range(len(files)):
for file in range(0,len(files)):
    run_spe = RunInfo(['D:/Xe/DAQ/'+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = 0.0032, poly_correct = True)
    runs.append(run_spe)
biases = [run.bias for run in runs]
#%% tests with different widths

# test1 = RunInfo(['D:/Xe/DAQ/'+files[2]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.5, baseline_correct = True, prominence = 0.0035, poly_correct = True, plot_waveforms = False, num_waveforms = 0)
# test2 = RunInfo(['D:/Xe/DAQ/'+files[2]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.5, baseline_correct = True, prominence = 0.0035, poly_correct = True, plot_waveforms = False, num_waveforms = 0, width = [80,100])
# test3 = RunInfo(['D:/Xe/DAQ/'+files[2]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.5, baseline_correct = True, prominence = 0.0035, poly_correct = True, plot_waveforms = False, num_waveforms = 0, width = [83,95])
# test4 = RunInfo(['D:/Xe/DAQ/'+files[2]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.5, baseline_correct = True, prominence = 0.0035, poly_correct = True, plot_waveforms = False, num_waveforms = 0, width = [87,93])
# test5 = RunInfo(['D:/Xe/DAQ/'+files[2]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.5, baseline_correct = True, prominence = 0.0035, poly_correct = True, plot_waveforms = False, num_waveforms = 0, width = [87,91])
# test6 = RunInfo(['D:/Xe/DAQ/'+files[2]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.5, baseline_correct = True, prominence = 0.0035, poly_correct = True, plot_waveforms = False, num_waveforms = 0, width = [88,91])
# test7 = RunInfo(['D:/Xe/DAQ/'+files[2]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = 0.5, baseline_correct = True, prominence = 0.0035, poly_correct = True, plot_waveforms = False, num_waveforms = 0, width = [87,92])

# test1.plot_hists()
# test2.plot_hists()
# test3.plot_hists()
# test4.plot_hists()
# test5.plot_hists()
# test6.plot_hists()
# test7.plot_hists()

#%%
cutoffs = []
centers_list=[]
centers_guesses = []

for run in runs:
    high = 3
    if run.bias < 33:
        bins = int(round(np.sqrt(len(run.all_peak_data))))*3
    else:
        bins = int(round(np.sqrt(len(run.all_peak_data))))*3
    print(bins)
    if run.bias > 34.5:
        high = 4
    count, edges = np.histogram(run.all_peak_data, bins=bins)
    centers = (edges[:-1] + edges[1:])/2
    peaks, props = signal.find_peaks(count, prominence=25, distance=3)
    print(peaks)
    
    plt.figure()
    plt.hist(run.all_peak_data, bins=bins)
    
    fitrange = ((centers[peaks[high]] - centers[peaks[0]])/2)
    range_low =  centers[peaks[0]]- 0.28*fitrange
    range_high = centers[peaks[high]]+ 0.39*fitrange
    
    cutoffs.append((range_low, range_high))
    centers_list.append(centers[peaks[0]])
    peaks = peaks[0:]
    centers_guesses.append([centers[peak] for peak in peaks])
    
    # plt.figure()
    # plt.hist(run.all_peak_data, bins=bins)
    for peak in peaks:
        plt.scatter(centers[peak],count[peak])
    plt.axvline(range_low, c='red')
    plt.axvline(range_high, c='black')
    # plt.xlim([0,1])
    plt.yscale('log')

#%% alter fit ranges
cutoffs[0] = (0.0033,0.022)  
cutoffs[3] = (0.00399,0.0285)   
cutoffs[1] = (0.0031,0.0239) 
# cutoffs[7] = (0.005,0.046)
# cutoffs[6] = (0.0051,0.0425)
# cutoffs[8] = (0.0052,0.0482)
#%% missing data
cutoffs[0] = (0.0033,0.022)  
cutoffs[3] = (0.00399,0.0285)   
cutoffs[1] = (0.0031,0.0239) 
cutoffs[5] = (0.005,0.046)
cutoffs[6] = (0.0052,0.0482)

#%% testing - plot with peak fit
# set conditions, temp
n=5
num=4
T = 170
con = 'LXe'
info_spe = MeasurementInfo()
info_spe.condition = con
info_spe.date = runs[n].date
info_spe.temperature = T
info_spe.bias = runs[n].bias
info_spe.baseline_numbins = 80
info_spe.peaks_numbins = 150
info_spe.data_type = 'h5'
wp = WaveformProcessor(info_spe, run_info_self = runs[n], run_info_solicit = run_spe_baseline, baseline_correct = True, cutoff = cutoffs[n], centers = centers_guesses[n], peak_range = (1,num))
wp.process(do_spe = True, do_alpha = False)
# wp.plot_peak_histograms(log_scale = False)
# wp.plot_spe()
wp.plot_both_histograms(with_fit = True, with_baseline_fit = True)
# wp.plot_baseline_histogram()

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
    num = 4
    if runs[i].bias > 34.5:
        num = 5
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_baseline, baseline_correct = True,  cutoff = cutoffs[i], centers = centers_guesses[i], no_solicit = False, peak_range = (1,num))
    wp_spe.process(do_spe = True, do_alpha = False)
    # wp_spe.plot_peak_histograms(log_scale = False, savefig=False, path = 'D:/Images/LXe May 2023/May_170K_gauss_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_spe(savefig=True, path = 'D:/Images/LXe May 2023/May_170K_spe_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_both_histograms(log_scale = True, savefig=True, path = 'D:/Images/LXe May 2023/May_170K_baseline_'+str(info_spe.bias)+'.png', with_fit = True, with_baseline_fit = True)
    # wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Images/LXe May 2023/May_170K_gauss_'+str(info_spe.bias)+'.svg')
    # wp_spe.plot_spe(savefig=True, path = 'D:/Images/LXe May 2023/May_170K_spe_'+str(info_spe.bias)+'.svg')
    # wp_spe.plot_both_histograms(log_scale = True, savefig=False, path = 'D:/Images/LXe May 2023/May_170K_baseline_'+str(info_spe.bias)+'.svg', with_fit = True, with_baseline_fit = True)
    plt.close()
    campaign_spe.append(wp_spe)
    print(info_spe.bias)
    
#%% plot linear fit to the breakdown voltage
curr_campaign = campaign_spe
filtered_spe = SPE_data(curr_campaign, invC_spe_filter, invC_spe_err_filter, filtered = True)
filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/May_170K_vbd_bias_20240318.csv') #, out_file = 'D:/Xe/AnalysisScripts/May_170K_vbd_bias_updated.csv'

print(filtered_spe.v_bd)
print(filtered_spe.v_bd_err)
# filtered_spe.plot_CA( out_file = 'D:/Xe/AnalysisScripts/May_170K_CA_test.csv')

#%% plot in units of absolute gain vs OV
filtered_spe.plot_spe(in_ov = False, absolute = False)
filtered_spe.plot_spe(in_ov = True, absolute = True, out_file = 'D:/Xe/AnalysisScripts/May_170K_vbd_ov_20240229.csv')
#%% record BD voltage
# v_bd = 27.68 # previous result
# v_bd_err = 0.05
# v_bd = 27.63
# v_bd_err = 0.07
v_bd = 27.52
v_bd_err = 0.05
#%%











#%% CORRELATED AVALANCHE SPE
# polynomial baseline correct off
# 405 nm 2.55V
path = 'C:/Users/Hannah/OneDrive - University of Massachusetts/Dataruns_2023-24/DAQ_folder_google_drive/June 28th & July 11th 13th Data/July 11th (Dark)/july_11th_dark_data/CA data/405nm/'
run_spe_solicited_CA = RunInfo([path+'Run_1689175515.hdf5'], do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
files = ['Run_1689178348','Run_1689177955', 'Run_1689177434', 'Run_1689176952', 'Run_1689175759', 'Run_1689175049'] #, 'Run_1689173976'
runs_CA = []
for file in range(len(files)):
    run_spe_CA = RunInfo([path + files[file] + '.hdf5'], do_filter = True, upper_limit = 1, baseline_correct = True, prominence = 0.0055)
    runs_CA.append(run_spe_CA)
biases = [run.bias for run in runs_CA] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

#%% CORRELATED AVALANCHE SPE
# 310 nm 3.65V
# run_spe_solicited_CA = RunInfo(['D:/Xe/DAQ/Run_1689154239.hdf5'], do_filter = False, is_solicit = True, upper_limit = 1, baseline_correct = True)
# files = ['Run_1689157735', 'Run_1689157266', 'Run_1689156867', 'Run_1689156453', 'Run_1689155459', 'Run_1689155036'] #, 'Run_1689154632'
# runs_CA = []
# for file in range(len(files)):
#     run_spe_CA = RunInfo(['D:/Xe/DAQ/' + files[file] + '.hdf5'], do_filter = False, upper_limit = 1, baseline_correct = True, prominence = 0.005)
#     runs_CA.append(run_spe_CA)
# biases = [run.bias for run in runs_CA] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
#%%
cutoffs = []
centers_list=[]
centers_guesses = []

for run in runs_CA:
    bins = int(round(np.sqrt(len(run.all_peak_data))))
    print(bins)
    count, edges = np.histogram(run.all_peak_data, bins=bins)
    centers = (edges[:-1] + edges[1:])/2
    peaks, props = signal.find_peaks(count, prominence=30, distance=3)
    print(peaks)
    fitrange = ((centers[peaks[3]] - centers[peaks[0]])/2)
    range_low =  centers[peaks[0]]- 0.36*fitrange
    range_high = centers[peaks[3]]+ 0.4*fitrange
    if run.bias < 33:
        range_low =  centers[peaks[0]]- 0.36*fitrange
        range_high = centers[peaks[3]]+ 0.27*fitrange
    
    cutoffs.append((range_low, range_high))
    centers_list.append(centers[peaks[0]])
    peaks = peaks[0:]
    centers_guesses.append([centers[peak] for peak in peaks])
    
    # plt.figure()
    # plt.hist(run.all_peak_data, bins=bins)
    # for peak in peaks:
    #     plt.scatter(centers[peak],count[peak])
    # plt.axvline(range_low, c='red')
    # plt.axvline(range_high, c='black')
    # plt.xlim([0,1])
    # plt.yscale('log')

# %% testing cell - plot with peak fit
# set conditions, temp
n=5
T = 171
con = 'GN'
info_spe = MeasurementInfo()
info_spe.condition = con
info_spe.date = runs_CA[n].date
info_spe.temperature = T
info_spe.bias = runs_CA[n].bias
info_spe.baseline_numbins = 50
info_spe.peaks_numbins = 500
info_spe.data_type = 'h5'
wp = WaveformProcessor(info_spe, run_info_self = runs_CA[n], run_info_solicit = run_spe_solicited_CA, baseline_correct = True, cutoff = cutoffs[n], centers = centers_guesses[n], numpeaks = 4)
wp.process(do_spe = True, do_alpha = False)
# wp.plot_peak_histograms(log_scale = False)
# wp.plot_spe()
wp.plot_both_histograms()
wp.plot_baseline_histogram()

#%% make a list of ProcessWaveforms objects
campaign_CA = []
for i in range(len(runs_CA)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'GN'
    info_spe.date = runs_CA[i].date
    info_spe.temperature = 171
    info_spe.bias = runs_CA[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 400
    info_spe.data_type = 'h5'
    num = 4
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs_CA[i], run_info_solicit = run_spe_solicited_CA, baseline_correct = True,  cutoff = cutoffs[i], centers = centers_guesses[i], no_solicit = False, numpeaks = num)
    wp_spe.process(do_spe = True, do_alpha = False)
    wp_spe.plot_peak_histograms(log_scale = False)
    campaign_CA.append(wp_spe)
#%%
invC_CA_filter = 0.01171 
invC_CA_err_filter = 4.8E-05
spe_CA = SPE_data(campaign_CA, invC_CA_filter, invC_CA_err_filter, filtered = True)
# spe_CA.plot_spe(in_ov = False, absolute = False, out_file = 'D:/Raw Results/July2023_CA_GN.csv') # , out_file = 'D:/Xe/AnalysisScripts/CA_vbd_bias.csv'
#%%
spe_CA.plot_CA(out_file='D:/Raw Results/July2023_CA_GN_probability.csv')
spe_CA.plot_CA_rms()
#%%
n = 0
for i in campaign_CA:
    n+=1
    i.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe May 2023/CA_GN_gauss_'+str(n)+'.png')
    plt.close()
    i.plot_peak_histograms(log_scale = False, savefig=True, path = 'D:/Xe/AnalysisScripts/LXe May 2023/CA_GN_gauss_'+str(n)+'.svg')
    plt.close()
    i.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/LXe May 2023/CA_GN_spe_'+str(n)+'.png')
    plt.close()
    i.plot_spe(savefig=True, path = 'D:/Xe/AnalysisScripts/LXe May 2023/CA_GN_spe_'+str(n)+'.svg')
    plt.close()
    i.plot_both_histograms(savefig=True, path = 'D:/Xe/AnalysisScripts/LXe May 2023/CA_GN_both_'+str(n)+'.png')
    plt.close()
    i.plot_both_histograms(savefig=True, path = 'D:/Xe/AnalysisScripts/LXe May 2023/CA_GN_both_'+str(n)+'.svg')
    plt.close()
#%%
n = 0
for i in campaign_CA:
    n+=1
    i.plot_peak_histograms(log_scale = False)
#%% ALPHA - 1us
#1us, no gain, no filter
invC_alpha_1us =  0.001142
invC_alpha_err_1us = 2.1E-6
files = ['Acquisition_1684439574', 'Acquisition_1684439834','Acquisition_1684440058', 'Acquisition_1684440328','Acquisition_1684440542', 'Acquisition_1684440766', 'Acquisition_1684441008','Acquisition_1684441249','Acquisition_1684441553', 'Acquisition_1684441781', 'Acquisition_1684442031', 'Acquisition_1684442292', 'Acquisition_1684442515', 'Acquisition_1684442713', 'Acquisition_1684442986', 'Acquisition_1684443190', 'Acquisition_1684443416']
proms = [0.007,0.007,0.007,0.01,0.01,0.01,0.02,0.05,0.05,0.06,0.07,0.07,0.07,0.07, 0.2, 0.35, 0.4, 0.5, 0.5]
upperlim = [0.18,0.18,0.18,0.35,0.35,0.35,0.5,0.8,0.8,0.8,0.8,1.75,1.75,1.75,10,10,10,10,10]
runs_alpha_1us = []
for file in range(len(files)):
    run_alpha_1us = RunInfo(['C:/Users/Hannah/Downloads/Run_1684439289.hdf5'], specifyAcquisition = True, acquisition = files[file], do_filter = False, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], is_solicit = False)
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
#%%
campaign_alpha_1us = []
runs_alpha = runs_alpha_1us
for i in range(len(runs_alpha)):
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.01
    if i > 7: # set lower cutoff to be higher for higher ov
        info_alpha.min_alpha_value = 0.15
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[i].date
    info_alpha.temperature = 170
    info_alpha.bias = runs_alpha[i].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = 50
    info_alpha.data_type = 'h5'
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[i], baseline_correct = True, no_solicit = True)
    wp.process(do_spe = False, do_alpha = True)
    j, k = wp.get_alpha()
    campaign_alpha_1us.append(wp)
    print(info_alpha.bias)
    print(runs_alpha[i].do_filter)
#%% testing
for i in runs_alpha_1us:
    i.plot_hists('','')
#%%
campaign_alpha_1us[0].plot_alpha_histogram()
#%%
alpha_data_1us = Alpha_data(campaign_alpha_1us, invC_alpha_1us, invC_alpha_err_1us, spe_CA, v_bd, v_bd_err)
alpha_data_1us.analyze_alpha()
#%%
alpha_data_1us.plot_sigma(color = 'green', out_file = 'D:/Xe/AnalysisScripts/LXe May 2023/May_170K_1us_sigmas_pe.csv', units = "PE")
#%%
alpha_data_1us.plot_alpha(color = 'green', units = "PE", x = "OV",  out_file = 'D:/Xe/AnalysisScripts/LXe May 2023/May2023_Alpha_1us_pe.csv') #,  out_file = 'D:/Xe/AnalysisScripts/LXe May 2023/May2023_Alpha_1us.csv'
alpha_data_1us.plot_num_det_photons(color = 'green', out_file = 'D:/Raw Results/Optical/May2023_1us_num_det_photons.csv')
#%% values
N = 5.49/(19.6E-6)  # based on Wesley's APS slides
PTE = 0.004928 # Sili's logbook post 12551 / copper reflection 100% specular reflective / diffusive teflon
alpha_data_1us.plot_PDE(N*PTE, color='green', out_file = 'D:/Xe/AnalysisScripts/May_170K_PDE_1us_specular_copper_on_diffusive.csv')

#%% values
N = 5.49/(19.6E-6)  # based on Wesley's APS slides
PTE = 0.014515 # Sili's logbook post 12551 / copper reflection 100% specular reflective / specular teflon
alpha_data_1us.plot_PDE(N*PTE, color='green', out_file = 'D:/Xe/AnalysisScripts/May_170K_PDE_1us_specular_copper_on_specular.csv')

#%%
N = 5.49/(19.6E-6)  # based on Wesley's APS slides
PTE = 0.003741 # Sili's logbook post 12551 / no copper reflection / diffusive teflon
alpha_data_1us.plot_PDE(N*PTE, color='green', out_file = 'D:/Xe/AnalysisScripts/May_170K_PDE_1us_copper_off_diffusive.csv')

#%% values
N = 5.49/(19.6E-6)  # based on Wesley's APS slides
PTE = 0.013578 # Sili's logbook post 12551 / no copper reflection / specular teflon
alpha_data_1us.plot_PDE(N*PTE, color='green', out_file = 'D:/Xe/AnalysisScripts/May_170K_PDE_1us_copper_off_specular.csv')

#%% values
PTE = 0.005122 # Sili's logbook post 12551 / copper reflection 100% diffusive / diffusive teflon
alpha_data_1us.plot_PDE(N*PTE, color='green', out_file = 'D:/Xe/AnalysisScripts/May_170K_PDE_1us_diffusive_copper_on_diffusive.csv')

#%% values
PTE = 0.014812 # Sili's logbook post 12551 / copper reflection 100% diffusive / specular teflon
alpha_data_1us.plot_PDE(N*PTE, color='green', out_file = 'D:/Xe/AnalysisScripts/May_170K_PDE_1us_diffusive_copper_on_specular.csv')




#%% ALPHA - 100ns 
#100ns, no gain, no filter
invC_alpha_100ns =  0.004074442176917963
invC_alpha_err_100ns = 1.4320358414009397E-05
files = ['Acquisition_1684429006', 'Acquisition_1684429323','Acquisition_1684429598','Acquisition_1684429901', 'Acquisition_1684430142', 'Acquisition_1684430326','Acquisition_1684430543','Acquisition_1684430787', 'Acquisition_1684431075', 'Acquisition_1684431293', 'Acquisition_1684431526', 'Acquisition_1684431824', 'Acquisition_1684432042', 'Acquisition_1684432244', 'Acquisition_1684432473', 'Acquisition_1684432689'] #,'Acquisition_1684432883','Acquisition_1684433075'
proms = [0.01,0.01,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07, 0.2, 0.35, 0.5, 0.5, 1]
upperlim = [0.15,0.15,0.35, 0.35,0.95,0.95,0.95,0.95,1.95,1.95,1.95,4.95,4.95,4.95,4.95,4.95]
runs_alpha_100ns = []
for file in range(len(files)):
    run_alpha_100ns = RunInfo(['C:/Users/Hannah/Downloads/Run_1684428416.hdf5'], specifyAcquisition = True, acquisition = files[file], do_filter = False, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], is_solicit = False)
    runs_alpha_100ns.append(run_alpha_100ns)
biases = [run.bias for run in runs_alpha_100ns] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
#%%
campaign_alpha_100ns = []
runs_alpha = runs_alpha_100ns
for i in range(len(runs_alpha)):
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.01
    # if i > 7: # set lower cutoff to be higher for higher ov
        # info_alpha.min_alpha_value = 0.15
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[i].date
    info_alpha.temperature = 170
    info_alpha.bias = runs_alpha[i].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = 50
    info_alpha.data_type = 'h5'
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[i], baseline_correct = True, no_solicit = True)
    wp.process(do_spe = False, do_alpha = True)
    j, k = wp.get_alpha()
    campaign_alpha_100ns.append(wp)
    print(info_alpha.bias)
#%%
for i in runs_alpha_100ns:
    i.plot_hists('','')
#%%
invC_alpha = invC_alpha_100ns
invC_alpha_err = invC_alpha_err_100ns
alpha_data_100ns = Alpha_data(campaign_alpha_100ns, invC_alpha, invC_alpha_err, spe, v_bd, v_bd_err)
alpha_data_100ns.analyze_alpha()
#%%
n = 0
for i in campaign_alpha_100ns:
    n+=1
    i.plot_alpha_histogram(peakcolor = 'purple')
    plt.savefig('D:/Xe/AnalysisScripts/LXe May 2023/100ns_'+str(n)+'.png')
    plt.close()
#%%
alpha_data_100ns.plot_alpha(color = 'purple', out_file = 'D:/Xe/AnalysisScripts/LXe May 2023/2023May_100ns_alpha.csv')
alpha_data_100ns.plot_num_det_photons()
#%% values
N = 5.49/(19.6E-6)  # based on Wesley's APS slides
PTE = 0.003741 # Sili's logbook post 12551
alpha_data_100ns.plot_PDE(N*PTE, color='purple', out_file = 'D:/Xe/AnalysisScripts/May_170K_PDE_100ns.csv')
