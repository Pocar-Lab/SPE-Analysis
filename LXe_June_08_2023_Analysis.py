# -*- coding: utf-8 -*-
"""
SPE Created on Mon Oct 16 18:41:24 2023

@author: Hannah
@author: Ed van Bruggen (evanbruggen@umass.edu)
"""


%load_ext autoreload
%autoreload 2
%autoindent

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
import pickle
import dill
import csv

plt.style.use('misc/nexo.mplstyle')

#%%
#1us, 10x gain, filter off
invC =0.0132
invC_err =0.000089
invC_spe_filter2 = 0.01171
invC_spe_err_filter2 = 0.000048
# USE DEG = 3 IN POLY_CORRECT
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
path ='data-june-08/'
run_spe_solicited = RunInfo([path+'Run_1686255720.hdf5'],  specifyAcquisition = False, do_filter = True, is_solicit = True, upper_limit = 0.2, baseline_correct = True, poly_correct = True)

files = ['Run_1686246831','Run_1686245382','Run_1686247699','Run_1686241351','Run_1686249902','Run_1686244907']
upperlim = [0.32, 0.33, 0.33, 0.33, 0.75, 0.75]
proms = [0.004, 0.004, 0.0044, 0.0044, 0.0044, 0.0044, 0.0044, 0.0044, 0.005]
runs = []
for file in range(len(files)):
    run_spe = RunInfo([path+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], is_led = True)
    runs.append(run_spe)
biases = [run.bias for run in runs]

#%%
# RunInfo(['D:/Xe/DAQ/'+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = 0.004, poly_correct = True, is_led = True, plot_waveforms = True, num_waveforms = 10,  width = [10,200])
#%%
file = 3
runs[3] = RunInfo([path+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = 0.0044, poly_correct = True, is_led = True)

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
    subtracted_counts[subtracted_counts < -80] = 0
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
    
# #%%
# for run in runs:
#     get_subtract_hist_mean(run.bias,run.all_led_peak_data, run.all_dark_peak_data, plot = True)
    
#%% variable peaks
range_lows2 = [0.004,0.0036,0.0033,0.0046, 0.0047,0.0048] 
centers2 = [0.0051,0.006,0.0063, 0.0076, 0.00819, 0.0087] #
range_highs2 = [0.02375,0.0264,0.029, 0.04, 0.051, 0.055] #only use from 33V and onwards
numpeaks = [4,4,4,5,6,6]

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
wp = WaveformProcessor(info_spe, run_info_self = runs[n], run_info_solicit = run_spe_solicited, baseline_correct = True, cutoff = (range_lows2[n],range_highs2[n]), centers = [centers2[n]*a for a in range(1,numpeaks[n]+1)], peak_range = (1,numpeaks[n]), subtraction_method=True)
wp.process(do_spe = True, do_alpha = False)
wp.plot_both_histograms(with_fit=True,with_baseline_fit=True)
wp.plot_peak_histograms()
# wp.get_subtract_hist_mean(runs2[8].all_led_peak_data,runs2[8].all_dark_peak_data,plot = True)
wp.plot_spe()
#%% 
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
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, cutoff = (range_lows2[i],range_highs2[i]), centers = [centers2[i]*a for a in range(1,numpeaks[i]+1)], peak_range = (1,numpeaks[i]), subtraction_method=True)
    wp_spe.process(do_spe = True, do_alpha = False)
    wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/June 8/06082023_gauss_'+str(info_spe.bias)+'.png')
    wp_spe.plot_spe(savefig=True, path = 'C:/Users/Hannah/Downloads/June 8/06082023_spe_'+str(info_spe.bias)+'.png')
    wp_spe.plot_both_histograms(with_fit = True, with_baseline_fit = True, log_scale = True, savefig=False, path = 'C:/Users/Hannah/Downloads/June 8/06082023_baseline_'+str(info_spe.bias)+'.png')
    wp_spe.plot_peak_histograms(log_scale = False, savefig=True, path = 'C:/Users/Hannah/Downloads/June 8/06082023_gauss_'+str(info_spe.bias)+'.svg')
    wp_spe.plot_spe(savefig=True, path = 'C:/Users/Hannah/Downloads/June 8/06082023_spe_'+str(info_spe.bias)+'.svg')
    wp_spe.plot_both_histograms(with_fit = True, with_baseline_fit = True, log_scale = True, savefig=False, path = 'C:/Users/Hannah/Downloads/June 8/06082023_baseline_'+str(info_spe.bias)+'.svg')
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
spe.plot_CA(out_file = 'C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/LXe_June_2023_CA_ov.csv')
spe.plot_spe(in_ov = False, absolute = False, out_file = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/LXe_June_2023_vbd_bias.csv') 
# spe.plot_CA(out_file = None)
file_path = 'data-june-08/' 
run_spe_solicited = RunInfo([file_path+'Run_1686255720.hdf5'], specifyAcquisition = False, do_filter = True, is_solicit = True, upper_limit = .5, baseline_correct = True)

files = ['Run_1686250924.hdf5', 'Run_1686252148.hdf5', 'Run_1686253721.hdf5', 'Run_1686255098.hdf5']
proms = [0.04,0.04,0.04,0.04]
upperlim = [4, 4, 4, 4]
# files = ['Run_1686251282.hdf5', 'Run_1686252429.hdf5', 'Run_1686254519.hdf5', 'Run_1686255420.hdf5']
# proms = [0.035,0.04,0.04,0.05]
# upperlim = [4, 4, 4, 4]
# files = ['Run_1686251590.hdf5', 'Run_1686253346.hdf5', 'Run_1686254720.hdf5' ]
# proms = [0.035,0.04,0.04,0.05]
# upperlim = [4, 4, 4, 4]
runs = []
for file in range(len(files)):
    run_spe = RunInfo([file_path+files[file]], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file])
    runs.append(run_spe)
biases = [run.bias for run in runs]

range_lows = [0.06, 0.08, 0.09, 0.1]
centers = [0.09, 0.11, 0.12, 0.14]
range_highs = [0.403, 0.5, 0.56, 0.67]
campaign_spe = []
for i in range(len(runs)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'GN'
    info_spe.date = runs[i].date
    info_spe.temperature = '167'
    info_spe.bias = biases[i]
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 220
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i])
    wp_spe.process(do_spe = True, do_alpha = False)
    campaign_spe.append(wp_spe)

invC_spe_filter = 0.19261918346201742
invC_spe_err_filter = 0.0021831140214106596
spe = SPE_data(campaign_spe, invC_spe_filter, invC_spe_err_filter, filtered = True)

##%% CORRELATED AVALANCHE SPE
#file_path = 'self/' # folder with H5 data files
#run_spe_solicited = RunInfo([file_path+'Run_1648191265.hdf5'], do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
#files = ['Run_1648176286','Run_1648179496', 'Run_1648181807', 'Run_1648184235', 'Run_1648186910'] #, 'Run_1648186910', 'Run_1648171846'
#proms = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
#upperlim = [5, 5, 5, 5, 5, 5]
#runs = []
#for file in range(len(files)):
#    run_spe = RunInfo([file_path + files[file] + '.hdf5'], do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file])
#    runs.append(run_spe)
#biases = [run.bias for run in runs] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
##%%
#a_guess = 0.01689
#b_guess = -0.4739
#range_lows = [0.05, 0.059, 0.071, 0.09, 0.09, 0.04]
#centers = [biases[i]*a_guess+b_guess for i in range(len(runs))]
#range_highs = [centers[i]*4 + range_lows[i] for i in range(len(runs))]
##%%
#campaign_spe = []
#for i in range(len(runs)):
#    info_spe = MeasurementInfo()
#    info_spe.condition = 'Vacuum'
#    info_spe.date = 'March 2022'
#    info_spe.temperature = 190
#    info_spe.bias = runs[i].bias
#    info_spe.baseline_numbins = 50
#    info_spe.peaks_numbins = 200
#    info_spe.data_type = 'h5'
#    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i])
#    wp_spe.process(do_spe = True, do_alpha = False)
#    campaign_spe.append(wp_spe)
##%%
#invC_spe_filter = 0.19261918346201742
#invC_spe_err_filter = 0.0021831140214106596
#spe = SPE_data(campaign_spe, invC_spe_filter, invC_spe_err_filter, filtered = True)






#%% ALPHA - 1us
#1us, no gain, no filter
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
file_path = 'data-june-08/' # folder with H5 data files
files = ['Run_1686250924.hdf5', 'Run_1686252148.hdf5', 'Run_1686253721.hdf5', 'Run_1686255098.hdf5']
proms = [0.04,0.04,0.04,0.04]
upperlim = [4, 1, 4, 4]
files += ['Run_1686251282.hdf5', 'Run_1686252429.hdf5', 'Run_1686254519.hdf5', 'Run_1686255420.hdf5']
proms += [0.035,0.04,0.04,0.05]
upperlim += [1, 1, .5, 1]
files += ['Run_1686251590.hdf5', 'Run_1686253346.hdf5', 'Run_1686254720.hdf5' ]
proms += [0.035,0.2,0.04]
upperlim += [.8, 4, 4, 4]
runs_alpha_1us = []
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file]], do_filter=False, upper_limit=upperlim[file], baseline_correct=True, prominence=proms[file], plot_waveforms=False)
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
# runs_alpha_1us[0].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
# TODO 29, 31
# TODO fix 33V? chi^2 is very high
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [35,35,34, 32,29,30, 33,32,37,35, 33,43,30]
for i in range(1):
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value =  0.029
    # info_alpha.min_alpha_value = 0.3 if i > 8 else 0.029
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[i].date
    info_alpha.temperature = 167
    info_alpha.bias = runs_alpha[i].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = bins[i]
    # print(f"{i=}")
    # print(f"{info_alpha.bias=}")
    # wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[i], baseline_correct = True, no_solicit = True, range_high = 10)
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[i], baseline_correct = True, no_solicit = True, cutoff=(0,10))
    wp.process(do_spe = False, do_alpha = True)
    j, k = wp.get_alpha()
    wp.plot_alpha_histogram(peakcolor = 'blue')
    campaign_alpha.append(wp)

#%%
# with open('LED-SPE/campaign_alpha.pickle', 'wb') as f:
#     dill.dump(campaign_alpha, f)

# p = dill.Unpickler(open("/media/ed/My Passport/ed/CA-july-12.pickle","rb"))
# p.fast = True
# spe = p.load()

v_bd = 27.69
v_bd_err = 0.06
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

alpha_data.analyze_alpha()

#%%
alpha_data.plot_alpha()

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE = .005530 # diffusive reflector
# PTE = .010736 # specular reflector
alpha_data.plot_PDE(N*PTE)
