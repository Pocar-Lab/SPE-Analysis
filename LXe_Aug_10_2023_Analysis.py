# -*- coding: utf-8 -*-
"""
Alpha: Created on Aug 10 2023
@author: Ed van Bruggen (evanbruggen@umass.edu)

SPE: Created on Mon Oct 16 18:41:24 2023
@author: Hannah
"""

%load_ext autoreload
%autoreload 2
%autoindent

import sys
from MeasurementInfo import MeasurementInfo
import numpy as np
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

#%% SPE
#1us, 10x gain, filter off
invC =0.0132
invC_err =0.000089
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
run_spe_solicited = RunInfo(['D:/Xe/DAQ/Run_1691696340.hdf5'],  specifyAcquisition = False, do_filter = False, is_solicit = True, upper_limit = 0.2, baseline_correct = True)
files = ['Run_1691691406','Run_1691690935','Run_1691688275','Run_1691694839','Run_1691693056','Run_1691694534','Run_1691695317','Run_1691689385','Run_1691689812','Run_1691693423', 'Run_1691694026'] #31.5, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36, 36.5
upperlim = [0.062, 0.062, 0.1, 0.1, 0.1, 0.1, 0.1, 0.18, 0.3, 0.3, 0.45]
proms = [0.0055, 0.0055, 0.0055, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006]
runs = []
for file in range(len(files)):
    run_spe = RunInfo(['D:/Xe/DAQ/'+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = False, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], poly_correct = True)
    runs.append(run_spe)
biases = [run.bias for run in runs]
#%%
#%%
cutoffs = []
centers_list=[]
centers_guesses = []

for run in runs:
    bins = int(round(np.sqrt(len(run.all_peak_data))))
    print(bins)
    count, edges = np.histogram(run.all_peak_data, bins=bins)
    centers = (edges[:-1] + edges[1:])/2
    if run.bias == 36.5:
        d = 2
    else:
        d = 3
    peaks, props = signal.find_peaks(count, prominence=25, distance=d)
    fitrange = ((centers[peaks[3]] - centers[peaks[0]])/2)
    if run.bias < 32.5:
        range_low =  centers[peaks[0]]- 0.32*fitrange
        range_high = centers[peaks[3]]+ 0.42*fitrange
    if run.bias == 32.5:
        range_low =  centers[peaks[0]]- 0.33*fitrange
        range_high = centers[peaks[3]]+ 0.45*fitrange
    if run.bias > 32.5:
        range_low =  centers[peaks[0]]- 0.24*fitrange
        range_high = centers[peaks[3]]+ 0.34*fitrange
    if run.bias >= 34:
        range_low =  centers[peaks[0]]- 0.22*fitrange
        range_high = centers[peaks[3]]+ 0.37*fitrange 
    if run.bias == 33.5:
        range_low =  centers[peaks[0]]- 0.27*fitrange
        range_high = centers[peaks[3]]+ 0.37*fitrange
    if run.bias > 34.5:
        range_low =  centers[peaks[0]]- 0.28*fitrange
        range_high = centers[peaks[3]]+ 0.31*fitrange
    if run.bias == 36.5:
        range_low =  centers[peaks[0]]- 0.34*fitrange
        range_high = centers[peaks[3]]+ 0.31*fitrange
    cutoffs.append((range_low, range_high))
    centers_list.append(centers[peaks[0]])
    peaks = peaks[0:]
    centers_guesses.append([centers[peak] for peak in peaks])
    
    # plt.figure()
    # plt.hist(run.all_peak_data, bins=bins)
    # for peak in peaks:
        # plt.scatter(centers[peak],count[peak])
    # plt.axvline(range_low, c='red')
    # plt.axvline(range_high, c='black')
    # plt.xlim([0,1])
    # plt.yscale('log')
#%% testing cell - plot with peak fit
# set conditions, temp
n=1
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
wp = WaveformProcessor(info_spe, run_info_self = runs[n], run_info_solicit = run_spe_solicited, baseline_correct = True, cutoff = cutoffs[n], centers = centers_guesses[n], numpeaks = 4)
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
    # if i == 0:
    #     num = 3
    # else:
    num = 4
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True,  cutoff = cutoffs[i], centers = centers_guesses[i], no_solicit = False, numpeaks = num)
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

#%% plot linear fit to the breakdown voltage
curr_campaign = campaign_spe
spe = SPE_data(curr_campaign, invC, invC_err, filtered = False)
spe.plot_spe(in_ov = False, absolute = False, out_file = 'D:/Xe/AnalysisScripts/LXe August 1 2023/vbd_bias_no_silicon.csv') #




#%% ALPHA - 1us
#1us, no gain, no filter
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
file_path = 'aug-alpha/' # folder with H5 data files
files = ['Run_1691679909.hdf5', 'Run_1691680314.hdf5',  'Run_1691680611.hdf5' ]
proms = [ .025, .015, .1 ]
upperlim = [ 1, .4, 2]
files += [ 'Run_1691682631.hdf5',  'Run_1691682887.hdf5']
proms += [ .01, .015 ]
upperlim += [ .3, .5 ]
files += ['Run_1691683735.hdf5', 'Run_1691684246.hdf5',
         'Run_1691683999.hdf5', 'Run_1691684510.hdf5', 'Run_1691684917.hdf5']
proms += [ .04, .015, .015, .015, .1]
upperlim += [ .6, .2, .5, .2, 1.5]
files += ['Run_1691681764.hdf5','Run_1691683166.hdf5'] # 28.5
proms += [ .005, .005]
upperlim += [ .025, .06]
files += ['Run_1691685248.hdf5', 'Run_1691685841.hdf5', 'Run_1691686415.hdf5', 'Run_1691686760.hdf5']
proms += [ .02, .005, .015, .05 ]
upperlim += [ .1, .05, .2, 1]
runs_alpha_1us = []
i = 0
# files = files[i:i+2]
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file]], do_filter=False,
                            upper_limit=upperlim[file+i], baseline_correct=True,
                            prominence=proms[file+i], plot_waveforms=False)
    # run_alpha_1us.plot_hists('','')
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

runs_alpha_1us[0].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [40, 40, 53, 40, 40]
bins += [40, 55, 40, 40, 40]
bins += [50, 40]
bins += [36, 39, 40, 42]
for n in range(len(runs_alpha)):
# if n := 15:
    # if n < 2:
    #     continue
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.001
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[n].date
    info_alpha.temperature = 167
    info_alpha.bias = runs_alpha[n].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = bins[n]
    # print(f'{n=}')
    # print(f"{info_alpha.bias=}")
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[n], baseline_correct = True,
                           no_solicit = True, cutoff=(0,10))
    wp.process(do_spe = False, do_alpha = True)
    # j, k = wp.get_alpha()
    wp.plot_alpha_histogram(peakcolor = 'blue', with_fit=True)
    campaign_alpha.append(wp)
    # break

#%%
storage_path = '/media/ed/My Passport/ed/'

with open(storage_path+'aug-10-campaign-alpha.pickle', 'wb') as f:
    dill.dump(campaign_alpha, f)

# with open('LED-SPE/SPE-June-28.pickle', 'rb') as f:
#     spe = dill.load(f)

p = dill.Unpickler(open("/media/ed/My Passport/ed/aug-10-campaign-alpha.pickle","rb"))
p.fast = True
# p.dump(campaign_alpha)
campaign_alpha = p.load()

p = dill.Unpickler(open(storage_path+"CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()



v_bd = 27.28
v_bd_err = 0.05
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

p = dill.Pickler(open("July-2023-alpha_data.pickle","wb"))
p.fast = True
p.dump(alpha_data)

alpha_data.analyze_alpha(out_file='aug-10-2023-lxe.csv')

#%%
alpha_data.plot_alpha()#out_file='aug-10-2023_alpha_amp.csv'

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
# PTE = 0.0042
PTE_dif = 0.001611
PTE_spec = 0.001352
PTE_none = 0.001360
alpha_data.plot_PDE(N*PTE_dif)
