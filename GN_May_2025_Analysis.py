#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPE: Created on Fri May 9 16:00:00 2025

Analysis for May 9th 2025 data

@author: Thomas S.
"""

import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from MeasurementInfo import MeasurementInfo
from src.ProcessHistograms import ProcessHist
from RunInfo import RunInfo
from AnalyzePDE import SPE_data
from ProcessWaveforms_MultiGaussian import WaveformProcessor

# plt.style.use('nexo_new.mplstyle')
plt.style.use('misc/nexo.mplstyle')


#%%
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098


#%%
# change path name to folder with data files
path ='data/20250509_SPE_GN/'

files = os.listdir(path)
files = [f for f in files if os.path.isfile(path+'/'+f)]

#print names of all files
print(files)

#%% read minimal waveforms as a shortcut to print all the metadata num_waveforms = 1

# all files without the baseline noise file at a bias voltage of 20V
# Excluding  32.0V
# We did two runs at a bias voltage of 36.0V because I initially saw no peaks in the finger plots. Upon further investigation, they were easy to see when adjusting the bin size.
# Therefore, I'm excluding here the second run with a bias voltage of 36.0V
files = ['Run_1746799234.hdf5', 'Run_1746797467.hdf5', 'Run_1746802343.hdf5',
         'Run_1746796959.hdf5', 'Run_1746800530.hdf5', 'Run_1746798018.hdf5',
         'Run_1746795858.hdf5', 'Run_1746810834.hdf5']


# loop over all files to find bias, scoperange, offset and upperlim
runs = []
# for file in range(len(files)):
for file in range(5,6):
    run_spe = RunInfo([path+files[file]], specifyAcquisition = False, do_filter = True,
                      upper_limit = 5, poly_correct=True, baseline_correct = True, prominence =
                      0.008, is_led = True, num_waveforms = 1)
    runs.append(run_spe)
biases = [run.bias for run in runs]
print(biases)
scoperange = [run.yrange for run in runs]
print(scoperange)
offset = [run.offset for run in runs]
print(offset)

# upper_limit = scoperange - offset - 0.001 is a good choice, learned from Hannahs older analysis
upperlim = np.array(scoperange)-np.array(offset)-0.001


#%%
# noise at a bias of 20V
# upperlimit from general equation
run_spe_solicited = RunInfo([path+'Run_1746805292.hdf5'],  specifyAcquisition = False, do_filter =
                            True, is_solicit = True, upper_limit = 0.063, baseline_correct = True)

#%%

# loop over all files
runs = []
# prominences=[.0055,  .0056, .0056,  .0056,  .0056,  .0053,  .0054, .0054]
# prominences=[.0054,  .0055, .0055,  .0055,  .0055,  .0055,  .0054, .0054]
# prominences=[.0053,  .0054, .0054,  .0054,  .0054,  .0054,  .0054, .0054]
# prominences = [.00545, .0053, .0053,  .00545, .00545, .00535, .0054, .0054]
# prominences = [.00545, .0052, .0052,  .00544, .00544, .00538, .0054, .0054]
prominences = [.00545, .00525, .00525, .00543, .00543, .0054,  .0054, .0054]
# for file in range(0, len(files)):
for file in range(5,6):
    run_spe = RunInfo([path+files[file]], specifyAcquisition=False, do_filter=True,
                      upper_limit=upperlim[0], baseline_correct=True,
                      prominence=prominences[file],
                      poly_correct=True, is_led=True)
    runs.append(run_spe)

#%% get the approximate locations of the SPE peaks

cutoffs = []
centers_list=[]
centers_guesses = []
numpeaks = []
for run in runs:
    bins = int(round(np.sqrt(len(run.all_peak_data))))
    prominence = 15
    if run.bias < 34.5 and run.bias >32.0:
        high = 2
        distance= 5
        numpeaks.append(high)
    if run.bias == 34.5:
        high = 2
        distance =3
        numpeaks.append(high)
    if run.bias > 34.5 and run.bias < 36.0:
        high = 3
        distance= 5
        numpeaks.append(high)
    if run.bias == 36.0:
        bins = bins * 3
        high = 3
        distance= 4
        numpeaks.append(high)
    count, edges = np.histogram(run.all_peak_data, bins=bins)
    centers = (edges[:-1] + edges[1:])/2
    peaks, props = signal.find_peaks(count, prominence=prominence, distance=distance)
    print(f"{run.bias}: {peaks=}")
    # plt.figure()
    # plt.hist(run.all_peak_data, bins=bins)
    # plt.xlim(0,0.3)
    # plt.show()
    fitrange = ((centers[peaks[high]] - centers[peaks[0]])/2)
    if run.bias < 33.0:
        range_low  = centers[peaks[0]]- 0.25*fitrange
        # range_high = centers[peaks[high]]+ 0.45*fitrange
        range_high = centers[peaks[high]]+ 0.5*fitrange
    elif run.bias == 33.0:
        range_low  = centers[peaks[0]]- 0.22*fitrange
        range_high = centers[peaks[high]]+ 0.55*fitrange
    elif run.bias < 35.0:
        range_low  = centers[peaks[0]]- 0.25*fitrange
        range_high = centers[peaks[high]]+ 0.45*fitrange
    elif run.bias < 35.5:
        range_low  = centers[peaks[0]]- 0.25*fitrange
        range_high = centers[peaks[high]]+ 0.5*fitrange
    else:
        range_low  = centers[peaks[0]]- 0.25*fitrange
        range_high = centers[peaks[high]]+ 0.35*fitrange
    range_high = .06 if run.bias != 34. else .055
    cutoffs.append((range_low, range_high))
    # cutoffs.append((range_low, run.yrange))
    centers_list.append(centers[peaks[0]])
    peaks = peaks[0:]
    centers_guesses.append([centers[peak] for peak in peaks])
    # plot histogram with found peaks
    #plt.hist(run.all_peak_data, bins=bins)
    #for peak in peaks:
    #    plt.scatter(centers[peak],count[peak])
    #plt.axvline(range_low, c='red')
    #plt.axvline(range_high, c='black')
    #plt.xlim(0,0.3)
    ##plt.yscale('log')
    #plt.show()

peak_max = [4, 4, 5, 5, 6, 6, 6, 6]
# Fit Gauss to peaks and find the best value for peak location
outpath = '/home/ed/pictures/0vbb/20250509/20250627_4pe_'
campaign_spe = []
for i in range(len(runs)): #
    info_spe = MeasurementInfo()
    info_spe.condition = 'GN'
    info_spe.date = runs[i].date
    info_spe.temperature = 171.0
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self=runs[i],
                               run_info_solicit=run_spe_solicited, baseline_correct=True,
                               cutoff=cutoffs[i], centers=centers_guesses[i],
                               background_linear=False,
                               peak_range=(1,6))
                               # peak_range=(1,3))
    wp_spe.process(do_spe=True, do_alpha=False)
    wp_spe.plot_peak_histograms(log_scale=False, savefig=False,
                                path=outpath+'gauss_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_spe_with_origin(savefig=True,
    #                             path=outpath+'spe_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)
    print(info_spe.bias)

# Fit Gauss to peaks and find the best value for peak location
outpath = '/home/ed/pictures/0vbb/20250509/20250627_4pe_'
campaign_spe = []
for i in range(1): #
    info_spe = MeasurementInfo()
    info_spe.condition = 'GN'
    info_spe.date = runs[i].date
    info_spe.temperature = 171.0
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = ProcessHist(info_spe, run_info_self=runs[i],
                         run_info_pre_bd=run_spe_solicited, baseline_correct=True,
                         cutoff=cutoffs[i], centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,6))
                         # peak_range=(1,3))
    wp_spe.process_spe()
    wp_spe.plot_peak_histograms(log_scale=False, savefig=False,
                                path=outpath+'gauss_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_spe_with_origin(savefig=False,
    #                             path=outpath+'spe_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)
    print(info_spe.bias)



for i in range(len(campaign_spe)):
    campaign_spe[i].plot_both_histograms()

#%% plot linear fit to the breakdown voltage
curr_campaign = campaign_spe
spe = SPE_data(curr_campaign, invC_filter, invC_err_filter, filtered = True)
spe.plot_spe(in_ov = False, absolute = True)

spe.plot_spe(in_ov = False, absolute = False)


spe.plot_CA()






