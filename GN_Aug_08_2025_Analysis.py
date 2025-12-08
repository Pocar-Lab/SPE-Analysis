#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intermediate Temperature GN Campaign
SPE analysis for data collected on Aug 8 2025 at 235K with LED at 2.6V and 2.75V

Created on Fri Aug 8 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
from AnalyzePDE import SPE_data
import h5py
import os
from scipy.optimize import curve_fit
import numpy as np
import pickle
from multiprocessing import Pool
plt.style.use('misc/nexo.mplstyle')

### Aug 8 2025 at 235K with LED at 2.6V and 2.75V

### 2.6V LED voltage for SPE and CA

condition = 'GN'
temperature = 218
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
# change path name to folder with data files
path = '../data/20250808_SPE_GN_218K/2.6_10/'
_, _, run_spe_pre_bd = get_files('../data/20250808_SPE_GN_218K/')
files, biases, _ = get_files(path)

def proc(file):
    return ProcessWaveforms([path+file], do_filter=True,
                            baseline_correct=True, poly_correct=True,
                            upper_limit=-1, prominence=.003)
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
savefig=False
pe_center_guesses = [4.9, 5.1, 5.3, 5.8, 6.0, 6.6, 7.1, 7.4, 7.8]
num_pes           = [3,   5,   6,   7,   8,   8,   8,   10,  10]
lower_cutoff      = [4.2, 4.2, 4.3, 4.6, 4.5, 4.5, 4.9, 5.0, 5.0]
outpath = '~/0vbb/20250808/2.6V/20250812_'
campaign_spe = []
for i in range(len(files)):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=(lower_cutoff[i]/1000, .09),
                         first_pe=pe_center_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)



### Aug 8 2.75V LED voltage for SPE and comparision with IV

condition = 'GN'
temperature = 218

#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
# change path name to folder with data files
path = 'data/20250808_SPE_GN_218K/'

# files = os.listdir(path+"2.75_10")
# # Sort files by bias voltage
# biases = []
# for f in files:
#     with h5py.File(path+"2.75_10/"+f, 'r') as hdf:
#         biases.append(hdf['RunData'][next(iter(hdf['RunData'].keys()))].attrs['Bias(V)'])
# # files = files[:-2]
# files = files[:2] + files[-2:]
# _, files = zip(*sorted(zip(biases, files)))
files, biases, _ = get_files(path+"2.75_10")

file_pre_bd = 'Run_1754658234.hdf5'
run_spe_pre_bd = ProcessWaveforms([path+file_pre_bd], do_filter=True,
                                  is_pre_bd=True, upper_limit=0.063, baseline_correct=True)

# loop over all files
runs = []
prom = 0.0050
upperlim = -1
for file in range(len(files)):
    run = ProcessWaveforms([path+"2.75_10/"+files[file]], do_filter=True,
                           baseline_correct=True, poly_correct=True,
                           upper_limit=upperlim, prominence=prom)
    runs.append(run)

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    # Plot preliminary histogram
    m.plot_histogram()
    measurments.append(m)


# Fit Gauss to peaks and find the best value for peak location
pe_center_guesses = [4.9, 5.1, 5.4, 5.8, 6.0, 6.6, 7.1, 7.4, 7.8]
num_pes = [4, 6, 7, 8, 8, 8, 8, 8, 8]
centers_guesses = []
for pe_guess, num_pe in zip(pe_center_guesses, num_pes):
    centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
outpath = '/home/ed/pictures/0vbb/20250808/2.75V/20250813_'
savefig = True
campaign_spe = []
for i in range(len(measurments)):
    if measurments[i].bias <= 33.5:
        cutoff = (.003,.03)
    elif measurments[i].bias == 34.:
        cutoff = (.003,.05)
    elif measurments[i].bias >= 36.:
        cutoff = (.0035,.10)
    elif measurments[i].bias >= 35.5:
        cutoff = (.0035,.08)
    else:
        cutoff = (.003,.07)
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=cutoff, centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()


### 2.75V LED voltage comparing the same bias voltages at the start and end of the data collection

condition = 'GN'
temperature = 218

#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
# change path name to folder with data files
path = 'data/20250808_SPE_GN_218K/'
files = os.listdir(path+"2.75_10")

# Sort files by bias voltage
biases = []
for f in files:
    with h5py.File(path+"2.75_10/"+f, 'r') as hdf:
        biases.append(hdf['RunData'][next(iter(hdf['RunData'].keys()))].attrs['Bias(V)'])
files = files[:2] + files[-2:]
_, files = zip(*sorted(zip(biases, files)))

file_pre_bd = 'Run_1754658234.hdf5'
run_spe_pre_bd = ProcessWaveforms([path+file_pre_bd], do_filter=True,
                                  is_pre_bd=True, upper_limit=0.063, baseline_correct=True)

# loop over all files
runs = []
prom = 0.0050
upperlim = -1
for file in range(len(files)):
    run = ProcessWaveforms([path+"2.75_10/"+files[file]], do_filter=True,
                           baseline_correct=True, poly_correct=True,
                           upper_limit=upperlim, prominence=prom)
    runs.append(run)

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    # Plot preliminary histogram
    m.plot_histogram()
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
pe_center_guesses = [4.9, 4.9, 5.4, 5.4]
num_pes = [4, 4, 6, 6]
centers_guesses = []
for pe_guess, num_pe in zip(pe_center_guesses, num_pes):
    centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
outpath = '/home/ed/pictures/0vbb/20250808/2.75V/20250813_'
savefig = False
campaign_spe = []
for i in range(2,4):
    if measurments[i].bias == 33.:
        cutoff = (.003,.03)
    elif measurments[i].bias == 34.:
        cutoff = (.003,.05)
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=cutoff, centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()



