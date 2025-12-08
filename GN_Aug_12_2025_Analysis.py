#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPE analysis for data collected on Aug 12 2025 at 252K

Created on Fri Aug 12 2025

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


### Aug 12th SPE

condition = 'GN'
temperature = 252
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '../data/20250812_SPE_GN_252K/'
files, biases, run_spe_pre_bd = get_files(path)

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
# pe_center_guesses = [4.9, 5.1, 5.3, 5.6, 6.1, 6.3, 6.7, 7.2, 7.5]
savefig = False
first_pe     = [4.9, 4.6, 5.1, 5.5, 5.9, 6.3, 6.7, 7.1, 7.4]
num_pes      = [4,   5,   5,   7,   7,   8,   9,   10,  10]
lower_cutoff = [4.0, 3.2, 3.1, 3.0, 3.2, 3.3, 3.4, 3.5, 3.8]
outpath = '/home/evanbruggen_umass_edu/0vbb/20250812/20250923_'
campaign_spe = []
for i in range(len(measurments)):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=(lower_cutoff[i]/1000, .1),
                         first_pe=first_pe[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    # wp_spe.plot_peak_histogram(log_scale=True,
    #                            savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    # wp_spe.plot_spe(fit_origin=False,
    #                 savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    # wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()

spe.plot_CA(other_ca='../results/CA/July13_171K_LXe_CA.csv', out_file='../results/20250812_CA_252K.csv')

### Aug 12th SPE, low prom test

condition = 'GN'
temperature = 252
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
# change path name to folder with data files
path = '../data/20250812_SPE_GN_252K/'
files, biases, run_spe_pre_bd = get_files(path)

def proc(file):
    return ProcessWaveforms([path+file], do_filter=True,
                            baseline_correct=True, poly_correct=True,
                            upper_limit=-1, prominence=.003) #.002
with Pool(len(files)) as p:
    runs = p.map(proc, iter(files))


# files = os.listdir(path)
# # Sort files by bias voltage
# biases = []
# for f in files:
#     with h5py.File(path+f, 'r') as hdf:
#         biases.append(hdf['RunData'][next(iter(hdf['RunData'].keys()))].attrs['Bias(V)'])
# biases, files = zip(*sorted(zip(biases, files)))

# # Extract pre-breakdown baseline if it was collected
# if biases[0] < 25:
#     file_pre_bd = files[0]
#     run_spe_pre_bd = ProcessWaveforms([path+file_pre_bd], do_filter=True,
#                                       is_pre_bd=True, upper_limit=0.063, baseline_correct=True)
#     files = files[1:]
#     biases = biases[1:]
# else:
#     file_pre_bd = None

# # loop over all files
# runs = []
# prom = .002
# upperlim = -1
# for file in range(4,5):
#     run = ProcessWaveforms([path+files[file]], do_filter=True,
#                            baseline_correct=True, poly_correct=True,
#                            upper_limit=upperlim, prominence=prom)
#     runs.append(run)


measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# with open('20250812_measurments.pkl', 'wb') as file:
#     pickle.dump(measurments, file)

# Fit Gauss to peaks and find the best value for peak location
savefig = True
# first_pe     = [4.9, 5.1, 5.3, 5.6, 6.1, 6.3, 6.7, 7.2, 7.5]
# num_pes      = [4,   5,   5,   7,   7,   8,   9,   10,  10]
# lower_cutoff = [2.5, 3.2, 3.3, 3.4, 3.5, 3.7, 3.7, 3.8, 4.0]
first_pe     = [4.4, 4.7, 5.1, 5.5, 5.9, 6.3, 6.6, 7.0, 7.5]
num_pes      = [4,   5,   5,   7,   7,   8,   9,   10,  10]
lower_cutoff = [2.7, 2.5, 3.0, 2.9, 3.2, 3.0, 3.2, 3.4, 3.6]
outpath = '/home/evanbruggen_umass_edu/0vbb/20250812/20250917_'
campaign_spe = []
for i in range(len(measurments)):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=(lower_cutoff[i]/1000, .1),
                         first_pe=first_pe[i],
                         # centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[0].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[0].bias}.png" if savefig else None)
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()

