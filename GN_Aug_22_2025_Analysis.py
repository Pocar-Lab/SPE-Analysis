#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPE analysis for data collected on Aug 22 2025 at 200K

Created on Fri Aug 22 2025

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


### Aug 22th SPE

condition = 'GN'
temperature = 200
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = 'data/20250822_SPE_GN_200K/'
files = os.listdir(path)

# Sort files by bias voltage
biases = []
for f in files:
    with h5py.File(path+f, 'r') as hdf: # Find bias for each file
        biases.append(hdf['RunData'][next(iter(hdf['RunData'].keys()))].attrs['Bias(V)'])
biases, files = zip(*sorted(zip(biases, files))) # Sort based on biases

# Extract pre-breakdown baseline if it was collected
if biases[0] < 25:
    file_pre_bd = files[0]
    run_spe_pre_bd = ProcessWaveforms([path+file_pre_bd], do_filter=True,
                                      is_pre_bd=True, upper_limit=0.063, baseline_correct=True)
    files = files[1:]
    biases = biases[1:]
else:
    file_pre_bd = None

# Loop over all files and find peaks
runs = []
proms = [0.0050, 0.0050, 0.0050,
         0.0050, 0.0050, 0.0050,
         0.0050, 0.0050, 0.0050]
prom = .002
upperlim = -1
for file in range(len(files)):
    run = ProcessWaveforms([path+files[file]], do_filter=True,
                           baseline_correct=True, poly_correct=True,
                           upper_limit=upperlim, prominence=prom)
    runs.append(run)

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# with open('20250822_measurments.pkl', 'wb') as file:
#     pickle.dump(measurments, file)

# Fit Gauss to peaks and find the best value for peak location
pe_center    = [4.9, 5.1, 5.7, 6.1, 6.8, 7.3, 7.7, 8.2, 8.5]
num_pes      = [4,   4,   5,   5,   5,   7,   7,   8,   8]
lower_cutoff = [3.8, 4.4, 4.7, 4.9, 4.6, 4.8, 4.8, 4.8, 4.8]
savefig = True
centers_guesses = []
for pe_guess, num_pe in zip(pe_center, num_pes):
    centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
outpath = '/home/ed/pictures/0vbb/20250822/20250825_'
campaign_spe = []
for i in range(len(measurments)):
    cutoff = (lower_cutoff[i]/1000, .1)
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
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()


