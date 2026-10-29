#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPE: Created on Fri May 9 16:00:00 2025

Example template for a new SPE analysis

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
from AnalyzePDE import SPE_data
import h5py
plt.style.use('misc/nexo.mplstyle')


condition = 'GN' # Vac, GN, GXe, LXe
temperature = 170

# Calibration Constant: 1us, 10x gain, filter on - values from Noahs calibration report
invC = ufloat(0.011547, 0.000098)

# NOTE Change path name to folder with data files
path ='../data/20250702_SPE_GN/'
# NOTE Add files here:
files = []
file_pre_bd = 'XXX.hdf5'

# Sort files by bias voltage
biases = []
for f in files:
    with h5py.File(f, 'r') as hdf:
        biases.append(hdf['RunData'][next(iter(hdf['RunData'].keys()))].attrs['Bias(V)'])
_, files = zip(*sorted(zip(biases, files)))

run_spe_pre_bd = ProcessWaveforms([path+file_pre_bd], do_filter=True, baseline_correct=True,
                                  upper_limit=0.063)
run_spe_pre_bd.get_baseline()

# loop over all files
# NOTE Modify prominence and upper limit for your data:
proms = [.005 for _ in files]
upperlim = .4

runs = []
for file in range(len(files)):
    runs += ProcessWaveforms([path+files[file]], do_filter=True,
                             baseline_correct=True, poly_correct=True,
                             upper_limit=upperlim, prominence=proms[file])

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    # Plot preliminary histogram
    m.plot_histogram()
    measurments += m

# Fit Gauss to peaks and find the best value for peak location
# NOTE Modify  for your data:
pe_center_guesses = [5.7, 4.8, 8.5, 6.1, 5.6, 7.1, 8.5, 7.7, 6.8]
num_pes = [4, 3, 5, 4, 3, 5, 5, 4, 5]
cutoff = (.004,.07)

centers_guesses = []
for pe_guess, num_pe in zip(pe_center_guesses, num_pes):
    centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
outpath = '/home/ed/pictures/0vbb/20250708/20250720_'
campaign_spe = []
for i in range(len(measurments)):
    wp_spe = ProcessHist(measurments[i], run_info_self=runs[i],
                         run_info_pre_bd=run_spe_pre_bd,
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         # cutoff=cutoffs[i], centers=centers_guesses[i],
                         cutoff=cutoff, centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
                         # peak_range=(1,3))
    wp_spe.process_spe()
    # wp_spe.plot_led_dark_hists(led_voltages[i])
    wp_spe.plot_peak_histogram(log_scale=False, savefig=True,
                               path=outpath+f'hist_{info_spe.bias}.png')
    wp_spe.plot_spe(savefig=True, path=outpath+'spe_'+str(info_spe.bias)+'.png',
                    fit_origin=True)
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)
    print(info_spe.bias)


for i in range(len(campaign_spe)):
    campaign_spe[i].plot_both_histograms()

#%% plot linear fit to the breakdown voltage
spe = SPE_data(campaign_spe, invC, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()
