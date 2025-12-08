#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anlaysis of Sep 22 2025 SPE data for characterziation in GN at 264K

Created on Mon Nov 3 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
from AnalyzePDE import SPE_data
from multiprocessing import Pool
plt.style.use('misc/nexo.mplstyle')

condition = 'GN'
temperature = 264
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '../data/20250922_SPE_GN_264K/'
files, biases, run_spe_pre_bd = get_files(path)

def proc(file):
    return ProcessWaveforms([path+file], do_filter=True,
                            baseline_correct=True, poly_correct=True,
                            upper_limit=-1, prominence=.002)
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    # m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
savefig = True
first_pe     = [3.6, 4.2, 4.6, 4.9, 5.4, 6.0, 6.1, 6.6, 6.8]
lower_cutoff = [2.2, 2.4, 2.0, 2.0, 2.3, 2.6, 2.7, 2.8, 3.2]
num_pes      = [3,   5,   6,   7,   7,   6,   10,  12,  8]
outpath = '/home/evanbruggen_umass_edu/0vbb/20250922/20251103_'
campaign_spe = []
for i in range(len(measurments)):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=(lower_cutoff[i]/1000, .1),
                         # cutoff=(lower_cutoff[i]/1000, upper_cutoff[i]/1000),
                         first_pe=first_pe[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    # wp_spe.plot_peak_histogram(log_scale=True,
    #                            savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    # wp_spe.plot_spe(fit_origin=False,
    #                 savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()

