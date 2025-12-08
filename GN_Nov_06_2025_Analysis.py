#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO

Created on Mon Nov 6 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

from typing import Counter
import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
from AnalyzePDE import SPE_data
from multiprocessing import Pool
plt.style.use('misc/nexo.mplstyle')

### Low (normal) Illumination at 2.6V 10%DC

condition = 'GN'
temperature = 185
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '../data/20251106_SPE_GN_190K/2.60V/'
files, biases, run_spe_pre_bd = get_files(path)

# Run_1762458140 should be 33.5

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
measurments = measurments[2:]

# Fit Gauss to peaks and find the best value for peak location
savefig = True
first_pe     = [5.6, 6.1, 6.0, 6.6, 7.4, 7.7, 8.2, 8.1, 8.1]
lower_cutoff = [4.9, 4.7, 5.0, 5.1, 4.8, 5.2, 5.3, 5.0, 5.4]
num_pes      = [3,   4,   4,   5,   5,   7,   4,   6,   6, ]
outpath = '/home/evanbruggen_umass_edu/0vbb/20251106/20251110_'
campaign_spe = []
for i in range(len(measurments) - 1):
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
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()



### High Illumination at 2.75V 90%DC

condition = 'GN'
temperature = 185
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '../data/20251106_SPE_GN_190K/2.75V/'
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
    m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
savefig = True
first_pe     = [4.5, 5.0, 5.3, 5.9, 6.1, 6.7, 7.1, 7.5, 7.8,]
lower_cutoff = [3.4, 3.2, 3.4, 3.5, 3.5, 2.5, 3.9, 4.2, 4.4,]
num_pes      = [4,   4,   5,   5,   6,   7,   7,   7,   7,  ]
outpath = '/home/evanbruggen_umass_edu/0vbb/20251106/bright/20251110_'
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
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()


