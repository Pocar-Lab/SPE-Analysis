#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redone analysis for the July 11 SPE data for CA in GN

Created on Tue Sep 23 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
from AnalyzePDE import SPE_data
from multiprocessing import Pool
plt.style.use('misc/nexo.mplstyle')


### July 11 2023 SPE for CA

condition = 'GN'
temperature = 171
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '../data/20230711_SPE_GN_171K/CA data/405nm/'
files, biases, run_spe_pre_bd = get_files(path)

# NOTE for prom > 5mV, numbins in ProcessHist has to be tripled
def proc(file):
    return ProcessWaveforms([path+file], do_filter=True,
                            baseline_correct=True, poly_correct=True,
                            upper_limit=-1, prominence=.004)
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    # m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
savefig = False
first_pe     = [0, 5.2, 6.1, 6.5, 7.2, 7.7, 8.0, 8.4]
lower_cutoff = [0, 4.8, 4.9, 5.0, 5.1, 5.1, 5.5, 5.1]
num_pes      = [0, 3,   3,   4,   5,   6,   5,   5]
# lower_cutoff = [0, 0, 5.2, 5.0, 5.6,  5.6,  5.2,  5.0]
# first_pe     = [0, 0, 6.8, 7.1, 8.2,  8.0,  6.5,  5.6]
# upper_cutoff = [0, 0, 31., 33., 53.5, 43.5, 27.5, 24.5]
# num_pes      = [0, 0, 4,   4,   6,    5,    4,    4]
outpath = '/home/evanbruggen_umass_edu/0vbb/20230711/20250923_'
campaign_spe = []
for i in range(2,len(measurments)):
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
    # wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA(other_ca=['../results/20250812_CA_252K.csv'],)
            # out_file='../results/20230711_CA_171K_prom4.csv')
