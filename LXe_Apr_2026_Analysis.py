# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2024

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
# from src.AnalyzePDE import SPE_data
from multiprocessing import Pool
from AnalyzePDE import Alpha_data
import pandas as pd
plt.style.use('misc/nexo.mplstyle')

condition = 'LXe'
temperature = 171
#1us, 10x gain, filter on - values from Noahs calibration report
# invC_filter = 0.011547
# invC_err_filter = 0.000098
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
path = '../data/20260429_SPE_LED_405_LXe/'
files, biases, run_spe_pre_bd = get_files(path)

def proc(file):
    return ProcessWaveforms([path+file], do_filter=False,
                            baseline_correct=True, poly_correct=False,
                            # TODO why does upper_limit=-1 not work
                            upper_limit=1, prominence=.01,
                            # plot_waveforms=True, num_waveforms=500,
                            )
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run)
    # Plot preliminary histogram
    # m.plot_histogram()
    measurments.append(m)

#%%
campaign_alpha = []
bins = [40, 43, 42]
bins += [41, 41, 44]
bins += [40, 39]
bins += [40, 40]
bins += [42, 39, 40, 40]
for n in range(0,len(measurments)):
    wp = ProcessHist(measurments[n],
                     baseline_correct=True,
                     # no_solicit=True,
                     cutoff=(0,10))
    wp.process_alpha(0.001 if measurments[n].bias < 36 else 0.02, bins[n])
    # j, k = wp.get_alpha()
    wp.plot_alpha_histogram(bins[n], peakcolor='blue', with_fit=True)
    campaign_alpha.append(wp)
    # break


bias_vals = []
bias_err = []
alpha_vals = []
alpha_err = []
sigma_vals = []
sigma_err = []
for wp in campaign_alpha:
    bias_vals.append(wp.info.bias)
    bias_err.append(0.0025 * wp.info.bias + 0.015)
    alpha_vals.append(wp.alpha_fit['fit'].params['center'].value)
    alpha_err.append(wp.alpha_fit['fit'].params['center'].stderr)
    sigma_vals.append(wp.alpha_fit['fit'].params['sigma'].value)
    sigma_err.append(wp.alpha_fit['fit'].params['sigma'].stderr)
data = {
    # 'ov': data_x, 'ov error': data_x_err,
    'bias': bias_vals, 'bias error': bias_err,
    'amps': alpha_vals, 'amps error': alpha_err,
    'sigma': sigma_vals, 'sigma error': sigma_err,
    # 'num_det': self.num_det_photons, 'num_det error': self.num_det_photons_err,
    # 'pde': data_y, 'pde error': data_y_err,
}
df = pd.DataFrame(data)
df.to_csv('../results/20241017_Alpha.csv')


alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)
alpha_data.analyze_alpha()

alpha_data.plot_alpha(x='OV')

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE =  0.001782
alpha_data.plot_PDE(N*PTE, out_file='20260429_Alpha.csv')
