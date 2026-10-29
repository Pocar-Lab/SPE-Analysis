# -*- coding: utf-8 -*-
"""
Created on Thu Sep 3 2026

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
from multiprocessing import Pool
import pandas as pd
plt.style.use('misc/nexo.mplstyle')

### Alpha

condition = 'LXe'
temperature = 170
#1us, 10x gain, filter on - values from Noahs calibration report
# invC_filter = 0.011547
# invC_err_filter = 0.000098
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
path = '../data/20260916_Alpha_LXe_170K/'
files, biases, run_spe_pre_bd = get_files(path)

def proc(file):
    return ProcessWaveforms([path+file], do_filter=False,
                            baseline_correct=True, poly_correct=False,
                            # TODO why does upper_limit=-1 not work
                            upper_limit=2, prominence=.01,
                            # plot_waveforms=True, num_waveforms=500,
                            )
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run)
    # Plot preliminary histogram
    m.plot_histogram()
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
    wp.process_alpha(0.001 if measurments[n].bias < 34 else 0.1, bins[n])
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
df.to_csv('../results/20260916_Alpha.csv')


alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)
alpha_data.analyze_alpha()

alpha_data.plot_alpha(x='OV')

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE =  0.001782
alpha_data.plot_PDE(N*PTE, out_file='20260429_Alpha.csv')

### SPE

condition = 'LXe'
temperature = 170
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '../data/20260916_SPE_LXe_170K/'

files, biases, pre_bd = rwf.get_files(path)
pre_bd_baseline = pre_bd and rwf.Waveform(path+pre_bd)

def proc(file):
    return rwf.Waveform(path+file, condition, temperature, pre_bd_baseline)
with Pool(len(files)+1) as p:
    measurements = p.map(proc, iter(files))

for m in measurements:
    prochist.plot_multigaussian(m.peaks, subtracted=True)

first_pe     = [4.8, 5.1, 5.9, 6.1, 6.4, 7.2, 7.2, 8.2, 8.5]
num_pes      = [4,   4,   4,   5,   6,   6,   6,   6,   6]
lower_cutoff = [3.8, 3.5, 3.8, 4,   4.1, 4.3, 4.3, 4.5, 4.6, ]

# noise_means = [m.baseline_mean for m in measurements]
# noise_stds = [m.baseline_std for m in measurements]
outpath = '/home/evanbruggen_umass_edu/0vbb/20260916/20260916_'
# outpath = '../results/pics/20260916/20260916_'
spe_amps = []
for i in range(0,len(measurements)):
    fit = prochist.fit_multigaussian(np.array(measurements[i].peaks['all']),
                            cutoff=(lower_cutoff[i], 100),
                            first_pe=first_pe[i],
                            peak_range=(1,num_pes[i]),
                            baseline_mean=pre_bd_baseline.baseline_mean,
                            baseline_width=pre_bd_baseline.baseline_sigma,
                            )
    prochist.plot_multigaussian(measurements[i].peaks['all'], fit,
                                # save_path=outpath+f"{biases[i]}_hist.png"
                                )
    spe_amp = prochist.plot_spe(fit, (1,num_pes[i]), pre_bd_baseline.baseline_mean,
                                pre_bd_baseline.baseline_sigma,
                                # save_path=outpath+f"{biases[i]}_spe.png"
                                )
    spe_amps.append(spe_amp)

ares.plot_breakdown(biases, spe_amps)
