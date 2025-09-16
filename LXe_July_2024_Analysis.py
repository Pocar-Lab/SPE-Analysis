# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 2024

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

%load_ext autoreload
%autoreload 2
import sys
from MeasurementInfo import MeasurementInfo
import numpy as np
from RunInfo import RunInfo
import heapq
from scipy import signal
from scipy.optimize import curve_fit
import AnalyzePDE
from AnalyzePDE import SPE_data
from AnalyzePDE import Alpha_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import ProcessWaveforms_MultiGaussian
from ProcessWaveforms_MultiGaussian import WaveformProcessor as WaveformProcessor
import pickle
import dill
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy
plt.style.use('misc/nexo.mplstyle')

#%% ALPHA - 1us
#1us, no gain
invC_alpha = 1143.98e-6
invC_alpha_err = 0.12e-6
file_path = 'july-9-2024/' # folder with H5 data files
files = [ 'Run_1720540763.hdf5', 'Run_1720542050.hdf5', 'Run_1720542913.hdf5', ]
proms = [ .02, .02, .01]
upperlim = [ 1, 1,1]
files += [ 'Run_1720543766.hdf5', 'Run_1720544600.hdf5', 'Run_1720545288.hdf5', ]
proms += [ .02, .02, .01]
upperlim += [ 1, 1,1]
files += [ 'Run_1720545767.hdf5', 'Run_1720541734.hdf5', 'Run_1720542682.hdf5', ]
proms += [ .06, .02, .01]
upperlim += [ 3, 1,1]
files += [ 'Run_1720543160.hdf5', 'Run_1720544304.hdf5', 'Run_1720544954.hdf5', 'Run_1720545596.hdf5' ]
proms += [ .02, .02, .05, .02]
upperlim += [ 1, 1, 2, 1]
runs_alpha_1us = []
i = 0
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file+i]], do_filter=False,
                            upper_limit=upperlim[file+i], baseline_correct=True,
                            prominence=proms[file+i], plot_waveforms=False,)
    # run_alpha_1us.plot_hists('','')
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

runs_alpha_1us[0].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [40, 40, 40]
bins += [40, 40, 40]
bins += [40, 40, 40]
bins += [40, 40, 40, 40]
for n in range(len(runs_alpha)):
# if n := 1:
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.001
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[n].date
    info_alpha.temperature = 170
    info_alpha.bias = runs_alpha[n].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = bins[n]
    # print(f'{n=}')
    # print(f"{info_alpha.bias=}")
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[n], baseline_correct = True,
                           no_solicit = True, cutoff=(0,10))
    wp.process(do_spe = False, do_alpha = True)
    # j, k = wp.get_alpha()
    wp.plot_alpha_histogram(peakcolor = 'blue', with_fit=True)
    campaign_alpha.append(wp)
    # break

invC_alpha = 1143.98e-6
invC_alpha_err = 0.12e-6
invC_spe = 11404e-6
invC_spe_err = 11e-6

v_bd = 27.08 # from Oct 17 2024 SPE
v_bd_err = 0.098

alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

alpha_data.analyze_alpha()

#%%
alpha_data.plot_alpha(x='OV')

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE =  0.001782
alpha_data.plot_PDE(N*PTE, out_file='2024July9_Alpha.csv')

bias_vals = []
bias_err = []
alpha_vals = []
alpha_err = []
for wp in campaign_alpha:
    bias_vals.append(wp.info.bias)
    bias_err.append(0.0025 * wp.info.bias + 0.015)
    # bias_err.append(0.005)
    curr_alpha = wp.get_alpha()
    alpha_vals.append(curr_alpha[0])
    alpha_err.append(curr_alpha[1])


data = {
    # 'ov': data_x, 'ov error': data_x_err,
    'Bias Voltage [V]': bias_vals, 'Bias Voltage error [V]': bias_err,
    'Alpha Pulse Amplitude [V]': alpha_vals, 'Alpha Pulse Amplitude error [V]': alpha_err,
    # 'num_det': self.num_det_photons, 'num_det error': self.num_det_photons_err,
    # 'pde': data_y, 'pde error': data_y_err,
}
df = pd.DataFrame(data)
df.to_csv('2024July9_Alpha.csv')
