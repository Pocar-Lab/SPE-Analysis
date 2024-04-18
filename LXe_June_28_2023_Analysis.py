# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:28:38 2023

@author: hpeltzsmalle
@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

%load_ext autoreload
%autoreload 2
%autoindent

import sys
import numpy as np
from MeasurementInfo import MeasurementInfo
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

run_spe_solicited = RunInfo(['data/Run_1666778594.hdf5'],  specifyAcquisition = True, acquisition ='Acquisition_1666781843', do_filter = True, is_solicit = True, upper_limit = .5, baseline_correct = True)

files = ['Acquisition_1666778815','Acquisition_1666779156','Acquisition_1666782030','Acquisition_1666780808','Acquisition_1666779491']
proms = [0.035,0.04,0.04,0.05,0.055]
upperlim = [4, 4, 4, 4, 4]
runs = []
for file in range(len(files)):
    run_spe = RunInfo(['Run_1666778594.hdf5'], specifyAcquisition = True, acquisition = files[file], do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file])
    runs.append(run_spe)
biases = [run.bias for run in runs]

range_lows = [0.06, 0.08, 0.09, 0.1]
centers = [0.09, 0.11, 0.12, 0.14]
range_highs = [0.403, 0.5, 0.56, 0.67]
campaign_spe = []
for i in range(len(runs)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'GN'
    info_spe.date = runs[i].date
    info_spe.temperature = '167'
    info_spe.bias = biases[i]
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 220
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i])
    wp_spe.process(do_spe = True, do_alpha = False)
    campaign_spe.append(wp_spe)
invC_spe_filter = 0.19261918346201742
invC_spe_err_filter = 0.0021831140214106596
spe = SPE_data(campaign_spe, invC_spe_filter, invC_spe_err_filter, filtered = True)

##%% CORRELATED AVALANCHE SPE
#file_path = 'self/' # folder with H5 data files
#run_spe_solicited = RunInfo([file_path+'Run_1648191265.hdf5'], do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
#files = ['Run_1648176286','Run_1648179496', 'Run_1648181807', 'Run_1648184235', 'Run_1648186910'] #, 'Run_1648186910', 'Run_1648171846'
#proms = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
#upperlim = [5, 5, 5, 5, 5, 5]
#runs = []
#for file in range(len(files)):
#    run_spe = RunInfo([file_path + files[file] + '.hdf5'], do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file])
#    runs.append(run_spe)
#biases = [run.bias for run in runs] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
##%%
#a_guess = 0.01689
#b_guess = -0.4739
#range_lows = [0.05, 0.059, 0.071, 0.09, 0.09, 0.04]
#centers = [biases[i]*a_guess+b_guess for i in range(len(runs))]
#range_highs = [centers[i]*4 + range_lows[i] for i in range(len(runs))]
##%%
#campaign_spe = []
#for i in range(len(runs)):
#    info_spe = MeasurementInfo()
#    info_spe.condition = 'Vacuum'
#    info_spe.date = 'March 2022'
#    info_spe.temperature = 190
#    info_spe.bias = runs[i].bias
#    info_spe.baseline_numbins = 50
#    info_spe.peaks_numbins = 200
#    info_spe.data_type = 'h5'
#    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i])
#    wp_spe.process(do_spe = True, do_alpha = False)
#    campaign_spe.append(wp_spe)
##%%
#invC_spe_filter = 0.19261918346201742
#invC_spe_err_filter = 0.0021831140214106596
#spe = SPE_data(campaign_spe, invC_spe_filter, invC_spe_err_filter, filtered = True)






#%% ALPHA - 1us
#1us, no gain, no filter
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
file_path = 'alpha-june-28/' # folder with H5 data files
files = ['Run_1687969527.hdf5', 'Run_1687969770.hdf5', 'Run_1687970855.hdf5',
         'Run_1687971807.hdf5', 'Run_1687972206.hdf5', 'Run_1687971430.hdf5',
         'Run_1687972646.hdf5', 'Run_1687973035.hdf5', 'Run_1687973389.hdf5',
         'Run_1687973682.hdf5', 'Run_1687974000.hdf5', 'Run_1687974651.hdf5', 'Run_1687975037.hdf5']
proms = [0.08,0.08,0.08, 0.09,0.05,0.2, 0.4,0.35,0.25, 0.25,0.5,0.8,0.2]
upperlim = [.2,.3,.4, 2,.2,.8, 1.3,1,0.5, 1.2,2.5,1.7,1]
runs_alpha_1us = []
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file]], do_filter=False, upper_limit=upperlim[file], baseline_correct=True, prominence=proms[file], plot_waveforms=False)
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
# runs_alpha_1us[0].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
# TODO 29, 31
# TODO fix 33V? chi^2 is very high
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [36,35,34, 32,29,30, 33,32,37,35, 33,43,30]
for i in range(len(runs_alpha)):
    info_alpha = MeasurementInfo()
    # info_alpha.min_alpha_value =  0.029
    info_alpha.min_alpha_value = 0.3 if i > 8 else 0.029
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[i].date
    info_alpha.temperature = 167
    info_alpha.bias = runs_alpha[i].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = bins[i]
    # print(f"{i=}")
    # print(f"{info_alpha.bias=}")
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[i], baseline_correct = True, no_solicit = True, cutoff=(0,10))
    wp.process(do_spe = False, do_alpha = True)
    j, k = wp.get_alpha()
    wp.plot_alpha_histogram(peakcolor = 'blue')
    campaign_alpha.append(wp)

#%%
# with open('LED-SPE/campaign_alpha.pickle', 'wb') as f:
#     dill.dump(campaign_alpha, f)

p = dill.Unpickler(open("/run/media/ed/My Passport/ed/CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()

v_bd = 27.69
v_bd_err = 0.06
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

alpha_data.analyze_alpha()

#%%
alpha_data.plot_alpha()

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE = .005530 # diffusive reflector
# PTE = .010736 # specular reflector
alpha_data.plot_PDE(N*PTE, out_file='2023_June_28_Alpha-new.csv')
