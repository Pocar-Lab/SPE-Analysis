# -*- coding: utf-8 -*-
"""
Created on Aug 10 2023

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

%load_ext autoreload
%autoreload 2
%autoindent

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

#%% ALPHA - 1us
#1us, no gain, no filter
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
file_path = 'aug-alpha/' # folder with H5 data files
files = ['Run_1691679909.hdf5', 'Run_1691680314.hdf5',  'Run_1691680611.hdf5' ]
proms = [ .025, .015, .1 ]
upperlim = [ 1, .4, 2]
files += [ 'Run_1691682631.hdf5',  'Run_1691682887.hdf5']
proms += [ .01, .015 ]
upperlim += [ .3, .5 ]
files += ['Run_1691683735.hdf5', 'Run_1691684246.hdf5',
         'Run_1691683999.hdf5', 'Run_1691684510.hdf5', 'Run_1691684917.hdf5']
proms += [ .04, .015, .015, .015, .1]
upperlim += [ .6, .2, .5, .2, 1.5]
files += ['Run_1691681764.hdf5','Run_1691683166.hdf5'] # 28.5
proms += [ .005, .005]
upperlim += [ .025, .06]
files += ['Run_1691685248.hdf5', 'Run_1691685841.hdf5', 'Run_1691686415.hdf5', 'Run_1691686760.hdf5']
proms += [ .02, .005, .015, .05 ]
upperlim += [ .1, .05, .2, 1]
runs_alpha_1us = []
i = 0
# files = files[i:i+2]
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file]], do_filter=False,
                            upper_limit=upperlim[file+i], baseline_correct=True,
                            prominence=proms[file+i], plot_waveforms=False)
    # run_alpha_1us.plot_hists('','')
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

runs_alpha_1us[0].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [40, 40, 53, 40, 40]
bins += [40, 55, 40, 40, 40]
bins += [50, 40]
bins += [36, 39, 40, 42]
for n in range(len(runs_alpha)):
# if n := 15:
    # if n < 2:
    #     continue
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.001
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[n].date
    info_alpha.temperature = 167
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

#%%
storage_path = '/media/ed/My Passport/ed/'

with open(storage_path+'aug-10-campaign-alpha.pickle', 'wb') as f:
    dill.dump(campaign_alpha, f)

# with open('LED-SPE/SPE-June-28.pickle', 'rb') as f:
#     spe = dill.load(f)

p = dill.Unpickler(open("/media/ed/My Passport/ed/aug-10-campaign-alpha.pickle","rb"))
p.fast = True
# p.dump(campaign_alpha)
campaign_alpha = p.load()

p = dill.Unpickler(open(storage_path+"CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()



v_bd = 27.28
v_bd_err = 0.05
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

p = dill.Pickler(open("July-2023-alpha_data.pickle","wb"))
p.fast = True
p.dump(alpha_data)

alpha_data.analyze_alpha(out_file='aug-10-2023-lxe.csv')

#%%
alpha_data.plot_alpha()#out_file='aug-10-2023_alpha_amp.csv'

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
# PTE = 0.0042
PTE_dif = 0.001611
PTE_spec = 0.001352
PTE_none = 0.001360
alpha_data.plot_PDE(N*PTE_dif)
