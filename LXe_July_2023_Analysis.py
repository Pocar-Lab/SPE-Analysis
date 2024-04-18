# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 2023

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

#%% ALPHA - 1us
#1us, no gain, no filter
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
file_path = 'alpha_july/' # folder with H5 data files
files = [ 'Run_1689278620.hdf5', 'Run_1689279162.hdf5', 'Run_1689281334.hdf5',
          'Run_1689281964.hdf5', 'Run_1689282438.hdf5', 'Run_1689278958.hdf5',
          'Run_1689280412.hdf5', 'Run_1689281693.hdf5', 'Run_1689282206.hdf5']
files += ['Run_1689276244.hdf5', 'Run_1689277404.hdf5', 'Run_1689280793.hdf5']
files += ['Run_1689282865.hdf5', 'Run_1689283492.hdf5', 'Run_1689278000.hdf5', 'Run_1689279935.hdf5' ]
proms = [0.15,0.15,0.15, 0.15,0.15,0.15, 0.15,0.15,0.15, 0.08,0.1,0.15, 0.05,0.8,0.03,0.15]
upperlim = [1.2,.5,2, 2,.8,10, 1,4,1.5, .3,.4,.8, .2,10,.15,.5]
runs_alpha_1us = []
i = 0
# files = files[i:i+1]
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file]], do_filter=False,
                            upper_limit=upperlim[file+i], baseline_correct=True,
                            prominence=proms[file+i], plot_waveforms=False)
    # run_alpha_1us.plot_hists('','')
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [34,40,40, 40,38,40, 36,40,41, 35,38,37, 40,37,31,37]
for n in range(len(runs_alpha)):
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.029
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
    j, k = wp.get_alpha()
    wp.plot_alpha_histogram(peakcolor = 'blue')
    campaign_alpha.append(wp)
    # break

#%%
p = dill.Pickler(open("/media/ed/My Passport/ed/Alpha-July-2023.pickle","wb"))
p.fast = True
p.dump(campaign_alpha)

p = dill.Unpickler(open("/media/ed/My Passport/ed/CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()



# v_bd = 27.69
# v_bd_err = 0.06
v_bd = 27.13
v_bd_err = 0.225
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

p = dill.Pickler(open("/media/ed/My Passport/ed/CA-july-12-alpha_data.pickle","wb"))
p.fast = True
p.dump(alpha_data)

p = dill.Unpickler(open("/media/ed/My Passport/ed/CA-july-12-alpha_data.pickle","rb"))
p.fast = True
alpha_data = p.load()

alpha_data.analyze_alpha()

#%%
alpha_data.plot_alpha(out_file="july-2023-alpha_amp.csv")

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE = 0.005221
alpha_data.plot_PDE(N*PTE)
