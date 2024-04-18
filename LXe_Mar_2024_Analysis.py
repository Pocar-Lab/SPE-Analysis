# -*- coding: utf-8 -*-
"""
Created on Thu March 28 2024

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

plt.style.use('misc/nexo.mplstyle')

#%% ALPHA - 1us
#1us, no gain
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
file_path = 'march-28-2024/' # folder with H5 data files
files = [ 'Run_1711658600.hdf5', 'Run_1711659190.hdf5', 'Run_1711659564.hdf5', ]
proms = [ .05, .015, .5 ]
upperlim = [ 5, 1, 6]
files += [ 'Run_1711660058.hdf5', 'Run_1711660299.hdf5', 'Run_1711660791.hdf5', ]
proms += [ .05, .3, .05 ]
upperlim += [ 5, 5, 6]
files += [ 'Run_1711661069.hdf5', 'Run_1711661510.hdf5', 'Run_1711661820.hdf5', ]
proms += [ .05, .05, .05 ]
upperlim += [ 2, 5, 6]
files += [ 'Run_1711662120.hdf5', 'Run_1711662413.hdf5', 'Run_1711662639.hdf5', 'Run_1711663027.hdf5']
proms += [ .05, .05, .1, .05 ]
upperlim += [ .5, 7, 2, 1]
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

runs_alpha_1us[2].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [36, 40, 39]
bins += [36, 40, 39]
bins += [34, 40, 40]
bins += [40, 40, 40, 40]
for n in range(len(runs_alpha)):
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
    # wp.plot_alpha_histogram(peakcolor = 'blue', with_fit=True)
    campaign_alpha.append(wp)
    # break

#%%
storage_path = '/run/media/ed/My Passport/ed/'

p = dill.Pickler(open(storage_path+'march-2024-campaign-alpha.pickle', 'wb'))
p.fast = True
p.dump(campaign_alpha)
# campaign_alpha = p.load()

p = dill.Unpickler(open(storage_path+"CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()


v_bd = 27.83 # from June 28 SPE
v_bd_err = 0.1
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

p = dill.Pickler(open("march-2024-alpha_data.pickle","wb"))
p.fast = True
p.dump(alpha_data)

alpha_data.analyze_alpha()

#%%
alpha_data.plot_alpha(x='OV')

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
# PTE = 0.0042
PTE_dif = 0.001611
PTE_spec = 0.001352
PTE_none = 0.001360
alpha_data.plot_PDE(N*PTE_dif, out_file='mar-28-2024-alpha-amp-lxe.csv')
