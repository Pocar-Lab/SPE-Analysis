# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2024

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
file_path = 'june-2024/' # folder with H5 data files
files = ['Run_1718916316.hdf5','Run_1718918399.hdf5']
proms = [ .01, .015]
upperlim = [ 5, 1]
files  += [ 'Run_1718921229.hdf5', 'Run_1718921904.hdf5', 'Run_1718928445.hdf5']
proms  += [ .04, .01, .06 ]
upperlim  += [ 1, 1, 5]
files += [ 'Run_1718922518.hdf5', 'Run_1718923142.hdf5',   'Run_1718923541.hdf5']
proms += [ .01, .01, .04 ]
upperlim += [ 5, 1, 5]
files += ['Run_1718924484.hdf5',    'Run_1718924966.hdf5',    'Run_1718925432.hdf5']
proms += [ .04, .04, .04]
upperlim += [ 1, 5, 5]
files += [ 'Run_1718926961.hdf5', 'Run_1718928042.hdf5', 'Run_1718927452.hdf5' ]
proms += [ .04, .03, .01]
upperlim += [ 1, 5, .3]
runs_alpha_1us = []
i = 0
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file]], do_filter=False,
                            upper_limit=upperlim[file+i], baseline_correct=True,
                            prominence=proms[file+i], plot_waveforms=False,)
    # run_alpha_1us.plot_hists('','')
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

runs_alpha_1us[3].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [40, 40]
bins += [40, 40, 40]
bins += [41, 38, 42]
bins += [40, 44, 42]
bins += [40, 35, 36]
for n in range(len(runs_alpha)):
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

#%%
storage_path = '/run/media/ed/My Passport/ed/'

p = dill.Pickler(open(storage_path+'june-20-2024-campaign-alpha.pickle', 'wb'))
p.fast = True
p.dump(campaign_alpha)
# campaign_alpha = p.load()

p = dill.Unpickler(open(storage_path+"CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()


v_bd = 26.93 # from June 20 SPE
v_bd_err = 0.198
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

p = dill.Pickler(open("june-20-2024-alpha_data.pickle","wb"))
p.fast = True
p.dump(alpha_data)

alpha_data.analyze_alpha()

#%%
alpha_data.plot_alpha(x='OV')

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE =  0.001782
alpha_data.plot_PDE(N*PTE, out_file='june-20-2024-alpha-amp-lxe.csv')
