# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:12:04 2023

@author: heps1
"""
import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
# sys.path.append('F:/Xe/AnalysisScripts/LXe March 2023')
from MeasurementInfo import MeasurementInfo
import ProcessWaveforms_MultiGaussian
from ProcessWaveforms_MultiGaussian import WaveformProcessor as WaveformProcessor
from RunInfo_LXe import RunInfo
import heapq
from scipy import signal
from scipy.optimize import curve_fit
# import AnalyzePDE_2022
# from AnalyzePDE_2022 import SPE_data
import matplotlib.pyplot as plt
import matplotlib as mpl
path  = 'C:/Users/Hannah/OneDrive - University of Massachusetts/Dataruns_2023-24/DAQ_folder_google_drive/20230608_SPE_Alpha_LXe/'
#%% Alpha 32.5V
A_325V = RunInfo([path+'Run_1686252148.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.3, plot_waveforms = False)
# r.plot_hists('169','')
#%% 34V solicited alpha
A_34V = RunInfo([path+'Run_1686250649.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.3, plot_waveforms = False)
# A_34V.plot_hists('','')
#%% Alpha 34.5 
A_345V = RunInfo([path+'Run_1686249902.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.3, plot_waveforms = False)
# r.plot_hists('169','')
#%% Alpha 33V
A_33V = RunInfo([path+'Run_1686251282.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.3, plot_waveforms = False)
# r.plot_hists('169','')
#%% Alpha 30.5V
A_305V = RunInfo([path+'Run_1686255420.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.1, plot_waveforms = False)
# r.plot_hists('169','')        
#%% Alpha 30V
A_30V = RunInfo([path+'Run_1686255098.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.1, plot_waveforms = False)
# r.plot_hists('169','')
#%% Alpha 29V
A_29V = RunInfo([path+'Run_1686254519.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.01, plot_waveforms = False)
# r.plot_hists('169','')
#%% Alpha 32V
A_32V = RunInfo([path+'Run_1686253721.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.1, plot_waveforms = False)
# r.plot_hists('169','')
#%% Alpha 35V
A_35V = RunInfo([path+'Run_1686253346.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.1, plot_waveforms = False)
A_35V.plot_hists('','')
#%% Alpha 31.5V
A_315V = RunInfo([path+'Run_1686252429.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.1, plot_waveforms = False)
A_315V.plot_hists('','')
#%% Alpha 31V
A_31V = RunInfo([path+'Run_1686251590.hdf5'], do_filter = False, is_solicit = False, upper_limit = 5, baseline_correct = True, prominence = 0.1, plot_waveforms = False)
# A_31V.plot_hists('','')
#%% Solicited
solicit = RunInfo([path+'Run_1686255720.hdf5'], do_filter = False, is_solicit = True, upper_limit = 5, baseline_correct = True)
#%%
alpha_campaign =[A_31V,A_315V,A_325V,A_34V,A_33V,A_305V,A_30V,A_29V,A_32V,A_35V]
x = []
xerr = []
s = []
serr = []
N = []
mean_vals = []
campaign_alpha = []
for i in range(len(alpha_campaign)):
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.01
    info_alpha.condition = 'LXe'
    info_alpha.date = alpha_campaign[i].date
    info_alpha.temperature = '169K'
    info_alpha.bias = alpha_campaign[i].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = 60
    info_alpha.data_type = 'h5'
    wp = WaveformProcessor(info_alpha, run_info_self = alpha_campaign[i], run_info_solicit = solicit, baseline_correct = True)
    wp.process(do_spe = False, do_alpha = True)
    j, k = wp.get_alpha()
    l = wp.get_alpha_sigma()
    m = wp.get_alpha_sigma_err()
    x.append(j)
    xerr.append(k)
    s.append(l)
    serr.append(m)
    N.append(wp.get_N())
    mean_vals.append(wp.get_mean_val())
    campaign_alpha.append(wp)
    wp.plot_alpha_histogram(savefig = True, path = 'C:/Users/Hannah/Documents/GitHub/SPE-Analysis/2023_June8_teflon/alpha_'+str(alpha_campaign[i].bias)+'.png')
    print(info_alpha.bias)

#%%
# biases = [31.0,31.5,32.5,34.0,33.0,30.5,30.0,29.0,32.0,35.0]
biases = [A.bias for A in alpha_campaign]
#%% figures
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(biases, x, yerr = xerr, fmt = 'bo')
ax.set_ylabel('Pulse Amplitude',fontsize=20)
ax.set_xlabel('Bias Voltage [V]',fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
#%%
import pandas as pd
data = {
        "Alpha Pulse Amplitude [V]": x,
        "Alpha Pulse Amplitude [V] error": xerr,
        "Bias Voltage [V]": biases,
        "Bias Voltage [V] error": [0.0025*V + 0.015 for V in biases]
        }
df = pd.DataFrame(data)
df.to_csv('C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/2023June8_Alphas.csv')