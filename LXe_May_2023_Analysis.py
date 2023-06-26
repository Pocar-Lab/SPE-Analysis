# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:28:38 2023

@author: hpeltzsmalle
"""

import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
sys.path.append('E:/Xe/AnalysisScripts/Starter_Pack')   
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


#%% SPE-VBD
#100ns, new shaper, 10x gain, filtered (change as needed; make sure it is the correct factor for the data set)
invC_spe_filter =  0.011959447603692185
invC_spe_err_filter = 3.881945391072933E-05
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
files = ['Run_1684433317','Run_1684434365','Run_1684434800','Run_1684435362','Run_1684435943','Run_1684436416','Run_1684436838','Run_1684437151','Run_1684437508'] #32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36
proms = [0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003]
upperlim = [0.1, 0.1, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
runs = []
for file in range(len(files)):
    run_spe = RunInfo(['E:/Xe/DAQ/'+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file])
    runs.append(run_spe)
biases = [run.bias for run in runs]
#%%
range_lows = [0.00225, 0.0032, 0.0031, 0.0037, 0.0037, 0.00475, 0.005, 0.0053, 0.006]
range_highs = [0.0205, 0.025, 0.025, 0.027, 0.03, 0.03, 0.033, 0.036, 0.04]
centers = [0.0046, 0.0049, 0.0059, 0.0065, 0.0069, 0.0072, 0.0075, 0.008, 0.008]
#%% testing cell - plot with peak fit
# set conditions, temp
n= 0
T = 170
con = 'LXe'
info_spe = MeasurementInfo()
info_spe.condition = con
info_spe.date = runs[n].date.decode('utf-8')
# info_spe.date = 'NA'
info_spe.temperature = T
info_spe.bias = biases[n]
info_spe.baseline_numbins = 50
info_spe.peaks_numbins = 150
info_spe.data_type = 'h5'
wp = WaveformProcessor(info_spe, run_info_self = runs[n], baseline_correct = True, range_low = range_lows[n], range_high = range_highs[n], center = centers[n], no_solicit = True)
wp.process(do_spe = True, do_alpha = False)
wp.plot_peak_histograms()
wp.plot_spe()

#%% make a list of ProcessWaveforms objects
campaign_spe = []
for i in range(len(runs)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'LXe'
    info_spe.date = runs[i].date.decode('utf-8')
    info_spe.temperature = 170
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 150
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i], no_solicit = True)
    wp_spe.process(do_spe = True, do_alpha = False)
    campaign_spe.append(wp_spe)
    
#%% plot linear fit to the breakdown voltage
curr_campaign = campaign_spe
filtered_spe = SPE_data(curr_campaign, invC_spe_filter, invC_spe_err_filter, filtered = True)
filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = None)

print(filtered_spe.v_bd)
print(filtered_spe.v_bd_err)

#%% plot in units of absolute gain vs OV
filtered_spe.plot_spe(in_ov = True, absolute = True, out_file = None)
#%% record BD voltage
v_bd = 27.69
v_bd_err = 0.06
#%%











#%% CORRELATED AVALANCHE SPE
run_spe_solicited = RunInfo(['E:/Xe/DAQ/Run_1648191265.hdf5'], do_filter = True, is_solicit = True, upper_limit = 1, baseline_correct = True)
files = ['Run_1648176286','Run_1648179496', 'Run_1648181807', 'Run_1648184235', 'Run_1648186910'] #, 'Run_1648186910', 'Run_1648171846'
proms = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
upperlim = [5, 5, 5, 5, 5, 5]
runs = []
for file in range(len(files)):
    run_spe = RunInfo(['E:/Xe/DAQ/' + files[file] + '.hdf5'], do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file])
    runs.append(run_spe)
biases = [run.bias for run in runs] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
#%%
a_guess = 0.01689
b_guess = -0.4739
range_lows = [0.05, 0.059, 0.071, 0.09, 0.09, 0.04]
centers = [biases[i]*a_guess+b_guess for i in range(len(runs))]
range_highs = [centers[i]*4 + range_lows[i] for i in range(len(runs))]
#%%
campaign_spe = []
for i in range(len(runs)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'Vacuum'
    info_spe.date = 'March 2022'
    info_spe.temperature = 190
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i])
    wp_spe.process(do_spe = True, do_alpha = False)
    campaign_spe.append(wp_spe)
#%%
invC_spe_filter = 0.19261918346201742
invC_spe_err_filter = 0.0021831140214106596
spe = SPE_data(campaign_spe, invC_spe_filter, invC_spe_err_filter, filtered = True)
#%%






#%% ALPHA - 100ns
#100ns, new shaper, no gain, no filter
invC_alpha_100ns =  0.004074442176917963
invC_alpha_err_100ns = 1.4320358414009397E-05
files = ['Acquisition_1684439574', 'Acquisition_1684439834','Acquisition_1684440058', 'Acquisition_1684440328','Acquisition_1684440542', 'Acquisition_1684440766', 'Acquisition_1684441008','Acquisition_1684441249','Acquisition_1684441553', 'Acquisition_1684441781', 'Acquisition_1684442031', 'Acquisition_1684442292', 'Acquisition_1684442515', 'Acquisition_1684442713', 'Acquisition_1684442986', 'Acquisition_1684443190', 'Acquisition_1684443416']
proms = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
upperlim = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
runs_alpha_100ns = []
for file in range(len(files)):
    run_alpha_100ns = RunInfo(['C:/Users/heps1/Downloads/Run_1684439289.hdf5'], specifyAcquisition = True, acquisition = files[file], do_filter = False, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], is_solicit = False)
    runs_alpha_100ns.append(run_alpha_100ns)
biases = [run.bias for run in runs_alpha_100ns] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)
#%% ALPHA - 1us
#1us, no gain, no filter
invC_alpha_1us =  0.004074442176917963 # *************PLACEHOLDER
invC_alpha_err_1us = 1.4320358414009397E-05 # ********PLACEHOLDER
files = ['Acquisition_1684429006', 'Acquisition_1684429323','Acquisition_1684429598', 'Acquisition_1684429828','Acquisition_1684429901', 'Acquisition_1684430142', 'Acquisition_1684430326','Acquisition_1684430543','Acquisition_1684430787', 'Acquisition_1684431075', 'Acquisition_1684431293', 'Acquisition_1684431526', 'Acquisition_1684431824', 'Acquisition_1684432042', 'Acquisition_1684432244', 'Acquisition_1684432473', 'Acquisition_1684432689'] #,'Acquisition_1684432883','Acquisition_1684433075'
proms = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01, 0.01]
upperlim = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
runs_alpha_1us = []
for file in range(len(files)):
    run_alpha_1us = RunInfo(['C:/Users/heps1/Downloads/Run_1684428416.hdf5'], specifyAcquisition = True, acquisition = files[file], do_filter = False, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file], is_solicit = False)
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [40,40,50,50,50,50,50,50,60,60,60,70,70,70,70,70,70,70,70]
for i in range(len(runs_alpha)):
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.029
    if i > 8: # set lower cutoff to be higher for higher ov
        info_alpha.min_alpha_value = 0.3
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[i].date.decode('utf-8')
    info_alpha.temperature = 170
    info_alpha.bias = runs_alpha[i].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = bins[i]
    info_alpha.data_type = 'h5'
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[i], baseline_correct = True, no_solicit = True, range_high = 10)
    wp.process(do_spe = False, do_alpha = True)
    j, k = wp.get_alpha()
    campaign_alpha.append(wp)
    print(info_alpha.bias)
#%%
runs_alpha[0].plot_hists('','')
campaign_alpha[0].plot_alpha_histogram(peakcolor = 'blue')
#%%
invC_alpha = invC_alpha_1us
invC_alpha_err = invC_alpha_err_1us
alpha_data = Alpha_data(campaign_alpha, invC_alpha, invC_alpha_err, spe, v_bd, v_bd_err)
alpha_data.analyze_alpha()
#%%
alpha_data.plot_alpha(color = 'purple')
alpha_data.plot_num_det_photons()
#%% values based on Wesley's APS slides
N = 5.49/(19.6E-6) 
PTE = 0.0042
alpha_data.plot_PDE(N*PTE)
