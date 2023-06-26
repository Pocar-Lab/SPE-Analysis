# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:13:35 2023

@author: heps1
"""
import sys
import numpy as np
# append necessary file paths, and change E -> D or vice versa
sys.path.append('E:/Xe/AnalysisScripts')
sys.path.append('E:/Xe/AnalysisScripts/August 2022 Low Illumination Campaign')   
from MeasurementInfo import MeasurementInfo
from RunInfo import RunInfo
import heapq
from scipy import signal
from scipy.optimize import curve_fit
import AnalyzePDE
from AnalyzePDE import SPE_data
import matplotlib.pyplot as plt
import matplotlib as mpl
import ProcessWaveforms_MultiGaussian
from ProcessWaveforms_MultiGaussian import WaveformProcessor as WaveformProcessor

#%% 167K 
# loop quickly loads in files
run_spe_solicited = RunInfo(['E:/Xe/DAQ/Run_1666778594.hdf5'],  specifyAcquisition = True, acquisition ='Acquisition_1666781843', do_filter = True, is_solicit = True, upper_limit = .5, baseline_correct = True)
files = ['Acquisition_1666778815','Acquisition_1666779156','Acquisition_1666782030','Acquisition_1666780808','Acquisition_1666779491']
proms = [0.033,0.038,0.04,0.054,0.068]
upperlim = [1.9, 1.9, 4.9, 4.9, 4.9]
runs = []
for file in range(len(files)):
    run_spe = RunInfo(['E:/Xe/DAQ/Run_1666778594.hdf5'], specifyAcquisition = True, acquisition = files[file], do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file])
    runs.append(run_spe)
biases = [run.bias for run in runs]
#%% gain calibration conversion factor (change as needed; make sure it is the correct factor for the data set)
#100ns, old shaper, 100x gain
invC_spe_filter = 0.19261918346201742 
invC_spe_err_filter = 0.0021831140214106596

#%% take a look at waveforms (solicited)
RunInfo(['E:/Xe/DAQ/Run_1666778594.hdf5'],  specifyAcquisition = True, acquisition ='Acquisition_1666781843', do_filter = False, is_solicit = True, upper_limit = .5, baseline_correct = True, plot_waveforms = True, fourier = False)
#%% take a look at finger plots
for run in runs:
    run.plot_hists('167', '.')
#%% set the initial guesses and fitting range
a_guess = 0.01674
b_guess = -0.4598
range_lows = [0.035, 0.05, 0.06, 0.07, 0.086]
centers = [biases[i]*a_guess+b_guess for i in range(len(runs))] #guess position of 1 p.e. for each bias
range_highs = [range_lows[i] + centers[i]*4 for i in range(len(runs))]
#%% plot with peak fit - test
n= 4
T = 167
con = 'GN'
info_spe = MeasurementInfo()
info_spe.condition = con
info_spe.date = runs[n].date.decode('utf-8')
info_spe.temperature = T
info_spe.bias = biases[n]
info_spe.baseline_numbins = 50
info_spe.peaks_numbins = 150
info_spe.data_type = 'h5'
wp = WaveformProcessor(info_spe, run_info_self = runs[n], run_info_solicit = run_spe_solicited, baseline_correct = True,  range_low = range_lows[n], range_high = range_highs[n], center = centers[n])
wp.process(do_spe = True, do_alpha = False)
wp.plot_both_histograms()
wp.plot_peak_histograms()
wp.plot_spe()

#%% make a list of ProcessWaveforms objects
campaign_spe = []
for i in range(len(runs)):
    if i == 2: #skipping this bias for now
        continue
    T = 167
    con = 'GN'
    info_spe = MeasurementInfo()
    info_spe.condition = con
    info_spe.date = runs[i].date.decode('utf-8')
    info_spe.temperature = T
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 20
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, run_info_self = runs[i], run_info_solicit = run_spe_solicited, baseline_correct = True, range_low = range_lows[i], range_high = range_highs[i], center = centers[i])
    wp_spe.process(do_spe = True, do_alpha = False)
    campaign_spe.append(wp_spe)
    
#%% check plots
for i in range(len(campaign_spe)):
    campaign_spe[i].plot_peak_histograms()

#%% check CA
for i in range(len(campaign_spe)):
    print('bias: ' + str(biases[i]) + '; ' + str(campaign_spe[i].get_CA()))

#%% plot linear fit to the breakdown voltage
    
curr_campaign = campaign_spe

filtered_spe = SPE_data(curr_campaign, invC_spe_filter, invC_spe_err_filter, filtered = True)
filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = None)

print(filtered_spe.v_bd)
print(filtered_spe.v_bd_err)

#%% plot in units of absolute gain vs OV

filtered_spe.plot_spe(in_ov = True, absolute = True, out_file = None)

#%% plot CA 
path = 'YOUR/PATH/HERE.csv'
filtered_spe.plot_CA(out_file=path) #saves values to spreadsheet, or leave blank
