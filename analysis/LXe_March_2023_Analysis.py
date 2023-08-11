# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 14:38:28 2023

@author: Jaime
"""
import os
os.chdir('C:/Users/Jaime/Desktop/Analysis/SPE-Analysis')
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

#%% SPE-VBD
#CR-Z Preamp, Shaper "0", 100x gain, filtered (change as needed; make sure it is the correct factor for the data set)
invC_spe_filter =  0.011959447603692185
invC_spe_err_filter = 3.881945391072933E-05
# loop quickly loads in files
# separate files for each bias voltage -> specifyAcquisition = False
files = ['Run_1680287257',
         'Run_1680286846',
         'Run_1680285923',
         'Run_1680285397',
         'Run_1680284969',
         'Run_1680282125']
proms = [0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025,0.025]
upperlim = [1, 1, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8]
runs_spe = []
for file in range(len(files)):
    run_spe = RunInfo(['C:/Users/Jaime/Desktop/Analysis/SPE-Analysis/'+files[file]+'.hdf5'], specifyAcquisition = False, do_filter = True, upper_limit = upperlim[file], baseline_correct = True, prominence = proms[file])
    runs_spe.append(run_spe)
biases = [run.bias for run in runs_spe]

#%%
cutoffs = []
centers_list=[]
centers_guesses = []

for run in runs_spe:
    bins = int(round(np.sqrt(len(run.all_peak_data))))
    count, edges = np.histogram(run.all_peak_data, bins=bins)
    centers = (edges[:-1] + edges[1:])/2
    
    peaks, props = signal.find_peaks(count, prominence=25, distance=5)
    
    fitrange = ((centers[peaks[3]] - centers[peaks[1]])/2)
    range_low =  centers[peaks[1]]- 0.28*fitrange
    range_high = centers[peaks[4]]+ 0.35*fitrange
    
    cutoffs.append((range_low, range_high))
    centers_list.append(centers[peaks[1]])
    peaks = peaks[1:]
    centers_guesses.append([centers[peak] for peak in peaks])
    
    plt.figure()
    plt.hist(run.all_peak_data, bins=bins)
    for peak in peaks:
        plt.scatter(centers[peak],count[peak])
    plt.axvline(range_low, c='red')
    plt.axvline(range_high, c='black')
    plt.xlim([0,1])
    plt.yscale('log')


#%% compute baseline info

run_baseline = RunInfo(['C:/Users/Jaime/Desktop/Analysis/SPE-Analysis/Run_1680287728.hdf5'],
                       is_solicit=True, do_filter=True, baseline_correct=True)
info_solicited = MeasurementInfo()
info_solicited.condition = 'LXe'
info_solicited.date = run_baseline.date.decode()
info_solicited.temperature = 170
info_solicited.baseline_numbins=300
info_solicited.date_type = 'h5'


#%% testing cell - plot with peak fit
# set conditions, temp
for i in range(len(biases)):
    T = 170.5
    con = 'LXe'
    info_spe = MeasurementInfo()
    info_spe.condition = con
    info_spe.date = runs_spe[i].date.decode('utf-8')
    info_spe.temperature = T
    info_spe.bias = biases[i]
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 150
    info_spe.data_type = 'h5'
    wp = WaveformProcessor(info_spe, centers = centers_guesses[i],
                           run_info_self = runs_spe[i],
                           run_info_solicit = run_baseline,
                           baseline_correct = True, cutoff = cutoffs[i])
    
    wp.process(do_spe = True, do_alpha = False)
    wp.plot_peak_histograms()
    # plt.savefig(f'hist_Run_{files[i]}_4')
    wp.plot_spe()
    # plt.savefig(f'SPE_Run_{files[i]}_4')

#%% make a list of ProcessWaveforms objects
campaign_spe = []
for i in range(len(runs_spe)):
    info_spe = MeasurementInfo()
    info_spe.condition = 'LXe'
    info_spe.date = runs_spe[i].date.decode('utf-8')
    info_spe.temperature = 170
    info_spe.bias = runs_spe[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 150
    info_spe.data_type = 'h5'
    wp_spe = WaveformProcessor(info_spe, centers = centers_guesses[i],
                               run_info_self = runs_spe[i],
                               run_info_solicit = run_baseline,
                               baseline_correct = True, cutoff = cutoffs[i])
    wp_spe.process(do_spe = True, do_alpha = False)
    campaign_spe.append(wp_spe)

#%%



#%%
curr_campaign = campaign_spe
filtered_spe = SPE_data(curr_campaign, invC_spe_filter, invC_spe_err_filter, filtered = True)
filtered_spe.plot_spe(in_ov = False, absolute = False, out_file = None)

print(filtered_spe.v_bd)
print(filtered_spe.v_bd_err)

#%% plot in units of absolute gain vs OV
filtered_spe.plot_spe(in_ov = True, absolute = True, out_file = None)
#%% record BD voltage
v_bd = 27.29
v_bd_err = 0.0579
spe = SPE_data(campaign_spe, invC_spe_filter, invC_spe_err_filter, filtered = True)

#%%
path = 'C:/Users/Jaime/Desktop/Analysis/SPE-Analysis/Run_1680279197.hdf5'
with h5py.File(path, 'r') as hdf:
    keys = list(hdf['RunData'].keys())

i=0
key = keys[i]
proms = [0.08,0.08,0.08, 0.09,0.05,0.2, 0.4,0.35,0.25, 0.25,0.5,0.8,0.2]
run_alpha = RunInfo(file, do_filter=False, upper_limit=5, 
                        baseline_correct=True, prominence=0.01,
                        specifyAcquisition=True, acquisition=key,
                        plot_waveforms=True)
#%% Alpha Data Time
runs_alpha = []
path = 'C:/Users/Jaime/Desktop/Analysis/SPE-Analysis/Run_1680279197.hdf5'
with h5py.File(path, 'r') as hdf:
    keys = list(hdf['RunData'].keys())

file = [path]
for i,key in enumerate(keys):
    
    run_alpha = RunInfo(file, do_filter=False, upper_limit=5, 
                        baseline_correct=True, prominence=0.01,
                        specifyAcquisition=True, acquisition=key)
    run_alpha.plot_hists('0','0')
    biases = [run.bias for run in runs_alpha]
    runs_alpha.append(run_alpha)


#%%

campaign_alpha = []
min_alphas = [1,0.3,0.3,0,0,0,0,0]
for i in range(len(runs_alpha)):
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = min_alphas[i]
    info_alpha.condition = 'LXe'
    info_alpha.date = str(runs_alpha[i].date)
    info_alpha.temperature = 170
    info_alpha.bias = runs_alpha[i].bias
    info_alpha.baseline_numbins = 40
    bins = int(round(np.sqrt(len(runs_alpha[i].all_peak_data))))
    info_alpha.peaks_numbins = bins
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[i],
                           run_info_solicit = run_baseline,
                           no_solicit=True,
                           baseline_correct = True)
    wp.process(do_spe = False, do_alpha = True)
    wp.plot_alpha_histogram(peakcolor = 'blue')
    campaign_alpha.append(wp)

#%%

# invC_spe_filter =  0.011959447603692185
# invC_spe_err_filter = 3.881945391072933E-05
# spedata = SPE_data(campaign_spe, invC_spe_filter, invC_spe_err_filter, filtered=True)

#%%
#GET NEW SPEDATA FROM GAS OR VACUUM
invC_alpha = 0.011959447603692185
invC_alpha_err = 3.881945391072933E-05
v_bd = spedata.v_bd
v_bd_err = spedata.v_bd_err
alphadata = Alpha_data(campaign_alpha, invC_alpha, invC_alpha_err, spedata, v_bd, v_bd_err)
