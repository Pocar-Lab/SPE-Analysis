#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intermediate Temperature GN Campaign
SPE analysis for data collected on Aug 6 2025 at 235K

Created on Fri Aug 8 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
from AnalyzePDE import SPE_data
import h5py
import os
from multiprocessing import Pool
plt.style.use('misc/nexo.mplstyle')


condition = 'GN'
temperature = 235


#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098

path ='data/20250806_SPE_GN/LED_Calibration/'
led_voltages = [2.50, 2.52, 2.54,
                2.56, 2.58, 2.60,
                2.62, 2.64, 2.66]
files = ['Run_1754500093.hdf5', 'Run_1754503380.hdf5', 'Run_1754504140.hdf5',
         'Run_1754504691.hdf5', 'Run_1754505241.hdf5', 'Run_1754505895.hdf5',
         'Run_1754506489.hdf5', 'Run_1754507113.hdf5', 'Run_1754507878.hdf5',
         'Run_1754508718.hdf5']

# loop over all files
# NOTE Modify prominence and upper limit for your data:
proms = [.005 for _ in files]
upperlim = .4

runs = []

for file in range(9,len(files)):
    runs += [ProcessWaveforms([path+files[file]], do_filter=True,
                              baseline_correct=True, poly_correct=True,
                              upper_limit=upperlim, prominence=proms[0])]

measurments = []
led_ratios = []
avg_amps = []

for run, voltage in zip(runs, led_voltages):
    m = MeasurementInfo(condition, temperature, run)
    (r, a) = m.plot_led_dark_hists(voltage)
    led_ratios.append(r)
    avg_amps.append(a)
    # m.plot_histogram() # Plot preliminary histogram
    # measurments.append(m)

plt.errorbar(led_voltages, [v['all'].n for v in avg_amps],  fmt='o', yerr=[v['all'].s for v in avg_amps], label='All peaks')
plt.errorbar(led_voltages, [v['led'].n for v in avg_amps],  fmt='o', yerr=[v['led'].s for v in avg_amps], label='LED peaks')
plt.errorbar(led_voltages, [v['dark'].n for v in avg_amps], fmt='o', yerr=[v['dark'].s for v in avg_amps], label='Dark peaks')
plt.xlabel('LED Voltage [V]', loc='right')
plt.ylabel('Average Amplitude [V]', loc='top')
plt.xlim(2.47, 2.7)
# plt.xlim(2.5, 2.65)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()





#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
# change path name to folder with data files
path = 'data/20250806_SPE_GN/'
files = ['Run_1754516545.hdf5', 'Run_1754515972.hdf5', 'Run_1754515431.hdf5',
         'Run_1754514829.hdf5', 'Run_1754514219.hdf5', 'Run_1754513577.hdf5',
         'Run_1754512886.hdf5', 'Run_1754511706.hdf5', 'Run_1754510958.hdf5', ]

# Sort files by bias voltage
biases = []
for f in files:
    with h5py.File(path+f, 'r') as hdf:
        biases.append(hdf['RunData'][next(iter(hdf['RunData'].keys()))].attrs['Bias(V)'])
_, files = zip(*sorted(zip(biases, files)))

condition = 'GN'
temperature = 235

# file_pre_bd = 'Run_1751987061.hdf5'
# run_spe_pre_bd = ProcessWaveforms([path+file_pre_bd], do_filter=True,
#                                   is_pre_bd=True, upper_limit=0.063, baseline_correct=True)


# loop over all files
runs = []
proms = [0.0050, 0.0050, 0.0050,
         0.0050, 0.0050, 0.0050,
         0.0050, 0.0050, 0.0050]
upperlim = -1
for file in range(len(files)):
    run = ProcessWaveforms([path+files[file]], do_filter=True,
                           baseline_correct=True, poly_correct=True,
                           upper_limit=upperlim, prominence=proms[file])
    runs.append(run)

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run)
    # Plot preliminary histogram
    m.plot_histogram()
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
pe_center_guesses = [4.8, 5.1, 5.7, 6.2, 6.6, 6.9, 7.2, 7.4, 7.9]
num_pes = [3, 6, 5, 5, 6, 7, 6, 7, 7]
centers_guesses = []
for pe_guess, num_pe in zip(pe_center_guesses, num_pes):
    centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
outpath = '/home/ed/pictures/0vbb/20250806/20250808_'
campaign_spe = []
cutoff = (.004,.07)
cutoff = (.002,.05)
for i in range(0,9):
    cutoff = (.0024, .09) if measurments[i].bias > 35 else (.0030, .04)
    if measurments[i].bias > 37:
        cutoff = (.0060, .17)
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=cutoff, centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=True,path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(savefig=True, fit_origin=True,path=outpath+f"spe_{measurments[i].bias}.png")
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)


spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()


### no LED

condition = 'GN'
temperature = 235
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '20250822_SPE_GN_235K/LED_Calibration/LED_Off/'
files = os.listdir(path)

# Sort files by bias voltage
biases = []
for f in files:
    with h5py.File(path+f, 'r') as hdf: # Find bias for each file
        biases.append(hdf['RunData'][next(iter(hdf['RunData'].keys()))].attrs['Bias(V)'])
biases, files = zip(*sorted(zip(biases, files))) # Sort based on biases

# Extract pre-breakdown baseline if it was collected
if biases[0] < 25:
    file_pre_bd = files[0]
    run_spe_pre_bd = ProcessWaveforms([path+file_pre_bd], do_filter=True,
                                      is_pre_bd=True, upper_limit=0.063, baseline_correct=True)
    files = files[1:]
    biases = biases[1:]
else:
    file_pre_bd = None

# Loop over all files and find peaks
runs = []
proms = [0.0050, 0.0050, 0.0050,
         0.0050, 0.0050, 0.0050,
         0.0050, 0.0050, 0.0050]
prom = .002
upperlim = -1
for file in range(len(files)):
    run = ProcessWaveforms([path+files[file]], do_filter=True,
                           baseline_correct=True, poly_correct=True,
                           upper_limit=upperlim, prominence=prom)
    runs.append(run)

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

with open('20250822_measurments.pkl', 'wb') as file:
    pickle.dump(measurments, file)

# Fit Gauss to peaks and find the best value for peak location
pe_center    = [4.9, 5.1, 5.7, 6.1, 6.8, 7.3, 7.7, 8.2, 8.5]
num_pes      = [4,   4,   5,   5,   5,   7,   7,   8,   8]
lower_cutoff = [3.8, 4.4, 4.7, 4.9, 4.6, 4.8, 4.8, 4.8, 4.8]
savefig = True
centers_guesses = []
for pe_guess, num_pe in zip(pe_center, num_pes):
    centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
outpath = '/home/ed/pictures/0vbb/20250822/20250825_'
campaign_spe = []
for i in range(len(measurments)):
    cutoff = (lower_cutoff[i]/1000, .1)
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=cutoff, centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)


### New analysis method

condition = 'GN'
temperature = 235
#1us, 10x gain, filter on - values from Noahs calibration report
# invC_filter = 0.011547
# invC_err_filter = 0.000098
invC_filter = 0.010040
invC_err_filter = 0.000500
path = '../data/20250806_SPE_GN_235K/'
files, biases, run_spe_pre_bd = get_files(path)

def proc(file):
    return ProcessWaveforms([path+file], do_filter=True,
                            baseline_correct=True, poly_correct=True,
                            upper_limit=-1, prominence=.003)
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    # m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
savefig = False
pe_center    = [4.9, 5.3, 5.8, 6.2, 6.5, 7.0, 7.4, 7.8, 8.1]
num_pes      = [4,   4,   5,   5,   5,   7,   7,   8,   8]
lower_cutoff = [3.9, 3.9, 3.7, 3.9, 4.1, 4.2, 4.4, 4.3, 4.3]
outpath = '/home/evanbruggen_umass_edu/0vbb/20250806/20250917_'
campaign_spe = []
for i in range(3,4):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=(lower_cutoff[i]/1000, .1),
                         first_pe=pe_center[i],
                         # centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    # wp_spe.plot_spe(fit_origin=False,
    #                 savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    # wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)

spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()
