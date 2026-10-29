#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO

Created on Mon Nov 6 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

from typing import Counter
import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms
from AnalyzePDE import SPE_data
from multiprocessing import Pool
plt.style.use('misc/nexo.mplstyle')

### Low (normal) Illumination at 2.6V 10%DC

condition = 'GN'
temperature = 185
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '../data/20251106_SPE_GN_190K/2.60V/'
files, biases, run_spe_pre_bd = get_files(path)

# Run_1762458140 should be 33.5

def proc(file):
    return ProcessWaveforms([path+file], do_filter=True,
                            baseline_correct=True, poly_correct=True,
                            upper_limit=-1, prominence=.002)
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    # m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)
measurments = measurments[2:]

# Fit Gauss to peaks and find the best value for peak location
savefig = True
first_pe     = [5.6, 6.1, 6.0, 6.6, 7.4, 7.7, 8.2, 8.1, 8.1]
lower_cutoff = [4.9, 4.7, 5.0, 5.1, 4.8, 5.2, 5.3, 5.0, 5.4]
num_pes      = [3,   4,   4,   5,   5,   7,   4,   6,   6, ]
outpath = '/home/evanbruggen_umass_edu/0vbb/20251106/20251110_'
campaign_spe = []
for i in range(len(measurments) - 1):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=(lower_cutoff[i]/1000, .1),
                         # cutoff=(lower_cutoff[i]/1000, upper_cutoff[i]/1000),
                         first_pe=first_pe[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()



### High Illumination at 2.75V 90%DC

condition = 'GN'
temperature = 185
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
path = '../data/20251106_SPE_GN_190K/2.75V/'
files, biases, run_spe_pre_bd = get_files(path)

def proc(file):
    return ProcessWaveforms([path+file], do_filter=True,
                            baseline_correct=True, poly_correct=True,
                            upper_limit=-1, prominence=.002)
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
savefig = True
first_pe     = [4.5, 5.0, 5.3, 5.9, 6.1, 6.7, 7.1, 7.5, 7.8,]
lower_cutoff = [3.4, 3.2, 3.4, 3.5, 3.5, 2.5, 3.9, 4.2, 4.4,]
num_pes      = [4,   4,   5,   5,   6,   7,   7,   7,   7,  ]
outpath = '/home/evanbruggen_umass_edu/0vbb/20251106/bright/20251110_'
campaign_spe = []
for i in range(len(measurments)):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=(lower_cutoff[i]/1000, .1),
                         # cutoff=(lower_cutoff[i]/1000, upper_cutoff[i]/1000),
                         first_pe=first_pe[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()



### 0PE CA Test

import numpy as np
from scipy import signal
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from scipy.special import gamma

def get_mode(hist_data: list or np.array) -> tuple[float, float]:
    counts = hist_data[0]
    bins = hist_data[1]
    centers = (bins[1:] + bins[:-1]) / 2.0
    max_index = np.argmax(counts)
    return centers[max_index], np.amax(counts)

peak_search_params = {
    "height": 0.0,
    "threshold": None,
    "distance": None,
    "prominence": .03,
    "width": None,
    "wlen": 100,
    "rel_height": None,
    "plateau_size": None,
}

def get_spe_peak_nums(file, acquisition, plot_num=None):
    # load in waveforms
    with h5py.File(file, "r") as hdf:
        group_names = list(hdf["RunData"].keys())
        if not acquisition:
            acquisition = group_names[0]
        print(f"Reading {file} with {acquisition}")
        data: np.ndarray = hdf["RunData"][acquisition][:]
        waveforms = data[:, 1:]
        time = data[:, 0]
    num_peaks = []
    for i in range(waveforms.shape[1]):
        if i % 1000 == 0:
            print(f"Processing {i} waveforms")
        # baseline correct
        use_bins = np.linspace(-10, 10, 1000)
        curr_hist = np.histogram(waveforms[:, i], bins=use_bins)
        baseline_level, _ = get_mode(curr_hist)
        waveforms[:, i] -= baseline_level
        # filter
        sos = signal.butter(3, 4e5, btype="lowpass", fs=2502502., output="sos")
        filtered = signal.sosfilt(sos, waveforms[:, i])
        peaks, _ = signal.find_peaks(filtered, **peak_search_params)
        # # debug waveforms
        # if i % 100 == 0:
        #     print(f"{peaks=}")
        # if len(peaks) == 0:
        #     plt.plot(time,filtered)
        #     plt.show()
        num_peaks.append(len(peaks))
        # plotting
        if plot_num:
            time_us = time*1e6-200
            plt.plot(time_us,filtered)
            for peak in peaks:
                plt.plot(time_us[peaks], filtered[peaks], '.')
            if i > plot_num: # only plot number of waveforms given
                break
    return num_peaks


nums = get_spe_peak_nums(path+files[0], None)

data = Counter(nums)
occur = list(data.keys()) # [1, 2, 3, 4, 5, 7, 6]
counts = list(data.values()) # [188535, 120383, 37110, 7523, 1019, 6, 114]
counts_err = [math.sqrt(c) for c in counts]

def poisson(r, k, dt):
    return ((r*dt)**k) * math.exp(-r*dt)/gamma(k+1)
def chi_square(N, P):
    chi = 0
    for i,j in zip(N,P):
        if i > 0:
            chi += (i-j)**2/j
        else:
            chi += 1
    return chi

# Find best decay rate
dt = 400e-6 # Time in seconds used in analysis, after trigger is 200 us
# rate = np.arange(2900,3300, 1)
rate = np.arange(000,7000, 1)
# rate = np.arange(3000,3200, 1)
profile = []
r_plot = []
for r in rate:
    fit = [poisson(r,k,dt) for k in range(len(counts))]
    norm = np.sum(counts)
    pdf = [norm*f/np.sum(fit) for f in fit]
    profile.append(chi_square(counts,pdf))
    r_plot.append(r)

best_fit = rate[np.where(profile == np.amin(profile))][0]


### 0PE CA Test

import numpy as np
from scipy import signal
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from scipy.special import gamma

def get_mode(hist_data: list or np.array) -> tuple[float, float]:
    counts = hist_data[0]
    bins = hist_data[1]
    centers = (bins[1:] + bins[:-1]) / 2.0
    max_index = np.argmax(counts)
    return centers[max_index], np.amax(counts)

peak_search_params = {
    "height": 0.0,
    "threshold": None,
    "distance": None,
    "prominence": .03,
    "width": None,
    "wlen": 100,
    "rel_height": None,
    "plateau_size": None,
}

def get_spe_peak_nums(file, acquisition, plot_num=None):
    # load in waveforms
    with h5py.File(file, "r") as hdf:
        group_names = list(hdf["RunData"].keys())
        if not acquisition:
            acquisition = group_names[0]
        print(f"Reading {file} with {acquisition}")
        data: np.ndarray = hdf["RunData"][acquisition][:]
        waveforms = data[:, 1:]
        time = data[:, 0]
    num_peaks = []
    for i in range(waveforms.shape[1]):
        if i % 1000 == 0:
            print(f"Processing {i} waveforms")
        # baseline correct
        use_bins = np.linspace(-10, 10, 1000)
        curr_hist = np.histogram(waveforms[:, i], bins=use_bins)
        baseline_level, _ = get_mode(curr_hist)
        waveforms[:, i] -= baseline_level
        # filter
        sos = signal.butter(3, 4e5, btype="lowpass", fs=2502502., output="sos")
        filtered = signal.sosfilt(sos, waveforms[:, i])
        peaks, _ = signal.find_peaks(filtered, **peak_search_params)
        # # debug waveforms
        # if i % 100 == 0:
        #     print(f"{peaks=}")
        # if len(peaks) == 0:
        #     plt.plot(time,filtered)
        #     plt.show()
        num_peaks.append(len(peaks))
        # plotting
        if plot_num:
            time_us = time*1e6-200
            plt.plot(time_us,filtered)
            for peak in peaks:
                plt.plot(time_us[peaks], filtered[peaks], '.')
            if i > plot_num: # only plot number of waveforms given
                break
    return num_peaks


nums = get_spe_peak_nums(path+files[0], None)

data = Counter(nums)
occur = list(data.keys()) # [1, 2, 3, 4, 5, 7, 6]
counts = list(data.values()) # [188535, 120383, 37110, 7523, 1019, 6, 114]
counts_err = [math.sqrt(c) for c in counts]

def poisson(r, k, dt):
    return ((r*dt)**k) * math.exp(-r*dt)/gamma(k+1)
def chi_square(N, P):
    chi = 0
    for i,j in zip(N,P):
        if i > 0:
            chi += (i-j)**2/j
        else:
            chi += 1
    return chi

# Find best decay rate
dt = 400e-6 # Time in seconds used in analysis, after trigger is 200 us
# rate = np.arange(2900,3300, 1)
rate = np.arange(000,7000, 1)
# rate = np.arange(3000,3200, 1)
profile = []
r_plot = []
for r in rate:
    fit = [poisson(r,k,dt) for k in range(len(counts))]
    norm = np.sum(counts)
    pdf = [norm*f/np.sum(fit) for f in fit]
    profile.append(chi_square(counts,pdf))
    r_plot.append(r)

best_fit = rate[np.where(profile == np.amin(profile))][0]
