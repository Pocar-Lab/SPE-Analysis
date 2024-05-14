# -*- coding: utf-8 -*-
"""
Count number of alpha peaks to calculate alpha rate

Created on Tue Apr 30 2024

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
from lmfit.models import LinearModel, GaussianModel, ExponentialModel
from typing import Any, Dict, List, Tuple, Optional
import lmfit as lm
import h5py
from collections import Counter
import os

plt.style.use('misc/nexo.mplstyle')
path = 'pre-bd-data/'

# count peaks over all bias voltages
num_peaks = []
directory = os.fsencode('march-28-2024/')
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".hdf5"):
        num_peaks += get_peak_nums(directory + file, None)

# count peaks of single bias voltage
num_peaks = get_peak_nums('march-28-2024/Run_1711658600.hdf5', None)



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
    "prominence": .09,
    "width": None,
    "wlen": 100,
    "rel_height": None,
    "plateau_size": None,
}

def get_peak_nums(file, acquisition):
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
        peaks, props = signal.find_peaks(filtered, **peak_search_params)
        # # debug waveforms
        # if i % 100 == 0:
        #     print(f"{peaks=}")
        # if len(peaks) == 0:
        #     plt.plot(time,filtered)
        #     plt.show()
        num = 0 # count peaks which are in first half of window before self trigged pulse
        for peak in peaks:
            if time[peak] > 200e-6:
                num += 1
        num_peaks.append(num)
        # # plotting
        # if len(peaks) > 0:  # only plot peaks
        #     plt.plot(time,filtered)
        #     for peak in peaks:
        #         plt.plot(time[peaks], filtered[peaks], '.')
        # if i > 1000: # only plot first 1000 waveforms
        #     break
    return num_peaks
