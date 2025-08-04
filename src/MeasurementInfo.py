# -*- coding: utf-8 -*-
"""
Created on Jul 26 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from uncertainties import ufloat

def avg_amp(peaks):
    avg_amp = np.mean(peaks)
    avg_amp_err = avg_amp * sem(peaks) / np.mean(peaks)
    return ufloat(avg_amp, avg_amp_err)

class MeasurementInfo:
    """Collect all the relevant measurement information for each data set"""
    # TODO: support alpha analysis
    def __init__(self, condition, temperature, run, run_pre_bd=None):
        self.condition = condition
        self.temperature = temperature
        self.date = run.date
        self.bias = run.bias

        self.prominence = run.prominence
        self.upperlim   = run.upper_limit

        self.all_peaks  = run.all_peak_data
        self.led_peaks  = run.all_led_peak_data
        self.dark_peaks = run.all_dark_peak_data
        self.peaks = {
            'all': run.all_peak_data,
            'led': run.all_led_peak_data,
            'dark': run.all_dark_peak_data,
        }

        self.avg_amp = {
            'all': avg_amp(self.all_peaks),
            'dark': avg_amp(self.dark_peaks),
            'led': avg_amp(self.led_peaks),
        }

        if run_pre_bd:
            self.baseline_mean = run_pre_bd.baseline_mean
            self.baseline_err  = run_pre_bd.baseline_err
            self.baseline_std  = run_pre_bd.baseline_std
        else: # If no pre-breakdown data collected, calculate baseline from SPE data
            self.baseline_mean = np.mean(run.baseline_levels)
            self.baseline_err  = sem(run.baseline_levels)
            self.baseline_std  = 0.25 * np.std(run.baseline_levels) # TODO why 1/4 ??
            # self.baseline_rms = run_info_self.baseline_mode_rms

    def plot_histogram(self, bins=1000) -> None:
        """Plot preliminary histograms of file. Used to find initial guess of first PE value"""
        plt.hist(self.all_peaks, bins=bins, histtype="step", density=False)
        plt.xlabel("Amplitude (V)", loc='right')
        plt.ylabel("Frequency", loc='top')
        plt.title(f"{self.date}, {self.condition}, {self.temperature} K, {self.bias} V")
        plt.subplots_adjust(top=0.9)
        plt.yscale("log")
        plt.tight_layout()
        plt.show()
