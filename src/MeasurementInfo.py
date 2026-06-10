# -*- coding: utf-8 -*-
"""
Created on Jul 26 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from uncertainties import ufloat
import os
import h5py
from src.ProcessWaveforms import ProcessWaveforms

def get_files(path: str):
    files = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]
    biases = []
    # Sort files by bias voltage
    for f in files:
        with h5py.File(path+f, 'r') as hdf: # Find bias for each file
            biases.append(hdf['RunData'][next(iter(hdf['RunData'].keys()))].attrs['Bias(V)'])
    biases, files = zip(*sorted(zip(biases, files))) # Sort based on biases

    # Extract pre-breakdown baseline if it was collected
    run_spe_pre_bd = None
    if biases[0] < 25:
        run_spe_pre_bd = ProcessWaveforms([path+files[0]], do_filter=True,
                                          is_pre_bd=True, upper_limit=-1, baseline_correct=True)
        files = files[1:]
        biases = biases[1:]

    return files, biases, run_spe_pre_bd

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

        # self.peaks = run.peaks
        self.peaks = {
            'all':  run.peaks['all'],
            'dark': run.peaks['dark'],
            'led':  run.peaks['led'],
        }

        self.avg_amp = {
            'all':  avg_amp(self.peaks['all']),
            'dark': avg_amp(self.peaks['dark']),
            'led':  avg_amp(self.peaks['led']),
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

    def plot_histogram(self, zoom=False, bins=1000) -> None:
        """Plot preliminary histograms of file. Used to find initial guess of first PE value"""
        plt.hist(self.peaks['all'], bins=bins, histtype="step", density=False)
        plt.xlabel("Amplitude (V)", loc='right')
        plt.ylabel("Frequency", loc='top')
        plt.title(f"{self.date}, {self.condition}, {self.temperature} K, {self.bias} V")
        plt.subplots_adjust(top=0.9)
        if zoom:
            plt.xlim(0,.02)
        plt.yscale("log")
        plt.tight_layout()
        plt.show()

    def plot_led_dark_hists(self, voltage):
        (n, b, p) = plt.hist(self.peaks['all'], bins=2000, histtype='step', density=False, label='All ')
        (n1, b1, p1) = plt.hist(self.peaks['led'], bins=b, histtype='step', density=False, label='LED on')
        (n2, b2, p2) = plt.hist(self.peaks['dark'], bins=b, histtype='step', density=False, label='LED off')
        plt.legend()
        font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 14,}
        plt.xlabel('Amplitude (V)', fontdict=font)
        plt.ylabel('Frequency', fontdict=font)

        dark_count = float(len(self.peaks['dark']))
        led_count = float(len(self.peaks['led'])) - dark_count
        ratio = led_count / dark_count

        txtstr = f'LED Voltage: {voltage}\n'
        txtstr += f'Ratio: {round(ratio,2)}\n'
        txtstr += f'Total Avg Amp: {self.avg_amp['all']:.3}\n'
        txtstr += f'LED   Avg Amp: {self.avg_amp['led']:.3}\n'
        txtstr += f'Dark  Avg Amp: {self.avg_amp['dark']:.3}\n'
        plt.annotate(txtstr, xy=(0.60, 0.60), xycoords='axes fraction', size=12)
        plt.subplots_adjust(top=0.9)
        plt.yscale('log')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        return (ratio, self.avg_amp)

    def get_mode(self, peaks='all', bins=1000):
        counts, edges = np.histogram(self.peaks[peaks], bins=bins)
        mode = edges[max(counts)]
        return mode
