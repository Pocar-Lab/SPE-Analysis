# -*- coding: utf-8 -*-
"""
Compare baseline histrograms in various conditions

Created on Thur Feb 22 2024

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

path = 'pre-bd-data/'

csvf = open('baseline.csv', 'w')
# plot_baseline_histogram(path+'Run_1666778594.hdf5', 'Acquisition_1666781843', color='blue', condition='GN')
# plot_baseline_histogram(path+'Run_1680287728.hdf5', 'Acquisition_1680287785', color='red')
plot_baseline_histogram(path+'Run_1689175515.hdf5', 'Acquisition_1689175575', color='yellow', condition='GN')
plot_baseline_histogram(path+'Run_1690920592.hdf5', 'Acquisition_1690920842', color='green')
plot_baseline_histogram(path+'Run_1691696340.hdf5', 'Acquisition_1691696355')
plot_baseline_histogram(path+'Run_1695300627.hdf5', 'Acquisition_1695300648', color='red', condition='Vac')
# plot_baseline_histogram(path+'Run_1695564279.hdf5', 'Acquisition_1695564289', color='blue', condition='Vac')
# plot_baseline_histogram(path+'Run_1697746173.hdf5', 'Acquisition_1697746526', color='aqua', condition='Vac')
# plot_baseline_histogram(path+'Run_1697818249.hdf5', 'Acquisition_1697819075', color='pink', condition='Vac')
plot_baseline_histogram(path+'Run_1699564042.hdf5', 'Acquisition_1699564067', color='purple')
csvf.close()
plt.show()

writer = csv.writer(open("baseline.csv", 'w'))



def plot_baseline_histogram(
    baseline_file, acquisition,
    with_fit: bool = True, 
    log_scale: bool = False, 
    color: str = "orange", 
    savefig: bool = False, 
    path: Optional[str] = None,
    condition = 'LXe'
) -> None:
    run_spe_solicited = RunInfo([baseline_file],  specifyAcquisition = True, acquisition=acquisition, do_filter = True, is_solicit = True, upper_limit = .5, baseline_correct = True)
    baseline_values = np.array(
        run_spe_solicited.peak_data[run_spe_solicited.hd5_files[0]][
            run_spe_solicited.acquisition
        ]
    )
    baseline_numbins = 50
    baseline_fit = fit_baseline_gauss(baseline_values, binnum=baseline_numbins, alpha=False)
    # fig = plt.figure()
    plt.hist(
        baseline_values,
        bins=baseline_numbins,
        # label="Solicited Baseline Data",
        color="tab:purple",
        alpha=.3,
        density=True,
    )
    # if with_fit:
    sigma = f" σ = {baseline_fit['fit'].params['sigma'].value*1000:0.4} ± {baseline_fit['fit'].params['sigma'].stderr*1000:0.1} mV"
    plot_fit(
        baseline_fit,
        baseline_values,
        binnum=baseline_numbins,
        plot_hists=False,
        label=condition + ': ' + run_spe_solicited.date.split(" ")[0] + sigma,
        color=color
    )
    plt.legend(loc = 'center left')
    plt.xlabel("Waveform Amplitude [V]")
    plt.ylabel("Counts")
    if log_scale:
        plt.yscale("log")
    # textstr = f"Date: {self.info.date}\n"
    # textstr += f"Condition: {self.info.condition}\n"
    # textstr += f"Bias: {self.info.bias:0.4} [V]\n"
    # textstr += f"RTD4: {self.info.temperature} [K]\n"
    # textstr += f"--\n"
    textstr = f"""Baseline Mean: {baseline_fit['fit'].params['center'].value:0.4} +- {baseline_fit['fit'].params['center'].stderr:0.1} [V]\n"""
    textstr += f"""Baseline Sigma: {baseline_fit['fit'].params['sigma'].value:0.4} +- {baseline_fit['fit'].params['sigma'].stderr:0.1} [V]\n"""
    textstr += f"""Reduced $chi^2$: {baseline_fit['fit'].redchi:0.4}"""
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # props = dict(boxstyle="round", facecolor=color, alpha=0.5)
    # fig.text(0.15, 0.9, textstr, fontsize=10, verticalalignment="top", bbox=props)
    plt.tight_layout()
    csvf.write(f"{run_spe_solicited.date},{baseline_fit['fit'].params['sigma'].value:0.4},{baseline_fit['fit'].params['sigma'].stderr:0.1}\n")
    # writer.writerow([run_spe_solicited.date, baseline_fit['fit'].params['sigma'].value, baseline_fit['fit'].params['sigma'].stderr ])
    # if savefig:
    #     plt.savefig(path)
    #     plt.close(fig)
    # else:
    #     plt.show()

def fit_baseline_gauss( values: list[float], binnum: int = 50, alpha: bool = False) -> dict[str, type[float | Any]]:
    f_range = {}
    if alpha:  # TODO no hardcoded parameters !!
        f_range["low"] = -0.0005
        # f_range['low'] = 0.0
        f_range["high"] = 0.0045
        # f_range['high'] = 0.003
        f_range["center"] = (f_range["high"] + f_range["low"]) / 2.0
    else:
        f_range["center"] = np.mean(values)
        std_guess = np.std(values)
        f_range["low"] = f_range["center"] - 2.0 * std_guess
        f_range["high"] = f_range["center"] + 2.0 * std_guess
    bin_density = float(binnum) / (np.amax(values) - np.amin(values))
    new_binnum = int(bin_density * (f_range["high"] - f_range["low"]))
    limit_values = values[(values >= f_range["low"]) & (values <= f_range["high"])]
    curr_hist = np.histogram(limit_values, bins=new_binnum)
    # plt.hist(values, bins= binnum)
    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1]) / 2
    model = lm.models.GaussianModel()
    params = model.make_params(
        amplitude=np.amax(counts), center=np.mean(limit_values), sigma=np.std(values)
    )
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1 / counts))
    # plt.step(centers, counts, where = 'mid')
    # plt.plot(centers, res.eval(params = res.params, x = centers), '--')
    f_range["fit"] = res
    # return {'center': np.mean(values), 'low': np.amin(values), 'high': np.amax(values), 'fit': res}
    return f_range

def plot_fit(
    fit_info: Dict[str, lm.model.ModelResult],
    values: np.ndarray,
    binnum: int = 20,
    plot_hists: bool = True,
    label: str | None = None,
    color = "red"
) -> None:
    fit_data = values[(values >= fit_info["low"]) & (values <= fit_info["high"])]
    numvalues = len(fit_data)
    h = 3.49 * (numvalues) ** (-1 / 3) * np.std(fit_data)
    binnum = int(np.ceil((max(fit_data) - min(fit_data)) / h))
    if plot_hists:
        curr_hist = plt.hist(fit_data, bins=binnum)
    x = np.linspace(fit_info["low"], fit_info["high"], num=200)
    plt.plot(
        x,
        fit_info["fit"].eval(params=fit_info["fit"].params, x=x),
        color=color,
        label=label,
        density=True,
    )

