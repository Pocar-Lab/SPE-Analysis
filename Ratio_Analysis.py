# -*- coding: utf-8 -*-
"""
Created on Fri Dec 6 2024

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

%load_ext autoreload
%autoreload 2
%autoindent
import sys
from MeasurementInfo import MeasurementInfo
import numpy as np
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
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy
from AnalyzeResults import AnalyzeResults
plt.style.use('misc/nexo.mplstyle')

invC_alpha = 1143.98e-6
invC_alpha_err = 0.12e-6
invC_spe = 11404e-6
invC_spe_err = 11e-6

v_bd = 27.08 # from Oct 17 2024 SPE
v_bd_err = 0.098

teflon = AnalyzeResults('results/20230608_Alpha.csv', 'results/LXe_June8_2023_vbd_bias.csv',
               'results/July13_171K_CA_GN.csv', invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
               27.13, .0342)

cu = AnalyzeResults('results/2023June28_Alpha.csv', 'results/LXe_June8_2023_vbd_bias.csv',
               'results/July13_171K_CA_GN.csv', invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
               27., .0342, ov_max=5)
plt.axes().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.5))
cu.fit_alpha(cu.ov, fit='exp1')
plt.legend()
plt.show()

short_si = AnalyzeResults('2023August01_Alpha.csv', 'results/August1_silicon_vbd.csv',
               'results/July13_171K_CA_GN.csv', invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
               27.23, .04)

teflon.plot_ratio(short_si)

no_short_si = AnalyzeResults('results/2023August10_Alpha.csv', 'results/20230810_SPE_no_si.csv',
               'results/July13_171K_CA_GN.csv', invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
               27., .0488, ov_max=8, ov_min=3)
# no_short_si.plot_num_det_photons()
plt.axes().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.5))
no_short_si.fit_alpha(no_short_si.ov, fit='exp1')
plt.legend()
plt.show()

tall_si_pre = AnalyzeResults('2024June20_Alpha.csv', 'results/20241017_LXe_SPE_vbd_bias.csv',
               'results/July13_171K_CA_GN.csv', invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
               v_bd, v_bd_err, ov_max=6, ov_min=2.5)

# tall_si_pre.plot_num_det_photons()
plt.axes().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.5))
tall_si_pre.fit_alpha(tall_si_pre.ov, fit='exp1')
plt.legend()
plt.show()

tall_si_atm = AnalyzeResults('2024July09_Alpha.csv', 'results/20241017_LXe_SPE_vbd_bias.csv',
               'results/July13_171K_CA_GN.csv', invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
               v_bd, v_bd_err, ov_max=8)
# tall_si_atm.plot_num_det_photons()

no_tall_si_pre = AnalyzeResults('2024Aug07_Alpha.csv', 'results/20241017_LXe_SPE_vbd_bias.csv',
               'results/July13_171K_CA_GN.csv', invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
               v_bd, v_bd_err, ov_max=8)
# no_tall_si.plot_num_det_photons()

no_tall_si = AnalyzeResults('2024Oct17_Alpha.csv', 'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv', invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
                            v_bd, v_bd_err, ov_max=6)
# no_tall_si.plot_num_det_photons()
# plt.axes().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.5))
# no_tall_si.fit_alpha(tall_si_pre.ov, fit='fullquad')
# plt.legend()
# plt.show()



tall_si_atm.plot_ratio(no_tall_si, fit='exp1')

tall_si_atm.plot_ratio(no_tall_si_pre)

tall_si_pre.plot_ratio(tall_si_atm)

tall_si_pre.plot_ratio(no_tall_si, fit='xexp', alpha_ylim=(0,.5), ratio_ylim=(.3,1.8))

tall_si_atm.plot_ratio(no_tall_si, fit='exp', alpha_ylim=(0,.5), ratio_ylim=(.3,1.8))
