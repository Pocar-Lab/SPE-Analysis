# -*- coding: utf-8 -*-
"""
Created on Fri Dec 6 2024

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

%load_ext autoreload
%autoreload 2
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

## teflon
teflon.plot_ratio(si_short)

cu_none = AnalyzeResults('Copper Spacing', '2023July13_Alpha.csv',
                         'results/August1_silicon_vbd.csv', 'results/July13_171K_CA_GN.csv',
                         invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27.22, .0686,
                         ov_max=6, ov_min=2.5)

si_short = AnalyzeResults('Si Short Reflectors', '2023August01_Alpha.csv',
                          'results/August1_silicon_vbd.csv', 'results/July13_171K_CA_GN.csv',
                          invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27.23, .04,
                          ov_max=6, ov_min=2.5)

si_short_none = AnalyzeResults('Si Short Spacing', 'results/2023August10_Alpha.csv',
                               'results/20230810_SPE_no_si.csv', 'results/July13_171K_CA_GN.csv',
                               invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27., .0488,
                               ov_max=6, ov_min=3)

si_tall_pre = AnalyzeResults('Si Tall Reflectors, Pre-baking', '2024June20_Alpha.csv',
                             'results/20241017_LXe_SPE_vbd_bias.csv',
                             'results/July13_171K_CA_GN.csv',
                             invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, v_bd, v_bd_err,
                             ov_max=6, ov_min=2.5)

si_tall_baked = AnalyzeResults('Si Tall Reflectors, Post-baking', '2024June27_Alpha.csv',
                               'results/20241017_LXe_SPE_vbd_bias.csv',
                               'results/July13_171K_CA_GN.csv',
                               invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, v_bd, v_bd_err,
                               ov_max=6, ov_min=2.5)

si_tall_atm = AnalyzeResults('Si Tall Reflectors, Post-Atmosphere', '2024July9_Alpha.csv',
                             'results/20241017_LXe_SPE_vbd_bias.csv',
                             'results/July13_171K_CA_GN.csv',
                             invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, v_bd, v_bd_err,
                             ov_max=6)

si_tall_none_gnd = AnalyzeResults('Si Tall Spacing, GND Loop', '2024Aug07_Alpha.csv',
                                'results/20241017_LXe_SPE_vbd_bias.csv',
                                'results/July13_171K_CA_GN.csv',
                                invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, v_bd, v_bd_err,
                                ov_max=6)

si_tall_none = AnalyzeResults('Si Tall Spacing', '2024Oct17_Alpha.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, v_bd, v_bd_err,
                            ov_max=6)


## No reflector
si_short_none.plot_ratio(si_tall_none, fit='xexp', alpha_ylim=(0,.5), ratio_ylim=(0,2.5))

cu_none.plot_ratio(si_tall_none, fit='xexp', alpha_ylim=(0,1.5), ratio_ylim=(0,10))

cu_none.plot_ratio(si_short_none, fit='xexp', alpha_ylim=(0,2), ratio_ylim=(0,5))

## Baking
si_tall_atm.plot_ratio(si_tall_pre, fit='exp1', alpha_ylim=(0,.7), ratio_ylim=(0,3.5))

si_tall_baked.plot_ratio(si_tall_pre, fit='exp1', alpha_ylim=(0,.5), ratio_ylim=(0,2.5))

si_tall_atm.plot_ratio(si_tall_baked, fit='exp1', alpha_ylim=(0,.5), ratio_ylim=(0,1.8))

## Si Short (8-inner-outer)
si_short.plot_ratio(si_short_none, fit='exp1', alpha_ylim=(0,.8), ratio_ylim=(0,3.5))

PTEs_si_short = ufloat(0.0030553, 5.5e-06)
PTEd_si_short = ufloat(0.0019055, 4.4e-06)
PTEn_si_short = ufloat(0.0013598, 3.7e-6)
S = PTEs_si_short/PTEn_si_short
D = PTEd_si_short/PTEn_si_short
s = (ufloat(2.4645, .0028) - D)/(S-D)
ufloat(2.4645, .0028) - S

## Si Tall (4-inner)
si_tall_pre.plot_ratio(si_tall_none, fit='exp', alpha_ylim=(0,.5), ratio_ylim=(0,2))

si_tall_baked.plot_ratio(si_tall_none, fit='exp1', alpha_ylim=(0,.8), ratio_ylim=(0,3.5))

si_tall_atm.plot_ratio(si_tall_none, fit='exp1', alpha_ylim=(0,1), ratio_ylim=(0,5))

PTEs_tall_si = ufloat(0.0025563, 5.1e-06)
PTEd_tall_si = ufloat(0.0016941, 4.1e-06)
PTEn_tall_si = ufloat(0.0012682, 3.6e-06)
S = PTEs_tall_si/PTEn_tall_si
D = PTEd_tall_si/PTEn_tall_si
# a = (np.mean(ratio) - D)/(S-D)
# a = (ufloat(1.338, .0026) - D)/(S-D)
s = (ufloat(3.1/1.68, .0026) - D)/(S-D)
s = (ufloat(2.2799, .0028) - D)/(S-D)
ufloat(2.2799, .0028) - S
