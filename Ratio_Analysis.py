# -*- coding: utf-8 -*-
"""
Created on Fri Dec 6 2024

@author: Ed van Bruggen (evanbruggen@umass.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# from ProcessWaveforms_MultiGaussian import WaveformProcessor as WaveformProcessor
from uncertainties import ufloat
from src.AnalyzeResults import AnalyzeResults
import pandas as pd
plt.style.use('misc/nexo.mplstyle')
plt.style.use('misc/theme.mplstyle')

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

cu_none = AnalyzeResults('Copper Spacing (2023/07/13)', 'results/2023July13_Alpha.csv',
                         'results/August1_silicon_vbd.csv', 'results/July13_171K_CA_GN.csv',
                         invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27.22, .0686,
                         ov_max=6, ov_min=2.5)


si_short8_refl = AnalyzeResults('Si 8 Short Reflectors (2023/08/01)', 'results/2023August01_Alpha.csv',
                          'results/August1_silicon_vbd.csv', 'results/July13_171K_CA_GN.csv',
                          invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27.23, .04,
                          ov_max=8, ov_min=2)


si_short8_none = AnalyzeResults('Si Short Spacing (2023/08/10)', 'results/2023August10_Alpha.csv',
                               'results/20230810_SPE_no_si.csv', 'results/July13_171K_CA_GN.csv',
                               invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27., .0488,
                               ov_max=8, ov_min=2)

cu_none.plot_ratio(si_short8_none, fit='exp1', alpha_ylim=(0, 1), ratio_ylim=(0,5))

si_short8_refl.plot_ratio(si_short8_none, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,3.5))

si_tall_none_sipm.plot_ratio(si_tall_none_pure, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,3.5))

si_short_none = AnalyzeResults('Si Short Spacing', 'results/2023August10_Alpha.csv',
                               'results/20230810_SPE_no_si.csv', 'results/July13_171K_CA_GN.csv',
                               invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27., .0488,
                               ov_max=6, ov_min=2.5)

si_tall_none_predict = AnalyzeResults('Si Tall Spacing Prediction',
                                      'results/Tall_Si_Spacing_predict_alpha.csv',
                                      'results/20230810_SPE_no_si.csv', 'results/July13_171K_CA_GN.csv',
                                      invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27., .0488,
                                      ov_max=6, ov_min=2.5)

si_short4_refl_pre = AnalyzeResults('Si 4 Short Reflectors (2024/06/20) Pre-Baking', 'results/2024June20_Alpha.csv',
                             'results/20240620_LXe_SPE_vbd_bias.csv',
                             'results/July13_171K_CA_GN.csv',
                             invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
                             26.96, .102, # Thomas's SPE
                             ov_max=6.2, ov_min=2.7)

si_short4_refl_post = AnalyzeResults('Si 4 Short Reflectors (2024/06/27) Post-baking', 'results/2024June27_Alpha.csv',
                               'results/20240627_LXe_SPE.csv',
                               'results/July13_171K_CA_GN.csv',
                               invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
                               27.16, .0263, # Thomas's SPE
                               ov_max=6.2, ov_min=2.7)

si_short4_refl_air = AnalyzeResults('Si 4 Short Reflectors (2024/07/09) Post-Baking/Air', 'results/2024July9_Alpha.csv',
                             'results/20241017_LXe_SPE_vbd_bias.csv',
                             'results/July13_171K_CA_GN.csv',
                             invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
                             27.11, .0583, # Thomas's SPE
                             ov_max=6.2, ov_min=2.7)

si_tall_none_gnd = AnalyzeResults('Si Tall Spacing, GND Loop', '2024Aug07_Alpha.csv',
                                'results/20241017_LXe_SPE_vbd_bias.csv',
                                'results/July13_171K_CA_GN.csv',
                                invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, v_bd, v_bd_err,
                                ov_max=6, ov_min=2.5)

si_tall_none = AnalyzeResults('Si Tall Spacing', 'results/2024Oct17_Alpha.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 26.91, 0.122, # v_bd, v_bd_err,
                            ov_max=7.5, ov_min=2)

si_tall_none_pure_old = AnalyzeResults('April 29th 2026', 'results/20260429_Alpha.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 26.68, .220, #v_bd, v_bd_err,
                            ov_max=7, ov_min=2)

si_tall_none_pure = AnalyzeResults('June 25th 2026', 'results/20260625_Alpha.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 0, 0, #v_bd, v_bd_err,
                            ov_max=37, ov_min=28)

si_short_none_sipm = AnalyzeResults('Si Short Spacing (2026/07/21)', 'results/20260721_Alpha.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, v_bd, v_bd_err,
                            ov_max=7, ov_min=2)

si_tall_refl = AnalyzeResults('Si Tall 4 Reflector (2026/09/03)', 'results/20260903_Alpha.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27.1, 0.076,
                            ov_max=8, ov_min=2)

si_tall_none = AnalyzeResults('Si Tall Spacing (2026/09/16)', 'results/20260916_Alpha.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27.48, 0.12,
                            ov_max=8, ov_min=2)

si_tall_refl.calc_ratio_mc(si_tall_none)

si_tall_refl.plot_ratio(si_tall_none, fit='exp1', alpha_ylim=(0, 1), ratio_ylim=(0,3.4))

si_tall_none.plot_ratio(si_short_none, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,3.5))

si_tall_refl.plot_ratio(si_short8_refl, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,3.5))

si_tall_refl_led = AnalyzeResults('Si Tall 4 Reflector LED (2026/09/03)', 'results/20260903_Alpha_LED.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27.1, 0.076,
                            ov_max=8, ov_min=2)

si_tall_refl.plot_ratio(si_tall_refl_led, fit='exp1', alpha_ylim=(0, 1), ratio_ylim=(0,2))

si_tall_none_predict_new = AnalyzeResults('Si Tall Spacing Predict (2026/07/21)', 'results/Tall_Si_Spacing_predict_20260721_alpha.csv',
                            'results/20241017_LXe_SPE_vbd_bias.csv',
                            'results/July13_171K_CA_GN.csv',
                            invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, v_bd, v_bd_err,
                            ov_max=7, ov_min=2)

si_tall_refl.plot_ratio(si_tall_none_predict_new, fit='exp1', alpha_ylim=(0, 1), ratio_ylim=(0,3.4))

si_tall_refl.plot_ratio(si_short8_none, fit='exp1', alpha_ylim=(0, 1), ratio_ylim=(0,3.4))

si_short8_refl.plot_ratio(si_short8_none, fit='exp1', alpha_ylim=(0, 1), ratio_ylim=(0,3.4))
si_short4_refl_pre.plot_ratio(si_short8_none, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,1))
si_short4_refl_post.plot_ratio(si_short8_none, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,2))
si_short4_refl_air.plot_ratio(si_short8_none, fit='exp1', alpha_ylim=(0, .4), ratio_ylim=(0,2))
si_short_none_sipm.plot_ratio(si_short8_none, fit='exp1', alpha_ylim=(0, .4), ratio_ylim=(0,2))

si_tall_pre.plot_ratio(si_tall_none_sipm, fit='exp1', alpha_ylim=(0, .7), ratio_ylim=(0,3.5))

si_tall_pre.plot_ratio(si_tall_none_predict, fit='exp1', alpha_ylim=(0, .7), ratio_ylim=(0,3.5))

si_tall_atm.plot_ratio(si_tall_none_sipm, fit='exp1', alpha_ylim=(0, .7), ratio_ylim=(0,3.5))

si_tall_none_sipm.plot_ratio(si_tall_none_pure, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,3.5))

si_tall_none_sipm.plot_ratio(si_tall_none_predict, si_short_none, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,1.5))

si_tall_none_pure.plot_ratio(si_tall_none_predict, fit='exp1', alpha_ylim=(0, .3), ratio_ylim=(0,1.5))

## No reflector
si_short_none.plot_ratio(si_tall_none, calibrate=baked, fit='exp1', alpha_ylim=(0,.5),
                         ratio_ylim=(0,7))

cu_none.plot_ratio(si_tall_none, calibrate=baked_atm, fit='exp1', alpha_ylim=(0,1.5),
                   ratio_ylim=(0,25))

cu_none.plot_ratio(si_short_none, fit='exp1', alpha_ylim=(0,2), ratio_ylim=(0,5))

## Baking
baked_atm = si_tall_atm.plot_ratio(si_tall_pre, fit='exp1', alpha_ylim=(0,.5), ratio_ylim=(0,3.6))

baked = si_tall_baked.plot_ratio(si_tall_pre, fit='exp1', alpha_ylim=(0,.5), ratio_ylim=(0,3))

baked_atm = si_tall_atm.plot_ratio(si_tall_baked, fit='exp1', alpha_ylim=(0,.5), ratio_ylim=(0,2))

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
si_tall_pre.plot_ratio(si_tall_none, fit='exp1', alpha_ylim=(0,.5), ratio_ylim=(0,2))

ratio = si_tall_baked.plot_ratio(si_tall_none, fit='exp1', alpha_ylim=(0,400), ratio_ylim=(0,3.5))

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

## Short/Tall Comparison
si_short.plot_sub_ratio(si_tall_pre, si_short_none, 1.072, fit='exp1', alpha_ylim=(0,.8), ratio_ylim=(0,3))

2.46 - 2.28*1.072



## Average and STD of SPE breakdowns
v  = [27.13, 27.23, 27.0,  27.51, 27.11, 27.08, 26.96, 27.16, ]
s  = [.0342, .04,   .0488, .0528, 0.06,  .096,  0.10,  0.03,  ]
t  = [167.8, 168.4, 168.5, 167.0, 169.8, 169.6, 169.7, 169.8, ]
ts = [.062,  .067,  .103,  .134,  .221,  .148,  .187,  .166,  ]
 # 27.22,
 # .0686,
np.average(v)
np.sqrt(np.average(s)**2 + np.std(v)**2)


# Predict Tall Si Spacing alpha ratio
# df = pd.read_csv("results/2023August10_Alpha.csv")
df = pd.read_csv("results/20260721_Alpha.csv")

si_short_PTE = 0.0013598
si_tall_PTE = 0.001268
PTE_ratio = si_tall_PTE/si_short_PTE
# df['Alpha Pulse Amplitude [V]'] = df['Alpha Pulse Amplitude [V]'] / 1.072
df['Alpha Pulse Amplitude [V]'] = df['Alpha Pulse Amplitude [V]'] * PTE_ratio
df.to_csv("results/Tall_Si_Spacing_predict_20260721_alpha.csv", index=False)
