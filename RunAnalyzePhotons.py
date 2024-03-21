# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:28:33 2024

@author: Hannah
"""
from AnalyzePhotons import CorrelatedAvalancheProbability
from AnalyzePhotons import AlphaData
from AnalyzePhotons import AnalyzePhotons
from AnalyzePhotons import AlphaRatio
from MeasurementInfo import MeasurementInfo
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('C:/Users/Hannah/Documents/GitHub/SPE-Analysis/nexo_new.mplstyle')
#%% Correlated avalanche options

CA_July13 = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/July13_171K_LXe_CA.csv')
CA_July13_GN = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/July13_171K_CA_GN.csv')
CA_Sep24 = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/Sept24th_405nm_CA_values.csv')
CA_Sep20 = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/Sept20th_405nm_CA_values.csv'
                                          ,ov_key = ('Overvoltage [V]', 'Bias Voltage [V] error'),)
CA_Nov9 =  CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/Nov9_169K_CA_ov_NoSource.csv')
#%% March 2023, No reflector
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter =  0.1288
invC_spe_err_filter = 0.00064
alpha_no_teflon = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/March2023_Alpha_1us.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/March2023_SPE.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  bias_key = ("Bias Voltage [V]","Bias Voltage [V] error"),
                  spe_bias_key= ("Bias Voltage [V]", "Bias Voltage [V] error"),
                  reflector = "None",
                  color = 'cyan'
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'March 2023'
info.temperature = 170
no_reflector = AnalyzePhotons(info, "No reflector", alpha_no_teflon, CA_Sep20, max_OV = 7, use_fit_results_only = True)
no_reflector.plot_num_det_photons(label=True)

#%% May 18 2023, Teflon
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter =  0.011959447603692185
invC_spe_err_filter = 3.881945391072933E-05
alpha_teflonI = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/May2023_Alpha_1us.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/May_170K_vbd_bias_20240318.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  reflector = 'Teflon',
                  color = "blue"
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'May 18th 2023'
info.temperature = 170
teflonI = AnalyzePhotons(info, "Teflon", alpha_teflonI, CA_Sep20, max_OV = 7, use_fit_results_only = True)
teflonI.plot_num_det_photons(label=False,color = 'blue')

#%%
CA_Sep20.fit_CA(max_OV=7, color = 'xkcd:green', show_label=True)
teflonI.plot_CA(color = 'purple')
#%%
ratio_tef = AlphaRatio(alpha_teflonI, alpha_no_teflon)
ratio_tef.plot_alpha()
ratio_tef.plot_alpha_ratio(ov_from_avg = True, color = 'tab:blue')
#%% August 10 2023, No silicon
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter = 0.01171
invC_spe_err_filter = 0.000048
alpha_si_none = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/2023August10_Alpha.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/vbd_bias_no_silicon.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  color = 'olive',
                  use_fit_result_only=True
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'May 18th 2023'
info.temperature = 170
# si_none = AnalyzePhotons(info, "None / July 13 GN CA", alpha, CA_July13)
# si_none = AnalyzePhotons(info, "None / Nov 9 LXe Dark CA", alpha, CA_Nov9)
# si_none = AnalyzePhotons(info, "None / Sep 24 GN CA", alpha, CA_Sep24)
si_none = AnalyzePhotons(info, "None / Sep 20 CA (Vacuum)", alpha_si_none, CA_Sep20, max_OV = 7, use_fit_results_only = True)
si_none.plot_num_det_photons(label=False)

#%% August 1 2023, silicon
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter = 0.01171
invC_spe_err_filter = 0.000048
alpha_si = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/2023August01_Alpha.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/August1_silicon_vbd.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  reflector = "Silicon",
                  color = 'green',
                  use_fit_result_only= True
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'August 1st 2023'
info.temperature = 170
si = AnalyzePhotons(info, "Silicon / Sep 20 CA (Vacuum)", alpha_si, CA_Sep20, max_OV = 7, use_fit_results_only = True)
si.plot_num_det_photons(label=False)
#%%
ratio_si = AlphaRatio(alpha_si, alpha_si_none)
ratio_si.plot_alpha_ratio(ov_from_avg = True, color = 'xkcd:brown')
ratio_si.plot_alpha()
#%%
alpha_si.plot_alpha(unit = 'PE')
alpha_si_none.plot_alpha(unit = 'PE')
#%%
CA_Sep20.fit_CA(max_OV=7, color = 'magenta', show_label=True)
si.plot_CA(color = 'green')

#%% June 28, copper
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter = 0.01171
invC_spe_err_filter = 0.000048
alpha_copper = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/2023June28_Alpha.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/LXe_June_28_2023_vbd_bias.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  # mean_subtract= 'C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/June2023_means_subtraction_method_2.csv',
                  use_fit_result_only = False,
                  reflector = "Copper",
                  color = "brown"
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'June 28th 2023'
info.temperature = 170
# alpha_copper.get_CA_from_means(out_file = 'C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/June28_CA_from_subtraction_method.csv')
#%%
CA = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/June28_CA_from_subtraction_method.csv')
# CA = CA_July13_GN
copper = AnalyzePhotons(info, "Copper", alpha_copper, CA, max_OV = 7)
# copper.plot_num_det_photons(label=False)
CA.fit_CA(plot = True, color = 'brown', max_OV = 7, show_label = True)
# copper.CA.fit_CA(plot = True, color = 'pink')
copper.plot_CA(color = 'brown')
#%% July 13, no copper
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter = 0.01171
invC_spe_err_filter = 0.000048
alpha_no_copper = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/2023July_Alpha.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/July13_171K_LED_SPE.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  # mean_subtract= 'C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/July13_means_subtraction_method_3.csv',
                  use_fit_result_only = False,
                  reflector = "None",
                  color = "orange"
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'July 13th 2023'
info.temperature = 170
# alpha.get_CA_from_means(out_file = 'C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/July13_CA_from_subtraction_method.csv')
#%%
CA_copper_none = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/July13_CA_from_subtraction_method.csv')
copper_none = AnalyzePhotons(info, "None", alpha_no_copper, CA_copper_none, max_OV = 7)
copper_none.plot_num_det_photons(label=False)
# CA_copper_none.fit_CA(plot = True)
# copper_none.CA.fit_CA(plot = True, max_OV = 7, color = 'orange', show_label=True)
# copper_none.plot_CA()
#%%
ratio_copper = AlphaRatio(alpha_copper, alpha_no_copper)
ratio_copper.plot_alpha()
ratio_copper.plot_alpha_ratio(ov_from_avg = True, color = 'xkcd:red')

#%% June 8 2023, teflon, flipped source
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter = 0.01171
invC_spe_err_filter = 0.000048
alpha_teflonII = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/2023June8_Alphas.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/LXe_June8_2023_vbd_bias.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  reflector = "Teflon",
                  color = "blue"
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'June 8th 2023'
info.temperature = 170
CA_teflonII = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/LXe_June_2023_CA_ov.csv')
teflonII = AnalyzePhotons(info, "Teflon", alpha_teflonII, CA_teflonII, max_OV =7)
teflonII.plot_num_det_photons(label=False,color='blue')
#%%
ratio_teflonII = AlphaRatio(alpha_teflonII, alpha_no_copper)
ratio_teflonII.plot_alpha()
ratio_teflonII.plot_alpha_ratio(ov_from_avg = True, color = 'xkcd:green')
#%%
CA_teflonII.fit_CA(max_OV=7, show_label=False, color = 'blue')
CA_copper_none.fit_CA(max_OV=7, show_label=False, color = 'orange')