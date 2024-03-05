# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:28:33 2024

@author: Hannah
"""
from AnalyzePhotons import CorrelatedAvalancheProbability
from AnalyzePhotons import AlphaData
from AnalyzePhotons import AnalyzePhotons
from MeasurementInfo import MeasurementInfo
import matplotlib.pyplot as plt
plt.style.use('C:/Users/Hannah/Documents/GitHub/SPE-Analysis/nexo_new.mplstyle')
#%% Correlated avalanche options

CA_July13 = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/July13_171K_LXe_CA.csv')
# CA_Sep24 = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/Sept24th_405nm_CA_values.csv')

#%% March 2023, No reflector
path_march = 'D:/Raw Results/Breakdown Voltage/SPE Amplitudes/March2023_SPE.csv'
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter =  0.1288
invC_spe_err_filter = 0.00064
alpha = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/March2023_Alpha_1us.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/March2023_SPE.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  bias_key = ("Bias Voltage [V]","Bias Voltage [V] error"),
                  spe_bias_key= ("Bias Voltage [V]", "Bias Voltage [V] error")
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'March 2023'
info.temperature = 170
no_reflector = AnalyzePhotons(info, "No reflector", alpha, CA_July13)
no_reflector.plot_num_det_photons(label=True)

#%% May 18 2023, Teflon
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter =  0.011959447603692185
invC_spe_err_filter = 3.881945391072933E-05
CA_July13 = CorrelatedAvalancheProbability('C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/July13_171K_LXe_CA.csv')
alpha = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/May2023_Alpha_1us.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/May_170K_vbd_bias_20240301.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error')
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'May 18th 2023'
info.temperature = 170
teflon = AnalyzePhotons(info, "Teflon", alpha, CA_July13)
teflon.plot_num_det_photons(label=False,color='green')

#%%

#%% August 10 2023, No silicon
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_filter = 0.01171
invC_err_filter = 0.000048
# CA_July13 = CorrelatedAvalancheProbability('D:/Raw Results/Correlated Avalanche Data/July13_171K_LXe_CA.csv')
alpha = AlphaData(path = 'C:/Users/Hannah/Documents/Alpha Amplitudes/2023August10_Alpha.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/vbd_bias_no_silicon.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'May 18th 2023'
info.temperature = 170
teflon = AnalyzePhotons(info, "Teflon", alpha, CA_July13)
teflon.plot_num_det_photons(label=False,color='green')

#%% August 1 2023, silicon
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter = 0.01171
invC_spe_err_filter = 0.000048
alpha = AlphaData(path = 'C:/Users/Hannah/Documents/Raw Results/Alpha Amplitudes/2023August01_Alpha.csv', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/August1_silicon_vbd.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
                  bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error')
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'August 1st 2023'
info.temperature = 170
teflon = AnalyzePhotons(info, "Silicon", alpha, CA_July13)
teflon.plot_num_det_photons(label=False,color='blue')
#%% June 8 2023, teflon
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter = 0.01171
invC_spe_err_filter = 0.000048
alpha = AlphaData(path = '', 
                  path_spe = 'C:/Users/Hannah/Documents/Raw Results/Breakdown Voltage/LXe_June_2023_vbd_bias.csv',
                  invC_alpha = (invC_alpha,invC_alpha_err),
                  invC_spe = (invC_spe_filter,invC_spe_err_filter),
                  spe_bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error')
                  )
info = MeasurementInfo()
info.condition = 'LXe'
info.date = 'June 8th 2023'
info.temperature = 170
teflon = AnalyzePhotons(info, "Silicon", alpha, CA_July13)
teflon.plot_num_det_photons(label=False,color='blue')