# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:43:41 2022

@author: lab-341
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
from RunInfo import RunInfo

    


def get_gain_hd5_11(plot = False, do_filter = True):
    # Gain calibration
    # CR-Z-SiPM, CR-S 100 ns, both gains on
    # 1000 Hz
    
    # 0.005 V
#    run_1 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648664466.hdf5'], do_filter = True),
#             'input voltage': 0.005,
#             'lower': 0.07,
#             'upper': 0.078}
    # 0.01 V
    #run_info_2 = RunInfo(['E:/Xe/DAQ/Run_1648665224.hdf5'])
    if do_filter:
        run_2 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648665224.hdf5'], do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.005,
                 'lower': 0.1175, #filtered
                 'upper': 0.1275} #filtered
        ## 0.015 V
        #run_info_3 = RunInfo(['E:/Xe/DAQ/Run_1648665421.hdf5'])
        run_3 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648665421.hdf5'], do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.0075,
                 'lower': 0.165, #filtered
                 'upper': 0.175} #filtered
        ## 0.02 V
        #run_info_4 = RunInfo(['E:/Xe/DAQ/Run_1648666140.hdf5'])
        run_4 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648666140.hdf5'], do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.010,
                 'lower': 0.212, #filtered
                 'upper': 0.222} #filtered
        ## 0.025 V
        #run_info_5 = RunInfo(['E:/Xe/DAQ/Run_1648666251.hdf5'])
        run_5 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648666251.hdf5'], do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.0125,
                 'lower': 0.2625, #filtered
                 'upper': 0.2725} #filtered
    else:
        run_2 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648665224.hdf5'], do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.005,
                 'lower': 0.37, #unfiltered
                 'upper': 0.41} #unfiltered
        ## 0.015 V
        #run_info_3 = RunInfo(['E:/Xe/DAQ/Run_1648665421.hdf5'])
        run_3 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648665421.hdf5'], do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.0075,
                 'lower': 0.55, #unfiltered
                 'upper': 0.59} #unfiltered
        ## 0.02 V
        #run_info_4 = RunInfo(['E:/Xe/DAQ/Run_1648666140.hdf5'])
        run_4 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648666140.hdf5'], do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.010,
                 'lower': 0.74, #unfiltered
                 'upper': 0.78} #unfiltered
        ## 0.025 V
        #run_info_5 = RunInfo(['E:/Xe/DAQ/Run_1648666251.hdf5'])
        run_5 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1648666251.hdf5'], do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.0125,
                 'lower': 0.93, #unfiltered
                 'upper': 0.97} #unfiltered
    if plot:
        fig = plt.figure(figsize = (8, 6))
    runs = [run_2, run_3, run_4, run_5]
    textstr = f'Date: 3/30/2022\n'
    textstr += f'Preamp: CR-Z-SiPM\n'
    textstr += f'Shaper: CR-S 100ns, 100x\n'
    if do_filter:
        textstr += f'Filtering: Lowpass, 400kHz\n'
    textstr += f'--\n'
    for curr_run in runs:
        curr_run_info = curr_run['run_info']
        for curr_file in curr_run_info.hd5_files:
            for curr_acquisition_name in curr_run_info.acquisition_names[curr_file]:
                peak_values = np.array(curr_run_info.peak_data[curr_file][curr_acquisition_name])
        curr_run['peak_data'] = peak_values
        curr_peak_values = peak_values[(peak_values >= curr_run['lower']) & (peak_values <= curr_run['upper'])]
        curr_hist = np.histogram(curr_peak_values, bins = 20)
        if plot:
            plt.hist(curr_peak_values, bins = 20, label = 'Input: ' + str(curr_run['input voltage']) + 'V')
        counts = curr_hist[0]
        bins = curr_hist[1]
        centers = (bins[1:] + bins[:-1])/2.0
        model = lm.models.GaussianModel()
        params = model.make_params(amplitude=max(counts), center=1, sigma=1)
        res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1/counts))
        curr_run['fit_res'] = res
        x = np.linspace(curr_run['lower'], curr_run['upper'], num = 200)
        if plot:
            plt.plot(x, res.eval(params = res.params, x = x), color='red')
            textstr += f'''{curr_run['input voltage']:0.3}V input, Mean: {res.params['center'].value:0.2} +- {res.params['center'].stderr:0.1} [V], '''
            textstr += f'''Sigma: {res.params['sigma'].value:0.2} +- {res.params['sigma'].stderr:0.1} [V], '''
            textstr += f'''Reduced $\chi^2$: {res.redchi:0.3}\n'''
    if plot:
        props = dict(boxstyle='round', facecolor='tab:red', alpha=0.4)
        fig.text(0.11, 0.95, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.legend(loc = 'upper right')
        plt.xlabel('Output Amplitude [V]')
        plt.ylabel('Counts')
        curr_lims = plt.ylim()
        plt.ylim(curr_lims[0], curr_lims[1] * 1.5)
        plt.tight_layout()
    
    input_voltages = []
    output_voltages = []
    output_err = []
    textstr = f'Date: 3/30/2022\n'
    textstr += f'Preamp: CR-Z-SiPM\n'
    textstr += f'Shaper: CR-S 100ns, 100x\n'
    if do_filter:
        textstr += f'Filtering: Lowpass, 400kHz\n'
    textstr += f'--\n'
    for curr_run in runs:
        input_voltages.append(curr_run['input voltage'] * 100.0)
        output_voltages.append(curr_run['fit_res'].params['center'].value)
        output_err.append(curr_run['fit_res'].params['center'].stderr)
    
    model_lin = lm.models.LinearModel()
    params_lin = model_lin.make_params(m = 1, b = -1)
    res_lin = model_lin.fit(np.array(output_voltages), params = params_lin, x = np.array(input_voltages), weights = np.sqrt(1.0 / np.array(output_err)))
    if plot:
        fig = plt.figure(figsize = ( 8, 6))
        plt.errorbar(input_voltages, output_voltages, fmt = '.', yerr = output_err, markersize = 10, color = 'black')
        x = np.linspace(np.amin(input_voltages), np.amax(input_voltages), num = 20)
        plt.plot(x, res_lin.eval(params = res_lin.params, x = x), color='red')
        res_lin.fit_report()
        textstr += f'''Slope: {res_lin.params['slope'].value:0.4} +- {res_lin.params['slope'].stderr:0.2} [V/pC]\n'''
        textstr += f'''Intercept: {res_lin.params['intercept'].value:0.4} +- {res_lin.params['intercept'].stderr:0.2} [V]\n'''
        textstr += f'''Reduced $\chi^2$: {res_lin.redchi:0.4}'''
        props = dict(boxstyle='round', facecolor='tab:red', alpha=0.4)
        fig.text(0.15, 0.9, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.xlabel('Equivalent Input Charge [pC]')
        plt.ylabel('Output Voltage [V]')
        plt.tight_layout()
    return (res_lin.params['slope'].value, res_lin.params['slope'].stderr)
#get_gain_hd5_11(plot = True, do_filter = True)


#%%
def get_gain_hd5_10(plot = False, do_filter = True):
    # Gain calibration
    # CR-Z-SiPM, CR-S 100 ns, one gain on
    # 1000 Hz


    if do_filter:        
        run_2 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253211', do_filter = do_filter, baseline_correct = False, prominence = 0.003),
                 'input voltage': 0.005,
                 'lower': 0.01210, #filtered
                 'upper': 0.01438,
                 'center': 0.0133} #filtered
        ##
        run_3 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253331', do_filter = do_filter, baseline_correct = False, prominence = 0.004),
                 'input voltage': 0.01,
                 'lower': 0.0215, #filtered
                 'upper': 0.0234,
                 'center': 0.022} #filtered
#        ## 
#
        run_4 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253433', do_filter = do_filter, baseline_correct = False, prominence = 0.01),
                 'input voltage': 0.025,
                 'lower': 0.0487, #filtered
                 'upper': 0.0505,
                 'center': 0.04969} #filtered
        # 

        run_5 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253541', do_filter = do_filter, baseline_correct = False, prominence = 0.01),
                 'input voltage': 0.05,
                 'lower': 0.0953, #filtered
                 'upper': 0.097,
                 'center': 0.0961} #filtered
        
        run_6 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253650', do_filter = do_filter, baseline_correct = False, prominence = 0.01),
                 'input voltage': 0.055,
                 'lower': 0.1073, #filtered
                 'upper': 0.1094,
                 'center': 0.1083} #filtered
#        
        run_7 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253753', do_filter = do_filter, baseline_correct = False, prominence = 0.1),
                 'input voltage': 0.1,
                 'lower': 0.1905, #filtered
                 'upper': 0.1925,
                 'center': 0.1914} #filtered
#        
        run_8 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253871', do_filter = do_filter, baseline_correct = False, prominence = 0.125),
                 'input voltage': 0.25,
                 'lower': 0.4812, #filtered
                 'upper': 0.4840,
                 'center': 0.4825} #filtered
#        
        run_9 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253949', do_filter = do_filter, baseline_correct = False, prominence = 0.125),
                 'input voltage': 0.5,
                 'lower': 0.981, #filtered
                 'upper': 0.9841,
                 'center': 0.9828} #filtered
        
    else: #NOTE--- these are placeholder values. Do not attempt to use the calibration for the unfiltered case. It is incomplete
        run_2 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253211', do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.005,
                 'lower': 0.03, #unfiltered
                 'upper': 0.041,
                 'center': 0.0439} #unfiltered
        ## 

        run_3 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253331', do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.01,
                 'lower': 0.055, #unfiltered
                 'upper': 0.059} #unfiltered
        ## 
        run_4 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253433', do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.025,
                 'lower': 0.074, #unfiltered
                 'upper': 0.078} #unfiltered
        ## 
        run_5 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253541', do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.05,
                 'lower': 0.095, #unfiltered
                 'upper': 0.097} #unfiltered
        
        
        run_6 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253650', do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.055,
                 'lower': 0.09,
                 'upper': 0.1} 
        
        run_7 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253753', do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.1,
                 'lower': 0.09,
                 'upper': 0.1} 
        
        run_8 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253871', do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.25,
                 'lower': 0.09,
                 'upper': 0.1} 
        
        run_9 = {'run_info': RunInfo(['E:/Xe/DAQ/Run_1663252916.hdf5'], specifyAcquisition = True, acquisition = 'Acquisition_1663253949', do_filter = do_filter, baseline_correct = False),
                 'input voltage': 0.5,
                 'lower': 0.09, 
                 'upper': 0.1} 
    if plot:
        fig = plt.figure(figsize = (8, 6))
    runs = [run_2, run_3, run_4, run_5, run_6, run_7, run_8, run_9]
    textstr = f'Date: 9/15/2022\n'
    textstr += f'Preamp: CR-Z-SiPM\n'
    textstr += f'Shaper: CR-S 100ns, 10x\n'
    if do_filter:
        textstr += f'Filtering: Lowpass, 400kHz\n'
    textstr += f'--\n'
    for curr_run in runs:
        curr_run_info = curr_run['run_info']
#        for curr_file in curr_run_info.hd5_files:
#            for curr_acquisition_name in curr_run_info.acquisition_names[curr_file]:
#                peak_values = np.array(curr_run_info.peak_data[curr_file][curr_acquisition_name])
        for curr_file in curr_run_info.hd5_files:
#            for curr_acquisition_name in curr_run_info.acquisition_names[curr_file]:
                peak_values = np.array(curr_run_info.peak_data[curr_file][curr_run_info.acquisition])
#            peak_values = np.array(curr_run_info.all_peak_data)
                
        curr_run['peak_data'] = peak_values
        curr_peak_values = peak_values

        curr_peak_values = peak_values[(peak_values >= curr_run['lower']) & (peak_values <= curr_run['upper'])]
        curr_hist = np.histogram(curr_peak_values, bins = 40)
        if plot:
            plt.hist(curr_peak_values, bins = 40, label = 'Input: ' + str(curr_run['input voltage']) + 'V')
        counts = curr_hist[0]
        bins = curr_hist[1]
        centers = (bins[1:] + bins[:-1])/2.0
        model = lm.models.GaussianModel()
        params = model.make_params(amplitude=max(counts), center=curr_run['center'], sigma=0.0018)
        res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1/counts))
        curr_run['fit_res'] = res
        x = np.linspace(curr_run['lower'], curr_run['upper'], num = 200)
        if plot:
            plt.plot(x, res.eval(params = res.params, x = x), color='red')
            textstr += f'''{curr_run['input voltage']:0.3}V input, Mean: {res.params['center'].value:0.2} +- {res.params['center'].stderr:0.1} [V], '''
            textstr += f'''Sigma: {res.params['sigma'].value:0.2} +- {res.params['sigma'].stderr:0.1} [V], '''
            textstr += f'''Reduced $\chi^2$: {res.redchi:0.3}\n'''
    if plot:
        props = dict(boxstyle='round', facecolor='tab:red', alpha=0.4)
        fig.text(0.11, 0.95, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.legend(loc = 'upper right')
        plt.xlabel('Output Amplitude [V]')
        plt.ylabel('Counts')
        curr_lims = plt.ylim()
        plt.ylim(curr_lims[0], curr_lims[1] * 1.5)
        plt.tight_layout()
    
    input_voltages = []
    output_voltages = []
    output_err = []
    textstr = f'Date: 3/30/2022\n'
    textstr += f'Preamp: CR-Z-SiPM\n'
    textstr += f'Shaper: CR-S 100ns, 100x\n'
    if do_filter:
        textstr += f'Filtering: Lowpass, 400kHz\n'
    textstr += f'--\n'
    for curr_run in runs:
        input_voltages.append(curr_run['input voltage'] * 100.0) # why x100 here?
        output_voltages.append(curr_run['fit_res'].params['center'].value)
        output_err.append(curr_run['fit_res'].params['center'].stderr)
    
    model_lin = lm.models.LinearModel()
    params_lin = model_lin.make_params(m = 1, b = -1)
    res_lin = model_lin.fit(np.array(output_voltages), params = params_lin, x = np.array(input_voltages), weights = np.sqrt(1.0 / np.array(output_err)))
    if plot:
        fig = plt.figure(figsize = ( 8, 6))
        plt.errorbar(input_voltages, output_voltages, fmt = '.', yerr = output_err, markersize = 10, color = 'black')
        x = np.linspace(np.amin(input_voltages), np.amax(input_voltages), num = 20)
        plt.plot(x, res_lin.eval(params = res_lin.params, x = x), color='red')
        res_lin.fit_report()
        textstr += f'''Slope: {res_lin.params['slope'].value:0.4} +- {res_lin.params['slope'].stderr:0.2} [V/pC]\n'''
        textstr += f'''Intercept: {res_lin.params['intercept'].value:0.4} +- {res_lin.params['intercept'].stderr:0.2} [V]\n'''
        textstr += f'''Reduced $\chi^2$: {res_lin.redchi:0.4}'''
        props = dict(boxstyle='round', facecolor='tab:red', alpha=0.4)
        fig.text(0.15, 0.9, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.xlabel('Equivalent Input Charge [pC]')
        plt.ylabel('Output Voltage [V]')
        plt.tight_layout()
    return (res_lin.params['slope'].value, res_lin.params['slope'].stderr)

#get_gain_hd5_10(plot = True, do_filter = True)

#%% write final result of calibration 
# slope, error = get_gain_hd5_11(plot = False, do_filter = True)
# print(slope,error)