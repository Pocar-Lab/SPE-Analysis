# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:14:37 2023

@author: Hannah
"""

import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
import pandas as pd
#%%

# May
v_bd = 27.63
v_bd_err = 0.07

# March (no reflector)
# v_bd = 27.3
# v_bd_err = 0.07
#%% May
path = 'D:/Raw Results/Alpha Amplitudes/May2023_Alpha_1us.csv'
df = pd.read_csv(path) 
bias = df['Bias voltage [V]']
bias_err = df['Bias voltage [V] error']
alpha_vals = np.array(df['Alpha Pulse Amplitude [V]'])
alpha_err = np.array(df['Alpha Pulse Amplitude [V] error'])

ov_alpha = np.array([V - v_bd for V in bias])
ov_alpha_err = np.array(bias_err)

#%% March
path = 'D:/Raw Results/Alpha Amplitudes/March2023_Alpha_1us.csv'
df = pd.read_csv(path) 
bias_nr = df['Bias']
bias_nr_err = [0.0025 * V + 0.015 for V in bias]
alpha_vals_nr = np.array(df['Amp'])
alpha_nr_err = np.array(df['Err'])

ov_alpha = np.array([V - v_bd for V in bias])
ov_alpha_err = np.array(bias_err)

#%% CA SPE
path = 'D:/Raw Results/July2023_CA_GN.csv'
df = pd.read_csv(path) 
bias_CA = df['Bias Voltage [V]']
bias_CA_err = df['Bias Voltage [V] error']
spe_vals_from_CA = np.array(df['SPE Amplitude [V]'])
spe_err_from_CA = np.array(df['SPE Amplitude [V] error'])

#%% May
df = pd.read_csv('D:/Raw Results/Breakdown Voltage/May_170K_vbd_bias.csv') 
bias_bd = df['Bias Voltage [V]']
bias_bd_err = df['Bias Voltage [V] error']
spe_vals_from_bd = np.array(df['SPE Amplitude [V]'])
spe_err_from_bd = np.array(df['SPE Amplitude [V] error'])
#%% CA Probability
df = pd.read_csv('D:/Raw Results/July2023_CA_GN_probability.csv')
ov_CA = np.array(df['Overvoltage [V]'])
ov_CA_err = np.array(df['Overvoltage [V] error'])
CA_vals = np.array(df['Number of CA [PE]'])
CA_vals_err = np.array(df['Number of CA [PE] error'])

#%%
def CA_func(x, A, B):
    return  (A * np.exp(B * x) + 1.0) / (1.0 + A) - 1.0
#%%
def plot_CA(ov, ov_err, CA_vals, CA_err, alpha_ov_vals, color = 'blue'):
    color = 'tab:' + color
    
    data_x = ov
    data_x_err = ov_err
    data_y = CA_vals
    data_y_err = CA_err
    
    CA_model = lm.Model(CA_func)
    CA_params = CA_model.make_params(A = 1, B = .1)
    CA_wgts = [1.0 / curr_std for curr_std in CA_err]
    CA_res = CA_model.fit(CA_vals, params=CA_params, x=ov, weights=CA_wgts)
        
    fit_x = np.linspace(0.0, 10.0, num = 100)
    fit_y = CA_res.eval(params=CA_res.params, x = fit_x)
    fit_y_err = CA_res.eval_uncertainty(x = fit_x, params = CA_res.params, sigma = 1)

    plt.fill_between(fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color = color, alpha = .5)
    plt.plot(fit_x, fit_y, color = color) #, label = r'$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit'
    plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', label = r'$\frac{1}{N}\sum_{i=1}^{N}{\frac{A_i}{\bar{A}_{1 PE}}-1}$')
        
    x_label = 'Overvoltage [V]'
    y_label = 'Number of CA [PE]'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    
    out_vals = CA_res.eval(params=CA_res.params, x=alpha_ov_vals)
    out_err = CA_res.eval_uncertainty(x=alpha_ov_vals, sigma=1)
    return out_vals, out_err
#%%
def get_v_bd(spe, spe_err, bias, bias_err): # gets v_bd of CA data
    model = lm.models.LinearModel()
    params = model.make_params()
    spe_wgts = spe_wgts = [1.0 / curr_std for curr_std in spe_err]
    spe_res = model.fit(spe, params=params, x=bias, weights=spe_wgts)

                        # linear fit
    b_spe = spe_res.params["intercept"].value
    m_spe = spe_res.params["slope"].value
    v_bd = -b_spe / m_spe 
    return v_bd

#%%
def get_spe_ov(bias, spe, spe_err, input_ov_vals):
    model = lm.models.LinearModel()
    params = model.make_params()
    spe_wgts = spe_wgts = [1.0 / curr_std for curr_std in spe_err]
    spe_res = model.fit(spe, params=params, x=bias, weights=spe_wgts)

    # linear fit
    b_spe = spe_res.params["intercept"].value
    m_spe = spe_res.params["slope"].value
    v_bd = -b_spe / m_spe 
    print(v_bd)
    
    input_bias_vals = input_ov_vals + v_bd
    out_vals = spe_res.eval(params=spe_res.params, x=input_bias_vals)
    out_err = spe_res.eval_uncertainty(x=input_bias_vals, sigma=1)
    return out_vals, out_err
#%%
out_spe, out_spe_err = get_spe_ov(bias_CA, spe_vals_from_CA, spe_err_from_CA, ov_alpha)
# out_spe, out_spe_err = get_spe_ov(bias_bd, spe_vals_from_bd, spe_err_from_bd, ov_alpha)
#%%

# def get_spe_amp(spe, spe_err):
    
#     model = lm.models.LinearModel()
#     params = model.make_params()
#     spe_wgts = spe_wgts = [1.0 / curr_std for curr_std in spe_err]
#     spe_res = model.fit(spe, params=params, x=bias, weights=spe_wgts)
#     b_spe = spe_res.params["intercept"].value
#     m_spe = spe_res.params["slope"].value
#     v_bd = -b_spe / m_spe 
    
#     spe_amp_from_fit = 
    





#%%
CA_vals_from_fit, CA_err_from_fit = plot_CA(ov_CA, ov_CA_err, CA_vals, CA_vals_err, ov_alpha)

out_spe, out_spe_err = get_spe_ov(bias_CA, spe_vals_from_CA, spe_err_from_CA, ov_alpha)
# out_spe, out_spe_err = get_spe_ov(bias_bd, spe_vals_from_bd, spe_err_from_bd, ov_alpha)
#%%
export = {"A_SPE": out_spe, "A_SPE error": out_spe_err, "OV": ov_alpha, "OV error": ov_alpha_err,}
save = pd.DataFrame(export)
save.to_csv('D:/Raw Results/A_SPE.csv')

#%%
CA_invC = 0.01171 
CA_invC_err = 4.8E-05
invC_alpha =  0.001142
invC_alpha_err = 2.1E-6
invC_spe_filter =  0.011959447603692185
invC_spe_err_filter = 3.881945391072933E-05
T = 170
date = 'May 18th 2023'
shaper = '1 $\mu$s'
invC_spe = CA_invC
invC_spe_err = CA_invC_err
# invC_spe = invC_spe_filter
# invC_spe_err = invC_spe_err_filter

def plot_num_det_photons(ov, ov_err, spe_vals, spe_err, CA_vals, CA_err, alpha_vals, alpha_err, color = "purple", out_file = None):
     """
     Plot the number of detected photons as a function of overvoltage.
     """
     
     num_det_photons = (
            alpha_vals * invC_spe
            / (spe_vals * invC_alpha * (1.0 + CA_vals))
        )
     num_det_photons_err = num_det_photons * np.sqrt(
            (alpha_err * alpha_err) / (alpha_vals * alpha_vals)
            + (invC_spe_err * invC_spe_err)
            / (CA_invC * CA_invC)
            + (spe_err * spe_err) / (spe_vals * spe_vals)
            + (invC_alpha_err * invC_alpha_err) / (invC_alpha * invC_alpha)
            + (CA_err * CA_err) / (CA_vals * CA_vals)
        )     
     color = "tab:" + color
     # fig = plt.figure()

     data_x = ov
     data_x_err = ov_err
     data_y = num_det_photons
     data_y_err = num_det_photons_err

     plt.errorbar(
         data_x,
         data_y,
         xerr=data_x_err,
         yerr=data_y_err,
         markersize=7,
         fmt=".",
         color=color,
     )

     plt.xlabel("Overvoltage [V]")
     plt.ylabel("Number of Detected Photons")
     textstr = f"Date: {date}\n"
     textstr = f"Shaper: {shaper}\n"
     textstr += f"RTD4: {T} [K]\n"
     textstr += f'Filtering: None'

     props = dict(boxstyle="round", facecolor=color, alpha=0.4)
     # fig.text(0.3, 0.4, textstr, fontsize=18, verticalalignment="top", bbox=props)
     plt.xlim(0, np.amax(ov) + 1.0)
     ylow, yhigh = plt.ylim()
     plt.ylim(-1, yhigh * 1.1)
     plt.grid(True)
     plt.tight_layout()
     
     x_label = "Overvoltage [V]"
     y_label = "Number of Detected Photons"
     if out_file:
            data = {
                x_label: data_x,
                x_label + " error": data_x_err,
                y_label: data_y,
                y_label + " error": data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)
#%%
plot_num_det_photons(ov_alpha, ov_alpha_err, out_spe, out_spe_err, CA_vals_from_fit, CA_err_from_fit, alpha_vals, alpha_err, color = "orange", out_file = 'D:/Raw Results/May2023_num_det_photons_1us.csv')

#%%
ratio = []
may = zip(bias, alpha_vals)
march = zip(bias_nr, alpha_vals_nr)
# r = zip(bias, alpha_vals, bias_nr, alpha_vals_nr)

# for i in bias: 
#     for j in bias_nr:
#         if i==j:
#             print(i,j)

for i,j in may:
    print(i,j)
    for k,l in march:
        print(k,l)
    