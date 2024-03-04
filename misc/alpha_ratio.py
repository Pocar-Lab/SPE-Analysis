# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:23:35 2023

@author: Hannah
"""
import numpy as np
import scipy
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import lmfit as lm
#%%
plt.style.use('D:/Xe/AnalysisScripts/LXe May 2023/nexo_new.mplstyle')

df = pd.read_csv('D:/Xe/AnalysisScripts/LXe May 2023/May2023_Alpha_1us.csv') 
bias = df['Bias voltage [V]']
bias_err = df['Bias voltage [V] error']
amp = df['Alpha Pulse Amplitude [V]']
amp_err = df['Alpha Pulse Amplitude [V] error']

#%%
df_100ns = pd.read_csv('D:/Xe/AnalysisScripts/LXe May 2023/May2023_100ns_alpha.csv') 
bias_100ns = df['Bias voltage [V]']
bias_err_100ns = df['Bias voltage [V] error']
amp_100ns = df['Alpha Pulse Amplitude [V]']
amp_err_100ns = df['Alpha Pulse Amplitude [V] error']
#%%
df_no_reflector = pd.read_csv('D:/Xe/AnalysisScripts/LXe May 2023/March2023_AlphaAmplitudes.csv') 
bias_nr = df_no_reflector['Bias']
bias_nr_err = [0.0025 * V + 0.015 for V in bias_nr] #error from keysight
amp_nr = df_no_reflector['Amp']
amp_nr_err = df_no_reflector['Err']
#%%
v_bd = 27.65
v_bd_err = 0.08
ov = [v-v_bd for v in bias]
ov_err = [np.sqrt(v_bd_err**2+v_err**2) for v_err in bias_err]

v_bd_nr = 27.3
v_bd_nr_err = 0.07
ov_nr = [v-v_bd_nr for v in bias_nr]
ov_nr_err = [np.sqrt(v_bd_nr_err**2+v_err**2) for v_err in bias_nr_err]
#%%
plt.errorbar(
            ov,
            amp,
            xerr=bias_err,
            yerr=amp_err,
            markersize=5,
            fmt=".",
            color='red',
            data_label = "1 $\mu$s"
        )
plt.errorbar(
            ov_nr,
            amp_nr,
            xerr=bias_nr_err,
            yerr=amp_nr_err,
            markersize=5,
            fmt=".",
            color='blue',
            data_label = "100 ns"
        )
plt.legend()
x_label = "Overvoltage [V]"
plt.xlabel(x_label)
y_label = "Alpha Pulse Amplitude [V]"
plt.ylabel(y_label)
#%% testing interpolator ?
interp = scipy.interpolate.BarycentricInterpolator(ov, amp)
interp_nr = scipy.interpolate.BarycentricInterpolator(ov_nr, amp_nr)
#%%
interp.__call__(1)
#%%
amp_ratio = [interp.__call__(i)/interp_nr.__call__(i) for i in range(3,9)]
#%%
plt.errorbar(
            range(3,9),
            amp_ratio,
            # xerr=0.0025 * range(2,7) + 0.015,
            # yerr=amp_err,
            markersize=5,
            fmt=".",
            color='purple',
        )

x_label = "Overvoltage [V]"
plt.xlabel(x_label)
y_label = "Alpha Pulse Amplitude [V]"
plt.ylabel(y_label)
#%%
def model_func(x,A,B):
    # return (A * np.exp(B*(x)) + 1.0) / (1.0 + A) - 1.0
    return A*np.sinh(B*x)
#%%
model = lm.Model(model_func)
params = model.make_params(A=.01, B=0)
# params['A'].min = 0
# params['B'].min = 0
# params['C'].min = 0
# params['C'].max = 0.1
wgts = [1.0 / curr_std for curr_std in amp_err]
res = model.fit(
            amp, params=params, x=ov, weights=wgts
        )
#%%
def get_amp(ov):
        out_vals = res.eval(params=res.params, x=ov)
        out_err = res.eval_uncertainty(x=ov, sigma=1)
        return out_vals, out_err
    
def plot_fit(ov,amp, err_x, err_y):
    fig = plt.figure()

    data_x = ov
    data_x_err = err_x
    data_y = amp
    data_y_err = err_y

    fit_x = np.linspace(0, np.amax(ov) + 1.0, num=100)
    fit_y = res.eval(params=res.params, x=fit_x)
    fit_y_err = res.eval_uncertainty(x=fit_x, params=res.params, sigma=1)

    plt.fill_between(
        fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color="deeppink", alpha=0.5
    )
    plt.plot(
        fit_x,
        fit_y,
        color="deeppink",
        label=r"$Ae^{B*V_{OV}}$ fit",
    )
    plt.errorbar(
        data_x,
        data_y,
        xerr=data_x_err,
        yerr=data_y_err,
        markersize=10,
        fmt=".",
        label=r"Alpha amplitude",
    )

    x_label = "Overvoltage [V]"
    y_label = "Alpha Pulse Amplitude"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper left")
    textstr = f"--\n"
    textstr += f"""A: {res.params['A'].value:0.3} $\pm$ {res.params['A'].stderr:0.3}\n"""
    textstr += f"""B: {res.params['B'].value:0.2} $\pm$ {res.params['B'].stderr:0.2}\n"""
    textstr += rf"""Reduced $\chi^2$: {res.redchi:0.4}"""
    props = dict(boxstyle="round", alpha=0.4)
    fig.text(0.15, 0.65, textstr, fontsize=8, verticalalignment="top", bbox=props)
#%%
plot_fit(ov,amp,bias_err,amp_err)
#%%
plot_fit(ov_nr,amp_nr,bias_nr_err,amp_nr_err)
