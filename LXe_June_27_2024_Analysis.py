# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2024

Si tall reflector, after baking

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

plt.style.use('misc/nexo.mplstyle')

#%% ALPHA - 1us
#1us, no gain
invC_alpha_1us = 1140e-6
invC_alpha_err_1us = 0.32e-6
file_path = 'june-27-2024/' # folder with H5 data files
files = ['Run_1719503841.hdf5',  'Run_1719504543.hdf5',  'Run_1719504868.hdf5',  'Run_1719505534.hdf5']
proms = [ .04, .02, .04, .01]
upperlim = [ 5, 1, 5, .1]
files += [ 'Run_1719505948.hdf5', 'Run_1719506619.hdf5', 'Run_1719506922.hdf5', 'Run_1719507150.hdf5', 'Run_1719507450.hdf5' ]
proms += [ .1, .04, .04, .04, .1]
upperlim += [ 5, 5, 1, 1, 5]
files += ['Run_1719508113.hdf5', 'Run_1719508578.hdf5', 'Run_1719508808.hdf5', 'Run_1719509115.hdf5', 'Run_1719509416.hdf5']
proms += [ .02, .02, .01, .04, .04]
upperlim += [ .3, .3, .3, 4, .5]
runs_alpha_1us = []
i = 0
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file]], do_filter=False,
                            upper_limit=upperlim[file+i], baseline_correct=True,
                            prominence=proms[file+i], plot_waveforms=False,)
    # run_alpha_1us.plot_hists('','')
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

runs_alpha_1us[3].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [40, 40, 40, 40]
bins += [40, 40, 40, 40, 40]
bins += [40, 40, 40, 40, 40]
for n in range(len(runs_alpha)):
    info_alpha = MeasurementInfo()
    info_alpha.min_alpha_value = 0.001
    info_alpha.condition = 'LXe'
    info_alpha.date = runs_alpha[n].date
    info_alpha.temperature = 170
    info_alpha.bias = runs_alpha[n].bias
    info_alpha.baseline_numbins = 40
    info_alpha.peaks_numbins = bins[n]
    # print(f'{n=}')
    # print(f"{info_alpha.bias=}")
    wp = WaveformProcessor(info_alpha, run_info_self = runs_alpha[n], baseline_correct = True,
                           no_solicit = True, cutoff=(0,10))
    wp.process(do_spe = False, do_alpha = True)
    # j, k = wp.get_alpha()
    wp.plot_alpha_histogram(peakcolor = 'blue', with_fit=True)
    campaign_alpha.append(wp)
    # break

# bias_vals = []
# bias_err = []
# alpha_vals = []
# alpha_err = []
# for wp in campaign_alpha:
#     bias_vals.append(wp.info.bias)
#     # self.bias_err.append(0.0025 * wp.info.bias + 0.015)
#     bias_err.append(0.005)
#     curr_alpha = wp.get_alpha()
#     alpha_vals.append(curr_alpha[0])
#     alpha_err.append(curr_alpha[1])
# ov = []
# ov_err = []
# for b, db in zip(bias_vals, bias_err):
#     curr_ov = b - 27.3
#     curr_ov_err = np.sqrt(db * db + .103 * .103)
#     ov.append(curr_ov)
#     ov_err.append(curr_ov_err)

# data = {
#     'ov': ov, 'ov error': ov_err,
#     'bias': bias_vals, 'bias err': bias_err,
#     'amps': alpha_vals, 'amps err': alpha_err,
# }
# df = pd.DataFrame(data)
# df.to_csv('june-27-2024-alpha-amp-lxe.csv')

bias_vals = []
bias_err = []
alpha_vals = []
alpha_err = []
for wp in campaign_alpha:
    bias_vals.append(wp.info.bias)
    bias_err.append(0.0025 * wp.info.bias + 0.015)
    # bias_err.append(0.005)
    curr_alpha = wp.get_alpha()
    alpha_vals.append(curr_alpha[0])
    alpha_err.append(curr_alpha[1])

data = {
    # 'ov': data_x, 'ov error': data_x_err,
    'Bias Voltage [V]': bias_vals, 'Bias Voltage error [V]': bias_err,
    'Alpha Pulse Amplitude [V]': alpha_vals, 'Alpha Pulse Amplitude error [V]': alpha_err,
    # 'num_det': self.num_det_photons, 'num_det error': self.num_det_photons_err,
    # 'pde': data_y, 'pde error': data_y_err,
}
df = pd.DataFrame(data)
df.to_csv('2024June27_Alpha.csv')

# plotting
xvar = 'bias'
df = pd.read_csv('june-20-2024-alpha-amp-lxe.csv').sort_values('ov')
june_20_x = df[xvar]
june_20_x_err = df[xvar+' err']
june_20_alpha = df['amps']
june_20_alpha_err = df['amps error']

df = pd.read_csv('june-27-2024-alpha-amp-lxe.csv').sort_values('ov')
june_27_x = df[xvar]
june_27_x_err = df[xvar+' err']
june_27_alpha = df['amps']
june_27_alpha_err = df['amps error']

# june_20_alpha = [.024, .031, .042, .057, .071, .087, .104, .126, .156, .190, .234, .294, .380, .506]
# june_27_alpha = [0.039, 0.056, 0.075, 0.096, 0.120, 0.148, 0.183, 0.222, 0.271, 0.335, 0.415, 0.524, 0.682, 0.909]

def exp(x, a, b):
    return a*np.exp(b*x)
def uexp(x, a, b):
    return a*unumpy.exp(b*x)

ujune_27_x = unumpy.uarray(june_27_x, june_27_x_err)
params, covar = curve_fit(exp, june_20_x, june_20_alpha)
perr = np.sqrt(np.diag(covar))
uparams = unumpy.uarray(params, perr)
ujune_20_fit = uexp(ujune_27_x, *uparams)

june_20_fitn = np.array([ r.n for r in ujune_20_fit ])
june_20_fits = np.array([ r.s for r in ujune_20_fit ])

# ratio = udata_y / udata10fit_y

ujune_20_alpha = unumpy.uarray(june_20_alpha, june_20_alpha_err)
ujune_27_alpha = unumpy.uarray(june_27_alpha, june_27_alpha_err)

# ratio = ujune_27_alpha/ujune_20_fit
ratio = ujune_27_alpha/ujune_20_alpha
ration = [ r.n for r in ratio ]
ratios = [ r.s for r in ratio ]

fig,ax = plt.subplots()
fig.tight_layout()
plt.rc("font", size=12)
# x_label = "Over Voltage [V]"
x_label = "Bias Voltage [V]"
y_label = "Alpha Pulse Amplitude [V]"
plt.errorbar(
    june_20_x,
    june_20_alpha,
    xerr=june_27_x_err,
    yerr=june_20_alpha_err,
    markersize=10,
    fmt=".",
    color='tab:blue',
    label='Before Baking'
)
plt.errorbar(
    june_27_x,
    june_27_alpha,
    xerr=june_27_x_err,
    yerr=june_27_alpha_err,
    markersize=10,
    fmt=".",
    color='tab:purple',
    label='After Baking'
)
# x = np.linspace(1, 9, 500)
# ax.plot(x, exp(x, *params))
# ax.fill_between(june_27_x, june_20_fitn - june_20_fits, june_20_fitn + june_20_fits, alpha=.3)
axr = ax.twinx()
axr.errorbar(
    june_27_x,
    ration,
    xerr=june_27_x_err,
    yerr=ratios,
    markersize=10,
    fmt=".",
    color='tab:green',
    label='Ratio'
)
# plt.errorbar(27.3,0,xerr=.103,fmt='.',color='purple')
# plt.errorbar(26.9,0,xerr=.198,fmt='.',color='blue')
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
axr.set_ylabel('Ratio')
axr.set_ylim(0,3)
ax.set_ylim(0,2)
textstr = f"Date: June 20th and 27th 2024\n"
textstr += f"Condition: LXe\n"
textstr += f"RTD4: 170 [K]\n"
textstr += f"Ratio: {ratio.mean():.3f}"
ax.grid(True)
ax.legend(loc="upper left")
axr.legend(loc="upper right")
props = dict(boxstyle="round", facecolor='tab:purple', alpha=0.4)
fig.text(0.1, 0.45, textstr, fontsize=10, verticalalignment="top", bbox=props)
# fig.text(0.1, 0.13, "Breakdown Voltages:", fontsize=8, verticalalignment="top")
plt.show()

#%%
storage_path = '/run/media/ed/My Passport/ed/'

p = dill.Pickler(open(storage_path+'june-20-2024-campaign-alpha.pickle', 'wb'))
p.fast = True
p.dump(campaign_alpha)
# campaign_alpha = p.load()

p = dill.Unpickler(open(storage_path+"CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()


v_bd = 26.93 # from June 20 SPE
v_bd_err = 0.198
alpha_data = Alpha_data(campaign_alpha, invC_alpha_1us, invC_alpha_err_1us, spe, v_bd, v_bd_err)

p = dill.Pickler(open("june-20-2024-alpha_data.pickle","wb"))
p.fast = True
p.dump(alpha_data)

alpha_data.analyze_alpha()

#%%
alpha_data.plot_alpha(x='OV')

alpha_data.plot_num_det_photons()

##%% values based on Wesley's APS slides
N = 5.49/(19.6E-6)
PTE =  0.001782
alpha_data.plot_PDE(N*PTE, out_file='june-20-2024-alpha-amp-lxe.csv')
