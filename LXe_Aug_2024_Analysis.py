# -*- coding: utf-8 -*-
"""
Created on Thu June 20 2024

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
from uncertainties import unumpy

plt.style.use('misc/nexo.mplstyle')

#%% ALPHA - 1us
#1us, no gain
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
file_path = 'aug-2024/' # folder with H5 data files
files = ['Run_1723059892.hdf5', 'Run_1723060554.hdf5']
proms = [ .01, .04]
upperlim = [ .5, 1]
files += ['Run_1723061041.hdf5', 'Run_1723060789.hdf5']
proms += [ .01, .04]
upperlim += [ 3, 3]
files += ['Run_1723061758.hdf5', 'Run_1723061416.hdf5']
proms += [ .04, .01]
upperlim += [ 3, 1]
files  += [ 'Run_1723061965.hdf5', 'Run_1723062362.hdf5', 'Run_1723062627.hdf5', 'Run_1723062830.hdf5', 'Run_1723063128.hdf5' ]
proms += [ .04, .04, .04, .04, .04]
upperlim += [ 3, 1, 1, 3, 1]
files += [ 'Run_1723063412.hdf5', 'Run_1723063659.hdf5', 'Run_1723063892.hdf5' ]
proms += [ .2, .01, .04]
upperlim += [ 3, .4, 1]
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
bins = [40, 40]
bins += [40, 40]
bins += [40, 40]
bins += [42, 40, 40, 38, 40]
bins += [40, 36, 28]
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

# v_bd = 27.57
# v_bd_err = 0.159
bias_vals = []
bias_err = []
alpha_vals = []
alpha_err = []
for wp in campaign_alpha:
    bias_vals.append(wp.info.bias)
    # self.bias_err.append(0.0025 * wp.info.bias + 0.015)
    bias_err.append(0.005)
    curr_alpha = wp.get_alpha()
    alpha_vals.append(curr_alpha[0])
    alpha_err.append(curr_alpha[1])
# ov = []
# ov_err = []
# for b, db in zip(bias_vals, bias_err):
#     curr_ov = b - v_bd
#     curr_ov_err = np.sqrt(db * db + v_bd_err * v_bd_err)
#     ov.append(curr_ov)
#     ov_err.append(curr_ov_err)

data = {
    # 'ov': ov, 'ov error': ov_err,
    'bias': bias_vals, 'bias error': bias_err,
    'amps': alpha_vals, 'amps error': alpha_err,
}
df = pd.DataFrame(data)
df.to_csv('2024Aug07_Alpha.csv')

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

# comparison plotting
xvar = 'Bias Voltage [V]'
ndf = pd.read_csv('2024Aug07_Alpha.csv').sort_values('bias').head(10)
none_x = ndf['bias']
none_x_err = ndf['bias error']
none_alpha = ndf['amps']
none_alpha_err = ndf['amps error']

rdf = pd.read_csv('2024July09_Alpha.csv').sort_values(xvar).head(8)
refl_x = rdf[xvar]
refl_x_err = rdf['Bias Voltage error [V]']
refl_alpha = rdf['Alpha Pulse Amplitude [V]']
refl_alpha_err = rdf['Alpha Pulse Amplitude error [V]']

def quad(x, a, c):
    return a*x*x +  c
def exp(x, a, b):
    return a*np.exp(b*x)
def uexp(x, a, b):
    return a*unumpy.exp(b*x)
def expc(x, a, b, c):
    return a*np.exp(b*x) + c
def uexpc(x, a, b, c):
    return a*unumpy.exp(b*x) + c

urefl_x = unumpy.uarray(refl_x, refl_x_err)
params, covar = curve_fit(exp, none_x, none_alpha)
perr = np.sqrt(np.diag(covar))
uparams = unumpy.uarray(params, perr)
unone_fit = uexp(urefl_x, *uparams)

none_fitn = np.array([ r.n for r in unone_fit ])
none_fits = np.array([ r.s for r in unone_fit ])
unone_alpha = unumpy.uarray(none_alpha, none_alpha_err)
urefl_alpha = unumpy.uarray(refl_alpha, refl_alpha_err)

ratio = urefl_alpha/unone_fit
# ratio = urefl_alpha/unone_alpha
ration = [ r.n for r in ratio ]
ratios = [ r.s for r in ratio ]

fig,ax = plt.subplots()
fig.tight_layout()
plt.rc("font", size=12)
# x_label = "Over Voltage [V]"
x_label = "Bias Voltage [V]"
y_label = "Alpha Pulse Amplitude [V]"
plt.errorbar(
    none_x,
    none_alpha,
    xerr=none_x_err,
    yerr=none_alpha_err,
    markersize=10,
    fmt=".",
    color='tab:blue',
    label='No Reflector'
)
plt.errorbar(
    refl_x,
    refl_alpha,
    xerr=refl_x_err,
    yerr=refl_alpha_err,
    markersize=10,
    fmt=".",
    color='tab:purple',
    label='4 Tall Si Reflector'
)
x = np.linspace(30, 34, 100)
ax.plot(x, quad(x, *params), label='a*x^2+b')
ax.fill_between(refl_x, none_fitn - none_fits, none_fitn + none_fits, alpha=.3)
axr = ax.twinx()
axr.errorbar(
    refl_x,
    ration,
    xerr=refl_x_err,
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
axr.set_ylim(0,4)
# ax.set_ylim(0,2.5)
textstr = f"Date: July 7th and August 9th 2024\n"
textstr += f"Condition: LXe\n"
textstr += f"RTD4: 170 [K]\n"
textstr += f"Ratio: {ratio.mean():.3f}"
ax.grid(True)
ax.legend(loc="upper left")
axr.legend(loc="upper right")
props = dict(boxstyle="round", facecolor='tab:purple', alpha=0.4)
fig.text(0.1, 0.7, textstr, fontsize=10, verticalalignment="top", bbox=props)
# fig.text(0.1, 0.13, "Breakdown Voltages:", fontsize=8, verticalalignment="top")
plt.show()

from uncertainties import ufloat
PTEs = ufloat(.0025563,  5.1e-06)
PTEd = ufloat(.0016941, 4.1e-06)
PTEn = ufloat(.0012682, 3.6e-06)
X = PTEs/PTEn
Y = PTEd/PTEn
a = (ufloat(1.34, .06) - Y)/(X-Y)

a = (np.mean(ratio) - Y)/(X-Y)
