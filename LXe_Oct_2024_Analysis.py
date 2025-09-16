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
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy
plt.style.use('misc/nexo.mplstyle')

#%% ALPHA - 1us
#1us, no gain
invC_alpha_1us = 0.001142
invC_alpha_err_1us = 0.0000021
file_path = 'oct-2024/' # folder with H5 data files
files = [ 'Run_1729186580.hdf5', 'Run_1729186383.hdf5', 'Run_1729186177.hdf5',
          'Run_1729185903.hdf5', 'Run_1729185648.hdf5', 'Run_1729185416.hdf5',
          'Run_1729185066.hdf5', 'Run_1729184763.hdf5']
proms = [ .02, .02, .01, .015, .02, .01, .02, .02]
upperlim = [ 1, 1,1,1,1,1, 1, 1]
# 'Run_1729186821.hdf5',
files += [ 'Run_1729187422.hdf5', 'Run_1729187628.hdf5',
          'Run_1729188168.hdf5', 'Run_1729188561.hdf5', 'Run_1729188879.hdf5' ]
proms += [ .02, .01,
          .08, .05, .01]
upperlim += [ 1, 1,
             1, 1, 1]
runs_alpha_1us = []
i = 0
for file in range(len(files)):
    run_alpha_1us = RunInfo([file_path+files[file+i]], do_filter=False,
                            upper_limit=upperlim[file+i], baseline_correct=True,
                            prominence=proms[file+i], plot_waveforms=False,)
    # run_alpha_1us.plot_hists('','')
    runs_alpha_1us.append(run_alpha_1us)
biases = [run.bias for run in runs_alpha_1us] # get all the bias voltages from RunInfo (enter manually if metadata is wrong)

runs_alpha_1us[0].plot_hists('169.9', '.1')
# runs_alpha_1us[0].plot_peak_waveform_hist()

#%%
campaign_alpha = []
runs_alpha = runs_alpha_1us #change as needed
bins = [40, 39, 42]
bins += [41, 41, 44]
bins += [40, 39]
bins += [40, 40]
bins += [42, 39, 40]
for n in range(len(runs_alpha)):
# if n := 1:
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

#%%
storage_path = '/run/media/ed/My Passport/ed/'

p = dill.Pickler(open(storage_path+'june-20-2024-campaign-alpha.pickle', 'wb'))
p.fast = True
p.dump(campaign_alpha)
# campaign_alpha = p.load()

p = dill.Unpickler(open(storage_path+"CA-july-12.pickle","rb"))
p.fast = True
spe = p.load()

invC_alpha = 1143.98e-6
invC_alpha_err = 0.12e-6
invC_spe = 11404e-6
invC_spe_err = 11e-6

v_bd = 27.08 # from Oct 17 2024 SPE
v_bd_err = 0.098

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
alpha_data.plot_PDE(N*PTE, out_file='2024Oct17_Alpha.csv')

data = 'amps'
dfr = pd.read_csv('2024Aug07_Alpha.csv').sort_values('ov').tail(-1)#.tail(9)
data_x = dfr['ov']
data_x_err = dfr['ov error']
data_y = dfr[data]
data_y_err = dfr[data+' error']

df10 = pd.read_csv('2024Oct17_Alpha.csv').sort_values('ov')#.head(10).tail(9)
d10_x = df10['ov']
d10_x_err = df10['ov error']
d10_y = df10[data]
d10_y_err = df10[data+' error']

# ratio = data_y / d10_y

udata_x = unumpy.uarray(data_x, data_x_err)
udata_y = unumpy.uarray(data_y, data_y_err)
udata10_y = unumpy.uarray(d10_y, d10_y_err)
# udata_x = np.array([ ufloat(v, e) for v, e in zip(data_x, data_x_err)])
# udata_y = np.array([ ufloat(v, e) for v, e in zip(data_y, data_y_err)])
# udata10_y = np.array([ ufloat(v, e) for v, e in zip(d10_y, d10_y_err)])

# udata10fit_y = np.array([ ufloat(v, e) for v, e in zip(exp(data_x, *params), d10_y_err)])

def exp(x, a, b):
    return a*np.exp(b*x)
def uexp(x, a, b):
    return a*unumpy.exp(b*x)

params, covar = curve_fit(exp, d10_x, d10_y) #, sigma=d10_y_err
perr = np.sqrt(np.diag(covar))
uparams = unumpy.uarray(params, perr)
udata10fit_y = uexp(udata_x, *uparams)

ratio = udata_y / udata10fit_y
ration = [ r.n for r in ratio ]
ratios = [ r.s for r in ratio ]
yfitn = np.array([ r.n for r in udata10fit_y ])
yfits = np.array([ r.s for r in udata10fit_y ])

color = 'tab:blue'
fig,ax = plt.subplots()
fig.tight_layout()
# plt.rc('font', size=22)
x = np.linspace(3, 9, 100)
ax.plot(x, exp(x, *params))
# ax.fill_between(data_x[:7], yfitn[:7] - yfits[:7], yfitn[:7] + yfits[:7], alpha=.3)
ax.fill_between(data_x, yfitn - yfits, yfitn + yfits, alpha=.3)
ax.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 8, fmt = '.',
            color = 'tab:purple', label='4 Tall Silicon Reflector')
ax.errorbar(d10_x, d10_y, xerr = d10_x_err, yerr = d10_y_err, markersize = 8, fmt = '.',
             color = 'tab:blue', label='No Reflector')
# plt.errorbar(data_x, ration, xerr=data_x_err, yerr=ratios, markersize = 8, fmt = '.', color =
#              'tab:green', label='Ratio (Average: 2.238±.00058)')
axr = ax.twinx()
axr.errorbar(data_x, ration, xerr = data_x_err, yerr=ratios, markersize=0, fmt='.',
             color='tab:green', label=f'Ratio {ratio.mean()}')
axr.set_ylim(0,3)
ax.set_ylim(0,1)
# axr.fill_between(data_x, ration - ratios, ration + ratios, alpha=.3)
ax.set_xlabel('Overvoltage [V]')
# plt.ylabel('Number of Detected Photons')
ax.set_ylabel('Alpha Amplitude [V]')
axr.set_ylabel('Ratio')
# textstr = f'Date: {self.campaign[0].info.date}\n'
textstr = f'Silicon Reflector\n'
textstr += f'Condition: LXe\n'
textstr += f'RTD4: 167 [K]'
props = dict(boxstyle='round', facecolor=color, alpha=0.4)
# fig.text(.5, .5, textstr, fontsize=10,
#         verticalalignment='top', bbox=props)
ax.legend(loc='upper left')
axr.legend(loc='upper right')
# plt.legend()
plt.show()

PTEs = ufloat(.00335, .0000596)
PTEd = ufloat(.00198, .0000445)
PTEn = ufloat(.00136, .0000368)
X = PTEs/PTEn
Y = PTEd/PTEn
# a = (np.mean(ratio) - Y)/(X-Y)
a = (ufloat(2.11,.103) - Y)/(X-Y)

# plt.errorbar(data_x, ration, xerr=data_x_err, yerr=ratios, markersize = 8, fmt = '.', color = 'tab:green', label='Ratio')
# plt.xlabel('Overvoltage [V]')
# plt.ylabel('Alpha Pulse Ratio')
# # textstr = f'Date: {self.campaign[0].info.date}\n'
# textstr = f'Silicon Reflector\n'
# textstr += f'Condition: LXe\n'
# # textstr += f'Ratio Average: 18.228 ±\n'
# textstr += f'RTD4: 167 [K]'
# props = dict(boxstyle='round', facecolor=color, alpha=0.4)
# fig.text(0.75, 0.2, textstr, fontsize=10,
#         verticalalignment='top', bbox=props)
# plt.legend()
# plt.show()
#
