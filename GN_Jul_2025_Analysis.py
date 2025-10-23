#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPE: Created on Fri May 9 16:00:00 2025

Analysis for July 2nd, 3rd, 8th 2025 data for LED calibration,
and SPE data on July 8th

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
from src.MeasurementInfo import MeasurementInfo, get_files
from src.ProcessHistograms import ProcessHist
from src.ProcessWaveforms import ProcessWaveforms, proc_wf_runner
from AnalyzePDE import SPE_data
from multiprocessing import Pool
plt.style.use('misc/nexo.mplstyle')

# import scienceplots
# plt.style.use('science')

plt.style.use('misc/science.mplstyle')


#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
# change path name to folder with data files
path ='data/20250702_SPE_GN/'

### LED Calibration

# July 2nd
led_voltages = [2.5, 2.52, 2.54, 2.56, 2.58, 2.60]
files = ['LED_sweep_250.hdf5',  'LED_sweep_252.hdf5',  'LED_sweep_254.hdf5',
         'LED_sweep_256.hdf5',  'LED_sweep_258.hdf5',  'LED_sweep_260.hdf5', ]
files = [path+file for file in files]

# July 3rd
led_voltages += [2.62, 2.64, 2.66, 2.68, 2.70]
files += ['Run_1751536179.hdf5',  'Run_1751536870.hdf5',  'Run_1751537655.hdf5',
          'Run_1751538300.hdf5', 'Run_1751539219.hdf5']
led_voltages += [2.58, 2.60, 2.61] # repeated
files += ['Run_1751533994.hdf5', 'Run_1751534732.hdf5', 'Run_1751549975.hdf5']

# led_voltages += [0.0]
# files += ['LED_sweep_000.hdf5',]

led_voltages += [2.55, 2.57]
files += ['Run_1751553184.hdf5', 'Run_1751554225.hdf5']
files = [path+file for file in files]

# July 8th
path = 'data/20250708_SPE_GN/LED Calibration/'
led_voltages = [2.55, 2.57, 2.59, 2.60, 2.61, 2.58]
files8th = ['Run_1751972009.hdf5', 'Run_1751972730.hdf5', 'Run_1751973782.hdf5',
            'Run_1751974647.hdf5', 'Run_1751975322.hdf5', 'Run_1751977312.hdf5']
files = [path+file for file in files8th]

# loop over all files
runs = []
for file in range(len(files)):
    run_spe = ProcessWaveforms([files[file]], do_filter=True,
                               upper_limit=.2, prominence=.005,
                               baseline_correct=True, poly_correct=True)
    runs.append(run_spe)

measurments = []
all_amps = []
led_amps = []
dark_amps = []
for run in runs:
    m = MeasurementInfo('GN', 168, run)
    m.plot_histogram() # Plot preliminary histogram
    all_amps.append(m.avg_amp['all'])
    led_amps.append(m.avg_amp['led'])
    dark_amps.append(m.avg_amp['dark'])
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
pe_center_guesses = [7.9, 7.9, 7.9,7.9, 7.9, 7.9,]
num_pes = [4, 4, 4, 4, 4, 4]
centers_guesses = []
for pe_guess, num_pe in zip(pe_center_guesses, num_pes):
    centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
# outpath = '/home/ed/pictures/0vbb/20250708/20250720_'
campaign_spe = []
cutoff = (.005,.07)
for i in range(len(measurments)):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         # cutoff=cutoffs[i], centers=centers_guesses[i],
                         cutoff=cutoff, centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    # wp_spe.plot_peak_histogram(log_scale=False, savefig=False,)
    # wp_spe.plot_spe(savefig=False, fit_origin=True)
    wp_spe.plot_led_only_hist()
    campaign_spe.append(wp_spe)



# runs = runs[:-1]

# led_ratios = [.06, .12, .19, .33, .42, .16]
# led_amps = [.0169, .0168, .0187, .0201, .0203, .018]
# led_amp_errs = []

led_ratios = []
led_amps = []
led_amp_errs = []
for run, voltage in zip(runs, led_voltages):
    (r, a, a_err) = run.plot_led_dark_hists(voltage)
    # (r, a, a_err) = run.plot_led_dark_hists(led_voltages[5])
    led_ratios.append(r)
    led_amps.append(a)
    led_amp_errs.append(a_err)

# Fit Gauss to peaks and find the best value for peak location
outpath = '/home/ed/pictures/0vbb/20250703/20250707_'
campaign_spe = []
cutoffs = (.005,.065)
center_guesses = [.0076, .015, .0225, .03]
for i in range(len(runs)): #
    info_spe = MeasurementInfo()
    info_spe.condition = 'GN'
    info_spe.date = runs[i].date
    info_spe.temperature = 168
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = ProcessHist(info_spe, run_info_self=runs[i],
                         baseline_correct=True,
                         # peaks='all',
                         peaks='dark',
                         # peaks='LED',
                         # cutoff=cutoffs[i], centers=centers_guesses[i],
                         cutoff=cutoffs, centers=center_guesses,
                         background_linear=False,
                         peak_range=(1,4))
                         # peak_range=(1,3))
    wp_spe.process_spe()
    # wp_spe.plot_led_dark_hists(led_voltages[i])
    wp_spe.plot_peak_histograms(log_scale=False, savefig=False,
                                path=outpath+'262_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_spe_with_origin(savefig=False,
    #                             path=outpath+'spe_'+str(info_spe.bias)+'.png')
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)
    print(info_spe.bias)


led_voltages_2nd = led_voltages[:6]
led_ratios_2nd   = led_ratios[:6]
led_amps_2nd     = led_amps[:6]
led_amp_errs_2nd = led_amp_errs[:6]
led_voltages_3rd = led_voltages[6:]
led_ratios_3rd   = led_ratios[6:]
led_amps_3rd     = led_amps[6:]
led_amp_errs_3rd = led_amp_errs[6:]

led_voltages_8th = [2.55, 2.57, 2.59, 2.6, 2.61, 2.58]
led_ratios_8th   = [0.06403269754768393,
                    0.12435072486239243,
                    0.18611353711790393,
                    0.33009138928481596,
                    0.4237142857142857,
                    0.15861364058725144]
led_amps_8th     = [0.01690935376820229,
                    0.01679126773669308,
                    0.018716437360274334,
                    0.02013707421723995,
                    0.020334641146336523,
                    0.018048086585203238]
led_amp_errs_8th = [0.00016347747549932848,
                    0.0001375147005110337,
                    0.00015242622345286466,
                    0.00015110160468264207,
                    0.0001359925452321009,
                    0.00015510332359456886]

# plt.plot(led_voltages, led_ratios, 'ro')
plt.plot(led_voltages_2nd, led_ratios_2nd, 'o', label='July 2nd')
plt.plot(led_voltages_3rd, led_ratios_3rd, 'o', label='July 3rd')
plt.plot(led_voltages_8th, led_ratios_8th, 'o', label='July 8th')
plt.xlabel('LED Voltage [V]', loc='right')
plt.ylabel('LED Subtracted Ratio', loc='top')
plt.xlim(2.46, 2.72)
plt.grid(True)
plt.tight_layout()
plt.yscale('log')
plt.legend()
plt.show()

# plt.plot(led_voltages, led_amps, 'bo')
# plt.errorbar(led_voltages_2nd, led_amps_2nd, fmt='o', yerr=led_amp_errs_2nd, label='July 2nd')
# plt.errorbar(led_voltages_3rd, led_amps_3rd, fmt='o', yerr=led_amp_errs_3rd, label='July 3rd')
# plt.errorbar(led_voltages_8th, led_amps_8th, fmt='o', yerr=led_amp_errs_8th, label='July 8th')

plt.errorbar(led_voltages, [v.n for v in all_amps],  fmt='o', yerr=[v.s for v in all_amps],  label='All peaks')
plt.errorbar(led_voltages, [v.n for v in led_amps],  fmt='o', yerr=[v.s for v in led_amps],  label='LED peaks')
plt.errorbar(led_voltages, [v.n for v in dark_amps], fmt='o', yerr=[v.s for v in dark_amps], label='Dark peaks')
plt.xlabel('LED Voltage [V]', loc='right')
plt.ylabel('Average Amplitude [V]', loc='top')
plt.xlim(2.44, 2.68)
# plt.xlim(2.5, 2.65)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


led_ratios = [0.007936507936507936,
 0.044271279328652165,
 0.07391347196215631,
 0.09641924654979485,
 0.28545957029508784,
 0.5883275261324041,
 2.0020719073735527,
 3.645070090705619,
 7.9351679104477615,
 16.124351162265853,
 31.806622659885292,
 0.4059690758719885,
 0.8313055685540477,
 1.3477394445136381,
 0.12537799488253082,
 0.30833721471821146 ]

led_amps = [0.016537188743721856,
 0.015522901652222587,
 0.015549146450591434,
 0.01525151233910552,
 0.015395386045768353,
 0.015965077697566065,
 0.02118932304718828,
 0.021861162333396654,
 0.023147666501964915,
 0.02403925342059092,
 0.02518486618238924,
 0.018502925934388004,
 0.019159652927635173,
 0.020506307781988992,
 0.017504360788011892,
 0.017787720842381883,
            ]
led_amp_errs = [0.00016672307564321773,
 0.00014016369439547827,
 0.00015270590494199593,
 0.00014590065612205414,
 0.00013826486308709464,
 0.00013059148533650799,
 0.00014404346824012802,
 0.00012123155054414894,
 9.3273366874001e-05,
 6.933936075974693e-05,
 5.1138332530415764e-05,
 0.00017040429825406766,
 0.0001563710691029544,
 0.00015335083343247997,
 0.00017393814856070223,
 0.00016309165984111531, ]



### July 8th SPE

#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = 0.011547
invC_err_filter = 0.000098
# change path name to folder with data files
path = '../SiPM_Data/20250708_SPE_GN/'
files = ['Run_1751980622.hdf5', 'Run_1751981408.hdf5', 'Run_1751982057.hdf5',
         'Run_1751982744.hdf5', 'Run_1751983369.hdf5', 'Run_1751983981.hdf5',
         'Run_1751984720.hdf5', 'Run_1751985574.hdf5', 'Run_1751986369.hdf5']

file_pre_bd = 'Run_1751987061.hdf5'
run_spe_pre_bd = RunInfo([path+file_pre_bd], specifyAcquisition=False, do_filter=True,
                         is_solicit=True, upper_limit=0.063, baseline_correct=True)

# loop over all files
runs = []
proms = [0.0054, 0.0053, 0.0056,
         0.0054, 0.0053, 0.0055,
         0.0056, 0.0056, 0.0055]
for file in range(len(files)):
    run_spe = RunInfo([path+files[file]], specifyAcquisition=False, do_filter=True,
                      upper_limit=.4, baseline_correct=True,
                      prominence=proms[file],
                      poly_correct=True, is_led=True)
    runs.append(run_spe)

# Preliminary histogram
for run in runs:
    run.plot_hists()

# Fit Gauss to peaks and find the best value for peak location
pe_center_guesses = [5.7, 4.8, 8.5, 6.1, 5.6, 7.1, 8.5, 7.7, 6.8]
num_pes = [4, 3, 5, 4, 3, 5, 5, 4, 5]
centers_guesses = []
for pe_guess, num_pe in zip(pe_center_guesses, num_pes):
    centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
outpath = '/home/ed/pictures/0vbb/20250708/20250720_'
campaign_spe = []
cutoff = (.004,.07)
# cutoff = (.002,.07)
# center_guesses = [.0076, .015, .0225, .03]
for i in range(len(runs)):
# if i := 6:
    info_spe = MeasurementInfo()
    info_spe.condition = 'GN'
    info_spe.date = runs[i].date
    info_spe.temperature = 168
    info_spe.bias = runs[i].bias
    info_spe.baseline_numbins = 50
    info_spe.peaks_numbins = 200
    info_spe.data_type = 'h5'
    wp_spe = ProcessHist(info_spe, run_info_self=runs[i],
                         run_info_pre_bd=run_spe_pre_bd,
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         # cutoff=cutoffs[i], centers=centers_guesses[i],
                         cutoff=cutoff, centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
                         # peak_range=(1,3))
    wp_spe.process_spe()
    # wp_spe.plot_led_dark_hists(led_voltages[i])
    wp_spe.plot_peak_histogram(log_scale=False, savefig=True,
                               path=outpath+f'hist_{info_spe.bias}.png')
    wp_spe.plot_spe(savefig=True, path=outpath+'spe_'+str(info_spe.bias)+'.png',
                    fit_origin=True)
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)
    print(info_spe.bias)


for i in range(len(campaign_spe)):
    campaign_spe[i].plot_both_histograms()

#%% plot linear fit to the breakdown voltage
spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()

### July 8th

condition = 'GN'
temperature = 168
#1us, 10x gain, filter on - values from Noahs calibration report
# invC_filter = 0.011547
# invC_err_filter = 0.000098
invC_filter = 0.010040
invC_err_filter = 0.000500
path = '../data/20250708_SPE_GN_168K/'
files, biases, run_spe_pre_bd = get_files(path)

# Loop over all files and find peaks
params = dict(
        do_filter=True,
        baseline_correct=True,
        poly_correct=True,
        upper_limit=-1,
        prominence=.003,
        )
runs = proc_wf_runner(files, params)

def proc(file):
    return ProcessWaveforms([path+file], do_filter=True,
                            baseline_correct=True, poly_correct=True,
                            upper_limit=-1, prominence=.003)
with Pool(len(files)+1) as p:
    runs = p.map(proc, iter(files))

measurments = []
for run in runs:
    m = MeasurementInfo(condition, temperature, run, run_spe_pre_bd)
    # m.plot_histogram() # Plot preliminary histogram
    measurments.append(m)

# Fit Gauss to peaks and find the best value for peak location
pe_center    = [5.1, 5.1, 5.8, 6.8, 7.0, 7.1, 7.6, 8.4, 8.8]
num_pes      = [3,   4,   4,   4,   4,   5,   5,   5,   5]
lower_cutoff = [5.1, 5.2, 5.3, 4.9, 5.3, 5.6, 5.4, 5.1, 5.2]
savefig = False
# centers_guesses = []
# for pe_guess, num_pe in zip(pe_center, num_pes):
#     centers_guesses.append([pe_guess * peak / 1000 for peak in range(1,num_pe+1)])
outpath = '/home/evanbruggen_umass_edu/0vbb/20250708/20250916_'
# outpath = None
campaign_spe = []
for i in range(len(measurments)):
    wp_spe = ProcessHist(measurments[i],
                         baseline_correct=True,
                         peaks='all',
                         # peaks='dark',
                         # peaks='LED',
                         cutoff=(lower_cutoff[i]/1000, .1),
                         first_pe=pe_center[i],
                         # centers=centers_guesses[i],
                         background_linear=False,
                         peak_range=(1,num_pes[i]))
    wp_spe.process_spe()
    wp_spe.plot_peak_histogram(log_scale=True,
                               savefig=savefig, path=outpath+f"hist_{measurments[i].bias}.png")
    wp_spe.plot_spe(fit_origin=False,
                    savefig=savefig, path=outpath+f"spe_{measurments[i].bias}.png")
    wp_spe.plot_ca(outpath+f"ca_{measurments[i].bias}.png" if savefig else None)
    # wp_spe.plot_led_only_hist()
    # wp_spe.plot_both_histograms(with_fit=True, savefig=False, path = outpath+'silicon_baseline_'+str(info_spe.bias)+'.png')
    campaign_spe.append(wp_spe)



spe = SPE_data(campaign_spe, invC_filter, invC_err_filter, filtered=True)

spe.plot_spe(in_ov=False, absolute=True)
spe.plot_spe(in_ov=False, absolute=False)

spe.plot_spe(in_ov=True, absolute=True)
spe.plot_spe(in_ov=True, absolute=False)

spe.plot_CA()
