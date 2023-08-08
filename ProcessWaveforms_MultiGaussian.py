# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:00:35 2022

@author: lab-341
"""
import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm
from scipy.stats import mode
from scipy.stats import sem
from scipy.optimize import curve_fit
from MeasurementInfo import MeasurementInfo
import glob
from scipy.fft import fft, fftfreq
from lmfit.models import LinearModel, GaussianModel, ExponentialModel

def get_waveform(w):
    time = []
    amp = []

    f = open(w, 'r')
    
    metadata = {}
    data = {}
    
    header = True
    for x in f:
        line = x.split('\t')
        if header:
            if line[0] == 'Time (s)':
                header = False
            elif len(line) < 10:
                continue
            else:
                metadata[line[0]] = line[1]
        else:
            t = float(line[0]) * 1E6
            a = float(line[1])
            time.append(t)
            amp.append(a)
    f.close()
    return (time, amp)

# REED DID THIS <3
def get_peaks(waveform_dir, peak_search_params):
    waveform_filenames = glob.glob(waveform_dir + 'w*.txt')
    all_peaks = []
    for idx, w in enumerate(waveform_filenames):
        if idx % 100 == 0:
            print(idx)
        time, amp = get_waveform(w)

        peaks, props = signal.find_peaks(amp, **peak_search_params)
        for peak in peaks:
            all_peaks.append(amp[peak])
    return all_peaks

def get_peak_waveforms(waveform_dir, num = -1):
    #wfs = fnmatch.filter(os.listdir(filepath), 'w*')
#   read in solicited trigger waveforms
    waveform_filenames = glob.glob(waveform_dir + 'w*.txt')
    values = []
    times = []
    num_w = 0
#   search each waveform for pulses, reject those with any
    if num > 0:
        waveform_filenames = waveform_filenames[:num]
    for idx, w in enumerate(waveform_filenames):
        if idx % 100 == 0:
            print(idx)
        time, amp = get_waveform(w)
        num_w += 1
        values += amp
        times += time
    return values, times, num_w

def get_baseline(waveform_dir, peak_search_params):
    #wfs = fnmatch.filter(os.listdir(filepath), 'w*')
#   read in solicited trigger waveforms
    waveform_filenames = glob.glob(waveform_dir + 'w*.txt')
    values = []
    times = []
    num_w = 0
#   search each waveform for pulses, reject those with any
    for idx, w in enumerate(waveform_filenames):
        if idx % 100 == 0:
            print(idx)
        time, amp = get_waveform(w)
        peaks, props = signal.find_peaks(amp, **peak_search_params)
#   aggregate all pulseless data
        if len(amp) < 1:
            continue
        if len(peaks) == 0 and np.amin(amp) > -.25:
            num_w += 1
            values += amp[300:-300]
            times += time[300:-300]
    return values, times, num_w

def save_baseline_csv(waveform_dir, savedir, peak_search_params):
    waveform_data, waveform_times, _ = get_baseline(waveform_dir, peak_search_params)
    data = {'waveform data': waveform_data}
    df = pd.DataFrame(data)
    df.to_csv(savedir)

def save_peaks_csv(waveform_dir, savedir, peak_search_params):
    peaks = get_peaks(waveform_dir, peak_search_params)
    data = {'peaks': peaks}
    df = pd.DataFrame(data)
    df.to_csv(savedir)

def read_data_csv(filename):
    df = pd.read_csv(filename)
    return df

def Gauss(x, A, B, C):
    y = A * np.exp(-(x - B) ** 2 / (2 * C * C))
    return y

def fit_gauss(values, range_low, range_high):
    histogram = np.histogram(values, bins = 40)
    counts = histogram[0]
    bins = histogram[1]
    centers = (bins[1:] + bins[:-1])/2
    model = lm.models.GaussianModel()
    params = model.make_params(amplitude=max(counts), center=np.mean(values), sigma=np.std(values))
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1/counts))
    return res

def fit_baseline_gauss(values, binnum = 50, alpha = False):
    f_range = {}
    if alpha:
        f_range['low'] = -0.0005
    #    f_range['low'] = 0.0
        f_range['high'] = 0.0045
    #    f_range['high'] = 0.003
        f_range['center'] = (f_range['high'] + f_range['low']) / 2.0
    else:
        f_range['center'] = np.mean(values)
        std_guess = np.std(values)
        f_range['low'] = f_range['center'] - 2.0 * std_guess
        f_range['high'] = f_range['center'] + 2.0 * std_guess
    bin_density = float(binnum) / (np.amax(values) - np.amin(values))
    new_binnum = int(bin_density * (f_range['high'] - f_range['low']))
    limit_values = values[(values >= f_range['low']) & (values <= f_range['high'])]
    curr_hist = np.histogram(limit_values, bins = new_binnum)
    # plt.hist(values, bins= binnum)
    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1])/2
  
    model = lm.models.GaussianModel()
    params = model.make_params(amplitude=np.amax(counts), center=np.mean(limit_values), sigma=np.std(values))
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1/counts))
    # plt.step(centers, counts, where = 'mid')
    # plt.plot(centers, res.eval(params = res.params, x = centers), '--')
    f_range['fit'] = res
#    return {'center': np.mean(values), 'low': np.amin(values), 'high': np.amax(values), 'fit': res}
    return f_range


def fit_peaks_multigauss(values, baseline_loc, baseline_width, binnum=400, range_low = 0, range_high = 2, center = 0.1, offset_num = 0, peak_range = (0,4)):
    
    low_peak = peak_range[0]
    high_peak = peak_range[1]
    
    fit_range = [] #defines estimated center and the locations of the left and right "edges" of the "finger"
    fit_range.append({'low': range_low, 'high': range_high})
    curr_peak_data = values[(values >= range_low) & (values <= range_high)]
    # binnum = int(np.sqrt(len(curr_peak_data))) # bin number = square root of number of data points
    curr_hist = np.histogram(curr_peak_data, bins = binnum)
    # plt.hist(curr_peak_data, bins = binnum)
    counts = curr_hist[0] 
    
    print('bins: ' + str(binnum))

    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1])/2
    
    
    
    for peak in range(low_peak, high_peak + 1):
        if peak == low_peak:
            model = GaussianModel(prefix='g' + str(low_peak) + '_')
        else:
            model = model + GaussianModel(prefix='g' + str(peak) + '_')
        
    model = model + LinearModel(prefix= 'l_')

    
    g_center = [center * (idx + offset_num) for idx in range(low_peak, high_peak + 1)]
    
    #constraints for center
    
    g_center_index = 0
    for peak in range(low_peak, high_peak + 1):
        if peak == low_peak:
            model.set_param_hint('g' + str(peak) + '_center', value = g_center[g_center_index], min = range_low, max = baseline_width + g_center[g_center_index])
            g_center_index += 1
        elif peak == high_peak:
            g_center_last = len(g_center) - 1 #last index of g_center
            model.set_param_hint('g' + str(peak) + '_center', value = g_center[g_center_last], min = g_center[g_center_last] - baseline_width, max = range_high)
        else: 
            model.set_param_hint('g' + str(peak) + '_center', value = g_center[ g_center_index], min = g_center[g_center_index] - baseline_width, max = baseline_width + g_center[g_center_index])
            g_center_index += 1
            
            
    # model.set_param_hint('g1_center', value = g_center[0], min = range_low, max = baseline_width + g_center[0])
    # model.set_param_hint('g2_center', value = g_center[1], min = g_center[1] - baseline_width, max = baseline_width + g_center[1])
    # model.set_param_hint('g3_center', value = g_center[2], min = g_center[2] - baseline_width, max = baseline_width + g_center[2])
    # model.set_param_hint('g4_center', value = g_center[3], min = g_center[3] - baseline_width, max = range_high)
    
    #constraints for sigma
    
    for peak in range(low_peak, high_peak + 1):
        model.set_param_hint('g' + str(peak) + '_sigma', value = 0.5 * baseline_width, min = 0, max = baseline_width)
        
    # model.set_param_hint('g1_sigma', value = 0.5 * baseline_width, min = 0, max = baseline_width)
    # model.set_param_hint('g2_sigma', value = 0.5 * baseline_width, min = 0, max = baseline_width)
    # model.set_param_hint('g3_sigma', value = 0.5 * baseline_width, min = 0, max = baseline_width)
    # model.set_param_hint('g4_sigma', value = 0.5 * baseline_width, min = 0, max = baseline_width)
    
    #constraints for amplitude
    g_amplitude = [np.amax(counts)*np.sqrt(2*np.pi)*baseline_width/(2**num) for num in range(low_peak, high_peak + 1)]
    
    g_amplitude_index = 0
    for peak in range(low_peak, high_peak + 1):
        model.set_param_hint('g' + str(peak) + '_amplitude', value = g_amplitude[g_amplitude_index], min = 0)
        g_amplitude_index += 1
        
    # model.set_param_hint('g1_amplitude', value = g_amplitude[0], min = 0)
    # model.set_param_hint('g2_amplitude', value = g_amplitude[1], min = 0)
    # model.set_param_hint('g3_amplitude', value = g_amplitude[2], min = 0)
    # model.set_param_hint('g4_amplitude', value = g_amplitude[3], min = 0)
    
    model.set_param_hint('l_slope', value = 0, max = 0) #constraint the slope fit to be less or equal to 0
    model.set_param_hint('l_intercept', value = counts[0])
    
    params = model.make_params()
    
    # params = model.make_params(
    #     g1_amplitude=max(counts)*np.sqrt(2*np.pi)*baseline_width/2, 
    #     g2_amplitude=max(counts)*np.sqrt(2*np.pi)*center/4, 
    #     g3_amplitude=max(counts)*np.sqrt(2*np.pi)*baseline_width/8, 
    #     g4_amplitude=max(counts)*np.sqrt(2*np.pi)*baseline_width/16,
    #     g1_center=center*(1+offset_num),
    #     g2_center=center*(2+offset_num),
    #     g3_center=center*(3+offset_num),
    #     g4_center=center*(4+offset_num),
    #     g1_sigma= 0.5*baseline_width,
    #     g2_sigma= 0.5*baseline_width,
    #     g3_sigma= 0.5*baseline_width,
    #     g4_sigma= 0.5*baseline_width,
    #     l_slope = 0,
    #     l_intercept = counts[0],
    #     )

    
    # params['g1_sigma'].max = baseline_width
    # params['g2_sigma'].max = baseline_width
    # params['g3_sigma'].max = baseline_width
    # params['g4_sigma'].max = baseline_width
    # params['g5_sigma'].max = baseline_width
    # params['g6_sigma'].max = baseline_width
    # params['g1_center'].min = range_low
    # params['g1_center'].max = baseline_width + g1_c
    # params['g2_center'].min = range_low
    # params['g2_center'].max = baseline_width + g2_c
    # params['g3_center'].min = g3_c - baseline_width
    # params['g3_center'].max = g3_c + baseline_width
    # params['g4_center'].min = g4_c - baseline_width
    # params['g4_center'].max = range_high
    # params['g5_center'].min = range_low
    # params['g5_center'].max = center*5 + 0.5*baseline_width 
    # params['g6_center'].min = range_low
    # params['g6_center'].max = center*5 + 0.5*baseline_width 
    # params['g1_amplitude'].min = 0
    # params['g2_amplitude'].min = 0
    # params['g3_amplitude'].min = 0
    # params['g4_amplitude'].min = 0
    # params['g5_amplitude'].min = 0
    # params['g6_amplitude'].min = 0
    
    # print(center*(1+offset_num))
    res = model.fit(counts, params=params, x=centers, weights = 1/np.sqrt(counts))
    
    # y_fit = res.eval(x=centers)
    # plt.plot(centers, y_fit)
        #****************
    # ci = res.conf_interval()
    # lm.printfuncs.report_ci(ci)
        #****************
    # print(res.fit_report())
    return res

def fit_alpha_gauss(values, binnum=20):
    f_range = {}
    curr_hist = np.histogram(values, bins = binnum)

    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1])/2
    f_range['center'] = centers[np.argmax(counts)]
    std_guess = np.std(values)
    mean_guess = centers[np.argmax(counts)]
    f_range['low'] = mean_guess - 0.25 * std_guess
    f_range['high'] = mean_guess + 0.5 * std_guess
#    print(f_range['center'], f_range['low'], f_range['high'])
    curr_peak_data = values[(values >= f_range['low']) & (values <= f_range['high'])]

#    high_val = 3.5
#    low_val = 2.4
#    center_val = (high_val - low_val) / 2.0
#    curr_peak_data = values[(values > low_val) & (values < high_val)]
    curr_hist = np.histogram(curr_peak_data, bins = binnum)
#    plt.hist(curr_peak_data, bins = binnum)
    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1])/2.0
    model = lm.models.GaussianModel()
    params = model.make_params(amplitude=max(counts), center=mean_guess, sigma=std_guess)
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1/counts))
    
    mean_guess = res.params['center'].value
    std_guess = res.params['sigma'].value
    f_range['low'] = mean_guess - 2.0 * std_guess
    f_range['high'] = mean_guess + 3.0 * std_guess 
    curr_peak_data = values[(values >= f_range['low']) & (values <= f_range['high'])]
    curr_hist = np.histogram(curr_peak_data, bins = binnum)
    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1])/2.0
    model = lm.models.GaussianModel()
    params = model.make_params(amplitude=max(counts), center=mean_guess, sigma=std_guess)
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1/counts))
    
    f_range['fit'] = res
    return f_range
    
def plot_fit(fit_info, values, binnum = 20, plot_hists = True, label = None):
    fit_data = values[(values >= fit_info['low']) & (values <= fit_info['high'])]
    numvalues = len(fit_data)
    h = 3.49*(numvalues)**(-1/3) * np.std(fit_data)
    binnum = int(np.ceil((max(fit_data) - min(fit_data))/h))
    if plot_hists:
        curr_hist = plt.hist(fit_data, bins = binnum)
    x = np.linspace(fit_info['low'],fit_info['high'],num=200)
    plt.plot(x, fit_info['fit'].eval(params=fit_info['fit'].params, x=x), color='red', label = label)
    
def get_mode(hist_data):
    counts = hist_data[0]
    bins = hist_data[1]
    centers = (bins[1:] + bins[:-1])/2.0
    max_index = np.argmax(counts)
    return centers[max_index], np.amax(counts)


# takes in measurement info, and processes it at waveform level, constructs different histograms, and does gaussian fits
class WaveformProcessor:
    def __init__(self, info, run_info_self = None, run_info_solicit = None, baseline_correct = False, range_low = 0, range_high = 2, center = 0.1, offset_num = 0, peak_range = (1,4), no_solicit = False, status = 0):
        self.baseline_correct = baseline_correct
        self.info = info
        self.run_info_self = run_info_self
        self.range_low = range_low
        self.range_high = range_high
        self.center = center
        self.numpeaks = peak_range[1] - (peak_range[0] - 1)
        self.offset_num = offset_num
        self.peak_range = peak_range #a tuple that contains the lowest peak and highest peak to Gauss fit
        self.low_peak = peak_range[0]
        self.high_peak = peak_range[1]
        self.no_solicit = no_solicit
        self.status = status
        if no_solicit == False:
            self.run_info_solicit = run_info_solicit
            self.baseline_mode = run_info_solicit.baseline_mode
        else:
            self.baseline_mode = run_info_self.baseline_mode_mean
            # self.baseline_mode = 1 #PLACEHOLDER
            self.baseline_rms = run_info_self.baseline_mode_rms
            self.baseline_std = 0.25*run_info_self.baseline_mode_std
            self.baseline_err = run_info_self.baseline_mode_err
            self.baseline_rms = run_info_self.baseline_mode_rms
            # self.baseline_std = 1
            # self.baseline_err = 1
        
    def process_h5(self):
        for curr_file in self.run_info_self.hd5_files:
            for curr_acquisition_name in self.run_info_self.acquisition_names[curr_file]:
#                self.peak_values = np.array(self.run_info_self.peak_data[curr_file][curr_acquisition_name])
                # if self.no_solicit == True:
                if self.status == 0:
                    self.all = np.array(self.run_info_self.all_peak_data)
                elif self.status == 1:
                    self.all = np.array(self.run_info_self.all_dark_peak_data)
                else:
                    self.all = np.array(self.run_info_self.all_led_peak_data)

                self.peak_values = self.all[(self.all >= self.range_low) & (self.all <= self.range_high)] #peaks in a range

            
        if self.no_solicit == False:
            for curr_file in self.run_info_solicit.hd5_files:
                for curr_acquisition_name in self.run_info_solicit.acquisition_names[curr_file]:
                    # try:
                    if self.run_info_solicit.specifyAcquisition:
                            curr_acquisition_name = self.run_info_solicit.acquisition
                    #except:
                    else:
                        self.baseline_values = np.array(self.run_info_solicit.peak_data[curr_file])
                    
                    self.baseline_values = np.array(self.run_info_solicit.peak_data[curr_file][curr_acquisition_name])
                
    def process_text(self, overwrite):
#        check if already saved as csv
        if not self.info.saved_to_csv or overwrite:
#            if not save, read in waveform data and save to .csv
            print('reading self-trig waveforms')
            save_peaks_csv(self.info.selftrig_path, self.info.selftrig_savedir, self.info.peak_search_params)
            print('reading solicit waveforms')
            save_baseline_csv(self.info.solicit_path, self.info.solicit_savedir, self.info.peak_search_params)
        self.baseline_values = read_data_csv(self.info.solicit_savedir)['waveform data']
        self.peak_values = read_data_csv(self.info.selftrig_savedir)['peaks']
        
#   reads in the waveform data either from the raw data or from a pre-saved .csv file
    def process(self, overwrite = False, do_spe = True, do_alpha = False, range_low = 0, range_high = 2, center = 0.1):  
        if self.info.data_type == 'text':
            self.process_text(overwrite)
        elif self.info.data_type == 'h5':
            self.process_h5()
        else:
            return

        if do_alpha:
            self.peak_values = self.peak_values[self.peak_values > self.info.min_alpha_value]
        
        # self.numbins = int(round(np.sqrt(len(self.peak_values))))  
        # self.numbins = self.info.peaks_numbins
        if self.peak_range != (1,4): #if doing 4 peaks, the bin number are calculated using proper stats
            self.numbins = self.info.peaks_numbins 
        else:
            self.numbins = int(np.sqrt(len(self.peak_values))) 
   
        if self.no_solicit == False: #added code for alpha analysis 
            self.baseline_fit = fit_baseline_gauss(
                    self.baseline_values,
                    binnum = self.info.baseline_numbins,
                    alpha = do_alpha
                    )
            self.baseline_std = self.baseline_fit['fit'].values['sigma']
            self.baseline_mean = self.baseline_fit['fit'].values['center']
            self.baseline_err = self.baseline_fit['fit'].params['center'].stderr
            self.baseline_rms = np.sqrt(np.mean(self.baseline_values**2))
            print('baseline mean: ' + str(self.baseline_mean))
            print('baseline std: ' + str(self.baseline_std))
        else:
            self.baseline_mean = self.baseline_mode
            self.baseline_std = 0.002 #arbitrary
            print('baseline mode: ' + str(self.baseline_mode))
            print('baseline std: ' + str(self.baseline_std))
        if do_spe:
            self.peak_fit = fit_peaks_multigauss(
                    self.peak_values,
                    self.baseline_mean,
                    2.0 * self.baseline_std, 
                    binnum = self.numbins,
                    range_low = self.range_low,
                    range_high = self.range_high,
                    center = self.center,
                    offset_num = self.offset_num,
                    peak_range = self.peak_range
                    )
            
            self.peak_locs = [self.peak_fit.params['g' + str(idx + 1) + '_center'].value for idx in range(self.low_peak-1, self.high_peak)]
            # self.peak_locs = [self.peak_fit.params['g1_center'].value, self.peak_fit.params['g2_center'].value, self.peak_fit.params['g3_center'].value, self.peak_fit.params['g4_center'].value,]
            print(self.peak_locs)
            
            self.peak_sigmas = [self.peak_fit.params['g' + str(idx + 1) + '_sigma'].value for idx in range(self.low_peak-1, self.high_peak)]
            # self.peak_sigmas = [self.peak_fit.params['g1_sigma'].value, self.peak_fit.params['g2_sigma'].value, self.peak_fit.params['g3_sigma'].value, self.peak_fit.params['g4_sigma'].value]
            print(self.peak_sigmas)
            
            self.peak_stds = [self.peak_fit.params['g' + str(idx + 1) + '_center'].stderr for idx in range(self.low_peak-1, self.high_peak)]
            # self.peak_stds = [self.peak_fit.params['g1_center'].stderr, self.peak_fit.params['g2_center'].stderr, self.peak_fit.params['g3_center'].stderr, self.peak_fit.params['g4_center'].stderr]
            print(self.peak_stds)
            
            # self.peak_err = [np.sqrt(sigma**2 - self.baseline_std**2) for sigma in self.peak_sigmas] #error on peak location as rms difference between peak and baseline width
            # self.peak_stds = self.peak_err
            
            self.peak_wgts = [1.0 / curr_std for curr_std in self.peak_stds]
            self.spe_num = []
        
            
            self.resolution = [(self.peak_locs[i+1]-self.peak_locs[i])/np.sqrt(self.peak_sigmas[i]**2 + self.peak_sigmas[i+1]**2) for i in range(len(self.peak_locs)-1)]
            print('sigma SNR: ' + str(self.resolution))
            
            for idx in range(self.low_peak-1, self.high_peak):
                self.spe_num.append(float(idx + 1 + self.offset_num))
            # self.peak_locs = sorted(self.peak_locs)
            
            # linear fit to the peak locations
            model = lm.models.LinearModel()
            params = model.make_params()
            
            self.spe_res = model.fit(self.peak_locs[:self.numpeaks], params=params, x=self.spe_num, weights=self.peak_wgts[:self.numpeaks]) # creates linear fit model
#            
            
            print('SNR: ' + str(self.spe_res.params['slope'].value/self.baseline_mode))
            print('SNR 2-3: ' + str((self.peak_locs[2]-self.peak_locs[1])/self.baseline_mode))
            print('SNR 1-2: ' + str((self.peak_locs[1]-self.peak_locs[0])/self.baseline_mode))
            
            if self.baseline_correct: #maybe changing this baseline correction might help clipping the baseline data to the right more
                # self.A_avg = np.mean(self.all) - self.spe_res.params['intercept'].value # spectrum specific baseline correction
                self.A_avg = np.mean(self.all) - self.spe_res.params['intercept'].value

                # #self.A_avg_err = self.A_avg * np.sqrt((sem(self.all) / np.mean(self.all))** 2 + (self.spe_res.params['intercept'].stderr / self.spe_res.params['intercept'].value)** 2)
                # self.A_avg_err = np.sqrt((sem(self.all))** 2 + (self.spe_res.params['intercept'].stderr)** 2)
                self.A_avg_err = np.sqrt((sem(self.all))** 2 + (self.spe_res.params['intercept'].stderr)** 2)
            else:
                # self.A_avg = np.mean(self.all)
                # self.A_avg_err = self.A_avg * np.sqrt((sem(self.all) / np.mean(self.all)) ** 2)   
                self.A_avg = np.mean(self.all)
                self.A_avg_err = self.A_avg * np.sqrt((sem(self.all) / np.mean(self.all)) ** 2) 
           
            if self.run_info_self.led:
                self.A_subtract_avg = self.get_subtract_hist_mean(self.run_info_self.all_led_peak_data, self.run_info_self.all_dark_peak_data, plot = False)
            
            # wesley_vals = (self.all - self.spe_res.params['intercept'].value) 
            # plt.figure()
            # plt.hist(wesley_vals, bins = 1000, label = self.info.bias)
            # plt.title(f'Average amplitude: {np.mean(wesley_vals):0.03}')
            # plt.legend()
            # plt.axvline(x = np.mean(wesley_vals), color = 'red')
            
            # print('here') #checks histograms and avg peak location
            # plt.figure()
            # plt.hist(self.led_off, bins = 150)
            # plt.title(str(self.A_avg) + ' ' + str(self.spe_res.params['slope'].value))
            
            # rest of this function is for CA     
            # self.A_avg = self.A_subtract_avg #changes the average value using the subtraction hist method
            self.CA = self.A_avg / self.spe_res.params['slope'].value - 1
            self.CA_err = self.CA * np.sqrt(
                    (self.A_avg_err / self.A_avg) ** 2 +
                    (self.spe_res.params['slope'].stderr / self.spe_res.params['slope'].value) ** 2)
      
            
        if do_alpha:
            self.alpha_fit = fit_alpha_gauss(self.peak_values, binnum = self.info.peaks_numbins)
            self.alpha_res = self.alpha_fit['fit'] 

    def get_alpha_data(self):
        return self.peak_values

    def get_baseline_data(self):
        return self.baseline_values
    
    def get_alpha_fit(self):
        return self.alpha_res

    def get_baseline_fit(self):
        return self.baseline_fit['fit']

    def get_spe(self):
        return (self.spe_res.params['slope'].value, self.spe_res.params['slope'].stderr)
    
    def get_CA(self):
        return (self.CA, self.CA_err)

    def get_CA_spe(self, spe, spe_err):
        # print('average A error', self.A_avg_err)
        currCA = self.A_avg / spe - 1
        currCA_err = currCA * np.sqrt(
                    (self.A_avg_err / self.A_avg) ** 2 +
                    (spe_err / spe) ** 2)

        return (currCA, currCA_err)
    
    def get_CA_rms(self, spe, spe_err):
        currCA = self.A_avg / spe - 1
        Q_twi = self.peak_values - self.spe_res.params['intercept'].value
        Q_1pe = spe
        sqrtval = Q_twi / Q_1pe - (currCA + 1)
        val = sqrtval * sqrtval
        rms = np.sqrt(np.mean(val))
        rms_err = rms * np.sqrt(
                    (self.A_avg_err / self.A_avg) ** 2 +
                    (spe_err / spe) ** 2)
        return (rms, rms_err)
        
    def get_alpha(self, sub_baseline = True):
        alpha_value = self.alpha_res.params['center'].value
        alpha_error = self.alpha_res.params['center'].stderr
        if sub_baseline:
            baseline_value = self.baseline_mean
            baseline_error = self.baseline_err
            alpha_value -= baseline_value
            alpha_error = np.sqrt(alpha_error * alpha_error + baseline_error * baseline_error)
        return alpha_value, alpha_error
    
    def get_alpha_std(self):
        alpha_value = self.alpha_res.params['sigma'].value
        alpha_error = self.alpha_res.params['sigma'].stderr

        return alpha_value, alpha_error
    
    def plot_spe(self, with_baseline = True, baselinecolor = 'orange', peakcolor = 'blue', savefig = False, path = None):
        fig = plt.figure()
        plt.errorbar(self.spe_num, self.peak_locs[:self.numpeaks], yerr = self.peak_stds[:self.numpeaks], fmt = '.', label = 'Solicited-LED-Triggered Peaks', color = 'tab:' + peakcolor, markersize = 10)
        if with_baseline:
            if self.no_solicit == False:
#            plt.errorbar(0, self.baseline_mean, yerr = self.baseline_std, fmt='.', label = 'Solicited Baseline Peak')
                plt.errorbar(0, self.baseline_mean, yerr = self.baseline_err, fmt='.', label = 'Solicited Baseline Peak', color = 'tab:' + baselinecolor, markersize = 10)
        
        b = self.spe_res.params['intercept'].value
        m = self.spe_res.params['slope'].value 
        x_values = np.linspace(0, len(self.spe_num) + 1, 20)
        y_values = m * x_values + b
        plt.plot(x_values, y_values, '--', color = 'tab:' + peakcolor, label = 'Solicited-LED-Triggered Fit')
#        dely = self.spe_res.eval_uncertainty(x=x_values, sigma=1)
#        plt.fill_between(x_values, y_values+dely, y_values-dely)
#        plt.plot(self.spe_num, self.spe_res.best_fit, 'r', label='Self-Triggered Fit')
        
        plt.xlabel('Photoelectron Peak Number')
        plt.ylabel('Peak Location [V]')
        
        plt.legend()
        
        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias:0.4} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'
        textstr += f'--\n'
        textstr += f'''Slope: {self.spe_res.params['slope'].value:0.4} +- {self.spe_res.params['slope'].stderr:0.2} [V/p.e.]\n'''
        textstr += f'''Intercept: {self.spe_res.params['intercept'].value:0.4} +- {self.spe_res.params['intercept'].stderr:0.2} [V]\n'''
        textstr += rf'''Reduced $\chi^2$: {self.spe_res.redchi:0.4}'''
        textstr += f'''\n'''
        textstr += f'--\n'
        if not self.no_solicit:
            textstr += f'Baseline: {self.baseline_mean:0.4} +- {self.baseline_err:0.2} [V]'

        props = dict(boxstyle='round', facecolor='tab:' + peakcolor, alpha=0.4)
        fig.text(0.6, 0.45, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(path) 
            plt.close(fig)

    def plot_baseline_histogram(self, with_fit = True, log_scale = False, color = 'orange', savefig = False, path = None):
        fig = plt.figure()
        plt.hist(self.baseline_values, bins = self.info.baseline_numbins, label = 'Solicited Baseline Data', color = 'tab:' + color)
        if with_fit:
            plot_fit(self.baseline_fit, self.baseline_values, binnum = self.info.baseline_numbins, plot_hists = False, label = 'Solicited Baseline Fit')
#        plt.legend(loc = 'center left')
        plt.xlabel('Waveform Amplitude [V]')
        plt.ylabel('Counts')
        if log_scale:
            plt.yscale('log')
        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias:0.4} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'
        textstr += f'--\n'
        textstr += f'''Baseline Mean: {self.baseline_fit['fit'].params['center'].value:0.4} +- {self.baseline_fit['fit'].params['center'].stderr:0.1} [V]\n'''
        textstr += f'''Baseline Sigma: {self.baseline_fit['fit'].params['sigma'].value:0.4} +- {self.baseline_fit['fit'].params['sigma'].stderr:0.1} [V]\n'''
        textstr += f'''Reduced $\chi^2$: {self.baseline_fit['fit'].redchi:0.4}'''
        
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        props = dict(boxstyle='round', facecolor='tab:' + color, alpha=0.5)
        fig.text(0.15, 0.9, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(path) 
            plt.close(fig)

    def plot_peak_histograms(self, with_fit = True, log_scale = True, peakcolor = 'blue', savefig = False, path = None):
        fig = plt.figure()

        # total_num_bins = self.numbins
        # bin_density = int(np.sqrt(len(self.peak_values))) / (self.range_high - self.range_low)
        if self.peak_range != (1,4): #if doing 4 peaks, the bin number are calculated using proper stats
            bin_density = self.info.peaks_numbins / (self.range_high - self.range_low)
        else:
            bin_density = int(np.sqrt(len(self.peak_values))) / (self.range_high - self.range_low)
            
      
            
        total_num_bins = bin_density * (np.amax(self.all) - np.amin(self.all))

        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'
        textstr += f'--\n'
        textstr += f'Peak Locations ($\mu$) [V]\n'
        for peak in range(len(self.peak_sigmas)):
            actual_peak = peak + 1
            textstr += f'''Peak {actual_peak}: {self.peak_fit.params['g' + str(actual_peak) + '_center'].value:0.2} $\pm$ {self.peak_fit.params['g' + str(actual_peak) + '_center'].stderr:0.2}\n'''
        # textstr += f'''SNR (quadrature): {self.resolution[0]:0.2}\n'''
        # textstr += f'''SNR 1-2 (mode): {(self.peak_locs[1]-self.peak_locs[0])/self.baseline_mode:0.2}\n'''
        # textstr += f'''SNR 2-3 (mode): {(self.peak_locs[2]-self.peak_locs[1])/self.baseline_mode:0.2}\n'''
       
        textstr += f'--\n'
        textstr += 'Peak Width (\u03C3) [V]\n'
        for peak in range(len(self.peak_sigmas)):
            curr_sigma_err = self.peak_fit.params['g' + str(peak + 1) + '_sigma'].stderr
            textstr += f'''{peak + 1}: {round(self.peak_sigmas[peak],5)} $\pm$ {curr_sigma_err:0.2}\n'''
         
        textstr += f'--\n'    
        textstr += f'''Reduced $\chi^2$: {self.peak_fit.redchi:0.2}\n'''
  

        curr_hist = np.histogram(self.peak_values, bins = self.numbins)
        counts = curr_hist[0]
        bins = curr_hist[1]
        centers = (bins[1:] + bins[:-1])/2
        y_line_fit = self.peak_fit.eval(x=centers)
        
        plt.plot(centers, y_line_fit,'r-', label='best fit')
        
        # plt.plot(np.linspace(self.range_low,self.range_high,len(self.peak_fit.best_fit)), self.peak_fit.best_fit, 'r-', label='best fit', color = 'red')        
        
        # x = np.linspace(self.range_low,self.range_high,200) 
        plt.plot(centers, self.peak_fit.best_values['l_intercept'] +  self.peak_fit.best_values['l_slope']*centers, 'b-', label='best fit - line')     
        
        # print('here')
        # print(self.peak_fit.best_values)
        
        props = dict(boxstyle='round', facecolor='tab:' + peakcolor, alpha=0.4)
        # plt.scatter(centers, counts, s = 7, color = 'black')
        # plt.hist(self.peak_values, bins = int(total_num_bins), color = 'tab:' + peakcolor)
        plt.hist(self.all, bins = int(total_num_bins), color = 'tab:' + peakcolor)
        
        fig.text(0.70, 0.925, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.ylabel('Counts')
        plt.xlabel('Pulse Amplitude [V]')
        
        if log_scale:
            plt.ylim(1E-1)
            plt.yscale('log')
        plt.tight_layout()
        
        if savefig:
            plt.savefig(path)
            plt.close(fig)

    def plot_alpha_histogram(self, with_fit = True, log_scale = False, peakcolor = 'purple'):
        fig = plt.figure()
        bin_density = self.info.peaks_numbins / (self.alpha_fit['high'] - self.alpha_fit['low'])
        total_num_bins = bin_density * (np.amax(self.peak_values) - np.amin(self.peak_values))
        plt.hist(self.peak_values, bins = int(total_num_bins), color = 'tab:' + peakcolor)
        if with_fit:
            plot_fit(self.alpha_fit, self.peak_values, binnum = self.info.peaks_numbins, plot_hists = False)
#        plt.legend(loc = 'center left')
        plt.xlabel('Waveform Amplitude [V]')
        plt.ylabel('Counts')
        plt.xlim(0.0)
        if log_scale:
            plt.yscale('log')
            plt.ylim(.1)
        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias:0.4} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'
        textstr += f'--\n'
        textstr += f'''Alpha Peak Mean: {self.alpha_fit['fit'].params['center'].value:0.4} +- {self.alpha_fit['fit'].params['center'].stderr:0.1} [V]\n'''
        textstr += f'''Alpha Peak Sigma: {self.alpha_fit['fit'].params['sigma'].value:0.4} +- {self.alpha_fit['fit'].params['sigma'].stderr:0.1} [V]\n'''
        textstr += f'''Reduced $\chi^2$: {self.alpha_res.redchi:0.4}'''
        
        
        props = dict(boxstyle='round', facecolor='tab:' + peakcolor, alpha=0.4)
        fig.text(0.175, 0.925, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        
    def plot_both_histograms(self, log_scale = True, density = True, alphas = False, baselinecolor = 'orange', peakcolor = 'blue', savefig = False, path = None):
        if self.no_solicit == True:
            print('NO PRE BREAKDOWN DATA TO PLOT')
        fig = plt.figure()
        plt.hist(self.baseline_values, bins = self.info.baseline_numbins, label = 'Solicited Baseline Data', density = density, color = 'tab:' + baselinecolor)
        if alphas:
            bin_density = self.info.peaks_numbins / (self.alpha_fit['high'] - self.alpha_fit['low'])
        else:
            # bin_density = self.info.peaks_numbins / (4.0 * self.baseline_std)
            bin_density = int(np.sqrt(len(self.peak_values))) / (self.range_high - self.range_low)
        total_num_bins = bin_density * (np.amax(self.all) - np.amin(self.all))
        # total_num_bins = self.info.peaks_numbins
        plt.hist(self.all, bins = int(total_num_bins), density = density, label = 'Solicited-LED-Triggered Pulse Height Data', color = 'tab:' + peakcolor)
        if log_scale:
            plt.ylim(1E-1)
            plt.yscale('log')
        if density:
            plt.ylabel('Frequency')
        else:
            plt.ylabel('Counts')
        plt.xlabel('Amplitude [V]')
        
        plt.legend()
        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias:0.4} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]'
        props = dict(boxstyle='round', facecolor='tab:' + peakcolor, alpha=0.4)
        fig.text(0.70, 0.75, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(path)
            plt.close(fig)
        
#currently broken:
    def plot_baseline_waveform_hist(self, num = -1, color = 'orange'):
        fig = plt.figure()
        waveform_data, waveform_times, num_w = get_baseline(self.info.solicit_path, self.info.peak_search_params)
        plt.hist2d(waveform_times, waveform_data, bins = 100, norm=mpl.colors.LogNorm())
        plt.xlabel(r'Time [$\mu$s]')
        plt.ylabel('Waveform Amplitude [V]')
        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias:0.4} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'
        textstr += f'Superposition of {num_w} waveforms'
        props = dict(boxstyle='round', facecolor='tab:' + color, alpha=0.5)
        fig.text(0.6, 0.3, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        
    def plot_peak_waveform_hist(self, num = -1, color = 'blue'):
        fig = plt.figure()
        waveform_data, waveform_times, num_w = get_peak_waveforms(self.info.selftrig_path, num)
        plt.hist2d(waveform_times, waveform_data, bins = 1000, norm=mpl.colors.LogNorm())
        plt.xlabel(r'Time [$\mu$s]')
        plt.ylabel('Waveform Amplitude [V]')
        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias:0.4} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'
        textstr += f'Superposition of {num_w} waveforms'
        props = dict(boxstyle='round', facecolor='tab:' + color, alpha=0.4)
        low, high = plt.ylim()
        plt.ylim(low, 4.5)
        fig.text(0.6, 0.9, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
    
    def plot_waveform(self, i, wf_type = 'selftrig'):
        if wf_type == 'solicit':
            x, y = get_waveform(self.info.solicit_path + f'w{i}.txt')
        elif wf_type == 'selftrig':
            x, y = get_waveform(self.info.selftrig_path + f'w{i}.txt')
        else:
            print('Fuck You <3')
            return

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(1,1)
            
        peaks, props = signal.find_peaks(y, **self.info.peak_search_params)
        for peak in peaks:
            plt.scatter(x[peak], y[peak])
        ax.plot(x, y, 'b-')
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title('Waveform ' + str(i) + ', ' + str(len(peaks)) + ' peaks')
        plt.draw_all()
        plt.show()
        return x, y
    
    def plot_fft(self, i, wf_type = 'solicit'):
        if wf_type == 'solicit':
            x, y = get_waveform(self.info.solicit_path + f'w{i}.txt')
        elif wf_type == 'selftrig':
            x, y = get_waveform(self.info.selftrig_path + f'w{i}.txt')
        else:
            print('Fuck You <3')
            return
        
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(1,1)
        
        N = len(x)
        T = (x[-1]-x[0])/len(x)
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]
        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Fourier Transform of Waveform {i}')

    def get_subtract_hist_mean(self, data1, data2, numbins = 2000, plot = False):
        if plot:
            plt.figure()
            (n, b, p) = plt.hist(data1, bins = numbins, density = False, label = 'LED-On', histtype='step')
            plt.axvline(x = np.mean(data1), color = 'red')
            print('LED on hist: ' + str(np.mean(data1)))
            print('LED off hist: ' + str(np.mean(data2)))
            plt.axvline(x = np.mean(data2), color = 'green')
            plt.hist(data2, bins = b, density = False, label = 'LED-Off', histtype='step')
        counts1, bins1 = np.histogram(data1, bins = numbins, density = False)
        counts2, bins2 = np.histogram(data2, bins = bins1, density = False)
        centers = (bins1[1:] + bins1[:-1])/2
        subtracted_counts = counts1 - counts2
        # subtracted_counts[subtracted_counts < 0] = 0
        if plot:
            plt.step(centers, subtracted_counts, label = 'subtracted hist')
            plt.legend()
            
        norm_subtract_hist = subtracted_counts / np.sum(subtracted_counts)
        # weights = 1.0 / subtracted_counts / 
        mean_value = np.sum(centers * norm_subtract_hist)
        ca_value = mean_value / self.spe_res.params['slope'].value - 1
        if plot:
            plt.title(f'OV: {round(self.run_info_self.bias - 27.1,3)}, CA value: {round(ca_value,3)}')
            plt.axvline(x = mean_value, color = 'orange')
        return mean_value
