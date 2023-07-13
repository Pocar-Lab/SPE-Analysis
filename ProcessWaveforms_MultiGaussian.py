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
    # wfs = fnmatch.filter(os.listdir(filepath), 'w*')
    # read in solicited trigger waveforms
    waveform_filenames = glob.glob(waveform_dir + 'w*.txt')
    values = []
    times = []
    num_w = 0
    # search each waveform for pulses, reject those with any
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
    # read in solicited trigger waveforms
    waveform_filenames = glob.glob(waveform_dir + 'w*.txt')
    values = []
    times = []
    num_w = 0
    # search each waveform for pulses, reject those with any
    for idx, w in enumerate(waveform_filenames):
        if idx % 100 == 0:
            print(idx)
        time, amp = get_waveform(w)
        peaks, props = signal.find_peaks(amp, **peak_search_params)
    # aggregate all pulseless data
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
        # f_range['low'] = 0.0
        f_range['high'] = 0.0045
        # f_range['high'] = 0.003
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
    # return {'center': np.mean(values), 'low': np.amin(values), 'high': np.amax(values), 'fit': res}
    return f_range

def fit_peaks_multigauss(values, baseline_width, centers, numpeaks = 4, cutoff = (0,np.infty)):
    '''
    Fit multiple gaussians to a finger plot made from values.
    
    Parameters
    ----------
    values : list
        height of peaks extracted from waveforms
    baseline_width : float
        estimate of the width in Volts of the noise
    centers : list
        initial guesses for centroid of each gaussian
    cutoff : tuple
        low and high cutoff values
    numpeaks : int
        the number of peaks you want to fit
    Returns
    -------
    res : lmfit.model.ModelResult
        an lmfit model result object containing all fit information
    '''
    curr_peak_data = values[(values >= cutoff[0]) & (values <= cutoff[1])]
    binnum = round(np.sqrt(len(curr_peak_data)))
    counts, bins = np.histogram(curr_peak_data, bins = binnum)
    centers = (bins[1:] + bins[:-1])/2
    model = (GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_') + GaussianModel(prefix='g3_') + GaussianModel(prefix='g4_') + LinearModel(prefix= 'l_') )
    
    params = model.make_params(
        l_slope = 0,
        l_intercept = counts[0],
        )
    
    peak_scale = max(counts)*np.sqrt(2*np.pi)*baeline_width
    for i in range(1, numpeaks+1):
        params[f'g{i}_amplitude'] = peak_scale/(2**i)
        
        params[f'g{i}_center'] = centers[i]
        params[f'g{i}_center'].max = params[f'g{i}_center'].value*1.3
        params[f'g{i}_center'].min = params[f'g{i}_center'].value*0.8
        
        params[f'g{i}_sigma'] = 0.5*baseline_width
        params[f'g{i}_sigma'].min = 0.3*0.5*baseline_width
        params[f'g{i}_sigma'].max = 2*0.5*baseline_width
        
        params[f'g{i}_amplitude'].min = 0
    
    res = model.fit(counts, params=params, x=centers, weights = 1/np.sqrt(counts))
    
    print(res.fit_report())
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
    # print(f_range['center'], f_range['low'], f_range['high'])
    curr_peak_data = values[(values >= f_range['low']) & (values <= f_range['high'])]

    # high_val = 3.5
    # low_val = 2.4
    # center_val = (high_val - low_val) / 2.0
    # curr_peak_data = values[(values > low_val) & (values < high_val)]
    curr_hist = np.histogram(curr_peak_data, bins = binnum)
    # plt.hist(curr_peak_data, bins = binnum)
    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1])/2.0
    model = lm.models.GaussianModel()
    params = model.make_params(amplitude=max(counts), center=mean_guess, sigma=std_guess)
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1/counts), nan_policy='omit')

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
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1/counts), nan_policy='omit')

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


# takes in measurement info, and processes it at waveform level
#constructs different histograms, and does gaussian fits
class WaveformProcessor:
    def __init__(self, info, centers, run_info_self = None,
                 run_info_solicit = None, baseline_correct = False,
                 cutoff = (0,np.infty), numpeaks = 4, no_solicit = False,
                 offset_num = 0):
        
        self.baseline_correct = baseline_correct
        self.info = info
        self.run_info_self = run_info_self
        self.cutoff = cutoff
        self.centers = centers
        self.numpeaks = numpeaks
        self.no_solicit = no_solicit
        self.offset_num = offset_num
        # options for if you forgot to take pre-breakdown data.......
        if no_solicit:
            self.baseline_mode = run_info_self.baseline_mode_mean
            # self.baseline_mode = 1 #PLACEHOLDER
            self.baseline_rms = run_info_self.baseline_mode_rms
            self.baseline_std = 0.25*run_info_self.baseline_mode_std
            self.baseline_err = run_info_self.baseline_mode_err
            self.baseline_rms = run_info_self.baseline_mode_rms
            # self.baseline_std = 1
            # self.baseline_err = 1
        else:
            self.run_info_solicit = run_info_solicit
            self.baseline_mode = run_info_solicit.baseline_mode

    def process_h5(self):
        for curr_file in self.run_info_self.hd5_files:
            for curr_acquisition_name in self.run_info_self.acquisition_names[curr_file]:
                # self.peak_values = np.array(self.run_info_self.peak_data[curr_file][curr_acquisition_name])
                self.peak_values = np.array(self.run_info_self.all_peak_data)
                self.peak_values = self.peak_values[(self.peak_values >= self.cutoff[0]) & (self.peak_values <= self.cutoff[1])] #peaks in a range
                self.all = np.array(self.run_info_self.all_peak_data) #all peaks

        if not self.no_solicit:
            for curr_file in self.run_info_solicit.hd5_files:
                for curr_acquisition_name in self.run_info_solicit.acquisition_names[curr_file]:
                    # try:
                    if self.run_info_solicit.specifyAcquisition:
                            curr_acquisition_name = self.run_info_solicit.acquisition
                    #except:
                    else:
                        self.baseline_values = np.array(self.run_info_solicit.peak_data[curr_file][curr_acquisition_name])
                    self.baseline_values = np.array(self.run_info_solicit.peak_data[curr_file][curr_acquisition_name])

    # reads in the waveform data either from the raw data or from a pre-saved .csv file
    def process(self, overwrite = False, do_spe = True, do_alpha = False):
        self.process_h5()

        if do_alpha:
            self.peak_values = self.peak_values[self.peak_values > self.info.min_alpha_value]

        self.numbins = int(round(np.sqrt(len(self.peak_values)))) #!!! attr defined outside init
        print(f"len: {len(self.peak_values)}")
        print(f"{self.numbins}")
        if self.no_solicit:
            self.baseline_mean = self.baseline_mode
            self.baseline_std = 0.002 #arbitrary
            print('baseline mode: ' + str(self.baseline_mode))
            print('baseline std: ' + str(self.baseline_std))
        else:
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

        if do_spe:
            self.peak_fit = fit_peaks_multigauss(
                    self.peak_values,
                    # self.baseline_mean,
                    2.0 * self.baseline_std,
                    # 0.002,
                    # binnum = self.info.peaks_numbins,
                    binnum = self.numbins,
                    range_low = self.range_low,
                    range_high = self.range_high,
                    center = self.center,
                    offset_num = self.offset_num
                    )

            self.peak_locs = [self.peak_fit.params['g1_center'].value, self.peak_fit.params['g2_center'].value, self.peak_fit.params['g3_center'].value, self.peak_fit.params['g4_center'].value,]
            print(self.peak_locs)
            self.peak_sigmas = [self.peak_fit.params['g1_sigma'].value, self.peak_fit.params['g2_sigma'].value, self.peak_fit.params['g3_sigma'].value, self.peak_fit.params['g4_sigma'].value]
            print(self.peak_sigmas)
            self.peak_stds = [self.peak_fit.params['g1_center'].stderr, self.peak_fit.params['g2_center'].stderr, self.peak_fit.params['g3_center'].stderr, self.peak_fit.params['g4_center'].stderr]
            print(self.peak_stds)

            # self.peak_err = [np.sqrt(sigma**2 - self.baseline_std**2) for sigma in self.peak_sigmas] #error on peak location as rms difference between peak and baseline width
            # self.peak_stds = self.peak_err

            self.peak_wgts = [1.0 / curr_std for curr_std in self.peak_stds]
            self.spe_num = []


            self.resolution = [(self.peak_locs[i+1]-self.peak_locs[i])/np.sqrt(self.peak_sigmas[i]**2 + self.peak_sigmas[i+1]**2) for i in range(len(self.peak_locs)-1)]
            print('sigma SNR: ' + str(self.resolution))

            for idx in range(self.numpeaks):
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

            if self.baseline_correct:
                self.A_avg = np.mean(self.all) - self.spe_res.params['intercept'].value # spectrum specific baseline correction
                #self.A_avg_err = self.A_avg * np.sqrt((sem(self.all) / np.mean(self.all))** 2 + (self.spe_res.params['intercept'].stderr / self.spe_res.params['intercept'].value)** 2)
                self.A_avg_err = np.sqrt((sem(self.all))** 2 + (self.spe_res.params['intercept'].stderr)** 2)
            else:
                self.A_avg = np.mean(self.all)
                self.A_avg_err = self.A_avg * np.sqrt((sem(self.all) / np.mean(self.all)) ** 2)


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
        fig.tight_layout()
        plt.rc('font', size=22)
        plt.errorbar(self.spe_num, self.peak_locs[:self.numpeaks], yerr = self.peak_stds[:self.numpeaks], fmt = '.', label = 'Self-Triggered Peaks', color = 'tab:' + peakcolor, markersize = 10)
        if with_baseline:
            if self.no_solicit == False:
                plt.errorbar(0, self.baseline_mean, yerr = self.baseline_err, fmt='.', label = 'Solicited Baseline Peak', color = 'tab:' + baselinecolor, markersize = 10)
            # else:
                # plt.errorbar(0, self.baseline_mode, yerr = self.baseline_err, fmt='.', label = 'Solicited Baseline Peak', color = 'tab:' + baselinecolor, markersize = 10)


        b = self.spe_res.params['intercept'].value
        m = self.spe_res.params['slope'].value
        x_values = np.linspace(0, len(self.spe_num) + 1, 20)
        y_values = m * x_values + b
        plt.plot(x_values, y_values, '--', color = 'tab:' + peakcolor, label = 'Self-Triggered Fit')
        # dely = self.spe_res.eval_uncertainty(x=x_values, sigma=1)
        # plt.fill_between(x_values, y_values+dely, y_values-dely)
        # plt.plot(self.spe_num, self.spe_res.best_fit, 'r', label='Self-Triggered Fit')

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
        fig.text(0.6, 0.4, textstr, fontsize=20,
                verticalalignment='top', bbox=props)

        if savefig:
            plt.savefig(path)
            plt.close(fig)

    def plot_baseline_histogram(self, with_fit = True, log_scale = False, color = 'orange', savefig = False, path = None):
        fig = plt.figure()
        plt.hist(self.baseline_values, bins = self.info.baseline_numbins, label = 'Solicited Baseline Data', color = 'tab:' + color)
        if with_fit:
            plot_fit(self.baseline_fit, self.baseline_values, binnum = self.info.baseline_numbins, plot_hists = False, label = 'Solicited Baseline Fit')
        # plt.legend(loc = 'center left')
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

#zoomed in version
    # def plot_peak_histograms(self, with_fit = True, log_scale = True, peakcolor = 'blue', savefig = False, path = None):
    #     fig = plt.figure()

    #     total_num_bins = self.numbins

    #     textstr = f'Date: {self.info.date}\n'
    #     textstr += f'Condition: {self.info.condition}\n'
    #     textstr += f'Bias: {self.info.bias} [V]\n'
    #     textstr += f'RTD4: {self.info.temperature} [K]\n'
    #     textstr += f'--\n'
    #     textstr += f'''Peak 1: {self.peak_fit.params['g1_center'].value:0.2} +- {self.peak_fit.params['g1_center'].stderr:0.2}\n'''
    #     textstr += f'''Peak 2: {self.peak_fit.params['g2_center'].value:0.2} +- {self.peak_fit.params['g2_center'].stderr:0.2}\n'''
    #     textstr += f'''Peak 3: {self.peak_fit.params['g3_center'].value:0.2} +- {self.peak_fit.params['g3_center'].stderr:0.2}\n'''
    #     textstr += f'''Peak 4: {self.peak_fit.params['g4_center'].value:0.2} +- {self.peak_fit.params['g4_center'].stderr:0.2}\n'''
    #     textstr += f'''Reduced $\chi^2$: {self.peak_fit.redchi:0.2}\n'''
    #     textstr += f'''SNR (quadrature): {self.resolution[0]:0.2}\n'''
    #     textstr += f'''SNR 1-2 (mode): {(self.peak_locs[1]-self.peak_locs[0])/self.baseline_mode:0.2}\n'''
    #     textstr += f'''SNR 2-3 (mode): {(self.peak_locs[2]-self.peak_locs[1])/self.baseline_mode:0.2}\n'''

    #     curr_hist = np.histogram(self.peak_values, bins = self.numbins)
    #     counts = curr_hist[0]
    #     bins = curr_hist[1]
    #     centers = (bins[1:] + bins[:-1])/2
    #     plt.plot(np.linspace(self.range_low,self.range_high,len(self.peak_fit.best_fit)), self.peak_fit.best_fit, '-', label='best fit', color = 'red')

    #     x = np.linspace(self.range_low,self.range_high,200)
    #     plt.plot(x, self.peak_fit.best_values['l_intercept'] +  self.peak_fit.best_values['l_slope']*x, '-', label='best fit - line', color = 'blue')

    #     props = dict(boxstyle='round', facecolor='tab:' + peakcolor, alpha=0.4)
    #     # plt.scatter(centers, counts, s = 7, color = 'black')
    #     plt.hist(self.peak_values, bins = int(total_num_bins), color = 'tab:' + peakcolor)

    #     fig.text(0.55, 0.925, textstr, fontsize=8,
    #             verticalalignment='top', bbox=props)
    #     plt.ylabel('Counts')
    #     plt.xlabel('Pulse Amplitude [V]')

    #     if log_scale:
    #         plt.ylim(1E-1)
    #         plt.yscale('log')
    #     plt.tight_layout()

    #     if savefig:
    #         plt.savefig(path)
    #         plt.close(fig)

    # see entire hist
    def plot_peak_histograms(self, with_fit = True, log_scale = True, peakcolor = 'blue', savefig = False, path = None):
        fig = plt.figure()\

        bin_width = (max(self.peak_values)-min(self.peak_values))/self.numbins
        # print(bin_width)

        total_num_bins = (max(self.all)-min(self.all))/bin_width

        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'
        textstr += f'--\n'
        textstr += f'''Peak 1: {self.peak_fit.params['g1_center'].value:0.2} +- {self.peak_fit.params['g1_center'].stderr:0.2}\n'''
        textstr += f'''Peak 2: {self.peak_fit.params['g2_center'].value:0.2} +- {self.peak_fit.params['g2_center'].stderr:0.2}\n'''
        textstr += f'''Peak 3: {self.peak_fit.params['g3_center'].value:0.2} +- {self.peak_fit.params['g3_center'].stderr:0.2}\n'''
        textstr += f'''Peak 4: {self.peak_fit.params['g4_center'].value:0.2} +- {self.peak_fit.params['g4_center'].stderr:0.2}\n'''
        textstr += f'''Reduced $\chi^2$: {self.peak_fit.redchi:0.2}\n'''
        textstr += f'''SNR (quadrature): {self.resolution[0]:0.2}\n'''
        textstr += f'''SNR 1-2 (mode): {(self.peak_locs[1]-self.peak_locs[0])/self.baseline_mode:0.2}\n'''
        textstr += f'''SNR 2-3 (mode): {(self.peak_locs[2]-self.peak_locs[1])/self.baseline_mode:0.2}\n'''

        curr_hist = np.histogram(self.all, bins = self.numbins)
        counts = curr_hist[0]
        bins = curr_hist[1]
        centers = (bins[1:] + bins[:-1])/2
        plt.plot(np.linspace(self.cutoff[0],self.cutoff[1],len(self.peak_fit.best_fit)), self.peak_fit.best_fit, '-', label='best fit', color = 'red') #plot the best fit model func

        x = np.linspace(self.cutoff[0],self.cutoff[1],200)
        plt.plot(x, self.peak_fit.best_values['l_intercept'] +  self.peak_fit.best_values['l_slope']*x, '-', label='best fit - line', color = 'blue') #plot linear component of best fit model func

        props = dict(boxstyle='round', facecolor='tab:' + peakcolor, alpha=0.4)
        # plt.scatter(centers, counts, s = 7, color = 'black')
        plt.hist(self.all, bins = int(total_num_bins), color = 'tab:' + peakcolor)

        fig.text(0.55, 0.925, textstr, fontsize=8,
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
        fig.tight_layout()
        plt.rc('font', size=22)
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
        fig.text(0.175, 0.925, textstr, fontsize=20,
                verticalalignment='top', bbox=props)
        plt.show()

    def plot_both_histograms(self, log_scale = True, density = True, alphas = False, baselinecolor = 'orange', peakcolor = 'blue', savefig = False, path = None):
        if self.no_solicit:
            print('NO PRE BREAKDOWN DATA TO PLOT')
        fig = plt.figure()
        plt.hist(self.baseline_values, bins = self.info.baseline_numbins, label = 'Solicited Baseline Data', density = density, color = 'tab:' + baselinecolor)
        if alphas:
            bin_density = self.info.peaks_numbins / (self.alpha_fit['high'] - self.alpha_fit['low'])
        else:
            bin_density = self.info.peaks_numbins / (4.0 * self.baseline_std)
        #total_num_bins = bin_density * (np.amax(self.peak_values) - np.amin(self.peak_values))
        total_num_bins = self.info.peaks_numbins
        plt.hist(self.peak_values, bins = int(total_num_bins), density = density, label = 'Self-Triggered Pulse Height Data', color = 'tab:' + peakcolor)
        if log_scale:
            plt.ylim(1E-1)
            plt.yscale('log')
        plt.ylabel('Frequency' if density else 'Counts')
        plt.xlabel('Amplitude [V]')

        plt.legend()
        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias:0.4} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]'
        props = dict(boxstyle='round', facecolor='tab:' + peakcolor, alpha=0.4)
        fig.text(0.75, 0.75, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()

        if savefig:
            plt.savefig(path)
            plt.close(fig)

    # currently broken:
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
