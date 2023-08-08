# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:07:39 2022

@author: lab-341
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from scipy import optimize
from scipy import interpolate
import peakutils
import matplotlib as mpl
import sys
from scipy.stats import sem
#%%

def get_data(h5path, groupname):
    with h5py.File(h5path, 'r') as hdf:
        data = hdf['RunData'].get(groupname)[:]
    return data
        
def get_grp_meta(h5path, groupname):
    with h5py.File(h5path, 'r') as hdf:
        meta_group = dict(hdf['RunData'][groupname].attrs)
    return meta_group

def get_run_meta(h5path):
    with h5py.File(h5path, 'r') as hdf:
        run_meta = dict(hdf['RunData'].attrs)
    return run_meta

def get_grp_names(h5path):
    with h5py.File(h5path, 'r') as hdf:
        group_names = list(hdf['RunData'].keys())
    return group_names

def get_mode(hist_data):
    counts = hist_data[0]
    bins = hist_data[1]
    centers = (bins[1:] + bins[:-1])/2.0
    max_index = np.argmax(counts)
    return centers[max_index], np.amax(counts)
    

class RunInfo:

    def __init__(self, f, acquisition = 'placeholder', is_solicit = False, do_filter = False, plot_waveforms = False, upper_limit = 4.4, baseline_correct = False, prominence = 0.005, specifyAcquisition = False, fourier = False, is_led = False):
        if not isinstance(f, list):
            raise TypeError('Files must be a in a list')
        self.do_filter = do_filter
        self.plot_waveforms = plot_waveforms
        self.hd5_files = f
        self.upper_limit = upper_limit
        self.baseline_correct = baseline_correct
#        holds all acquisition data index by first file name then by acquisition name
        self.acquisitions_data = {}
#        holds all acquisition names indexed by file name indexed by file name then by acquisition name
        self.acquisition_names = {}
#        holds acquisition time axis for all acquisitions in a single acquisition data array
        self.acquisitions_time = {}
#        holds all acquisition meta data, indexed first by file name then by acquisition name
        self.acquisition_meta_data = {}
        self.all_peak_data = []
        self.led = is_led
        if self.led:
            self.all_dark_peak_data = []
            self.all_led_peak_data = []
        
        # self.a_led = np.mean(self.all_led_peak_data)
        # self.a_dark = np.mean(self.all_dark_peak_data)

        self.acquisition = acquisition
        self.specifyAcquisition = specifyAcquisition
        self.fourier=fourier
        self.prominence = prominence
        self.baseline_levels = [] #list of mode of waveforms
        
        for curr_file in self.hd5_files:
            self.acquisition_names[curr_file] = get_grp_names(curr_file)
            print(self.acquisition_names[curr_file])
            # for curr_acquisition_name in self.acquisition_names[curr_file]:    
                
#        holds all run data indexed by file name
        self.run_meta_data = {}
        # self.LED_operating_voltage = []
        for curr_file in self.hd5_files:
            
            self.acquisition_names[curr_file] = get_grp_names(curr_file)
            self.acquisitions_time[curr_file] = {}
            self.acquisitions_data[curr_file] = {}
            self.acquisition_meta_data[curr_file] = {}
            
            self.run_meta_data[curr_file] = get_run_meta(curr_file)
            led_operating_v = self.run_meta_data[curr_file].get("RunNotes")
            # self.LED_operating_voltage.append(led_operating_v)
            
            if specifyAcquisition:
                    curr_data = get_data(curr_file, acquisition)
                    self.acquisitions_data[curr_file][acquisition] = curr_data[:, 1:]
                    self.acquisitions_time[curr_file][acquisition] = curr_data[:, 0]
                    self.acquisition_meta_data[curr_file][acquisition] = get_grp_meta(curr_file, acquisition)
                    self.bias = self.acquisition_meta_data[curr_file][acquisition]['Bias(V)']
                    self.condition = 'Liquid Xenon'
                    self.date = self.acquisition_meta_data[curr_file][acquisition]['AcquisitionStart']
                    self.trig = self.acquisition_meta_data[curr_file][acquisition]['LowerLevel']
                    self.yrange = self.acquisition_meta_data[curr_file][acquisition]['Range']
                    self.offset = self.acquisition_meta_data[curr_file][acquisition]['Offset']
                
            else:
                for curr_acquisition_name in self.acquisition_names[curr_file]:
                    curr_data = get_data(curr_file, curr_acquisition_name)
                    self.acquisitions_data[curr_file][curr_acquisition_name] = curr_data[:, 1:]
                    self.acquisitions_time[curr_file][curr_acquisition_name] = curr_data[:, 0]
                    self.acquisition_meta_data[curr_file][curr_acquisition_name] = get_grp_meta(curr_file, curr_acquisition_name)
                    self.bias = self.acquisition_meta_data[curr_file][curr_acquisition_name]['Bias(V)']
                    self.condition = '1426.93g LXe'
                    self.date = self.acquisition_meta_data[curr_file][curr_acquisition_name]['AcquisitionStart']
                    self.trig = self.acquisition_meta_data[curr_file][curr_acquisition_name]['LowerLevel']
                    self.yrange = self.acquisition_meta_data[curr_file][curr_acquisition_name]['Range']
                    self.offset = self.acquisition_meta_data[curr_file][curr_acquisition_name]['Offset']
                
        if not is_solicit:
            self.peak_search_params = {
                'height':0.0,# SPE
                'threshold':None,# SPE
                'distance':None,# SPE
                'prominence':prominence, 
                'width':None,# SPE
                'wlen':100, # SPE
                'rel_height':None,# SPE
                'plateau_size':None,# SPE
                # 'distance':10 #ADDED 2/25/2023
                }
            self.get_peak_data() 
        else:
            self.get_peak_data_solicit()
            
        self.baseline_mode_err = sem(self.baseline_levels)
        self.baseline_mode_std = np.std(self.baseline_levels)
        self.baseline_mode_mean= np.mean(self.baseline_levels)
        rms = [i**2 for i in self.baseline_levels]
        self.baseline_mode_rms=np.sqrt(np.mean(np.sum(rms)))        
    
        if not is_solicit:
            print('mean mode of amplitudes, standard deviation, SEM: ' + str(self.baseline_mode_mean) + ', ' + str(self.baseline_mode_std) + ',' + str(self.baseline_mode_err))
            
    def plot_hists(self, temp_mean, temp_std, new = False):
        if new:
            plt.figure() #makes new 
        (n, b, p) = plt.hist(self.all_peak_data, bins = 200, histtype = 'step', density = False)
        for curr_file in self.hd5_files:
            print(curr_file)
            for curr_acquisition_name in self.acquisition_names[curr_file]:
                print(curr_acquisition_name)
              
        if not self.plot_waveforms:
            font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 14,}
            plt.xlabel('Amplitude (V)', fontdict = font)
            plt.ylabel('Frequency', fontdict = font)
            bias = self.bias
            condition = self.condition
            date = self.date
            trig = self.trig
            yrange = self.yrange
            offset = self.offset
            plt.annotate(' Trig: ' + str(trig) + '\n Range: ' + str(yrange) + '\n Offset: ' + str(offset), xy=(0.65, 0.75), xycoords = 'axes fraction', size=15)
            #plt.title(str(date.decode('utf-8')) + ', ' + str(condition) + ', ' + temp_mean + ' $\pm$ ' + temp_std + ' K, ' + str(bias) + ' V', fontdict=font)
            plt.title(date + ', ' + str(condition) + ', ' + temp_mean + ' $\pm$ ' + temp_std + ' K, ' + str(bias) + ' V', fontdict=font) #old way
            plt.subplots_adjust(top=0.9)
            plt.yscale('log')
            plt.tight_layout()
            plt.show()
            
    def plot_led_dark_hists(self, temp_mean, temp_std, new = False):
        if not self.led:
            return
        if new:
            plt.figure() #makes new 
        (n, b, p) = plt.hist(self.all_peak_data, bins = 1500, histtype = 'step', density = False, label = 'All ')
        (n1, b1, p1) = plt.hist(self.all_led_peak_data, bins = b, histtype = 'step', density = False, label = 'LED on')
        (n2, b2, p2) = plt.hist(self.all_dark_peak_data, bins = b, histtype = 'step', density = False, label = 'LED off')
        plt.legend()
        for curr_file in self.hd5_files:
            print(curr_file)
            for curr_acquisition_name in self.acquisition_names[curr_file]:
                print(curr_acquisition_name)
              
        if not self.plot_waveforms:
            font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 14,}
            plt.xlabel('Amplitude (V)', fontdict = font)
            plt.ylabel('Frequency', fontdict = font)
            bias = self.bias
            condition = self.condition
            date = self.date
            trig = self.trig
            yrange = self.yrange
            offset = self.offset
            
            dark_count = float(len(self.all_dark_peak_data))
            led_count = float(len(self.all_led_peak_data)) - dark_count
            
            ratio1 = led_count / dark_count 
            
            self.led_ratio = ratio1
            
            ratio2 = float(len(self.all_led_peak_data)) / dark_count
            #print(ratio1)
            #print(ratio2)
                
            plt.annotate(' Trig: ' + str(trig) + '\n Range: ' + str(yrange) + '\n Offset: ' + str(offset) + '\n Ratio: ' + str(round(ratio1,2)), xy=(0.80, 0.60), xycoords = 'axes fraction', size=12)
            #plt.title(str(date.decode('utf-8')) + ', ' + str(condition) + ', ' + temp_mean + ' $\pm$ ' + temp_std + ' K, ' + str(bias) + ' V', fontdict=font)
            plt.title(date + ', ' + str(condition) + ', ' + temp_mean + ' $\pm$ ' + temp_std + ' K, ' + str(bias) + ' V', fontdict=font) #old way
            print(condition)
            plt.subplots_adjust(top=0.9)
            plt.yscale('log')
            plt.tight_layout()
            plt.show()
            
#            
    def get_peaks(self, filename, acquisition_name):
        all_peaks = []
        if self.led:
            dark_peaks = []
            led_peaks = []
        print(filename, acquisition_name)
        curr_data = self.acquisitions_data[filename][acquisition_name]
        time = self.acquisitions_time[filename][acquisition_name]
        window_length = time[-1] - time[0]
        num_points = float(len(time))
        fs = num_points / window_length
        num_wavefroms = np.shape(curr_data)[1]
        if self.plot_waveforms:
            num_wavefroms = 20
            # fig = plt.figure()
#            fig, axs = plt.subplots(1, 2)
        for idx in range(num_wavefroms):
            
#            idx = idx + 8000 #uncomment if plotting waveforms and want to see waveforms at different indices
            
            time = self.acquisitions_time[filename][acquisition_name]

            if idx % 1000 == 0:
                print(idx)
            
            amp = curr_data[:, idx]

            if np.amax(amp) > self.upper_limit:
                continue
            
            # peaks, props = signal.find_peaks(amp, height = 0.0, prominence = 0.3) #***
            
            use_bins = np.linspace(-self.upper_limit, self.upper_limit, 1000) #added code for alpha
            curr_hist = np.histogram(amp, bins = use_bins)
            baseline_mode_raw, max_counts = get_mode(curr_hist)
            self.baseline_levels.append(baseline_mode_raw)
            
            
            if self.baseline_correct:
                
                
# #                print('baseline:', baseline_level)

                baseline_level = peakutils.baseline(amp, deg=2)
                self.baseline_mode = baseline_level
                amp = amp - baseline_level
                
                #second baseline correction
                
                use_bins = np.linspace(-self.upper_limit, self.upper_limit, 2500)
            
                curr_hist = np.histogram(amp, bins = use_bins)
                
                baseline_level, max_counts = get_mode(curr_hist)
                
                self.baseline_mode = baseline_level
                amp = amp - baseline_level
                
            if self.do_filter and np.shape(amp) != (0,):
    #            sos = signal.butter(3, 1E6, btype = 'lowpass', fs = fs, output = 'sos')
                sos = signal.butter(3, 4E5, btype = 'lowpass', fs = fs, output = 'sos') # SPE dark/10g 
                filtered = signal.sosfilt(sos, amp)
                amp = filtered            
                
            peaks, props = signal.find_peaks(amp, **self.peak_search_params)
            
            if self.plot_waveforms:
                plt.title(acquisition_name)
                plt.tight_layout()
                if len(peaks) > 0:  #only plot peaks
                    plt.plot(time,amp)
                    for peak in peaks:
                        plt.plot(time[peaks], amp[peaks], '.')
            
            if self.led:
                led_time_thresh = (time[-1] + time[1]) / 2.0
            

            for peak in peaks:
                all_peaks.append(amp[peak])
                if self.led:
                    curr_time = time[peak]
                    if curr_time < led_time_thresh:
                        dark_peaks.append(amp[peak])
                    else:
                        led_peaks.append(amp[peak])
                    
        
        if self.led:
            # print(f'LED off: {np.mean(dark_peaks)} $+-$  {np.std(dark_peaks,ddof=1)/np.sqrt(len(dark_peaks))}')
            # print(f'LED on: {np.mean(led_peaks)} $+-$  {np.std(led_peaks,ddof=1)/np.sqrt(len(led_peaks))}')
            return all_peaks, dark_peaks, led_peaks
        else:
            return all_peaks
                
        
    def get_peak_data(self):
        self.peak_data = {}
        for curr_file in self.hd5_files:
            self.peak_data[curr_file] = {}
            for curr_acquisition_name in self.acquisition_names[curr_file]:
                if self.specifyAcquisition:
                    curr_acquisition_name = self.acquisition
                if self.led:
                    curr_peaks, curr_dark_peaks, curr_led_peaks = self.get_peaks(curr_file, curr_acquisition_name)
                else:
                    curr_peaks = self.get_peaks(curr_file, curr_acquisition_name)
                self.peak_data[curr_file][curr_acquisition_name] = curr_peaks
                self.all_peak_data = self.all_peak_data + curr_peaks
                if self.led:
                    self.all_dark_peak_data = self.all_dark_peak_data + curr_dark_peaks
                    self.all_led_peak_data = self.all_led_peak_data + curr_led_peaks
                if self.plot_waveforms == True or self.specifyAcquisition:
                    break

    def get_peaks_solicit(self, filename, acquisition_name):
        all_peaks = []
        curr_data = self.acquisitions_data[filename][acquisition_name]
        time = self.acquisitions_time[filename][acquisition_name]
        window_length = time[-1] - time[0]
        num_points = float(len(time))
        fs = num_points / window_length
#        print(fs)
        num_wavefroms = np.shape(curr_data)[1]
#        print(num_wavefroms)
        if self.plot_waveforms:
            num_wavefroms = 20
        for idx in range(num_wavefroms):
            if idx % 100 == 0:
                print(idx)
            
            amp = curr_data[:, idx]
            if np.amax(amp) > self.upper_limit:
                continue

            if self.baseline_correct:
                # use_bins = np.linspace(-self.upper_limit, self.upper_limit, 1000)
                # curr_hist = np.histogram(amp, bins = use_bins)
                # baseline_level, _ = get_mode(curr_hist)
                
                baseline_level = peakutils.baseline(amp, deg=2)
                amp = amp - baseline_level
                self.baseline_mode = baseline_level
                
                #second baseline correction
                
                use_bins = np.linspace(-self.upper_limit, self.upper_limit, 1000)
                curr_hist = np.histogram(amp, bins = use_bins)  
                baseline_level, max_counts = get_mode(curr_hist)      
                self.baseline_mode = baseline_level
                amp = amp - baseline_level

            if self.do_filter:
                sos = signal.butter(3, 4E5, btype = 'lowpass', fs = fs, output = 'sos')
                filtered = signal.sosfilt(sos, amp)
                amp = filtered

#            peaks, props = signal.find_peaks(amp, **self.peak_search_params)
            if self.plot_waveforms:
                if self.fourier:
                    fourier = np.fft.fft(amp)
                    n = amp.size
                    duration = 1E-4
                    freq = np.fft.fftfreq(n, d= duration/n)
                    colors  = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'aquamarine', 'pink', 'gray']
                    marker, stemlines, baseline = plt.stem(freq, np.abs(fourier), linefmt=colors[0], use_line_collection = True, markerfmt = " ")
                    plt.setp(stemlines, linestyle = "-", linewidth = 1, color = colors[0], alpha = 5/num_wavefroms)
                    plt.yscale('log')
                    plt.show()

                else:
                    plt.title(acquisition_name)
                    plt.tight_layout()
                    plt.plot(time,amp)
            amp = list(amp[100:])
            all_peaks+=amp
        return all_peaks
                
        
    def get_peak_data_solicit(self):
        self.peak_data = {}
        for curr_file in self.hd5_files:
            self.peak_data[curr_file] = {}
            for curr_acquisition_name in self.acquisition_names[curr_file]:
                if self.specifyAcquisition:
                    curr_acquisition_name = self.acquisition
                curr_peaks = self.get_peaks_solicit(curr_file, curr_acquisition_name)
                self.peak_data[curr_file][curr_acquisition_name] = curr_peaks  
                if self.plot_waveforms == True or self.specifyAcquisition:
                    break
                
    def plot_peak_waveform_hist(self, num = -1, color = 'blue'):
        fig = plt.figure()

        waveform_data = []
        waveform_times = []
        num_w = -1
        for curr_file in self.hd5_files:
            for curr_acquisition_name in self.acquisition_names[curr_file]:
                if self.specifyAcquisition:
                    curr_acquisition_name = self.acquisition
                curr_data = self.acquisitions_data[curr_file][curr_acquisition_name]
                time = self.acquisitions_time[curr_file][curr_acquisition_name]
                window_length = time[-1] - time[0]
                num_points = float(len(time))
                fs = num_points / window_length
        #        print(fs)
                if num < 1:
                    num_w = np.shape(curr_data)[1]
                else:
                    num_w = num
        #        print(num_wavefroms)

                for idx in range(num_w):
        #        for idx in range(200):
                    if idx % 100 == 0:
                        print(idx)
                    
                    amp = curr_data[:, idx]
                        
                    if np.amax(amp) > self.upper_limit: #SPE
                        continue
                    
                    #first baseline correction
                    
                    baseline_level = peakutils.baseline(amp, deg=2)
                    amp = amp - baseline_level
                    self.baseline_mode = baseline_level
                    
                    #second baseline correction
                    
                    use_bins = np.linspace(-self.upper_limit, self.upper_limit, 1000)
                    curr_hist = np.histogram(amp, bins = use_bins)  
                    baseline_level, max_counts = get_mode(curr_hist)      
                    self.baseline_mode = baseline_level
                    amp = amp - baseline_level
                    
                    sos = signal.butter(3, 4E5, btype = 'lowpass', fs = fs, output = 'sos')
                    filtered = signal.sosfilt(sos, amp)
                    amp = filtered
                    good_idx = (amp > 0.05) & (time > 1E-5)
                    amp = amp[good_idx]
                    time_curr = time[good_idx]
                    waveform_data += list(amp)
                    waveform_times += list(time_curr)
                if self.plot_waveforms == True or self.specifyAcquisition:
                    break
        
        plt.hist2d(waveform_times, waveform_data, bins = 50, norm=mpl.colors.LogNorm())
        plt.xlabel(r'Time [$\mu$s]')
        plt.ylabel('Waveform Amplitude [V]')
        
        led_time_thresh = (time[-1] + time[1]) / 2.0
        
        plt.axvline(x = led_time_thresh, color = 'r', label = 'LED on/off time')
        # textstr = f'Date: {self.info.date}\n'
        # textstr += f'Condition: {self.info.condition}\n'
        # textstr += f'Bias: {self.info.bias:0.4} [V]\n'
        # textstr += f'RTD4: {self.info.temperature} [K]\n'
        # textstr += f'Superposition of {num_w} waveforms'
        # props = dict(boxstyle='round', facecolor='tab:' + color, alpha=0.4)
        low, high = plt.ylim()
        # plt.ylim(low, 4.5)
        # fig.text(0.6, 0.9, textstr, fontsize=8,
        #         verticalalignment='top', bbox=props)
        plt.tight_layout()
        
    def average_all_waveforms(self):
        waveform_data = []
        for curr_file in self.hd5_files:
            for curr_acquisition_name in self.acquisition_names[curr_file]:
                curr_data = self.acquisitions_data[curr_file][curr_acquisition_name]
                waveform_data.append(curr_data)
                time_data = self.acquisitions_time[curr_file][curr_acquisition_name]
        all_data = np.array(waveform_data)
        averaged_data = np.average(all_data, axis = (0, 2))
        return averaged_data, time_data
