# -*- coding: utf-8 -*-
"""
Created on June 25 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import numpy as np
import lmfit as lm
from lmfit.models import LinearModel, GaussianModel, ExponentialModel
import matplotlib.pyplot as plt
from scipy.stats import sem
from MeasurementInfo import MeasurementInfo
from typing import Dict, List, Tuple, Optional

### Guide to porting ProcessWaveforms_MultiGaussian to ProcessHistograms:
# - Replace info and run_info_self with new MeasurementInfo class
# - Remove is_solicit and run_info_solict, now include within MeasurementInfo
# - Remove do_spe or do_alpha, call process_spe() or process_alpha() instead of process()

def fit_alpha_gauss(
    values: List[float], binnum: int = 20
) -> Dict[str, lm.model.ModelResult]:
    """
    Performs a Gaussian fit to a given data set. The fitting process is repeated twice to
    refine the center and standard deviation estimates, and provide a narrower fit range.

    Args:
        values (List[float]): List of values to perform the Gaussian fit on.
        binnum (int, optional): The number of bins to use when creating the histogram. Defaults to 20.

    Returns:
        Dict[str, ModelResult]: A dictionary containing the range ('low', 'high', 'center') and the final fit result ('fit') from the Gaussian model.
    """
    # TODO fit_info should be a class
    f_range = {}
    curr_hist = np.histogram(values, bins=binnum)

    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1]) / 2
    f_range["center"] = centers[np.argmax(counts)]
    std_guess = np.std(values)
    mean_guess = centers[np.argmax(counts)]
    f_range["low"] = mean_guess - 0.25 * std_guess
    f_range["high"] = mean_guess + 0.5 * std_guess
    # print(f_range['center'], f_range['low'], f_range['high'])
    curr_peak_data = values[(values >= f_range["low"]) & (values <= f_range["high"])]

    # high_val = 3.5
    # low_val = 2.4
    # center_val = (high_val - low_val) / 2.0
    # curr_peak_data = values[(values > low_val) & (values < high_val)]
    curr_hist = np.histogram(curr_peak_data, bins=binnum)
    # plt.hist(curr_peak_data, bins = binnum)
    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1]) / 2.0
    model = lm.models.GaussianModel()
    params = model.make_params(
        amplitude=max(counts), center=mean_guess, sigma=std_guess
    )
    res = model.fit(
        counts, params=params, x=centers, weights=np.sqrt(1 / counts), nan_policy="omit"
    )

    mean_guess = res.params["center"].value
    std_guess = res.params["sigma"].value
    f_range["low"] = mean_guess - 2.0 * std_guess
    f_range["high"] = mean_guess + 3.0 * std_guess
    curr_peak_data = values[(values >= f_range["low"]) & (values <= f_range["high"])]
    curr_hist = np.histogram(curr_peak_data, bins=binnum)
    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1]) / 2.0
    model = lm.models.GaussianModel()
    params = model.make_params(
        amplitude=max(counts), center=mean_guess, sigma=std_guess
    )
    res = model.fit(
        counts, params=params, x=centers, weights=np.sqrt(1 / counts), nan_policy="omit"
    )

    f_range["fit"] = res
    return f_range

def fit_peaks_multigauss(
    values: np.ndarray,
    baseline_width: float,
    centers: list[float],
    peak_range: tuple[float,float]=(1,4),
    cutoff: tuple[float, float] = (0, np.inf),
    background_linear: bool = True,
    bins = None,
) -> lm.model.ModelResult:
    """
    Fits multiple Gaussian functions to a 'finger plot' made from given values using the
    lmfit.models.GaussianModel method.

    Args:
        values (List[float]): Heights of peaks extracted from waveforms.
        baseline_width (float): An estimate of the width in Volts of the noise.
        centers (List[float]): Initial guesses for the centroid of each Gaussian.
        peak_range (tuple[int, int]): The number of peaks you want to fit. Defaults to 4.
        cutoff (Tuple[float, float], optional): Low and high cutoff values. Defaults to (0, np.inf).

    Returns:
        ModelResult: An lmfit ModelResult object containing all fit information.
    """
    curr_peak_data = values[(values >= cutoff[0]) & (values <= cutoff[1])]
    if not bins:
        bins = round(np.sqrt(len(curr_peak_data)))
    counts, b = np.histogram(curr_peak_data, bins=bins)
    bin_centers = (b[1:] + b[:-1]) / 2

    low_peak = peak_range[0]
    high_peak = peak_range[1]
    range_low = cutoff[0]
    range_high = cutoff[1]

    for peak in range(low_peak, high_peak + 1):
        if peak == low_peak:
            model = GaussianModel(prefix='g' + str(low_peak) + '_')
        else:
            model = model + GaussianModel(prefix='g' + str(peak) + '_')
    if background_linear:
        model = model + LinearModel(prefix= 'l_')
    else:
        model = model + ExponentialModel(prefix= 'e_')
    g_center = centers[:(peak_range[1]-peak_range[0]+1)]
    print('CENTER GUESSES TO BE USED IN FIT: ',g_center)

    # constraints for center
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
            # print('max ', baseline_width + g_center[g_center_index])
    # constraints for sigma
    for peak in range(low_peak, high_peak + 1):
        model.set_param_hint('g' + str(peak) + '_sigma', value = 0.5 * baseline_width, min = 0, max = baseline_width)

    # constraints for amplitude
    g_amplitude = [np.amax(counts)*np.sqrt(2*np.pi)*baseline_width/(2**num) for num in range(low_peak, high_peak + 1)]
    g_amplitude_index = 0
    for peak in range(low_peak, high_peak + 1):
        model.set_param_hint('g' + str(peak) + '_amplitude', value = g_amplitude[g_amplitude_index], min = 0)
        g_amplitude_index += 1

    if background_linear:
        # constraints for linear fit
        model.set_param_hint('l_slope', value=0, max=0) # constraint the slope fit to be less or equal to 0
        model.set_param_hint('l_intercept', value=counts[0])
    else:
        # constraints for exponential fit
        model.set_param_hint('e_decay', value=84e-3, min=0, max=1)
        model.set_param_hint('e_amplitude', value=68) # counts[0]

    params = model.make_params()
    # params.pretty_print()
    res = model.fit(counts, params=params, x=bin_centers, weights = 1/np.sqrt(counts), nan_policy='omit')
    # print(res.fit_report())
    return res


class ProcessHist:
    def __init__(
        self,
        info: MeasurementInfo,
        centers: List[float] | np.ndarray = [], #initializes centers as an empty list (so code still works for alpha data)
        baseline_correct: bool = False,
        cutoff: Tuple[float, float] = (0, np.inf),
        peak_range: Tuple[int,int] = (1,4),
        subtraction_method: bool = False,
        background_linear: bool = True,
        peaks: str = 'all',
    ):
        """
        Process and histogram the identified peaks and fit with a multi-gaussian to extract the
        SPE amplitude.

        Args:
            info (MeasurementInfo): Class containing info regarding measurement.
            centers (List[float]): Initial guesses for centroid of each gaussian.
            baseline_correct (bool): Boolean value indicating if baseline correction needs to be applied. Defaults to False.
            cutoff (Tuple[float, float]): Low and high cutoff values. Defaults to (0,np.inf).
            peak_range: The number of peaks you want to fit. Defaults to 4.
            background_linear: If to fit a linear or exponential background to the multi-gaussian
            peaks: Which peaks to include, either all, LED, or dark.
        """

        self.baseline_correct = baseline_correct
        self.info = info
        self.cutoff = cutoff
        self.low_peak = peak_range[0]
        self.high_peak = peak_range[1]
        self.numpeaks = peak_range[1] - (peak_range[0] - 1)
        # TODO make center guess only the guess for the first peak, and the rest int multiples
        self.centers = centers[peak_range[0]-1:]
        self.peak_range = peak_range
        self.range_high = cutoff[1]
        self.range_low = cutoff[0]
        # self.no_solicit = no_solicit
        self.subtraction_method = subtraction_method
        self.background_linear = background_linear
        self.peaks = peaks

        self.baseline_mean = self.info.baseline_mean
        self.baseline_err  = self.info.baseline_err
        self.baseline_std  = self.info.baseline_std


    def process_peaks(self) -> None:
        """Processes the .h5 files associated with the WaveformProcessor instance.

        The method extracts peak data and, if available, baseline data from .h5 files.
        It filters the peak data based on a predefined cutoff range and also handles solicit data if it's not disabled.
        """
        # TODO convert to mV
        if self.peaks == 'all':
            self.all_peaks = np.array(self.info.all_peaks) # * 1000
        elif self.peaks == 'dark':
            self.all_peaks = np.array(self.info.dark_peaks)
        elif self.peaks == 'LED':
            self.all_peaks = np.array(self.info.led_peaks)
        self.peak_values = self.all_peaks[
            (self.all_peaks >= self.cutoff[0])
            & (self.all_peaks <= self.cutoff[1])
        ]  # peaks in a range
        # self.all = np.array(self.info.all_peak_data) #all peaks

        self.all_led_peaks = np.array(self.info.led_peaks) # * 1000
        self.peak_led_values = self.all_led_peaks[
            (self.all_led_peaks >= self.cutoff[0])
            & (self.all_led_peaks <= self.cutoff[1])
        ]  # peaks in a range
        self.all_dark_peaks = np.array(self.info.dark_peaks) # * 1000
        self.peak_dark_values = self.all_dark_peaks[
            (self.all_dark_peaks >= self.cutoff[0])
            & (self.all_dark_peaks <= self.cutoff[1])
        ]  # peaks in a range

    def process_alpha(self, overwrite=False, subtraction_method=False):
        """Processes the waveform data, extracting various statistical information from it.

        Args:
            overwrite (bool, optional): If True, any previous processing results are overwritten. Defaults to False.
            do_spe (bool, optional): If True, Single Photoelectron (SPE) data is processed, including fitting multiple peaks and calculating signal-to-noise ratio (SNR). Defaults to True.
            do_alpha (bool, optional): If True, alpha particle data is processed. Defaults to False.
        """
        self.process_peaks()

        self.peak_values = self.peak_values[
            self.peak_values > self.info.min_alpha_value
        ]

        print('processing alpha data...')
        self.alpha_fit = fit_alpha_gauss(
            self.peak_values, binnum=self.info.peaks_numbins
        )
        self.alpha_res = self.alpha_fit["fit"]

    # reads in the waveform data either from the raw data or from a pre-saved .csv file
    def process_spe(self, overwrite=False, subtraction_method=False):
        """Processes the waveform data, extracting various statistical information from it.

        Args:
            overwrite (bool, optional): If True, any previous processing results are overwritten. Defaults to False.
            do_spe (bool, optional): If True, Single Photoelectron (SPE) data is processed, including fitting multiple peaks and calculating signal-to-noise ratio (SNR). Defaults to True.
            do_alpha (bool, optional): If True, alpha particle data is processed. Defaults to False.
        """
        self.process_peaks()

        # if self.peak_range != (1,4): # if doing 4 peaks, the bin number are calculated using proper stats
        #     self.numbins = self.info.peaks_numbins
        # else:
        self.numbins = int(np.sqrt(len(self.peak_values)))
          #!!! attr defined outside init

        self.peak_fit = fit_peaks_multigauss(
                values = self.peak_values,
                baseline_width = 2.0 * self.baseline_std,
                centers = self.centers,
                peak_range = self.peak_range,
                cutoff = self.cutoff,
                background_linear=self.background_linear,
                bins=round(np.sqrt(len(self.all_peaks)))
                )

        self.peak_locs = [self.peak_fit.params['g' + str(idx + 1) + '_center'].value for idx in range(self.low_peak-1, self.high_peak)]
        #pprint.pprint('peak locations from fit: '+ str(self.peak_locs))
        self.peak_sigmas = [self.peak_fit.params['g' + str(idx + 1) + '_sigma'].value for idx in range(self.low_peak-1, self.high_peak)]
        #pprint.pprint('peak sigmas (widths) from fit: '+ str(self.peak_sigmas))
        self.peak_stds = [self.peak_fit.params['g' + str(idx + 1) + '_center'].stderr for idx in range(self.low_peak-1, self.high_peak)]
        self.peak_sigmas_stds = [self.peak_fit.params['g' + str(idx + 1) + '_sigma'].stderr for idx in range(self.low_peak-1, self.high_peak)]
        for i in range(len(self.peak_sigmas_stds)):
            if not self.peak_sigmas_stds[i]:
                self.peak_sigmas_stds[i] = 1.0

        self.peak_amps = [self.peak_fit.params[f'g{idx}_amplitude'].value
                          for idx in range(self.low_peak, self.high_peak+1)]
        self.peak_amp_errs = [self.peak_fit.params[f'g{idx}_amplitude'].stderr
                              for idx in range(self.low_peak, self.high_peak+1)]
        for i in range(len(self.peak_amp_errs)):
            if not self.peak_amp_errs[i]:
                self.peak_amp_errs[i] = 1.0
        self.peak_heights = [self.peak_amps[peak] / (self.peak_sigmas[peak]*np.sqrt(2*np.pi))
                             for peak in range(len(self.peak_amps))]
        self.peak_height_errs = [self.peak_heights[peak] *
                                 np.sqrt((self.peak_amp_errs[peak]/self.peak_amps[peak])**2 +
                                         (self.peak_sigmas_stds[peak]/self.peak_sigmas[peak])**2)
                                 for peak in range(len(self.peak_amps))]

        #check if any of the fitted sigmas are less than the sigma of baseline noise (unphysical):
        for s in self.peak_sigmas:
            # TODO only warn if more than 1 sigma different
            if s < self.baseline_std:
                print('WARNING! Fitted sigma ' + str(s) + ' is less than baseline sigma ' + str(self.baseline_std) +'!')

        for i in range(len(self.peak_stds)):
            if self.peak_stds[i] is None:
                print('WARNING! Fit failed to return a standard error on the peak locations and returned None! Setting std = 1')
                self.peak_stds[i] = 1.0
            if self.peak_sigmas_stds[i] is None:
                print('WARNING! Fit failed to return a standard error on the peak sigmas and returned None! Setting std = 1')
                self.peak_sigmas_stds[i] = 1

        self.peak_wgts = [1.0 / curr_std for curr_std in self.peak_stds]

        self.resolution = [
            (self.peak_locs[i + 1] - self.peak_locs[i])
            / np.sqrt(self.peak_sigmas[i] ** 2 + self.peak_sigmas[i + 1] ** 2)
            for i in range(len(self.peak_locs) - 1)
        ]
        print("sigma SNR: " + str(self.resolution))

        self.spe_num = []
        for idx in range(self.low_peak-1, self.high_peak):
            self.spe_num.append(float(idx + 1))
        # self.peak_locs = sorted(self.peak_locs)

        # linear fit to the peak locations
        model = lm.models.LinearModel()
        params = model.make_params()
        self.spe_res = model.fit(
            self.peak_locs[: self.numpeaks],
            params=params,
            x=self.spe_num,
            weights=self.peak_wgts[: self.numpeaks],
        ) # creates linear fit model

        if self.baseline_correct:
            self.A_avg = ( np.mean(self.peak_values) - self.spe_res.params["intercept"].value)  # spectrum specific baseline correction
            # self.A_avg_err = self.A_avg * np.sqrt((sem(self.all) / np.mean(self.all))** 2 + (self.spe_res.params['intercept'].stderr / self.spe_res.params['intercept'].value)** 2)
            self.A_avg_err = np.sqrt(
                (sem(self.peak_values)) ** 2
                + (self.spe_res.params["intercept"].stderr) ** 2
            )
        else:
            self.A_avg = np.mean(self.peak_values)
            self.A_avg_err = self.A_avg * np.sqrt(
                (sem(self.peak_values) / np.mean(self.peak_values)) ** 2
            )

        self.CA = self.A_avg / self.spe_res.params["slope"].value - 1
        self.CA_err = self.CA * np.sqrt(
            (self.A_avg_err / self.A_avg) ** 2
            + (self.spe_res.params["slope"].stderr / self.spe_res.params["slope"].value) ** 2
        )
        print('CA at this bias voltage: mean of all amplitudes / SPE amplitude (gain slope) = '+str(self.CA) + ' +/- ' + str(self.CA_err))


    def get_spe(self) -> Tuple[float, float]:
        """Retrieves the slope value and its error from the spe fit results.

        Returns:
            Tuple[float, float]: A tuple containing the slope value and its error.
        """
        return (self.spe_res.params["slope"].value, self.spe_res.params["slope"].stderr)

    def get_CA_spe(self, spe: float, spe_err: float) -> Tuple[float, float]:
        """Computes the Correlated Avalanche (CA) factor and its error based on given spe and its error.

        Args:
            spe (float): The spe value.
            spe_err (float): The error in the spe value.

        Returns:
            Tuple[float, float]: A tuple containing the computed CA factor and its error.
        """
        currCA = self.A_avg / spe - 1
        currCA_err = currCA * np.sqrt(
            (self.A_avg_err / self.A_avg) ** 2 + (spe_err / spe) ** 2
        )

        return (currCA, currCA_err)

    def get_CA_rms(self, spe: float, spe_err: float) -> Tuple[float, float]:
        """Computes the root mean square (rms) of the Correlated Avalanche (CA) factor and its error based on given spe and its error.

        Args:
            spe (float): The spe value.
            spe_err (float): The error in the spe value.

        Returns:
            Tuple[float, float]: A tuple containing the rms value of the computed CA factor and its error.
        """
        currCA = self.A_avg / spe - 1
        Q_twi = self.peak_values - self.spe_res.params["intercept"].value
        Q_1pe = spe
        sqrtval = Q_twi / Q_1pe - (currCA + 1)
        val = sqrtval * sqrtval
        rms = np.sqrt(np.mean(val))
        rms_err = rms * np.sqrt(
            (self.A_avg_err / self.A_avg) ** 2 + (spe_err / spe) ** 2
        )
        return (rms, rms_err)


    def get_subtract_hist_mean(self, data1, data2, numbins=2000, plot=False):
        if plot:
            plt.figure()
            (n, b, p) = plt.hist(data1, bins=numbins, density=False, label='LED-On', histtype='step')
            plt.axvline(x = np.mean(data1), color='blue')
            print(f'LED on hist: {np.mean(data1)}')
            print(f'LED off hist: {np.mean(data2)}')
            plt.axvline(x=np.mean(data2), color='orange')
            plt.hist(data2, bins=b, density=False, label='LED-Off', histtype='step')
        counts1, bins1 = np.histogram(data1, bins=numbins, density=False)
        counts2, bins2 = np.histogram(data2, bins=bins1, density=False)
        centers = (bins1[1:] + bins1[:-1])/2
        subtracted_counts = counts1 - counts2

        if plot:
            plt.step(centers, subtracted_counts, label = 'subtracted hist')
            plt.legend()

        big_n = np.sum(subtracted_counts)
        norm_subtract_hist = subtracted_counts / big_n

        # weights = 1.0 / subtracted_counts /
        mean_value = np.sum(centers * norm_subtract_hist)
        ca_value = mean_value / self.spe_res.params['slope'].value - 1
        print('CA: ', ca_value)

        if plot:
            plt.title(f'Bias: 35V, mean_value: {mean_value}')
            plt.axvline(x = mean_value, color = 'green')
        # mean_err = np.sum((centers/big_n) ** 2)(subtracted_counts) + (np.sum(subtracted_counts*centers)/(big_n)**2) ** 2 * (np.sum(subtracted_counts)) #overestimation
        a = np.sum(subtracted_counts * centers)
        mean_err = np.sqrt(np.sum( ((a - centers * big_n)/ big_n ** 2) ** 2 * (counts1 + counts2)))
        return (mean_value, mean_err)

    def plot_led_dark_hists(self, led_voltage, new=False):
        if new:
            plt.figure() #makes new
        (_, b, _) = plt.hist(self.peak_values, bins=2000, histtype='step', density=False, label='All ')
        (_, _, _) = plt.hist(self.peak_led_values, bins=b, histtype='step', density=False, label='LED on')
        (_, _, _) = plt.hist(self.peak_dark_values, bins=b, histtype='step', density=False, label='LED off')
        print(f"{b=}")
        plt.legend()
        plt.tight_layout()
        plt.xlabel('Amplitude (V)', size=14, loc="right")
        plt.ylabel('Frequency', size=14, loc="top")
        # trig = self.info.trig
        # yrange = self.info.yrange
        # offset = self.info.offset
        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'

        dark_count = float(len(self.peak_dark_values))
        led_count = float(len(self.peak_led_values)) - dark_count
        self.led_ratio = led_count / dark_count

        # ratio2 = float(len(self.all_led_peak_data)) / dark_count
        #print(self.led_ratio)
        #print(ratio2)

        A_avg = np.mean(self.all_peaks)
        A_avg_err = A_avg * sem(self.all_peaks) / np.mean(self.all_peaks)

        # txtstr = f' Range: {yrange}\n Offset: {offset}\n'
        textstr =  f'LED Voltage: {led_voltage}\n'
        textstr += f'Ratio: {round(self.led_ratio,2)}\n'
        textstr += f'Avg Amp: {A_avg:.3} ± {A_avg_err:.3}\n'
        plt.annotate(textstr, xy=(0.60, 0.60), xycoords='axes fraction', size=12)
        plt.subplots_adjust(top=0.9)
        plt.yscale('log')
        plt.grid(True)
        plt.show()
        return (self.led_ratio, A_avg, A_avg_err)

    def plot_peak_histogram(
            self,
            with_fit: bool = True,
            log_scale: bool = True,
            peakcolor: str = "blue",
            savefig: bool = False,
            path: Optional[str] = None
        ) -> None:

        """
        Args:
            with_fit (bool, optional): If True, overlays the fitted model of the data on the plot. Defaults to True.
            log_scale (bool, optional): If True, sets the y-axis to a logarithmic scale. Defaults to True.
            peakcolor (str, optional): The color of the histogram bars. Defaults to "blue".
            savefig (bool, optional): If True, saves the figure to the provided path. Defaults to False.
            path (str, optional): Path where the figure should be saved. Used only if savefig is set to True. Defaults to None.
        """

        fig = plt.figure()
        plt.tight_layout()

        if self.peaks == 'all':
            peakcolor = 'blue'
        elif self.peaks == 'dark':
            peakcolor = 'grey'
        elif self.peaks == 'LED':
            peakcolor = 'olive'

        dark_count = float(len(self.peak_dark_values))
        led_count = float(len(self.peak_led_values)) - dark_count
        self.led_ratio = led_count / dark_count

        A_avg = np.mean(self.peak_values)
        A_avg_err = A_avg * sem(self.peak_values) / np.mean(self.peak_values)

        # # txtstr = f' Range: {yrange}\n Offset: {offset}\n'
        # textstr =  f'LED Voltage: {led_voltage}\n'

        textstr = f'Date: {self.info.date}\n'
        textstr += f'Condition: {self.info.condition}\n'
        textstr += f'Bias: {self.info.bias} [V]\n'
        textstr += f'RTD4: {self.info.temperature} [K]\n'
        textstr += f'--\n'
        textstr += f'Peaks: {self.peaks}\n'
        textstr += f'LED Ratio: {round(self.led_ratio,2)}\n'
        textstr += f'Avg Amp: {A_avg:.3} ± {A_avg_err:.3} V\n'
        textstr += f'--\n'
        textstr += f'Peak Locations ($\\mu$) [V]:\n'
        for peak in range(len(self.peak_sigmas)):
            actual_peak = self.peak_range[0] + peak
            # actual_peak = peak + 1
            if self.peak_fit.params['g' + str(actual_peak) + '_center'].stderr is not None:
                textstr += f'''Peak {actual_peak}: {self.peak_fit.params['g' + str(actual_peak) + '_center'].value:0.2} $\\pm$ {self.peak_fit.params['g' + str(actual_peak) + '_center'].stderr:0.2}\n'''
        textstr += f'--\n'
        textstr += 'Peak Width (\u03C3) [V]:\n'
        for peak in range(0,len(self.peak_sigmas)):
            actual_peak = self.peak_range[0] + peak
            curr_sigma_err = self.peak_fit.params['g' + str(actual_peak) + '_sigma'].stderr
            if curr_sigma_err is not None:
                textstr += f'''{peak + 1}: {round(self.peak_sigmas[peak],5)} $\\pm$ {curr_sigma_err:0.2}\n'''
        textstr += f'--\n'
        # textstr += 'Peak Amplitude (A)\n'
        textstr += 'Peak Height [Counts]:\n'
        for peak in range(len(self.peak_sigmas)):
            actual_peak = self.peak_range[0] + peak
            # amp = self.peak_fit.params['g' + str(actual_peak) + '_amplitude'].value
            # amp = self.peak_amps[peak]
            # amp_err = self.peak_fit.params['g' + str(actual_peak) + '_amplitude'].stderr
            # amp_err = self.peak_amp_errs[peak]
            # amp_height = amp / (self.peak_sigmas[peak]*np.sqrt(2*np.pi))
            amp_height = self.peak_heights[peak]
            # sigma_err = self.peak_fit.params['g' + str(actual_peak) + '_sigma'].stderr
            # amp_height_err = amp_height * np.sqrt((amp_err/amp)**2 +
            #                                       (sigma_err/self.peak_sigmas[peak])**2)
            amp_height_err = self.peak_height_errs[peak]
            # textstr += f'''{peak + 1}: {amp:0.4} $\\pm$ {amp_err:0.4}\n'''
            textstr += f'''{peak + 1}: {amp_height:0.4} $\\pm$ {amp_height_err:0.4}\n'''
        # for peak in range(len(self.peak_heights)):
        #     ratio = self.peak_heights
        textstr += f'--\n'
        if self.background_linear:
            textstr += f'''Linear Intercept: {self.peak_fit.best_values['l_intercept']:0.4}\n'''
            textstr += f'''Linear Slope: {self.peak_fit.best_values['l_slope']:0.4}\n'''
        else:
            textstr += f'''Exp Amplitude: {self.peak_fit.best_values['e_amplitude']:0.4}\n'''
            textstr += f'''Exp Decay: {self.peak_fit.best_values['e_decay']:0.4}\n'''
        textstr += f'--\n'
        textstr += f'''Reduced $\\chi^2$: {self.peak_fit.redchi:0.2}'''

        curr_hist = np.histogram(self.peak_values, bins=self.numbins)
        counts = curr_hist[0]
        bins = curr_hist[1]
        centers = (bins[1:] + bins[:-1])/2
        y_line_fit = self.peak_fit.eval(x=centers)

        plt.plot(centers, y_line_fit,'r-', label='best fit')
        if self.background_linear:
            background_fit = self.peak_fit.best_values['l_intercept'] + self.peak_fit.best_values['l_slope']*centers
        else:
            background_fit = self.peak_fit.best_values['e_amplitude'] * np.exp(-centers/self.peak_fit.best_values['e_decay'])
        plt.plot(centers, background_fit, 'b-', label='best fit - line')
        # total_num_bins = round(np.sqrt(len(self.peak_values)))
        total_num_bins = round(np.sqrt(len(self.all_peaks)))
        plt.hist(self.peak_values, bins=total_num_bins) #zoom
        # plt.hist(self.peak_values, bins = self.numbins, color = 'tab:' + peakcolor) #zoom
        # total_num_bins = round(np.sqrt(len(self.all_peaks)))
        # plt.hist(self.all_peaks, bins = int(total_num_bins), color = 'tab:' + peakcolor)
        plt.grid(True)

        props = dict(boxstyle='round', facecolor='tab:' + peakcolor, alpha=0.7)
        fig.text(0.67, 0.85, textstr, fontsize=6, verticalalignment='top', bbox=props)
        plt.ylabel('Counts', loc='top')
        plt.xlabel('Pulse Amplitude [V]', loc='right')
        if log_scale:
            plt.ylim(1E-1)
            plt.yscale('log')

        if savefig:
            plt.savefig(path)
            plt.close(fig)
        else:
            plt.show()

    def plot_spe(
        self,
        with_baseline: bool = True,
        fit_origin: bool = True,
        baselinecolor: str = "orange",
        peakcolor: str = "blue",
        savefig: bool = False,
        path: Optional[str] = None,
    ) -> None:
        """Plots average pulse amplitudes as a function of # of Photoelectrons (PE).

        Args:
            with_baseline (bool, optional): If True, plots the baseline data. Defaults to True.
            fit_origin: Constrain linear fit to go through origin.
            baselinecolor (str, optional): Color used for the baseline data. Defaults to "orange".
            peakcolor (str, optional): Color used for the SPE peak data. Defaults to "blue".
            savefig (bool, optional): If True, saves the figure to the provided path. Defaults to False.
            path (str, optional): Path where the figure should be saved. Used only if savefig is set to True. Defaults to None.
        """
        fig = plt.figure()
        fig.tight_layout()

        plt.rc("font", size=12)
        plt.errorbar(
            self.spe_num,
            self.peak_locs[: self.peak_range[1]],
            yerr=self.peak_stds[: self.peak_range[1]],
            fmt=".",
            label="Photoelectron Peaks",
            color="tab:" + peakcolor,
            markersize=10,
        )
        if with_baseline:
            # if self.run_info_pre_bd:
            plt.errorbar(
                0,
                self.baseline_mean,
                yerr=self.baseline_err,
                fmt=".",
                label="Pre-breakdown Baseline",
                color="tab:" + baselinecolor,
                markersize=10,
            )
            # else:
            # plt.errorbar(0, self.baseline_mode, yerr = self.baseline_err, fmt='.', label = 'Solicited Baseline Peak', color = 'tab:' + baselinecolor, markersize = 10)

        b = 0 if fit_origin else self.spe_res.params["intercept"].value
        m = self.spe_res.params["slope"].value
        x_values = np.linspace(0, len(self.spe_num) + 1, 20)
        y_values = m * x_values + b
        plt.plot(
            x_values,
            y_values,
            "--",
            color="tab:" + peakcolor,
            label="Linear Fit",
        )
        #dely = self.spe_res.eval_uncertainty(x=x_values, sigma=1)
        #plt.fill_between(x_values, y_values+dely, y_values-dely)
        #plt.plot(self.spe_num, self.spe_res.best_fit, 'r', label='Self-Triggered Fit')

        plt.xlabel("Photoelectron Peak Number", loc='right')
        plt.ylabel("Peak Location [V]", loc='top')
        plt.legend()
        plt.grid(True)

        textstr = f"Date: {self.info.date}\n"
        textstr += f"Condition: {self.info.condition}\n"
        textstr += f"Bias: {self.info.bias:0.4} [V]\n"
        textstr += f"RTD4: {self.info.temperature} [K]\n"
        textstr += f"--\n"
        textstr += f"""Slope: {self.spe_res.params['slope'].value:0.4} $\\pm$ {self.spe_res.params['slope'].stderr:0.2} [V/pe]\n"""
        textstr += f"""Reduced $\\chi^2$: {self.spe_res.redchi:0.4}\n"""
        # textstr += f"--\n"
        # if self.run_info_pre_bd:
        textstr += (
            f"Baseline: {self.baseline_mean:0.4} +- {self.baseline_err:0.2} [V]"
        )

        props = dict(boxstyle="round", facecolor="tab:" + peakcolor, alpha=0.4)
        fig.text(0.6, 0.48, textstr, fontsize=8, verticalalignment="top", bbox=props)
        fig.tight_layout()

        if savefig:
            plt.savefig(path)
            plt.close(fig)
        else:
            plt.show()
