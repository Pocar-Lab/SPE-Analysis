"""
Silicon PhotoMultipler Single PhotoElectron Analysis

Characterizes silicon photomultipliers (SiPMs) from pulse height spectra.
Loads a directory of waveform acquisitions from HDF5 files collected at
different bias voltages. Pulse amplitudes are extracted and the resulting pulse
height spectra is fit to determine the single photoelectron (SPE) amplitude,
from which gain and breakdown voltage are calculated.

Created on Oct 25 2026

@author: Edryd van Bruggen <ed@edryd.org>
"""

import h5py
import time
import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from scipy import signal
# import pprint
from typing import Any, Optional
from scipy.stats import sem
from uncertainties import ufloat
import os
from dataclasses import dataclass
from multiprocessing import Pool

from lmfit.models import LinearModel, GaussianModel, ExponentialModel
from matplotlib.offsetbox import AnchoredText
from scipy.stats import sem
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def get_mode(hist_data: list or np.array) -> tuple[float, float]:
    """Get the mode of histogram data

    Args:
        hist_data (list or np.array): Histogram data

    Returns:
        tuple[float, float]: value (in volts), number of counts
    """
    counts = hist_data[0]
    bins = hist_data[1]
    centers = (bins[1:] + bins[:-1]) / 2.0
    max_index = np.argmax(counts)
    return centers[max_index], np.amax(counts)

def get_waveforms(file: str):
    with h5py.File(file, 'r') as hdf: # Find bias for each file
        waveforms = []
        for acq_name in hdf['RunData']:
            acq = hdf['RunData'][acq_name]
            # TODO append columns since 0 col is time which is same
            waveforms.append(acq[:])
            # self.data.append(curr_data[:, 1:])
            # self.time.append(curr_data[:, 0])

            # Get last bias voltage in case it was incorrectly entered at first aquiations
            bias = acq.attrs['Bias(V)']
            start_time = acq.attrs['AcquisitionStart']
            trig = acq.attrs['LowerLevel']
            yrange = acq.attrs['Range']
            offset = acq.attrs['Offset']
    return waveforms, bias, start_time, trig, yrange, offset

def process_amp(amp: np.array, fs, upper_limit):
    """Process amplitudes before peaks are found.

    Applies upper limit cut, baseline correction, and filtering"""

    # Skip waveforms which go out of the oscilloscope range, such as an alpha pulse
    if np.amax(amp) > upper_limit:
        return [], []

    # if self.baseline_correct:
    #     if self.poly_correct:
    amp -= peakutils.baseline(amp, deg=2)

    use_bins = np.linspace(-upper_limit, upper_limit, 1000) # TODO why this binning
    curr_hist = np.histogram(amp, bins=use_bins)
    baseline_level, _ = get_mode(curr_hist)
    # self.baseline_levels.append(baseline_level)
    # self.baseline_mode = baseline_level
    # print('baseline:', baseline_level)
    amp -= baseline_level

    # TODO why check shape? not done for solicited
    # if self.do_filter and np.shape(amp) != (0,):
    sos = signal.butter(3, 4E5, btype='lowpass', fs=fs, output='sos') # SPE dark/10g
    filtered = signal.sosfilt(sos, amp)
    amp = filtered

    return amp, baseline_level

def get_peaks(waveforms, filter=True, upper_limit=None, prominence=0.0005, plot_waveforms=False):

    all_peaks = { 'all': [], 'dark': [], 'led': [], 'baseline': [] }

    for wf in waveforms:
        data = wf[:, 1:]
        time = wf[:, 0]
        window_length = time[-1] - time[0]
        num_points = float(len(time))
        fs = num_points / window_length

        # if self.num_waveforms == 0:
        #     print('reading all waveforms')
        #     num_wavefroms = np.shape(data)[1]
        # else:
        #     num_wavefroms = self.num_waveforms
        #     print(f'reading {num_wavefroms} waveforms')
        print(f"{np.shape(data)=}")
        num_wavefroms = np.shape(data)[1]

        for idx in range(num_wavefroms):
            # TODO better way to do this
            # idx = idx + 8000 # uncomment if plotting waveforms and want to see waveforms at different indices
            if idx % 1000 == 0:
                print(f'{idx} read so far')

            # TODO split out to be called in analysis script
            amp, baseline_level = process_amp(data[:, idx], fs, upper_limit)
            all_peaks['baseline'].append(baseline_level)
            # waveform.baseline_mode = baseline_level

            # # If processing prebreakdown baseline, add all amplitudes and don't find peaks
            # if self.is_pre_bd or len(amp) == 0:
            #     all_peaks += list(amp) # amp[100:]
            #     continue

            peak_search_params = {
                "height": 0.0,  # SPE
                "threshold": None,  # SPE
                "distance": None,  # SPE
                "prominence": prominence, #specified by user
                "width": None,  # SPE
                "wlen": 100,  # SPE
                "rel_height": None,  # SPE
                "plateau_size": None,  # SPE
            }
            peaks, props = signal.find_peaks(amp, **peak_search_params)
            # peaks = peaks[peaks > lower_cutoff]

            # # TODO split into different method
            # if plot_waveforms:
            #     plt.tight_layout()
            #     if len(peaks) > 0: # only plot peaks
            #         plt.plot(time,amp)
            #         for peak in peaks:
            #             plt.plot(time[peaks], amp[peaks], '.')
            #     plt.show()

            # TODO exlcude peaks < low bound
            # peak_hist[len(peaks)] = peak_hist.get(len(peaks), 0) + 1

            led_time_thresh = (time[-1] + time[1]) / 2.0
            for peak in peaks:
                all_peaks['all'].append(amp[peak])
                curr_time = time[peak]
                if curr_time < led_time_thresh:
                    all_peaks['dark'].append(amp[peak])
                else:
                    all_peaks['led'].append(amp[peak])

        # print(f'peak_hist[{self.bias}] = {peak_hist}')
        # print(f'all  peaks: {np.mean(all_peaks):.3} ± {np.std(all_peaks,ddof=1)/np.sqrt(len(all_peaks)):.3}')
        # if not self.is_pre_bd:
        #     print(f'dark peaks: {np.mean(dark_peaks):.3} ± {np.std(dark_peaks,ddof=1)/np.sqrt(len(all_peaks)):.3}')
        #     print(f'LED  peaks: {np.mean(led_peaks):.3} ± {np.std(led_peaks,ddof=1)/np.sqrt(len(led_peaks)):.3}')

        if plot_waveforms:
            break

        # # TODO: call in analysis script or measurement info?
        # if is_pre_bd:
        #     self.get_baseline()

    return all_peaks

def fit_baseline(values: list[float], binnum: int = 50):
    """
    Fits a Gaussian model to the provided data using the lmfit.models.GaussianModel method.
    This function also allows to handle alpha-type peak fitting based on the 'alpha' parameter.

    Args:
        values (List[float]): A list of values to which a Gaussian function will be fitted.
        binnum (int, optional): The number of bins to use for histogramming the data. Defaults to 50.
        alpha (bool, optional): If True, the function assumes the data represents an alpha
                                 peak, and sets fitting range accordingly. If False, the function
                                 estimates the fitting range based on the data's standard deviation.
                                 Defaults to False.

    Returns:
        Dict[str, Union[float, Any]]: A dictionary containing the center, low, and high bounds of the
                                       fitting range, as well as the fitted model result.
    """
    std_guess = np.std(values)
    center = np.mean(values)
    low = center - 2.0 * std_guess
    high = center + 2.0 * std_guess
    peaks = np.array(values)
    peaks = peaks[(peaks >= low) & (peaks <= high)]
    counts, bins = np.histogram(peaks, bins=int(np.sqrt(len(peaks))))
    # plt.hist(peaks, bins=binnum)
    centers = (bins[1:] + bins[:-1]) / 2

    model = lm.models.GaussianModel()
    params = model.make_params(
        amplitude=np.amax(counts), center=np.mean(peaks), sigma=np.std(values)
    )
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1 / counts))
    # plt.step(centers, counts, where = 'mid')
    # plt.plot(centers, res.eval(params = res.params, x = centers), '--')
    return res
    # return {'center': np.mean(values), 'low': np.amin(values), 'high': np.amax(values), 'fit': res}

    # TODO mv to plot_baseline
    # # TODO scale x axis to be units of uV
    # plt.hist(peaks, bins=bins)
    # plt.plot(
    #     centers,
    #     res.eval(params=res.params, x=centers),
    #     # color="red",
    #     label=rf"$\mu = {res.values['center']*1e6:.3g} \pm {res.params['center'].stderr*1e6:.3g} \mu V, \sigma = {res.values['sigma']*1e6:.3g} \pm {res.params['sigma'].stderr*1e6:.3g} \mu V$",
    # )
    # plt.legend()
    # plt.show()


@dataclass
class Measurement:
    # Metadata
    bias: float
    start_time: str
    trig: float
    yrange: float
    offset: float

    # Multi-Gaussian
    peaks: dict
    peak_fit: Optional[lm.model.ModelResult] = None

    # SPE Fit
    # TODO generalize for alpha pulse, eg rm and ret vals from funcs
    spe_fit: Optional[lm.model.ModelResult] = None
    spe_amp: Optional[ufloat] = None
    spe_gain: Optional[ufloat] = None

    @classmethod
    def from_file(cls, f) -> "Measurement":
        waveforms, bias, start_time, trig, yrange, offset = get_waveforms(f)
        print(f"Finding peaks for {bias}V ...")
        peaks = get_peaks(waveforms, upper_limit=yrange - offset - 0.001)
        print(f"Found {np.size(peaks)} for {bias}V")
        return cls(bias, start_time, trig, yrange, offset, peaks)


# TODO auto cache measurements peaks
# TODO plot waveforms
class Analysis:
    def __init__(self, path, calibration: ufloat,
                 condition="", temperature=273,
                 baseline_cutoff=25):
        self.calibration = calibration
        self.condition = condition
        self.temperature = temperature

        start_time = time.time()

        # Collection Measurments
        files = [ os.path.join(path, f) for f in os.listdir(path)
                    if os.path.isfile(os.path.join(path,f)) ]
        with Pool(len(files)) as p:
            self.measurements = p.map(Measurement.from_file, iter(files))

        # Sort files by bias voltage
        self.measurements.sort(key=lambda x: x.bias)
        self.date = self.measurements[0].start_time # TODO convert to date

        # self.avg_amp = {
        #     'all':  avg_amp(self.peaks['all']),
        #     'dark': avg_amp(self.peaks['dark']),
        #     'led':  avg_amp(self.peaks['led']),
        # }

        if self.measurements[0].bias <= baseline_cutoff:
            # run_spe_pre_bd = ProcessWaveforms([path+files[0]], do_filter=True,
            #                                   is_pre_bd=True, upper_limit=-1, baseline_correct=True)
            self.baseline = self.measurements[0]
            self.measurements = self.measurements[1:]
            baseline_fit = fit_baseline(self.baseline.peaks['all']) # TODO use basline instead?
        else: # If no pre-breakdown data collected, calculate baseline from SPE data
            self.baseline = None
            baseline_fit = fit_baseline(self.measurements[0].peaks['all']) # TODO use basline instead?
        # self.baseline_mean = baseline_fit.values["center"]
        # self.baseline_err  = baseline_fit.params["center"].stderr
        self.baseline_mean = ufloat(baseline_fit.values["center"],
                                    baseline_fit.params["center"].stderr)
        # self.baseline_std  = baseline_fit.values["sigma"]
        # self.baseline_std_err  = baseline_fit.params["sigma"].stderr
        self.baseline_sigma = ufloat(baseline_fit.values["sigma"],
                                        baseline_fit.params["sigma"].stderr)
        # self.baseline_mean = np.mean(self.peaks['baseline'])
        # self.baseline_err  = sem(self.peaks['baseline'])
        # self.baseline_std  = 0.25 * np.std(self.peaks['baseline']) # TODO why 1/4 ??
        # self.baseline_rms = run_info_self.baseline_mode_rms

        end_time = time.time()
        time_elapsed =  end_time - start_time
        print(f'Time Elapsed: {time_elapsed:.2f} seconds')


    ### Multi-Gaussian

    def plot_multigaussian(self, span=None, subtracted=False, zoom=False, save_path=None) -> None:
        """Plot preliminary histograms of file. Used to find initial guess of first PE value"""
        for m in self.measurements[slice(*span) if isinstance(span, tuple) else slice(span)]:
            # print(f"{m=}")
            fig, ax = plt.subplots()
            # bins = int(np.sqrt(len(m.peaks['all'])))
            bins = np.arange(0, m.yrange, m.yrange/2**12) # ZTEC ZT4421 resolution 12 bits

            # Plot pulse height spectrum
            if subtracted:
                plt.hist(m.peaks['led'], bins=bins, histtype="step", density=False, label="LED")
                plt.hist(m.peaks['dark'], bins=bins, histtype="step", density=False, label="No LED")
                # centers = np.histogram_bin_edges(m.peaks, bins=bins)
                counts_led, b = np.histogram(m.peaks['led'], bins=bins)
                counts_dark, b = np.histogram(m.peaks['dark'], bins=bins)
                counts = counts_led - counts_dark

                centers = (b[1:] + b[:-1])/2
                plt.step(centers, counts, label = 'Subtracted')
            else:
                plt.hist(m.peaks['all'], bins=bins, density=False, label="Data")
                # centers = np.histogram_bin_edges(peaks, bins=bins)
                counts, b = np.histogram(m.peaks['all'], bins=bins)
                centers = (b[1:] + b[:-1])/2

            # Plot Multi-Gaussian fit
            if m.peak_fit:
                y_line_fit = m.peak_fit.eval(x=centers)
                plt.plot(centers, y_line_fit, label='Multi-Gaussian Fit')

                bv = m.peak_fit.best_values
                background_fit = bv['e_amplitude'] * np.exp(-centers/bv['e_decay'])
                plt.plot(centers, background_fit, label='Baseline Guassian Fit')

                # noise_fit = gaussian(centers, bv['noise_amplitude'], bv['noise_center'], bv['noise_sigma'])
                # plt.plot(centers, noise_fit, label='best fit - line')

                textstr = f"Date: {self.date}\n"
                textstr += f"Condition: {self.condition}\n"
                textstr += f"Bias: {m.bias:0.4} V\n"
                textstr += f"Temperature: {self.temperature} K\n"
                # textstr += f"""SPE Amplitude: {self.spe_res.params['slope'].value:0.4} $\\pm$ {self.spe_res.params['slope'].stderr:0.2} [V/pe]\n"""
                textstr += f"""Reduced $\\chi^2$: {m.peak_fit.redchi:0.4}"""
                # if self.run_info_pre_bd:
                # textstr += (
                #     f"Baseline: {self.baseline_mean:0.4} ± {self.baseline_err:0.2} [V]"
                # )
                text_box = AnchoredText(textstr, frameon=True, loc="lower right", pad=0.5)
                plt.setp(text_box.patch, facecolor='white', edgecolor="black", boxstyle="round", alpha=0.9)
                ax.add_artist(text_box)

            plt.xlabel("Amplitude (V)", loc='right')
            plt.ylabel("Counts", loc='top')
            plt.subplots_adjust(top=0.9)
            if zoom:
                plt.xlim(0,.02)
            plt.ylim(1)
            plt.yscale("log")
            plt.legend(loc="upper right")
            plt.tight_layout()

            # bin_max = np.max(m.peaks['all'])
            # bin_max = .2
            bin_max = b[np.where(counts == 0)[0][0]] + 0.05
            # print(f"{bin_max=}")
            plt.xlim(0, bin_max)

            if save_path is not None:
                plt.savefig(save_path.replace("%b", str(m.bias)))
                plt.close(fig)
            else:
                plt.show()

    def fit_multigaussian(
        self,
        first_guesses: list[float],
        peak_ranges: list[tuple[int,int]],
        cutoffs: list[tuple[float, float]],
    ) -> lm.model.ModelResult:
        """
        Fits multiple Gaussian functions to a 'finger plot' with a linear or exponential background.

        Args:
            firt_pe (List[float]): Initial guesses for the centroid of each Gaussian.
            peak_range (tuple[int, int]): The number of peaks you want to fit. Defaults to 4.
            cutoff (Tuple[float, float], optional): Low and high cutoff values in mV. Defaults to (0, np.inf).

        Returns:
            ModelResult: An lmfit ModelResult object containing all fit information.
        """
        # TODO optional peak_ranges and cutoffs
        # TODO increment peak_ranges to find max number
        # TODO auto determine cutoff from first minimal, or make uncesarcy with subtraction
        # cutoff = (0, np.inf)
        for m, first_pe, peak_range, cutoff in zip(self.measurements, first_guesses, peak_ranges, cutoffs):
            values = np.array(m.peaks['all']) # TODO fit to subtracted
            # TODO before or after?
            peaks = values[(values >= cutoff[0]/1e3) & (values <= cutoff[1]/1e3)]
            # FIXME sometimes number of bins is too small to find peaks
            # bins = int(np.sqrt(len(peaks)))
            # bins = np.histogram_bin_edges(peaks, bins=numbins)

            bins = np.arange(0, m.yrange, m.yrange/2**12) # ZTEC ZT4421 resolution 12 bits

            counts_led, b = np.histogram(m.peaks['led'], bins=bins)
            counts_dark, b = np.histogram(m.peaks['dark'], bins=bins)
            counts = counts_led - counts_dark

            # counts, b = np.histogram(peaks, bins=bins)
            bin_centers = (b[1:] + b[:-1]) / 2
            # bin_centers = np.histogram_bin_edges(peaks, bins=bins)

            assert len(peaks) != 0
            assert len(counts) != 0
            assert len(bin_centers) != 0

            low_peak = peak_range[0]
            high_peak = peak_range[1]

            # model  = GaussianModel(prefix="noise_")
            model = ExponentialModel(prefix='e_')
            for peak in range(low_peak, high_peak + 1):
                model += GaussianModel(prefix=f'g{peak}_')

            # # Constraints for noise
            # if noise_mean:
            #     model.set_param_hint("noise_center", value=noise_mean, vary=False)
            # if noise_sigma:
            #     model.set_param_hint("noise_sigma", value=noise_sigma, vary=False)
            # print(f"{np.max(counts)}")
            # model.set_param_hint("noise_amplitude", value=np.max(counts)*np.sqrt(np.pi)*noise_sigma,
            #                      vary=False)

            # Constraints for background
            model.set_param_hint('e_decay', value=.01, min=0, max=1)
            model.set_param_hint('e_amplitude', value=61, min=0)

            # Constraints for gaussian centers
            pe_centers = [first_pe * peak / 1e3 for peak in range(peak_range[0], peak_range[1]+1)]
            # pe_centers = centers[:(peak_range[1]-peak_range[0]+1)]
            print('Center guesses used in fit: ', pe_centers)
            for peak in range(low_peak, high_peak + 1):
                    model.set_param_hint(f'g{peak}_center', value=pe_centers[peak-1],
                                        min=pe_centers[peak-1] - 5*self.baseline_sigma.n,
                                        max=pe_centers[peak-1] + 5*self.baseline_sigma.n)

            # Constraints for gaussian sigmas
            # baseline_std = .5*baseline_width
            # baseline_std = .0012
            for peak in range(low_peak, high_peak + 1):
                # model.set_param_hint(f'g{peak}_sigma', value=2*baseline_std,
                #                      min=.50*baseline_std, max=2.*baseline_std)
                # model.set_param_hint(f'g{peak}_sigma', value=0.5 * baseline_width,
                #                      # min=.3*baseline_width, max=.7*baseline_width)
                #                      min=.2*baseline_width, max=.8*baseline_width)
                #                      # min=0, max=baseline_width)
                model.set_param_hint(f'g{peak}_sigma', value=self.baseline_sigma.n,
                                    # min=baseline_sigma.n - 5*baseline_sigma.s,
                                    # max=baseline_sigma.n + 5*baseline_sigma.s
                                    )

            # Constraints for gaussian amplitudes
            g_amplitude = [np.amax(counts)*np.sqrt(2*np.pi)*self.baseline_sigma.n/(2**num)
                        for num in range(low_peak, high_peak + 1)]
            for peak in range(low_peak, high_peak + 1):
                model.set_param_hint(f'g{peak}_amplitude', value=g_amplitude[peak-1], min=0)
            # assert not np.any(np.isinf(g_amplitude)) and not np.any(np.isnan(g_amplitude))

            params = model.make_params()
            # params.pretty_print()
            y_model = model.eval(params, x=bin_centers)
            assert y_model.size != 0

            m.peak_fit = model.fit(counts, params=params, x=bin_centers, weights=1/np.sqrt(counts), nan_policy='omit')
            print(m.peak_fit.fit_report())
            if m.peak_fit.aborted:
                print("Multi-Gaussian fit failed!!")
            if not m.peak_fit.errorbars:
                print("Multi-Gaussian fit could not calculate error bars!!")
            print(f"{m.peak_fit.success=}")
        # TODO print error if any fit fails or has no err bars

    ### SPE Amplitude

    # TODO option to now show
    # TODO option to loop over at same time as plot_multigaussian
    def plot_spe(
        self,
        peak_ranges: tuple,
        fit_origin: bool = False,
        save_path: Optional[str] = None,
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
        spe_amps = []
        for m, peak_range in zip(self.measurements, peak_ranges):
            # TODO rename to better desc
            spe_num = np.arange(peak_range[0], peak_range[1]+1)

            if not m.peak_fit:
                raise ValueError("Multi-Gaussian not fit yet, run `fit_multigaussian` first")

            peak_locs = [m.peak_fit.params[f'g{idx}_center'].value for idx in spe_num]
            peak_locs_err = [m.peak_fit.params[f'g{idx}_center'].stderr for idx in spe_num]
            peak_sigmas = [m.peak_fit.params[f'g{idx}_sigma'].value for idx in spe_num]
            peak_sigmas_err = [m.peak_fit.params[f'g{idx}_sigma'].stderr for idx in spe_num]

            fig, (ax_spe, ax_sig) = plt.subplots(2, 1, figsize=(8, 6),
                                                gridspec_kw={'height_ratios': [2, 1]},
                                                sharex=True)
            ax_spe.errorbar(
                spe_num,
                peak_locs,
                yerr=peak_locs_err,
                fmt=".",
                label="Photoelectron Peaks",
                markersize=10,
            )
            ax_spe.errorbar(
                0,
                self.baseline_mean.n,
                yerr=self.baseline_mean.s,
                fmt=".",
                label="Pre-breakdown Baseline",
                markersize=10,
            )

            # Create linear fit model
            model = lm.models.LinearModel()
            params = model.make_params()
            spe_fit = model.fit(
                peak_locs,
                params=params,
                x=spe_num,
                weights=[1.0 / curr_std for curr_std in peak_locs_err]
            )
            # TODO should be not?
            b = 0 if fit_origin else spe_fit.params["intercept"].value
            slope = spe_fit.params["slope"].value
            x = np.linspace(0, peak_range[1]+1, 20)
            y = slope * x + b
            ax_spe.plot(
                x,
                y,
                "--",
                label="Linear Fit",
            )
            # TODO show error on fit
            #dely = self.spe_res.eval_uncertainty(x=x, sigma=1)
            #ax_spe.fill_between(x, y+dely, y-dely)
            #ax_spe.plot(spe_num, self.spe_res.best_fit, 'r', label='Self-Triggered Fit')

            textstr = f"Condition: {self.condition}\n"
            textstr += f"Bias: {m.bias:0.4} [V]\n"
            textstr += f"Temperature: {self.temperature} [K]\n"
            textstr += f"Date: {self.date}\n"
            textstr += (
                f"Baseline: {self.baseline_mean:0.4} ± {self.baseline_sigma:0.2} [V]"
            )
            textstr = f"""SPE Amplitude: {spe_fit.params['slope'].value:0.4} \
$\\pm$ {spe_fit.params['slope'].stderr:0.4} V
Y-Intercept: {spe_fit.params['intercept'].value:0.4} \
$\\pm$ {spe_fit.params['intercept'].stderr:0.4} V
Reduced $\\chi^2$: {spe_fit.redchi:0.4}\
            """
            text_box = AnchoredText(textstr, frameon=True, loc="lower right", pad=0.5)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5, edgecolor="black", boxstyle="round")
            ax_spe.add_artist(text_box)
            # props = dict(boxstyle="round", facecolor="tab:" + peakcolor, alpha=0.4)
            # fig.text(0.6, 0.48, textstr, fontsize=8, verticalalignment="top", bbox=props)

            ax_sig.errorbar(
                spe_num,
                peak_sigmas,
                yerr=peak_sigmas_err,
                fmt=".",
                label="Photoelectron Peaks Width",
                markersize=10,
            )
            ax_sig.errorbar(
                0,
                self.baseline_sigma.n,
                yerr=self.baseline_sigma.s,
                fmt=".",
                label="Pre-breakdown Baseline Width",
                markersize=10,
            )

            textstr = f"""\
Width Average: {np.mean(peak_sigmas):.4} $\\pm$ {np.mean(peak_sigmas_err):.4} V
Width Spread: {np.std(peak_sigmas):.4} $\\pm$ {np.std(peak_sigmas_err):.4} V"""
            text_box = AnchoredText(textstr, frameon=True, loc="lower right", pad=0.5)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5, edgecolor="black", boxstyle="round")
            ax_sig.add_artist(text_box)

            plt.xlabel("Photoelectron Peak Number", loc='right')
            ax_spe.set_ylabel("Peak Location [V]", loc='top')
            ax_sig.set_ylabel("Peak Width [V]", loc='top')
            ax_spe.set_ylim(0)
            ax_spe.legend()
            ax_spe.grid(True)
            plt.grid(True)
            plt.setp(ax_spe.get_xticklabels(), visible=False)
            fig.tight_layout()

            if save_path is not None:
                plt.savefig(save_path.replace("%b", str(m.bias)))
                plt.close(fig)
            else:
                plt.show()
            m.spe_amp = ufloat(spe_fit.params['slope'].value, spe_fit.params['slope'].stderr)
            spe_amps.append(m.spe_amp)
        return spe_amps

    # Breakdown Voltage

    def plot_breakdown(self, save_path=None):
        if not self.measurements[0].spe_amp:
            print('Need to calculate SPE amplitude first')

        biases = [m.bias for m in self.measurements]
        biases_err = [0.0025 * bias + 0.015 for bias in biases]

        spe_amps_val = [m.spe_amp.n for m in self.measurements]
        spe_amps_err = [m.spe_amp.s for m in self.measurements]
        spe_wgts = [1.0 / m.spe_amp.s for m in self.measurements]

        # spe_gains = [m.spe_amp/self.calibration/e_charge for m in self.measurements]

        fig, ax = plt.subplots()

        model = lm.models.LinearModel()
        params = model.make_params()
        spe_res = model.fit(spe_amps_val, params=params, x=biases, weights=spe_wgts)
        b_spe = ufloat(spe_res.params["intercept"].value, spe_res.params["intercept"].stderr)
        m_spe = ufloat(spe_res.params["slope"].value, spe_res.params["slope"].stderr)
        print(f"{b_spe=}, {m_spe=}")
        print(f"{spe_res.covar=}")
        v_bd = -b_spe / m_spe

        fit_bias = np.linspace(v_bd.n, np.amax(biases) + 1, 20)
        fit_y = spe_res.eval(params=spe_res.params, x=fit_bias)
        fit_y_err = spe_res.eval_uncertainty(x=fit_bias, params=spe_res.params, sigma=1) # TODO sigma=1?

        line = plt.plot(fit_bias, fit_y, label="SPE Amplitude Best Fit")
        plt.fill_between(
            fit_bias, fit_y + fit_y_err, fit_y - fit_y_err, alpha=0.5,
            color=line[0].get_color(),
        )
        # TODO plot SPE gain
        plt.errorbar(
            biases,
            spe_amps_val*1e3,
            xerr=biases_err,
            yerr=spe_amps_err,
            markersize=6,
            fmt=".",
            label="SPE Amplitude Values",
        )

        plt.ylim(0)
        plt.xlim(0)
        plt.xlabel("Bias Voltage [V]", loc='right')
        plt.ylabel("SPE Amplitude [V]", loc='top')
        plt.legend()
        plt.tight_layout()

        # textstr = f"Date: {campaign[0].info.date}\n"
        # textstr += f"Condition: {campaign[0].info.condition}\n"
        # textstr += f"RTD4: {campaign[0].info.temperature} [K]\n"
        # if self.filtered:
        #     textstr += f"Filtering: Lowpass, 400kHz\n"
        # else:
        #     textstr += f"Filtering: None\n"
        textstr = f"""\
Slope: {m_spe:0.4} V/V
Intercept: {b_spe:0.4} V
Reduced $\\chi^2$: {spe_res.redchi:0.4}
Breakdown Voltage: {v_bd:0.4} V\
"""
        text_box = AnchoredText(textstr, frameon=True, loc="lower right", pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=0.5)
        ax.add_artist(text_box)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close(fig)
