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
from typing import Any, Dict, List, Tuple, Optional

from RunInfo import *
from RunInfoBaseline import RunInfoBaseline


def get_waveform(w: str) -> tuple[list[float], list[float]]:
    # TODO: getting the axe
    """
    Processes a waveform file, extracting metadata and data points. Metadata is skipped until
    the line 'Time (s)' is found. Subsequent lines are treated as data with the first column
    as time and the second as amplitude.

    Args:
        w (str): Path to the waveform file. The file should be tab-delimited with the first
                 column being time in seconds, and the second column being the amplitude.
                 The first non-metadata line may start with 'Time (s)'.

    Returns:
        tuple: Two lists, first one containing the time data in microseconds, and the second
               one containing the amplitude data, both extracted from the waveform file.
    """
    time = []
    amp = []

    f = open(w, "r")

    metadata = {}
    data = {}

    header = True
    for x in f:
        line = x.split("\t")
        if header:
            if line[0] == "Time (s)":
                header = False
            elif len(line) < 10:
                continue
            else:
                metadata[line[0]] = line[1]
        else:
            t = float(line[0]) * 1e6
            a = float(line[1])
            time.append(t)
            amp.append(a)
    f.close()
    return (time, amp)


# REED DID THIS <3
def get_peaks(
    waveform_dir: str, peak_search_params: dict[str, type[int | float]]
) -> list[float]:
    """
    Searches for peaks in a set of waveforms specified in a directory using parameters for peak
    searching. This function applies signal.find_peaks to each waveform and stores the amplitude
    of each peak.

    Args:
        waveform_dir (str): The directory containing waveform files. Each file should be named
                            with the prefix 'w' and have the extension '.txt'.
        peak_search_params (Dict[str, Union[int, float]]): Parameters to be passed to
                               scipy.signal.find_peaks. These parameters may include
                               properties like 'height', 'threshold', 'distance', etc.,
                               depending on the requirements for a valid peak.

    Returns:
        List[float]: A list containing the amplitudes of all detected peaks across all
                     waveforms in the specified directory.
    """
    waveform_filenames = glob.glob(waveform_dir + "w*.txt")
    all_peaks = []
    for idx, w in enumerate(waveform_filenames):
        if idx % 100 == 0:
            print(idx)
        time, amp = get_waveform(w)

        peaks, props = signal.find_peaks(amp, **peak_search_params)
        for peak in peaks:
            all_peaks.append(amp[peak])
    return all_peaks


def get_peak_waveforms(
    waveform_dir: str, num: int = -1
) -> tuple[list[float], list[float], int]:
    """
    Reads a specified number of waveform files from a directory, and combines their time and
    amplitude values into two lists. Also returns the total number of waveforms processed.

    Args:
        waveform_dir (str): The directory containing waveform files. Each file should be named
                            with the prefix 'w' and have the extension '.txt'.
        num (int, optional): The maximum number of waveform files to process. If this is set to
                             a positive number, only the first 'num' waveforms will be processed.
                             If it is set to -1 (default), all waveforms in the directory will
                             be processed.

    Returns:
        Tuple[List[float], List[float], int]: A tuple containing three elements:
            1. A list of amplitude values combined from all processed waveforms.
            2. A list of corresponding time values combined from all processed waveforms.
            3. The total number of waveforms processed.
    """
    # wfs = fnmatch.filter(os.listdir(filepath), 'w*')
    # read in solicited trigger waveforms
    waveform_filenames = glob.glob(waveform_dir + "w*.txt")
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
    """_summary_

    Args:
        waveform_dir (_type_): _description_
        peak_search_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    # wfs = fnmatch.filter(os.listdir(filepath), 'w*')
    # read in solicited trigger waveforms
    waveform_filenames = glob.glob(waveform_dir + "w*.txt")
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
        if len(peaks) == 0 and np.amin(amp) > -0.25:
            num_w += 1
            values += amp[300:-300]
            times += time[300:-300]
    return values, times, num_w


def save_baseline_csv(waveform_dir, savedir, peak_search_params):
    waveform_data, waveform_times, _ = get_baseline(waveform_dir, peak_search_params)
    data = {"waveform data": waveform_data}
    df = pd.DataFrame(data)
    df.to_csv(savedir)


def save_peaks_csv(waveform_dir, savedir, peak_search_params):
    peaks = get_peaks(waveform_dir, peak_search_params)
    data = {"peaks": peaks}
    df = pd.DataFrame(data)
    df.to_csv(savedir)


def read_data_csv(filename):
    df = pd.read_csv(filename)
    return df


def Gauss(x: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """
    Calculates the value of a Gaussian function at a given point or array of points.

    Args:
        x (Union[float, np.ndarray]): The point(s) at which to evaluate the Gaussian function.
                                       This can be a single float or a numpy array of floats.
        A (float): The height of the Gaussian's peak.
        B (float): The position of the center of the peak.
        C (float): The standard deviation, which controls the width of the peak.

    Returns:
        Union[float, np.ndarray]: The value(s) of the Gaussian function at the given point(s).
    """
    y = A * np.exp(-((x - B) ** 2) / (2 * C * C))
    return y


def fit_gauss(
    values: list[float], range_low: float, range_high: float
) -> lm.model.ModelResult:
    """
    Fits a Gaussian model to the provided data within the specified range using the
    lmfit.models.GaussianModel method.

    Args:
        values (List[float]): A list of values to which a Gaussian function will be fitted.
        range_low (float): The lower bound of the range to consider for fitting.
        range_high (float): The upper bound of the range to consider for fitting.

    Returns:
        lm.model.ModelResult: An instance of the ModelResult class in the lmfit package, containing the results
             of the model fitting process. This includes parameters of the fitted model and
             other statistical information.
    """
    histogram = np.histogram(values, bins=40)
    counts = histogram[0]
    bins = histogram[1]
    centers = (bins[1:] + bins[:-1]) / 2
    model = lm.models.GaussianModel()
    params = model.make_params(
        amplitude=max(counts), center=np.mean(values), sigma=np.std(values)
    )
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1 / counts))
    return res


def fit_baseline_gauss(
    values: list[float], binnum: int = 50, alpha: bool = False
) -> dict[str, type[float | Any]]:
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
    f_range = {}
    if alpha:  # TODO no hardcoded parameters !!
        f_range["low"] = -0.0005
        # f_range['low'] = 0.0
        f_range["high"] = 0.0045
        # f_range['high'] = 0.003
        f_range["center"] = (f_range["high"] + f_range["low"]) / 2.0
    else:
        f_range["center"] = np.mean(values)
        std_guess = np.std(values)
        f_range["low"] = f_range["center"] - 2.0 * std_guess
        f_range["high"] = f_range["center"] + 2.0 * std_guess
    bin_density = float(binnum) / (np.amax(values) - np.amin(values))
    new_binnum = int(bin_density * (f_range["high"] - f_range["low"]))
    limit_values = values[(values >= f_range["low"]) & (values <= f_range["high"])]
    curr_hist = np.histogram(limit_values, bins=new_binnum)
    # plt.hist(values, bins= binnum)
    counts = curr_hist[0]
    bins = curr_hist[1]
    centers = (bins[1:] + bins[:-1]) / 2

    model = lm.models.GaussianModel()
    params = model.make_params(
        amplitude=np.amax(counts), center=np.mean(limit_values), sigma=np.std(values)
    )
    res = model.fit(counts, params=params, x=centers, weights=np.sqrt(1 / counts))
    # plt.step(centers, counts, where = 'mid')
    # plt.plot(centers, res.eval(params = res.params, x = centers), '--')
    f_range["fit"] = res
    # return {'center': np.mean(values), 'low': np.amin(values), 'high': np.amax(values), 'fit': res}
    return f_range


def fit_peaks_multigauss(
    values: np.ndarray,
    baseline_width: float,
    centers: list[float],
    numpeaks: int = 4,
    cutoff: tuple[float, float] = (0, np.infty),
) -> lm.model.ModelResult:
    """
    Fits multiple Gaussian functions to a 'finger plot' made from given values using the
    lmfit.models.GaussianModel method.

    Args:
        values (List[float]): Heights of peaks extracted from waveforms.
        baseline_width (float): An estimate of the width in Volts of the noise.
        centers (List[float]): Initial guesses for the centroid of each Gaussian.
        numpeaks (int, optional): The number of peaks you want to fit. Defaults to 4.
        cutoff (Tuple[float, float], optional): Low and high cutoff values. Defaults to (0, np.infty).

    Returns:
        ModelResult: An lmfit ModelResult object containing all fit information.
    """
    curr_peak_data = values[(values >= cutoff[0]) & (values <= cutoff[1])]
    binnum = round(np.sqrt(len(curr_peak_data)))
    counts, bins = np.histogram(curr_peak_data, bins=binnum)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    model = LinearModel(prefix="l_")

    for i in range(1, numpeaks + 1):
        model += GaussianModel(prefix=f"g{i}_")

    params = model.make_params(
        l_slope=0,
        l_intercept=counts[0],
    )

    peak_scale = max(counts) * np.sqrt(2 * np.pi) * baseline_width
    for i in range(1, numpeaks + 1):
        params[f"g{i}_amplitude"].value = peak_scale / (2**i)
        params[f"g{i}_center"].value = centers[i - 1]
        params[f"g{i}_center"].max = params[f"g{i}_center"].value * 1.3
        params[f"g{i}_center"].min = params[f"g{i}_center"].value * 0.8

        params[f"g{i}_sigma"].value = 0.5 * baseline_width
        params[f"g{i}_sigma"].min = 0.3 * 0.5 * baseline_width
        params[f"g{i}_sigma"].max = 2 * 0.5 * baseline_width

        params[f"g{i}_amplitude"].min = 0

    res = model.fit(counts, params=params, x=bin_centers, weights=1 / np.sqrt(counts))

    print(res.fit_report())
    return res


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


def plot_fit(
    fit_info: Dict[str, lm.model.ModelResult],
    values: np.ndarray,
    binnum: int = 20,
    plot_hists: bool = True,
    label: str | None = None,
) -> None:
    """
    Plots the histogram of the data and the Gaussian fit.

    Args:
        fit_info (Dict[str, ModelResult]): A dictionary containing the range ('low', 'high', 'center') and
                                           the final fit result ('fit') from the Gaussian model.
        values (List[float]): List of values to plot in the histogram.
        binnum (int, optional): The number of bins to use when creating the histogram. Defaults to 20.
        plot_hists (bool, optional): If True, the histogram is plotted. Defaults to True.
        label (str, optional): Label for the Gaussian fit line. Defaults to None.
    """
    fit_data = values[(values >= fit_info["low"]) & (values <= fit_info["high"])]
    numvalues = len(fit_data)
    h = 3.49 * (numvalues) ** (-1 / 3) * np.std(fit_data)
    binnum = int(np.ceil((max(fit_data) - min(fit_data)) / h))
    if plot_hists:
        curr_hist = plt.hist(fit_data, bins=binnum)
    x = np.linspace(fit_info["low"], fit_info["high"], num=200)
    plt.plot(
        x,
        fit_info["fit"].eval(params=fit_info["fit"].params, x=x),
        color="red",
        label=label,
    )


def get_mode(hist_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, int]:
    """
    Determines the mode (the value that appears most often) from histogram data.

    Args:
        hist_data (Tuple[np.ndarray, np.ndarray]): A tuple containing the counts per bin and the bin edges, typically the output of a numpy histogram function.

    Returns:
        Tuple[float, int]: The center of the bin with the highest count (mode) and the maximum count.
    """
    counts = hist_data[0]
    bins = hist_data[1]
    centers = (bins[1:] + bins[:-1]) / 2.0
    max_index = np.argmax(counts)
    return centers[max_index], np.amax(counts)


# takes in measurement info, and processes it at waveform level
# constructs different histograms, and does gaussian fits
class WaveformProcessor:
    def __init__(
        self,
        info: MeasurementInfo,
        run_info_self: Optional[RunInfo] = None,
        run_info_solicit: Optional[RunInfoBaseline] = None,
        baseline_correct: bool = False,
        cutoff: Tuple[float, float] = (0, np.infty),
        offset_num: int = 0,
    ):
        """
        Initializes the WaveformProcessor class with the provided parameters.

        Args:
            info (MeasurementInfo): Class containing info regarding measurement.
            centers (List[float]): Initial guesses for centroid of each gaussian.
            run_info_self (Optional[RunInfo]): RunInfo class containing SPE or Alpha data.
            run_info_solicit (Optional[RunInfo]): RunInfo class containing baseline data.
            baseline_correct (bool): Boolean value indicating if baseline correction needs to be applied. Defaults to False.
            cutoff (Tuple[float, float]): Low and high cutoff values. Defaults to (0,np.infty).
            numpeaks (int): The number of peaks you want to fit. Defaults to 4.
            no_solicit (bool): A flag indicating if there is no solicited waveform available. Defaults to False.
            offset_num (int): The number of peaks to skip in the histogram. Defaults to 0.
        """

        self.baseline_correct = baseline_correct
        self.info = info
        self.run_info_self = run_info_self
        self.cutoff = cutoff

        self.offset_num = offset_num
        # options for if you forgot to take pre-breakdown data.......
        self.baseline_mode = run_info_self.baseline_mode_mean
        # self.baseline_mode = 1 #PLACEHOLDER
        self.baseline_rms = run_info_self.baseline_mode_rms
        self.baseline_std = 0.25 * run_info_self.baseline_mode_std
        self.baseline_err = run_info_self.baseline_mode_err
        self.baseline_rms = run_info_self.baseline_mode_rms
        # self.baseline_std = 1
        # self.baseline_err = 1

    def process_h5(self) -> None:
        """Processes the .h5 files associated with the WaveformProcessor instance.

        The method extracts peak data and, if available, baseline data from .h5 files.
        It filters the peak data based on a predefined cutoff range and also handles solicit data if it's not disabled.
        """
        for curr_file in self.run_info_self.hd5_files:
            for curr_acquisition_name in self.run_info_self.acquisition_names[
                curr_file
            ]:
                # self.peak_values = np.array(self.run_info_self.peak_data[curr_file][curr_acquisition_name])
                self.peak_values = np.array(self.run_info_self.all_peak_data)
                self.peak_values = self.peak_values[
                    (self.peak_values >= self.cutoff[0])
                    & (self.peak_values <= self.cutoff[1])
                ]  # peaks in a range

        if not self.no_solicit:
            for curr_file in self.run_info_solicit.hd5_files:
                for curr_acquisition_name in self.run_info_solicit.acquisition_names[
                    curr_file
                ]:
                    # try:
                    if self.run_info_solicit.specifyAcquisition:
                        curr_acquisition_name = self.run_info_solicit.acquisition
                    # except:
                    else:
                        self.baseline_values = np.array(
                            self.run_info_solicit.peak_data[curr_file][
                                curr_acquisition_name
                            ]
                        )
                    self.baseline_values = np.array(
                        self.run_info_solicit.peak_data[curr_file][
                            curr_acquisition_name
                        ]
                    )

    # reads in the waveform data either from the raw data or from a pre-saved .csv file
    def process(self, overwrite=False, do_spe=True, do_alpha=False):
        """Processes the waveform data, extracting various statistical information from it.

        Args:
            overwrite (bool, optional): If True, any previous processing results are overwritten. Defaults to False.
            do_spe (bool, optional): If True, Single Photoelectron (SPE) data is processed, including fitting multiple peaks and calculating signal-to-noise ratio (SNR). Defaults to True.
            do_alpha (bool, optional): If True, alpha particle data is processed. Defaults to False.
        """
        self.process_h5()

        if do_alpha:
            self.peak_values = self.peak_values[
                self.peak_values > self.info.min_alpha_value
            ]

        self.numbins = int(
            round(np.sqrt(len(self.peak_values)))
        )  #!!! attr defined outside init
        print(f"len: {len(self.peak_values)}")
        print(f"{self.numbins}")
        if self.no_solicit:
            self.baseline_mean = self.baseline_mode
            self.baseline_std = 0.002  # arbitrary
            print("baseline mode: " + str(self.baseline_mode))
            print("baseline std: " + str(self.baseline_std))
        else:
            self.baseline_fit = fit_baseline_gauss(
                self.baseline_values, binnum=self.info.baseline_numbins, alpha=do_alpha
            )
            self.baseline_std = self.baseline_fit["fit"].values["sigma"]
            self.baseline_mean = self.baseline_fit["fit"].values["center"]
            self.baseline_err = self.baseline_fit["fit"].params["center"].stderr
            self.baseline_rms = np.sqrt(np.mean(self.baseline_values**2))
            print("baseline mean: " + str(self.baseline_mean))
            print("baseline std: " + str(self.baseline_std))

        if do_spe:
            self.peak_fit = fit_peaks_multigauss(
                values=self.peak_values,
                baseline_width=2.0 * self.baseline_std,
                centers=self.centers,
                numpeaks=self.numpeaks,
                cutoff=self.cutoff,
            )

            self.peak_locs = [
                self.peak_fit.params[f"g{i}_center"].value
                for i in range(1, numpeaks + 1)
            ]
            self.peak_sigmas = [
                self.peak_fit.params[f"g{i}_sigma"].value
                for i in range(1, numpeaks + 1)
            ]
            self.peak_stds = [
                self.peak_fit.params[f"g{i}_center"].stderr
                for i in range(1, numpeaks + 1)
            ]

            self.peak_wgts = [1.0 / curr_std for curr_std in self.peak_stds]
            self.spe_num = []

            self.resolution = [
                (self.peak_locs[i + 1] - self.peak_locs[i])
                / np.sqrt(self.peak_sigmas[i] ** 2 + self.peak_sigmas[i + 1] ** 2)
                for i in range(len(self.peak_locs) - 1)
            ]
            print("sigma SNR: " + str(self.resolution))

            for idx in range(self.numpeaks):
                self.spe_num.append(float(idx + 1 + self.offset_num))
            # self.peak_locs = sorted(self.peak_locs)

            # linear fit to the peak locations
            model = lm.models.LinearModel()
            params = model.make_params()

            self.spe_res = model.fit(
                self.peak_locs[: self.numpeaks],
                params=params,
                x=self.spe_num,
                weights=self.peak_wgts[: self.numpeaks],
            )  # creates linear fit model
            #

            print(
                "SNR: " + str(self.spe_res.params["slope"].value / self.baseline_mode)
            )
            print(
                "SNR 2-3: "
                + str((self.peak_locs[2] - self.peak_locs[1]) / self.baseline_mode)
            )
            print(
                "SNR 1-2: "
                + str((self.peak_locs[1] - self.peak_locs[0]) / self.baseline_mode)
            )

            if self.baseline_correct:
                self.A_avg = (
                    np.mean(self.peak_values) - self.spe_res.params["intercept"].value
                )  # spectrum specific baseline correction
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
                + (
                    self.spe_res.params["slope"].stderr
                    / self.spe_res.params["slope"].value
                )
                ** 2
            )

        if do_alpha:
            self.alpha_fit = fit_alpha_gauss(
                self.peak_values, binnum=self.info.peaks_numbins
            )
            self.alpha_res = self.alpha_fit["fit"]

    def get_alpha_data(self):
        """Retrieves the alpha pulse peak heights.

        Returns:
            numpy.ndarray: An array of processed alpha particle data.
        """
        return self.peak_values

    def get_baseline_data(self):
        """Retrieves the raw aggregated baseline data.

        Returns:
            numpy.ndarray: An array of processed baseline data values.
        """
        return self.baseline_values

    def get_alpha_fit(self) -> lm.model.ModelResult:
        """Retrieves the fit results for the alpha particle data.

        Returns:
            object: An object that represents the fitting results of the alpha particle data.
        """
        return self.alpha_res

    def get_baseline_fit(self):
        """Retrieves the fit results for the baseline data.

        Returns:
            object: An object that represents the fitting results of the baseline data.
        """
        return self.baseline_fit["fit"]

    def get_spe(self) -> Tuple[float, float]:
        """Retrieves the slope value and its error from the spe fit results.

        Returns:
            Tuple[float, float]: A tuple containing the slope value and its error.
        """
        return (self.spe_res.params["slope"].value, self.spe_res.params["slope"].stderr)

    def get_CA(self) -> Tuple[float, float]:
        """Retrieves the Correlated Avalanche (CA) correction factor and its error.

        Returns:
            Tuple[float, float]: A tuple containing the CA factor and its error.
        """
        return (self.CA, self.CA_err)

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

    def get_alpha(self, sub_baseline: bool = True) -> Tuple[float, float]:
        """Retrieves the center value and its error from the alpha fit results. It subtracts the baseline mean if sub_baseline is set to True.

        Args:
            sub_baseline (bool, optional): If True, subtracts the baseline mean from the alpha center value. Defaults to True.

        Returns:
            Tuple[float, float]: A tuple containing the center value of alpha and its error.
        """
        alpha_value = self.alpha_res.params["center"].value
        alpha_error = self.alpha_res.params["center"].stderr
        if sub_baseline:
            baseline_value = self.baseline_mean
            baseline_error = self.baseline_err
            alpha_value -= baseline_value
            alpha_error = np.sqrt(
                alpha_error * alpha_error + baseline_error * baseline_error
            )
        return alpha_value, alpha_error

    def get_alpha_std(self) -> Tuple[float, float]:
        """Retrieves the standard deviation and its error from the alpha fit results.

        Returns:
            Tuple[float, float]: A tuple containing the standard deviation of alpha and its error.
        """
        alpha_value = self.alpha_res.params["sigma"].value
        alpha_error = self.alpha_res.params["sigma"].stderr

        return alpha_value, alpha_error

    def plot_baseline_histogram(
        self,
        with_fit: bool = True,
        log_scale: bool = False,
        color: str = "orange",
        savefig: bool = False,
        path: Optional[str] = None,
    ) -> None:
        """Plots a histogram of the baseline data.

        Args:
            with_fit (bool, optional): If True, overlays the fit of the data on the plot. Defaults to True.
            log_scale (bool, optional): If True, sets the y-axis to a logarithmic scale. Defaults to False.
            color (str, optional): The color of the histogram bars. Defaults to "orange".
            savefig (bool, optional): If True, saves the figure to the provided path. Defaults to False.
            path (str, optional): Path where the figure should be saved. Used only if savefig is set to True. Defaults to None.
        """
        fig = plt.figure()
        plt.hist(
            self.baseline_values,
            bins=self.info.baseline_numbins,
            label="Solicited Baseline Data",
            color="tab:" + color,
        )
        if with_fit:
            plot_fit(
                self.baseline_fit,
                self.baseline_values,
                binnum=self.info.baseline_numbins,
                plot_hists=False,
                label="Solicited Baseline Fit",
            )
        # plt.legend(loc = 'center left')
        plt.xlabel("Waveform Amplitude [V]")
        plt.ylabel("Counts")
        if log_scale:
            plt.yscale("log")
        textstr = f"Date: {self.info.date}\n"
        textstr += f"Condition: {self.info.condition}\n"
        textstr += f"Bias: {self.info.bias:0.4} [V]\n"
        textstr += f"RTD4: {self.info.temperature} [K]\n"
        textstr += f"--\n"
        textstr += f"""Baseline Mean: {self.baseline_fit['fit'].params['center'].value:0.4} +- {self.baseline_fit['fit'].params['center'].stderr:0.1} [V]\n"""
        textstr += f"""Baseline Sigma: {self.baseline_fit['fit'].params['sigma'].value:0.4} +- {self.baseline_fit['fit'].params['sigma'].stderr:0.1} [V]\n"""
        textstr += f"""Reduced $\chi^2$: {self.baseline_fit['fit'].redchi:0.4}"""

        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        props = dict(boxstyle="round", facecolor="tab:" + color, alpha=0.5)
        fig.text(0.15, 0.9, textstr, fontsize=8, verticalalignment="top", bbox=props)
        plt.tight_layout()

        if savefig:
            plt.savefig(path)
            plt.close(fig)

    # zoomed in version
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
    def plot_peak_histograms(
        self,
        with_fit: bool = True,
        log_scale: bool = True,
        peakcolor: str = "blue",
        savefig: bool = False,
        path: Optional[str] = None,
    ) -> None:
        """Plots a histogram of peak data with possible fitted models.

        Args:
            with_fit (bool, optional): If True, overlays the fitted model of the data on the plot. Defaults to True.
            log_scale (bool, optional): If True, sets the y-axis to a logarithmic scale. Defaults to True.
            peakcolor (str, optional): The color of the histogram bars. Defaults to "blue".
            savefig (bool, optional): If True, saves the figure to the provided path. Defaults to False.
            path (str, optional): Path where the figure should be saved. Used only if savefig is set to True. Defaults to None.
        """
        fig = plt.figure()
        bin_width = (max(self.peak_values) - min(self.peak_values)) / self.numbins
        # print(bin_width)

        total_num_bins = (max(self.peak_values) - min(self.peak_values)) / bin_width

        textstr = f"Date: {self.info.date}\n"
        textstr += f"Condition: {self.info.condition}\n"
        textstr += f"Bias: {self.info.bias} [V]\n"
        textstr += f"RTD4: {self.info.temperature} [K]\n"
        textstr += f"--\n"
        textstr += f"""Peak 1: {self.peak_fit.params['g1_center'].value:0.2} +- {self.peak_fit.params['g1_center'].stderr:0.2}\n"""
        textstr += f"""Peak 2: {self.peak_fit.params['g2_center'].value:0.2} +- {self.peak_fit.params['g2_center'].stderr:0.2}\n"""
        textstr += f"""Peak 3: {self.peak_fit.params['g3_center'].value:0.2} +- {self.peak_fit.params['g3_center'].stderr:0.2}\n"""
        textstr += f"""Peak 4: {self.peak_fit.params['g4_center'].value:0.2} +- {self.peak_fit.params['g4_center'].stderr:0.2}\n"""
        textstr += f"""Reduced $\chi^2$: {self.peak_fit.redchi:0.2}\n"""
        textstr += f"""SNR (quadrature): {self.resolution[0]:0.2}\n"""
        textstr += f"""SNR 1-2 (mode): {(self.peak_locs[1]-self.peak_locs[0])/self.baseline_mode:0.2}\n"""
        textstr += f"""SNR 2-3 (mode): {(self.peak_locs[2]-self.peak_locs[1])/self.baseline_mode:0.2}\n"""

        curr_hist = np.histogram(self.peak_values, bins=self.numbins)
        counts = curr_hist[0]
        bins = curr_hist[1]
        centers = (bins[1:] + bins[:-1]) / 2
        plt.plot(
            np.linspace(self.cutoff[0], self.cutoff[1], len(self.peak_fit.best_fit)),
            self.peak_fit.best_fit,
            "-",
            label="best fit",
            color="red",
        )  # plot the best fit model func

        x = np.linspace(self.cutoff[0], self.cutoff[1], 200)
        plt.plot(
            x,
            self.peak_fit.best_values["l_intercept"]
            + self.peak_fit.best_values["l_slope"] * x,
            "-",
            label="best fit - line",
            color="blue",
        )  # plot linear component of best fit model func

        props = dict(boxstyle="round", facecolor="tab:" + peakcolor, alpha=0.4)
        # plt.scatter(centers, counts, s = 7, color = 'black')
        plt.hist(self.peak_values, bins=int(total_num_bins), color="tab:" + peakcolor)

        fig.text(0.55, 0.925, textstr, fontsize=8, verticalalignment="top", bbox=props)
        plt.ylabel("Counts")
        plt.xlabel("Pulse Amplitude [V]")

        if log_scale:
            plt.ylim(1e-1)
            plt.yscale("log")
        plt.tight_layout()

        if savefig:
            plt.savefig(path)
            plt.close(fig)

    def plot_alpha_histogram(
        self, with_fit: bool = True, log_scale: bool = False, peakcolor: str = "purple"
    ) -> None:
        """Plots a histogram of alpha values with or without fitting.

        This method creates a histogram of alpha values (self.peak_values), calculated as the number of peaks (self.info.peaks_numbins) over the range of alpha fit (self.alpha_fit["high"] - self.alpha_fit["low"]).

        It supports options to show/hide fitted model on the plot (with_fit), use a logarithmic scale (log_scale), and change the color of the histogram bars (peakcolor).

        Additional information about the measurement, such as date, condition, bias, temperature, and peak statistics are displayed on the plot.

        Args:
            with_fit (bool, optional): If True, overlays the fitted model of the alpha values on the histogram. Defaults to True.
            log_scale (bool, optional): If True, sets the y-axis to a logarithmic scale. Defaults to False.
            peakcolor (str, optional): The color of the histogram bars. Defaults to "purple".
        """
        fig = plt.figure()
        fig.tight_layout()
        plt.rc("font", size=22)
        bin_density = self.info.peaks_numbins / (
            self.alpha_fit["high"] - self.alpha_fit["low"]
        )
        total_num_bins = bin_density * (
            np.amax(self.peak_values) - np.amin(self.peak_values)
        )
        plt.hist(self.peak_values, bins=int(total_num_bins), color="tab:" + peakcolor)
        if with_fit:
            plot_fit(
                self.alpha_fit,
                self.peak_values,
                binnum=self.info.peaks_numbins,
                plot_hists=False,
            )
        #        plt.legend(loc = 'center left')
        plt.xlabel("Waveform Amplitude [V]")
        plt.ylabel("Counts")
        plt.xlim(0.0)
        if log_scale:
            plt.yscale("log")
            plt.ylim(0.1)
        textstr = f"Date: {self.info.date}\n"
        textstr += f"Condition: {self.info.condition}\n"
        textstr += f"Bias: {self.info.bias:0.4} [V]\n"
        textstr += f"RTD4: {self.info.temperature} [K]\n"
        textstr += f"--\n"
        textstr += f"""Alpha Peak Mean: {self.alpha_fit['fit'].params['center'].value:0.4} +- {self.alpha_fit['fit'].params['center'].stderr:0.1} [V]\n"""
        textstr += f"""Alpha Peak Sigma: {self.alpha_fit['fit'].params['sigma'].value:0.4} +- {self.alpha_fit['fit'].params['sigma'].stderr:0.1} [V]\n"""
        textstr += f"""Reduced $\chi^2$: {self.alpha_res.redchi:0.4}"""

        props = dict(boxstyle="round", facecolor="tab:" + peakcolor, alpha=0.4)
        fig.text(
            0.175, 0.925, textstr, fontsize=20, verticalalignment="top", bbox=props
        )
        plt.show()

    def plot_both_histograms(
        self,
        log_scale: bool = True,
        density: bool = True,
        alphas: bool = False,
        baselinecolor: str = "orange",
        peakcolor: str = "blue",
        savefig: bool = False,
        path: Optional[str] = None,
    ):
        """Plots histograms for both baseline and peak values.

        This method creates histograms for baseline and peak values with control over the
        logarithmic scale, normalization (density), inclusion of alpha peaks, and color schemes
        for baseline and peak histograms.

        It can also save the plot to a specified file.

        Args:
            log_scale (bool, optional): If True, sets the y-axis to a logarithmic scale. Defaults to True.
            density (bool, optional): If True, normalizes the histogram to form a probability density. Defaults to True.
            alphas (bool, optional): If True, includes alpha peaks in the histogram. Defaults to False.
            baselinecolor (str, optional): The color of the baseline histogram bars. Defaults to "orange".
            peakcolor (str, optional): The color of the peak histogram bars. Defaults to "blue".
            savefig (bool, optional): If True, saves the plot to the file specified in 'path'. Defaults to False.
            path (str, optional): The file path to save the plot. Used only if 'savefig' is True. Defaults to None.
        """
        if self.no_solicit:
            print("NO PRE BREAKDOWN DATA TO PLOT")
        fig = plt.figure()
        plt.hist(
            self.baseline_values,
            bins=self.info.baseline_numbins,
            label="Solicited Baseline Data",
            density=density,
            color="tab:" + baselinecolor,
        )
        if alphas:
            bin_density = self.info.peaks_numbins / (
                self.alpha_fit["high"] - self.alpha_fit["low"]
            )
        else:
            bin_density = self.info.peaks_numbins / (4.0 * self.baseline_std)
        # total_num_bins = bin_density * (np.amax(self.peak_values) - np.amin(self.peak_values))
        total_num_bins = self.info.peaks_numbins
        plt.hist(
            self.peak_values,
            bins=int(total_num_bins),
            density=density,
            label="Self-Triggered Pulse Height Data",
            color="tab:" + peakcolor,
        )
        if log_scale:
            plt.ylim(1e-1)
            plt.yscale("log")
        plt.ylabel("Frequency" if density else "Counts")
        plt.xlabel("Amplitude [V]")

        plt.legend()
        textstr = f"Date: {self.info.date}\n"
        textstr += f"Condition: {self.info.condition}\n"
        textstr += f"Bias: {self.info.bias:0.4} [V]\n"
        textstr += f"RTD4: {self.info.temperature} [K]"
        props = dict(boxstyle="round", facecolor="tab:" + peakcolor, alpha=0.4)
        fig.text(0.75, 0.75, textstr, fontsize=8, verticalalignment="top", bbox=props)
        plt.tight_layout()

        if savefig:
            plt.savefig(path)
            plt.close(fig)

    # currently broken:
    def plot_baseline_waveform_hist(self, num: int = -1, color: str = "orange"):
        """Plots a 2D histogram of baseline waveform data over time.

        The method generates a 2D histogram of baseline waveform data over time.
        The number of waveforms and their color in the plot can be controlled.

        Args:
            num (int, optional): Number of waveforms to include in the superposition. Defaults to -1, indicating all.
            color (str, optional): Color of the histogram. Defaults to "orange".
        """
        fig = plt.figure()
        waveform_data, waveform_times, num_w = get_baseline(
            self.info.solicit_path, self.info.peak_search_params
        )
        plt.hist2d(waveform_times, waveform_data, bins=100, norm=mpl.colors.LogNorm())
        plt.xlabel(r"Time [$\mu$s]")
        plt.ylabel("Waveform Amplitude [V]")
        textstr = f"Date: {self.info.date}\n"
        textstr += f"Condition: {self.info.condition}\n"
        textstr += f"Bias: {self.info.bias:0.4} [V]\n"
        textstr += f"RTD4: {self.info.temperature} [K]\n"
        textstr += f"Superposition of {num_w} waveforms"
        props = dict(boxstyle="round", facecolor="tab:" + color, alpha=0.5)
        fig.text(0.6, 0.3, textstr, fontsize=8, verticalalignment="top", bbox=props)
        plt.tight_layout()

    def plot_peak_waveform_hist(self, num: int = -1, color: str = "blue"):
        """Plots a 2D histogram of peak waveform data over time.

        The method generates a 2D histogram of peak waveform data over time.
        The number of waveforms and their color in the plot can be controlled.

        Args:
            num (int, optional): Number of waveforms to include in the superposition. Defaults to -1, indicating all.
            color (str, optional): Color of the histogram. Defaults to "blue".
        """
        fig = plt.figure()
        waveform_data, waveform_times, num_w = get_peak_waveforms(
            self.info.selftrig_path, num
        )
        plt.hist2d(waveform_times, waveform_data, bins=1000, norm=mpl.colors.LogNorm())
        plt.xlabel(r"Time [$\mu$s]")
        plt.ylabel("Waveform Amplitude [V]")
        textstr = f"Date: {self.info.date}\n"
        textstr += f"Condition: {self.info.condition}\n"
        textstr += f"Bias: {self.info.bias:0.4} [V]\n"
        textstr += f"RTD4: {self.info.temperature} [K]\n"
        textstr += f"Superposition of {num_w} waveforms"
        props = dict(boxstyle="round", facecolor="tab:" + color, alpha=0.4)
        low, high = plt.ylim()
        plt.ylim(low, 4.5)
        fig.text(0.6, 0.9, textstr, fontsize=8, verticalalignment="top", bbox=props)
        plt.tight_layout()
