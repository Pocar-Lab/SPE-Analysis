# -*- coding: utf-8 -*-
"""
Created on Jul 25 2025

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""

import h5py
import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from scipy import signal
import pprint
from typing import Any, Optional

def get_data(h5path: str, groupname: str) -> np.ndarray:
    """Extrats data from an h5 file provided by h5 path

    Args:
        h5path (str): Path to h5 file.
        groupname (str): Group name of desired group in h5 file

    Returns:
        np.ndarray : waveform data
    """
    with h5py.File(h5path, "r") as hdf:
        # TODO: check data format can be cast to ndarray
        data: np.ndarray = hdf["RunData"][groupname][:]
    return data

def get_run_meta(h5path: str) -> dict:
    """Get metadata for specific run

    Args:
        h5path (str): path to h5 file

    Returns:
        dict : hdf5 run metadata
    """
    with h5py.File(h5path, "r") as hdf:
        run_meta = dict(hdf["RunData"].attrs)
    return run_meta

def get_grp_meta(h5path: str, groupname: str) -> dict:
    """Get metadata of specified group

    Args:
        h5path (str): Path to h5 file
        groupname (str): Name of group to accquire metadata from

    Returns:
        dict: hdf5 group metadata
    """
    with h5py.File(h5path, "r") as hdf:
        meta_group = dict(hdf["RunData"][groupname].attrs)
    return meta_group

def get_grp_names(h5path: str) -> list:
    """Get names of groups in specified h5 file

    Args:
        h5path (str): Path to h5 file

    Returns:
        list : acquisition names
    """
    with h5py.File(h5path, "r") as hdf:
        group_names = list(hdf["RunData"].keys())
    return group_names

def get_mode(hist_data: list | np.array) -> tuple[float, float]:
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


def fit_baseline_gauss(
    values: list[float], binnum: int = 50
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
    f_range["center"] = np.mean(values)
    std_guess = np.std(values)
    f_range["low"] = f_range["center"] - 2.0 * std_guess
    f_range["high"] = f_range["center"] + 2.0 * std_guess
    bin_density = float(binnum) / (np.amax(values) - np.amin(values))
    new_binnum = int(bin_density * (f_range["high"] - f_range["low"]))
    values = np.array(values)
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


class ProcessWaveforms:
    # TODO: return single peaks object with fields for all, led, or dark peaks
    def __init__(
        self,
        f: list,
        acquisition: Optional[str] = None,
        is_pre_bd: bool = False,
        do_filter: bool = False,
        plot_waveforms: bool = False,
        upper_limit: float = 4.4,
        baseline_correct: bool = False,
        poly_correct: bool = False,
        prominence: float = 0.005,
        fourier: bool = False,
        condition: str = "unspecified medium (GN/LXe/Vacuum)",
        num_waveforms: float = 0
    ):
        """
        Process waveform data from hdf5 file to find peaks.

        Apply filtering and baseline correciton to waveforms. Can process alpha pulse waveforms,
        SPE waveforms, or baseline data. Records alpha pulse heights, SPE pulse heights, or
        aggregate all ampltiudes respectively. Optionally plots waveforms to inspect data for
        tuning peak finding.

        Args:
            f list: list of h5 file names
            acquisition (str, optional): specified file name. Defaults to 'placeholder'.
            is_pre_bd (bool, optional): specified whether data is solicited, AKA 'empty' baseline data. Defaults to False.
            do_filter (bool, optional): activates butterworth lowpass filter if True. Defaults to False.
            plot_waveforms (bool, optional): plots waveforms if True. Defaults to False.
            upper_limit (float, optional): amplitude threshold for discarding waveforms. Set to -1 for automatic setting. Defaults to 4.4.
            baseline_correct (bool, optional): baseline corrects waveforms if True. Defaults to False.
            prominence (float, optional): parameter used for peak finding algo. Defaults to 0.005.
            fourier (bool, optional): if True performs fourier frequency subtraction. Defaults to False.
            num_waveforms: (float, optional): number of oscilloscope traces to read in before stopping; default 0 reads everything
        Raises:
            TypeError: _description_
        """
        self.do_filter = do_filter
        self.plot_waveforms = plot_waveforms
        self.hd5_files = f
        self.upper_limit = upper_limit
        self.baseline_correct = baseline_correct
        self.poly_correct = poly_correct
        # holds all acquisition data index by first file name then by acquisition name
        self.acquisitions_data = {}
        # holds all acquisition names indexed by file name indexed by file name then by acquisition name
        self.acquisition_names = {}
        # holds acquisition time axis for all acquisitions in a single acquisition data array
        self.acquisitions_time = {}
        # holds all acquisition meta data, indexed first by file name then by acquisition name
        self.acquisition_meta_data = {}
        self.all_peak_data = []
        self.acquisition = acquisition
        self.fourier = fourier
        self.prominence = prominence
        self.baseline_levels = []  # list of mode of waveforms
        self.num_waveforms = num_waveforms
        self.condition = condition
        self.is_pre_bd = is_pre_bd

        # if self.led: # initialize led on/off lists
        self.all_dark_peak_data = []
        self.all_led_peak_data = []

        for curr_file in self.hd5_files:
            self.acquisition_names[curr_file] = get_grp_names(curr_file)
            # for curr_acquisition_name in self.acquisition_names[curr_file]:

        # holds all run data indexed by file name
        self.run_meta_data = {}
        for curr_file in self.hd5_files:
            self.acquisition_names[curr_file] = get_grp_names(curr_file)
            self.acquisitions_time[curr_file] = {}
            self.acquisitions_data[curr_file] = {}
            self.acquisition_meta_data[curr_file] = {}

            self.run_meta_data[curr_file] = get_run_meta(curr_file)
            print(f"Run Meta: {self.run_meta_data[curr_file]}")

            # TODO: simplify reducency, if single acquisition given put it in a list
            if acquisition:
                curr_data = get_data(curr_file, acquisition)
                self.acquisitions_data[curr_file][acquisition] = curr_data[:, 1:]
                self.acquisitions_time[curr_file][acquisition] = curr_data[:, 0]
                self.acquisition_meta_data[curr_file][acquisition] = get_grp_meta(
                    curr_file, acquisition
                )
                # print(f"Group Meta: {self.acquisition_meta_data[curr_file][acquisition]}")
                pprint.pprint(self.acquisition_meta_data[curr_file][acquisition])
                self.bias = self.acquisition_meta_data[curr_file][acquisition][
                    "Bias(V)"
                ]
                self.date = self.acquisition_meta_data[curr_file][acquisition][
                    "AcquisitionStart"
                ]
                self.trig = self.acquisition_meta_data[curr_file][acquisition][
                    "LowerLevel"
                ]
                self.yrange = self.acquisition_meta_data[curr_file][acquisition][
                    "Range"
                ]
                self.offset = self.acquisition_meta_data[curr_file][acquisition][
                    "Offset"
                ]

            else:
                for curr_acquisition_name in self.acquisition_names[curr_file]:
                    curr_data = get_data(curr_file, curr_acquisition_name)
                    self.acquisitions_data[curr_file][
                        curr_acquisition_name
                    ] = curr_data[:, 1:]
                    self.acquisitions_time[curr_file][
                        curr_acquisition_name
                    ] = curr_data[:, 0]
                    self.acquisition_meta_data[curr_file][
                        curr_acquisition_name
                    ] = get_grp_meta(curr_file, curr_acquisition_name)
                    # pprint.pprint(self.acquisition_meta_data[curr_file][curr_acquisition_name])
                    self.bias = self.acquisition_meta_data[curr_file][
                        curr_acquisition_name
                    ]["Bias(V)"]
                    self.date = self.acquisition_meta_data[curr_file][
                        curr_acquisition_name
                    ]["AcquisitionStart"]
                    self.trig = self.acquisition_meta_data[curr_file][
                        curr_acquisition_name
                    ]["LowerLevel"]
                    self.yrange = self.acquisition_meta_data[curr_file][
                        curr_acquisition_name
                    ]["Range"]
                    self.offset = self.acquisition_meta_data[curr_file][
                        curr_acquisition_name
                    ]["Offset"]

        if self.upper_limit == -1:
            self.upper_limit = self.yrange - self.offset - 0.001

        # Searches for peaks using provided parameters
        self.peak_search_params = {
            "height": 0.0,  # SPE
            "threshold": None,  # SPE
            "distance": None,  # SPE
            "prominence": prominence, #specified by user
            "width": None,  # SPE
            "wlen": 100,  # SPE
            "rel_height": None,  # SPE
            "plateau_size": None,  # SPE
        }
        self.get_peak_data()

        # TODO: call in analysis script or measurement info?
        if is_pre_bd:
            self.get_baseline()

    def get_peak_data(self):
        """ collects peak data and puts in dict """
        self.peak_data = {}
        for curr_file in self.hd5_files:
            self.peak_data[curr_file] = {}
            for curr_acquisition_name in self.acquisition_names[curr_file]:
                if self.acquisition:
                    curr_acquisition_name = self.acquisition
                curr_peaks, curr_dark_peaks, curr_led_peaks = self.get_peaks(curr_file, curr_acquisition_name)
                self.peak_data[curr_file][curr_acquisition_name] = curr_peaks
                self.all_peak_data += curr_peaks
                self.all_dark_peak_data += curr_dark_peaks
                self.all_led_peak_data += curr_led_peaks
                if self.plot_waveforms or self.acquisition:
                    break

    def get_peaks(self, filename: str, acquisition_name: str) -> list[float]:
        """Uses scipy.signal.find_peaks to find the peaks of the data.
        Args:
            filename (str): Name of file to analyze
            acquisition_name (str): Name of particular acquisition
        Returns:
            list: List of peak heights.
        """
        all_peaks = []
        dark_peaks = []
        led_peaks = []
        curr_data = self.acquisitions_data[filename][acquisition_name]
        time = self.acquisitions_time[filename][acquisition_name]
        window_length = time[-1] - time[0]
        num_points = float(len(time))
        fs = num_points / window_length

        print(f'filename: {filename}, acquisition name: {acquisition_name}')
        if self.poly_correct:
            print('using polynomial baseline correction...')
        if self.num_waveforms ==0:
            print('reading all waveforms')
            num_wavefroms = np.shape(curr_data)[1]
        else:
            num_wavefroms = self.num_waveforms
            print(f'reading {num_wavefroms} waveforms')

        for idx in range(num_wavefroms):
            # TODO better way to do this
            # idx = idx + 8000 # uncomment if plotting waveforms and want to see waveforms at different indices
            if idx % 1000 == 0:
                print(f'{idx} read so far')

            amp = self.process_amp(curr_data[:, idx], fs)

            # If processing prebreakdown baseline, add all amplitudes and don't find peaks
            if self.is_pre_bd:
                all_peaks += list(amp) # amp[100:]
                continue

            peaks, props = signal.find_peaks(amp, **self.peak_search_params)

            # TODO split into different method
            if self.plot_waveforms:
                plt.title(acquisition_name)
                plt.tight_layout()
                if len(peaks) > 0: # only plot peaks
                    plt.plot(time,amp)
                    for peak in peaks:
                        plt.plot(time[peaks], amp[peaks], '.')

            led_time_thresh = (time[-1] + time[1]) / 2.0
            for peak in peaks:
                all_peaks.append(amp[peak])
                curr_time = time[peak]
                if curr_time < led_time_thresh:
                    dark_peaks.append(amp[peak])
                else:
                    led_peaks.append(amp[peak])

        print(f'all  peaks: {np.mean(all_peaks):.3} ± {np.std(all_peaks,ddof=1)/np.sqrt(len(all_peaks)):.3}')
        print(f'dark peaks: {np.mean(dark_peaks):.3} ± {np.std(dark_peaks,ddof=1)/np.sqrt(len(all_peaks)):.3}')
        print(f'LED  peaks: {np.mean(led_peaks):.3} ± {np.std(led_peaks,ddof=1)/np.sqrt(len(led_peaks)):.3}')
        return all_peaks, dark_peaks, led_peaks

    def process_amp(self, amp, fs):
        """Process amplitudes before peaks are found.

        Applies upper limit cut, baseline correction, and filtering"""

        # Skip waveforms which go out of the oscilloscope range, such as an alpha pulse
        if np.amax(amp) > self.upper_limit:
            return []

        if self.baseline_correct:
            if self.poly_correct:
                amp -= peakutils.baseline(amp, deg=2)

            use_bins = np.linspace(-self.upper_limit, self.upper_limit, 1000)
            curr_hist = np.histogram(amp, bins=use_bins)
            baseline_level, _ = get_mode(curr_hist)
            self.baseline_levels.append(baseline_level)
            self.baseline_mode = baseline_level
            # print('baseline:', baseline_level)
            amp -= baseline_level

        # TODO why check shape? not done for solicited
        if self.do_filter and np.shape(amp) != (0,):
            sos = signal.butter(3, 4E5, btype='lowpass', fs=fs, output='sos') # SPE dark/10g
            filtered = signal.sosfilt(sos, amp)
            amp = filtered

        return amp

    def get_baseline(self):
        baseline_fit = fit_baseline_gauss(self.all_peak_data)
        self.baseline_mean = baseline_fit["fit"].values["center"]
        self.baseline_err  = baseline_fit["fit"].params["center"].stderr
        self.baseline_std  = baseline_fit["fit"].values["sigma"]
