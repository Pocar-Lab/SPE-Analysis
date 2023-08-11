from typing import List, Optional, Tuple
import numpy as np
from MeasurementInfo import MeasurementInfo
from ProcessWaveforms import *
from RunInfo import RunInfo, np
from RunInfoBaseline import RunInfoBaseline
from numpy import ndarray


class WaveformProcessorSPE(WaveformProcessor):
    def __init__(
        self,
        info: MeasurementInfo,
        centers: List[float] | ndarray,
        run_info_self: RunInfo | None = None,
        run_info_solicit: RunInfoBaseline | None = None,
        baseline_correct: bool = False,
        cutoff: Tuple[float, float] = ...,
        numpeaks: int = 4,
        no_solicit: bool = False,
        offset_num: int = 0,
    ):
        super().__init__(
            info,
            run_info_self,
            run_info_solicit,
            baseline_correct,
            cutoff,
            offset_num,
        )

        self.centers = centers
        self.numpeaks = numpeaks
        self.no_solicit = no_solicit

        # TODO fix red lines
        if no_solicit:
            self.baseline_mode = run_info_self.baseline_mode_mean
            # self.baseline_mode = 1 #PLACEHOLDER
            self.baseline_rms = run_info_self.baseline_mode_rms
            self.baseline_std = 0.25 * run_info_self.baseline_mode_std
            self.baseline_err = run_info_self.baseline_mode_err
            self.baseline_rms = run_info_self.baseline_mode_rms
            # self.baseline_std = 1
            # self.baseline_err = 1
        else:
            self.run_info_solicit = run_info_solicit
            self.baseline_mode = run_info_solicit.baseline_mode
    def process(self, overwrite=False):
        """Processes the waveform data, extracting various statistical information from it.

        Args:
            overwrite (bool, optional): If True, any previous processing results are overwritten. Defaults to False.
            do_spe (bool, optional): If True, Single Photoelectron (SPE) data is processed, including fitting multiple peaks and calculating signal-to-noise ratio (SNR). Defaults to True.
            do_alpha (bool, optional): If True, alpha particle data is processed. Defaults to False.
        """
        self.process_h5()


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

        self.peak_fit = fit_peaks_multigauss(
                values = self.peak_values,
                baseline_width = 2.0 * self.baseline_std,
                centers = self.centers,
                numpeaks = self.numpeaks, 
                cutoff = self.cutoff
                )

        self.peak_locs = [self.peak_fit.params[f'g{i}_center'].value for i in range(1, numpeaks + 1)]
        self.peak_sigmas = [self.peak_fit.params[f'g{i}_sigma'].value for i in range(1, numpeaks + 1)]
        self.peak_stds = [self.peak_fit.params[f'g{i}_center'].stderr for i in range(1, numpeaks + 1)]
        
        

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
    def plot(
        self,
        with_baseline: bool = True,
        baselinecolor: str = "orange",
        peakcolor: str = "blue",
        savefig: bool = False,
        path: Optional[str] = None,
    ) -> None:
        """Plots average pulse amplitudes as a function of # of Photoelectrons (PE).

        Args:
            with_baseline (bool, optional): If True, plots the baseline data. Defaults to True.
            baselinecolor (str, optional): Color used for the baseline data. Defaults to "orange".
            peakcolor (str, optional): Color used for the SPE peak data. Defaults to "blue".
            savefig (bool, optional): If True, saves the figure to the provided path. Defaults to False.
            path (str, optional): Path where the figure should be saved. Used only if savefig is set to True. Defaults to None.
        """       
        fig = plt.figure()
        fig.tight_layout()
        plt.rc("font", size=22)
        plt.errorbar(
            self.spe_num,
            self.peak_locs[: self.numpeaks],
            yerr=self.peak_stds[: self.numpeaks],
            fmt=".",
            label="Self-Triggered Peaks",
            color="tab:" + peakcolor,
            markersize=10,
        )
        if with_baseline:
            if self.no_solicit == False:
                plt.errorbar(
                    0,
                    self.baseline_mean,
                    yerr=self.baseline_err,
                    fmt=".",
                    label="Solicited Baseline Peak",
                    color="tab:" + baselinecolor,
                    markersize=10,
                )
            # else:
            # plt.errorbar(0, self.baseline_mode, yerr = self.baseline_err, fmt='.', label = 'Solicited Baseline Peak', color = 'tab:' + baselinecolor, markersize = 10)

        b = self.spe_res.params["intercept"].value
        m = self.spe_res.params["slope"].value
        x_values = np.linspace(0, len(self.spe_num) + 1, 20)
        y_values = m * x_values + b
        plt.plot(
            x_values,
            y_values,
            "--",
            color="tab:" + peakcolor,
            label="Self-Triggered Fit",
        )
        # dely = self.spe_res.eval_uncertainty(x=x_values, sigma=1)
        # plt.fill_between(x_values, y_values+dely, y_values-dely)
        # plt.plot(self.spe_num, self.spe_res.best_fit, 'r', label='Self-Triggered Fit')

        plt.xlabel("Photoelectron Peak Number")
        plt.ylabel("Peak Location [V]")

        plt.legend()

        textstr = f"Date: {self.info.date}\n"
        textstr += f"Condition: {self.info.condition}\n"
        textstr += f"Bias: {self.info.bias:0.4} [V]\n"
        textstr += f"RTD4: {self.info.temperature} [K]\n"
        textstr += f"--\n"
        textstr += f"""Slope: {self.spe_res.params['slope'].value:0.4} +- {self.spe_res.params['slope'].stderr:0.2} [V/p.e.]\n"""
        textstr += f"""Intercept: {self.spe_res.params['intercept'].value:0.4} +- {self.spe_res.params['intercept'].stderr:0.2} [V]\n"""
        textstr += rf"""Reduced $\chi^2$: {self.spe_res.redchi:0.4}"""
        textstr += f"""\n"""
        textstr += f"--\n"
        if not self.no_solicit:
            textstr += (
                f"Baseline: {self.baseline_mean:0.4} +- {self.baseline_err:0.2} [V]"
            )

        props = dict(boxstyle="round", facecolor="tab:" + peakcolor, alpha=0.4)
        fig.text(0.6, 0.4, textstr, fontsize=20, verticalalignment="top", bbox=props)

        if savefig:
            plt.savefig(path)
            plt.close(fig)
