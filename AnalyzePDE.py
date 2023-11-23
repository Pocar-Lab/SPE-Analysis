# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:55:20 2022

@author: lab-341
"""

import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt

# import GainCalibration_2022 as GainCalibration
import pandas as pd
from ProcessWaveforms_MultiGaussian import WaveformProcessor


# def CA_func(x, A, B, C):
#     return (A * np.exp(B*(x)) + 1.0) / (1.0 + A) - 1.0 + C
def CA_func(x, A, B):
    return (A * np.exp(B*(x)) + 1.0) / (1.0 + A) - 1.0


class SPE_data:
    def __init__(
        self,
        campaign: list[WaveformProcessor],
        invC: float,
        invC_err: float,
        filtered: bool,
    ) -> None:
        """
        Initialize the SPE_data class. This class is used for collecting all the WaveformProcessor results for
        an entire campaign into one object. Can plot absolute gain and Correlated Avalance (CA) as a
        function of bias voltage.

        Parameters:
        campaign (List[WaveformProcessor]): The campaign data.
        invC (float): The inverse of the capacitance.
        invC_err (float): The error in the inverse of the capacitance.
        filtered (bool): A flag indicating whether the data is filtered.

        Returns:
        None
        """
        self.campaign = campaign
        self.invC = invC
        self.invC_err = invC_err
        self.filtered = filtered
        self.analyze_spe()

    def analyze_spe(self) -> None:
        """
        Analyze the single photoelectron (SPE) data.

        This method unpacks the SPE amplitudes, bias voltages, and CA calculation from the
        WaveformProcessor objects. It then performs a linear fit on the SPE values as a
        function of bias voltage, and determines the breakdown voltage. It also calculates
        and populates the absolute gain and overvoltage attributes.

        Returns:
        None
        """
        self.bias_vals = []
        self.bias_err = []
        self.spe_vals = []
        self.spe_err = []
        self.absolute_spe_vals = []
        self.absolute_spe_err = []
        self.CA_vals = []
        self.CA_err = []
        self.CA_rms_vals = []
        self.CA_rms_err = []
        for wp in self.campaign:
            self.bias_vals.append(wp.info.bias)
            self.bias_err.append(0.0025 * wp.info.bias + 0.015)  # error from keysight
            curr_spe = wp.get_spe()
            self.spe_vals.append(curr_spe[0])
            self.spe_err.append(curr_spe[1])

        self.bias_vals = np.array(self.bias_vals)
        self.bias_err = np.array(self.bias_err)
        self.spe_vals = np.array(self.spe_vals)
        self.spe_err = np.array(self.spe_err)
        self.absolute_spe_vals = self.spe_vals / (self.invC * 1.60217662e-7)
        self.absolute_spe_err = self.absolute_spe_vals * np.sqrt(
            (self.spe_err * self.spe_err) / (self.spe_vals * self.spe_vals)
            + (self.invC_err * self.invC_err) / (self.invC * self.invC)
        )
        spe_wgts = [1.0 / curr_std for curr_std in self.spe_err]
        absolute_spe_wgts = [1.0 / curr_std for curr_std in self.absolute_spe_err]

        model = lm.models.LinearModel()
        params = model.make_params()
        self.spe_res = model.fit(
            self.spe_vals, params=params, x=self.bias_vals, weights=spe_wgts
        )
        self.absolute_spe_res = model.fit(
            self.absolute_spe_vals,
            params=params,
            x=self.bias_vals,
            weights=absolute_spe_wgts,
        )  # linear fit
        b_spe = self.spe_res.params["intercept"].value
        m_spe = self.spe_res.params["slope"].value
        self.v_bd = -b_spe / m_spe

        vec_spe = np.array([b_spe / (m_spe * m_spe), -1.0 / m_spe])
        # print('check ' + str(self.bias_vals))
        self.v_bd_err = np.sqrt(
            np.matmul(
                np.reshape(vec_spe, (1, 2)),
                np.matmul(self.spe_res.covar, np.reshape(vec_spe, (2, 1))),
            )[0, 0]
        )  # breakdown error calculation using covariance matrix

        self.ov = []
        self.ov_err = []

        for b, db in zip(self.bias_vals, self.bias_err):
            curr_ov = b - self.v_bd
            curr_ov_err = np.sqrt(db * db)
            self.ov.append(curr_ov)
            self.ov_err.append(curr_ov_err)

        # self.bias_vals.append(self.v_bd)
        # self.bias_err.append(self.v_bd_err)
        
        # self.ov.append(0.0)
        # self.ov_err.append(self.v_bd_err)
        
        self.ov = np.array(self.ov)
        self.ov_err = np.array(self.ov_err)
        
        spe = self.spe_res.eval(params=self.spe_res.params, x=self.bias_vals)
        spe_err = self.spe_res.eval_uncertainty(params=self.spe_res.params, x=self.bias_vals, sigma=1)
        for wp, curr_bias, curr_spe, curr_spe_err in zip(self.campaign, self.bias_vals, spe, spe_err):
            curr_CA_val, curr_CA_err = wp.get_CA_spe(curr_spe, curr_spe_err)
            curr_CA_rms_val, curr_CA_rms_err = wp.get_CA_rms(curr_spe, curr_spe_err)
            self.CA_vals.append(curr_CA_val)
            self.CA_err.append(curr_CA_err)
            self.CA_rms_vals.append(curr_CA_rms_val)
            self.CA_rms_err.append(curr_CA_rms_err)

#include interpolated v_bd value in CA model fit
        # bias_inclusive_bd = np.append(self.bias_vals,self.v_bd)
        # bias_inclusive_bd_err = np.append(self.bias_err,self.v_bd_err)
        
        # self.ov = np.append(self.ov, 0.0)
        # self.bias_err = np.append(self.ov_err,self.v_bd_err)
        # self.CA_vals.append(0.0) #no CA activity at v_bd by definition
        # self.CA_err.append(self.campaign[0].get_baseline_fit().params["sigma"].value)
        
        # self.CA_vals = np.array(self.CA_vals)
        # self.CA_err = np.array(self.CA_err)

        # self.CA_rms_vals = np.array(self.CA_rms_vals)
        # self.CA_rms_err = np.array(self.CA_rms_err)

        # i am stinky

        CA_model = lm.Model(CA_func)
        # CA_params = CA_model.make_params(A=1, B= 1, C = 0)
        CA_params = CA_model.make_params(A=1, B=1)
        # CA_params['A'].min = 0.1
        # CA_params['C'].min = -5
        # CA_params['C'].max = 5
        CA_wgts = [1.0 / curr_std for curr_std in self.CA_err]
        self.CA_res = CA_model.fit(
            self.CA_vals, params=CA_params, x=self.ov, weights=CA_wgts
        )

    def get_CA_ov(self, input_ov_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the CA values and their uncertainties at the given overvoltage values.

        Parameters:
        input_ov_vals (np.ndarray): The input overvoltage values.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the evaluated CA values and their uncertainties.
        """
        out_vals = self.CA_res.eval(params=self.CA_res.params, x=input_ov_vals)
        out_err = self.CA_res.eval_uncertainty(x=input_ov_vals, sigma=1)
        return out_vals, out_err

    def get_spe_ov(self, input_ov_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the SPE values and their uncertainties at the given overvoltage values.

        This method first calculates the bias values by adding the breakdown voltage to the input
        overvoltage values. It then evaluates the SPE values and their uncertainties at these bias values.

        Parameters:
        input_ov_vals (np.ndarray): The input overvoltage values.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the evaluated SPE values and their uncertainties.
        """
        input_bias_vals = input_ov_vals + self.v_bd
        out_vals = self.spe_res.eval(params=self.spe_res.params, x=input_bias_vals)
        out_err = self.spe_res.eval_uncertainty(x=input_bias_vals, sigma=1)
        return out_vals, out_err

    def plot_spe(
        self,
        in_ov: bool = False,
        absolute: bool = False,
        color: str = "blue",
        out_file: str = None,
    ) -> None:
        """
        Plot the single photoelectron (SPE) data.

        This method creates a plot of the SPE data. It can plot either the absolute gain or the SPE amplitude,
        and it can plot the data as a function of either the bias voltage or the overvoltage. The plot includes
        the best fit line and its uncertainty, the data points with their errors, and a text box with additional
        information about the data and the fit. The plot can be saved to a file.

        Parameters:
        in_ov (bool, optional): If True, plot the data as a function of the overvoltage. If False, plot the data as a function of the bias voltage. Default is False.
        absolute (bool, optional): If True, plot the absolute gain. If False, plot the SPE amplitude. Default is False.
        color (str, optional): The color to use for the text box. Default is 'blue'.
        out_file (str, optional): The name of the file to save the plot to. If None, the plot is not saved to a file. Default is None.

        Returns:
        None
        """

        color = "tab:" + color
        fig = plt.figure()

        start_bias = self.v_bd
        end_bias = np.amax(self.bias_vals) + 1.0
        fit_bias = np.linspace(start_bias, end_bias, 20)

        if absolute:
            fit_y = self.absolute_spe_res.eval(
                params=self.absolute_spe_res.params, x=fit_bias
            )
            fit_y_err = self.absolute_spe_res.eval_uncertainty(
                x=fit_bias, params=self.absolute_spe_res.params, sigma=1
            )
            fit_label = "Absolute Gain Best Fit"
            data_label = "Absolute Gain values"
            y_label = "Absolute Gain"
            data_y = self.absolute_spe_vals
            data_y_err = self.absolute_spe_err
            chi_sqr = self.absolute_spe_res.redchi
            slope_text = rf"""Slope: {self.absolute_spe_res.params['slope'].value:0.4} $\pm$ {self.absolute_spe_res.params['slope'].stderr:0.2} [1/V]"""
            intercept_text = rf"""Intercept: {self.absolute_spe_res.params['intercept'].value:0.4} $\pm$ {self.absolute_spe_res.params['intercept'].stderr:0.2} [V]"""
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        else:
            fit_y = self.spe_res.eval(params=self.spe_res.params, x=fit_bias)
            fit_y_err = self.spe_res.eval_uncertainty(
                x=fit_bias, params=self.spe_res.params, sigma=1
            )
            fit_label = "SPE Amplitude Best Fit"
            data_label = "SPE Amplitude values"
            y_label = "SPE Amplitude [V]"
            data_y = self.spe_vals
            data_y_err = self.spe_err
            chi_sqr = self.spe_res.redchi
            slope_text = rf"""Slope: {self.spe_res.params['slope'].value:0.4} $\pm$ {self.spe_res.params['slope'].stderr:0.2} [V/V]"""
            intercept_text = rf"""Intercept: {self.spe_res.params['intercept'].value:0.4} $\pm$ {self.spe_res.params['intercept'].stderr:0.2} [V]"""

        parameter_text = slope_text
        if in_ov:
            fit_x = np.linspace(start_bias - self.v_bd, end_bias - self.v_bd, 20)
            data_x = self.ov
            data_x_err = self.ov_err
            x_label = "Overvoltage [V]"
        else:
            fit_x = fit_bias
            data_x = self.bias_vals
            data_x_err = self.bias_err
            x_label = "Bias Voltage [V]"
            parameter_text += f"""\n"""
            parameter_text += intercept_text

        parameter_text += f"""\n"""
        parameter_text += rf"""Reduced $\chi^2$: {chi_sqr:0.4}"""
        parameter_text += f"""\n"""
        plt.fill_between(
            fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color="red", alpha=0.5
        )
        plt.plot(fit_x, fit_y, color="red", label=fit_label)
        plt.errorbar(
            data_x,
            data_y,
            xerr=data_x_err,
            yerr=data_y_err,
            markersize=6,
            fmt=".",
            label=data_label,
        )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

        textstr = f"Date: {self.campaign[0].info.date}\n"
        textstr += f"Condition: {self.campaign[0].info.condition}\n"
        textstr += f"RTD4: {self.campaign[0].info.temperature} [K]\n"
        if self.filtered:
            textstr += f"Filtering: Lowpass, 400kHz\n"
        else:
            textstr += f"Filtering: None\n"
        textstr += f"--\n"
        textstr += parameter_text
        textstr += f"--\n"
        textstr += rf"Breakdown Voltage: {self.v_bd:0.4} $\pm$ {self.v_bd_err:0.3} [V]"

        props = dict(boxstyle="round", facecolor=color, alpha=0.4)
        fig.text(0.6, 0.45, textstr, fontsize=8, verticalalignment="top", bbox=props)
        plt.tight_layout()

        if out_file:
            data = {
                x_label: data_x,
                x_label + " error": data_x_err,
                y_label: data_y,
                y_label + " error": data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)

    def plot_CA(self, color: str = "blue", out_file: str = None) -> None:
        """
        Plot the correlated avalanche (CA) as a function of overvoltage.

        This method creates a plot of the CA as a function of overvoltage. The plot includes
        the best fit line and its uncertainty, the data points with their errors, and a text box
        with additional information about the data and the fit. The plot can be saved to a file.

        Parameters:
        color (str, optional): The color to use for the text box. Default is 'blue'.
        out_file (str, optional): The name of the file to save the plot to. If None, the plot is not saved to a file. Default is None.

        Returns:
        None
        """
        color = "tab:" + color
        fig = plt.figure()

        data_x = self.ov
        data_x_err = self.bias_err
        data_y = self.CA_vals
        data_y_err = self.CA_err

        fit_x = np.linspace(0, np.amax(self.ov) + 1.0, num=100)
        fit_y = self.CA_res.eval(params=self.CA_res.params, x=fit_x)
        # fit_y = [CA_func(x, self.CA_res.params['A'].value, self.CA_res.params['B'].value, self.CA_res.params['C'].value) for x in fit_x]
        # print(self.CA_res.params['A'].value)
        fit_y_err = self.CA_res.eval_uncertainty(
            x=fit_x, params=self.CA_res.params, sigma=1
        )

        plt.fill_between(
            fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color="deeppink", alpha=0.5
        )
        plt.plot(
            fit_x,
            fit_y,
            color="deeppink",
            label=r"$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit",
        )
        plt.errorbar(
            data_x,
            data_y,
            xerr=data_x_err,
            yerr=data_y_err,
            markersize=10,
            fmt=".",
            label=r"$\frac{1}{N}\sum_{i=1}^{N}{\frac{A_i}{\bar{A}_{1 PE}}-1}$",
        )

        x_label = "Overvoltage [V]"
        y_label = "Number of CA [PE]"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper left")
        textstr = f"Date: {self.campaign[0].info.date}\n"
        textstr += f"Condition: {self.campaign[0].info.condition}\n"
        textstr += f"RTD4: {self.campaign[0].info.temperature} [K]\n"
        if self.filtered:
            textstr += f"Filtering: Lowpass, 400kHz\n"
        else:
            textstr += f"Filtering: None\n"
        textstr += f"--\n"
        textstr += f"""A: {self.CA_res.params['A'].value:0.3} $\pm$ {self.CA_res.params['A'].stderr:0.3}\n"""
        textstr += f"""B: {self.CA_res.params['B'].value:0.2} $\pm$ {self.CA_res.params['B'].stderr:0.2}\n"""
        # textstr += f"""C: {self.CA_res.params['C'].value:0.2} $\pm$ {self.CA_res.params['C'].stderr:0.2}\n"""
        textstr += rf"""Reduced $\chi^2$: {self.CA_res.redchi:0.4}"""
        props = dict(boxstyle="round", facecolor=color, alpha=0.4)
        fig.text(0.15, 0.65, textstr, fontsize=8, verticalalignment="top", bbox=props)
        plt.tight_layout()
        if out_file:
            data = {
                x_label: data_x,
                x_label + " error": data_x_err,
                y_label: data_y,
                y_label + " error": data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)

    def plot_CA_rms(self, color: str = "blue", out_file: str = None) -> None:
        """
        Plot the root mean square (RMS) of the charge amplification (CA) as a function of overvoltage.

        This method creates a plot of the RMS of the CA as a function of overvoltage. The plot includes
        the data points with their errors and a text box with additional information about the data.
        The plot can be saved to a file.

        Parameters:
        color (str, optional): The color to use for the text box. Default is 'blue'.
        out_file (str, optional): The name of the file to save the plot to. If None, the plot is not saved to a file. Default is None.

        Returns:
        None
        """
        color = "tab:" + color
        fig = plt.figure()

        data_x = self.bias_vals
        data_x_err = self.bias_err
        data_y = self.CA_rms_vals
        data_y_err = self.CA_rms_err

        plt.errorbar(
            data_x,
            data_y,
            xerr=data_x_err,
            yerr=data_y_err,
            markersize=10,
            fmt=".",
            label=r"$\sqrt{\frac{\sum_{i=1}^{N}\left(\frac{A_i}{\bar{A}_{1 PE}}-\left(\langle\Lambda\rangle+1\right)\right)^2}{N}}$",
        )

        x_label = "Bias voltage [V]"
        y_label = "RMS CAs [PE]"
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper left")
        textstr = f"Date: {self.campaign[0].info.date}\n"
        textstr += f"Condition: {self.campaign[0].info.condition}\n"
        textstr += f"RTD4: {self.campaign[0].info.temperature} [K]\n"
        if self.filtered:
            textstr += f'Filtering: Lowpass, 400kHz\n'
            # textstr += f"Filtering: Bandpass [1E4, 1E6]\n"
        else:
            textstr += f"Filtering: None\n"

        props = dict(boxstyle="round", facecolor=color, alpha=0.4)
        fig.text(0.15, 0.65, textstr, fontsize=8, verticalalignment="top", bbox=props)
        plt.tight_layout()
        if out_file:
            data = {
                x_label: data_x,
                x_label + " error": data_x_err,
                y_label: data_y,
                y_label + " error": data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)

    # I HAVE A SECRET TO TELL YOU! (it was reed who wrote that message and thwy are pinning it on me)


#    it worked.


class Alpha_data:
    def __init__(
        self,
        campaign: list[WaveformProcessor],
        invC: float,
        invC_err: float,
        spe_data: SPE_data,
        v_bd: float,
        v_bd_err: float,
    ) -> None:
        """
        Initialize the Alpha_data class. This class is used for collecting all the WaveformProcessor results for
        an entire campaign into one object. Can then be used to determine PDE and the equivalent number of detected photons.

        Parameters:
        campaign (List): The campaign data.
        invC (float): The inverse of the capacitance.
        invC_err (float): The error in the inverse of the capacitance.
        spe_data (SPE_data): The single photoelectron data.
        v_bd (float): The breakdown voltage.
        v_bd_err (float): The error in the breakdown voltage.

        Returns:
        None
        """
        self.campaign = campaign
        self.invC = invC
        self.invC_err = invC_err
        self.spe_data = spe_data
        self.v_bd = v_bd
        self.v_bd_err = v_bd_err
        self.analyze_alpha()

    def analyze_alpha(self) -> None:
        """
        Analyze the alpha data.

        This method calculates various parameters related to the alpha data such as bias values,
        alpha values, overvoltage values, and the number of detected photons.

        Returns:
        None
        """
        self.bias_vals = []
        self.bias_err = []
        self.alpha_vals = []
        self.alpha_err = []
        
        self.sigma_vals = []
        self.sigma_err = []

        for wp in self.campaign:
            self.bias_vals.append(wp.info.bias)
            self.bias_err.append(0.0025 * wp.info.bias + 0.015)
            # self.bias_err.append(0.005)
            curr_alpha = wp.get_alpha()
            self.alpha_vals.append(curr_alpha[0])
            self.alpha_err.append(curr_alpha[1])
            curr_sigma = wp.get_sigma()
            self.sigma_vals.append(curr_sigma[0])
            self.sigma_err.append(curr_sigma[1])            

        self.bias_vals = np.array(self.bias_vals)
        self.bias_err = np.array(self.bias_err)
        self.alpha_vals = np.array(self.alpha_vals)
        self.alpha_err = np.array(self.alpha_err)

        self.ov = []
        self.ov_err = []

        for b, db in zip(self.bias_vals, self.bias_err):
            curr_ov = b - self.v_bd
            curr_ov_err = np.sqrt(db * db + self.v_bd_err * self.v_bd_err)
            self.ov.append(curr_ov)
            self.ov_err.append(curr_ov_err)

        self.ov = np.array(self.ov)
        self.ov_err = np.array(self.ov_err)

        self.sigma_vals = np.array(self.sigma_vals)
        self.sigma_err = np.array(self.sigma_err)

        self.CA_vals, self.CA_err = self.spe_data.get_CA_ov(self.ov)
        self.spe_vals, self.spe_err = self.spe_data.get_spe_ov(self.ov)
        print("CA Vals: " + str(self.CA_vals))
        print("SPE Vals: " + str(self.spe_vals))

# in p.e. units
        self.sigma_in_spe_units = self.sigma_vals/self.spe_vals * self.spe_data.invC/self.invC
        self.sigma_in_spe_units_err = self.sigma_in_spe_units * np.sqrt((self.sigma_err * self.sigma_err)/(self.sigma_vals * self.sigma_vals) 
                                                                        + (self.spe_err*self.spe_err)/(self.spe_vals*self.spe_vals) 
                                                                        + (self.invC_err*self.invC_err)/(self.invC*self.invC) 
                                                                        + (self.spe_data.invC_err*self.spe_data.invC_err)/(self.spe_data.invC*self.spe_data.invC)
                                                                        )
        self.alpha_in_spe_units = self.alpha_vals/self.spe_vals * self.spe_data.invC/self.invC
        self.alpha_in_spe_units_err = self.alpha_in_spe_units * np.sqrt((self.alpha_err * self.alpha_err)/(self.alpha_vals * self.alpha_vals) 
                                                                        + (self.spe_err*self.spe_err)/(self.spe_vals*self.spe_vals) 
                                                                        + (self.invC_err*self.invC_err)/(self.invC*self.invC) 
                                                                        + (self.spe_data.invC_err*self.spe_data.invC_err)/(self.spe_data.invC*self.spe_data.invC) 
                                                                        )                                                                        

        self.num_det_photons = (
            self.alpha_vals
            * self.spe_data.invC
            / (self.spe_vals * self.invC * (1.0 + self.CA_vals))
        )
        self.num_det_photons_err = self.num_det_photons * np.sqrt(
            (self.alpha_err * self.alpha_err) / (self.alpha_vals * self.alpha_vals)
            + (self.spe_data.invC_err * self.spe_data.invC_err)
            / (self.spe_data.invC * self.spe_data.invC)
            + (self.spe_err * self.spe_err) / (self.spe_vals * self.spe_vals)
            + (self.invC_err * self.invC_err) / (self.invC * self.invC)
            + (self.CA_err * self.CA_err) / (self.CA_vals * self.CA_vals)
        )

    def plot_alpha(self, color: str = "purple", units = "Volts", x = "Bias", out_file: str = None) -> None:
        """
        Plot the alpha data as a function of overvoltage.

        This method creates a plot of the alpha data as a function of overvoltage. The plot includes
        the data points with their errors and a text box with additional information about the data.
        The plot can be saved to a file.

        Parameters:
        color (str, optional): The color to use for the data points. Default is 'purple'.
        out_file (str, optional): The name of the file to save the plot to. If None, the plot is not saved to a file. Default is None.

        Returns:
        None
        """
        color = "tab:" + color
        fig = plt.figure()
        fig.tight_layout()
        plt.rc("font", size=12)

        
        if x == "OV":
            data_x = self.ov
            data_x_err = self.bias_err
            x_label = "Overvoltage [V]"
        if x == "Bias":
            data_x = self.bias_vals
            data_x_err = self.bias_err
            x_label = "Bias voltage [V]"
        
        if units == "Volts":
            data_y = self.alpha_vals
            data_y_err = self.alpha_err
            y_label = "Alpha Pulse Amplitude [V]"
        if units == "PE":
            data_y = self.alpha_in_spe_units
            data_y_err = self.alpha_in_spe_units_err
            y_label = "Alpha Amplitude [p.e.]"
            
        # fit_x = np.linspace(0.0, np.amax(self.ov) + 1.0, num = 100)
        # fit_y = self.CA_res.eval(params=self.CA_res.params, x = fit_x)
        # fit_y_err = self.CA_res.eval_uncertainty(x = fit_x, params = self.CA_res.params, sigma = 1)

        # plt.fill_between(fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color = 'red', alpha = .5)
        # plt.plot(fit_x, fit_y, color = 'red', label = r'$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit')
        plt.errorbar(
            data_x,
            data_y,
            xerr=data_x_err,
            yerr=data_y_err,
            markersize=10,
            fmt=".",
            color=color,
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        textstr = f"Date: {self.campaign[0].info.date}\n"
        textstr += f"Condition: {self.campaign[0].info.condition}\n"
        textstr += f"RTD4: {self.campaign[0].info.temperature} [K]"
        # if self.filtered:
        #     textstr += f'Filtering: Lowpass, 400kHz\n'
        # else:
        #     textstr += f'Filtering: None\n'
        # textstr += f'--\n'
        # textstr += f'''A: {self.CA_res.params['A'].value:0.3f} $\pm$ {self.CA_res.params['A'].stderr:0.3f}\n'''
        # textstr += f'''B: {self.CA_res.params['B'].value:0.2} $\pm$ {self.CA_res.params['B'].stderr:0.2}\n'''
        # textstr += rf'''Reduced $\chi^2$: {self.CA_res.redchi:0.4}'''

        plt.grid(True)

        props = dict(boxstyle="round", facecolor=color, alpha=0.4)
        fig.text(0.15, 0.75, textstr, fontsize=12, verticalalignment="top", bbox=props)
        if out_file:
            data = {
                x_label: data_x,
                x_label + " error": data_x_err,
                y_label: data_y,
                y_label + " error": data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)
        plt.show()
        
        
    def plot_sigma(self, color: str = "purple", units = "Volts", x = "Bias", out_file: str = None) -> None:
        """
        Plot the fitted alpha sigmas as a function of overvoltage.

        Parameters:
        color (str, optional): The color to use for the data points. Default is 'purple'.
        out_file (str, optional): The name of the file to save the plot to. If None, the plot is not saved to a file. Default is None.

        Returns:
        None
        """
        color = "tab:" + color
        fig = plt.figure()
        fig.tight_layout()
        plt.rc("font", size=12)
        
        if x == "OV":
            data_x = self.ov
            data_x_err = self.bias_err
            x_label = "Overvoltage [V]"
        if x == "Bias":
            data_x = self.bias_vals
            data_x_err = self.bias_err
            x_label = "Bias voltage [V]"
        
        if units == "Volts":
            data_y = self.sigma_vals
            data_y_err = self.sigma_err
            y_label = "Fitted Pulse Sigma [V]"
        if units == "PE":
            data_y = self.sigma_in_spe_units
            data_y_err = self.sigma_in_spe_units_err
            y_label = "Fitted Pulse Sigma [p.e.]"

        plt.errorbar(
            data_x,
            data_y,
            xerr=data_x_err,
            yerr=data_y_err,
            markersize=10,
            fmt=".",
            color=color,
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        textstr = f"Date: {self.campaign[0].info.date}\n"
        textstr += f"Condition: {self.campaign[0].info.condition}\n"
        textstr += f"RTD4: {self.campaign[0].info.temperature} [K]"
        # if self.filtered:
        #     textstr += f'Filtering: Lowpass, 400kHz\n'
        # else:
        #     textstr += f'Filtering: None\n'
        # textstr += f'--\n'
        # textstr += f'''A: {self.CA_res.params['A'].value:0.3f} $\pm$ {self.CA_res.params['A'].stderr:0.3f}\n'''
        # textstr += f'''B: {self.CA_res.params['B'].value:0.2} $\pm$ {self.CA_res.params['B'].stderr:0.2}\n'''
        # textstr += rf'''Reduced $\chi^2$: {self.CA_res.redchi:0.4}'''

        plt.grid(True)

        props = dict(boxstyle="round", facecolor=color, alpha=0.4)
        fig.text(0.15, 0.75, textstr, fontsize=18, verticalalignment="top", bbox=props)
        if out_file:
            data = {
                x_label: data_x,
                x_label + " error": data_x_err,
                y_label: data_y,
                y_label + " error": data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)
        plt.show()

    def plot_num_det_photons(self, color: str = "purple", out_file: str = None) -> None:
        """
        Plot the number of detected photons as a function of overvoltage.

        This method creates a plot of the number of detected photons as a function of overvoltage.
        The plot includes the data points with their errors and a text box with additional information
        about the data. The plot can be saved to a file.

        Parameters:
        color (str, optional): The color to use for the data points. Default is 'purple'.
        out_file (str, optional): The name of the file to save the plot to. If None, the plot is not saved to a file. Default is None.

        Returns:
        None
        """
        color = "tab:" + color
        fig = plt.figure()

        data_x = self.ov
        data_x_err = self.ov_err
        data_y = self.num_det_photons
        data_y_err = self.num_det_photons_err

        plt.errorbar(
            data_x,
            data_y,
            xerr=data_x_err,
            yerr=data_y_err,
            markersize=10,
            fmt=".",
            color=color,
        )

        plt.xlabel("Overvoltage [V]")
        plt.ylabel("Number of Detected Photons")
        textstr = f"Date: {self.campaign[0].info.date}\n"
        textstr += f"Condition: {self.campaign[0].info.condition}\n"
        textstr += f"RTD4: {self.campaign[0].info.temperature} [K]"
        # if self.filtered:
        #     textstr += f'Filtering: Lowpass, 400kHz\n'
        # else:
        #     textstr += f'Filtering: None\n'

        props = dict(boxstyle="round", facecolor=color, alpha=0.4)
        fig.text(0.3, 0.4, textstr, fontsize=18, verticalalignment="top", bbox=props)
        plt.xlim(0, np.amax(self.ov) + 1.0)
        ylow, yhigh = plt.ylim()
        plt.ylim(-1, yhigh * 1.1)
        plt.tight_layout()

    def plot_PDE(
        self,
        num_incident: int,
        color: str = "purple",
        other_data: list = None,
        out_file: str = None,
    ) -> None:
        """
        Plot the Photon Detection Efficiency (PDE) as a function of overvoltage.

        This method creates a plot of the PDE as a function of overvoltage. The plot includes
        the data points with their errors and a text box with additional information about the data.
        The plot can also include data from other sources. The plot can be saved to a file.

        Parameters:
        num_incident (int): The number of incident photons.
        color (str, optional): The color to use for the data points. Default is 'purple'.
        other_data (List, optional): A list of other data to include in the plot. Each item in the list should be a dictionary with keys 'ov', 'pde', 'ov_err', 'pde_err', and 'label'. Default is None.
        out_file (str, optional): The name of the file to save the plot to. If None, the plot is not saved to a file. Default is None.

        Returns:
        None
        """
        color = "tab:" + color
        fig = plt.figure()

        data_x = self.ov
        data_x_err = self.ov_err
        data_y = self.num_det_photons / num_incident
        data_y_err = self.num_det_photons_err / num_incident

        plt.errorbar(
            data_x,
            data_y,
            xerr=data_x_err,
            yerr=data_y_err,
            markersize=10,
            fmt=".",
            color=color,
            label="UMass, 175nm, 190K",
        )

        if other_data:
            for od in other_data:
                plt.errorbar(
                    od.ov,
                    od.pde,
                    xerr=od.ov_err,
                    yerr=od.pde_err,
                    markersize=10,
                    fmt=".",
                    label=od.label,
                )

        plt.xlabel("Overvoltage [V]")
        plt.ylabel("Photon Detection Efficiency")
        textstr = f"Date: {self.campaign[0].info.date}\n"
        textstr += f"Condition: {self.campaign[0].info.condition}\n"
        textstr += f"RTD4: {self.campaign[0].info.temperature} [K]"

        props = dict(boxstyle="round", facecolor=color, alpha=0.4)
        fig.text(0.3, 0.4, textstr, fontsize=18, verticalalignment="top", bbox=props)
        plt.xlim(0, np.amax(self.ov) + 1.0)
        ylow, yhigh = plt.ylim()
        plt.ylim(-0.01, yhigh * 1.1)
        if other_data:
            plt.legend(loc="lower left")
        plt.tight_layout()
        if out_file:
            x_label = "Overvoltage [V]"
            y_label = "Photon Detection Efficiency"
            data = {
                x_label: data_x,
                x_label + " error": data_x_err,
                y_label: data_y,
                y_label + " error": data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)


class Collab_PDE:
    """
    A class used to represent the Photon Detection Efficiency (PDE) data from a collaboration.

    Attributes:
    filename (str): The name of the file containing the PDE data.
    groupname (str): The name of the group that collected the data.
    wavelength (float): The wavelength of the photons used in the experiment.
    temp (float): The temperature at which the data was collected.
    df (DataFrame): A pandas DataFrame containing the data from the file.
    ov (np.array): An array of overvoltage values.
    ov_err (np.array): An array of errors in the overvoltage values.
    pde (np.array): An array of PDE values.
    pde_err (np.array): An array of errors in the PDE values.
    """

    def __init__(
        self, filename: str, groupname: str, wavelength: float, temp: float
    ) -> None:
        """
        Initialize a Collab_PDE object.

        Parameters:
        filename (str): The name of the file containing the PDE data.
        groupname (str): The name of the group that collected the data.
        wavelength (float): The wavelength of the photons used in the experiment.
        temp (float): The temperature at which the data was collected.

        Returns:
        None
        """
        self.filename = filename
        self.groupname = groupname
        self.wavelength = wavelength
        self.temp = temp
        self.df = pd.read_csv(self.filename)
        self.ov = np.array(self.df["OV"])
        self.ov_err = np.array(self.df["OV error"])
        self.pde = np.array(self.df["PDE"]) / 100.0
        self.pde_err = np.array(self.df["PDE error"]) / 100.0


# %%
class multi_campaign:  # class to compile multiple campaigns
    def __init__(
        self, campaigns: list, invC: float, invC_err: float, filtered: bool
    ) -> None:
        """
        Initialize a multi_campaign object.

        Parameters:
        campaigns (List): A list of campaigns to compile.
        invC (float): The inverse of the capacitance of the device.
        invC_err (float): The error in the inverse of the capacitance.
        filtered (bool): Whether the data has been filtered.

        Returns:
        None
        """
        self.campaigns = campaigns
        self.invC = invC
        self.invC_err = invC_err
        self.filtered = filtered
        self.create_SPEs()

    def create_SPEs(
        self,
    ) -> None:  # does SPE_data on all the campaigns and returns a list of objects
        """
        Create SPE_data objects for all the campaigns.

        This method creates an SPE_data object for each campaign in the list of campaigns and stores them in the data attribute.

        Parameters:
        None

        Returns:
        None
        """
        self.data = []
        for curr_campaign in self.campaigns:
            self.data.append(
                SPE_data(
                    curr_campaign, invC_spe_filter, invC_spe_err_filter, filtered=True
                )
            )


# %%
# ihep_pde = Collab_PDE('C:/Users/lab-341/Desktop/Analysis/fresh_start/PDE_175nm_HD3_iHEP_233K.csv', 'IHEP', 175, 233)
# triumf_pde = Collab_PDE('C:/Users/lab-341/Desktop/Analysis/fresh_start/PDE_176nm_HD3_Triumf_163K.csv', 'TRIUMF', 176, 163)
