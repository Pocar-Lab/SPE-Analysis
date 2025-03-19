# -*- coding: utf-8 -*-
"""
Created on Nov 13 2024

@author: Ed van Bruggen evanbruggen@mass.edu
"""

import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import typing
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy
from functools import partial

# import GainCalibration_2022 as GainCalibration
from ProcessWaveforms_MultiGaussian import WaveformProcessor

# def ca_func(x, A, B, C):
#     return (A * np.exp(B*(x)) + 1.0) / (1.0 + A) - 1.0 + C
def ca_func(x, A, B):
    return (A * np.exp(B*(x)) + 1.0) / (1.0 + A) - 1.0

def fullquad(x, a, b, c):
    return a*x*x + b*x + c
def quad(x, a, c):
    return a*x*x + c
def quad0(x, a, b):
    return a*x*x + b*x
def linear(x, b, c):
    return b*x + c

def exp(x, a, b):
    return a*np.exp(b*x)
def expda(x, a, b):
    return np.exp(b*x)
def expdb(x, a, b):
    return a*x*np.exp(b*x)
def uexp(x, a, b):
    return a*unumpy.exp(b*x)

def exp1(x, a, b):
    return a*np.exp(b*x) - a
def exp1da(x, a, b):
    return np.exp(b*x) - 1
def exp1db(x, a, b):
    return a*x*np.exp(b*x)
def uexp1(x, a, b):
    return a*unumpy.exp(b*x) - a

def xexp(x, a, b):
    return a*x*np.exp(b*x)
def xexpda(x, a, b):
    return x*np.exp(b*x)
def xexpdb(x, a, b):
    return a*x*x*np.exp(b*x)
def uxexp(x, a, b):
    return a*x*unumpy.exp(b*x)



def alpha_csv(wp, v_bd, v_bd_err):
    bias_vals  = []
    bias_err   = []
    alpha_vals = []
    alpha_err  = []
    sigma_vals = []
    sigma_err  = []
    for wp in campaign:
        bias_vals.append(wp.info.bias)
        bias_err.append(0.0025 * wp.info.bias + 0.015)
        # bias_err.append(0.005)
        curr_alpha = wp.get_alpha()
        alpha_vals.append(curr_alpha[0])
        alpha_err.append(curr_alpha[1])
        curr_sigma = wp.get_sigma()
        sigma_vals.append(curr_sigma[0])
        sigma_err.append(curr_sigma[1])
    bias_vals = np.array(bias_vals)
    bias_err = np.array(bias_err)
    alpha_vals = np.array(alpha_vals)
    alpha_err = np.array(alpha_err)
    sigma_vals = np.array(sigma_vals)
    sigma_err = np.array(sigma_err)

    ov = []
    ov_err = []
    for b, db in zip(bias_vals, bias_err):
        curr_ov = b - v_bd
        curr_ov_err = np.sqrt(db * db + v_bd_err * v_bd_err)
        ov.append(curr_ov)
        ov_err.append(curr_ov_err)
    ov = np.array(ov)
    ov_err = np.array(ov_err)

    data = {
        'ov': ov, 'ov error': ov_err,
        'bias': bias_vals, 'bias error': bias_err,
        'amps': alpha_vals, 'amps error': alpha_err,
        # 'num_det': self.num_det_photons, 'num_det error': self.num_det_photons_err,
        # 'pde': data_y, 'pde error': data_y_err,
    }
    df = pd.DataFrame(data)
    df.to_csv(out_file)

def read_spe(df, new_bias, invC, invC_err):
    bias_vals = df['Bias Voltage [V]']
    bias_err = df['Bias Voltage [V] error']
    spe_vals = df['SPE Amplitude [V]']
    spe_err = df['SPE Amplitude [V] error']
    absolute_spe_vals = spe_vals / (invC * 1.60217662e-7)
    absolute_spe_err = absolute_spe_vals * np.sqrt(
        (spe_err * spe_err) / (spe_vals * spe_vals)
        + (invC_err * invC_err) / (invC * invC)
    )
    spe_wgts = [1.0 / curr_std for curr_std in spe_err]
    absolute_spe_wgts = [1.0 / curr_std for curr_std in absolute_spe_err]

    model = lm.models.LinearModel()
    params = model.make_params()
    spe_res = model.fit(
        spe_vals, params=params, x=bias_vals, weights=spe_wgts
    )
    absolute_spe_res = model.fit(
        absolute_spe_vals, params=params, x=bias_vals, weights=absolute_spe_wgts,
    )

    # input_bias_vals = ov + v_bd
    out_vals = spe_res.eval(params=spe_res.params, x=new_bias)
    out_err = spe_res.eval_uncertainty(x=new_bias, sigma=1)
    return out_vals, out_err

def read_ca(df, input_ov_vals):
    ov_vals = df['Overvoltage [V]']
    ov_err = df['Overvoltage [V] error']
    ca_vals = df['Number of CA [PE]']
    ca_err = df['Number of CA [PE] error']

    ca_model = lm.Model(ca_func)
    # ca_params = ca_model.make_params(A=1, B= 1, C = 0)
    ca_params = ca_model.make_params(A=1, B=1)
    # ca_params['A'].min = 0.1
    # ca_params['C'].min = -5
    # ca_params['C'].max = 5
    ca_wgts = [1.0 / curr_std for curr_std in ca_err]
    ca_res = ca_model.fit(
        ca_vals, params=ca_params, x=ov_vals, weights=ca_wgts
    )

    out_vals = ca_res.eval(params=ca_res.params, x=input_ov_vals)
    out_err = ca_res.eval_uncertainty(x=input_ov_vals, sigma=1)
    return out_vals, out_err

# TODO integrate MeasurmentInfo which includes calibration numbers
class AnalyzeResults:
    def __init__(
        self,
        config,
        alpha_csv,
        spe_csv,
        ca_csv,
        invC_alpha,
        invC_alpha_err,
        invC_spe,
        invC_spe_err,
        v_bd,
        v_bd_err,
        ov_min = 2,
        ov_max = 9,
    ) -> None:
        self.config = config
        self.invC_alpha = invC_alpha
        self.invC_alpha_err = invC_alpha_err
        self.invC_spe = invC_spe
        self.invC_spe_err = invC_spe_err
        self.v_bd = v_bd
        self.v_bd_err = v_bd_err

        alpha_df = pd.read_csv(alpha_csv).sort_values('Bias Voltage [V]')
        alpha_df = alpha_df[alpha_df['Bias Voltage [V]'] <= ov_max + v_bd]
        alpha_df = alpha_df[alpha_df['Bias Voltage [V]'] >= ov_min + v_bd]
        # alpha_df = alpha_df[alpha_df['Bias Voltage [V]'] <= 35]
        # alpha_df = alpha_df[alpha_df['Bias Voltage [V]'] >= 31]
        self.bias_vals  = alpha_df['Bias Voltage [V]']
        self.bias_err   = alpha_df['Bias Voltage error [V]']
        self.alpha_vals = alpha_df['Alpha Pulse Amplitude [V]']
        self.alpha_err  = alpha_df['Alpha Pulse Amplitude error [V]']
        # self.ov = alpha_df['ov']
        # self.ov_err = alpha_df['ov error']
        self.ov = self.bias_vals - v_bd
        self.ov_err = self.bias_err # TODO incorporate v_bd_err
        self.ov_min = ov_min
        self.ov_max = ov_max

        spe_df = pd.read_csv(spe_csv)
        ca_df = pd.read_csv(ca_csv)
        self.spe_vals, self.spe_err = read_spe(spe_df, self.bias_vals, invC_spe, invC_spe_err)
        self.ca_vals, self.ca_err = read_ca(ca_df, self.ov)
        # self.spe_vals, self.spe_err = self.spe_data.get_spe_ov(self.ov)
        # self.CA_vals, self.CA_err = self.spe_data.get_CA_ov(self.ov)

        # # in p.e. units
        # self.sigma_in_spe_units = self.sigma_vals/self.spe_vals * self.spe_data.invC/self.invC
        # self.sigma_in_spe_units_err = self.sigma_in_spe_units * np.sqrt((self.sigma_err * self.sigma_err)/(self.sigma_vals * self.sigma_vals)
        #                                                                 + (self.spe_err*self.spe_err)/(self.spe_vals*self.spe_vals)
        #                                                                 + (self.invC_err*self.invC_err)/(self.invC*self.invC)
        #                                                                 + (self.spe_data.invC_err*self.spe_data.invC_err)/(self.spe_data.invC*self.spe_data.invC)
        #                                                                 )
        # self.alpha_in_spe_units = self.alpha_vals/self.spe_vals * self.spe_data.invC/self.invC
        # self.alpha_in_spe_units_err = self.alpha_in_spe_units * np.sqrt((self.alpha_err * self.alpha_err)/(self.alpha_vals * self.alpha_vals)
        #                                                                 + (self.spe_err*self.spe_err)/(self.spe_vals*self.spe_vals)
        #                                                                 + (self.invC_err*self.invC_err)/(self.invC*self.invC)
        #                                                                 + (self.spe_data.invC_err*self.spe_data.invC_err)/(self.spe_data.invC*self.spe_data.invC)
        #                                                                 )

        self.num_det_photons = (
            self.alpha_vals * self.invC_spe
            / (self.spe_vals * self.invC_alpha * (1.0 + self.ca_vals))
        )
        self.num_det_photons_err = self.num_det_photons * np.sqrt(
            (self.alpha_err * self.alpha_err) / (self.alpha_vals * self.alpha_vals)
            + (self.invC_spe_err * self.invC_spe_err)
            / (self.invC_spe * self.invC_spe)
            + (self.spe_err * self.spe_err) / (self.spe_vals * self.spe_vals)
            + (self.invC_alpha_err * self.invC_alpha_err) / (self.invC_alpha * self.invC_alpha)
            + (self.ca_err * self.ca_err) / (self.ca_vals * self.ca_vals)
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
        # textstr = f"Date: {self.info.date}\n"
        # textstr += f"Condition: {self.info.condition}\n"
        # textstr += f"RTD4: {self.info.temperature} [K]"
        textstr = f"Alpha: 10/17/2024 (LXe @ 170K)\n"
        textstr += f"SPE: 10/17/2024 (LXe @ 170K)\n"
        textstr += f"CA: 07/12/2024 (GN @ 171K)"
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
        data_x_err = self.bias_err # use error from voltage source, not over correlated error from OV
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
        textstr = f"Alpha: 10/17/2024 (LXe @ 170K)\n"
        textstr += f"SPE: 10/17/2024 (LXe @ 170K)\n"
        textstr += f"CA: 07/12/2023 (GN @ 171K)"
        # textstr = f"Date: {self.campaign[0].info.date}\n"
        # textstr += f"Condition: {self.campaign[0].info.condition}\n"
        # textstr += f"RTD4: {self.campaign[0].info.temperature} [K]"
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
        plt.grid(True)
        plt.show()

        if out_file:
            x_label = "Overvoltage [V]"
            y_label = "Number of Detected Photons"
            data = {
                x_label: data_x,
                x_label + " error": data_x_err,
                y_label: data_y,
                y_label + " error": data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)

    def plot_PDE(
        self,
        num_incident: int,
        color: str = "purple",
        other_data: list = None,
        out_file: str = None,
        legtext: str = "",
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
        textstr += f"RTD4: {self.campaign[0].info.temperature} [K]\n"
        textstr += f"PTE: {legtext}"

        props = dict(boxstyle="round", facecolor=color, alpha=0.4)
        fig.text(0.3, 0.4, textstr, fontsize=18, verticalalignment="top", bbox=props)
        plt.xlim(0, np.amax(self.ov) + 1.0)
        ylow, yhigh = plt.ylim()
        plt.ylim(-0.01, yhigh * 1.1)
        if other_data:
            plt.legend(loc="lower left")
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        if out_file:
            data = {
                'ov': data_x, 'ov error': data_x_err,
                'bias': self.bias_vals, 'bias error': self.bias_err,
                'amps': self.alpha_vals, 'amps error': self.alpha_err,
                'num_det': self.num_det_photons, 'num_det error': self.num_det_photons_err,
                'pde': data_y, 'pde error': data_y_err,
            }
            df = pd.DataFrame(data)
            df.to_csv(out_file)

    def fit_alpha(self, bias, fit='exp', color="purple"):
        if fit == 'exp':
            fit_func = exp
            fit_ufunc = uexp
            fit_funcda = expda
            fit_funcdb = expdb
            init_guess = dict(a=1, b=1)
            eq_label = '$a e^{b x}$'
        elif fit == 'exp1':
            fit_func = exp1
            fit_ufunc = uexp1
            fit_funcda = exp1da
            fit_funcdb = exp1db
            init_guess = dict(a=1, b=1)
            eq_label = '$a (e^{b x} - 1)$'
        elif fit == 'xexp':
            fit_func = xexp
            fit_ufunc = uxexp
            fit_funcda = xexpda
            fit_funcdb = xexpdb
            init_guess = dict(a=1, b=1)
            eq_label = '$a x e^{b x}$'
        elif fit == 'quad':
            fit_func = fit_ufunc = quad
            init_guess = dict(a=1, c=0)
            eq_label = '$a x^2 + c$'
        elif fit == 'fullquad':
            fit_func = fit_ufunc = fullquad
            init_guess = dict(a=1, b=1, c=0)
            eq_label = '$a x^2 + b x + c$'
        elif fit == 'quad0':
            fit_func = fit_ufunc = quad0
            init_guess = dict(a=1, b=1)
            eq_label = '$a x^2 + b x$'
        elif fit == 'linear':
            fit_func = fit_ufunc = linear
            init_guess = dict(b=1, c=0)
            eq_label = '$y = b x + c$'
        else:
            raise ValueError('Incorrect fit parameter, expected: exp, quad, or fullquad')

        model = lm.Model(fit_func)
        res = model.fit(self.alpha_vals, x=self.ov, **init_guess)
        print(res.redchi)
        params, covar = list(res.best_values.values()), res.covar

        # params, covar = curve_fit(fit_func, self.ov, self.alpha_vals) #, sigma=d10_y_err
        print(f"{self.config}: {covar=}")
        perr = np.sqrt(np.diag(covar))
        uparams = unumpy.uarray(params, perr)
        x = np.linspace(.1, 10, 100)
        # x = np.linspace(self.ov_min+.3, self.ov_max, num=100)
        yfitn = fit_func(x, *params)
        yfits = np.sqrt(fit_funcda(x, *params)**2 * perr[0]**2
                        +fit_funcdb(x, *params)**2 * perr[1]**2
                        +2*fit_funcda(x, *params)*fit_funcdb(x, *params)*covar[1,0]
                        )
        ufit = unumpy.uarray(yfitn, yfits)
        if color:
            plt.grid(True)
            plt.plot(x, fit_func(x, *params), color=color,
                     label=eq_label + f" with $\chi^2 = $ {res.redchi:.3g},\n"
                     + f"$a$ = {res.best_values['a']:.3g}, "
                     + f"$b$ = {res.best_values['b']:.3g}, "
                     # + f"$c$ = {res.best_values['c']:.3g}"
                     )
            plt.fill_between(x, yfitn - yfits, yfitn + yfits, alpha=.3, color=color)
            plt.errorbar(self.ov, self.alpha_vals, xerr=self.ov_err, yerr=self.alpha_err,
                         markersize=8, fmt='.', color=color, label=self.config)
        # plt.plot(self.ov, res.best_fit, color='tab:green', label=eq_label)

        return ufit

    def plot_ratio(self, other, color="tab:purple", color_other="tab:blue", fit='exp',
                   alpha_ylim=(0, 1.5), ratio_ylim=(1,4)):
        udata_x = unumpy.uarray(self.ov, self.ov_err)
        udata_y = unumpy.uarray(self.alpha_vals, self.alpha_err)
        # udata10_y = unumpy.uarray(other.alpha_vals, other.alpha_err)

        fig,ax = plt.subplots()
        fig.tight_layout()
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1/4))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1/4))

        x = np.linspace(self.ov_min+.3, self.ov_max, num=100)
        # x = np.linspace(self.ov_min-1, self.ov_max+1, num=100)
        # x = np.linspace(0.1, 10, num=100)
        udata10fit_y = other.fit_alpha(x, fit, color_other)
        udata_y = self.fit_alpha(x, fit, color)

        ratio = udata_y / udata10fit_y
        # ratio = udata10fit_y / udata_y
        ration = np.array([ r.n for r in ratio ])
        ratios = np.array([ r.s for r in ratio ])

        # plt.rc('font', size=22)
        # ax.fill_between(data_x[:7], yfitn[:7] - yfits[:7], yfitn[:7] + yfits[:7], alpha=.3)
        # ax.errorbar(self.ov, self.alpha_vals, xerr = self.ov_err, yerr = self.alpha_err, markersize = 8, fmt = '.',
        #             color=color, label=self.config)
        # ax.errorbar(other.ov, other.alpha_vals, xerr = other.ov_err, yerr = other.alpha_err, markersize = 8, fmt = '.',
        #             color = 'tab:blue', label='8 Short Si')
        # plt.errorbar(self.ov, ration, xerr=self.ov_err, yerr=ratios, markersize = 8, fmt = '.', color =
        #              'tab:green', label='Ratio (Average: 2.238±.00058)')
        axr = ax.twinx()
        axr.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1/4))
        # , xerr = self.ov_err
        # axr.errorbar(x, ration, yerr=ratios, markersize=0, fmt='.',
        #             color='tab:green', label=f'Ratio {ratio.mean()}')

        axr.plot(x, ration, color='tab:green', label=f'Ratio {ratio.mean()}')
        axr.fill_between(x, ration - ratios, ration + ratios, alpha=.3, color='tab:green')
        axr.set_ylim(*ratio_ylim)
        ax.set_ylim(*alpha_ylim)
        # axr.fill_between(self.ov, ration - ratios, ration + ratios, alpha=.3)
        ax.set_xlabel('Overvoltage [V]', loc='right')
        # plt.ylabel('Number of Detected Photons')
        ax.set_ylabel('Alpha Amplitude [V]', loc='top')
        axr.set_ylabel('Ratio', loc='top')
        # textstr = f'Date: {self.campaign[0].info.date}\n'
        # textstr = f'Tall Silicon Reflector\n' # UPDATE
        # textstr += f'Condition: LXe\n'
        # textstr += f'RTD4: 167 [K]'
        # props = dict(boxstyle='round', facecolor=color, alpha=0.4)
        # fig.text(.5, .5, textstr, fontsize=10,
        #         verticalalignment='top', bbox=props)
        ax.legend(loc='upper left')
        axr.legend(loc='upper right')
        # plt.legend()
        plt.show()

    def plot_sub_ratio(self, sub, denom, factor
                       color="tab:purple", color_other="tab:blue", fit='exp',
                       alpha_ylim=(0, 1.5), ratio_ylim=(1,4)):

        fig,ax = plt.subplots()
        fig.tight_layout()
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1/4))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(.1/4))

        x = np.linspace(self.ov_min+.3, self.ov_max, num=100)
        subfit = sub.fit_alpha(x, fit, color_other)
        denfit = denom.fit_alpha(x, fit, "tab:pink")
        selffit = self.fit_alpha(x, fit, color)

        ratio = (selffit - subfit * factor) / denfit
        ration = np.array([ r.n for r in ratio ])
        ratios = np.array([ r.s for r in ratio ])

        # subr = selffit - subfit # * factor
        # subrn = np.array([ r.n for r in subr ])
        # subrs = np.array([ r.s for r in subr ])
        # plt.plot(x, subrn, color='tab:cyan')
        # plt.fill_between(x, subrn - subrs, subrn + subrs, alpha=.3, color='tab:cyan')

        axr = ax.twinx()
        axr.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1/4))

        axr.plot(x, ration, color='tab:green', label=f'Ratio {ratio.mean()}')
        axr.fill_between(x, ration - ratios, ration + ratios, alpha=.3, color='tab:green')
        axr.set_ylim(*ratio_ylim)
        ax.set_ylim(*alpha_ylim)
        ax.set_xlabel('Overvoltage [V]', loc='right')
        ax.set_ylabel('Alpha Amplitude [V]', loc='top')
        axr.set_ylabel('Ratio', loc='top')
        ax.legend(loc='upper left')
        axr.legend(loc='upper right')
        plt.show()
        return ratio
