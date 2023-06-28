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

def CA_func(x, A, B):
    return  (A * np.exp(B * x) + 1.0) / (1.0 + A) - 1.0

class SPE_data:
    def __init__(self, campaign, invC, invC_err, filtered):
        self.campaign = campaign
        self.invC = invC
        self.invC_err = invC_err
        self.filtered = filtered
        self.analyze_spe()

    def analyze_spe(self):
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
            self.bias_err.append(0.0025*wp.info.bias + 0.015)  # error from keysight
            curr_spe = wp.get_spe()
            self.spe_vals.append(curr_spe[0])
            self.spe_err.append(curr_spe[1])

        self.bias_vals = np.array(self.bias_vals)
        self.bias_err = np.array(self.bias_err)
        self.spe_vals = np.array(self.spe_vals)
        self.spe_err = np.array(self.spe_err)
        self.absolute_spe_vals = self.spe_vals / (self.invC * 1.60217662E-7)
        self.absolute_spe_err = self.absolute_spe_vals * np.sqrt(
                (self.spe_err * self.spe_err) / (self.spe_vals * self.spe_vals) +
                (self.invC_err * self.invC_err) / (self.invC * self.invC))
        spe_wgts = [1.0 / curr_std for curr_std in self.spe_err]
        absolute_spe_wgts = [1.0 / curr_std for curr_std in self.absolute_spe_err]


        model = lm.models.LinearModel()
        params = model.make_params()
        self.spe_res = model.fit(self.spe_vals, params=params, x=self.bias_vals, weights=spe_wgts)
        self.absolute_spe_res = model.fit(self.absolute_spe_vals, params=params, x=self.bias_vals, weights=absolute_spe_wgts) # linear fit
        b_spe = self.spe_res.params['intercept'].value
        m_spe = self.spe_res.params['slope'].value
        self.v_bd = -b_spe / m_spe

        vec_spe = np.array([b_spe / (m_spe * m_spe), -1.0/m_spe])
        # print('check ' + str(self.bias_vals))
        self.v_bd_err = np.sqrt(np.matmul(np.reshape(vec_spe, (1, 2)), np.matmul(self.spe_res.covar, np.reshape(vec_spe, (2, 1))))[0, 0]) # breakdown error calculation using covariance matrix

        self.ov = []
        self.ov_err = []

        for b, db in zip(self.bias_vals, self.bias_err):
            curr_ov = b - self.v_bd
            curr_ov_err = np.sqrt(db * db + self.v_bd_err * self.v_bd_err)
            self.ov.append(curr_ov)
            self.ov_err.append(curr_ov_err)

        self.ov = np.array(self.ov)
        print(self.ov)
        self.ov_err = np.array(self.ov_err)
        for wp, curr_bias, curr_bias_err in zip(self.campaign, self.bias_vals, self.bias_err):
            curr_spe = self.spe_res.eval(params=self.spe_res.params, x=curr_bias)
            curr_spe_err = self.spe_res.eval_uncertainty(x = curr_bias, params = self.spe_res.params, sigma = 1)[0]
            curr_CA_val, curr_CA_err = wp.get_CA_spe(curr_spe, curr_spe_err)
            curr_CA_rms_val, curr_CA_rms_err = wp.get_CA_rms(curr_spe, curr_spe_err)
            self.CA_vals.append(curr_CA_val)
            self.CA_err.append(curr_CA_err)
            self.CA_rms_vals.append(curr_CA_rms_val)
            self.CA_rms_err.append(curr_CA_rms_err)

        self.CA_vals = np.array(self.CA_vals)
        self.CA_err = np.array(self.CA_err)

        self.CA_rms_vals = np.array(self.CA_rms_vals)
        self.CA_rms_err = np.array(self.CA_rms_err)

    #i am stinky

        CA_model = lm.Model(CA_func)
        CA_params = CA_model.make_params(A = 1, B = .1)
        CA_wgts = [1.0 / curr_std for curr_std in self.CA_err]
        self.CA_res = CA_model.fit(self.CA_vals, params=CA_params, x=self.ov, weights=CA_wgts)

    def get_CA_ov(self, input_ov_vals):
        out_vals = self.CA_res.eval(params = self.CA_res.params, x = input_ov_vals)
        out_err = self.CA_res.eval_uncertainty(x = input_ov_vals, sigma = 1)
        return out_vals, out_err

    def get_spe_ov(self, input_ov_vals):
        input_bias_vals = input_ov_vals + self.v_bd
        out_vals = self.spe_res.eval(params = self.spe_res.params, x = input_bias_vals)
        out_err = self.spe_res.eval_uncertainty(x = input_bias_vals, sigma = 1)
        return out_vals, out_err

    def plot_spe(self, in_ov = False, absolute = False, color = 'blue', out_file = None):
        color = 'tab:' + color
        fig = plt.figure()

        start_bias = self.v_bd
        end_bias = np.amax(self.bias_vals) + 1.0
        fit_bias = np.linspace(start_bias, end_bias, 20)

        if absolute:
            fit_y = self.absolute_spe_res.eval(params=self.absolute_spe_res.params, x = fit_bias)
            fit_y_err = self.absolute_spe_res.eval_uncertainty(x = fit_bias, params = self.absolute_spe_res.params, sigma = 1)
            fit_label = 'Absolute Gain Best Fit'
            data_label = 'Absolute Gain values'
            y_label = 'Absolute Gain'
            data_y = self.absolute_spe_vals
            data_y_err = self.absolute_spe_err
            chi_sqr = self.absolute_spe_res.redchi
            slope_text = rf'''Slope: {self.absolute_spe_res.params['slope'].value:0.4} $\pm$ {self.absolute_spe_res.params['slope'].stderr:0.2} [1/V]'''
            intercept_text = rf'''Intercept: {self.absolute_spe_res.params['intercept'].value:0.4} $\pm$ {self.absolute_spe_res.params['intercept'].stderr:0.2} [V]'''
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            fit_y = self.spe_res.eval(params=self.spe_res.params, x = fit_bias)
            fit_y_err = self.spe_res.eval_uncertainty(x = fit_bias, params = self.spe_res.params, sigma = 1)
            fit_label = 'SPE Amplitude Best Fit'
            data_label = 'SPE Amplitude values'
            y_label = 'SPE Amplitude [V]'
            data_y = self.spe_vals
            data_y_err = self.spe_err
            chi_sqr = self.spe_res.redchi
            slope_text = rf'''Slope: {self.spe_res.params['slope'].value:0.4} $\pm$ {self.spe_res.params['slope'].stderr:0.2} [V/V]'''
            intercept_text = rf'''Intercept: {self.spe_res.params['intercept'].value:0.4} $\pm$ {self.spe_res.params['intercept'].stderr:0.2} [V]'''

        parameter_text = slope_text
        if in_ov:
            fit_x = np.linspace(start_bias - self.v_bd, end_bias - self.v_bd, 20)
            data_x = self.ov
            data_x_err = self.ov_err
            x_label = 'Overvoltage [V]'
        else:
            fit_x = fit_bias
            data_x = self.bias_vals
            data_x_err = self.bias_err
            x_label = 'Bias Voltage [V]'
            parameter_text += f'''\n'''
            parameter_text += intercept_text

        parameter_text += f'''\n'''
        parameter_text += rf'''Reduced $\chi^2$: {chi_sqr:0.4}'''
        parameter_text += f'''\n'''
        plt.fill_between(fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color = 'red', alpha = .5)
        plt.plot(fit_x, fit_y, color = 'red', label = fit_label)
        plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', label = data_label)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

        textstr = f'Date: {self.campaign[0].info.date}\n'
        textstr += f'Condition: {self.campaign[0].info.condition}\n'
        textstr += f'RTD4: {self.campaign[0].info.temperature} [K]\n'
        if self.filtered:
            textstr += f'Filtering: Lowpass, 400kHz\n'
        else:
            textstr += f'Filtering: None\n'
        textstr += f'--\n'
        textstr += parameter_text
        textstr += f'--\n'
        textstr += rf'Breakdown Voltage: {self.v_bd:0.4} $\pm$ {self.v_bd_err:0.3} [V]'

        props = dict(boxstyle='round', facecolor=color, alpha=0.4)
        fig.text(0.6, 0.45, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()

        if out_file:
            data = {x_label: data_x, x_label + ' error': data_x_err, y_label: data_y, y_label + ' error': data_y_err}
            df = pd.DataFrame(data)
            df.to_csv(out_file)

    def plot_CA(self, color = 'blue', out_file = None):
        color = 'tab:' + color
        fig = plt.figure()

        data_x = self.ov
        data_x_err = self.ov_err
        data_y = self.CA_vals
        data_y_err = self.CA_err

        fit_x = np.linspace(0.0, np.amax(self.ov) + 1.0, num = 100)
        fit_y = self.CA_res.eval(params=self.CA_res.params, x = fit_x)
        fit_y_err = self.CA_res.eval_uncertainty(x = fit_x, params = self.CA_res.params, sigma = 1)

        plt.fill_between(fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color = 'deeppink', alpha = .5)
        plt.plot(fit_x, fit_y, color = 'deeppink', label = r'$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit')
        plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', label = r'$\frac{1}{N}\sum_{i=1}^{N}{\frac{A_i}{\bar{A}_{1 PE}}-1}$')

        x_label = 'Overvoltage [V]'
        y_label = 'Number of CA [PE]'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc = 'upper left')
        textstr = f'Date: {self.campaign[0].info.date}\n'
        textstr += f'Condition: {self.campaign[0].info.condition}\n'
        textstr += f'RTD4: {self.campaign[0].info.temperature} [K]\n'
        if self.filtered:
            textstr += f'Filtering: Lowpass, 400kHz\n'
        else:
            textstr += f'Filtering: None\n'
        textstr += f'--\n'
        textstr += f'''A: {self.CA_res.params['A'].value:0.3f} $\pm$ {self.CA_res.params['A'].stderr:0.3f}\n'''
        textstr += f'''B: {self.CA_res.params['B'].value:0.2} $\pm$ {self.CA_res.params['B'].stderr:0.2}\n'''
        textstr += rf'''Reduced $\chi^2$: {self.CA_res.redchi:0.4}'''

        props = dict(boxstyle='round', facecolor=color, alpha=0.4)
        fig.text(0.15, 0.65, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        if out_file:
            data = {x_label: data_x, x_label + ' error': data_x_err, y_label: data_y, y_label + ' error': data_y_err}
            df = pd.DataFrame(data)
            df.to_csv(out_file)

    def plot_CA_rms(self, color = 'blue', out_file = None):
        color = 'tab:' + color
        fig = plt.figure()

        data_x = self.ov
        data_x_err = self.ov_err
        data_y = self.CA_rms_vals
        data_y_err = self.CA_rms_err

        plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', label = r'$\sqrt{\frac{\sum_{i=1}^{N}\left(\frac{A_i}{\bar{A}_{1 PE}}-\left(\langle\Lambda\rangle+1\right)\right)^2}{N}}$')

        x_label = 'Overvoltage [V]'
        y_label = 'RMS CAs [PE]'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc = 'upper left')
        textstr = f'Date: {self.campaign[0].info.date}\n'
        textstr += f'Condition: {self.campaign[0].info.condition}\n'
        textstr += f'RTD4: {self.campaign[0].info.temperature} [K]\n'
        if self.filtered:
            # textstr += f'Filtering: Lowpass, 400kHz\n'
            textstr += f'Filtering: Bandpass [1E4, 1E6]\n'
        else:
            textstr += f'Filtering: None\n'

        props = dict(boxstyle='round', facecolor=color, alpha=0.4)
        fig.text(0.15, 0.65, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        if out_file:
            data = {x_label: data_x, x_label + ' error': data_x_err, y_label: data_y, y_label + ' error': data_y_err}
            df = pd.DataFrame(data)
            df.to_csv(out_file)
    # I HAVE A SECRET TO TELL YOU! (it was reed who wrote that message and thwy are pinning it on me)
#    it worked.

class Alpha_data:
    def __init__(self, campaign, invC, invC_err, spe_data, v_bd, v_bd_err):
        self.campaign = campaign
        self.invC = invC
        self.invC_err = invC_err
        self.spe_data = spe_data
        self.v_bd = v_bd
        self.v_bd_err = v_bd_err
        self.analyze_alpha()

    def analyze_alpha(self):
        self.bias_vals = []
        self.bias_err = []
        self.alpha_vals = []
        self.alpha_err = []

        for wp in self.campaign:
            self.bias_vals.append(wp.info.bias)
            self.bias_err.append(0.005)
            curr_alpha = wp.get_alpha()
            self.alpha_vals.append(curr_alpha[0])
            self.alpha_err.append(curr_alpha[1])

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

        self.CA_vals, self.CA_err = self.spe_data.get_CA_ov(self.ov)
        self.spe_vals, self.spe_err = self.spe_data.get_spe_ov(self.ov)
        print('CA Vals: ' + str(self.CA_vals))
        print('SPE Vals: ' + str(self.spe_vals))

        self.num_det_photons = self.alpha_vals * self.spe_data.invC / (self.spe_vals * self.invC * (1.0 + self.CA_vals))
        self.num_det_photons_err = self.num_det_photons * np.sqrt((self.alpha_err * self.alpha_err) / (self.alpha_vals * self.alpha_vals) +
                                                        (self.spe_data.invC_err * self.spe_data.invC_err) / (self.spe_data.invC * self.spe_data.invC) +
                                                        (self.spe_err * self.spe_err) / (self.spe_vals * self.spe_vals) +
                                                        (self.invC_err * self.invC_err) / (self.invC * self.invC) +
                                                        (self.CA_err * self.CA_err) / (self.CA_vals * self.CA_vals))

    def plot_alpha(self, color = 'purple', out_file = None):
        color = 'tab:' + color
        fig = plt.figure()

        data_x = self.ov
        data_x_err = self.ov_err
        data_y = self.alpha_vals
        data_y_err = self.alpha_err

        # fit_x = np.linspace(0.0, np.amax(self.ov) + 1.0, num = 100)
        # fit_y = self.CA_res.eval(params=self.CA_res.params, x = fit_x)
        # fit_y_err = self.CA_res.eval_uncertainty(x = fit_x, params = self.CA_res.params, sigma = 1)

        # plt.fill_between(fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color = 'red', alpha = .5)
        # plt.plot(fit_x, fit_y, color = 'red', label = r'$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit')
        plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', color = color)

        plt.xlabel('Overvoltage [V]')
        plt.ylabel('Alpha Pulse Amplitude [V]')
        textstr = f'Date: {self.campaign[0].info.date}\n'
        textstr += f'Condition: {self.campaign[0].info.condition}\n'
        textstr += f'RTD4: {self.campaign[0].info.temperature} [K]'
        # if self.filtered:
        #     textstr += f'Filtering: Lowpass, 400kHz\n'
        # else:
        #     textstr += f'Filtering: None\n'
        # textstr += f'--\n'
        # textstr += f'''A: {self.CA_res.params['A'].value:0.3f} $\pm$ {self.CA_res.params['A'].stderr:0.3f}\n'''
        # textstr += f'''B: {self.CA_res.params['B'].value:0.2} $\pm$ {self.CA_res.params['B'].stderr:0.2}\n'''
        # textstr += rf'''Reduced $\chi^2$: {self.CA_res.redchi:0.4}'''

        props = dict(boxstyle='round', facecolor=color, alpha=0.4)
        fig.text(0.75, 0.25, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        if out_file:
            data = {x_label: data_x, x_label + ' error': data_x_err, y_label: data_y, y_label + ' error': data_y_err}
            df = pd.DataFrame(data)
            df.to_csv(out_file)

    def plot_num_det_photons(self, color = 'purple', out_file = None):
        color = 'tab:' + color
        fig = plt.figure()

        data_x = self.ov
        data_x_err = self.ov_err
        data_y = self.num_det_photons
        data_y_err = self.num_det_photons_err

        plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', color = color)

        plt.xlabel('Overvoltage [V]')
        plt.ylabel('Number of Detected Photons')
        textstr = f'Date: {self.campaign[0].info.date}\n'
        textstr += f'Condition: {self.campaign[0].info.condition}\n'
        textstr += f'RTD4: {self.campaign[0].info.temperature} [K]'
        # if self.filtered:
        #     textstr += f'Filtering: Lowpass, 400kHz\n'
        # else:
        #     textstr += f'Filtering: None\n'


        props = dict(boxstyle='round', facecolor=color, alpha=0.4)
        fig.text(0.75, 0.25, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.xlim(0, np.amax(self.ov) + 1.0)
        ylow, yhigh = plt.ylim()
        plt.ylim(-1, yhigh * 1.1)
        plt.tight_layout()

    def plot_PDE(self, num_incident, color = 'purple', other_data = None, out_file = None):
        color = 'tab:' + color
        fig = plt.figure()

        data_x = self.ov
        data_x_err = self.ov_err
        data_y = self.num_det_photons / num_incident
        data_y_err = self.num_det_photons_err / num_incident

        plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', color = color, label = 'UMass, 175nm, 190K')

        if other_data:
            for od in other_data:
                plt.errorbar(od.ov, od.pde, xerr = od.ov_err, yerr = od.pde_err, markersize = 10, fmt = '.', label = od.label)

        plt.xlabel('Overvoltage [V]')
        plt.ylabel('Photon Detection Efficiency')
        textstr = f'Date: {self.campaign[0].info.date}\n'
        textstr += f'Condition: {self.campaign[0].info.condition}\n'
        textstr += f'RTD4: {self.campaign[0].info.temperature} [K]'

        props = dict(boxstyle='round', facecolor=color, alpha=0.4)
        fig.text(0.75, 0.25, textstr, fontsize=8,
                verticalalignment='top', bbox=props)
        plt.xlim(0, np.amax(self.ov) + 1.0)
        ylow, yhigh = plt.ylim()
        plt.ylim(-0.01, yhigh * 1.1)
        if other_data:
            plt.legend(loc = 'lower left')
        plt.tight_layout()
        if out_file:
            data = {x_label: data_x, x_label + ' error': data_x_err, y_label: data_y, y_label + ' error': data_y_err}
            df = pd.DataFrame(data)
            df.to_csv(out_file)

class Collab_PDE:
    def __init__(self, filename, groupname, wavelength, temp):
        self.filename = filename
        self.groupname = groupname
        self.wavelength = wavelength
        self.temp = temp
        self.df = pd.read_csv(self.filename)
        self.ov = np.array(self.df['OV'])
        self.ov_err = np.array(self.df['OV error'])
        self.pde = np.array(self.df['PDE']) / 100.
        self.pde_err = np.array(self.df['PDE error']) / 100.

#%%
class multi_campaign: #class to compile multiple campaigns
    def __init__(self, campaigns, invC, invC_err, filtered):
        self.campaigns = campaigns
        self.invC = invC
        self.invC_err = invC_err
        self.filtered = filtered
        self.create_SPEs()

    def create_SPEs(self): #does SPE_data on all the campaigns and returns a list of objects
        self.data = []
        for curr_campaign in self.campaigns:
            self.data.append(SPE_data(curr_campaign, invC_spe_filter, invC_spe_err_filter, filtered = True))


#%%
# ihep_pde = Collab_PDE('C:/Users/lab-341/Desktop/Analysis/fresh_start/PDE_175nm_HD3_iHEP_233K.csv', 'IHEP', 175, 233)
# triumf_pde = Collab_PDE('C:/Users/lab-341/Desktop/Analysis/fresh_start/PDE_176nm_HD3_Triumf_163K.csv', 'TRIUMF', 176, 163)
