# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:35:32 2024

@author: Hannah
"""

import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from scipy import optimize
from scipy import interpolate
import matplotlib as mpl
import pprint
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from MeasurementInfo import MeasurementInfo
#%%

#%%
def get_v_bd(spe, spe_err, bias, bias_err): # gets v_bd of CA data
    model = lm.models.LinearModel()
    params = model.make_params()
    spe_wgts = spe_wgts = [1.0 / curr_std for curr_std in spe_err]
    spe_res = model.fit(spe, params=params, x=bias, weights=spe_wgts)

                        # linear fit
    b_spe = spe_res.params["intercept"].value
    m_spe = spe_res.params["slope"].value
    v_bd = -b_spe / m_spe 
    return v_bd
#%%
def plot_CA(ov, ov_err, CA_vals, CA_err, alpha_ov_vals, color = 'blue'):
    color = 'tab:' + color
    
    data_x = ov
    data_x_err = ov_err
    data_y = CA_vals
    data_y_err = CA_err
    
    CA_model = lm.Model(CA_func)
    CA_params = CA_model.make_params(A = 1, B = .1)
    CA_wgts = [1.0 / curr_std for curr_std in CA_err]
    CA_res = CA_model.fit(CA_vals, params=CA_params, x=ov, weights=CA_wgts)
        
    fit_x = np.linspace(0.0, 10.0, num = 100)
    fit_y = CA_res.eval(params=CA_res.params, x = fit_x)
    fit_y_err = CA_res.eval_uncertainty(x = fit_x, params = CA_res.params, sigma = 1)

    plt.fill_between(fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color = color, alpha = .5)
    plt.plot(fit_x, fit_y, color = color) #, label = r'$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit'
    plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', label = r'$\frac{1}{N}\sum_{i=1}^{N}{\frac{A_i}{\bar{A}_{1 PE}}-1}$')
        
    x_label = 'Overvoltage [V]'
    y_label = 'Number of CA [PE]'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    
    out_vals = CA_res.eval(params=CA_res.params, x=alpha_ov_vals)
    out_err = CA_res.eval_uncertainty(x=alpha_ov_vals, sigma=1)
    return out_vals, out_err

def load_CA(path, 
            v_bd, 
            return_bias = False,
            ov_key = ('Overvoltage [V]', 'Overvoltage [V] error'),
            CA_key = ('Number of CA [PE]', 'Number of CA [PE] error')
            ):
    print('reading in CAs...')
    df = pd.read_csv(path)
    print('reading columns ' + ov_key[0] + ' , ' + ov_key[1])
    ov_vals = np.array(df[ov_key[0]])
    ov_err = np.array(df[ov_key[1]])
    print('reading columns ' + CA_key[0] + ' , ' + CA_key[1])
    CA_vals = np.array(df[CA_key[0]])
    CA_vals_err = np.array(df[CA_key[1]])
    
    bias_vals = np.array([V + v_bd for V in ov_vals])
    if not return_bias:
        return (ov_vals,ov_err), (CA_vals,CA_vals_err)
    else:
        return (bias_vals,ov_err), (CA_vals,CA_vals_err)

def load_data(path, 
            v_bd = 0.0,
            return_ov = False,
            data_type = 'Alpha',
            bias_key = ('Bias Voltage [V]', 'Bias Voltage [V] error'),
            data_key = ('Alpha Pulse Amplitude [V]', 'Alpha Pulse Amplitude [V] error')
            ):
    
    print('reading in data of type '+ data_type)
    df = pd.read_csv(path) 
    print('reading column ' + bias_key[0])
    bias_vals = df[bias_key[0]]
    print('reading column ' + bias_key[1])
    bias_err = df[bias_key[1]]
    print('reading column ' + data_key[0])
    vals = np.array(df[data_key[0]])
    print('reading column ' + data_key[1])
    vals_err = np.array(df[data_key[1]])
    
    if data_type == 'SPE' and data_key == 'Alpha Pulse Amplitude [V]' or None:
        print('Warning: data_type ' + data_type + ' does not match provided key name ' + data_key + '. Changing key name...')
        data_key = ('SPE Amplitude [V]','SPE Amplitude [V] error')
        # bias_key = bias_key = ('Bias voltage [V]', 'Bias voltage [V] error')
    
    if v_bd == 0.0 and return_ov:
        print('Breakdown voltage not provided. Returning bias values.')
    if return_ov and v_bd != 0.0:
            ov_vals = np.array([V - v_bd for V in bias_vals])
            ov_err = np.array(bias_err)
            return (ov_vals,ov_err), (vals,vals_err)
    else:
        return (bias_vals,bias_err), (vals,vals_err)
    
#%%
def CA_func(x, A, B):
    return  (A * np.exp(B * x) + 1.0) / (1.0 + A) - 1.0
class CorrelatedAvalancheProbability:
    def __init__(
        self,
        path: str,
        v_bd: float = 0.0,
        return_bias: bool = False,
        ov_key = ('Overvoltage [V]', 'Overvoltage [V] error'),
        CA_key = ('Number of CA [PE]', 'Number of CA [PE] error')
    ):
        self.path = path
        self.v_bd = v_bd
        self.return_bias = return_bias
        self.ov_key = ov_key
        self.CA_key = CA_key
        
        if self.v_bd == 0.0 and self.return_bias == True:
            print('Breakdown voltage not provided. Setting return_bias to False.')
            self.return_bias == False
            
        self.x, self.y = load_CA(self.path, self.v_bd, ov_key = self.ov_key, CA_key = self.CA_key, return_bias = self.return_bias)
    
    def fit_CA(self, plot = True, with_fit = True, color = 'blue', evaluate = True, eval_x = []):
        if self.return_bias:
            print('Warning: data was loaded with bias values, not OV. CA fit will not function')
        color = 'tab:' + color
        data_x = self.x[0]
        data_x_err = self.x[1]
        data_y = self.y[0]
        data_y_err = self.y[1]
            
        CA_model = lm.Model(CA_func)
        CA_params = CA_model.make_params(A = 1, B = .1)
        CA_wgts = [1.0 / curr_std for curr_std in data_y_err]
        CA_res = CA_model.fit(data_y, params=CA_params, x=data_x, weights=CA_wgts)
        fit_x = np.linspace(0.0, 10.0, num = 100)
        fit_y = CA_res.eval(params=CA_res.params, x = fit_x)
        fit_y_err = CA_res.eval_uncertainty(x = fit_x, params = CA_res.params, sigma = 1)
            
        if plot:
                if with_fit:
                    plt.fill_between(fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color = color, alpha = .5)
                    plt.plot(fit_x, fit_y, color = color, label = r'$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit')
                plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 10, fmt = '.', label = r'$\frac{1}{N}\sum_{i=1}^{N}{\frac{A_i}{\bar{A}_{1 PE}}-1}$')
                x_label = 'Overvoltage [V]'
                y_label = 'Number of CA [PE]'
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.tight_layout()
                plt.legend()
                plt.grid(True)
            
        if evaluate and len(eval_x) > 0:
                out_vals = CA_res.eval(params=CA_res.params, x=eval_x)
                out_err = CA_res.eval_uncertainty(x=eval_x, sigma=1)
                return out_vals, out_err
            
    def return_CA_vals(self, printout = True):
        if printout:
            print('(CA, CA_err):')
        return self.y[0], self.y[1]

    def get_V(self):
        return self.x[0], self.x[1]
    
    def get_CA_eval(self, values = []): #gets CA evaluated at a chosen value
        out_vals, out_err = self.fit_CA(plot = False, with_fit = False, evaluate = True, eval_x = values)
        return out_vals, out_err

#%%
class AlphaData: #class for alpha data loaded from CSV. assumes columns of alpha amplitudes and bias voltages. SPE values must be loaded concurrently for complete functionality
    def __init__(
        self,
        path: str, # path to file containing alpha amplitudes
        invC_alpha: tuple, # conversion factor from electronics calibration and error
        invC_spe: tuple = (None, None),
        path_spe: str = None, # path to file containing SPE amplitudes
        v_bd: float = 0.0, # breakdown voltage, optional
        return_ov: bool = True, # work in terms of OV or not
        #set these if for some reason CSV file has differently named columns
        bias_key: tuple = ('Bias voltage [V]', 'Bias voltage [V] error'), #name of column in CSV file
        alpha_key: tuple = ('Alpha Pulse Amplitude [V]', 'Alpha Pulse Amplitude [V] error'),
        spe_bias_key: tuple = (None,None),
        spe_key: tuple = ('SPE Amplitude [V]','SPE Amplitude [V] error'),
        shaper: str = '1 $\mu$s',
        filtering: str = '400 kHz'
    ):
        self.path = path
        self.path_spe = path_spe
        self.invC, self.invC_err = invC_alpha
        self.invC_spe, self.invC_spe_err = invC_spe
        self.v_bd = v_bd
        self.return_ov = return_ov
        self.bias_key = bias_key
        self.alpha_key = alpha_key
        self.spe_key = spe_key
        self.shaper = shaper
        self.filtering = filtering
        
        print('Using conversion invC = ' + str(self.invC) + ' +/- ' + str(self.invC_err) + ' for alpha data')
        print('Using conversion invC = ' + str(self.invC_spe) + ' +/- ' + str(self.invC_spe_err) + ' for spe data')
        
        if spe_bias_key == (None,None):
            self.spe_bias_key = bias_key
        else:
            self.spe_bias_key = spe_bias_key
        
        if self.path_spe == None:
            print('Warning: No SPE data provided. get_SPEs method will not function. Use kwarg: path_spe = "..."')
    
        
        print('loading SPE data from file: ' + str(self.path_spe))
        self.x_spe, self.y_spe = load_data(self.path_spe,
                                   self.v_bd,
                                   data_type = 'SPE', 
                                   bias_key = self.spe_bias_key, 
                                   data_key = self.spe_key, 
                                   return_ov = False # do not put in terms of OV right now
                                   )
        if self.v_bd == 0.0:
            print('Breakdown voltage not provided. Calculating breakdown voltage from user-provided SPE data...')
            self.v_bd = self.get_v_bd()

        print('loading alpha data from file: ' + str(self.path))
        self.x, self.y = load_data(self.path,
                                   self.v_bd,
                                   data_type = 'Alpha', 
                                   bias_key = self.bias_key, 
                                   data_key = self.alpha_key, 
                                   return_ov = self.return_ov
                                   )
        
        self.spe_for_alpha, self.spe_for_alpha_err = self.get_SPEs(self.x[0], in_ov = self.return_ov) #get the SPE values at specifically the OVs indicated for alpha
        
        self.alpha_in_spe_units = self.y[0]/self.spe_for_alpha * self.invC_spe/self.invC
        self.alpha_in_spe_units_err = self.alpha_in_spe_units * np.sqrt((self.y[1] * self.y[1])/(self.y[0] * self.y[0]) 
                                                                        + (self.spe_for_alpha_err*self.spe_for_alpha_err)/(self.spe_for_alpha*self.spe_for_alpha) 
                                                                        + (self.invC_err*self.invC_err)/(self.invC*self.invC) 
                                                                        + (self.invC_spe_err*self.invC_spe_err)/(self.invC_spe*self.invC_spe) 
                                                                        )   
        
    def get_SPEs(self, input_vals, in_ov = False): # method to get SPE value at a list of given OV or bias
        spe = self.y_spe[0]
        spe_err = self.y_spe[1]
        bias = self.x_spe[0]
        model = lm.models.LinearModel()
        params = model.make_params()
        spe_wgts = spe_wgts = [1.0 / curr_std for curr_std in spe_err]
        spe_res = model.fit(spe, params=params, x=bias, weights=spe_wgts)

        # linear fit:
        b_spe = spe_res.params["intercept"].value
        m_spe = spe_res.params["slope"].value
        v_bd = -b_spe / m_spe 
        
        if in_ov or min(input_vals) < 10:
            input_vals = input_vals + v_bd
            if min(input_vals < 10):
                print('Warning: SPE requested for bias voltage of less than 10V. Did you mean to set in_ov = True? Assuming OV input...')
            
        out_vals = spe_res.eval(params=spe_res.params, x=input_vals)
        out_err = spe_res.eval_uncertainty(x=input_vals, sigma=1)
        return out_vals, out_err
    
    def get_v_bd(self): # method to get the breakdown voltage from provided SPE values
        spe = self.y_spe[0]
        spe_err = self.y_spe[1]
        bias = self.x_spe[0]
        model = lm.models.LinearModel()
        params = model.make_params()
        spe_wgts = spe_wgts = [1.0 / curr_std for curr_std in spe_err]
        spe_res = model.fit(spe, params=params, x=bias, weights=spe_wgts)

        # linear fit
        b_spe = spe_res.params["intercept"].value
        m_spe = spe_res.params["slope"].value
        v_bd = -b_spe / m_spe 
        print('Breakdown voltage for alpha data: ' + str(v_bd))

        return v_bd
        
    def plot_alpha(self, color = 'red', unit = 'V'):
        if unit == 'V':
            y = self.y[0]
            y_err = self.y[1]
            y_label = self.alpha_key[0]
        if unit == 'PE':
            y = self.alpha_in_spe_units
            y_err = self.alpha_in_spe_units_err
            y_label = "Alpha Amplitude [p.e.]"
            
        plt.errorbar(
                    self.x[0],
                    y,
                    xerr=self.x[1],
                    yerr=y_err,
                    markersize=5,
                    fmt=".",
                    color=color,
                )
        plt.legend()
        
        if self.return_ov == True:
            x_label = "Overvoltage [V]"
        else:
            x_label = self.bias_key[0]
            plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
            
    def get_alpha(self):
        return self.y[0],self.y[1]
    def get_V(self):
        return self.x[0],self.x[1]

#%%
class AnalyzePhotons:
    def __init__(
        self,
        info: MeasurementInfo,
        reflector: str,
        alpha: AlphaData,
        CA: CorrelatedAvalancheProbability
    ):
        self.alpha = alpha
        self.CA = CA
        self.info = info
        self.reflector = reflector
        self.v, self.v_err = alpha.get_V()
        self.alpha_vals, self.alpha_err = alpha.get_alpha()
        self.CA_vals, self.CA_err = CA.get_CA_eval(values = self.v)
        self.spe_vals, self.spe_err = alpha.get_SPEs(self.v)
        self.invC_spe, self.invC_spe_err = alpha.invC_spe, alpha.invC_spe_err
        self.invC_alpha, self.invC_alpha_err = alpha.invC, alpha.invC_err
        
        #calculate number of detected photons
        self.num_det_photons = (
               self.alpha_vals * self.invC_spe
               / (self.spe_vals * self.invC_alpha * (1.0 + self.CA_vals))
           ) 
        self.num_det_photons_err = self.num_det_photons * np.sqrt(
               (self.alpha_err * self.alpha_err) / (self.alpha_vals * self.alpha_vals)
               + (self.invC_spe_err * self.invC_spe_err) / (self.invC_spe * self.invC_spe)
               + (self.spe_err * self.spe_err) / (self.spe_vals * self.spe_vals)
               + (self.invC_alpha_err * self.invC_alpha_err) / (self.invC_alpha * self.invC_alpha)
               + (self.CA_err * self.CA_err) / (self.CA_vals * self.CA_vals)
           )     
        
    def plot_num_det_photons(self, color = "purple", label = True, out_file = None):
         """
         Plot the number of detected photons as a function of overvoltage.
         """
         color = "tab:" + color
         if label:
             fig = plt.figure()
         data_x = self.v
         data_x_err = self.v_err
         data_y = self.num_det_photons
         data_y_err = self.num_det_photons_err

         plt.errorbar(
             data_x,
             data_y,
             xerr=data_x_err,
             yerr=data_y_err,
             markersize=7,
             fmt=".",
             color=color,
             label = f"{self.reflector}"
         )

         plt.xlabel("Overvoltage [V]",fontsize=17)
         plt.ylabel("Number of Detected Photons",fontsize = 17)
         textstr = f"Date: {self.info.date}\n"
         textstr = f"Shaper: {self.alpha.shaper}\n"
         textstr += f"RTD4: {self.info.temperature} [K]\n"
         textstr += f'Filtering: {self.alpha.filtering}'

         props = dict(boxstyle="round", facecolor=color, alpha=0.4)
         if label:
             fig.text(0.3, 0.4, textstr, fontsize=14, verticalalignment="top", bbox=props)
         plt.xlim(0, np.amax(self.v) + 1.0)
         ylow, yhigh = plt.ylim()
         plt.ylim(-1, yhigh * 1.1)
         plt.grid(True)
         plt.legend()
         plt.tight_layout()
         
         x_label = "Overvoltage [V]"
         y_label = "Number of Detected Photons"
         if out_file:
                data = {
                    x_label: data_x,
                    x_label + " error": data_x_err,
                    y_label: data_y,
                    y_label + " error": data_y_err,
                }
                df = pd.DataFrame(data)
                df.to_csv(out_file)

