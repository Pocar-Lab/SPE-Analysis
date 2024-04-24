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
from scipy.stats import sem
import statistics
import matplotlib as mpl
import pprint
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional
from MeasurementInfo import MeasurementInfo
#%%
def errorprop(tupled_err1, tupled_err2, combo = "multiplication"):
    val1, error1 = tupled_err1
    val2, error2 = tupled_err2
    if combo == "multiplication":
        percent1 = error1/val1
        percent2 = error2/val2
        prop = np.sqrt(percent1**2 + percent2**2)
    else:
        prop = np.sqrt(error1**2+error2**2)
    return prop
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
    plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 7, fmt = '.', label = r'$\frac{1}{N}\sum_{i=1}^{N}{\frac{A_i}{\bar{A}_{1 PE}}-1}$')
        
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
            
        self.x, self.y = load_CA(self.path, self.v_bd, ov_key = self.ov_key, CA_key = self.CA_key, return_bias = self.return_bias) # x = voltage, y = CA
    
    def fit_CA(self, plot = True, with_fit = True, color = 'blue', evaluate = True, eval_x = [], max_OV = 10, show_label = True):
        if self.return_bias:
            print('Warning: data was loaded with bias values, not OV. CA fit will not function')
        color = color
        
        data = np.array(list(zip(self.x[0],self.x[1],self.y[0],self.y[1]))) 
        data = data[data[:,0] < max_OV] # get only the data with OV < max_OV
        
        data_x, data_x_err = data[:,0], data[:,1]
        data_y, data_y_err = data[:,2], data[:,3]
            
        CA_model = lm.Model(CA_func)
        CA_params = CA_model.make_params(A = 1, B = .1)
        CA_wgts = [1.0 / curr_std for curr_std in data_y_err]
        CA_res = CA_model.fit(data_y, params=CA_params, x=data_x, weights=CA_wgts)
        fit_x = np.linspace(0.0, float(max_OV)+1.0, num = 100)
        fit_y = CA_res.eval(params=CA_res.params, x = fit_x)
        fit_y_err = CA_res.eval_uncertainty(x = fit_x, params = CA_res.params, sigma = 1)
            
        if plot:
                if with_fit:
                    plt.fill_between(fit_x, fit_y + fit_y_err, fit_y - fit_y_err, color = color, alpha = .5)
                    plt.plot(fit_x, fit_y, color = color, label = r'$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit')
                plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, color = color, markersize = 7, fmt = '.', label = r'$\frac{1}{N}\sum_{i=1}^{N}{\frac{A_i}{\bar{A}_{1 PE}}-1}$')
                x_label = 'Overvoltage [V]'
                y_label = 'Number of CA [PE]'
                textstr = f"""A: {CA_res.params['A'].value:0.3} $\pm$ {CA_res.params['A'].stderr:0.3}\n"""
                textstr += f"""B: {CA_res.params['B'].value:0.2} $\pm$ {CA_res.params['B'].stderr:0.2}\n"""
                textstr += rf"""Reduced $\chi^2$: {CA_res.redchi:0.4}"""
                props = dict(boxstyle="round", facecolor=color, alpha=0.4)
                if show_label == True:
                    plt.text(0.17, 1, textstr, fontsize=10, verticalalignment="top", bbox=props)
                plt.xlabel(x_label, fontsize=14)
                plt.ylabel(y_label, fontsize=14)
                plt.tight_layout()
                plt.legend()
                plt.grid(True)
            
        if evaluate and len(eval_x) > 0:
                eval_x = eval_x[eval_x < max_OV]
                out_vals = CA_res.eval(params=CA_res.params, x=eval_x)
                out_err = CA_res.eval_uncertainty(x=eval_x, sigma=1)
                return out_vals, out_err
            
    def return_CA_vals(self, printout = True):
        if printout:
            print('(CA, CA_err):')
        return self.y[0], self.y[1]

    def get_V(self, tupled = True):
        if tupled:
            return list(zip(self.x[0],self.x[1]))
        else:
            return self.x[0], self.x[1]
    
    def get_CA_eval(self, tupled = True, values = [], max_OV = 10.0): #gets CA evaluated at a chosen value
        out_vals, out_err = self.fit_CA(plot = False, with_fit = False, evaluate = True, eval_x = values, max_OV = max_OV)
        if tupled:
            return list(zip(out_vals, out_err))
        return out_vals, out_err

#%%
class SPEData:
    def __init__(
        self,
        path: str, # path to file 
        invC_spe: tuple = (None, None),
        return_ov: bool = True, # work in terms of OV or not
        #set these if for some reason CSV file has differently named columns
        bias_key: tuple = (None,None),
        spe_key: tuple = ('SPE Amplitude [V]','SPE Amplitude [V] error'),
        shaper: str = '1 $\mu$s',
        filtering: str = '400 kHz',
        color: str = "red",
    ):
        self.path = path
        self.invC_spe, self.invC_spe_err = invC_spe
        self.return_ov = return_ov
        self.bias_key = bias_key
        self.spe_key = spe_key
        self.shaper = shaper
        self.filtering = filtering
        self.color = color
        
        print('Using conversion invC = ' + str(self.invC_spe) + ' +/- ' + str(self.invC_spe_err) + ' for spe data')
        
        print('loading SPE data from file: ' + str(self.path))
        self.bias_spe, self.amp_spe = load_data(self.path,
                                   data_type = 'SPE', 
                                   bias_key = self.bias_key, 
                                   data_key = self.spe_key, 
                                   return_ov = False # do not put in terms of OV right now
                                   )
        
        spe = self.amp_spe[0]
        spe_err = self.amp_spe[1]
        bias = self.bias_spe[0]
        bias_err = self.bias_spe[1]
        model = lm.models.LinearModel()
        params = model.make_params()
        spe_wgts = spe_wgts = [1.0 / curr_std for curr_std in spe_err]
        self.spe_res = model.fit(spe, params=params, x=bias, weights=spe_wgts)
        
    def get_v_bd(self, plot = False): # method to get the breakdown voltage from provided SPE values
        spe = self.amp_spe[0]
        spe_err = self.amp_spe[1]
        bias = self.bias_spe[0]
        bias_err = self.bias_spe[1]
        spe_res = self.spe_res

        # linear fit
        b_spe = spe_res.params["intercept"].value
        m_spe = spe_res.params["slope"].value
        v_bd = -b_spe / m_spe 
        vec_spe = np.array([b_spe / (m_spe * m_spe), -1.0 / m_spe])
        v_bd_err = np.sqrt(
                    np.matmul(
                        np.reshape(vec_spe, (1, 2)),
                        np.matmul(spe_res.covar, np.reshape(vec_spe, (2, 1))),
                    )[0, 0]
                ) 
        if plot:
            start_bias = v_bd
            end_bias = np.amax(bias) + 1.0
            fit_bias = np.linspace(start_bias, end_bias, 20)
            fit_y = spe_res.eval(params=spe_res.params, x=fit_bias)
            fit_y_err = spe_res.eval_uncertainty(
                x=fit_bias, params=spe_res.params, sigma=1
            )
            fit_label = "SPE Amplitude Best Fit"
            data_label = "SPE Amplitude values"
            y_label = "SPE Amplitude [V]"
            data_y = spe
            data_y_err = spe_err
            chi_sqr = spe_res.redchi
            slope_text = rf"""Slope: {self.spe_res.params['slope'].value:0.4} $\pm$ {self.spe_res.params['slope'].stderr:0.2} [V/V]"""
            intercept_text = rf"""Intercept: {self.spe_res.params['intercept'].value:0.4} $\pm$ {self.spe_res.params['intercept'].stderr:0.2} [V]"""
            parameter_text = slope_text
            fit_x = fit_bias
            data_x = bias
            data_x_err = bias_err
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
            textstr = self.filtering
            textstr += f"--\n"
            textstr += parameter_text
            textstr += f"--\n"
            textstr += rf"Breakdown Voltage: {v_bd:0.4} $\pm$ {v_bd_err:0.3} [V]"

            props = dict(boxstyle="round", facecolor='blue', alpha=0.4)
            plt.text(0.6, 0.45, textstr, fontsize=8, verticalalignment="top", bbox=props)
            plt.tight_layout()
            plt.grid(True)

        print('Breakdown voltage: ' + str(v_bd) + ' +/- ' + str(v_bd_err))
        return v_bd, v_bd_err
    
    def get_SPEs(self, input_vals, in_ov = False, tupled = True): # method to get SPE value at a list of given OV or bias
        spe_res = self.spe_res
        b_spe = spe_res.params["intercept"].value
        m_spe = spe_res.params["slope"].value
        v_bd = -b_spe / m_spe 
        if in_ov or min(input_vals) < 10:
            input_vals = input_vals + v_bd
            if min(input_vals < 10):
                print('Warning: SPE requested for bias voltage of less than 10V. Did you mean to set in_ov = True?')
            
        out_vals = spe_res.eval(params=spe_res.params, x=input_vals)
        out_err = spe_res.eval_uncertainty(x=input_vals, sigma=1)
        if tupled:
            return list(zip(out_vals, out_err))
        else:
            return out_vals, out_err
#%%
class AlphaData: #class for alpha data loaded from CSV. assumes columns of alpha amplitudes and bias voltages. SPE values must be loaded concurrently for complete functionality
    def __init__(
        self,
        path: str, # path to file containing alpha amplitudes
        spe: SPEData,
        invC_alpha: tuple, # conversion factor from electronics calibration and error
        path_spe: str = None, # path to file containing SPE amplitudes
        v_bd: float = 0.0, # breakdown voltage, optional
        return_ov: bool = True, # work in terms of OV or not
        reflector: str = "None", 
        #set these if for some reason CSV file has differently named columns
        bias_key: tuple = ('Bias voltage [V]', 'Bias voltage [V] error'), #name of column in CSV file
        alpha_key: tuple = ('Alpha Pulse Amplitude [V]', 'Alpha Pulse Amplitude [V] error'),
        shaper: str = '1 $\mu$s',
        color: str = "red",
        mean_subtract: str = None, #optional path to file containing mean of subtracted histograms (if calculating CA from mean, slope. requires SPE data to be loaded)
        use_fit_result_only: bool = False #set if you want to use fit evals only or the actual data points
    ):
        self.path = path
        self.spe = spe
        self.path_spe = path_spe
        self.invC, self.invC_err = invC_alpha
        self.invC_spe, self.invC_spe_err = spe.invC_spe, spe.invC_spe_err
        self.v_bd = v_bd
        self.return_ov = return_ov
        self.bias_key = bias_key
        self.alpha_key = alpha_key
        self.shaper = shaper
        self.reflector = reflector
        self.filtering = self.spe.filtering
        self.mean_subtract = mean_subtract
        self.use_fit_result_only = use_fit_result_only
        self.color = color

        print('Using conversion invC = ' + str(self.invC) + ' +/- ' + str(self.invC_err) + ' for alpha data')

        self.x_spe, self.y_spe = self.spe.bias_spe, self.spe.amp_spe
        
        if self.v_bd == 0.0:
            print('Breakdown voltage not provided. Calculating breakdown voltage from user-provided SPE data...')
            self.v_bd, self.v_bd_err = self.get_v_bd()

        print('loading alpha data from file: ' + str(self.path))
        self.x, self.y = load_data(self.path,
                                   self.v_bd,
                                   data_type = 'Alpha', 
                                   bias_key = self.bias_key, 
                                   data_key = self.alpha_key, 
                                   return_ov = self.return_ov
                                   )
        
        self.spe_for_alpha, self.spe_for_alpha_err = self.get_SPEs(self.x[0], in_ov = self.return_ov, tupled = False) #get the SPE values at specifically the OVs indicated for alpha
        
        self.alpha_in_spe_units = self.y[0]/self.spe_for_alpha * self.invC_spe/self.invC
        self.alpha_in_spe_units_err = self.alpha_in_spe_units * np.sqrt((self.y[1] * self.y[1])/(self.y[0] * self.y[0]) 
                                                                        + (self.spe_for_alpha_err*self.spe_for_alpha_err)/(self.spe_for_alpha*self.spe_for_alpha) 
                                                                        + (self.invC_err*self.invC_err)/(self.invC*self.invC) 
                                                                        + (self.invC_spe_err*self.invC_spe_err)/(self.invC_spe*self.invC_spe) 
                                                                        ) 
        # list of tuples with : V, V err, amp, amp err, spe, spe err
        self.data = list(zip(self.x[0],self.x[1],self.y[0],self.y[1], self.spe_for_alpha, self.spe_for_alpha_err))
        self.data_array = np.array(self.data)
        
        if not self.use_fit_result_only:
            self.alpha_bias = self.x[0]
            if self.x_spe[0][0] > 10 : #this is misleadingly named but bias => OV
                self.spe_bias = self.x_spe[0] - self.v_bd
            else:
                self.spe_bias = self.x_spe[0]
            self.spe_vals_raw = self.y_spe[0]
            self.spe_vals_raw_err = self.y_spe[1]
            self.color_tracker = np.array([False for i in range(len(self.alpha_bias))])
            for v in self.alpha_bias: #look in list alpha_bias (OV) 
                if (min(self.spe_bias) - 0.05) < v < (max(self.spe_bias) + 0.05): #for value v that has a measurement of SPE associated with it (OV)
                    i, = np.where(np.isclose(self.alpha_bias, v)) #get the index of the bias (OV) we are replacing in alpha bias list
                    j, = np.where(np.isclose(self.spe_bias, v)) #get the index of the bias (OV) we are replacing in spe bias list
                    if len(self.spe_vals_raw[j]) > 0 and len(self.spe_for_alpha[i])>0: # accounts for situations where SPEs, Alphas are not spaced equally (30.5, 30, 30.5, 31 v.s. 30,31)
                        print('replacing SPE = '+str(self.spe_for_alpha[i])+' with ' + 'SPE = '+str(self.spe_vals_raw[j]) + ' at OV ' + str(self.alpha_bias[i]) + ' == ' + str(self.spe_bias[j]))
                        self.spe_for_alpha[i] = self.spe_vals_raw[j] #replace value obtained from doing a fit with actual CA value calculated from mean of amplitudes
                        self.spe_for_alpha_err[i] = self.spe_vals_raw_err[j] #do the same for the error (we do not want to use the fit error!)
                        self.color_tracker[i] = True
                    else:
                        print('Alpha pulse measurement at ' + str(v) + '[V] does not have an associated SPE gain measurement. Using extrapolated value from linear fit.')
                        continue
                else:
                    print('Alpha pulse measurement at ' + str(v) + '[V] does not have an associated SPE gain measurement. Using extrapolated value from linear fit.')
            
        self.alpha_in_spe_units_raw = self.y[0]/self.spe_for_alpha * self.invC_spe/self.invC
        self.alpha_in_spe_units_err_raw = self.alpha_in_spe_units * np.sqrt((self.y[1] * self.y[1])/(self.y[0] * self.y[0]) 
                                                                        + (self.spe_for_alpha_err*self.spe_for_alpha_err)/(self.spe_for_alpha*self.spe_for_alpha) 
                                                                        + (self.invC_err*self.invC_err)/(self.invC*self.invC) 
                                                                        + (self.invC_spe_err*self.invC_spe_err)/(self.invC_spe*self.invC_spe) 
                                                                        )   
        
    def get_SPEs(self, input_vals, in_ov = False, tupled = True): # method to get SPE value at a list of given OV or bias
        spe = self.y_spe[0]
        spe_err = self.y_spe[1]
        bias = self.x_spe[0]
        model = lm.models.LinearModel()
        params = model.make_params()
        spe_wgts = spe_wgts = [1.0 / curr_std for curr_std in spe_err]
        spe_res = model.fit(spe, params=params, x=bias, weights=spe_wgts)
        spe_res = self.spe.spe_res
        # linear fit:
        b_spe = spe_res.params["intercept"].value
        m_spe = spe_res.params["slope"].value
        v_bd = -b_spe / m_spe 
        
        if in_ov or min(input_vals) < 10:
            input_vals = input_vals + v_bd
            if min(input_vals < 10):
                print('Warning: SPE requested for bias voltage of less than 10V. Did you mean to set in_ov = True?')
            
        out_vals = spe_res.eval(params=spe_res.params, x=input_vals)
        out_err = spe_res.eval_uncertainty(x=input_vals, sigma=1)
        if tupled:
            return list(zip(out_vals, out_err))
        else:
            return out_vals, out_err
    
    def get_v_bd(self): # method to get the breakdown voltage from provided SPE values
        return self.spe.get_v_bd(plot = False)
        
    def plot_alpha(self, color = None, unit = 'V'):
        if color == None:
            color = self.color
        if unit == 'V':
            y = self.y[0]
            y_err = self.y[1]
            y_label = self.alpha_key[0]
        if unit == 'PE':
            y = self.alpha_in_spe_units
            y_err = self.alpha_in_spe_units_err
            y_label = "Alpha Amplitude [p.e.] (from SPE gain fit)" 
            if not self.use_fit_result_only:
                y_raw = [b for a, b in zip(self.color_tracker, self.alpha_in_spe_units_raw) if a]
                y_raw_err = [b for a, b in zip(self.color_tracker,self.alpha_in_spe_units_err_raw) if a]
                y_raw_label = "Alpha Amplitude [p.e.] (from data)"
            
        plt.errorbar(
                    self.x[0],
                    y,
                    xerr=self.x[1],
                    yerr=y_err,
                    markersize=7,
                    fmt=".",
                    capsize = 0,
                    color=color,
                    label = str(self.reflector) + ' / ' + str(y_label)
                )
        plt.legend()
        
        if not self.use_fit_result_only and unit == 'PE':
            plt.errorbar(
                        [b for a, b in zip(self.color_tracker, self.x[0]) if a],
                        y_raw,
                        xerr=[b for a, b in zip(self.color_tracker, self.x[1]) if a],
                        yerr=y_raw_err,
                        markersize=7,
                        capsize = 0,
                        fmt=".",
                        color='green',
                        label = str(self.reflector) + ' / ' + y_raw_label
                    )
            plt.legend()
        
        if self.return_ov == True:
            x_label = "Overvoltage [V]"
        else:
            x_label = self.bias_key[0]
            plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
            
    def get_alpha(self, tupled  = True, out_bias = False):
        if out_bias:
            V = self.x[0] + self.v_bd
            return V, self.y[0],self.y[1]
        if tupled:
            zipped  = zip(self.y[0],self.y[1])
            return list(zipped)            
        else:
            return self.y[0],self.y[1]
    
    def get_V(self, ov = True):
        v, amp = load_data(self.path,
                         self.v_bd,
                         data_type = 'Alpha', 
                         bias_key = self.bias_key, 
                         data_key = self.alpha_key, 
                         return_ov = ov
                         )
        zipped  = zip(v[0],v[1])
        return list(zipped)
        # return (v[0], v[1])

    
    def get_CA_from_means(self, #get the CA from user-provided CSV of mean subtracted histogram values
                return_bias = True,
                key = ("mean", "mean error"),
                out_file: str = None
                ):
        print('reading in mean of subtracted histogram...')
        df = pd.read_csv(self.mean_subtract)
        print('reading columns ' + key[0] + ' , ' + key[1])
        vals = np.array(df[key[0]])
        err = np.array(df[key[1]])
        print('reading columns ' + self.spe_bias_key[0] + ' , ' + self.spe_bias_key[1])
        bias_vals = np.array(df[self.spe_bias_key[0]])
        bias_err = np.array(df[self.spe_bias_key[1]])
        
        ov_vals = np.array([V - self.v_bd for V in bias_vals])
        print('calculating SPE at bias voltages ' + str(bias_vals))
        spe, spe_err = self.get_SPEs(bias_vals, in_ov=False, tupled = False)
        print('SPE values: ' + str(spe))
        print('mean values: ' + str(vals))
        CA_vals = vals / spe - 1
        CA_err = CA_vals * np.sqrt((err/vals)**2 + (spe_err/spe)**2)
        
        if out_file != None: #easiest way to deal with this is to save it to its own CSV file and treat it like any other CA data set
            data = {
                    "Number of CA [PE]": CA_vals,
                    "Number of CA [PE] error": CA_err,
                    "Overvoltage [V]": ov_vals,
                    "Overvoltage [V] error": bias_err # just use error on keysight
                    }
            df = pd.DataFrame(data)
            df.to_csv(out_file)
        
        if not return_bias:
            return (ov_vals,bias_err), (CA_vals,CA_err)
        else:
            return (bias_vals,bias_err), (CA_vals,CA_err)
    

#%%
class AlphaRatio:
    def __init__(
            self,
            alpha_1: AlphaData,
            alpha_2: AlphaData,
            ):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.color1, self.color2 = alpha_1.color, alpha_2.color
        
        self.list_of_volt_tuples_1, self.list_of_volt_tuples_2 = sorted(alpha_1.get_V(ov = True)), sorted(alpha_2.get_V(ov = True))
        self.list_of_amp_tuples_1, self.list_of_amp_tuples_2 = sorted(alpha_1.get_alpha()), sorted(alpha_2.get_alpha())
        # all the sorted tuples help ensure OVs don't get out of order
        self.v_1, self.v_2 = np.array(self.list_of_volt_tuples_1)[:,0],  np.array(self.list_of_volt_tuples_2)[:,0]
        self.v_1_err, self.v_2_err = np.array(self.list_of_volt_tuples_1)[:,1], np.array(self.list_of_volt_tuples_2)[:,1]
        self.alpha_vals_1, self.alpha_vals_2 = np.array(self.list_of_amp_tuples_1)[:,0], np.array(self.list_of_amp_tuples_2)[:,0]
        self.alpha_err_1, self.alpha_err_2 = np.array(self.list_of_amp_tuples_1)[:,1], np.array(self.list_of_amp_tuples_2)[:,1]
        
        #inherit breakdown voltage and bias values
        self.v_bd_1, self.v_bd_2 =  alpha_1.v_bd, alpha_2.v_bd
        self.v_bd_1_err, self.v_bd_2_err =  alpha_1.v_bd_err, alpha_2.v_bd_err
        self.alpha_1_bias, self.alpha_2_bias = self.v_1 + alpha_1.v_bd, self.v_2 + alpha_2.v_bd
        
        self.average_v_bd = (self.v_bd_2+self.v_bd_1)/2
        
        self.average_v_bd_err = np.sqrt((self.v_bd_1_err)**2 + (self.v_bd_2_err)**2)
        print('AVERAGE BREAKDOWN VOLTAGE = ' +str(self.average_v_bd)+'+/-'+str(self.average_v_bd_err))
        
        
        #all this allows for the alpha data sets to be of different lengths:
        self.longer_data = max([self.alpha_1_bias,self.alpha_2_bias] , key = len)
        self.longer_data_err = max([self.v_1_err,self.v_2_err] , key = len)
        
        self.ov_from_avg = self.longer_data - self.average_v_bd
        
        self.shorter_data = min([self.alpha_1_bias,self.alpha_2_bias] , key = len)
        self.shorter_data_err = min([self.v_1_err,self.v_2_err] , key = len)
        self.longer_alpha_vals = max([self.alpha_vals_1,self.alpha_vals_2] , key = len)
        self.longer_alpha_vals_err = max([self.alpha_err_1,self.alpha_err_2] , key = len)
        self.shorter_alpha_vals = min([self.alpha_vals_1,self.alpha_vals_2] , key = len)
        self.shorter_alpha_vals_err = min([self.alpha_err_1,self.alpha_err_2] , key = len)
        self.alpha_ratio = []
        self.alpha_ratio_err = []
        self.bias_for_ratio = []
        self.bias_for_ratio_err = []
        for i in range(len(self.longer_data)):
            if self.longer_data[i] in self.shorter_data:
                print(str(self.longer_data[i]) + ' is common to both data sets; computing ratio')
                j, = np.where(np.isclose(self.shorter_data, self.longer_data[i]))
                print(str(self.longer_data[i]) +'=='+str(self.shorter_data[j[0]]))
                # alpha_ratio = self.alpha_vals_1[i]/self.alpha_vals_2[j]
                # amp_long, amp_short = self.longer_data[i], self.shorter_data[j]
                if len(self.longer_data) != len(self.shorter_data):
                    alpha_ratio = max(self.longer_alpha_vals[i], self.shorter_alpha_vals[j])/min(self.longer_alpha_vals[i], self.shorter_alpha_vals[j])
                else: 
                    alpha_ratio = self.alpha_vals_1[i]/self.alpha_vals_2[j]
                print(alpha_ratio)
                self.alpha_ratio.append(alpha_ratio[0]) # alpha_ratio is a single-element np array by default
                err = np.sqrt((self.longer_alpha_vals_err[i]/self.longer_alpha_vals[i])**2+(self.shorter_alpha_vals_err[j]/self.shorter_alpha_vals[j])**2)
                self.alpha_ratio_err.append(err[0]) # err is a single element np array by default
                print('alpha ratios: ' + str(self.alpha_ratio))
                self.bias_for_ratio.append(self.longer_data[i])
                self.bias_for_ratio_err.append(self.longer_data_err[i])
            else:
                continue
        
        #average ratio
        self.average_alpha_ratio = np.mean(self.alpha_ratio)
        self.average_alpha_ratio_SEM = stats.sem(self.alpha_ratio)
        self.average_alpha_ratio_std = statistics.stdev(self.alpha_ratio)
        
        #save ratio of SPE values. if the ratio is systematically > or < 1 num det photons could be misleading
        self.spes_1, self.spes_1_err = self.alpha_1.get_SPEs(np.array(self.bias_for_ratio), in_ov = False, tupled = False)
        self.spes_2, self.spes_2_err= self.alpha_2.get_SPEs(np.array(self.bias_for_ratio), in_ov = False, tupled = False)
        self.spe_ratio = self.spes_1/self.spes_2
        self.spe_ratio_err = errorprop((self.spes_1, self.spes_1_err),(self.spes_2, self.spes_2_err))
        
    def plot_spe_ratio(self, color = 'red'):
        plt.errorbar(self.bias_for_ratio, self.spe_ratio, xerr = self.bias_for_ratio_err, yerr = self.spe_ratio_err, 
                     markersize = 3, fmt = 's', color = color, capsize=0
                     )        

    def get_average(self, return_std = False):
        if return_std: 
            a = self.average_alpha_ratio_std
        else:
            a = self.average_alpha_ratio_SEM
        return self.average_alpha_ratio, a
    
    def plot_alpha(self, unit = 'V'):
        color1 = self.color1
        color2 = self.color2
        self.alpha_1.plot_alpha(color = color1, unit = unit)
        self.alpha_2.plot_alpha(color = color2, unit = unit)
        
    def plot_alpha_ratio(self, color = 'purple', ov_from_avg = False): #ov_from_avg: display in OV with respect to the average of the two breakdown voltages
        config1, config2 = self.alpha_1.reflector, self.alpha_2.reflector
        data_y = self.alpha_ratio
        data_y_err = self.alpha_ratio_err
        
        if ov_from_avg:
            data_x = np.array(self.bias_for_ratio) - self.average_v_bd
            data_x_err = np.sqrt(np.array(self.average_v_bd_err)**2 +np.array(self.bias_for_ratio_err)**2)
        else:
            data_x = np.array(self.bias_for_ratio)
            data_x_err = np.array(self.bias_for_ratio_err)
        plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 3, fmt = 's', color = color, capsize=0,
                     # label = 'Ratio = '+str(self.average_alpha_ratio)+ r' $\pm$ ' + str(self.average_alpha_ratio_SEM))  
                     label = 'Ratio = '+f'{self.get_average()[0]:0.3}'+ r' $\pm$ ' + f'{self.get_average()[1]:0.2}'
                     + '\n' + 'Standard Deviation: ' + f'{self.average_alpha_ratio_std:0.3}'
                     )  
        plt.legend()
        if ov_from_avg:
            x_label = 'Overvoltage [V] w.r.t. average breakdown voltage'
        else:
            x_label = 'Bias Voltage [V]'
        y_label = 'Ratio ' + config1 + '/' +config2
        plt.axhline(y = self.average_alpha_ratio, color = color)
        plt.fill_between(
            data_x, self.average_alpha_ratio + self.average_alpha_ratio_std, self.average_alpha_ratio - self.average_alpha_ratio_std, 
            color=color, alpha=0.25
        )
        plt.xlabel(x_label,fontsize = 14)
        plt.ylabel(y_label, fontsize=16)
        plt.tight_layout()
        
#%%
class AnalyzePhotons:
    def __init__(
        self,
        info: MeasurementInfo,
        reflector: str,
        alpha: AlphaData,
        CA: CorrelatedAvalancheProbability,
        ratio: AlphaRatio = None,
        use_fit_results_only: bool = False,
        max_OV: float = 10.0,
        use_average_v_bd: bool = False
    ):
        self.alpha = alpha
        self.alpha_data_list = alpha.data
        self.CA = CA
        self.info = info
        self.reflector = reflector
        self.max_OV = max_OV
        self.use_average_v_bd = use_average_v_bd
        
        self.ratio = ratio
        self.ov_from_avg = ratio.ov_from_avg
        
        self.alpha_data_all = np.array(sorted(self.alpha_data_list))
        self.alpha_data = self.alpha_data_all[self.alpha_data_all[:,0] < self.max_OV]
        
        self.use_fit_results_only = use_fit_results_only
        self.v = np.array(sorted(alpha.get_V(ov = True)))[:,0]
        self.v_err = np.array(sorted(alpha.get_V(ov = True)))[:,1]
        self.alpha_vals = self.alpha_data[:,2]
        self.alpha_err = self.alpha_data[:,3]
        
        self.spe_vals, self.spe_err = self.alpha_data[:,4], self.alpha_data[:,5]
        self.invC_spe, self.invC_spe_err = alpha.invC_spe, alpha.invC_spe_err
        self.invC_alpha, self.invC_alpha_err = alpha.invC, alpha.invC_err
        
        self.alpha_bias, self.alpha_bias_err = np.array(sorted(self.alpha_data[:,0])), np.array(sorted(self.alpha_data[:,1]))
        
        self.CA_vals, self.CA_err = np.array(sorted(CA.get_CA_eval(values = self.alpha_bias, max_OV = self.max_OV)))[:,0], np.array(sorted(CA.get_CA_eval(values = self.alpha_bias,  max_OV = self.max_OV)))[:,1]
        
        self.CA_measured = np.array(sorted(zip(self.CA.x[0],self.CA.x[0],self.CA.y[0],self.CA.y[1]))) #sort by voltage
        self.CA_measured = self.CA_measured[self.CA_measured[:,0] < self.max_OV]
        self.CA_raw, self.CA_raw_err = self.CA_measured[:,2], self.CA_measured[:,3]
        self.CA_raw_bias, self.CA_raw_bias_err = self.CA_measured[:,0], self.CA_measured[:,1]
        
        if not use_fit_results_only:
            self.color_tracker = np.array([False for i in range(len(self.CA_vals))])
            for v in self.alpha_bias: #look in list alpha_bias (OV) 
                if min(self.CA_raw_bias)-0.05 < v < max(self.CA_raw_bias)+0.05: #for value v that has a measurement of CA (OV)
                    i, = np.where(np.isclose(self.alpha_bias, v)) #get the index of the bias (OV) we are replacing
                    j, = np.where(np.isclose(self.CA_raw_bias, v)) #get the index of the bias (OV) we are replacing
                    if len(self.CA_raw[j]) > 0:
                        print('replacing CA = '+str(self.CA_vals[i])+' with ' + str(self.CA_raw[j]) + ' at OV ' + str(self.v[i]) + ' == ' + str(self.CA_raw_bias[j]))
                        self.CA_vals[i] = self.CA_raw[j] #replace value obtained from doing a fit with actual CA value calculated from mean of amplitudes
                        self.CA_err[i] = self.CA_raw_err[j] #do the same for the error (we do not want to use the fit error!)
                        self.color_tracker[i] = True
                else:
                    print('OV value ' + str(v) + ' does not have an associated CA measurement. Using value from fit.')
    
        if use_average_v_bd and ratio != None and use_fit_results_only == False:
            self.CA_vals_from_avg, self.CA_err_from_avg = np.array(sorted(CA.get_CA_eval(values = self.ov_from_avg, max_OV = self.max_OV)))[:,0], np.array(sorted(CA.get_CA_eval(values = self.ov_from_avg,  max_OV = self.max_OV)))[:,1]
            self.spe_from_avg, self.spe_from_avg = alpha.get_SPEs(self.ov_from_avg, in_ov = True, tupled = False)    
            self.spe_vals, self.spe_err = self.spe_from_avg, self.spe_from_avg
            self.CA_vals, self.CA_err = self.CA_vals_from_avg, self.CA_err_from_avg
        
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
        
    def plot_num_det_photons(self, color = None, label = True, out_file = None):
         """
         Plot the number of detected photons as a function of overvoltage.
         """
         if color == None:
             color = self.alpha.color
         color = "tab:" + color
         if label:
             fig = plt.figure()
             
         data_y = self.num_det_photons
         data_y_err = self.num_det_photons_err
         if self.use_average_v_bd:
             data_x = self.ov_from_avg[:len(data_y)]
         else:
             data_x = self.alpha_bias
         data_x_err = self.alpha_bias_err

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

    def plot_CA(self, plot = True, color = 'blue'): #plots the CA values *specifically used to find num det photons*
        color = color
        data_x = self.alpha_bias
        if not self.use_fit_results_only:
            data_x_with_raw_CA = [b for a, b in zip(self.color_tracker, self.alpha_bias) if a]
            data_x_err_raw_CA = [b for a, b in zip(self.color_tracker, self.alpha_bias_err) if a]
            data_y_with_raw_CA = [b for a, b in zip(self.color_tracker, self.CA_vals) if a]
            data_y_err_raw_CA = [b for a, b in zip(self.color_tracker, self.CA_err) if a]
        data_x_err = self.alpha_bias_err
        data_y = self.CA_vals
        data_y_err = self.CA_err
        
        
        plt.errorbar(data_x, data_y, xerr = data_x_err, yerr = data_y_err, markersize = 7, color ='xkcd:'+color, fmt = '.', label = r'$\frac{Ae^{B*V_{OV}}+1}{A + 1} - 1$ fit')
        if not self.use_fit_results_only:
            plt.errorbar(data_x_with_raw_CA, data_y_with_raw_CA, xerr = data_x_err_raw_CA, color = color, yerr = data_y_err_raw_CA, markersize = 7, fmt = '.', label = r'$\frac{1}{N}\sum_{i=1}^{N}{\frac{A_i}{\bar{A}_{1 PE}}-1}$'+ ' from LED data')
        x_label = 'Overvoltage [V]'
        y_label = 'Number of CA [PE]'
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.legend()
        plt.grid(True)