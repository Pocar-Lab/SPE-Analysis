# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 22:11:30 2022

@author: lab-341
"""
from os.path import exists
# holds the meta data for any given measurement
class MeasurementInfo:
    def __init__(self):
        self._selftrig_path = None
        self._solicit_path = None
        self._selftrig_savedir = None
        self._solicit_savedir = None
        self._condition = None
        self._date = None
        self._temperature = None
        self._spe_a = None
        self._spe_b = None
        self._bias = None
        self._spe_guess = None
        self._which_spes = None
        self._baseline_numbins = None
        self._peaks_numbins = None
        self._saved_to_csv = None
        self._peak_search_params = None
        self._min_alpha_value = None
        self._data_type = None
    
    @property
    def selftrig_path(self):
        return self._selftrig_path
    
    @selftrig_path.setter
    def selftrig_path(self, p):
        if not isinstance(p, str):
            raise TypeError('Path must be a string')
        self._selftrig_path = p
    
    @property
    def solicit_path(self):
        return self._solicit_path
    
    @solicit_path.setter
    def solicit_path(self, p):
        if not isinstance(p, str):
            raise TypeError('Path must be a string')
        self._solicit_path = p

    @property
    def selftrig_savedir(self):
        return self._selftrig_savedir
    
    @selftrig_savedir.setter
    def selftrig_savedir(self, d):
        if not isinstance(d, str):
            raise TypeError('Save dir must be a string')
        self._selftrig_savedir = d

    @property
    def solicit_savedir(self):
        return self._solicit_savedir
    
    @solicit_savedir.setter
    def solicit_savedir(self, d):
        if not isinstance(d, str):
            raise TypeError('Save dir must be a string')
        self._solicit_savedir = d

    @property
    def condition(self):
        return self._condition
    
    @condition.setter
    def condition(self, c):
        if not isinstance(c, str):
            raise TypeError('Condition must be a string')
        self._condition = c

    @property
    def date(self):
        return self._date
    
    @date.setter
    def date(self, d):
        if not isinstance(d, str):
            raise TypeError('Date must be a string')
        self._date = d
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, t):
        self._temperature = t

    @property
    def spe_a(self):
        return self._spe_a
    
    @spe_a.setter
    def spe_a(self, a):
        self._spe_a = a

    @property
    def spe_b(self):
        return self._spe_b
    
    @spe_b.setter
    def spe_b(self, b):
        self._spe_b = b

    @property
    def bias(self):
        return self._bias
    
    @bias.setter
    def bias(self, b):
        self._bias = b
    
    @property
    def min_alpha_value(self):
        return self._min_alpha_value
    
    @min_alpha_value.setter
    def min_alpha_value(self, m):
        self._min_alpha_value = m

    @property
    def spe_guess(self):
        if not self._spe_a:
            raise ValueError('Must set spe_a to estimate spe')
        if not self._spe_b:
            raise ValueError('Must set spe_b to estimate spe')
        if not self._bias:
            raise ValueError('Must set bias to estimate spe')
        self._spe_guess = self._spe_estimate(self._bias, self._spe_a, self._spe_b)
        return self._spe_guess
        
    def _spe_estimate(self, bias, a, b):
        return a * bias + b

    @property
    def which_spes(self):
        return self._which_spes
    
    @which_spes.setter
    def which_spes(self, w):
        if not isinstance(w, list):
            raise TypeError('which_spes must be a list')
        else:
            for item in w:
                if not isinstance(item, bool):
                    raise TypeError('which_spes values must be a bool')
        self._which_spes = w
    
    @property
    def baseline_numbins(self):
        return self._baseline_numbins
    
    @baseline_numbins.setter
    def baseline_numbins(self, n):
        if not isinstance(n, int):
            raise TypeError('baseline_numbins must be a int')
        self._baseline_numbins = n

    @property
    def peaks_numbins(self):
        return self._peaks_numbins
    
    @peaks_numbins.setter
    def peaks_numbins(self, n):
        if not isinstance(n, int):
            raise TypeError('peaks_numbins must be a int')
        self._peaks_numbins = n

    @property
    def saved_to_csv(self):
        if not self._selftrig_savedir:
            raise ValueError('Must supply selftrig_savedir')
        if not self._solicit_savedir:
            raise ValueError('Must supply selftrig_savedir')
        selftrig_exists = exists(self._selftrig_savedir)
        solicit_exists = exists(self._solicit_savedir)
        self._saved_to_csv = selftrig_exists and solicit_exists
        return self._saved_to_csv

    @property
    def peak_search_params(self):
        return self._peak_search_params
    
    @peak_search_params.setter
    def peak_search_params(self, p):
        if not isinstance(p, dict):
            raise TypeError('peak_search_params must be a dict')
        self._peak_search_params = p

    @property
    def data_type(self):
        return self._data_type
    
    @data_type.setter
    def data_type(self, d):
        if not isinstance(d, str):
            raise TypeError('Data type must be a string')
        self._data_type = d