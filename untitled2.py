# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:23:12 2024

@author: Hannah
"""
from AnalyzePhotons import CorrelatedAvalancheProbability
from AnalyzePhotons import AlphaData
from AnalyzePhotons import AnalyzePhotons
from AnalyzePhotons import AlphaRatio
from AnalyzePhotons import SPEData
from MeasurementInfo import MeasurementInfo
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('C:/Users/Hannah/Documents/GitHub/SPE-Analysis/nexo_new.mplstyle')
#%%
#%%
CA_Sep20 = CorrelatedAvalancheProbability(
    'C:/Users/Hannah/Documents/Raw Results/Correlated Avalanche Data/Sept20th_405nm_CA_values.csv',
    ov_key = ('Overvoltage [V]', 'Bias Voltage [V] error')
    )