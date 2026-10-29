#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example template for a new SPE analysis

Created on Fri May 9 16:00:00 2025
Update to new analysis on Wed Oct 28 2026

@author: Ed van Bruggen <evanbruggen@umass.edu>
"""


import matplotlib.pyplot as plt
plt.style.use('misc/nexo.mplstyle')
import numpy as np
from uncertainties import ufloat

import SiPM


# NOTE Change
condition = 'GN' # Vac, GN, GXe, LXe
temperature = 170
# Calibration Constant: 1us, 10x gain, filter on - values from Noahs calibration report
invC = ufloat(0.011547, 0.000098)
path ='../data/20250702_SPE_GN/'

# NOTE Modify prominence for your data:
run = SiPM.Analysis(path, invC, 'LXe', 171)

run.plot_multigaussian(
    subtracted=True,
    # span=(1,2),
)

# Fit Gauss to peaks and find the best value for peak location
# NOTE Modify  for your data:
pe_center_guesses = [5.7, 4.8, 8.5, 6.1, 5.6, 7.1, 8.5, 7.7, 6.8]
num_pes = [4, 3, 5, 4, 3, 5, 5, 4, 5]
lower_cutoffs = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
outpath = '../results/20260702/20250509_'

run.fit_multigaussian(
    cutoffs=[ (c, 70) for c in lower_cutoffs ],
    first_guesses=first_pe,
    peak_ranges=[ (1, n) for n in num_pes ],
)

run.plot_multigaussian(
    subtracted=True,
    save_path=outpath+"%b_hist.png"
)

spe_amps = run.plot_spe(
    peak_ranges=[ (1, n) for n in num_pes ],
    save_path=outpath+"%b_spe.png"
)

# Plot linear gain fit to the breakdown voltage
run.plot_breakdown()
