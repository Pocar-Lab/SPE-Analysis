#!/usr/bin/env python3
"""
SPE analysis for data collected on Sep 16 2026 at 171K

Created on Sep 16 2026

@author: Edryd van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
plt.style.use('theme.mplstyle')
import numpy as np
from uncertainties import ufloat

import SiPM


condition = 'LXe'
temperature = 170
#1us, 10x gain, filter on - values from Noahs calibration report
invC = ufloat(0.011547, 0.000098)
path = '../data/20260916_SPE_LXe_170K/'

first_pe     = [4.8, 5.1, 5.9, 6.1, 6.4, 7.2, 7.2, 8.2, 8.5]
num_pes      = [4,   4,   4,   5,   6,   6,   6,   6,   6]
lower_cutoff = [3.8, 3.5, 3.8, 4,   4.1, 4.3, 4.3, 4.5, 4.6, ]
outpath = '../results/pics/20260916/20260924_'

run = SiPM.Analysis(path, invC, 'LXe', 171)

run.plot_multigaussian(
    subtracted=True,
    # span=(1,2),
)

run.fit_multigaussian(
    cutoffs=[ (c, 100) for c in lower_cutoff ],
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

run.plot_breakdown()
