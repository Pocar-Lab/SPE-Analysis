#!/usr/bin/env python3
"""
SPE analysis for data collected on Sep 9 2026 at 171K

Created on Sep 3 2026

@author: Edryd van Bruggen <evanbruggen@umass.edu>
"""

import matplotlib.pyplot as plt
plt.style.use('nexo.mplstyle')
from uncertainties import ufloat

import SiPM

condition = 'LXe'
temperature = 170
#1us, 10x gain, filter on - values from Noahs calibration report
invC_filter = ufloat(0.011547, 0.000098)
path = '../data/20260903_SPE_LXe_170K/'

first_pe     = [4.8, 5.4, 5.9, 6.2, 6.8, 7.1, 7.7, 8.1, 8.7]
num_pes      = [4,   5,   4,   5,   6,   5,   5,   5,   5]
lower_cutoff = [3.8, 3.5, 3.8, 4,   4.1, 4.3, 4.3, 4.5, 4.6, ]
outpath = '../results/pics/20260903/20260925_'

run = SiPM.Analysis(path, invC_filter, 'LXe', 171)

run.plot_multigaussian(
    subtracted=True,
    # span=(-3,-1),
)

run.fit_multigaussian(
    cutoffs=[ (1, 200) for c in lower_cutoff ],
    first_guesses=first_pe,
    peak_ranges=[ (1, n) for n in num_pes ],
)

run.plot_multigaussian(
    subtracted=True,
    # save_path=outpath+f"%b_hist.png"
)

spe_amps = run.plot_spe(
    peak_ranges=[ (1, n) for n in num_pes ],
    # save_path=outpath+f"%b_spe.png"
)

run.plot_breakdown()
