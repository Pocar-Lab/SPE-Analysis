import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit
import numpy as np

plt.style.use('misc/nexo.mplstyle')


### Breakdon Voltage vs Temperature

# 2025 Campaign
temp     = [128,   137.2, 144.7, 186.0, 264.3, 234.71, 217.55, 252.3, 200.5, 171.3, 168]
temp_err = [0.0,   .2,    .3,    0.3,   0.146, 0.17,   0.16,   0.18,  0.20,  0.136, .3]
vbd      = [26.28, 26.36, 26.19, 27.07, 29.88, 28.44,  27.77,  29.15, 27.32, 26.97, 26.23]
vbd_err  = [0.061, 0.15,  0.195, 0.23,  0.093, .102,   0.10,   0.083, 0.208, 0.15,  0.14]
skip_last_num = 2
# 171: 27.10
# 168: 26.69 +/- 0.099
# 217: 27.96 +/- .157
# 235: 28.46 +/- .168

# 2022 Campaign
temp2022     = [242.42, 234,   221,  190.5, 181,  153,   158,  ]
temp2022_err = [0.08,   0.05,  0.05, 0.05,  0.05, 0.05,  0.05, ]
vbd2022      = [29.2,   28.39, 28.1, 28.27, 27.66,26.92, 26.9, ]
vbd2022_err  = [0.1,    0.07,  0.1,  0.07,  0.05, 0.08,  0.2,  ]
# 2023 Campaign
temp2023     = [171,   135.22, 140.59, 217,]
temp2023_err = [0.05,  0.05,   0.08,   0.05,]
vbd2023      = [27.41, 26.88,  26.6,   28.17,]
vbd2023_err  = [0.08,  0.07,   0.1,    0.09,]
# 2022 Vac
temp2022vac     = [190.39, ]
temp2022vac_err = [.04,    ]
vbd2022vac      = [27.93, ]
vbd2022vac_err  = [.004,  ]

# Above 170K 2022 and 2023
# temp2022     = [190.39, 171,   217,   242.42, 234,   221,  190.5, 181,   153,   158,  ]
# temp2022_err = [.04,    0.05,  0.05,  0.08,   0.05,  0.05, 0.05,  0.05,  0.05,  0.05, ]
# vbd2022      = [27.93,  27.41, 28.17, 29.2,   28.39, 28.1, 28.27, 27.66, 26.92, 26.9, ]
# vbd2022_err  = [.004,   0.08,  0.09,  0.1,    0.07,  0.1,  0.07,  0.05,  0.08,  0.2,  ]

def linear_func(x, m, b):
    return m*x+b

def temp_fit(popt, pcov, min_temp=160, max_temp=300):
    # Extract fitted parameters and their uncertainties
    slope = popt[0]
    intercept = popt[1]
    slope_error = np.sqrt(pcov[0, 0])
    intercept_error = np.sqrt(pcov[1, 1])
    # Calculate fitted line
    # temp_fit = np.linspace(min(temp) - 5, max(temp) + 5, 100)
    temp_fit = np.linspace(min_temp, max_temp, 100)
    vbd_fit = linear_func(temp_fit, slope, intercept)
    vbd_uncertainty = np.sqrt(intercept_error**2 + temp_fit**2 * slope_error**2 +
                              2*temp_fit*pcov[0,1])
    vbd_upper = vbd_fit + 2*vbd_uncertainty
    vbd_lower = vbd_fit - 2*vbd_uncertainty
    p = plt.plot(temp_fit, vbd_fit, linewidth=2)
    plt.fill_between(temp_fit, vbd_lower, vbd_upper, alpha=0.2, color=p[-1].get_color())
                    # label='Fit uncertainty (1σ)')
    label = f'{slope*1e3:.1f} ± {slope_error*1e3:.2f} mV/K'
    # label = f'$V_{{bd}}$ = ({slope:.4f} ± {slope_error:.4f})·T + ({intercept:.3f} ± {intercept_error:.3f})\n'
    return p[-1].get_color(), label

c = temp_fit(*curve_fit(linear_func, temp2022, vbd2022, sigma=vbd2022_err, absolute_sigma=True),
             min_temp=130)
plt.errorbar(
    temp2022,
    vbd2022,
    xerr=temp2022_err,
    yerr=vbd2022_err,
    markersize=6,
    fmt=".",
    color=c,
    label='2022 GN Campaign - SPE',
)
# plt.errorbar(
#     temp2023,
#     vbd2023,
#     xerr=temp2023_err,
#     yerr=vbd2023_err,
#     markersize=6,
#     fmt=".",
#     label='2023 GN Campaign - SPE',
# )

plt.errorbar(
    170,
    27.2,
    xerr=.3,
    yerr=.16,
    markersize=6,
    fmt=".",
    label='Average LXe - SPE',
)

plt.xlabel('Temperature [K]')
plt.ylabel('Breakdown Voltage [V]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


### IV Breakdown

## LXe 2024-25
# a = [
# (167.254,0.139,27.73,0.015),
# (169.12,0.149,27.87,0.1),
# (169.448,0.144,27.7,0.067),
# (169.486,0.142,27.74,0.15),
# (169.521,0.144,28.19,0.19),
# (169.548,0.143,28.03,0.23),
# (169.755,0.157,27.45,0.27),
# (169.811,0.123,27.79,0.016),
# (170.326,0.166,27.73,0.036),
# ]
# list(zip(*a))
tempIVLXe     = [167.254, 169.12, 169.448, 169.486, 169.521, 169.548, 169.755, 169.811, 170.326]
tempIVLXe_err = [0.139,   0.149,  0.144,   0.142,   0.144,   0.143,   0.157,   0.123,   0.166]
vbdIVLXe      = [27.73,   27.87,  27.7,    27.74,   28.19,   28.03,   27.45,   27.79,   27.73]
vbdIVLXe_err  = [0.015,   0.1,    0.067,   0.15,    0.19,    0.23,    0.27,    0.016,   0.036]

## GN
tempIVGN_all     = [167.776, 167.787, 167.804, 167.839, 168.501, 176.9, 182.83, 182.861, 182.966, 183.198, 183.296, 183.654, 184.824, 186.64, 194.3, 207.6,  217.12, 217.452, 217.456, 217.457, 235.482, 235.533, 243,   252.511, 286,   286,   286,   288.649, 289.805],
tempIVGN_all_err = [0.13,    0.129,   0.122,   0.133,   0.131,   0.2,   0.149,  0.154,   0.15,    0.168,   0.186,   0.206,   0.506,   0.02,   0.2,   0.2,    0.034,  0.153,   0.152,   0.152,   0.14,    0.147,   0.2,   0.159,   0.133, 0.156, 0.143, 0.14,    0.119],
vbdIVGN_all      = [27.7,    27.69,   27.71,   27.67,   27.77,   27.96, 28.27,  28.39,   28.11,   28.28,   28.11,   28.14,   28.17,   28.2,   28.45, 28.79,  29.11,  29.09,   29.11,   29.16,   29.6,    29.66,   29.88, 30.2,    31.18, 31.41, 31.06, 31.39,   31.44],
vbdIVGN_all_err  = [0.021,   0.049,   0.014,   0.01,    0.085,   0.011, 0.14,   0.089,   0.09,    0.062,   0.0093,  0.046,   0.25,    0.046,  0.008, 0.0084, 0.012,  0.042,   0.001,   0.034,   0.051,   0.009,   0.012, 0.0073,  0.093, 0.12,  0.21,  0.0098,  0.014]

# 2025
a = [
(167.776,0.13,27.7,0.021),
(167.787,0.129,27.69,0.049),
(167.804,0.122,27.71,0.014),
(167.839,0.133,27.67,0.01),
(168.501,0.131,27.77,0.085),
(182.83,0.149,28.27,0.14),
(182.861,0.154,28.39,0.089),
(182.966,0.15,28.11,0.09),
(183.198,0.168,28.28,0.062),
(183.296,0.186,28.11,0.0093),
(183.654,0.206,28.14,0.046),
(184.824,0.506,28.17,0.25),
(217.452,0.153,29.09,0.042),
(217.456,0.152,29.11,0.001),
(217.457,0.152,29.16,0.034),
(235.482,0.14,29.6,0.051),
(235.533,0.147,29.66,0.009),
(252.511,0.159,30.2,0.0073),
(286,0.133,31.18,0.093),
(286,0.156,31.41,0.12),
(286,0.143,31.06,0.21),
(288.649,0.14,31.39,0.0098),
]
list(zip(*a))

# tempIVGN2025     = [167.776, 167.787, 167.804, 167.839, 168.501, 182.83, 182.861, 182.966, 183.198, 183.296, 183.654, 184.824, 217.452, 217.456, 217.457, 235.482, 235.533, 252.511, 286,   286,   286,   288.649]
# tempIVGN2025_err = [0.13,    0.129,   0.122,   0.133,   0.131,   0.149,  0.154,   0.15,    0.168,   0.186,   0.206,   0.506,   0.153,   0.152,   0.152,   0.14,    0.147,   0.159,   0.133, 0.156, 0.143, 0.14]
# vbdIVGN2025      = [27.7,    27.69,   27.71,   27.67,   27.77,   28.27,  28.39,   28.11,   28.28,   28.11,   28.14,   28.17,   29.09,   29.11,   29.16,   29.6,    29.66,   30.2,    31.18, 31.41, 31.06, 31.39]
# vbdIVGN2025_err  = [0.021,   0.049,   0.014,   0.01,    0.085,   0.14,   0.089,   0.09,    0.062,   0.0093,  0.046,   0.25,    0.042,   0.001,   0.034,   0.051,   0.009,   0.0073,  0.093, 0.12,  0.21,  0.0098]

tempIVGN2025     = [167.839, 183.296, 217.456, 235.533, 252.511, 286,   ]
tempIVGN2025_err = [0.133,   0.186,   0.152,   0.147,   0.159,   0.133, ]
vbdIVGN2025      = [27.67,   28.11,   29.11,   29.66,   30.2,    31.18, ]
vbdIVGN2025_err  = [0.01,    0.0093,  0.001,   0.009,   0.0073,  0.093, ]

# 2024
# a =[
# (289.805,0.119,31.44,0.014),
# (186.64,0.02,28.2,0.046),]
tempIVGN2024 = [289.805, 186.64]
tempIVGN2024_err =[0.119, 0.02]
vbdIVGN2024 = [31.44, 28.2]
vbdIVGN2024_err = [0.014, 0.046]

# 2022
# a = [
# (176.9,0.2,27.96,0.011),
# (194.3,0.2,28.45,0.008),
# (207.6,0.2,28.79,0.0084),
# (217.12,0.034,29.11,0.012),
# (235,0.2,29.67,0.014),
# (243,0.2,29.88,0.012),
# ]
# list(zip(*a))
tempIVGN2022     = [176.9, 194.3, 207.6,  217.12, 235,   243]
tempIVGN2022_err = [0.2,   0.2,   0.2,    0.034,  0.2,   0.2]
vbdIVGN2022      = [27.96, 28.45, 28.79,  29.11,  29.67, 29.88]
vbdIVGN2022_err  = [0.011, 0.008, 0.0084, 0.012,  0.014, 0.012]




c, iv_gn_2022 = temp_fit(*curve_fit(linear_func, tempIVGN2022, vbdIVGN2022, sigma=vbdIVGN2022_err, absolute_sigma=True),)
plt.errorbar(
    tempIVGN2022,
    vbdIVGN2022,
    xerr=tempIVGN2022_err,
    yerr=vbdIVGN2022_err,
    markersize=6,
    fmt=".",
    color=c,
    label='2022 GN - IV',
)
c, iv_gn_2025 = temp_fit(*curve_fit(linear_func, tempIVGN2025, vbdIVGN2025, sigma=vbdIVGN2025_err, absolute_sigma=True),)
plt.errorbar(
    tempIVGN2025,
    vbdIVGN2025,
    xerr=tempIVGN2025_err,
    yerr=vbdIVGN2025_err,
    markersize=6,
    fmt=".",
    color=c,
    label='2025 GN - IV',
)
# plt.errorbar(
#     tempIVGN2024,
#     vbdIVGN2024,
#     xerr=tempIVGN2024_err,
#     yerr=vbdIVGN2024_err,
#     markersize=6,
#     fmt=".",
#     label='2024 GN - IV',
# )
# popt, pcov = curve_fit(linear_func, temp[:-skip_last_num], vbd[:-skip_last_num],
#                        sigma=vbd_err[:-skip_last_num], absolute_sigma=True)

popt, pcov = curve_fit(linear_func, temp, vbd, sigma=vbd_err, absolute_sigma=True)
c, spe_2025 = temp_fit(popt, pcov, min_temp=125)
plt.errorbar(
    temp,
    vbd,
    xerr=temp_err,
    yerr=vbd_err,
    markersize=6,
    fmt=".",
    color=c,
    label='2025 GN - SPE',
)
popt, pcov = curve_fit(linear_func, temp2022, vbd2022, sigma=vbd2022_err, absolute_sigma=True)
c, spe_2022 = temp_fit(popt, pcov, min_temp=125)
plt.errorbar(
    temp2022,
    vbd2022,
    xerr=temp2022_err,
    yerr=vbd2022_err,
    markersize=6,
    fmt=".",
    color=c,
    label='2022 GN - SPE',
)


v  = [27.13, 27.23, 27.0,  27.51, 27.11, 27.08, 26.96, 27.16, ]
s  = [.0342, .04,   .0488, .0528, 0.06,  .096,  0.10,  0.03,  ]
t  = [167.8, 168.4, 168.5, 167.0, 169.8, 169.6, 169.7, 169.8, ]
ts = [.062,  .067,  .103,  .134,  .221,  .148,  .187,  .166,  ]
plt.errorbar(
    t,
    v,
    xerr=ts,
    yerr=s,
    markersize=6,
    fmt=".",
    label='2023-24 LXe - SPE',
)
# plt.errorbar(
#     tempIVLXe,
#     vbdIVLXe,
#     xerr=tempIVLXe_err,
#     yerr=vbdIVLXe_err,
#     markersize=6,
#     fmt=".",
#     label='2023-24 LXe - IV',
# )

label = f"""\
Change in Breakdown Voltage:
2025 SPE: {spe_2025}
2022 SPE: {spe_2022}\
"""
text_box = AnchoredText(label, frameon=True, loc=4, pad=0.7, prop={'fontsize':14})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.xlabel('Temperature [K]')
plt.ylabel('Breakdown Voltage [V]')
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)
plt.show()


### CA analysis

import glob
import src.AnalyzeResults as ar
import pandas as pd

files = glob.glob("unity/*.csv")

ar.plot_CA(files[:1]+files[-1:], ['July 11th 2023, GN 171K', 'August 12th 2025, GN 252K'])

ar.plot_CA(files[:-1])


### PDE

invC_alpha = 1143.98e-6
invC_alpha_err = 0.12e-6
invC_spe = 997e-6
invC_spe_err = .50e-6

before = ar.AnalyzeResults('Copper Spacing', 'results/2023July13_Alpha.csv',
                         'results/August1_silicon_vbd.csv', 'results/July13_171K_CA_GN.csv',
                         invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27.22, .0686,
                         ov_max=6, ov_min=2.5)

after = ar.AnalyzeResults('Si Tall Reflectors, Pre-baking', 'results/2024June20_Alpha.csv',
                             'results/20240620_LXe_SPE_vbd_bias.csv',
                             'results/July13_171K_CA_GN.csv',
                             invC_alpha, invC_alpha_err, invC_spe, invC_spe_err,
                             26.96, .102, # Thomas's SPE
                             ov_max=6.2, ov_min=2.7)

si_short_none = ar.AnalyzeResults('August 2023', 'results/2023August10_Alpha.csv',
                                  'results/20230810_SPE_no_si.csv', 'results/July13_171K_CA_GN.csv',
                                  invC_alpha, invC_alpha_err, invC_spe, invC_spe_err, 27., .0488,
                                  ov_max=6, ov_min=2.5)
PTE = 0.0013598 # +/- 3.7e-6
pde = si_short_none.plot_PDE(N*PTE, legtext='0.136 ± 0.00037 %')
plt.show()

out_file = 'results/20230713_PDE.csv'
N = 5.49/(19.6E-6)
PTE = 0.0052206 # ± 7.2e-06
before_pde = before.plot_PDE(N*PTE, legtext='0.522 %')

# PTE = 0.0025563
PTE = 0.0016941
PTE_err = 5.1e-06
after.plot_PDE(N*PTE, other_data=[before_pde])
plt.show()
