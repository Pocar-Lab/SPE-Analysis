# SPE Analysis

Single Photo Electron analysis code for the Pocar Lab LXe system.

## File Structure

* **Analysis** Script: Main script which contains how to analyze specific data, calls other files
  from here
  * Format: CONDITION_MONTH_YEAR_Analysis (eg LXe_May_2023_Analysis)
* **RunInfo**: Load data from hdf5, baseline correct, perform digital filtering, find pulses
* **MeasurementInfo**: Organize metadata into object
* **ProcessWaveforms**: Performs Gaussian and linear fits on finger plots, performs CA calculation and fit
* **Analyze_PDE**:
  * SPE_data class: Fit to breakdown voltage, error propagation from electronics calibration, CA plots
  * Alpha_data class: Plot amplitudes, calculate PDE, calculate number incident photons

## Contributing

When creating a new analysis for a new data set, copy an existing analysis script to a new file
with the naming convention `CONDITION_MONTH_YEAR_Analysis.py`. Add you name, date, and brief
description of run conditions of the data at the top for future reference. Do not make changes to
an existing analysis script unless you are analysing the same data.

For details on how to use git see the [git guide][1], in summary:
* Always pull latest changes before you commit new ones!! This is to avoid annoying merge
  conflicts.
* Make sure each commit has a clear and concise commit message describing the changes made.
* Commit often. Commits should only change/do one thing at a time.  Only commit changes that are
  done and working, not work in progress code.


[1]: https://umass-my.sharepoint.com/:b:/r/personal/pocar_umass_edu/Documents/PocarLab-021%20Repository/Lab%20Documentation/Git_Guide.pdf?csf=1&web=1&e=mtK75I
