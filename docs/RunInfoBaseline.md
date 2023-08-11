Module RunInfoBaseline
======================

Classes
-------

`RunInfoBaseline(f: list, acquisition: str = 'placeholder', do_filter: bool = False, plot_waveforms: bool = False, upper_limit: float = 4.4, baseline_correct: str = '', prominence: float = 0.005, specifyAcquisition: bool = False, fourier: bool = False)`
:   Extract raw waveform data from either txt or h5 files. Apply filtering and baseline correciton
    to waveforms. Can process alpha pulse waveforms, SPE waveforms (in LXe or Vacuum), or baseline data.
    Records alpha pulse heights, SPE pulse heights, or aggregate y-axis data respectively.
    Can also collect baseline info from SPE data in the absence of dedicated baseline data (why?).
    Optionally plots waveforms to inspect data for tuning peak finding algo.
    
    Args:
        f list: list of h5 file names
        acquisition (str, optional): specified file name. Defaults to 'placeholder'.
        do_filter (bool, optional): activates butterworth lowpass filter if True. Defaults to False.
        plot_waveforms (bool, optional): plots waveforms if True. Defaults to False.
        upper_limit (float, optional): amplitude threshold for discarding waveforms. Defaults to 4.4.
        baseline_correct (str, optional): if "mode" uses mode subtraction, if "peakutils" uses 2nd degree polyfit subtraction,
            if falsey does not do baseline correction. Defaults to "".
        prominence (float, optional): parameter used for peak finding algo. Defaults to 0.005.
        specifyAcquisition (bool, optional): if True, uses acquisition to extract just one acquisition dataset. Defaults to False.
        fourier (bool, optional): if True performs fourier frequency subtraction. Defaults to False.
    
    Raises:
        TypeError: _description_

    ### Ancestors (in MRO)

    * RunInfo.RunInfo

    ### Methods

    `get_data(self) ‑> None`
    :   Get peak data and add as a dict to self.peak_data. No return value.

    `get_peaks(self, filename: str, acquisition_name: str) ‑> list`
    :   Aggregates y-axis data of all solicited waveforms after optionally
        filtering (why?), baseline subtracting (why?), and removing first
        100 data points (why?). Also optionally plots waveforms.
        
        Args:
            filename (str): path on disk of hdf5 data to be processed
            acquisition_name (str): name of acquisition to be processed
        
        Returns:
            list: concatened list of all amplitude values for all waveforms