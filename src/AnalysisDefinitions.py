import numpy as np
import matplotlib.pyplot as plt
import os
from ProcessWaveforms import ProcessWaveforms
from MeasurementInfo import MeasurementInfo
from ProcessHistograms import ProcessHist
from AnalyzePDE import SPE_data
import h5py
import time

class AnalysisUltimate:
    def __init__(self,
                 h5_folder_path: str,
                 ):
        """
        Creates an AnalysisUltimate obejct for performing SPE analysis.

        Args:
            h5_folder_path (str): the path to the folder that holds all the hdf5 datafiles (and nothing else). Must end in '/'.
        """
        self.folder_path = h5_folder_path
        self.files = os.listdir(self.folder_path)
        self.measurementinfos = {} # keys are hdf5 file names, data is a ProcessWaveforms object
        self.biases = {} # keys are hdf5 file names, data is bias voltage
        for name in self.files:
            f = h5py.File(self.folder_path+name,'r') # get hdf5 file
            group_names = list(f['RunData'].keys())
            self.biases[name] = f['RunData'].get(group_names[0]).attrs['Bias(V)']
        self.files = {f:self.biases[f] for f in self.files}
        self.biases = {self.biases[f]:f for f in self.biases}

    def args_processwaveforms(self,
                            acquisition: dict[str] | None = None,
                            do_filter: dict | bool = False,
                            plot_waveforms: dict | bool = False,
                            upper_limit: dict | float = -1,
                            baseline_correct: dict | bool = False,
                            poly_correct: dict | bool = False,
                            prominence: dict | float = 0.005,
                            fourier: dict | bool = False,
                            condition: dict | str = "unspecified medium (GN/LXe/Vacuum)",
                            num_waveforms: dict | float = 0,
                        ):
        """
        Initialize the arguments for Process waveform data from hdf5 file to find peaks.

        Apply filtering and baseline correciton to waveforms. Can process alpha pulse waveforms,
        SPE waveforms, or baseline data. Records alpha pulse heights, SPE pulse heights, or
        aggregate all ampltiudes respectively. Optionally plots waveforms to inspect data for
        tuning peak finding.

        All arguments should be dictionaries with biases as keys, specifying any non-default parameters.
        Arguments that are not given as dictionaries will be applied to all, including any pre-breakdown.

        Once run, these can be freely inspected and edited via the attribute self.processwaveforms_arguments.

        To create the MeasurementInfo objects for each bias, run the do_processwaveforms method.

        Args:
            f (list): list of h5 file names
            acquisition (str, optional): specified file name. Defaults to 'placeholder'.
            do_filter (bool, optional): activates butterworth lowpass filter if True. Defaults to False.
            plot_waveforms (bool, optional): plots waveforms if True. Defaults to False.
            upper_limit (float, optional): amplitude threshold for discarding waveforms. Set to -1 for automatic setting. Defaults to -1.
            baseline_correct (bool, optional): baseline corrects waveforms if True. Defaults to False.
            prominence (float, optional): parameter used for peak finding algo. Defaults to 0.005.
            fourier (bool, optional): if True performs fourier frequency subtraction. Defaults to False.
            num_waveforms: (float, optional): number of oscilloscope traces to read in before stopping; default 0 reads everything
        """
        self.processwaveforms_arguments = {'acquisition': acquisition, 
                                           'do_filter': do_filter, 
                                           'plot_waveforms': plot_waveforms, 
                                           'upper_limit': upper_limit, 
                                           'baseline_correct': baseline_correct, 
                                           'poly_correct': poly_correct, 
                                           'prominence': prominence, 
                                           'fourier': fourier, 
                                           'condition': condition, 
                                           'num_waveforms': num_waveforms,
                                           }
        for a in self.processwaveforms_arguments:
            if type(self.processwaveforms_arguments[a]) is not dict:
                self.processwaveforms_arguments[a] = {v: self.processwaveforms_arguments[a] for v in self.biases}
            for v in self.biases:
                if v not in self.processwaveforms_arguments[a]:
                    self.processwaveforms_arguments[a][v] = 'not set'

    def do_processwaveforms(self,
                            temperature: float,
                            biases: list[float] | None = None,
                            timeit: bool = False,
                            ):
        """
        Create and run temporary ProcessWaveform objects on the given biases using the arguments created in self.args_processwaveforms.
        MeasurementInfo objects are created and stored in self.measurementinfos as a dictionary, keyed by bias.

        Args:
            temperature (float): temperature of the experiment
            biases (list of floats, optional): which biases to evaluate or reevaluate. Does not automatically include the breakdown. None does all given biases. Defaults to None.
            timeit (bool, optional): time how long this method takes. Defaults to False.
        """
        if timeit:
            t0 = time.time()
        if not biases:
            biases = list(self.biases)

        if not set(biases).issubset(set(list(self.biases))):
            raise Exception("biases must either be a subset of self.biases's keys or be None")


        runs = {} # keys are bias voltages, data is ProcessWaveforms object
        for bias in biases:
            f = self.folder_path + self.biases[bias]
            warg = {k: self.processwaveforms_arguments[k][bias] for k in self.processwaveforms_arguments if self.processwaveforms_arguments[k][bias] != 'not set'}
            if bias < 25:
                warg['is_pre_bd'] = True
                bias = 'prebreakdown'
            runs[bias] = ProcessWaveforms([f],**warg)
        if 'prebreakdown' not in runs:
            runs['prebreakdown'] = None
        for bias in biases:
            if bias > 25:
                self.measurementinfos[bias] = MeasurementInfo(self.processwaveforms_arguments['condition'][bias],temperature,runs[bias],runs['prebreakdown'])
        if timeit:
            t1 = time.time()
            print(f'do_processwaveforms took {t1-t0:0.4} seconds to complete.')
    def guess_pe_locations(self,
                           get_values: bool = False,
                           show_plots: bool = True,
                           ) -> dict | None:
        """
        Creates an estimation for the 1PE amplitude for use in ProcessHist via taking the mode.
        These are not stored.

        Args:
            get_values (bool, optional): do you want to return the PE values? If so, it will be in a dictionary keyed by bias. Defaults to False.
            show_plots (bool, optional): do you want to see the histograms? Defaults to True.
        """
        maxes = {}
        figs = {}
        axes = {}
        for bias in self.biases:
            if bias < 25:
                continue
            m = self.measurementinfos[bias]
            figs[bias],axes[bias] = plt.subplots(1,1)
            n,bins,_ = axes[bias].hist(m.all_peaks,bins = 1000,color='blue')
            maxes[bias] = [bins[np.argmax(n)] * i for i in range(1,10)]
            [axes[bias].axvline(x=v,color='red') for v in maxes[bias]]
            axes[bias].set_title(f'{bias} V bias')
        if show_plots:
            plt.show()
        [plt.close(figs[v]) for v in figs]
        if get_values:
            return maxes


    def args_processhistograms(self,
                               centers: dict[list[float]],
                               baseline_correct: dict[bool] = False,
                               cutoff: dict[tuple[float,float]] = (0,np.inf),
                               peak_range: dict[int] = 4,
                               background_linear: dict[bool] = True,
                               peaks: dict[str] = 'all'
                               ):
        """
        Process and histogram the identified peaks and fit with a multi-gaussian to extract the
        SPE amplitude.

        Once run, these can be freely inspected and edited via the attribute self.processhsit_arguments.

        To create the ProcessHist objects, run self.create_processhistograms.

        Args:
            centers (list[float]): Initial guesses for centroid of each gaussian.
            baseline_correct (bool,optional): Boolean value indicating if baseline correction needs to be applied. Defaults to False.
            cutoff (tuple[float, float],optional): Low and high cutoff values. Defaults to (0,np.inf).
            peak_range (int,optional): The number of peaks you want to fit. Defaults to 4.
            background_linear (bool,optional): If to fit a linear or exponential background to the multi-gaussian. Defaults to True.
            peaks (str): Which peaks to include, either all, LED, or dark. Defaults to all
        """
        self.processhist_arguments = {'info':self.measurementinfos,
                                      'centers': centers,
                                      'baseline_correct': baseline_correct,
                                      'cutoff': cutoff,
                                      'peak_range': peak_range,
                                      'background_linear': background_linear,
                                      'peaks': peaks}
        for a in self.processhist_arguments:
            if type(self.processhist_arguments[a]) is not dict:
                self.processhist_arguments[a] = {v: self.processhist_arguments[a] for v in self.biases}
            for v in self.biases:
                if v not in self.processhist_arguments[a] and v > 25:
                    self.processhist_arguments[a][v] = 'not set'

    def create_processhistograms(self):
        """
        Create ProcessHist objects on the given biases using the arguments created in args_processhistograms.
        These are stored in self.histograms as a dictionary, keyed by bias. They can be called and run manually,
        or they can be run automatically via self.do_processhistograms.

        Does not create a ProcessHist for pre-breakdown voltages
        """
        self.histograms = {}
        for bias in self.biases:
            if bias < 25:
                continue
            warg = {k: self.processhist_arguments[k][bias] for k in self.processhist_arguments if self.processhist_arguments[k][bias] != 'not set'}
            self.histograms[bias] = ProcessHist(**warg)

    def do_processhistograms(self,
                             biases: list[float] | None = None,
                             ):
        """
        Run the ProcessHist.process_spe for the specified biases.

        Args:
            biases (list of floats or None, optional): which biases to evaluate or reevaluate. Automatically excludes pre-breakdown data. None does all biases. Defaults to None.
        """
        if not biases:
            biases = list(self.biases)

        if not set(biases).issubset(set(list(self.biases))):
            raise Exception("biases must either be a subset of self.biases's keys or be None")            

        for bias in biases:
            if bias < 25:
                continue
            self.histograms[bias].process_spe()
        
    def do_spe_data(self,
                    invC: float,
                    invC_err: float,
                    filtered: bool,
                    do_CA: bool = False,
                    biases: list[float] = None,
                    ):
        """
        Create and run the SPE_data for the current ProcessHists in self.histograms. The class is stored in self.spe_data

        Args:
            invC (float): The inverse of the capacitance.
            invC_err (float): The error in the inverse of the capacitance.
            filtered (bool): A flag indicating whether the data is filtered.
            do_CA (bool, optional): A flag for whether the class should attempt CA calculation.
            biases (list of floats or None, optional): which biases to include. None does all given biases. Defaults to None.
        """
        if not biases:
            biases = list(self.biases)

        if not (biases <= list(self.biases)):
            raise Exception("biases must either be a subset of self.biases's keys or be None")            

        campaign = [self.histograms[v] for v in self.histograms]
        self.spe_data = SPE_data(campaign,invC,invC_err,filtered,do_CA)