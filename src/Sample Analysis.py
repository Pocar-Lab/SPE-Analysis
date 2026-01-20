from AnalysisDefinitions import AnalysisUltimate as AU
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':

    folder = "20251114_SPE_GN_137K" #requires a baseline file
    ult = AU(h5_folder_path='DAQ_data/'+folder+'/',)

    biases = list(ult.biases) #what biases to analyze
    biases = sorted(list(biases))
    num_pes = [5,   5,   5,   5,   5,   6,   6,   6,   6, ] #the number of peaks to fit, ordered by bias in increasing order
    pes_dict = {biases[i+1]: num_pes[i] for i in range(len(biases)-1)} #create a dictinoary. Can be manual: keys are biases, data is number of peaks as int
    lowerlim = [3.8, 3.8, 3.8, 3.9, 4.0, 4.1, 4.2, 4.4, 4.7] #in mV #the lowerlims to be used, ordered by bias in increasing order
    lowerlimdict = {biases[i+1]:lowerlim[i]/1000 for i in range(len(lowerlim))} #create a dictionary. Can be manual: keys are biases, data is lowerlim as float
    pe_center_guesses = None#[5.0, 5.2, 5.5, 5.9, 6.6, 7.0, 7.4, 7.8, 8.3] #centers of the first pe, ordered by bias in increasing order
    if pe_center_guesses:
        centers = {biases[i+1]: [pe_center_guesses[i] * peak / 1000 for peak in range(1,num_pes[i]+1)] for i in range(len(biases)-1)} #create a dictinoary. Can be manual: keys are biases, data is list of centers guesses for the multigaussian
    multicores = 6 #how many cores to use multiprocessing with. None for no multiprocessing
    invC = 0.011547 #inverse capacitance of pream-shaper circuit, see Noah's calibration report
    invCerr = 0.000098 #error on above term

    #print(biases)
    uplims = {v:-1 for v in biases}
    uplims[20] = 0.063
    poly = {v:True for v in biases}
    poly[20] = False
    proms = {v:0.002 for v in biases}
    proms[20] = 0.008

    ult.args_processwaveforms(do_filter=True,
                            prominence=proms,
                            condition = 'GN',
                            baseline_correct=True,
                            upper_limit=uplims,
                            poly_correct=poly)
    #print(ult.processwaveforms_arguments)

    ult.do_processwaveforms(temperature = 137,timeit=False,biases=biases,multiprocess_cores=multicores)

    cutoffs = {}
    for v in biases:
        if v <25:
            continue
        cutoffs[v] = (lowerlimdict[v], .1)
    peak_ranges = {biases[i+1]: (1,num_pes[i]) for i in range(len(biases)-1)}

    if not pe_center_guesses:
        centers = ult.guess_pe_locations(get_values=True,show_plots=False,cutoff=cutoffs)

    ult.args_processhistograms(centers = centers, 
                            baseline_correct=True,
                            peaks='all',
                            cutoff=cutoffs,
                            background_linear=False,
                            peak_range=peak_ranges)

    ult.create_processhistograms()

    ult.do_processhistograms(biases=biases)
    for bias in biases:
        if bias < 25:
            continue
        ult.histograms[bias].plot_peak_histogram()

    ult.do_spe_data(invC=invC,invC_err=invCerr,biases=biases)
    ult.spe_data.plot_spe(in_ov=False,absolute=False)