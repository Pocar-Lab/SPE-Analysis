from AnalysisDefinitions import AnalysisUltimate as AU
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':

    folder = "20251114_SPE_GN_137K" #requires a baseline file
    ult = AU(h5_folder_path='DAQ_data/'+folder+'/',)

    biases = list(ult.biases)
    pe_center_guesses = [5.0, 5.2, 5.5, 5.9, 6.6, 7.0, 7.4, 7.8, 8.3]
    num_pes = [6,   6,   6,   6,   6,   6,   6,   6,   6, ]
    lower_cutoff = None #[3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.4, 4.7] #the actual lowerlims used in the analysis, if any

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

    startlim = 3.5 #in mV
    endlim = 4.5 #in mV
    fine = 20
    lowerlimlist = [(startlim+(endlim-startlim)*i/fine)/1000 for i in range(fine)]
    ult.do_processwaveforms(temperature = 235,timeit=False,biases=biases,multiprocess_cores=6)

    biases = sorted(list(biases))
    if lower_cutoff:
        lower_cutoffdict = {biases[i+1]:lower_cutoff[i] for i in range(len(lower_cutoff))}
    centers = {biases[i+1]: [pe_center_guesses[i] * peak / 1000 for peak in range(1,num_pes[i]+1)] for i in range(len(biases)-1)}
    cutoffs = {}
    for v in biases:
        if v <25:
            continue
        cutoffs[v] = (.0035, .1)
    peak_ranges = {biases[i+1]: (1,num_pes[i]) for i in range(len(biases)-1)}
    pes_dict = {biases[i+1]: num_pes[i] for i in range(len(biases)-1)}
    redchidict = {bias: [] for bias in biases if bias > 25}
    spedict = {bias:[] for bias in biases if bias > 25}

    ult.args_processhistograms(centers = centers, 
                            baseline_correct=True,
                            peaks='all',
                            cutoff=cutoffs,
                            background_linear=False,
                            peak_range=peak_ranges)

    ult.create_processhistograms()

    for bias in biases: #ult.biases:
        if bias < 25:
            continue
        redchilist = []
        #debugmajor = []
        for lowerlim in lowerlimlist:
            ult.histograms[bias].cutoff = (lowerlim,.1)
            ult.do_processhistograms(biases=[bias])
        #    debugminor = []
        #    for peak in range(0,len(ult.histograms[bias].peak_sigmas)):
        #        actual_peak = ult.histograms[bias].peak_range[0] + peak
        #        curr_sigma_err = ult.histograms[bias].peak_fit.params['g' + str(actual_peak) + '_sigma'].stderr
        #        debugminor.append(curr_sigma_err)
        #    if all(debugminor):
        #        debugmajor.append(max(debugminor))
        #    else:
        #        debugmajor.append(False)
        #        continue
            redchidict[bias].append(ult.histograms[bias].peak_fit.redchi)
            spedict[bias].append(ult.histograms[bias].get_spe())
            #ult.histograms[bias].plot_peak_histogram()

    xdata = [1000*v for v in lowerlimlist]
    plt.figure()
    for bias in redchidict:
        plt.plot(xdata,redchidict[bias],label=f'{bias} V bias, {pes_dict[bias]} PEs fitted')
        if lower_cutoff:
            a = np.absolute(np.array(xdata)-lower_cutoffdict[bias])
            i = np.where(a == a.min())[0][0]
            plt.scatter(xdata[i],redchidict[bias][i],c='r')
    plt.title(f'{folder}, num_pes = {num_pes}')
    plt.xlabel('Lower Limit (mV)')
    plt.ylabel('Reduced $\chi^2$ of Multigaussian fit')
    plt.yscale('log')
    plt.legend()

    plt.figure()
    for bias in spedict:
        spedata = [l[0] for l in spedict[bias]]
        plt.errorbar(x=xdata,y=spedata,yerr=[l[1] for l in spedict[bias]],label=f'{bias} V bias, {pes_dict[bias]} PEs fitted')
        if lower_cutoff:
            a = np.absolute(np.array(xdata)-lower_cutoffdict[bias])
            i = list(np.where(a == a.min()))[0][0]
            plt.scatter(xdata[i],spedata[i],c='r')
    plt.title(f'{folder}, num_pes = {num_pes}')
    plt.xlabel('Lower Limit (mV)')
    plt.ylabel('SPE Amplitude')
    plt.legend()

    plt.show()