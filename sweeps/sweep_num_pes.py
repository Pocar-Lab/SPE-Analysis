import AnalysisDefinitions as AU
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    folder = "20251114_SPE_GN_137K" #requires a baseline file
    ult = AU.AnalysisUltimate(h5_folder_path='DAQ_data/'+folder+'/',)
    biases = sorted(list(ult.biases))
    low_pe = 4 #start amount of pes, inclusive
    high_pe = 8 #last amount of pes, inclusive
    lowerlim = 4.35 #in mV
    redchidict = {bias: [] for bias in biases if bias > 25}
    spedict = {bias:[] for bias in biases if bias > 25}
    pe_center_guesses = [5.0, 5.2, 5.5, 5.9, 6.6, 7.0, 7.4, 7.8, 8.3] #in mV
    actual_pes = [4,   4,   5,   5,   5,   6,   6,   6,   6, ] #what was used in the actual analysis of the data


    uplims = {v:-1 for v in biases}
    uplims[20] = 0.063
    poly = {v:True for v in biases}
    poly[20] = False
    proms = {v:0.0035 for v in biases}
    proms[20] = 0.008
    actual_pesdict = {biases[i+1]:actual_pes[i] for i in range(len(actual_pes))}

    ult.args_processwaveforms(do_filter=True,
                            prominence=proms,
                            condition = 'GN',
                            baseline_correct=True,
                            upper_limit=uplims,
                            poly_correct=poly)
    #print(ult.processwaveforms_arguments)
    ult.do_processwaveforms(temperature = 235,
                            timeit=False,
                            multiprocess_cores=4)



    for num_pes in range(low_pe,high_pe+1):
        
        #num_pes = np.array([3, 6, 5, 5, 6, 7, 6, 7, 7])
        biases = sorted(list(biases))
        centers = {biases[i+1]: [pe_center_guesses[i] * peak / 1000 for peak in range(1,num_pes+1)] for i in range(len(biases)-1)}
        cutoffs = {}
        for v in biases:
            if v <25:
                continue
            cutoffs[v] = (lowerlim/1000, .1)
        peak_ranges = {biases[i+1]: (1,num_pes) for i in range(len(biases)-1)}
        pes_dict = {biases[i+1]: num_pes for i in range(len(biases)-1)}

        ult.args_processhistograms(centers = centers, 
                                baseline_correct=True,
                                peaks='all',
                                cutoff=cutoffs,
                                background_linear=False,
                                peak_range=peak_ranges)

        ult.create_processhistograms()

        #lowerlimlist = [(3.5+(5.5-3.5)*i/fine)/1000 for i in range(fine)]
        #debugmajor = []
        #for lowerlim in lowerlimlist:
        ult.do_processhistograms()
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
        for bias in redchidict:
            redchidict[bias].append(ult.histograms[bias].peak_fit.redchi)
            #if redchilist[-1]>=10:
            #    ult.histograms[bias].plot_peak_histogram()
            spedict[bias].append(ult.histograms[bias].get_spe())
        #ult.do_spe_data(invC=0.011547,
        #                invC_err=0.000098)
        #ult.spe_data.plot_spe(False,False)

        #print(debugmajor)
    xdata = range(low_pe,high_pe+1)
    plt.figure()
    for bias in redchidict:
        plt.plot(xdata,redchidict[bias],label=f'{bias} V bias, {lowerlim} mV cutoff')
        a = np.absolute(np.array(xdata)-actual_pesdict[bias])
        i = np.where(a == a.min())[0][0]
        plt.scatter(xdata[i],redchidict[bias][i],c='r')
    plt.title(f'{folder}, lowerlim = {lowerlim}')
    plt.xlabel('Number of fitted PEs')
    plt.ylabel('Reduced $\chi^2$ of Multigaussian fit')
    plt.yscale('log')
    plt.legend()

    plt.figure()
    for bias in spedict:
        spedata = [l[0] for l in spedict[bias]]
        plt.errorbar(x=xdata,y=spedata,yerr=[l[1] for l in spedict[bias]],label=f'{bias} V bias, {lowerlim} mV cutoff')
        a = np.absolute(np.array(xdata)-actual_pesdict[bias])
        i = np.where(a == a.min())[0][0]
        plt.scatter(xdata[i],spedata[i],c='r')
    plt.title(f'{folder}, lowerlim = {lowerlim}')
    plt.xlabel('Number of fitted PEs')
    plt.ylabel('SPE Amplitude')
    plt.legend()

    plt.show()