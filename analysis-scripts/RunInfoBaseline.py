from RunInfo import *


class RunInfoBaseline(RunInfo):
    def __init__(
        self,
        f: list,
        acquisition: str = "placeholder",
        do_filter: bool = False,
        plot_waveforms: bool = False,
        upper_limit: float = 4.4,
        baseline_correct: str = "",
        prominence: float = 0.005,
        specifyAcquisition: bool = False,
        fourier: bool = False,
    ):
        self.peak_search_params = {
            "height": 0.0,  # SPE
            "threshold": None,  # SPE
            "distance": None,  # SPE
            "prominence": prominence,
            "width": None,  # SPE
            "wlen": 100,  # SPE
            "rel_height": None,  # SPE
            "plateau_size": None,  # SPE
            # 'distance':10 #ADDED 2/25/2023
        }

        super().__init__(
            f,
            acquisition,
            do_filter,
            plot_waveforms,
            upper_limit,
            baseline_correct,
            prominence,
            specifyAcquisition,
            fourier,
        )

    def get_peaks(self, filename: str, acquisition_name: str) -> list:
        """Aggregates y-axis data of all solicited waveforms after optionally
        filtering (why?), baseline subtracting (why?), and removing first
        100 data points (why?). Also optionally plots waveforms.

        Args:
            filename (str): path on disk of hdf5 data to be processed
            acquisition_name (str): name of acquisition to be processed

        Returns:
            list: concatened list of all amplitude values for all waveforms
        """
        all_peaks = []
        curr_data = self.acquisitions_data[filename][acquisition_name]
        time = self.acquisitions_time[filename][acquisition_name]
        window_length = time[-1] - time[0]
        num_points = float(len(time))
        fs = num_points / window_length
        # print(fs)
        num_wavefroms = np.shape(curr_data)[1]
        print(f"num_wavefroms: {num_wavefroms}")
        if (
            self.plot_waveforms
        ):  # TODO replace w/ num_wavefroms if not self.plot_waveforms else 20
            num_wavefroms = 20
        for idx in range(num_wavefroms):
            if idx % 100 == 0:
                print(idx)

            amp = curr_data[:, idx]
            if (
                np.amax(amp) > self.upper_limit
            ):  # skips waveform if amplitude exceeds upper_limit
                continue

            if self.baseline_correct:
                use_bins = np.linspace(-self.upper_limit, self.upper_limit, 1000)
                curr_hist = np.histogram(amp, bins=use_bins)
                baseline_level, _ = get_mode(curr_hist)
                amp = amp - baseline_level
                self.baseline_mode = baseline_level

            if self.do_filter:
                sos = signal.butter(3, 4e5, btype="lowpass", fs=fs, output="sos")
                filtered = signal.sosfilt(sos, amp)
                amp = filtered

            # peaks, props = signal.find_peaks(amp, **self.peak_search_params)
            if self.plot_waveforms:
                if self.fourier:
                    fourier = np.fft.fft(amp)
                    n = amp.size
                    duration = 1e-4
                    freq = np.fft.fftfreq(n, d=duration / n)
                    colors = [
                        "b",
                        "g",
                        "r",
                        "m",
                        "c",
                        "y",
                        "k",
                        "aquamarine",
                        "pink",
                        "gray",
                    ]
                    marker, stemlines, baseline = plt.stem(
                        freq,
                        np.abs(fourier),
                        linefmt=colors[0],
                        use_line_collection=True,
                        markerfmt=" ",
                    )
                    plt.setp(
                        stemlines,
                        linestyle="-",
                        linewidth=1,
                        color=colors[0],
                        alpha=5 / num_wavefroms,
                    )  # num_wavefroms always 20?
                    plt.yscale("log")
                    plt.show()
                else:
                    plt.title(acquisition_name)
                    plt.tight_layout()
                    plt.plot(time, amp)
                    plt.show()
            amp = list(amp[100:])
            all_peaks += amp
        return all_peaks

    def get_data(self) -> None:
        """Get peak data and add as a dict to self.peak_data. No return value."""
        self.peak_data = {}
        for curr_file in self.hd5_files:
            self.peak_data[curr_file] = {}
            for curr_acquisition_name in self.acquisition_names[curr_file]:
                if self.specifyAcquisition:
                    curr_acquisition_name = self.acquisition
                curr_peaks = self.get_peaks(curr_file, curr_acquisition_name)
                self.peak_data[curr_file][curr_acquisition_name] = curr_peaks
                if self.plot_waveforms or self.specifyAcquisition:
                    break
