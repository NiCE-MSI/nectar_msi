"""Class to select the peaks from the mean spectrum (peak picking routine)."""

"""___Built-In Modules___"""
from nectar_msi.gaussian_fitting.gaussian_fitting import GaussianFitting
from nectar_msi.plotting.plotting import Plotting
from nectar_msi.noise_correction.noise_determination import NoiseDetermination

"""___Third-Party Modules___"""
import numpy as np
import pandas as pd

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class PeakPicking:
    def __init__(self):
        self.noisedetermination = NoiseDetermination()
        self.gaussian_fitting = GaussianFitting()
        self.plotting = Plotting()
        pass

    def peak_picking(self, mean_spectrum, path_outputs, plot_peaks=False, save_tables=False, save_fitting=False):
        """
        Function to select peaks in the mean spectrum. 1) selects all peaks by first derivative, 2) calculates resolving
         power to determine theoretical peak-width, 3) does gaussian fitting with constrained parameters, 4) creates
         final tables.
        :param mean_spectrum: mean spectrum used to find the peaks
        :type mean_spectrum: DataSpectrum
        :param path_outputs: path where to save the outputs
        :type path_outputs: str
        :param plot_peaks: option to plot the selected peaks
        :type plot_peaks: bool
        :param save_tables: option to save the tables as .csv files
        :type save_tables: bool
        :param save_fitting: option to save the gaussian fitting plots for all peaks
        :type save_fitting: bool
        :return: list of 200 most intense peaks, list of all selected peaks
        :rtype: pandas dataframe, pandas dataframe
        """

        print("Finding potential peaks by first derivatives...")
        # We select potential peaks by finding the first derivative (noise has been already masked):
        mz_mean = []
        peak_intensity_mean = []
        ppm = 30.0  # allowed distance between peaks. Farther than 30ppm are considered independent peaks.

        for i in range(2, len(mean_spectrum.intensities) - 2):
            # The noise is filtered out of the peak picking routine.
            if mean_spectrum.mzmask[i] == 0.0 and mean_spectrum.mzmask[i + 1] == 1.0:
                istart = i
            if mean_spectrum.mzmask[i] == 1.0 and mean_spectrum.mzmask[i + 1] == 0.0:
                spectrum_window = mean_spectrum.intensities[istart: i + 2]
                xaxis_window = mean_spectrum.mzs[istart: i + 2]
                # One extra value to each side of the window are added to avoid errors (Intensity = 0.).
                xaxis_window = np.append(2 * xaxis_window[0] - xaxis_window[1], xaxis_window)
                xaxis_window = np.append(xaxis_window, 2 * xaxis_window[len(xaxis_window) - 1]
                                         - xaxis_window[len(xaxis_window) - 2],)
                spectrum_window = np.append([0], spectrum_window)
                spectrum_window = np.append(spectrum_window, [0])

                # First derivatives are calculated in the window. These peaks are selected as potential peaks.
                dydx = np.diff(spectrum_window) / np.diff(xaxis_window)

                peaks_centroids = []
                peak_intensities = []
                for jjj in range(len(dydx) - 1):
                    if np.sign(dydx[jjj + 1]) < np.sign(dydx[jjj]):
                        local_mz = xaxis_window[jjj + 1]
                        local_intensity = spectrum_window[jjj + 1]
                        peaks_centroids = np.append(peaks_centroids, local_mz)
                        peak_intensities = np.append(peak_intensities, local_intensity)

                # If there is only one peak in the window, it is selected as potential peak.
                if len(peaks_centroids) == 1:
                    mz_mean = np.append(mz_mean, local_mz)
                    peak_intensity_mean = np.append(peak_intensity_mean, local_intensity)

                # If there are more than one peak in the window, the peaks are classified as independent if they are
                # located farther than 30ppm. The most intense peak is selected as reference if there is confusion.
                else:
                    for iii in range(len(peaks_centroids)):
                        intensities_near = peak_intensities[
                            np.where(np.abs(peaks_centroids - peaks_centroids[iii]) <
                                     (ppm * peaks_centroids[iii]) / 10**6)]

                        if max(intensities_near) == peak_intensities[iii]:
                            mz_mean = np.append(mz_mean, peaks_centroids[iii])
                            peak_intensity_mean = np.append(peak_intensity_mean, peak_intensities[iii])

        # To plot the selected peaks:
        if plot_peaks:
            self.plotting.plot_peak_picking(mean_spectrum, mz_mean, peak_intensity_mean)

        print("Number of detected peaks", len(mz_mean))
        print("Calculating resolving power...")
        # The 200 most intense peaks are chosen to estimate the resolving power of the dataset:
        peak_intensity_mean = np.argpartition(-peak_intensity_mean, 200)
        result_args = peak_intensity_mean[:200]  # This gives the index
        most_intense_peaks_mz = []
        for i in result_args:
            most_intense_peaks_mz = np.append(most_intense_peaks_mz, (mz_mean[i]))
        most_intense_peaks_mz = np.sort(most_intense_peaks_mz)
        # Gaussian fitting in the 200 most intense peaks is applied.
        # The correlation is calculated forcing the intercept to be at zero.
        list_of_fittings_subset = self.gaussian_fitting.gaussian_fitting_spectrum(
            mean_spectrum, most_intense_peaks_mz, path_outputs, save_fitting=save_fitting)
        # list_of_fittings is formed by:
        # 0          1               2       3               4           5           6
        # centroid   peak_intensity  width   chi2/Nsamples   Nsamples    Peak_type   Fitting_success

        mz_best_success = []
        width_best_success = []
        for succ in range(len(list_of_fittings_subset[:, 0])):
            if list_of_fittings_subset[:, 6][succ] == "True":
                mz_best_success = np.append(mz_best_success, float(list_of_fittings_subset[:, 0][succ]))
                width_best_success = np.append(width_best_success, float(list_of_fittings_subset[:, 2][succ]))

        # Correlation that gives the resolving power for the 200 most intense peaks. The function uses a sigma clipping
        # approximation to select only those peaks that are at 3sigma from the new-std with a tolerance of 0.01 from the
        # old-std. The intercept of the correlation is forced to be at zero.
        (func_best, distance_std_best, std_new_best, mz_corr_best, width_corr_best) = \
            self.noisedetermination.sigmaClip_stdv_zero_intercept(width_best_success, mz_best_success)

        print("Correlation most intense peaks:", func_best, std_new_best)
        self.plotting.plot_correlation(mz_corr_best, width_corr_best, func_best, mean_spectrum.mzs, path_outputs)

        # Once the resolving power is calculated, the boundaries of the width are constrained to +-3sigma the
        # theoretical value.
        # ref_m/z_mean centroid_gaussian peak_intensity width chi2/Nsamples Nsamples Peak_type Fitting_success"
        # 0            1                 2              3     4             5        6         7
        print("Creating final list of present peaks...")
        full_list_of_fittings = (self.gaussian_fitting.gaussian_fitting_spectrum_constrained(
            mean_spectrum, mz_mean, func_best, std_new_best, path_outputs, save_fitting=save_fitting))

        a = np.polyfit(full_list_of_fittings[:, 1].astype(float), full_list_of_fittings[:, 5].astype(float), 1)
        func_all_peaks = np.poly1d(a)
        # plots the resolving power for all peaks (upper limits applied on peak-width)
        self.plotting.plot_correlation_all_peaks(full_list_of_fittings[:, 1].astype(float),
                                                 full_list_of_fittings[:, 5].astype(float),
                                                 func_best, func_all_peaks, mean_spectrum.mzs, path_outputs)

        # Creates dataframe for 200 most intense peaks
        list_of_fittings_subset = pd.DataFrame(list_of_fittings_subset,
                                               columns=["gaussian centroid", "peak intensity", "width",
                                                        "chi2/Nsamples", "Nsamples", "Peak type", "Fitting success"])
        # Creates dataframe for all peaks
        full_list_of_fittings = pd.DataFrame(full_list_of_fittings,
                                             columns=["left min", "meas mz", "right min", "gaussian centroid",
                                                      "peak intensity", "width", "chi2/Nsamples", "Nsamples",
                                                      "Peak type", "S/N", "Fitting success", "Peak shape"])

        if save_tables:
            list_of_fittings_subset.to_csv(path_outputs + "list_of_fittings_subset.csv", index=False)
            full_list_of_fittings.to_csv(path_outputs + "full_list_of_fittings.csv", index=False)
            print("!List of peaks saved in: " + path_outputs)

        return list_of_fittings_subset, full_list_of_fittings
