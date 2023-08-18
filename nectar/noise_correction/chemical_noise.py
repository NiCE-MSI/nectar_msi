"""Class to remove the chemical noise following the Adaptive Background Subtraction"""

"""___Built-In Modules___"""
from nectar.plotting.plotting import Plotting
from nectar.data_formats.data_spectrum import DataSpectrum

"""___Third-Party Modules___"""
import numpy as np
import scipy.interpolate

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class ChemicalNoise:
    def __init__(self):
        self.plotting = Plotting()
        pass

    def chemical_noise_correction_45percentile(self, mean_noise_spectrum, mean_spectrum, plot_chemicalnoise=False):
        """
        Function to apply chemical noise correction in the mean spectrum

        :param mean_noise_spectrum: noise spectrum
        :type mean_noise_spectrum: DataSpectrum
        :param mean_spectrum: baseline corrected mean spectrum
        :type mean_spectrum: DataSpectrum
        :param plot_chemicalnoise: option to plot the original, final and chemical noise spectra
        :type plot_chemicalnoise: Boolean
        :return: corrected mean spectrum
        :type: DataSpectrum

        """
        xaxis_steps = np.arange(mean_spectrum.mzs[0] + 1, mean_spectrum.mzs[-1] - 1, 1)
        intensities_new, chemical_noise = self.adaptive_background_subtraction(
            mean_spectrum, mean_noise_spectrum.intensities, xaxis_steps, percentile=45)

        if plot_chemicalnoise:
            self.plotting.plot_spectra_with_chemicalnoise(mean_spectrum, intensities_new, chemical_noise)

        spectrum_corrected = DataSpectrum(-99, [-99, -99], mean_spectrum.mzs, intensities_new)

        return spectrum_corrected

    def adaptive_background_subtraction(self, mean_spectrum, intensities_noise, xaxis_steps, percentile=45):
        """
        Function to estimate and subtract the chemical noise (Adaptive Background Subtraction approach)

        :param mean_spectrum: baseline corrected mean spectrum
        :type mean_spectrum: DataSpectrum
        :param intensities_noise: intensity values masked as noise
        :type intensities_noise: array of floats
        :param xaxis_steps: size of the x-axis bins
        :type xaxis_steps: array of floats
        :param percentile: percentile to be subtracted (default 45)
        :type percentile: int
        :return: corrected intensities, chemical_noise model intensities
        :type: array, array
        """

        intensities_new = np.zeros_like(mean_spectrum.intensities)  # corrected intensity values
        chemical_noise = np.zeros_like(mean_spectrum.intensities)  # chemical noise model

        for xaxis_current in xaxis_steps:
            bins = np.empty((21, 10))
            xbins = (np.linspace(
                xaxis_current - 10.5, xaxis_current + 10.5, 210, endpoint=False).reshape((21, 10)) + 0.05)
            for kk in range(0, 21, 1):
                for kkk in range(0, 10, 1):
                    bins[kk, kkk] = np.average(intensities_noise[np.where(
                        (mean_spectrum.mzs > xbins[kk, kkk] - 0.05) & (mean_spectrum.mzs <= xbins[kk, kkk] + 0.05))])

            xaxis_centroids = (np.arange(xaxis_current - 0.6, xaxis_current + 0.6, 0.1) + 0.05)
            background_noise = np.nanpercentile(bins, percentile, axis=0)
            background_noise = np.append(background_noise, background_noise[0])
            background_noise = np.insert(background_noise, 0, background_noise[-1])

            interpolfunc = scipy.interpolate.interp1d(
                xaxis_centroids[np.where(np.isfinite(background_noise))],
                background_noise[np.where(np.isfinite(background_noise))],
                bounds_error=False,
                fill_value="extrapolate")

            xaxis_currentbin = mean_spectrum.mzs[np.where(
                (mean_spectrum.mzs > xaxis_current - 0.5) & (mean_spectrum.mzs <= xaxis_current + 0.5))]
            yvals = interpolfunc(xaxis_currentbin)[:]

            chemical_noise[np.where(
                (mean_spectrum.mzs > xaxis_current - 0.5) & (mean_spectrum.mzs <= xaxis_current + 0.5))] = yvals

            intensities_original_1bin = mean_spectrum.intensities[np.where(
                (mean_spectrum.mzs > xaxis_current - 0.5) & (mean_spectrum.mzs <= xaxis_current + 0.5))]

            background_subtraction = intensities_original_1bin - yvals

            intensities_new[np.where(
                (mean_spectrum.mzs > xaxis_current - 0.5) &
                (mean_spectrum.mzs <= xaxis_current + 0.5))] = background_subtraction

        return intensities_new, chemical_noise
