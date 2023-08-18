"""Class to correct from baseline and chemical noise.
It determines as well the noise and the signal above certain sigma"""

"""___Built-In Modules___"""
from nectar.data_formats.data_spectrum import DataSpectrum
from nectar.noise_correction.noise_determination import NoiseDetermination
from nectar.noise_correction.chemical_noise import ChemicalNoise
from nectar.noise_correction.background_noise import BackgroundNoise
from nectar.plotting.plotting import Plotting

"""___Third-Party Modules___"""
import numpy as np

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class NoiseCorrection:
    def __init__(self):
        self.noisedetermination = NoiseDetermination()
        self.chemicalnoise = ChemicalNoise()
        self.backgroundnoise = BackgroundNoise()
        self.plotting = Plotting()
        pass

    def noise_correction(self, mean_spectrum, plot_noise=False):
        """
        Function to correct baseline and determine noise level in the mean spectrum

        :param mean_spectrum: mean spectrum to determine the noise
        :type mean_spectrum: DataSpectrum
        :param plot_noise: option to plot the results
        :param plot_noise: Boolean
        :return: mean spectrum baseline corrected and S/N masked
        :rtype: DataSpectrum
        """

        print("Applying baseline correction to the mean spectrum...")
        mean_spectrum_baseline_corr = self.noisedetermination.baseline_correction(mean_spectrum, save_baseline=False)
        intensities_peaks = np.copy(mean_spectrum_baseline_corr.intensities)
        intensities_noise = np.copy(mean_spectrum_baseline_corr.intensities)

        print("Masking the noise...")
        (mask, noiseaveragearray2, noisestvedarray2) = self.noisedetermination.clipandmasklocal_ppmwindow_interpolate(
            mean_spectrum_baseline_corr.intensities, mean_spectrum_baseline_corr.mzs,
            sigma_threshold=3.0, channels=20000)

        intensities_peaks[np.where(mask == 0)] = np.nan  # To choose the peaks of the spectrum according with the S/N
        intensities_noise[np.where(mask == 1)] = np.nan
        mean_spectrum_baseline_corr.set_mzmask(mask)

        if plot_noise:
            self.plotting.plot_noise_spectrum(mean_spectrum_baseline_corr, intensities_peaks, intensities_noise)

        print('Mean spectrum correction done!')
        return mean_spectrum_baseline_corr

    def noise_correction_with_chemical_noise(self, mean_spectrum, plot_noise=False, plot_chemicalnoise=False):
        """
        Function to correct baseline, determine noise level and correct chemical noise in the mean spectrum

        :param mean_spectrum: mean spectrum to determine/correct noise
        :type mean_spectrum: DataSpectrum
        :param plot_noise: Option to plot the signal/noise level in the spectrum
        :type plot_noise: Boolean
        :param plot_chemicalnoise: Option to plot the corrected spectrum and the modelled chemical noise
        :type plot_chemicalnoise: Boolean
        :return: mean spectrum corrected
        :rtype: DataSpectrum
        """

        print("Applying baseline correction...")
        mean_spectrum_baseline_corr = self.noisedetermination.baseline_correction(mean_spectrum, save_baseline=False)

        intensities_peaks = np.copy(mean_spectrum_baseline_corr.intensities)
        intensities_noise = np.copy(mean_spectrum_baseline_corr.intensities)

        print("Masking the noise...")
        (mask, noiseaveragearray, noisestvedarray) = self.noisedetermination.clipandmasklocal_ppmwindow_interpolate(
            mean_spectrum_baseline_corr.intensities, mean_spectrum_baseline_corr.mzs, sigma_threshold=3, channels=20000)

        intensities_peaks[np.where(mask == 0)] = np.nan  # To choose the peaks of the spectrum according with the S/N
        intensities_noise[np.where(mask == 1)] = np.nan
        mean_spectrum_baseline_corr.set_mzmask(mask)

        if plot_noise:
            self.plotting.plot_noise_spectrum(mean_spectrum_baseline_corr, intensities_peaks, intensities_noise)

        # converts noise to a DataSpectrum object
        noise_spectrum = DataSpectrum(-99, [-99, -99], mean_spectrum_baseline_corr.mzs, intensities_noise)

        # chemical noise correction with 45% percentile subtraction
        noise_corrected_spectrum = self.chemicalnoise.chemical_noise_correction_45percentile(
            noise_spectrum, mean_spectrum_baseline_corr, plot_chemicalnoise=plot_chemicalnoise)

        print("Masking the noise after chemical background...")
        (mask2, noiseaveragearray2, noisestvedarray2) = self.noisedetermination.clipandmasklocal_ppmwindow_interpolate(
            noise_corrected_spectrum.intensities, noise_corrected_spectrum.mzs, sigma_threshold=3, channels=20000.0)

        noise_corrected_spectrum.set_mzmask(mask2)

        if plot_noise:
            # To plot the noise and the signal
            noise_corrected_spectrum_peaks = np.copy(noise_corrected_spectrum.intensities)
            noise_corrected_spectrum_noise = np.copy(noise_corrected_spectrum.intensities)
            # To choose the peaks of the spectrum according with the S/N
            noise_corrected_spectrum_peaks[np.where(mask2 == 0)] = np.nan
            noise_corrected_spectrum_noise[np.where(mask2 == 1)] = np.nan

            self.plotting.plot_noise_spectrum(noise_corrected_spectrum, noise_corrected_spectrum_peaks,
                                              noise_corrected_spectrum_noise)

        print('Mean spectrum correction done!')
        return noise_corrected_spectrum
