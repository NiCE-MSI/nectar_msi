"""Class to determine and mask the noise level on the data, as well as to determine the
scatter in the resolving power"""

"""___Built-In Modules___"""
from nectar.data_formats.data_spectrum import DataSpectrum
from nectar.plotting.plotting import Plotting

"""___Third-Party Modules___"""
import numpy as np
from scipy import interpolate

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class NoiseDetermination:
    def __init__(self):
        self.plotting = Plotting()
        pass

    def SigmaClip(self, values, tolerance=0.01, median=True, sigma_thresh=3.0, no_zeros=True):
        """
        Function to remove outliers in an array via iteration with a threshold value

        :param values: Values used to apply the function
        :type values: array
        :param tolerance: tolerance value to stop the iteration (default = 0.01)
        :type tolerance: float
        :param median: option to choose the mean or the median (default = median)
        :type median: bool
        :param sigma_thresh: sigma threshold value (default = 3)
        :type sigma_thresh: float
        :param no_zeros: if sigma_new is zero, returns the std of the original array
        :type no_zeros: bool
        :return: [sigma_new, average, values]
        :rtype: float, float, array
        """

        # Remove NaNs from input values
        values = np.array(values)
        values = values[np.where(np.isnan(values) == False)]
        values_original = np.copy(values)
        # Continue loop until result converges
        diff = 10e10
        while diff > tolerance:
            # Assess current input iteration
            if median == False:
                average = np.mean(values)
            elif median == True:
                average = np.median(values)
            sigma_old = np.std(values)

            # Mask those pixels that lie more than 3 stdv away from mean
            check = np.zeros([len(values)])
            check[np.where(values > (average + (sigma_thresh * sigma_old)))] = 1
            values = values[np.where(check < 1)]

            # Re-measure sigma and test for convergence
            sigma_new = np.std(values)
            diff = abs(sigma_old - sigma_new) / sigma_old

        # Perform final mask
        check = np.zeros([len(values)])
        check[np.where(values > (average + (sigma_thresh * sigma_old)))] = 1
        check[np.where(values < (average - (sigma_thresh * sigma_old)))] = 1
        values = values[np.where(check < 1)]

        # If required, check if calculated sigma is zero
        if no_zeros:
            if sigma_new == 0.0:
                sigma_new = np.std(values_original)
                if median == False:
                    average = np.mean(values)
                elif median == True:
                    average = np.median(values)
        return [sigma_new, average, values]

    def sigmaClip_stdv(self, width_corr, mz_corr, tolerance=0.01, sigma_thresh=3):
        """
        Function to define resolving power of the sample. Take all peaks and apply
        SigmaClipping to eliminate outliers from the correlation. (intercept not forced to zero)

        :param width_corr: Values on the y-axis (widths from the gaussian fitting)
        :type width_corr: array
        :param mz_corr: Values on the x-axis (centroids of the peaks)
        :type mz_corr: array
        :param tolerance: tolerance value to stop the iteration (default = 0.01)
        :type tolerance: float
        :param sigma_thresh: sigma threshold value (default = 3)
        :type sigma_thresh: float
        :return: func, distance_std, std_new, mz_corr, width_corr
        :rtype:
        """
        diff = 10e10
        while diff > tolerance:
            a = np.polyfit(mz_corr, width_corr, 1)
            func = np.poly1d(a)
            std_old = np.sqrt(np.sum(((width_corr) - func(mz_corr)) ** 2) / len(width_corr))
            distance_std = (width_corr - func(mz_corr)) / std_old
            mz_corr = mz_corr[np.where(abs(distance_std) < sigma_thresh)]
            width_corr = width_corr[np.where(abs(distance_std) < sigma_thresh)]
            a = np.polyfit(mz_corr, width_corr, 1)
            func = np.poly1d(a)
            std_new = np.sqrt(np.sum(((width_corr) - func(mz_corr)) ** 2) / len(width_corr))
            distance_std = (width_corr - func(mz_corr)) / std_old
            diff = abs(std_old - std_new) / std_old

        return func, distance_std, std_new, mz_corr, width_corr

    def sigmaClip_stdv_zero_intercept(self, width_corr, mz_corr, tolerance=0.01, sigma_thresh=3):
        """
        Function to define resolving power of the sample. Takes most intense 200 peaks and apply
         SigmaClipping to eliminate outliers from the correlation. (intercept of the correlation forced to zero)

        :param width_corr: Values on the y-axis (widths from the gaussian fitting)
        :type width_corr: array
        :param mz_corr: Values on the x-axis (centroids of the peaks)
        :type mz_corr: array
        :param tolerance: tolerance value to stop the iteration (default = 0.01)
        :type tolerance: float
        :param sigma_thresh: sigma threshold value (default = 3)
        :type sigma_thresh: float
        :return: func (correlation), distance_std, std_new, mz_corr (selected values), width_corr (selected values)
        :rtype: poly1d, float, float, array, array
        """

        diff = 10e10
        while diff > tolerance:
            mz_corrT = mz_corr[:, np.newaxis]
            a, _, _, _ = np.linalg.lstsq(mz_corrT, width_corr)
            func = np.poly1d([a[0], 0])
            std_old = np.sqrt(np.sum(((width_corr) - func(mz_corr)) ** 2) / len(width_corr))
            distance_std = (width_corr - func(mz_corr)) / std_old
            mz_corr = mz_corr[np.where(abs(distance_std) < sigma_thresh)]
            width_corr = width_corr[np.where(abs(distance_std) < sigma_thresh)]
            mz_corrT = mz_corr[:, np.newaxis]
            a, _, _, _ = np.linalg.lstsq(mz_corrT, width_corr)
            func = np.poly1d([a[0], 0])
            std_new = np.sqrt(np.sum(((width_corr) - func(mz_corr)) ** 2) / len(width_corr))
            distance_std = (width_corr - func(mz_corr)) / std_old
            diff = abs(std_old - std_new) / std_old

        return func, distance_std, std_new, mz_corr, width_corr

    def clipandmask(self, intensities):
        """
        Function to clip and mask an array with the SigmaClipping function

        :param intensities: values to apply the SigmaClipping function on
        :type intensities: array
        :return: mask for signal and noise, average noise, stv noise
        :rtype: array, float, float
        """

        # Create an empty array to determine the mask noise and signal mask
        flag = np.zeros(len(intensities))
        # Apply SigmaClipping
        noise = self.SigmaClip(intensities)  # outputs [sigma_new, average, values]
        noiseavg = noise[1]
        noisestd = noise[0]
        # Apply the mask
        for k in range(len(intensities)):
            if intensities[k] >= noiseavg + 3.0 * noisestd:
                flag[k] = 1
        return flag, noiseavg, noisestd  # flag is an array with ones for the peaks and zeros for the noise

    def clipandmasklocal_ppmwindow_interpolate(self, intensities, xaxis, sigma_threshold=3.0,
                                               channels=20000.0, interpolation_steps=250):
        """
        Function to apply SigmaClipping within a window that changes size with ppm, and interpolation to make it faster

        :param intensities: values to apply the SigmaClipping function on
        :type intensities: array
        :param xaxis: sliding window
        :type xaxis: array
        :param sigma_threshold: sigma threshold value (default = 3)
        :type sigma_threshold: float
        :param channels: number of channels in the window
        :type channels: float
        :param interpolation_steps: number of steps to do the interpolation
        :type interpolation_steps: int
        :return: mask for signal and noise, average noise, stv noise
        :rtype: array, float, float
        """

        # Sigma clipping function local. Sliding window of 20000bins, with interpolation to make it faster.
        flag = np.zeros(len(intensities))
        noiseavgarrayint = np.zeros(int(len(intensities) / interpolation_steps))
        noisestdarrayint = np.zeros(int(len(intensities) / interpolation_steps))
        xarrayint = np.zeros(int(len(intensities) / interpolation_steps))
        counti = 0

        for k in range(500, len(intensities) - 500, interpolation_steps):
            xarrayint[counti] = xaxis[k]
            noise = self.SigmaClip(intensities[np.where(abs(xaxis - xaxis[k]) < channels * xaxis[k] / 1e6)],
                                   sigma_thresh=sigma_threshold)
            noiseavgarrayint[counti] = noise[1]  # Local noise - average
            noisestdarrayint[counti] = noise[0]  # Local stdv - sigma new
            counti += 1
        favg = interpolate.interp1d(xarrayint, noiseavgarrayint, bounds_error=False)
        fstd = interpolate.interp1d(xarrayint, noisestdarrayint, bounds_error=False)
        noiseavgarray = favg(xaxis)
        noisestdarray = fstd(xaxis)
        flag[np.where(intensities > noiseavgarray + 3.0 * noisestdarray)] = 1

        return flag, noiseavgarray, noisestdarray

    def local_noise_around_peak(self, spectrum, peak, window_size=30):
        """
        Function to calculate the S/N around a peak in a given window size (Dalton)

        :param spectrum: spectrum
        :type spectrum: DataSpectrum
        :param peak: m/z centroid value
        :type peak: float
        :param window_size: size of the window around the peak to determine the local noise
        :type window_size: float
        :return: local noise around the peak in +-30Da
        :rtype: float
        """

        # xaxis_window = spectrum.mzs[np.where((spectrum.mzs > peak - window_size)
        #                                      & (spectrum.mzs < peak + window_size))]

        intensities_window = spectrum.intensities[np.where((spectrum.mzs > peak - window_size)
                                                           & (spectrum.mzs < peak + window_size))]

        mask, noiseaveragearray, local_noise = self.clipandmask(intensities_window)

        return local_noise

    def baseline_correction(self, mean_spectrum, save_baseline=False):
        """
        Function to apply baseline correction to the mean spectrum

        :param mean_spectrum: mean spectrum without baseline correction
        :type mean_spectrum: DataSpectrum
        :param save_baseline: option to save the baseline as a DataSpectrum
        :type save_baseline: Boolean
        :return mean_spectrum_baseline_corr: baseline corrected spectrum
        :rtype mean_spectrum_baseline_corr: DataSpectrum
        """

        # Baseline correction: noiseavgarray is the average noise array, and noisestdarray is the sdtve noise array.
        # Default: Both are calculated by interpolating in a window of 250
        (flag, noiseavgarray, noisestdarray) = self.clipandmasklocal_ppmwindow_interpolate(
            mean_spectrum.intensities, mean_spectrum.mzs, sigma_threshold=2.0, channels=20000)

        mean_spectrum_baseline_corr = DataSpectrum(-99, [-99, -99], mean_spectrum.mzs, mean_spectrum.intensities)
        mean_spectrum_baseline_corr.intensities = (mean_spectrum.intensities - noiseavgarray)

        if save_baseline:
            baseline = DataSpectrum(-99, [-99, -99], mean_spectrum.mzs, noiseavgarray)
            return mean_spectrum_baseline_corr, baseline
        else:
            return mean_spectrum_baseline_corr

    def clipandmasklocal_daltonwindow_interpolate(self, intensities, xaxis, sigma_threshold=3.0, window=30.0,
                                                  interpolation_steps=1):
        """
        Function to apply SigmaClipping within a given window size (Dalton). Interpolation to make it faster

        :param intensities:  values to apply the SigmaClipping function on
        :type intensities: array
        :param xaxis: sliding window
        :type xaxis: array
        :param sigma_threshold: sigma threshold value (default = 3)
        :type sigma_threshold: float
        :param window: window size in Dalton
        :type window: float
        :param interpolation_steps: number of steps to do the interpolation
        :type interpolation_steps: int
        :return: mask for signal and noise, average noise, stv noise
        :rtype: array, float, float
        """
        # Sigma clipping function local. Sliding window of 20000bins, with interpolation to make it faster.
        peaks = range(int(np.ceil(xaxis[0])), int(np.ceil(xaxis[-1])), interpolation_steps)
        flag = np.zeros(len(intensities))
        noiseavgarrayint = np.zeros(len(peaks))
        noisestdarrayint = np.zeros(len(peaks))
        counti = 0
        for peak in peaks:
            intensities_window = intensities[np.where((xaxis > peak - window) & (xaxis < peak + window))]
            noise = self.SigmaClip(intensities_window, sigma_thresh=sigma_threshold)
            noiseavgarrayint[counti] = noise[1]  # Local noise - average
            noisestdarrayint[counti] = noise[0]  # Local stdv - sigma new
            counti += 1
        favg = interpolate.interp1d(peaks, noiseavgarrayint, bounds_error=False)
        fstd = interpolate.interp1d(peaks, noisestdarrayint, bounds_error=False)
        noiseavgarray = favg(xaxis)
        noisestdarray = fstd(xaxis)
        flag[np.where(intensities > noiseavgarray + 3.0 * noisestdarray)] = 1

        return flag, noiseavgarray, noisestdarray
