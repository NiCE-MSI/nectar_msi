"""Class for Gaussian fitting of peaks"""

"""___Built-In Modules___"""
from nectar_msi.plotting.plotting import Plotting
from nectar_msi.gaussian_fitting.gaussian_fitting_definitions import GaussianFittingDefinitions
from nectar_msi.data_processing.data_operations import DataOperations
from nectar_msi.noise_correction.noise_determination import NoiseDetermination

"""___Third-Party Modules___"""
import numpy as np
import os

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class GaussianFitting:
    def __init__(self):
        self.plotting = Plotting()
        self.gaussian_def = GaussianFittingDefinitions()
        self.dataop = DataOperations()
        self.noisedetermination = NoiseDetermination()
        pass

    def gaussian_fitting_spectrum(self, mean_spectrum, list_of_peaks, path_outputs, save_fitting=False):
        """
        Function to fit a Gaussian by chi2 minimization. The function saves plots of Gaussian fitting.

        :param mean_spectrum: mean spectrum to take as reference to fit the Gaussian
        :type mean_spectrum: DataSpectrum
        :param list_of_peaks: list of peaks of interest (list created in peak_picking.py)
        :type list_of_peaks: list
        :param path_outputs: outputs path
        :type path_outputs: string
        :param save_fitting: option to save plots of the Gaussian fitting (default = False)
        :type save_fitting: bool
        :return: table with the list of peaks and its characteristics
        (centroid, intensity, width, chi2/Nsamples, Number of sample bins, type of peak,
        success in the fitting)
        :rtype: array
        """

        if save_fitting:
            '''It creates a file where it is going to save all the relevant values of the fitting'''
            dirName = path_outputs + "Gaussian_fitting_most_intense_peaks"
            if not os.path.exists(dirName):
                os.mkdir(dirName)

        # It selects the part of the mean spectrum that has been identified as signal previously
        mask = mean_spectrum.mzmask
        list_of_fittings = np.empty((0, 7))
        # centroid   peak_intensity  width   chi2/Nsamples   Nsamples    Peak_type   Fitting_success
        # 0         1               2       3               4           5           6
        for i in range(2, len(mean_spectrum.intensities) - 2):
            # It selects windows of signal to determine how many peaks are on them
            if mask[i] == 0.0 and mask[i + 1] == 1.0:
                istart = i
            if mask[i] == 1.0 and mask[i + 1] == 0.0:
                spectrum_window2 = mean_spectrum.intensities[istart: i + 2]
                xaxis_window2 = mean_spectrum.mzs[istart: i + 2]

                local_max = []
                # It finds the number of local max in the window to determine how many
                # peaks are in the window.
                for ii in range(len(list_of_peaks)):
                    if max(xaxis_window2) > list_of_peaks[ii] > min(xaxis_window2):
                        local_max = np.append(local_max, list_of_peaks[ii])

                '''Fits one peak'''
                if len(local_max) == 1:
                    res = self.gaussian_def.gaussian_fit_one_element(xaxis_window2, spectrum_window2)
                    if save_fitting:
                        self.plotting.plot_one_gaussian(xaxis_window2, spectrum_window2, res, path_outputs)

                    list_of_fittings = np.append(list_of_fittings,
                                                 np.array([[float(res.x[0]), res.x[1],
                                                            abs(res.x[2]), (self.gaussian_def.cal_chi2
                                                                            (res.x, spectrum_window2, xaxis_window2) /
                                                                            res.x[1]/len(spectrum_window2)),
                                                            len(spectrum_window2), "single", res.success]]), axis=0)

                # If there are two peaks in the window, it determines if they are too
                # close to each other. It they are 15ppm apart, it fits each peak
                # as independent peaks. If not it fits them together as there might be
                # contamination between them.

                if len(local_max) == 2:
                    if abs(local_max[1] - local_max[0]) >= 15:
                        '''Fits two separated peaks'''
                        for j in range(0, 2, 1):
                            peak_max = spectrum_window2[np.where(local_max[j] == xaxis_window2)]
                            xaxis2_max = local_max[j]

                            res = self.gaussian_def.gaussian_fit_one_elements_separated(
                                xaxis_window2, spectrum_window2, xaxis2_max, peak_max)
                            if save_fitting:
                                self.plotting.plot_one_gaussian_separated(xaxis_window2, spectrum_window2,
                                                                         xaxis2_max, peak_max, res, path_outputs)
                            '''Add parameters to final table'''
                            list_of_fittings = np.append(list_of_fittings,
                                                         np.array([[float(res.x[0]), res.x[1], abs(res.x[2]),
                                                                    self.gaussian_def.cal_chi2(
                                                                        res.x, spectrum_window2, xaxis_window2)
                                                                    / res.x[1] / len(spectrum_window2),
                                                                    len(spectrum_window2), "single", res.success,]]),
                                                         axis=0)
                    else:
                        '''Fits two peaks together'''
                        res = self.gaussian_def.gaussian_fit_two_elements(
                            xaxis_window2, spectrum_window2,
                            local_max[0], spectrum_window2[np.where(local_max[0] == xaxis_window2)[0][0]],
                            local_max[1], spectrum_window2[np.where(local_max[1] == xaxis_window2)[0][0]])
                        if save_fitting:
                            self.plotting.plot_two_gaussians(xaxis_window2, spectrum_window2, local_max[0],
                                                             spectrum_window2[np.where(
                                                                 local_max[0] == xaxis_window2)[0][0]], local_max[1],
                                                             spectrum_window2[np.where(
                                                                 local_max[1] == xaxis_window2)[0][0]],
                                                             res, path_outputs)
                        '''Add parameters to final table'''
                        list_of_fittings = np.append(
                            list_of_fittings, np.array([[
                                        float(res.x[0]), res.x[1], abs(res.x[2]),
                                        self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window2, xaxis_window2)
                                        / res.x[1] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                res.success]]), axis=0)

                        list_of_fittings = np.append(
                            list_of_fittings, np.array([[
                                        float(res.x[3]), res.x[4], abs(res.x[5]),
                                        self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window2, xaxis_window2)
                                        / res.x[4] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                res.success]]), axis=0)

                # If there are tree peaks in the window, it determines the distance among
                # them and fit them accordingly.
                if len(local_max) == 3:
                    if len(local_max) == 3:
                        '''Fits three peaks together'''
                        if np.abs(local_max[1] - local_max[0]) <= 15 and np.abs(local_max[2] - local_max[1]) <= 15:
                            res = self.gaussian_def.gaussian_fit_three_elements(
                                xaxis_window2, spectrum_window2, local_max[0],
                                spectrum_window2[np.where(local_max[0] == xaxis_window2)[0][0]], local_max[1],
                                spectrum_window2[np.where(local_max[1] == xaxis_window2)[0][0]], local_max[2],
                                spectrum_window2[np.where(local_max[2] == xaxis_window2)[0][0]])

                            if save_fitting:
                                self.plotting.plot_three_gausssians(
                                    xaxis_window2, spectrum_window2, local_max[0],
                                    spectrum_window2[np.where(local_max[0] == xaxis_window2)[0][0]],
                                    local_max[1], spectrum_window2[np.where(local_max[1] == xaxis_window2)[0][0]],
                                    local_max[2], spectrum_window2[np.where(local_max[2] == xaxis_window2)[0][0]],
                                    res, path_outputs)
                            '''Add parameters to final table'''
                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                            float(res.x[0]), res.x[1], abs(res.x[2]),
                                            self.gaussian_def.cal_chi2_multigaussian_3g(
                                                res.x, spectrum_window2, xaxis_window2)
                                            / res.x[1] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                            res.success]]), axis=0)

                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                            float(res.x[3]), res.x[4], abs(res.x[5]),
                                            self.gaussian_def.cal_chi2_multigaussian_3g(
                                                res.x, spectrum_window2, xaxis_window2)
                                            / res.x[4] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                            res.success]]), axis=0)

                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                            float(res.x[6]), res.x[7], abs(res.x[8]),
                                            self.gaussian_def.cal_chi2_multigaussian_3g(
                                                res.x, spectrum_window2, xaxis_window2)
                                            / res.x[7] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                            res.success]]), axis=0)

                        '''Fits two peaks together and one separated'''
                        if np.abs(local_max[1] - local_max[0]) <= 15 and np.abs(local_max[2] - local_max[1]) > 15:
                            res = self.gaussian_def.gaussian_fit_two_elements(
                                xaxis_window2, spectrum_window2, local_max[0],
                                spectrum_window2[np.where(local_max[0] == xaxis_window2)[0][0]], local_max[1],
                                spectrum_window2[np.where(local_max[1] == xaxis_window2)[0][0]])
                            if save_fitting:
                                self.plotting.plot_two_gaussians(
                                    xaxis_window2, spectrum_window2, local_max[0],
                                    spectrum_window2[np.where(local_max[0] == xaxis_window2)[0][0]],
                                    local_max[1], spectrum_window2[np.where(local_max[1] == xaxis_window2)[0][0]],
                                    res, path_outputs)

                            '''Add parameters to final table'''
                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                    float(res.x[0]), res.x[1], abs(res.x[2]),
                                    self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window2, xaxis_window2)
                                    / res.x[1] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                    res.success]]), axis=0)

                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                    float(res.x[3]), res.x[4], abs(res.x[5]),
                                    self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window2, xaxis_window2)
                                    / res.x[4] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                    res.success]]), axis=0)
                            '''Separated peak'''
                            res = self.gaussian_def.gaussian_fit_one_element_separated(
                                xaxis_window2, spectrum_window2, local_max[2],
                                spectrum_window2[np.where(local_max[2] == xaxis_window2)[0][0]])
                            if save_fitting:
                                self.plotting.plot_one_gaussian_separated(
                                    xaxis_window2, spectrum_window2, local_max[2], local_max[2], res, path_outputs)

                            '''Add parameters to final table'''
                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                    float(res.x[0]), res.x[1], abs(res.x[2]),
                                    self.gaussian_def.cal_chi2(res.x, spectrum_window2, xaxis_window2)
                                    / res.x[1] / len(spectrum_window2), len(spectrum_window2), "separated",
                                    res.success]]), axis=0)

                        # The last two peaks are together, they are fitted together. the first one is separated.
                        if np.abs(local_max[2] - local_max[1]) <= 15 and np.abs(local_max[1] - local_max[0]) > 15:
                            res = self.gaussian_def.gaussian_fit_two_elements(
                                xaxis_window2, spectrum_window2,
                                local_max[1], spectrum_window2[np.where(local_max[1] == xaxis_window2)[0][0]],
                                local_max[2], spectrum_window2[np.where(local_max[2] == xaxis_window2)[0][0]])

                            if save_fitting:
                                self.plotting.plot_two_gaussians(
                                    xaxis_window2, spectrum_window2,
                                    local_max[1], spectrum_window2[np.where(local_max[1] == xaxis_window2)[0][0]],
                                    local_max[2], spectrum_window2[np.where(local_max[2] == xaxis_window2)],
                                    res, path_outputs)

                            '''Add parameters to final table'''
                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                    float(res.x[0]), res.x[1], abs(res.x[2]),
                                    self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window2, xaxis_window2)
                                    / res.x[1] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                    res.success]]), axis=0)

                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                    float(res.x[3]), res.x[4], abs(res.x[5]),
                                    self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window2, xaxis_window2)
                                    / res.x[4] / len(spectrum_window2), len(spectrum_window2), "multiple",
                                    res.success]]), axis=0)

                            '''separated peak'''
                            res = self.gaussian_def.gaussian_fit_one_element_separated(
                                xaxis_window2, spectrum_window2,
                                local_max[0], spectrum_window2[np.where(local_max[0] == xaxis_window2)[0][0]])

                            if save_fitting:
                                self.plotting.plot_one_gaussian_separated(
                                    xaxis_window2, spectrum_window2,
                                    local_max[0], spectrum_window2[np.where(local_max[0] == xaxis_window2)[0][0]],
                                    res, path_outputs)

                            '''Add parameters to final table'''
                            list_of_fittings = np.append(
                                list_of_fittings, np.array([[
                                    float(res.x[0]), res.x[1], abs(res.x[2]),
                                    self.gaussian_def.cal_chi2(res.x, spectrum_window2, xaxis_window2)
                                    / res.x[1] / len(spectrum_window2), len(spectrum_window2), "separated",
                                    res.success]]), axis=0)

                        # The three peaks are independent and therefore fitted independently
                        if np.abs(local_max[1] - local_max[0]) >= 15 and np.abs(local_max[2] - local_max[1]) > 15:
                            for j in range(0, 3, 1):
                                peak_max = spectrum_window2[np.where(local_max[j] == xaxis_window2)]
                                xaxis_max = local_max[j]

                                res = self.gaussian_def.gaussian_fit_one_element_separated(
                                    xaxis_window2, spectrum_window2, xaxis_max, peak_max)

                                if save_fitting:
                                    self.plotting.plot_one_gaussian_separated(
                                        xaxis_window2, spectrum_window2, xaxis_max, peak_max, res, path_outputs)

                                '''Add parameters to final table'''
                                list_of_fittings = np.append(
                                    list_of_fittings, np.array([[
                                        float(res.x[0]), res.x[1], abs(res.x[2]),
                                        self.gaussian_def.cal_chi2(res.x, spectrum_window2, xaxis_window2)
                                        / res.x[1] / len(spectrum_window2), len(spectrum_window2), "separated",
                                        res.success]]), axis=0)
        return list_of_fittings

    def gaussian_fitting_spectrum_constrained(self, mean_spectrum, list_of_peaks, func, std, path_outputs,
                                              save_fitting=False):
        """
        Function for Gaussian fitting by chi2 minimization.
        The fitting constrains the Gaussian in centroid, intensity and width values.

        :param mean_spectrum: mean spectrum to take as reference to fit the Gaussian
        :type mean_spectrum: DataSpectrum
        :param list_of_peaks: list of peaks of interest (list created in peak_picking.py)
        :type list_of_peaks: list
        :param func: correlation function of the resolving power for the dataset
        :type func: poly1d
        :param std: intercept of the resolving power correlation for the dataset
        :type std: float
        :param path_outputs: outputs path
        :type path_outputs: string
        :param save_fitting: option to save plots of the Gaussian fitting (default = False)
        :type save_fitting: bool
        :return: table with the list of peaks and its characteristics
        (centroid, intensity, width, chi2/Nsamples, Number of sample bins, type of peak,
        success in the fitting, good peak shape)
        :rtype: array
        """

        if save_fitting:
            dirName = path_outputs + "Gaussian_fitting_width_constrained"
            if not os.path.exists(dirName):
                os.mkdir(dirName)

        mask = mean_spectrum.mzmask
        full_list_of_fittings = np.empty((0, 12))
        # 0              1          2               3                    4         5
        # left_min     meas mz   right_min   centroid gaussian    peak_intensity  width
        #     6              7            8       9      10                   11
        # chi2/Nsamples   Nsamples    Peak_type   S/N   Fitting_success    PeakShape"

        # It selects windows of signal to determine how many peaks are on them
        for i in range(2, len(mean_spectrum.intensities) - 2):
            if mask[i] == 0.0 and mask[i + 1] == 1.0:
                istart = i
            if mask[i] == 1.0 and mask[i + 1] == 0.0:
                spectrum_window = mean_spectrum.intensities[istart: i + 2]
                xaxis_window = mean_spectrum.mzs[istart: i + 2]

                local_maximum = []
                # It finds the number of local max in the window to determine how many
                # peaks are in the window.
                for ii in range(len(list_of_peaks)):
                    if max(xaxis_window) > list_of_peaks[ii] > min(xaxis_window):
                        local_maximum = np.append(local_maximum, list_of_peaks[ii])

                # If there is only one peak in the window it makes a single gaussian fitting
                '''Fits one peak'''
                if len(local_maximum) == 1:
                    xaxis_window_plot = []
                    spectrum_window_plot = []
                    centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum))
                    for kkk in range((centroid - 50), (centroid + 50), 1):
                        xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                        spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])
                    res = self.gaussian_def.gaussian_fit_one_element_width_constraint(
                        xaxis_window_plot, spectrum_window_plot, local_maximum[0],
                        spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]], func, std)
                    if save_fitting:
                        self.plotting.plot_one_gaussian_width_constraint(
                            xaxis_window_plot, spectrum_window_plot, local_maximum[0], res, path_outputs)

                    '''peak shape peak1'''
                    if (abs(res.x[2]) == func(local_maximum[0]) + 5 * std) or \
                            (abs(res.x[2]) == func(local_maximum[0]) - 5 * std):
                        flag = "bad"
                    else:
                        flag = "good"
                    '''Finds minima and local noise of peak1'''
                    left_min, right_min = self.dataop.find_minima_around_peak(
                        xaxis_window_plot, spectrum_window_plot, local_maximum[0])
                    # It estimates the local noise around the peak in a window of 20 Da
                    # to each size of the peak centroid
                    local_noise = self.noisedetermination.local_noise_around_peak(mean_spectrum, local_maximum[0],
                                                                                  window_size=20)
                    # It estimates the S/N as the peak intensity over the local noise
                    snr = res.x[1] / local_noise
                    '''Add parameters to final table'''
                    full_list_of_fittings = np.append(
                        full_list_of_fittings, np.array([[
                            left_min, local_maximum[0], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                            self.gaussian_def.cal_chi2(res.x, spectrum_window, xaxis_window) / res.x[1]
                            / len(spectrum_window), len(spectrum_window), "single", snr, res.success, flag]]), axis=0)

                # If there are two peaks in the window, it determines if they are too
                # close to each other. It they are 15ppm apart, it fits each peak
                # as independent peaks. If not it fits them together as there might be
                # contamination between them.

                '''Fits two peaks'''
                if len(local_maximum) == 2:
                    if np.abs(local_maximum[1] - local_maximum[0]) >= 6 * func(local_maximum[0]):
                        '''Separated fit'''
                        for j in range(0, 2, 1):
                            peak_max = spectrum_window[np.where(local_maximum[j] == xaxis_window)[0][0]]
                            centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[j]))

                            xaxis_window_plot = []
                            spectrum_window_plot = []

                            for kkk in range((centroid - 50), (centroid + 50), 1):
                                xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                                spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])

                            res = self.gaussian_def.gaussian_fit_one_element_width_constraint(
                                xaxis_window_plot, spectrum_window_plot, local_maximum[j], peak_max, func, std)
                            if save_fitting:
                                self.plotting.plot_one_gaussian_separated_width_constraint(
                                    xaxis_window_plot, spectrum_window_plot, local_maximum[j], peak_max, res, func,
                                    std, path_outputs)

                            '''peak shape peak_j'''
                            if (abs(res.x[2]) == func(local_maximum[j]) + 5 * std) or \
                                    (abs(res.x[2]) == func(local_maximum[j]) - 5 * std):
                                flag = "bad"
                            else:
                                flag = "good"
                            '''Find minima and local noise around peak_j'''
                            left_min, right_min = self.dataop.find_minima_around_peak(
                                xaxis_window_plot, spectrum_window_plot, local_maximum[j])
                            local_noise = self.noisedetermination.local_noise_around_peak(
                                mean_spectrum, local_maximum[j], window_size=20)
                            snr = res.x[1] / local_noise
                            '''Add parameters to final table peak_j'''
                            full_list_of_fittings = np.append(
                                full_list_of_fittings, np.array([[
                                    left_min, local_maximum[j], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                                    self.gaussian_def.cal_chi2(res.x, spectrum_window, xaxis_window) / res.x[1]
                                    / len(spectrum_window), len(spectrum_window), "single", snr, res.success, flag]]),
                                axis=0)

                    else:  # If they are close to each other, it fits them together.
                        '''Two fits together'''
                        xaxis_window_plot = []
                        spectrum_window_plot = []
                        centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[0]))

                        for kkk in range((centroid - 150), (centroid + 150), 1):
                            xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                            spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])

                        res = self.gaussian_def.gaussian_fit_two_elements_width_constraint(
                            xaxis_window_plot, spectrum_window_plot,
                            local_maximum[0], spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]],
                            local_maximum[1], spectrum_window[np.where(local_maximum[1] == xaxis_window)[0][0]],
                            func, std)

                        if save_fitting:
                            self.plotting.plot_two_gaussians_width_constraint(
                                xaxis_window_plot, spectrum_window_plot,
                                local_maximum[0], spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]],
                                local_maximum[1], spectrum_window[np.where(local_maximum[1] == xaxis_window)[0][0]],
                                func, std, res, path_outputs)

                        '''peak shape peak1'''
                        if (abs(res.x[2]) == func(local_maximum[0]) + 5 * std) or \
                                (abs(res.x[2]) == func(local_maximum[0]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak1'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot, spectrum_window_plot, local_maximum[0])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[0], window_size=20)
                        snr = res.x[1] / local_noise
                        '''Add parameters to final table peak1'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[0], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                                self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window, xaxis_window)
                                / res.x[1] / len(spectrum_window), len(spectrum_window), "multiple", snr,
                                res.success, flag]]), axis=0)

                        '''peak shape peak2'''
                        if (abs(res.x[5]) == func(local_maximum[1]) + 5 * std) or \
                                (abs(res.x[5]) == func(local_maximum[1]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak2'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot, spectrum_window_plot, local_maximum[1])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[1], window_size=20)
                        snr = res.x[4] / local_noise
                        '''Add parameters to final table peak2'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[1], right_min, float(res.x[3]), res.x[4], abs(res.x[5]),
                                self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window, xaxis_window)
                                / res.x[4] / len(spectrum_window), len(spectrum_window), "multiple", snr,
                                res.success, flag]]), axis=0)

                # If there are tree peaks in the window, it determines the distance among
                # them and fit them accordingly.
                '''Fits three peaks'''
                if len(local_maximum) == 3:
                    # The three peaks are close to each other, they are fitted together
                    '''Three peaks together'''
                    if np.abs(local_maximum[1] - local_maximum[0]) <= 6 * func(local_maximum[0]) \
                            and np.abs(local_maximum[2] - local_maximum[1]) <= 6 * func(local_maximum[1]):

                        xaxis_window_plot = []
                        spectrum_window_plot = []
                        centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[1]))

                        for kkk in range((centroid - 150), (centroid + 150), 1):
                            xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                            spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])

                        res = self.gaussian_def.gaussian_fit_three_elements_width_constraint(
                            xaxis_window_plot, spectrum_window_plot,
                            local_maximum[0], spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]],
                            local_maximum[1], spectrum_window[np.where(local_maximum[1] == xaxis_window)[0][0]],
                            local_maximum[2], spectrum_window[np.where(local_maximum[2] == xaxis_window)[0][0]],
                            func, std)

                        if save_fitting:
                            self.plotting.plot_three_gausssians_width_constraint(
                                xaxis_window_plot, spectrum_window_plot,
                                local_maximum[0], spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]],
                                local_maximum[1], spectrum_window[np.where(local_maximum[1] == xaxis_window)[0][0]],
                                local_maximum[2], spectrum_window[np.where(local_maximum[2] == xaxis_window)[0][0]],
                                func, std, res, path_outputs)

                        '''peak shape peak1'''
                        if (abs(res.x[2]) == func(local_maximum[0]) + 5 * std) or \
                                (abs(res.x[2]) == func(local_maximum[0]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak1'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot, spectrum_window_plot, local_maximum[0])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[0], window_size=20)
                        snr = res.x[1] / local_noise
                        '''Add parameters to final table peak1'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[0], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                                self.gaussian_def.cal_chi2_multigaussian_3g(res.x, spectrum_window, xaxis_window)
                                / res.x[1] / len(spectrum_window), len(spectrum_window), "multiple",
                                snr, res.success, flag]]), axis=0)

                        '''peak shape peak2'''
                        if (abs(res.x[5]) == func(local_maximum[1]) + 5 * std) or \
                                (abs(res.x[5]) == func(local_maximum[1]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak2'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot, spectrum_window_plot, local_maximum[1])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[1], window_size=20)
                        snr = res.x[4] / local_noise
                        '''Add parameters to final table peak2'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[1], right_min, float(res.x[3]), res.x[4], abs(res.x[5]),
                                self.gaussian_def.cal_chi2_multigaussian_3g(res.x, spectrum_window, xaxis_window)
                                / res.x[4] / len(spectrum_window), len(spectrum_window), "multiple",
                                snr, res.success, flag]]), axis=0)

                        '''peak shape peak3'''
                        if (abs(res.x[8]) == func(local_maximum[2]) + 5 * std) or \
                                (abs(res.x[8]) == func(local_maximum[2]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak3'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot, spectrum_window_plot, local_maximum[2])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[2], window_size=20)
                        snr = res.x[7] / local_noise
                        '''Add parameters to final table peak3'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[2], right_min, float(res.x[6]), res.x[7], abs(res.x[8]),
                                self.gaussian_def.cal_chi2_multigaussian_3g(res.x, spectrum_window, xaxis_window)
                                / res.x[7] / len(spectrum_window), len(spectrum_window), "multiple",
                                snr, res.success, flag]]), axis=0)

                    '''Two peaks together + separated'''
                    # The first two peaks are together, they are fitted together. the last one is separated.
                    if np.abs(local_maximum[1] - local_maximum[0]) <= 6 * func(local_maximum[0]) \
                            and np.abs(local_maximum[2] - local_maximum[1]) > 6 * func(local_maximum[1]):

                        xaxis_window_plot_2 = []
                        spectrum_window_plot_2 = []
                        centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[1]))

                        for kkk in range((centroid - 150), (centroid + 150), 1):
                            xaxis_window_plot_2 = np.append(xaxis_window_plot_2, mean_spectrum.mzs[kkk])
                            spectrum_window_plot_2 = np.append(spectrum_window_plot_2, mean_spectrum.intensities[kkk])

                        res = self.gaussian_def.gaussian_fit_two_elements_width_constraint(
                            xaxis_window_plot_2, spectrum_window_plot_2,
                            local_maximum[0], spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]],
                            local_maximum[1], spectrum_window[np.where(local_maximum[1] == xaxis_window)[0][0]],
                            func, std)

                        if save_fitting:
                            self.plotting.plot_two_gaussians_width_constraint(
                                xaxis_window_plot_2, spectrum_window_plot_2,
                                local_maximum[0], spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]],
                                local_maximum[1], spectrum_window[np.where(local_maximum[1] == xaxis_window)[0][0]],
                                func, std, res, path_outputs)

                        '''peak shape peak1'''
                        if (abs(res.x[2]) == func(local_maximum[0]) + 5 * std) or \
                                (abs(res.x[2]) == func(local_maximum[0]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak1'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot_2, spectrum_window_plot_2, local_maximum[0])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[0], window_size=20)
                        snr = res.x[1] / local_noise
                        '''Add parameters to final table peak1'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[0], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                                self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window, xaxis_window)
                                / res.x[1] / len(spectrum_window), len(spectrum_window), "multiple",
                                snr, res.success, flag]]), axis=0)

                        '''peak shape peak2'''
                        if (abs(res.x[5]) == func(local_maximum[1]) + 5 * std) or \
                                (abs(res.x[5]) == func(local_maximum[1]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak2'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot_2, spectrum_window_plot_2, local_maximum[1])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[1], window_size=20)
                        snr = res.x[4] / local_noise
                        '''Add parameters to final table peak2'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[1], right_min, float(res.x[3]), res.x[4], abs(res.x[5]),
                                self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window, xaxis_window)
                                / res.x[4] / len(spectrum_window), len(spectrum_window), "multiple",
                                snr, res.success, flag]]), axis=0)

                        '''Separated fitting'''
                        xaxis_window_plot = []
                        spectrum_window_plot = []
                        centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[2]))

                        for kkk in range((centroid - 50), (centroid + 50), 1):
                            xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                            spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])

                        res = (self.gaussian_def.gaussian_fit_one_element_width_constraint(
                            xaxis_window_plot, spectrum_window_plot,
                            local_maximum[2], spectrum_window[np.where(local_maximum[2] == xaxis_window)[0][0]],
                            func, std))

                        if save_fitting:
                            self.plotting.plot_one_gaussian_separated_width_constraint(
                                xaxis_window_plot, spectrum_window_plot,
                                local_maximum[2], spectrum_window[np.where(local_maximum[2] == xaxis_window)[0][0]],
                                res, func, std, path_outputs)

                        '''peak shape peak3'''
                        if (abs(res.x[2]) == func(local_maximum[2]) + 5 * std) or \
                                (abs(res.x[2]) == func(local_maximum[2]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak3'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot_2, spectrum_window_plot_2, local_maximum[2])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[2], window_size=20)
                        snr = res.x[1] / local_noise
                        '''Add parameters to final table peak3'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[2], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                                self.gaussian_def.cal_chi2(res.x, spectrum_window, xaxis_window)
                                / res.x[1] / len(spectrum_window), len(spectrum_window), "separated",
                                snr, res.success, flag]]), axis=0)

                    '''Separated + two peaks together'''
                    # The last two peaks are together, they are fitted together. the first one is separated.
                    if np.abs(local_maximum[2] - local_maximum[1]) <= 6 * func(local_maximum[1])\
                            and np.abs(local_maximum[1] - local_maximum[0]) > 6 * func(local_maximum[0]):
                        '''Two gaussian fit'''
                        xaxis_window_plot_2 = []
                        spectrum_window_plot_2 = []
                        centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[1]))

                        for kkk in range((centroid - 50), (centroid + 50), 1):
                            xaxis_window_plot_2 = np.append(xaxis_window_plot_2, mean_spectrum.mzs[kkk])
                            spectrum_window_plot_2 = np.append(spectrum_window_plot_2, mean_spectrum.intensities[kkk])

                        res = self.gaussian_def.gaussian_fit_two_elements_width_constraint(
                            xaxis_window_plot_2, spectrum_window_plot_2,
                            local_maximum[1], spectrum_window[np.where(local_maximum[1] == xaxis_window)[0][0]],
                            local_maximum[2], spectrum_window[np.where(local_maximum[2] == xaxis_window)[0][0]],
                            func, std)

                        if save_fitting:
                            self.plotting.plot_two_gaussians_width_constraint(
                                xaxis_window_plot_2, spectrum_window_plot_2,
                                local_maximum[1], spectrum_window[np.where(local_maximum[1] == xaxis_window)[0][0]],
                                local_maximum[2], spectrum_window[np.where(local_maximum[2] == xaxis_window)[0][0]],
                                func, std, res, path_outputs)

                        '''peak shape peak2'''
                        if (abs(res.x[2]) == func(local_maximum[1]) + 3 * std) \
                                or (abs(res.x[2]) == func(local_maximum[1]) - 3 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak2'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot_2, spectrum_window_plot_2, local_maximum[1])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[1], window_size=20)
                        snr = res.x[1] / local_noise
                        '''Add parameters to final table peak2'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[1], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                                self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window, xaxis_window)
                                / res.x[1] / len(spectrum_window), len(spectrum_window), "multiple",
                                snr, res.success, flag]]), axis=0)

                        '''peak shape peak3'''
                        if (abs(res.x[5]) == func(local_maximum[2]) + 5 * std) or \
                                (abs(res.x[5]) == func(local_maximum[2]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak3'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot_2, spectrum_window_plot_2, local_maximum[2])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[2], window_size=20)
                        snr = res.x[4] / local_noise
                        '''Add parameters to final table peak3'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[2], right_min, float(res.x[3]), res.x[4], abs(res.x[5]),
                                self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window, xaxis_window)
                                / res.x[4] / len(spectrum_window), len(spectrum_window), "multiple",
                                snr, res.success, flag]]), axis=0)

                        '''Separated peak - peak1'''
                        xaxis_window_plot = []
                        spectrum_window_plot = []
                        centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[0]))
                        for kkk in range((centroid - 50), (centroid + 50), 1):
                            xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                            spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])

                        res = (self.gaussian_def.gaussian_fit_one_element_width_constraint(
                            xaxis_window_plot_2, spectrum_window_plot_2,
                            local_maximum[0], spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]],
                            func, std))

                        if save_fitting:
                            self.plotting.plot_one_gaussian_separated_width_constraint(
                                xaxis_window_plot_2, spectrum_window_plot_2,
                                local_maximum[0], spectrum_window[np.where(local_maximum[0] == xaxis_window)[0][0]],
                                res, func, std, path_outputs)

                        '''peak shape peak1'''
                        if (abs(res.x[2]) == func(local_maximum[0]) + 5 * std) \
                                or (abs(res.x[2]) == func(local_maximum[0]) - 5 * std):
                            flag = "bad"
                        else:
                            flag = "good"
                        '''Find minima and local noise around peak1'''
                        left_min, right_min = self.dataop.find_minima_around_peak(
                            xaxis_window_plot_2, spectrum_window_plot_2, local_maximum[0])
                        local_noise = self.noisedetermination.local_noise_around_peak(
                            mean_spectrum, local_maximum[0], window_size=20)
                        snr = res.x[1] / local_noise
                        '''Add parameters to final table peak1'''
                        full_list_of_fittings = np.append(
                            full_list_of_fittings, np.array([[
                                left_min, local_maximum[0], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                                self.gaussian_def.cal_chi2(res.x, spectrum_window, xaxis_window) / res.x[1]
                                / len(spectrum_window), len(spectrum_window), "separated", snr, res.success, flag]]),
                            axis=0)

                '''More than 3 peaks on window'''
                if len(local_maximum) > 3:
                    local_maximum = np.append(local_maximum, 0)
                    for j in range(len(local_maximum) - 1):
                        if j == 0:
                            xaxis_window_plot = []
                            spectrum_window_plot = []
                            centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[j]))

                            for kkk in range((centroid - 80), (centroid + 80), 1):
                                xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                                spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])

                            res = self.gaussian_def.gaussian_fit_two_elements_width_constraint(
                                xaxis_window_plot, spectrum_window_plot,
                                local_maximum[j], spectrum_window[np.where(local_maximum[j] == xaxis_window)[0][0]],
                                local_maximum[j + 1], spectrum_window[np.where(local_maximum[j + 1] ==
                                                                               xaxis_window)[0][0]],
                                func, std)

                            if save_fitting:
                                self.plotting.plot_two_gaussians_width_constraint(
                                    xaxis_window_plot, spectrum_window_plot,
                                    local_maximum[j], spectrum_window[np.where(local_maximum[j] == xaxis_window)[0][0]],
                                    local_maximum[j + 1], spectrum_window[np.where(local_maximum[j + 1] ==
                                                                                   xaxis_window)[0][0]],
                                    func, std, res, path_outputs)

                            '''peak shape peak_j'''
                            if (abs(res.x[2]) == func(local_maximum[j]) + 5 * std) \
                                    or (abs(res.x[2]) == func(local_maximum[j]) - 5 * std):
                                flag = "bad"
                            else:
                                flag = "good"
                            '''Find minima and local noise around peak_j'''
                            left_min, right_min = self.dataop.find_minima_around_peak(
                                xaxis_window_plot, spectrum_window_plot, local_maximum[j])
                            local_noise = self.noisedetermination.local_noise_around_peak(
                                mean_spectrum, local_maximum[j], window_size=20)
                            snr = res.x[1] / local_noise
                            '''Add parameters to final table peak_j'''
                            full_list_of_fittings = np.append(
                                full_list_of_fittings, np.array([[
                                    left_min, local_maximum[j], right_min, float(res.x[0]), res.x[1], abs(res.x[2]),
                                    self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window, xaxis_window)
                                    / res.x[1] / len(spectrum_window), len(spectrum_window), "multiple",
                                    snr, res.success, flag]]), axis=0)

                        elif j == (len(local_maximum) - 2):
                            xaxis_window_plot = []
                            spectrum_window_plot = []
                            centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[j]))

                            for kkk in range((centroid - 80), (centroid + 80), 1):
                                xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                                spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])

                            res = self.gaussian_def.gaussian_fit_two_elements_width_constraint(
                                xaxis_window_plot, spectrum_window_plot,
                                local_maximum[j - 1], spectrum_window[np.where(local_maximum[j - 1] ==
                                                                               xaxis_window)[0][0]],
                                local_maximum[j], spectrum_window[np.where(local_maximum[j] == xaxis_window)[0][0]],
                                func, std)

                            if save_fitting:
                                self.plotting.plot_two_gaussians_width_constraint(
                                    xaxis_window_plot, spectrum_window_plot,
                                    local_maximum[j - 1], spectrum_window[np.where(local_maximum[j - 1] ==
                                                                                   xaxis_window)[0][0]],
                                    local_maximum[j], spectrum_window[np.where(local_maximum[j] == xaxis_window)[0][0]],
                                    func, std, res, path_outputs)

                            '''peak shape peak_j'''
                            if (abs(res.x[5]) == func(local_maximum[j]) + 5 * std) \
                                    or (abs(res.x[5]) == func(local_maximum[j]) - 5 * std):
                                flag = "bad"
                            else:
                                flag = "good"
                            '''Add parameters to final table peak_j'''
                            left_min, right_min = self.dataop.find_minima_around_peak(
                                xaxis_window_plot, spectrum_window_plot, local_maximum[j])
                            local_noise = self.noisedetermination.local_noise_around_peak(
                                mean_spectrum, local_maximum[j], window_size=30)
                            snr = res.x[4] / local_noise
                            '''Add parameters to final table peak_j'''
                            full_list_of_fittings = np.append(
                                full_list_of_fittings, np.array([[
                                    left_min, local_maximum[j], right_min, float(res.x[3]), res.x[4], abs(res.x[5]),
                                    self.gaussian_def.cal_chi2_multigaussian(res.x, spectrum_window, xaxis_window)
                                    / res.x[4] / len(spectrum_window), len(spectrum_window), "multiple",
                                    snr, res.success, flag]]), axis=0)

                        else:
                            xaxis_window_plot = []
                            spectrum_window_plot = []
                            centroid = np.argmin(np.abs(mean_spectrum.mzs - local_maximum[j]))

                            for kkk in range((centroid - 80), (centroid + 80), 1):
                                xaxis_window_plot = np.append(xaxis_window_plot, mean_spectrum.mzs[kkk])
                                spectrum_window_plot = np.append(spectrum_window_plot, mean_spectrum.intensities[kkk])

                            res = self.gaussian_def.gaussian_fit_three_elements_width_constraint(
                                xaxis_window_plot, spectrum_window_plot,
                                local_maximum[j - 1], spectrum_window[np.where(local_maximum[j - 1] ==
                                                                               xaxis_window)[0][0]],
                                local_maximum[j], spectrum_window[np.where(local_maximum[j] ==
                                                                           xaxis_window)[0][0]],
                                local_maximum[j + 1], spectrum_window[np.where(local_maximum[j + 1] ==
                                                                               xaxis_window)[0][0]],
                                func, std)

                            if save_fitting:
                                self.plotting.plot_three_gausssians_width_constraint(
                                    xaxis_window_plot, spectrum_window_plot,
                                    local_maximum[j - 1], spectrum_window[np.where(local_maximum[j - 1] ==
                                                                                   xaxis_window)[0][0]],
                                    local_maximum[j], spectrum_window[np.where(local_maximum[j] ==
                                                                               xaxis_window)[0][0]],
                                    local_maximum[j + 1], spectrum_window[np.where(local_maximum[j + 1] ==
                                                                                   xaxis_window)[0][0]],
                                    func, std, res, path_outputs)

                            '''peak shape peak_j'''
                            if (abs(res.x[5]) == func(local_maximum[j]) + 5 * std) \
                                    or (abs(res.x[5]) == func(local_maximum[j]) - 5 * std):
                                flag = "bad"
                            else:
                                flag = "good"
                            '''Add parameters to final table peak_j'''
                            left_min, right_min = self.dataop.find_minima_around_peak(
                                xaxis_window_plot, spectrum_window_plot, local_maximum[j])
                            local_noise = self.noisedetermination.local_noise_around_peak(
                                mean_spectrum, local_maximum[j], window_size=30)
                            snr = res.x[4] / local_noise
                            '''Add parameters to final table peak_j'''
                            full_list_of_fittings = np.append(
                                full_list_of_fittings, np.array([[
                                    left_min, local_maximum[j], right_min, float(res.x[3]), res.x[4], abs(res.x[5]),
                                    self.gaussian_def.cal_chi2_multigaussian_3g(res.x, spectrum_window, xaxis_window)
                                    / res.x[4] / len(spectrum_window), len(spectrum_window), "multiple",
                                    snr, res.success, flag]]), axis=0)

        return full_list_of_fittings
