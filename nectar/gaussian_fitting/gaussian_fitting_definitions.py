"""Class to define Gaussian fitting functions"""

"""___Built-In Modules___"""

"""___Third-Party Modules___"""
import numpy as np
from scipy.optimize import minimize

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class GaussianFittingDefinitions:
    def __init__(self):
        pass

    def gaussian(self, centroid, peak_intensity, width, xaxis):
        """
        Function to define a gaussian

        :param centroid: centroid of the Gaussian
        :type centroid: float
        :param peak_intensity: amplitude of the Gaussian
        :type peak_intensity: float
        :param width: width of the Gaussian
        :type width: float
        :param xaxis: x-axis window
        :type xaxis: array
        :return: array of intensities - gaussian curve
        :rtype: array
        """
        gausisintensities = peak_intensity * np.exp(-((xaxis - centroid) ** 2) / (2 * width**2))
        return gausisintensities

    def cal_chi2(self, input_array, intensities, xaxis):
        """
        Function to minimise

        :param input_array: array with centroid, intensity and width of the gaussian
        :type input_array: array
        :param intensities: intensities on the x-axis window
        :type intensities: array of floats
        :param xaxis: x-axis window
        :type xaxis: array of floats
        :return: chi2 to minimise
        :rtype: float
        """

        centroid, peak_intensity, width = input_array
        gaussintensities = self.gaussian(centroid, peak_intensity, width, xaxis)

        return np.sum((gaussintensities - intensities) ** 2)

    def multigaussian(self, centroid_1, peak_intensity_1, width_1, centroid_2, peak_intensity_2, width_2, xaxis):
        """
        Function to defined two overlapping Gaussian

        :param centroid_1: centroid of the first Gaussian
        :type centroid_1: float
        :param peak_intensity_1: amplitude of the first Gaussian
        :type peak_intensity_1: float
        :param width_1: width of the first Gaussian
        :type width_1: float
        :param centroid_2: centroid of the second Gaussian
        :type centroid_2: float
        :param peak_intensity_2: amplitude of the second Gaussian
        :type peak_intensity_2: float
        :param width_2: width of the second Gaussian
        :type width_2: float
        :param xaxis: x-axis window
        :type xaxis: array
        :return: total intensity under the curves of two Gaussian
        :rtype: array
        """

        gausisintensities_1 = peak_intensity_1 * np.exp(-((xaxis - centroid_1) ** 2) / (2 * width_1**2))
        gausisintensities_2 = peak_intensity_2 * np.exp(-((xaxis - centroid_2) ** 2) / (2 * width_2**2))
        multigausisintensities = gausisintensities_1 + gausisintensities_2

        return multigausisintensities

    def cal_chi2_multigaussian(self, input_array, intensities, xaxis):
        """
        Function to minimise

        :param input_array: array with centroid, intensity and width of each gaussian
        :type input_array: array
        :param intensities: intensities on the x-axis window
        :type intensities: array
        :param xaxis: x-axis window
        :type: array
        :return: chi2 to minimise
        :rtype: float
        """

        (centroid_1, peak_intensity_1, width_1, centroid_2, peak_intensity_2, width_2,) = input_array
        multigaussintensities = self.multigaussian(centroid_1, peak_intensity_1, width_1, centroid_2,
                                                   peak_intensity_2, width_2, xaxis)
        return np.sum((multigaussintensities - intensities) ** 2)

    def multigaussian_3g(self, centroid_1, peak_intensity_1, width_1,
                         centroid_2, peak_intensity_2, width_2,
                         centroid_3, peak_intensity_3, width_3, xaxis):
        """
        Function to defined three overlapping Gaussian

        :param centroid_1: centroid of the first Gaussian
        :type centroid_1: float
        :param peak_intensity_1: amplitude of the first Gaussian
        :type peak_intensity_1: float
        :param width_1: width of the first Gaussian
        :type width_1: float
        :param centroid_2: centroid of the second Gaussian
        :type centroid_2: float
        :param peak_intensity_2: amplitude of the second Gaussian
        :type peak_intensity_2: float
        :param width_2: width of the second Gaussian
        :type width_2: float
        :param centroid_3: centroid of the third Gaussian
        :type centroid_3: float
        :param peak_intensity_3: amplitude of the third Gaussian
        :type peak_intensity_3: float
        :param width_3: width of the third Gaussian
        :type width_3: float
        :param xaxis: x-axis window
        :type xaxis: array
        :return: total intensity under the curves of two Gaussian
        :rtype: array
        """

        gausisintensities_1 = peak_intensity_1 * np.exp(-((xaxis - centroid_1) ** 2) / (2 * width_1**2))
        gausisintensities_2 = peak_intensity_2 * np.exp(-((xaxis - centroid_2) ** 2) / (2 * width_2**2))
        gausisintensities_3 = peak_intensity_3 * np.exp(-((xaxis - centroid_3) ** 2) / (2 * width_3**2))

        multigausisintensities_3g = (gausisintensities_1 + gausisintensities_2 + gausisintensities_3)

        return multigausisintensities_3g

    def cal_chi2_multigaussian_3g(self, input_array, intensities, xaxis):
        """
        Function to minimise

        :param input_array: array with centroid, intensity and width of each gaussian
        :type input_array: array
        :param intensities: intensities on the x-axis window
        :type intensities: array
        :param xaxis: x-axis window
        :type: array
        :return: chi2 to minimise
        :rtype: float
        """
        (centroid_1, peak_intensity_1, width_1,
         centroid_2, peak_intensity_2, width_2,
         centroid_3, peak_intensity_3, width_3) = input_array

        multigaussintensities_3g = self.multigaussian_3g(
            centroid_1, peak_intensity_1, width_1,
            centroid_2, peak_intensity_2, width_2,
            centroid_3, peak_intensity_3, width_3, xaxis)

        return np.sum((multigaussintensities_3g - intensities) ** 2)

    def gaussian_fit_one_element(self, xaxis, intensities):
        """
        Function to fit one gaussian without constrains

        :param xaxis: x-axis window where the peak is located
        :type xaxis: array
        :param intensities: intensities inside the x-axis window
        :type intensities: array
        :return: optimised result
        :rtype: class 'scipy.optimize._optimize.OptimizeResult'>
        """

        bnds = ((min(xaxis), max(xaxis)), (0, None), (0, 0.1))
        # cal_chi2: object to minimise; x0: initial guess
        res = minimize(self.cal_chi2, x0=[xaxis[np.argmax(intensities)], np.max(intensities), 0.01],
                       args=(intensities, xaxis), method="TNC", bounds=bnds, options={"maxiter": 50000, "ftol": 0.0001})

        return res

    def gaussian_fit_one_element_separated(self, xaxis, intensities, xaxis_local, intensities_local):
        """
        Function to fit one gaussian without constrains (when two peaks are in the same window)

        :param xaxis: x-axis window where the peak is located
        :type xaxis: array
        :param intensities: intensities inside the x-axis window
        :type intensities: array
        :param xaxis_local: x-axis around the separated peak
        :param intensities_local: intensities around the separated peak
        :return: optimised result
        :rtype: class 'scipy.optimize._optimize.OptimizeResult'>
        """
        bnds = ((min(xaxis), max(xaxis)), (0, None), (0, 0.1))
        res = minimize(self.cal_chi2, x0=[xaxis_local, intensities_local, 0.01],
                       args=(intensities[np.where(abs(xaxis - xaxis_local) < 0.01)],
                             xaxis[np.where(abs(xaxis - xaxis_local) < 0.02)]),
                       method="TNC", bounds=bnds, options={"maxiter": 50000, "ftol": 0.0001})

        return res

    def gaussian_fit_two_elements(self, xaxis, intensities, xaxis_1, intensities_1, xaxis_2, intensities_2):
        """
        Function to fit two gaussian without constrains

        :param xaxis: x-axis window where the peak is located
        :type xaxis: array
        :param intensities: intensities inside the x-axis window
        :type intensities: array
        :param xaxis_1: local x-axis window for the first peak
        :type xaxis_1: array
        :param intensities_1: local intensities window for the first peak
        :type intensities_1: array
        :param xaxis_2: local x-axis window for the second peak
        :type xaxis_2: array
        :param intensities_2: local intensities window for the second peak
        :type intensities_2: array
        :return: optimised result
        :rtype: class 'scipy.optimize._optimize.OptimizeResult'>
        """
        bnds = ((min(xaxis), max(xaxis)), (0, None), (0, 0.1),
                (min(xaxis), max(xaxis)), (0, None), (0, 0.1))
        res = minimize(self.cal_chi2_multigaussian, x0=[xaxis_1, intensities_1, 0.01,
                                                        xaxis_2, intensities_2, 0.01],
                       args=(intensities[np.where(abs(xaxis - xaxis_2) < 0.03)],
                             xaxis[np.where(abs(xaxis - xaxis_2) < 0.03)]),
                       method="TNC", bounds=bnds, options={"maxiter": 50000, "ftol": 0.0001})

        return res

    def gaussian_fit_three_elements(self, xaxis, intensities, xaxis_1, intensities_1, xaxis_2, intensities_2,
                                    xaxis_3, intensities_3):
        """
        Function to fit three gaussian without constrains

        :param xaxis: x-axis window where the peak is located
        :type xaxis: array
        :param intensities: intensities inside the x-axis window
        :type intensities: array
        :param xaxis_1: local x-axis window for the first peak
        :type xaxis_1: array
        :param intensities_1: local intensities window for the first peak
        :type intensities_1: array
        :param xaxis_2: local x-axis window for the second peak
        :type xaxis_2: array
        :param intensities_2: local intensities window for the second peak
        :type intensities_2: array
        :param xaxis_3: local x-axis window for the third peak
        :type xaxis_3: array
        :param intensities_3: local intensities window for the third peak
        :type intensities_3: array
        :return: optimised result
        :rtype: class 'scipy.optimize._optimize.OptimizeResult'>
        """
        bnds = ((min(xaxis), max(xaxis)), (0, None), (0, 0.1),
                (min(xaxis), max(xaxis)), (0, None), (0, 0.1),
                (min(xaxis), max(xaxis)), (0, None), (0, 0.1))

        res = minimize(self.cal_chi2_multigaussian_3g,
                       x0=[xaxis_1, intensities_1, 0.01,
                           xaxis_2, intensities_2, 0.01,
                           xaxis_3, intensities_3, 0.01],
                       args=(intensities, xaxis), method="TNC", bounds=bnds, options={"maxiter": 50000, "ftol": 0.0001})

        return res

    def gaussian_fit_one_element_width_constraint(self, xaxis, intensities, xaxis_max, peak_max, func, std):
        """
        Function to fit one gaussian with constrains on centroid, width and intensity

        :param xaxis: x-axis window where the peak is located
        :type xaxis: array
        :param intensities: intensities inside the x-axis window
        :type intensities: array
        :param xaxis_max: m/z centroid value of the peak
        :type xaxis_max: float
        :param peak_max: intensity value at the m/z centroid xaxis_max
        :type peak_max: float
        :param func: correlation function to determine constrains according to the resolving power
        :type func: poly1d
        :param std: intercept of the resolving power to determine constrains
        :type std: float
        :return: optimised result
        :rtype: class 'scipy.optimize._optimize.OptimizeResult'>
        """
        bnds = ((xaxis_max - func(xaxis_max) - 1 * std, xaxis_max + func(xaxis_max) + 1 * std),
                (peak_max - 0.5 * peak_max, peak_max + 0.5 * peak_max),
                (func(xaxis_max) - 5 * std, func(xaxis_max) + 5 * std))

        peak_fit = intensities[(xaxis > xaxis_max - func(xaxis_max) - 6 * std)
                               & (xaxis < xaxis_max + func(xaxis_max) + 6 * std)]

        xaxis_fit = xaxis[(xaxis > xaxis_max - func(xaxis_max) - 6 * std)
                          & (xaxis < xaxis_max + func(xaxis_max) + 6 * std)]

        res = minimize(self.cal_chi2, x0=[xaxis_max, peak_max, func(xaxis_max)],
                       args=(peak_fit, xaxis_fit), method="TNC", bounds=bnds,
                       options={"maxiter": 50000, "ftol": 0.0001})

        return res

    def gaussian_fit_two_elements_width_constraint(self, xaxis, intensities, xaxis_1, intensities_1,
                                                   xaxis_2, intensities_2, func, std):
        """
        Function to fit two gaussian with constrains on centroid, width and intensity

        :param xaxis: x-axis window where the peak is located
        :type xaxis: array
        :param intensities: intensities inside the x-axis window
        :type intensities: array
        :param xaxis_1: m/z centroid value of the first peak
        :type xaxis_1: float
        :param intensities_1: intensity value at the m/z centroid of the first peak xaxis_1
        :type intensities_1: float
        :param xaxis_2: m/z centroid value of the second peak
        :type xaxis_2: float
        :param intensities_2: intensity value at the m/z centroid of the second peak xaxis_2
        :type intensities_2: float
        :param func: correlation function to determine constrains according to the resolving power
        :type func: poly1d
        :param std: intercept of the resolving power to determine constrains
        :type std: float
        :return: optimised result
        :rtype: class 'scipy.optimize._optimize.OptimizeResult'>
        """

        bnds = ((xaxis_1 - func(xaxis_1) - 1 * std, xaxis_1 + func(xaxis_1) + 1 * std),
                (intensities_1 - 0.5 * intensities_1, intensities_1 + 0.5 * intensities_1),
                (func(xaxis_1) - 5 * std, func(xaxis_1) + 5 * std),
                (xaxis_2 - func(xaxis_2) - 1 * std, xaxis_2 + func(xaxis_2) + 1 * std),
                (intensities_2 - 0.5 * intensities_2, intensities_2 + 0.5 * intensities_2),
                (func(xaxis_2) - 5 * std, func(xaxis_2) + 5 * std))

        peak_fit = intensities[(xaxis > xaxis_1 - func(xaxis_1) - 6 * std)
                               & (xaxis < xaxis_2 + func(xaxis_2) + 6 * std)]

        xaxis_fit = xaxis[(xaxis > xaxis_1 - func(xaxis_1) - 6 * std)
                          & (xaxis < xaxis_2 + func(xaxis_2) + 6 * std)]

        res = minimize(self.cal_chi2_multigaussian, x0=[xaxis_1, intensities_1, func(xaxis_1),
                                                        xaxis_2, intensities_2, func(xaxis_2)],
                       args=(peak_fit, xaxis_fit), method="TNC", bounds=bnds,
                       options={"maxiter": 50000, "ftol": 0.0001})

        return res

    def gaussian_fit_three_elements_width_constraint(self, xaxis, intensities, xaxis_1, intensities_1,
                                                     xaxis_2, intensities_2, xaxis_3, intensities_3, func, std):
        """
        Function to fit three gaussian with constrains on centroid, width and intensity

        :param xaxis: x-axis window where the peak is located
        :type xaxis: array
        :param intensities: intensities inside the x-axis window
        :type intensities: array
        :param xaxis_1: m/z centroid value of the first peak
        :type xaxis_1: float
        :param intensities_1: intensity value at the m/z centroid of the first peak xaxis_1
        :type intensities_1: float
        :param xaxis_2: m/z centroid value of the second peak
        :type xaxis_2: float
        :param intensities_2: intensity value at the m/z centroid of the second peak xaxis_2
        :type intensities_2: float
        :param xaxis_3: m/z centroid value of the third peak
        :type xaxis_3: float
        :param intensities_3: intensity value at the m/z centroid of the third peak xaxis_2
        :type intensities_3: float
        :param func: correlation function to determine constrains according to the resolving power
        :type func: poly1d
        :param std: intercept of the resolving power to determine constrains
        :type std: float
        :return: optimised result
        :rtype: class 'scipy.optimize._optimize.OptimizeResult'>
        """

        bnds = ((xaxis_1 - func(xaxis_1) - 1 * std, xaxis_1 + func(xaxis_1) + 1 * std),
                (intensities_1 - 0.5 * intensities_1, intensities_1 + 0.5 * intensities_1),
                (func(xaxis_1) - 5 * std, func(xaxis_1) + 5 * std),
                (xaxis_2 - func(xaxis_2) - 1 * std, xaxis_2 + func(xaxis_2) + 1 * std),
                (intensities_2 - 0.5 * intensities_2, intensities_2 + 0.5 * intensities_2),
                (func(xaxis_2) - 5 * std, func(xaxis_2) + 5 * std),
                (xaxis_3 - func(xaxis_3) - 1 * std, xaxis_3 + func(xaxis_3) + 1 * std),
                (intensities_3 - 0.5 * intensities_3, intensities_3 + 0.5 * intensities_3),
                (func(xaxis_3) - 5 * std, func(xaxis_3) + 5 * std))

        xaxis_fit = xaxis[(xaxis > xaxis_1 - func(xaxis_1) - 6 * std) & (xaxis < xaxis_3 + func(xaxis_3) + 6 * std)]
        peak_fit = intensities[(xaxis > xaxis_1 - func(xaxis_1) - 6 * std)
                               & (xaxis < xaxis_3 + func(xaxis_3) + 6 * std)]

        res = minimize(self.cal_chi2_multigaussian_3g, x0=[xaxis_1, intensities_1, func(xaxis_1),
                                                           xaxis_2, intensities_2, func(xaxis_2),
                                                           xaxis_3, intensities_3, func(xaxis_3)],
                       args=(peak_fit, xaxis_fit), method="TNC", bounds=bnds,
                       options={"maxiter": 50000, "ftol": 0.0001})

        return res
