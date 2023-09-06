"""Class for performing operations on the data"""

"""___Built-In Modules___"""
from nectar_msi.data_formats.data_spectrum import DataSpectrum
from nectar_msi.data_formats.data_cube import DataCube
from nectar_msi.noise_correction.noise_determination import NoiseDetermination
from nectar_msi.data_processing.savers import Savers

"""___Third-Party Modules___"""
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import h5py
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class DataOperations:
    def __init__(self):
        self.noisedetermination = NoiseDetermination()
        self.saver = Savers()
        pass

    def get_mean_spectrum(self, data_cube, start_mz=None, end_mz=None, xaxis=None, scaling_power=0.5):
        """
        Function to obtain the mean spectrum of a data_cube without assigned mask

        :param data_cube: data_cube of interest
        :type data_cube: DataCube
        :param start_mz: first value in the x axis of the mean spectrum (default to None)
        :type start_mz: float
        :param end_mz: last value in the x axis of the mean spectrum (default to None)
        :type end_mz: float
        :param xaxis: x axis of the spectrum (default to None, in which case is created)
        :type xaxis: array
        :param scaling_power: the difference in m/z between two neighbouring channels mz_1 and mz_2 is given by
        mz2-mz_1 = a * mz_1**p, where a is the scaling factor and p is the scaling power (default = 0.5)
        :type scaling_power: float
        :return: mean spectrum
        :rtype: DataSpectrum
        """

        # Creates common x-axis for all pixels
        if xaxis is None:
            xaxis, scaling_factor = self.make_xaxis(data_cube, start_mz=start_mz, end_mz=end_mz,
                                                    scaling_power=scaling_power)

        print("Creating mean spectrum...")
        mean_intensities = np.zeros_like(xaxis)
        n_valid_spectra = 0  # only selects pixels with data on it
        for index in tqdm(range(len(data_cube.idxs))):
            spectrum = data_cube.get_spectrum_idx(data_cube.idxs[index])
            if spectrum is not None:
                intensities_pixeli = self.process_pixel_intensities(spectrum, xaxis)
                mean_intensities += intensities_pixeli
                n_valid_spectra += 1
        mean_intensities = mean_intensities / n_valid_spectra  # mean intensities
        spectrum = DataSpectrum(-99, [-99, -99], xaxis, mean_intensities)  # mean spectrum as a DataSpectrum object
        print("Mean spectrum created!")
        return spectrum

    def get_mean_spectrum_tissue_background(self, data_cube, start_mz=None, end_mz=None, xaxis=None, mean_tissue=True,
                                            mean_background=False, scaling_power=0.5):

        """
        Function to create the mean spectrum of the tissue and/or background based in a
        previously set mask

        :param data_cube: data_cube from where we want to calculate the mean spectra
        :type data_cube: DataCube
        :param start_mz: first value in the x axis of the mean spectrum (default to None)
        :type start_mz: float
        :param end_mz: last value in the x axis of the mean spectrum (default to None)
        :type end_mz: float
        :param xaxis: x axis of the spectrum (default to None, in which case is created)
        :type xaxis: array
        :param mean_tissue: Option to choose if we want to create the mean spectrum
        of the tissue (default True)
        :type mean_tissue: bool
        :param mean_background: Option to choose if we want to create the mean spectrum
        of the background (default False)
        :type mean_background: bool
        :param scaling_power: the difference in m/z between two neighbouring channels mz_1 and mz_2 is given by
        mz2-mz_1 = a * mz_1**p, where a is the scaling factor and p is the scaling power (default = 0.5)
        :type scaling_power: float
        :return: spectrum for tissue and/or background
        :rtype: DataSpectrum
        """
        if xaxis is None:
            xaxis, scaling_factor = self.make_xaxis(data_cube,
                                                    start_mz=start_mz, end_mz=end_mz, scaling_power=scaling_power)

        cluster_tissue = int(input("Please enter the tissue cluster number: "))
        cluster_background = int(input("Please enter the background cluster number: "))

        if mean_tissue == True and mean_background == True:
            print("Creating mean spectrum from tissue...")
            n_valid_spectra = 0
            mean_intensities = np.zeros_like(xaxis)
            for index in tqdm(range(len(data_cube.idxs))):
                if data_cube.pixelmask[index] == cluster_tissue:
                    spectrum = data_cube.get_spectrum_idx(data_cube.idxs[index])
                    if spectrum is not None:
                        intensities_pixeli = self.process_pixel_intensities(spectrum, xaxis)
                        mean_intensities += intensities_pixeli
                        n_valid_spectra += 1
            mean_intensities = mean_intensities / n_valid_spectra
            spectrum_tissue = DataSpectrum(-99, [-99, -99], xaxis, mean_intensities)
            print("Tissue-spectrum created!")

            print("Creating mean spectrum from background...")
            n_valid_spectra = 0
            mean_intensities = np.zeros_like(xaxis)
            for index in tqdm(range(len(data_cube.idxs))):
                if data_cube.pixelmask[index] == cluster_background:
                    spectrum = data_cube.get_spectrum_idx(data_cube.idxs[index])
                    if spectrum is not None:
                        spectrum = data_cube.get_spectrum_idx(data_cube.idxs[index])
                        intensities_pixeli = self.process_pixel_intensities(spectrum, xaxis)
                        mean_intensities += intensities_pixeli
                        n_valid_spectra += 1
            mean_intensities = mean_intensities / n_valid_spectra
            spectrum_background = DataSpectrum(-99, [-99, -99], xaxis, mean_intensities)
            print("Background-spectrum created!")

            return spectrum_tissue, spectrum_background

        elif mean_background:
            print("Creating mean spectrum from background...")
            n_valid_spectra = 0
            mean_intensities = np.zeros_like(xaxis)
            for index in tqdm(range(len(data_cube.idxs))):
                if data_cube.pixelmask[index] == cluster_background:
                    spectrum = data_cube.get_spectrum_idx(data_cube.idxs[index])
                    if spectrum is not None:
                        spectrum = data_cube.get_spectrum_idx(data_cube.idxs[index])
                        intensities_pixeli = self.process_pixel_intensities(spectrum, xaxis)
                        mean_intensities += intensities_pixeli
                        n_valid_spectra += 1
            mean_intensities = mean_intensities / n_valid_spectra
            spectrum_background = DataSpectrum(-99, [-99, -99], xaxis, mean_intensities)
            print("Background-spectrum created!")

            return spectrum_background

        elif mean_tissue:
            print("Creating mean spectrum from tissue...")
            n_valid_spectra = 0
            mean_intensities = np.zeros_like(xaxis)
            for index in tqdm(range(len(data_cube.idxs))):
                if data_cube.pixelmask[index] == cluster_tissue:
                    spectrum = data_cube.get_spectrum_idx(data_cube.idxs[index])
                    if spectrum is not None:
                        intensities_pixeli = self.process_pixel_intensities(spectrum, xaxis)
                        mean_intensities += intensities_pixeli
                        n_valid_spectra += 1
            mean_intensities = mean_intensities / n_valid_spectra
            spectrum_tissue = DataSpectrum(-99, [-99, -99], xaxis, mean_intensities)
            print("Tissue-spectrum created!")

            return spectrum_tissue

    def process_pixel_intensities(self, data_spectrum, xaxis):
        """
        Function to assign intensities of each pixel into the corresponding channel of the common x-axis

        :param data_spectrum: spectrum of each pixel
        :type data_spectrum: DataSpectrum
        :param xaxis: common x-axis for all pixels
        :type xaxis: array
        :return: Array of intensities with the same shape as x-axis
        :rtype: array
        """
        mzs, intensities = data_spectrum.mzs, data_spectrum.intensities
        yvals = np.zeros_like(xaxis)

        for i, mz in enumerate(mzs):
            yvals[np.argmin(np.abs(xaxis - mz))] = intensities[i]
        return yvals

    def make_xaxis(self, data_cube, start_mz=None, end_mz=None, scaling_power=0.5):
        """
        Function to create a common x-axis for all pixels

        :param data_cube: data cube of interest
        :type data_cube: DataCube
        :param start_mz: first m/z value of the spectrum
        :type start_mz: float
        :param end_mz: last m/z value of the spectrum
        :type end_mz: float
        :param scaling_power: the difference in m/z between two neighbouring channels mz_1 and mz_2 is given by
        mz2-mz_1 = a * mz_1**p, where a is the scaling factor and p is the scaling power (default = 0.5)
        :type scaling_power: float
        :return: new x axis for the spectrum and scaling factor
        :rtype: array, float
        """
        print("Creating x axis")
        indexmax = -99
        lenmax = 0
        # finds the spectrum with more measurements in the m/z range to calculate the scaling factor
        for index in range(len(data_cube.coord)):
            if data_cube.get_spectrum_idx(data_cube.idxs[index]) is not None:
                len_spectrum_per_pixel = len(data_cube.get_spectrum_idx(data_cube.idxs[index]).mzs)
                if len_spectrum_per_pixel > lenmax:
                    lenmax = len_spectrum_per_pixel
                    indexmax = index

        spectrum = data_cube.get_spectrum_idx(data_cube.idxs[indexmax])
        # Uses this spectrum as reference to calculate the scaling factor
        # scaling factor determines the channel size (bin) across the m/z range
        scaling_factor, scaling_std = self.find_scaling_factor_xaxis(spectrum.mzs, scaling_power=scaling_power)
        # print("scaling factor", scaling_factor)
        # Creates the x-axis for the mean spectrum
        new_mz = self.fill_xaxis(spectrum.mzs, scaling_factor, scaling_power, start_mz=start_mz, end_mz=end_mz)
        return new_mz, scaling_factor

    def find_scaling_factor_xaxis(self, xaxis, scaling_power=0.5):
        """
        Function to find the scaling factor to be used to create the x-axis for the mean spectrum

        :param xaxis: x-axis of the selected spectrum with most number of measurements
        :type xaxis: array
        :param scaling_power: the difference in m/z between two neighbouring channels mz_1 and mz_2 is given by
        mz2-mz_1 = a * mz_1**p, where a is the scaling factor and p is the scaling power (default=0.5)
        :type scaling_power: float
        :return: scaling factor and standard deviation of the scaling factor
        :rtype: floats
        """

        # Distances among all measurement in the spectrum
        scaling_candidates = (xaxis[1:] - xaxis[:-1]) / xaxis[1:] ** scaling_power
        # Sigma clipping to determine the most common value, and therefore select the scaling factor
        sigma_new, average, values = self.noisedetermination.SigmaClip(scaling_candidates)
        # print(scaling_power,np.round(scaling_candidates/average,1),st.mode(np.round(scaling_candidates/average,1))[0][0])
        if st.mode(np.round(scaling_candidates / average, 1))[0][0] != 1:
            print(
                "Are you sure the scaling power of your x-axis is correct? It looks like the data indicates it is not."
            )
        return average, sigma_new

    def fill_xaxis(self, xaxis, scaling_factor, scaling_power, start_mz=None, end_mz=None):
        """
        Function to create the common x-axis for all pixels

        :param xaxis: x-axis of the reference spectrum
        :type xaxis: array
        :param scaling_factor: the difference in m/z between two neighbouring channels mz_1 and mz_2 is given by
        mz2-mz_1 = a * mz_1**p, where a is the scaling factor and p is the scaling power
        :type scaling_factor: float
        :param scaling_power: the difference in m/z between two neighbouring channels mz_1 and mz_2 is given by
        mz2-mz_1 = a * mz_1**p, where a is the scaling factor and p is the scaling power
        :type scaling_power: float
        :param start_mz: first value in the x axis of the mean spectrum (default to None,
        we need to set this value if xaxis = None)
        :type start_mz: float
        :param end_mz: last value in the x axis of the mean spectrum (default to None)
        :type end_mz: float
        :return: new x-axis for mean spectrum
        :rtype: array
        """
        xaxis2 = np.zeros(5000000)
        iextra = 0
        i = 0
        if start_mz is None:
            start_mz = xaxis[0]
        if end_mz is None:
            end_mz = xaxis[-1]
        xaxis2[0] = start_mz
        # (xaxis[1:]-xaxis[:-1])/xaxis[1:]**1.5 =scaling
        while xaxis2[i] < end_mz and i - iextra + 1 < len(xaxis):
            # distance between previous xaxis point and next theoretical xaxis point (i.e. dist between x[i+1] and x[i])
            dist_theor = xaxis[i - iextra + 1] ** scaling_power * scaling_factor
            # distance between previous xaxis point and next measured xaxis point
            dist_measured = xaxis[i - iextra + 1] - xaxis2[i]
            if dist_measured / dist_theor > 1.5:
                iextra += 1
                xaxis2[i + 1] = xaxis2[i] + xaxis2[i] ** scaling_power * scaling_factor
            else:
                # print(xaxis[i-iextra+1])
                xaxis2[i + 1] = xaxis[i - iextra + 1]
            i += 1

        xaxis2 = xaxis2[np.where((xaxis2 >= start_mz) & (xaxis2 < end_mz))]
        return xaxis2

    def find_minima_around_peak(self, xaxis_window, spectrum_window, peak):
        """
        Function to find the minima of a given m/z value

        :param xaxis_window: Window size of mz values around where to look for the minima
        :type xaxis_window: array
        :param spectrum_window: Intensity values of the xaxis_window
        :type spectrum_window: array
        :param peak: centroid value
        :type peak: float
        :return: minima around the m/z value
        :rtype: floats
        """

        dydx = np.diff(spectrum_window) / np.diff(xaxis_window)
        right = [xaxis_window[-1]]
        left = [xaxis_window[0]]
        for ii in range(len(dydx) - 1):
            if np.sign(dydx[ii + 1]) > np.sign(dydx[ii]):
                local_mz = xaxis_window[ii + 1]
                if local_mz > peak:
                    right = np.append(right, xaxis_window[ii + 1])
                else:
                    left = np.append(left, xaxis_window[ii + 1])
        left_min = np.argmin(np.abs(left - peak))
        right_min = np.argmin(np.abs(right - peak))
        mzlow = left[left_min]
        mzhigh = right[right_min]
        return mzlow, mzhigh

    def background_subtraction(self, data_cube, data_mean, path_outputs, n_clusters=2, show_plot=False):
        """
        Function to separate between tissue and background by K-means

        :param data_cube: data_cube from which we want to separate tissue and background
        :type data_cube: DataCube
        :param data_mean: total mean spectrum (tissue+background)
        :type data_mean: DataSpectrum
        :param path_outputs: path to save the KMeans image
        :type path_outputs: str
        :param n_clusters: Number of clusters for k-means
        :type n_clusters: int
        :param show_plot: Option to show the KMeans resulted plot (default False)
        :type show_plot: bool
        :return: data_cube with a set mask separating tissue and background
        :rtype: DataCube
        """
        dirName = path_outputs
        if not os.path.exists(dirName):
            os.mkdir(dirName)

        print("Background subtraction...")
        index_intensities = np.argsort(data_mean.intensities[np.where(~np.isnan(data_mean.intensities))])[::-1][:500]
        peaks_mz = data_mean.mzs[index_intensities]
        peaks_mz.sort()

        cube = np.zeros((data_cube.shape[0] * data_cube.shape[1], len(peaks_mz)))
        for i in range(len(peaks_mz)):
            for j in range(data_cube.shape[0] * data_cube.shape[1]):
                spec = data_cube.get_spectrum_idx(data_cube.idxs[j])
                if spec is not None:
                    cube[j, i] = spec.get_intensity_between_mzs(
                        peaks_mz[i] * (1 - 30 * 10**-6),
                        peaks_mz[i] * (1 + 30 * 10**-6))

        print("Applying K-means...")
        cluster = KMeans(init="random", n_clusters=n_clusters, n_init="auto", max_iter=300).fit_predict(cube)
        cluster_reshape = cluster.reshape(data_cube.shape[1], data_cube.shape[0], 1)
        cluster_reshape = np.moveaxis(cluster_reshape, 0, 1)

        print("Setting tissue-background mask...")
        data_cube.set_pixel_mask_2D(cluster_reshape)

        plt.matshow(data_cube.get_pixel_mask_2D(), cmap=cm.get_cmap("RdBu", 2))
        plt.colorbar()
        if show_plot:
            plt.show()
        plt.savefig(path_outputs + "KMeans_tissue_background.png")
        print('Figure save in ' + path_outputs + 'KMeans_tissue_background.png')
        plt.clf()
        return data_cube

    def read_mean_from_sa(self, path_sa):
        """
        Function to read the spectrum created in spectral analysis
        :param path_sa: path to the files created by spectral analysis
        :type path_sa: string
        :return: spectrum
        :rtype: DataSpectrum
        """
        mz = h5py.File(path_sa+'totalSpectrum_mzvalues.mat')
        intensities = h5py.File(path_sa+'totalSpectrum_intensities.mat')

        mz = mz["totalSpectrum_mzvalues"]
        intensities = intensities["totalSpectrum_intensities"]
        mz = [item for sublist in mz for item in sublist]
        intensities = [item for sublist in intensities for item in sublist]
        mean = DataSpectrum(-99, [-99, -99], mz, intensities)
        print('Mean spectrum from Spectral Analysis created')
        return mean
