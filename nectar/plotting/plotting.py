"""Class for performing plots (e.g. plotting mean, Gaussian fittings)"""

"""___Built-In Modules___"""
from nectar.gaussian_fitting.gaussian_fitting_definitions import GaussianFittingDefinitions

"""___Third-Party Modules___"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class Plotting:
    def __init__(self):
        self.gaussian_def = GaussianFittingDefinitions()
        pass

    def plot_mean_spectrum(self, data_spectrum):
        """
        Function to plot the mean spectrum

        :param data_spectrum: data spectrum to plot
        :type data_spectrum: DataSpectrum
        :return: None
        :rtype: None
        """

        plt.plot(data_spectrum.mzs, data_spectrum.intensities)
        plt.title('Mean spectrum')
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.show()
        plt.clf()

    def plot_noise_spectrum(self, mean_spectrum, intensities_peaks, intensities_noise):
        """
        Function to show the separation between signal and noise in the mean spectrum

        :param mean_spectrum: mean spectrum in which the noise correction has been applied
        :type mean_spectrum: DataSpectrum
        :param intensities_peaks: intensities masked as signal
        :type intensities_peaks: array of floats
        :param intensities_noise: intensities masked as noise
        :type intensities_noise: array of floats
        :return: None
        :rtype: None
        """

        plt.plot(mean_spectrum.mzs, mean_spectrum.intensities, color="black", linewidth=0.5, label='Signal')
        plt.plot(mean_spectrum.mzs, intensities_peaks, color="black", linewidth=0.5)
        plt.plot(mean_spectrum.mzs, intensities_noise, color="orange", label='Noise')
        plt.title('Noise/signal - mean spectrum')
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
        plt.clf()

    def plot_spectra_with_chemicalnoise(self, spectrum, intensities_new, chemical_noise):
        """
        Function to show the original mean spectrum, the subtracted modeled chemical noise and
        the corrected mean spectrum.

        :param spectrum: original mean spectrum
        :type spectrum: DataSpectrum
        :param intensities_new: intensities of the spectrum after correction
        :type intensities_new: array
        :param chemical_noise: modelled chemical noise that has been subtracted from the original spectrum
        :type chemical_noise: array
        :return: None
        :rtype: None
        """

        plt.plot(spectrum.mzs, spectrum.intensities, color="blue", label="Original", alpha=0.7)
        plt.plot(spectrum.mzs, chemical_noise, color="red", label="Chemical noise", alpha=0.6)
        plt.plot(spectrum.mzs, intensities_new, color="green", label="Corrected", alpha=0.7)
        plt.title('Chemical noise correction')
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
        plt.clf()

    def plot_peak_picking(self, mean_spectrum, mz_mean, peak_intensity_mean):
        """
        Function to plot the selected peaks after first noise correction

        :param mean_spectrum: spectrum used to determine the peaks
        :type mean_spectrum: DataSpectrum
        :param mz_mean: centroid values of the peaks
        :type mz_mean: array
        :param peak_intensity_mean: peaks intensity
        :type peak_intensity_mean: array
        :return: None
        :rtype: None
        """

        plt.plot(mean_spectrum.mzs, mean_spectrum.intensities, color="black")
        plt.scatter(mz_mean, peak_intensity_mean, color="red")
        plt.title('Selected peaks')
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.show()
        plt.clf()

    def plot_correlation(self, mz, width, func, xaxis, path_outputs):
        """
        Function to plot the resolving power

        :param mz: peaks centroid value
        :type mz: array
        :param width: width obtained in the gaussian fitting step
        :type width: array
        :param func: linear correlation with the ordinate forced to zero
        :type func: poly1d
        :param xaxis: x-axis range of the spectrum
        :type xaxis: array
        :param path_outputs: path where to save the plot
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        plt.scatter(mz, width, color="blue", alpha=0.3, s=50)
        plt.plot(xaxis, func(xaxis), linestyle="dashed", color="aqua", label="200 peaks correlation")
        plt.title("Resolving power - most intense peaks")
        plt.xlabel("Centroid")
        plt.ylabel("Width")
        plt.savefig(path_outputs + "Resolving_power.png")
        plt.clf()

    def plot_correlation_all_peaks(self, mz, width, func, func_all_peaks, xaxis, path_outputs):
        """
        Function to plot the correlation obtained with all peaks compared with the resolving power

        :param mz: peaks centroid value
        :type mz: array
        :param width: width obtained in the gaussian fitting step
        :type width: array
        :param func: linear correlation with the ordinate forced to zero
        :type func: poly1d
        :param func_all_peaks: linear correlation when using all peaks
        :type func_all_peaks: poly1d
        :param xaxis: x axis range of the spectrum
        :type xaxis: array
        :param path_outputs: path where to save the plot
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        plt.scatter(mz, width, color="blue", alpha=0.3, s=50)
        plt.plot(xaxis, func(xaxis), linestyle="dashed", color="aqua", label="Resolving power")
        plt.plot(xaxis, func_all_peaks(xaxis), linestyle="dashed", color="lawngreen", label="All peaks correlation")
        plt.title("All peaks correlation")
        plt.xlabel("Centroid")
        plt.ylabel("Width")
        plt.legend(ncol=2)
        plt.savefig(path_outputs + "All_peaks_correlation.png")
        plt.clf()

    def plot_tic(self, data_cube, data_spectrum):
        """
        Function to plot the Total Ion Count image from the full data cube

        :param data_cube: data cube
        :type data_cube: DataCube
        :param data_spectrum: mean spectrum to define first and last m/z values
        :type data_spectrum: DataSpectrum
        :return: total ion counts image
        :rtype: image
        """

        image = data_cube.get_intensities_between_mzs(data_spectrum.mzs[0], data_spectrum.mzs[-1])
        im = plt.imshow(image, cmap=cm.YlGnBu_r)
        plt.colorbar(im)
        plt.show()
        plt.clf()
        return im

    def plot_sii_final_list(self, data_cube, data_mean, final_table, path_outputs, save_fig=True):
        """
        Function to create single ion images from a list of interest

        :param data_cube: data cube from where the singles ion images are created
        :type data_cube: DataCube
        :param data_mean: mean spectrum to select and plot the specific peak
        :type data_mean: DataSpectrum
        :param final_table: table with peaks to plot
        :type final_table: pandas dataframe
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :param save_fig: option to save the images
        :type save_fig: bool
        :return: None
        :rtype: None
        """

        '''Creates directory to save images'''
        if save_fig:
            dirName = path_outputs + "Single_ion_images"
            if not os.path.exists(dirName):
                os.mkdir(dirName)

        '''Read table'''
        print('Creating ion images...')
        for i in range(len(final_table["meas mz"])):
            mzlow = final_table.iloc[i, 0]
            mzhigh = final_table.iloc[i, 2]
            image = data_cube.get_intensities_between_mzs(mzlow, mzhigh)
            '''define figure'''
            fig, (ax0, ax1) = plt.subplots(ncols=2)
            '''Creates single ion image'''
            im = ax0.imshow(image, cmap=cm.YlGnBu_r)
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

            '''Creates peak image'''
            intensity_window = data_mean.intensities[np.where((data_mean.mzs > mzlow) & (data_mean.mzs < mzhigh))]
            peak = np.max(intensity_window) + 2
            ax1.plot(data_mean.mzs, data_mean.intensities, color="black")
            ax1.set_xlim((mzlow - 0.05, mzhigh + 0.05))
            ax1.set_ylim((-0.1, peak + peak*0.02))
            ax1.axvline(mzlow, color="green")
            ax1.axvline(mzhigh, color="green")
            ax1.axvline(final_table.iloc[i, 1], color="red")
            ax1.scatter(final_table.iloc[i, 1], np.max(intensity_window), color="green", marker="o")

            plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
            fig.set_size_inches(14, 8)
            plt.title(str(round(final_table.iloc[i, 1], 3)) + " m/z", style='italic')

            '''Saves or shows figure'''
            if save_fig:
                plt.savefig(path_outputs + "\\Single_ion_images\\" + str(round(final_table.iloc[i, 1], 3)) + "mz.png",
                            dpi=100)
            else:
                plt.show()

            plt.close(fig)
        print('All images saved!')

    def plot_backgroundNoise(self, image, data_spectrum, mz, mzlow, mzhigh, noise_sii_background, a, mask, sn_tissue,
                             sn_background, path_outputs):
        """
        Function to plot the information related to the background noise of the peak

        :param image: image to study
        :type image: image
        :param data_spectrum: mean spectrum to read the peak
        :type data_spectrum: DataSpectrum
        :param mz: peak centroid
        :type mz: float
        :param mzlow: peak's left minima
        :type mzlow: float
        :param mzhigh:peak's right minima
        :type mzhigh: float
        :param noise_sii_background: noise in background
        :type noise_sii_background: array
        :param a: estimated noise value
        :type a: float
        :param mask: mask to identify tissue only
        :type mask: array
        :param sn_tissue: s/n tissue
        :type sn_tissue: float
        :param sn_background: s/n background
        :type sn_background: float
        :param path_outputs: path where to save the outputs
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        '''Creates directory to save background noise related images'''
        dirName = path_outputs + "SII_BackgroundNoise"
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        '''Defines the figure'''
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

        '''Single ion image figure'''
        ratio_sn = sn_tissue / sn_background
        diff_sn = sn_tissue - sn_background

        im = ax0.imshow(image, cmap=cm.YlGnBu_r)
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax0.title.set_text("S/N tissue:" + str(round(sn_tissue))
                           + "\n S/N bgr:" + str(round(sn_background))
                           + "\n ratio_SN:" + str(round(ratio_sn, 3))
                           + "\n diff_SN:" + str(round(diff_sn, 3)))

        '''Plot peak associated with image'''
        self.plot_peak_spectrum(ax1, data_spectrum, mz, mzlow, mzhigh)
        '''Plot binned image'''
        self.plot_sii_binned(ax2, image, npixels=3)
        '''Plot noise estimation'''
        self.plot_estimated_background(ax3, noise_sii_background, a, mask)

        plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
        fig.set_size_inches(14, 8)
        fig.suptitle(str(round(mz, 3)) + "\n" + str(round(mzlow, 3)) + "-" + str(round(mzhigh, 3)))
        plt.savefig(path_outputs + "/SII_BackgroundNoise/" + str(round(mz, 3)) + ".png", dpi=100)
        #plt.show()
        plt.clf()
        plt.close('all')

    def plot_peak_spectrum(self, ax1, data_spectrum, mz, mzlow, mzhigh):
        """
        Function to plot peak of interest

        :param ax1: plot reference
        :type ax1: figure
        :param data_spectrum: mean spectrum to select the peak
        :type data_spectrum: DataSpectrum
        :param mz: peak centroid
        :type mz: float
        :param mzlow: peak's left minima
        :type mzlow: float
        :param mzhigh: peak's right minima
        :type mzhigh: float
        :return: plot of the peak
        :rtype: figure
        """

        intensity_window = data_spectrum.intensities[
            np.where((data_spectrum.mzs > mzlow) & (data_spectrum.mzs < mzhigh))]
        peak = np.max(intensity_window) + 2
        ax1.plot(data_spectrum.mzs, data_spectrum.intensities, color="black")
        ax1.set_xlim((mzlow - 0.05, mzhigh + 0.05))
        ax1.set_ylim((-0.1, peak + peak * 0.02))

        '''limits used to create the image'''
        ax1.axvline(mzlow, color="green")
        ax1.axvline(mzhigh, color="green")
        '''centroid'''
        ax1.axvline(mz, color="red")
        ax1.scatter(mz, np.max(intensity_window), color="green", marker="o")
        ax1.title.set_text("Mean spectrum")
        ax1.set_xlabel("mz")
        ax1.set_ylabel("Intensity")
        return ax1

    def plot_sii_binned(self, ax2, image, npixels=3):
        """
        Function to create a binned image. Useful for low intensity peaks

        :param ax2: plot reference
        :type ax2: figure
        :param image: image to binned
        :type image: array
        :param npixels: number of pixels to bin
        :type npixels: int
        :return: binned image
        :rtype: figure
        """

        dim0 = int(np.ceil(image.shape[0] / npixels))
        dim1 = int(np.ceil(image.shape[1] / npixels))
        im_bin = np.zeros((dim0, dim1))
        for i in range(dim0 - 1):
            for j in range(dim1 - 1):
                im_bin[i, j] = np.mean(image[npixels * i: npixels * (i + 1), npixels * j: npixels * (j + 1)])
            # fill in the edges
            for ii in range(dim0 - 1):
                im_bin[i, dim1 - 1] = np.mean(image[npixels * ii: npixels * (ii + 1), npixels * (dim1 - 1)::])
            for j in range(dim1 - 1):
                im_bin[dim0 - 1, j] = np.mean(image[npixels * (dim0 - 1)::, npixels * j: npixels * (j + 1)])
            # fill the final corner
            im_bin[dim0 - 1, dim1 - 1] = np.mean(image[npixels * (dim0 - 1)::, npixels * (dim1 - 1)::])

        im = ax2.imshow(im_bin, cmap=cm.YlGnBu_r)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax2.title.set_text("Binned: " + str(npixels) + " pixels")
        return ax2

    def plot_estimated_background(self, ax3, noise_sii_background, a, mask):
        """
        Function to plot the calculated and extrapolated noise values

        :param ax3: image reference
        :type ax3: figure
        :param noise_sii_background: noise background
        :type noise_sii_background: numpy array
        :param a: estimated noise value
        :type a: float
        :param mask: mask of the tissue only region (to know number of pixels)
        :type mask: array
        :return: plot with estimated noise value for specific number of pixels
        :rtype: figure
        """

        ax3.scatter(np.log10(noise_sii_background[:, 0]),
                    np.log10(noise_sii_background[:, 1]), color="red", label="Measured")
        ax3.scatter(np.log10(len(mask[np.where(mask == 0)])), a, color="blue", label="Estimated")
        ax3.set_title("Background S/N", fontsize=20)
        ax3.set_xlabel("log10(N-pixels)", fontsize=16)
        ax3.set_ylabel("log10(std)", fontsize=16)
        ax3.tick_params(axis="both", which="minor", labelsize=14)
        ax3.locator_params(axis="y", nbins=4)
        ax3.locator_params(axis="x", nbins=4)
        ax3.legend()
        return ax3

    def plot_one_gaussian(self, xaxis, intensities, res, path_outputs):
        """
        Function to plot one gaussian fitting

        :param xaxis: x-axis range
        :type xaxis: array
        :param intensities: intensities in x-axis range
        :type intensities: array
        :param res: minimisation results
        :type res: class 'scipy.optimize._optimize.OptimizeResult'>
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        plt.plot(xaxis, intensities, color="black", label="spectrum")
        plt.plot(xaxis[np.where(intensities == np.max(intensities))], np.max(intensities), "o", color="red")
        plt.plot(xaxis, self.gaussian_def.gaussian(res.x[0], res.x[1], res.x[2], xaxis),
            color="lawngreen", label="gaussian fit")
        plt.savefig(path_outputs + "Gaussian_fitting_most_intense_peaks/" + str(round(res.x[0], 3)) + "_simple.png")
        # plt.show()
        plt.clf()

    def plot_one_gaussian_separated(self, xaxis, intensities, xaxis_local, intensities_local, res, path_outputs):
        """
        Function to plot one gaussian fitting when there are more than one peak in the window

        :param xaxis: x-axis range
        :type xaxis: array
        :param intensities: intensities in x-axis range
        :type intensities: array
        :param xaxis_local: centroid value peak
        :type xaxis_local: float
        :param intensities_local: intensities value peak
        :type intensities_local: float
        :param res: minimisation results
        :type res: class 'scipy.optimize._optimize.OptimizeResult'>
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        plt.plot(xaxis_local, intensities_local, "o", color="red")
        plt.plot(xaxis[np.where(abs(xaxis - xaxis_local) < 0.2)], intensities[np.where(abs(xaxis - xaxis_local) < 0.2)],
                 color="black", label="spectrum")
        plt.plot(xaxis[np.where(abs(xaxis - xaxis_local) < 0.2)],
                 self.gaussian_def.gaussian(res.x[0],
                                            res.x[1],
                                            res.x[2],
                                            xaxis[np.where(abs(xaxis - xaxis_local) < 0.2)]),
                 color="lawngreen", label="gaussian fit")
        plt.savefig(path_outputs + "Gaussian_fitting_most_intense_peaks/" + str(round(res.x[0], 3)) + "_simple.png")
        plt.clf()

    def plot_two_gaussians(self, xaxis, intensities, xaxis_local_1, intensities_local_1,
                           xaxis_local_2, intensities_local_2, res, path_outputs):
        """
        Function to plot two gaussian together

        :param xaxis: x-axis range
        :type xaxis: array
        :param intensities: intensities in x-axis range
        :type intensities: array
        :param xaxis_local_1: centroid value for first peak
        :type xaxis_local_1: float
        :param intensities_local_1: intensity value first peak
        :type intensities_local_1: float
        :param xaxis_local_2: centroid value for second peak
        :type xaxis_local_2: float
        :param intensities_local_2: intensities value second peak
        :type intensities_local_2: float
        :param res: minimisation results
        :type res: class 'scipy.optimize._optimize.OptimizeResult'>
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        plt.plot(xaxis_local_1, intensities_local_1, "o", color="red")
        plt.plot(xaxis_local_2, intensities_local_2, "o", color="red")
        plt.plot(xaxis, intensities, color="black", alpha=0.8)
        plt.plot(xaxis,
                 self.gaussian_def.multigaussian(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], xaxis),
                 color="lawngreen")
        plt.plot(xaxis,
                 self.gaussian_def.gaussian(res.x[0], res.x[1], res.x[2], xaxis),
                 color="magenta", linestyle="dashed", alpha=0.5)
        plt.plot(xaxis,
                 self.gaussian_def.gaussian(res.x[3], res.x[4], res.x[5], xaxis),
                 color="orange", linestyle="dashed", alpha=0.5)
        plt.savefig(path_outputs + "Gaussian_fitting_most_intense_peaks/" + str(round(res.x[0], 3)) + "_multiple.png")
        plt.clf()

    def plot_three_gausssians(self, xaxis, intensities, xaxis_local_1, intensities_local_1,
                              xaxis_local_2, intensities_local_2,
                              xaxis_local_3, intensities_local_3,
                              res, path_outputs):
        """
        Function to plot three gaussian together

        :param xaxis: x-axis range
        :type xaxis: array
        :param intensities: intensities in x-axis range
        :type intensities: array
        :param xaxis_local_1: centroid value first peak
        :type xaxis_local_1: float
        :param intensities_local_1: intensities value first peak
        :type intensities_local_1: float
        :param xaxis_local_2: centroid value for second peak
        :type xaxis_local_2: float
        :param intensities_local_2: intensity value second peak
        :type intensities_local_2: float
        :param xaxis_local_3: centroid value third peak
        :type xaxis_local_3: flat
        :param intensities_local_3: intensities value third peak
        :type intensities_local_3: float
        :param res: minimisation results
        :type res: class 'scipy.optimize._optimize.OptimizeResult'>
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        xaxis_fit_1 = xaxis[np.where(abs(xaxis - xaxis_local_1) < 0.2)]
        xaxis_fit_2 = xaxis[np.where(abs(xaxis - xaxis_local_2) < 0.2)]
        xaxis_fit_3 = xaxis[np.where(abs(xaxis - xaxis_local_3) < 0.2)]

        plt.plot(xaxis_local_1, intensities_local_1, "o", color="red")
        plt.plot(xaxis_local_2, intensities_local_2, "o", color="red")
        plt.plot(xaxis_local_3, intensities_local_3, "o", color="red")
        plt.plot(xaxis, intensities, color="black")
        plt.plot(xaxis,
                 self.gaussian_def.multigaussian_3g(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5],
                                                    res.x[6], res.x[7], res.x[8], xaxis), color="lawngreen")
        plt.plot(xaxis_fit_1, self.gaussian_def.gaussian(res.x[0], res.x[1], res.x[2], xaxis_fit_1),
                 color="magenta", linestyle="dashed", alpha=0.5)
        plt.plot(xaxis_fit_2, self.gaussian_def.gaussian(res.x[3], res.x[4], res.x[5], xaxis_fit_2),
                 color="orange", linestyle="dashed", alpha=0.5)
        plt.plot(xaxis_fit_3, self.gaussian_def.gaussian(res.x[6], res.x[7], res.x[8], xaxis_fit_3),
                 color="lime", linestyle="dashed", alpha=0.5)
        plt.savefig(path_outputs + "Gaussian_fitting_most_intense_peaks/" + str(round(res.x[0], 3)) + "_multiple.png")
        plt.clf()

    def plot_one_gaussian_width_constraint(self, xaxis, intensities, local_peak, res, path_outputs):
        """
        Function to plot one gaussian fitting with constrains

        :param xaxis: x-axis range
        :type xaxis: array
        :param intensities: intensities in x-axis range
        :type intensities: array
        :param local_peak: centroid peak value
        :type local_peak: float
        :param res: minimisation results
        :type res: class 'scipy.optimize._optimize.OptimizeResult'>
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        plt.plot(xaxis, intensities, color="black", label="spectrum")
        plt.plot(xaxis[np.where(xaxis == local_peak)], intensities[np.where(xaxis == local_peak)], "o", color="red")
        plt.plot(xaxis, self.gaussian_def.gaussian(res.x[0], res.x[1], res.x[2], xaxis),
            color="lawngreen", label="gaussian fit")
        plt.savefig(path_outputs + "Gaussian_fitting_width_constrained/" + str(round(res.x[0], 3)) + "_single.png")
        plt.clf()

    def plot_one_gaussian_separated_width_constraint(self, xaxis, intensities, xaxis_local, peak_local, res, func, std,
                                                     path_outputs):
        """
        Function to plot one gaussian fitting with constrains

        :param xaxis: x-axis range
        :type xaxis: array
        :param intensities: intensities in x-axis range
        :type intensities: array
        :param xaxis_local: centroid peak value
        :type xaxis_local: float
        :param peak_local: intensity peak value
        :type peak_local: float
        :param res: minimisation results
        :type res: class 'scipy.optimize._optimize.OptimizeResult'>
        :param func: correlation function
        :type func: poly1d
        :param std: new standard deviation
        :type std: float
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        plt.plot(xaxis, intensities, color="black", label="spectrum")
        plt.plot(xaxis_local, peak_local, "o", color="red")
        # xaxis_1_fit = xaxis[(xaxis > xaxis_local - func(xaxis_local) - 6 * std)
        #                     & (xaxis < xaxis_local + func(xaxis_local) + 6 * std)]

        plt.plot(xaxis, self.gaussian_def.gaussian(res.x[0], res.x[1], res.x[2], xaxis),
            color="lawngreen", label="gaussian fit")
        plt.savefig(path_outputs + "Gaussian_fitting_width_constrained/" + str(round(res.x[0], 3)) + "_single.png")
        # plt.show()
        plt.clf()

    def plot_two_gaussians_width_constraint(self, xaxis, intensities, xaxis_local_1, peak_local_1, xaxis_local_2,
                                            peak_local_2, func, std, res, path_outputs):
        """
        Function to plot two gaussian with constrains

        :param xaxis: x-axis range
        :type xaxis: array
        :param intensities: intensities in x-axis range
        :type intensities: array
        :param xaxis_local_1: centroid value for first peak
        :type xaxis_local_1: float
        :param peak_local_1: intensity peak value first peak
        :type peak_local_1: float
        :param xaxis_local_2: centroid value for second peak
        :type xaxis_local_2: float
        :param peak_local_2: intensity peak value second peak
        :type peak_local_2: float
        :param func: correlation function
        :type func: poly1d
        :param std: new standard deviation
        :type std: float
        :param res: minimisation results
        :type res: class 'scipy.optimize._optimize.OptimizeResult'>
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :return: None
        :rtype: None
        """

        plt.plot(xaxis_local_1, peak_local_1, "o", color="red")
        plt.plot(xaxis_local_2, peak_local_2, "o", color="red")
        plt.plot(xaxis, intensities, color="black", alpha=0.8)
        # xaxis_fit = xaxis[(xaxis > xaxis_local_1 - func(xaxis_local_1) - 6 * std)
        #                   & (xaxis < xaxis_local_2 + func(xaxis_local_2) + 6 * std)]

        plt.plot(xaxis, self.gaussian_def.multigaussian(
            res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], xaxis),
                 color="lawngreen")

        xaxis_1_fit = xaxis[(xaxis > xaxis_local_1 - func(xaxis_local_1) - 6 * std)
                            & (xaxis < xaxis_local_1 + func(xaxis_local_1) + 6 * std)]
        xaxis_2_fit = xaxis[(xaxis > xaxis_local_2 - func(xaxis_local_2) - 6 * std)
                            & (xaxis < xaxis_local_2 + func(xaxis_local_2) + 6 * std)]

        plt.plot(xaxis_1_fit, self.gaussian_def.gaussian(res.x[0], res.x[1], res.x[2], xaxis_1_fit),
                 color="magenta", linestyle="dashed", alpha=0.5)
        plt.plot(xaxis_2_fit, self.gaussian_def.gaussian(res.x[3], res.x[4], res.x[5], xaxis_2_fit),
                 color="orange", linestyle="dashed", alpha=0.5)
        plt.savefig(path_outputs + "Gaussian_fitting_width_constrained/" + str(round(res.x[0], 3)) + "_double.png")
        # plt.show()
        plt.clf()

    def plot_three_gausssians_width_constraint(self, xaxis, intensities,
                                               xaxis_local_1, peak_local_1,
                                               xaxis_local_2, peak_local_2,
                                               xaxis_local_3, peak_local_3, func, std, res, path_outputs):
        """
        Function to plot three gaussian with constrains

        :param xaxis: x-axis range
        :type xaxis: array
        :param intensities: intensities in x-axis range
        :type intensities: array
        :param xaxis_local_1: centroid value for first peak
        :type xaxis_local_1: float
        :param peak_local_1: intensity peak value first peak
        :type peak_local_1: float
        :param xaxis_local_2: centroid value for second peak
        :type xaxis_local_2: float
        :param peak_local_2: intensity peak value second peak
        :type peak_local_2: float
        :param xaxis_local_3: centroid value for third peak
        :type xaxis_local_3: float
        :param peak_local_3: intensity peak value third peak
        :type peak_local_3: float
        :param func: correlation function
        :type func: poly1d
        :param std: new standard deviation
        :type std: float
        :param res: minimisation results
        :type res: class 'scipy.optimize._optimize.OptimizeResult'>
        :param path_outputs: path where to save the images
        :type path_outputs: str
        :return: None
        :rtype: None
        :return:
        """

        xaxis_fit_1 = xaxis[(xaxis > xaxis_local_1 - func(xaxis_local_1) - 6 * std)
                            & (xaxis < xaxis_local_1 + func(xaxis_local_1) + 6 * std)]

        xaxis_fit_2 = xaxis[(xaxis > xaxis_local_2 - func(xaxis_local_2) - 6 * std)
                            & (xaxis < xaxis_local_2 + func(xaxis_local_2) + 6 * std)]

        xaxis_fit_3 = xaxis[(xaxis > xaxis_local_3 - func(xaxis_local_3) - 6 * std)
                            & (xaxis < xaxis_local_3 + func(xaxis_local_3) + 6 * std)]

        plt.plot(xaxis_local_1, peak_local_1, "o", color="red")
        plt.plot(xaxis_local_2, peak_local_2, "o", color="red")
        plt.plot(xaxis_local_3, peak_local_3, "o", color="red")
        plt.plot(xaxis, intensities, color="black")
        plt.plot(xaxis, self.gaussian_def.multigaussian_3g(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4],
                                                           res.x[5], res.x[6], res.x[7], res.x[8], xaxis),
                 color="lawngreen")
        plt.plot(xaxis_fit_1, self.gaussian_def.gaussian(res.x[0], res.x[1], res.x[2], xaxis_fit_1),
                 color="magenta", linestyle="dashed", alpha=0.5)
        plt.plot(xaxis_fit_2, self.gaussian_def.gaussian(res.x[3], res.x[4], res.x[5], xaxis_fit_2),
                 color="orange", linestyle="dashed", alpha=0.5)
        plt.plot(xaxis_fit_3, self.gaussian_def.gaussian(res.x[6], res.x[7], res.x[8], xaxis_fit_3),
                 color="purple", linestyle="dashed", alpha=0.5)
        plt.savefig(path_outputs + "Gaussian_fitting_width_constrained/" + str(round(res.x[0], 3)) + "_triple.png")
        plt.clf()
