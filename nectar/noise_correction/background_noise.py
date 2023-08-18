"""Class to correct from background noise/contamination from matrix/solvent"""

"""___Built-In Modules___"""
from nectar.plotting.plotting import Plotting

"""___Third-Party Modules___"""
import numpy as np
from tqdm import tqdm

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class BackgroundNoise:
    def __init__(self):
        self.plotting = Plotting()
        pass

    def backgroundnoise_identification(self, data_ori, data_spectrum, list_of_peaks, path_outputs,
                                       save_plot_backgroundNoise=False):
        """
        Function to identified and remove from the final list of peaks the background noise peaks
        :param data_ori: MSI datacube
        :type data_ori: DataCube
        :param data_spectrum: mean spectrum
        :type data_spectrum: DataSpectrum
        :param list_of_peaks: First de-noised list of peaks
        :type list_of_peaks: pandas DataFrame
        :param path_outputs: outputs path
        :type path_outputs: str
        :param save_plot_backgroundNoise: option to save the single ion images and the estimated S/N
        :type save_plot_backgroundNoise: bool
        :return: list_of_peaks (pandas DataFrame)
        """

        ''' The input DataCube needs to have tissue and background mask to be able to use this function'''
        mask = data_ori.pixelmask

        if len(mask) == (mask == 0).sum():
            print("!Error: Your data does not have a mask to apply the background correction. \n"
                  'Please, apply the "background_subtraction" function first.')
            exit()

        ''' Select what cluster number corresponds to tissue and background'''
        cluster_tissue = int(input("Please enter the tissue cluster number: "))
        cluster_background = int(input("Please enter the background cluster number: "))
        mask_2d = np.reshape(mask, (data_ori.shape[1], data_ori.shape[0])).T
        print("Number of background pixels", len(mask_2d[np.where(mask_2d == cluster_tissue)]))
        print("Number of tissue pixels", len(mask_2d[np.where(mask_2d == cluster_background)]))

        print("Background noise correction running...")
        '''Including new columns in the dataframe for the final list'''
        list_of_peaks["meanIntTotal"] = ""
        list_of_peaks["meanTissue"] = ""
        list_of_peaks["meanBackground"] = ""
        list_of_peaks["log2(meanTissue/meanBackground)"] = ""
        list_of_peaks["Tissue S/N"] = ""
        list_of_peaks["Background S/N"] = ""
        list_of_peaks["ratio[Tis/bak] S/N"] = ""
        list_of_peaks["diff[Tis/bak] S/N"] = ""
        list_of_peaks["left min"] = list_of_peaks["left min"].astype(float)
        list_of_peaks["right min"] = list_of_peaks["right min"].astype(float)

        '''Calculate the S/N for background and tissue'''
        for i in tqdm(range(len(list_of_peaks["meas mz"]))):
            image = data_ori.get_intensities_between_mzs(list_of_peaks["left min"][i], list_of_peaks["right min"][i])
            background = image[np.where(mask_2d == cluster_background)]
            background_sum = np.mean(background) * len(mask_2d[np.where(mask_2d == cluster_background)])
            background_std = np.std(background) * np.sqrt(len(mask_2d[np.where(mask_2d == cluster_background)]))
            sn_backgroundpixels = background_sum / background_std
            signal_tissue = np.sum(image[np.where(mask_2d == cluster_tissue)])

            noise_sii_background = np.zeros((6, 2))
            if not np.isfinite(sn_backgroundpixels):
                print("finite")
                continue

            '''These values are specific for the case study we were studying. Estimation of the background noise
            for the number of pixels on tissue'''
            list = [1, 5, 10, 50, 100, 250]
            list_n = [3000, 1000, 500, 100, 50, 20]
            for j, pixel in enumerate(list):
                pixel_noise_average_bg = np.zeros(list_n[j])
                for jj in range(0, list_n[j], 1):
                    pixel_noise_bg = np.random.choice(image[np.where(mask_2d == cluster_background)],
                                                      pixel, replace=False)
                    pixel_noise_average_bg[jj] = np.sum(pixel_noise_bg)

                noise_sii_background[j] = [pixel, np.std(pixel_noise_average_bg)]

            a = np.polyfit(np.log10(noise_sii_background[:, 0]).astype(np.float32),
                           np.log10(noise_sii_background[:, 1]).astype(np.float32), 1)
            func_all_peaks = np.poly1d(a)
            a = func_all_peaks(np.log10(len(mask_2d[np.where(mask_2d == cluster_tissue)])))
            bckgr_noise = 10**a

            '''Calculation of S/N, ration and difference of tissue and background'''
            sn_tissue = signal_tissue / bckgr_noise
            sn_background = background_sum / bckgr_noise
            ratio_sn = sn_tissue / sn_background
            diff_sn = sn_tissue - sn_background

            if save_plot_backgroundNoise == True:
                self.plotting.plot_backgroundNoise(image, data_spectrum,
                                                   list_of_peaks["meas mz"][i],
                                                   list_of_peaks["left min"][i],
                                                   list_of_peaks["right min"][i],
                                                   noise_sii_background, a, mask_2d,
                                                   sn_tissue, sn_background, path_outputs)

            '''Filling the final table'''
            image = image.flatten()
            mean_intensities = np.mean(image)
            mean_tissue = np.sum(image[np.where(mask == cluster_tissue)]) / len(image[np.where(mask == cluster_tissue)])
            mean_background = np.sum(image[np.where(mask == cluster_background)]) / \
                              len(image[np.where(mask == cluster_background)])

            list_of_peaks["meanIntTotal"][i] = mean_intensities
            list_of_peaks["meanTissue"][i] = mean_tissue
            list_of_peaks["meanBackground"][i] = mean_background
            list_of_peaks["log2(meanTissue/meanBackground)"][i] = np.log2(mean_tissue / mean_background)
            list_of_peaks["Tissue S/N"][i] = sn_tissue
            list_of_peaks["Background S/N"][i] = sn_background
            list_of_peaks["ratio[Tis/bak] S/N"][i] = ratio_sn
            list_of_peaks["diff[Tis/bak] S/N"][i] = diff_sn

        list_of_peaks.to_csv(path_outputs + "peaks_classification.csv", index=False)
        print("!Final list of peaks saved in: " + path_outputs)

        return list_of_peaks
