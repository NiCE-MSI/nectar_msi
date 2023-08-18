"""Main file to be run in python to apply the pre-processing in a MSI dataset.
Several tasks can be done with the nectar package. Here we show an example of the most
standard options used to obtain a final list of peaks for further analysis."""

"""___Built-In Modules___"""
from nectar import Readers, Savers, DataOperations, NoiseCorrection, PeakPicking, DatabaseMatching, Plotting

"""___Third-Party Modules___"""
import numpy as np
from multiprocessing import Pool
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import scipy.io
import h5py
from skimage.segmentation import watershed
from skimage.filters import sobel
import copy
from tqdm import tqdm
from scipy.io import savemat
import matplotlib.cm as cm

"""___Authorship___"""
__author__ = "Ariadna Gonzalez"
__created__ = "24/02/2022"
__maintainer__ = "Ariadna Gonzalez"
__email__ = "ariadna.gonzalez@npl.co.uk"

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def main():
    reader = Readers()
    saver = Savers()
    dataop = DataOperations()
    noisecorrection = NoiseCorrection()
    databasematching = DatabaseMatching()
    peakpicking = PeakPicking()
    plotting = Plotting()

    '''Define paths'''
    path_data = "X:\\Ariadna\\PDAC\\MALDI\\SpectralAnalysis\\"
    file = "20181115_PDAC_uMALDI_POS_DHB_slides2and16 Slide 2 - C2.imzML"
    path_outputs = "X:\\Ariadna\\PDAC\\MALDI\\nectar_outputs\\"
    polarity = "positive"
    modality = "maldi"

    '''Read imzML data'''
    # data = reader.read_imzml(path_data + file)
    #
    # '''Create full mean spectrum'''
    # total_mean = dataop.get_mean_spectrum(data)
    # saver.save_spectrum_hdf5(path_outputs + 'total_mean.hdf5', total_mean)
    #
    # '''Separate tissue from background'''
    # data_masked = dataop.background_subtraction(data, total_mean, path_outputs, n_clusters=2, show_plot=True)
    # saver.save_hdf5(path_outputs + '20181115_PDAC_uMALDI_POS_DHB_slides2and16 Slide 2 - C2_masked.hdf5', data_masked)
    data_masked = reader.read_hdf5(path_outputs +
    '20181115_PDAC_uMALDI_POS_DHB_slides2and16 Slide 2 - C2_masked.hdf5')

    '''Creates mean spectra for tissue and background'''
    mean_background = dataop.get_mean_spectrum_tissue_background(data_masked, mean_tissue=False,
                                                                              mean_background=True)
    saver.save_spectrum_hdf5(path_outputs + 'mean_background.hdf5', mean_background)
    #saver.save_spectrum_hdf5(path_outputs + 'mean_tissue.hdf5', mean_tissue)
    mean_tissue = reader.read_hdf5_spectrum(path_outputs + 'mean_tissue.hdf5')
    # mean_background = reader.read_hdf5_spectrum(path_outputs + 'mean_background.hdf5')

    '''Apply noise correction to spectrum'''
    mean_tissue_corrected = noisecorrection.noise_correction_with_chemical_noise(mean_tissue,
                                                                                 plot_noise=True,
                                                                                 plot_chemicalnoise=True)

    saver.save_spectrum_hdf5(path_outputs + 'mean_tissue_corrected.hdf5', mean_tissue_corrected)
    # mean_tissue_corrected = reader.read_hdf5_spectrum(path_outputs + 'mean_tissue_corrected.hdf5')

    '''Peak picking and background contamination removal'''
    list_of_fittings, full_list_of_fittings = peakpicking.peak_picking(mean_tissue_corrected, path_outputs,
                                                                       plot_peaks=True,
                                                                       save_tables=True,
                                                                       save_fitting=True)

    # full_list_of_fittings = pd.read_csv("X:\\Ariadna\\PDAC\\MALDI\\nectar_outputs\\full_list_of_fittings.csv")

    peaks_classification = noisecorrection.backgroundnoise.backgroundnoise_identification(data_masked,
                                                                                          mean_tissue_corrected,
                                                                                          full_list_of_fittings,
                                                                                          path_outputs,
                                                                                          save_plot_backgroundNoise=True
                                                                                          )

    # peaks_classification = pd.read_csv("X:\\Ariadna\\PDAC\\MALDI\\nectar_outputs\\Peaks_classification.csv")

    '''Final selection of peaks according to S/N'''
    final_list = peaks_classification.loc[(peaks_classification['diff[Tis/bak] S/N'] >= 0) &
                                          (peaks_classification['ratio[Tis/bak] S/N'] >= 5)]

    #final_list.to_csv("X:\\Ariadna\\PDAC\\MALDI\\nectar_outputs\\Compounds_of_interest_final_list.csv", index=False)

    '''Save reduced data cubes'''
    saver.save_final_DataCube(data_masked, final_list, path_outputs, save_imzml=True, save_hdf5=True, save_mat=True)

    '''HMDB database cross matching'''
    databasematching.database_matching(final_list, polarity, modality, path_outputs, ppm=30.)

    '''Plot of selected peaks'''
    plotting.plot_sii_final_list(data_masked, mean_tissue_corrected, final_list, path_outputs, save_fig=True)

    print('Finished!')

if __name__ == "__main__":
    main()
