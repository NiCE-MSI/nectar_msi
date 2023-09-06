"""Class for saving data"""

"""___Built-In Modules___"""
from nectar_msi.data_formats.data_spectrum import DataSpectrum
from nectar_msi.data_formats.data_cube import DataCube

"""___Third-Party Modules___"""
import h5py
from pyimzml.ImzMLWriter import ImzMLWriter
from scipy.io import savemat
import numpy as np
from tqdm import tqdm
import copy


"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class Savers:
    def __init__(self):
        pass

    def save_imzml(self, path_outputs, data_cube):
        """
        Function to save data in imzML format

        :param path_outputs: path to save the data_cube
        :type path_outputs: string
        :param data_cube: data_cube to be saved
        :type data_cube: DataCube
        :return: None
        :rtype: None
        """
        print("Saving imzML file...")
        with ImzMLWriter(path_outputs) as writer:
            for index in range(len(data_cube.coord)):
                spec = data_cube.get_spectrum_idx(data_cube.idxs[index])
                if spec is not None:
                    writer.addSpectrum(spec.mzs, spec.intensities,
                                       (spec.coordinates[0] + 1, spec.coordinates[1] + 1, 1))
        print("imzML saved in " + path_outputs)

    def save_hdf5(self, path_outputs, data_cube):
        """
        Function to save a data_cube into hdf5 format

        :param path_outputs: path to save the data_cube
        :type path_outputs: string
        :param data_cube: data_cube to be saved
        :type data_cube: DataCube
        :return: None
        :rtype: None
        """

        print("Saving hdf5 file...")
        f = h5py.File(path_outputs, "w")
        f.create_dataset("pixelmask", data=data_cube.pixelmask)
        f.create_dataset("idxs", data=data_cube.idxs)
        grp_meta = f.create_group("metadata")
        for key, value in data_cube.cubemetadata.items():
            grp_meta.attrs[key] = value
        grp_data = f.create_group("data")
        grp_data.attrs["shape"] = data_cube.shape
        for index in range(len(data_cube.coord)):
            spec = data_cube.get_spectrum_idx(data_cube.idxs[index])
            if spec is not None:
                grp_spec = grp_data.create_group("spectrum_%s" % data_cube.idxs[index])
                grp_spec.create_dataset("mzs", data=spec.mzs)
                grp_spec.create_dataset("intensities", data=spec.intensities)
                grp_spec.create_dataset("mzmask", data=spec.mzmask)
                grp_spec.attrs["coordinates"] = spec.coordinates
                for key, value in spec.spectrummetadata.items():
                    grp_spec.attrs[key] = value
                for key, value in spec.offsets.items():
                    grp_spec.attrs[key] = value
                del spec
        print("HDF5 saved in " + path_outputs)

    def save_spectrum_hdf5(self, path_outputs, spectrum):
        """
        Function to save a spectrum into hdf5 format

        :param path_outputs: path to save the spectrum
        :type path_outputs: string
        :param spectrum: spectrum to be saved
        :type spectrum: DataSpectrum
        :return: None
        :rtype: None
        """

        print("Saving hdf5 file...")
        f = h5py.File(path_outputs, "w")
        grp_spec = f.create_group("spectrum_data")
        grp_spec.create_dataset("mzs", data=spectrum.mzs)
        grp_spec.create_dataset("intensities", data=spectrum.intensities)
        grp_spec.create_dataset("mzmask", data=spectrum.mzmask)
        grp_spec.attrs["coordinates"] = spectrum.coordinates
        grp_spec.attrs["idx"] = spectrum.idx
        for key, value in spectrum.spectrummetadata.items():
            grp_spec.attrs[key] = value
        print("HDF5 saved in " + path_outputs)

    def save_final_DataCube(self, data_cube, selected_peaks, path_outputs,
                            save_imzml=True, save_hdf5=False, save_mat=False):
        """
        Function to save the de-noised datacube

        :param data_cube: data cube to be saved
        :type data_cube: DataCube
        :param selected_peaks: peaks to be included in the final datacube
        :type selected_peaks: pandas dataframe
        :param path_outputs: path where to save the datacube
        :type path_outputs: str
        :param save_imzml: option to save imzML format
        :type save_imzml: bool
        :param save_hdf5: option to save hdf5 format
        :type save_hdf5: bool
        :param save_mat: option to save .mat format
        :type save_mat: bool
        :return: None
        :rtype: None
        """

        print('Saving final datacube...')
        data_cube_final = DataCube(data_cube.shape, metadata=data_cube.cubemetadata)
        data_cube_final.coord = copy.deepcopy(data_cube.coord)
        data_cube_final.pixelmask = copy.deepcopy(data_cube.pixelmask)
        data_cube_final.idxs = copy.deepcopy(data_cube.idxs)

        data_cube_matfile = np.zeros([len(data_cube.idxs), len(selected_peaks['left min'])])
        peaks = selected_peaks['meas mz'].to_numpy()

        for index in tqdm(range(len(data_cube.idxs))):
            idx = data_cube.idxs[index]
            spectrum_per_pixel = data_cube.get_spectrum_idx(idx)

            xaxis = spectrum_per_pixel.mzs[:]
            intensities = spectrum_per_pixel.intensities[:]
            new_intensities = np.zeros(len(intensities))
            add_intensities = np.zeros(len(selected_peaks['left min']))

            for mzi in range(len(selected_peaks['left min'])):
                new_intensities[
                    np.where((xaxis >= selected_peaks.iloc[mzi, 0]) & (xaxis <= selected_peaks.iloc[mzi, 2]))] = \
                    intensities[
                        np.where((xaxis >= selected_peaks.iloc[mzi, 0]) & (xaxis <= selected_peaks.iloc[mzi, 2]))]

                add_intensities[mzi] = spectrum_per_pixel.get_intensity_between_mzs(selected_peaks.iloc[mzi, 0],
                                                                                    selected_peaks.iloc[mzi, 2])
            data_cube_matfile[index, :] = add_intensities
            spectrum_per_pixel.set_intensities(new_intensities)
            data_cube_final.set_spectrum_idx(idx, spectrum_per_pixel)

        if save_imzml:
            self.save_imzml(path_outputs + 'datacube_reduced.imzML', data_cube_final)
            print('datacube_reduced.imzML saved in ' + path_outputs)
        if save_hdf5:
            self.save_hdf5(path_outputs + 'datacube_reduced.hdf5', data_cube_final)
            print('datacube_reduced.hdf5 saved in ' + path_outputs)
        if save_mat:
            mdic = {"data_cube": data_cube_matfile, "selected_peaks": peaks}
            savemat(path_outputs + "datacube_reduced.mat", mdic)
            print('datacube_reduced.mat saved in ' + path_outputs)

