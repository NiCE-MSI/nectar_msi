"""Class for reading data"""

"""___Built-In Modules___"""
from nectar_msi.data_formats.data_cube import DataCube
from nectar_msi.data_formats.data_spectrum import DataSpectrum

"""___Third-Party Modules___"""
from pyimzml.ImzMLParser import ImzMLParser
import h5py

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"

class Readers:
    def __init__(self):
        pass

    def read_imzml(self, filepath):
        """
        Function to read imzML data and convert it into DataCube class

        :param filepath: path where the data are located
        :type filepath: string
        :return: data_cube object
        :rtype: DataCube
        """
        p = ImzMLParser(filepath)
        dict = p.imzmldict
        x_shape = dict.get("max count of pixels x")
        y_shape = dict.get("max count of pixels y")
        shape = (x_shape, y_shape)
        datac = DataCube(shape)
        print("Image shape", x_shape, y_shape)
        print("Reading imzML file at: " + filepath)
        for idx, (x, y, z) in enumerate(p.coordinates):
            mzs, intensities = p.getspectrum(idx)
            spectrum = DataSpectrum(idx, [x - 1, y - 1], mzs, intensities)
            datac.set_spectrum_idx(idx, spectrum)
            del spectrum
        print("Reading imzML done!")
        return datac

    def read_imzml_missing(self, filepath):
        """
        Function to read imzML data with missing data points (NaNs)

        :param filepath: path where the data are located
        :type filepath: string
        :return: data_cube object
        :rtype: DataCube
        """
        p = ImzMLParser(filepath)
        dict = p.imzmldict
        x_shape = dict.get("max count of pixels x")
        y_shape = dict.get("max count of pixels y")
        shape = (x_shape, y_shape)
        datac = DataCube(shape)
        print("Image shape", x_shape, y_shape)
        print("Reading imzML file at: " + filepath)
        for indexp, (x, y, z) in enumerate(p.coordinates):
            mzs, intensities = p.getspectrum(indexp)
            idx = datac.get_idx_from_coor([x - 1, y - 1])
            spectrum = DataSpectrum(idx, [x - 1, y - 1], mzs, intensities)
            datac.set_spectrum_idx(idx, spectrum)
            del spectrum
        print("Reading imzML done!")
        return datac

    def read_hdf5(self, filepath):
        """
        Function to read hdf5 datacubes

        :param filepath: path where the data is located
        :type filepath: string
        :return: data_cube object
        :rtype: DataCube object
        """
        print("Reading hdf5 file at: " + filepath)
        f = h5py.File(filepath, "r")
        datac = DataCube(f["data"].attrs["shape"], f["idxs"], f["pixelmask"], f["metadata"].attrs)
        for idx in datac.idxs:
            speckey = "spectrum_%s" % idx
            metadata = {}
            offsets = {}
            for key, value in f["data"][speckey].attrs.items():
                if key == "coordinates":
                    coordinates = value
                elif "offset" in key:
                    offsets[key] = value
                else:
                    metadata[key] = value
            spectrum = DataSpectrum(idx, coordinates, f["data"][speckey]["mzs"], f["data"][speckey]["intensities"],
                                    f["data"][speckey]["mzmask"], offsets, metadata)
            datac.set_spectrum_idx(idx, spectrum)
            del spectrum
        print("Reading hdf5 done!")
        return datac

    def read_hdf5_spectrum(self, filepath):
        """
        Function to read a hdf5 spectrum

        :param filepath: path where the spectrum is located
        :type filepath: string
        :return: spectrum object
        :rtype: DataSpectrum
        """
        print("Reading hdf5 file at: " + filepath)
        f = h5py.File(filepath, "r")
        metadata = {}
        offsets = {}
        for key, value in f["spectrum_data"].attrs.items():
            if key == "idx":
                idx = value
            elif key == "coordinates":
                coordinates = value
            elif "offset" in key:
                offsets[key[7::]] = value
            else:
                metadata[key] = value
        spectrum = DataSpectrum(idx, coordinates, f["spectrum_data"]["mzs"],
                                f["spectrum_data"]["intensities"], f["spectrum_data"]["mzmask"], offsets, metadata)
        print("Reading hdf5 done!")
        return spectrum
