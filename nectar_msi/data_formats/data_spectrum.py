"""Class for creating and reading spectra"""

"""___Third-Party Modules___"""
import numpy as np

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class DataSpectrum:
    def __init__(self, idx, coordinates, mzs, intensities, mzmask=None, offsets=None, metadata={}):
        """
        DataSpectrum class initialiser

        :param idx: Id of the spectrum in the datacube
        :type idx: int
        :param coordinates: coordinates of the spectrum in the data_cube
        :type coordinates: tuple
        :param mzs: mz values in the spectrum
        :type mzs: array float64
        :param intensities: intensity values in the spectrum
        :type intensities: array float32
        :param mzmask: mask to mask part of the mz values (default to zeros)
        :type mzmask: array of int (optional)
        :param offsets: given reference mzs values, offsets to those references (default to empty)
        :type offsets: dict (optional)
        :param metadata:NOT DEVELOPED - Metadata of the spectrum (default to empty)
        :type metadata: dict (optional)
        """
        self.idx = idx
        self.coordinates = coordinates
        self.mzs = np.array(mzs, dtype="float64")
        self.intensities = np.array(intensities, dtype="float32")
        if mzmask is None:
            self.mzmask = np.zeros_like(mzs, dtype="u2")
        else:
            self.mzmask = np.array(mzmask, dtype="u2")
        if offsets is None:
            self.offsets = {}
        else:
            self.offsets = offsets
        self.spectrummetadata = metadata

    def set_mzs(self, mzs_new):
        """
        Function to set new mzs values in the spectrum

        :param mzs_new: new mzs values to set
        :type mzs_new: np.array
        :return: new mzs values in the spectrum
        :rtype: array of floats
        """
        self.mzs = np.array(mzs_new, dtype="float64")

    def set_intensities(self, new_intensities):
        """
        Funtion to set new intensity values in the spectrum

        :param new_intensities: new intensity values to set
        :type new_intensities: np.array
        :return: new intensity values in the spectrum
        :rtype: array of floats
        """
        self.intensities = np.array(new_intensities, dtype="float32")

    def set_mzmask(self, new_mask):
        """
        Function to set the mzmask in the spectrum class

        :param new_mask: mask to be included in the spectrum class
        :type new_mask: np.array
        :return: mask
        :rtype: array of int
        """
        self.mzmask = np.array(new_mask, dtype="u2")

    def set_offset(self, mz, offset):
        """
        Function to set the dictionary of offset values

        :param mz: mz reference value
        :type mz: float
        :param offset: offset values
        :type offset: float
        :return: dictionary of offset values
        :rtype: dict
        """
        self.offsets["offset_" + str(mz)] = offset

    def add_metadata(self, key, value):
        """
        Function to add the metadata of the spectrum  (NOT TESTED)

        :param key: attribute of the dictionary
        :type key: string
        :param value: value of the attribute
        :type value: string
        :return: dictionary of the metadata
        :rtype: dict
        """
        self.spectrummetadata[key] = value

    def get_intensity_at_mz(self, mz):
        """
        Function to obtain the intensity at a given mz value in the spectrum

        :param mz: centroid of interest
        :type mz: float
        :return: intensity at the given mz value
        :rtype: float
        """
        return self.intensities[np.where(self.mzs == mz)]

    def get_intensity_between_mzs(self, mzlow, mzhigh):
        """
        Function to calculate the intensity between two mz values. The intensity is the sum
        of all the intensities between these two values

        :param mzlow: left value of the interval
        :type mzlow: float
        :param mzhigh: right value of the interval
        :type mzhigh: float
        :return: Total intensity between two mz values
        :rtype: float
        """
        return np.sum(self.intensities[np.where((self.mzs >= mzlow) & (self.mzs <= mzhigh))])

    def get_offset(self, mz):
        """
        Function to calculate the offset of one mz value given a mz reference value

        :param mz: reference centroid of interest
        :type mz: float
        :return: offset of the mz value in the spectrum
        :rtype: dict
        """
        try:
            return self.offsets["offset_" + str(mz)]
        except:
            return np.nan
