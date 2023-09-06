"""Class for creating and reading data_cubes"""

"""___Third-Party Modules___"""
import numpy as np

"""___Authorship___"""
__author__ = "Ariadna Gonzalez-Fernandez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class DataCube:
    def __init__(self, shape, idxs=None, pixelmask=None, metadata={}):
        """
        Datacube class initialiser

        :param shape: shape of the datacube
        :type shape: tuple
        :param idxs: number of ids of the spectra in the datacube (defaults to None,
        in which case uses the shape to determine number)
        :type idxs: int (optional)
        :param pixelmask: mask to indicate Regions Of Interest (defaults to zeros)
        :type pixelmask: array of int (optional)
        :param metadata: NOT DEVELOPED - Metadata of the cube (default to empty)
        :type metadata: dict (optional)
        """
        self.shape = shape
        self.data = np.empty((shape[0] * shape[1]), dtype=object)
        self.coord = np.array([(i, j) for j in range(shape[1]) for i in range(shape[0])])
        if idxs is None:
            self.idxs = np.arange(shape[0] * shape[1])
        else:
            self.idxs = np.array(idxs, dtype="int")
        if pixelmask is None:
            self.pixelmask = np.zeros(shape[0] * shape[1])
        else:
            self.pixelmask = np.array(pixelmask, dtype="u2")
        self.cubemetadata = metadata

    def get_coor_from_idx(self, idx):
        """
        Function to obtain the coordinates of the spectrum of interest

        :param idx: id of the spectra in the data_cube
        :type idx: int
        :return: coordinates in the data_cube
        :rtype: tuple
        """
        index = np.argmin(abs(idx - self.idxs))
        return self.coord[index]

    def get_idx_from_coor(self, coor):
        """
        Function to obtain the idx given the coordinates of the pixel in the data_cube

        :param coor: coordinates in the data_cube
        :type coor: np.array
        :return: id of the spectrum
        :rtype: int
        """
        return int(self.idxs[np.where((self.coord[:, 0] == coor[0]) & (self.coord[:, 1] == coor[1]))])

    def get_index_from_coor(self, coor):
        """
        Function to obtain the index given the coordinates of the pixel in the data_cube

        :param coor: coordinates in the data_cube
        :type coor: np.array
        :return: index of the array
        :rtype: int
        """
        return coor[0] + coor[1] * self.shape[0]

    def get_spectrum_coor(self, coor):
        """
        Function to obtain the spectrum of a pixel given the coordinates

        :param coor: coordinates of the pixel in the data_cube
        :type coor: tuple
        :return: spectrum of certain pixel
        :rtype: object
        """
        index = self.get_index_from_coor(coor)
        return self.data[index]

    def get_spectrum_idx(self, idx):
        """
        Function to obtain the spectrum of a given idx in the data_cube

        :param idx: idx of the data_cube
        :type idx: int
        :return: spectrum of certain idx
        :rtype: DataSpectrum object
        """
        try:
            return self.data[idx == self.idxs][0]
        except:
            print("The idx given does not seem to be valid.")

    def set_spectrum_coor(self, coor, spectrum):
        """
        Function to set a new spectrum in an specific location given in coordinates

        :param coor: coordinates where to set the new spectrum
        :type coor: tuple
        :param spectrum: spectrum to set in the data_cube
        :type spectrum: DataSpectrum object
        :return: modified data_cube
        :rtype: DataCube
        """
        index = self.get_index_from_coor(coor)
        self.data[index] = spectrum

    def set_spectrum_idx(self, idx, spectrum):
        """
        Function to set a new spectrum in an specific idx location

        :param idx: idx where to set the new spectrum
        :type idx: int
        :param spectrum: spectrum to set in the data_cube
        :type spectrum: DataSpectrum object
        :return: modified data_cube
        :rtype: DataCube
        """
        index = np.argmin(abs(idx - self.idxs))
        self.data[index] = spectrum

    def set_mzs_coor(self, idx, mzs_new):
        """
        Function to set new mz values in a specific pixel selected by its coordinates

        :param idx: id where to set the new mz values
        :type idx: int
        :param mzs_new: new mz values
        :type mzs_new: array
        :return: modified data_cube
        :rtype: DataCube
        """
        index = np.argmin(abs(idx - self.idxs))
        self.data[index].set_mzs(mzs_new)

    def set_mzs_idx(self, idx, mzs_new):
        """
        Function to set new mz values in a specific pixel selected by its idx

        :param idx: idx of the target pixel in the data_cube
        :type idx: int
        :param mzs_new: new mz values
        :type mzs_new: array
        :return: modified data_cube
        :rtype: DataCube
        """
        index = np.argmin(abs(idx - self.idxs))
        self.data[index].set_mzs(mzs_new)

    def set_pixel_mask_1D(self, new_mask):
        """
        Function to set the mask in the data_cube

        :param new_mask: mask to be set
        :type new_mask: intarray
        :return: data_cube with mask
        :rtype: DataCube
        """
        if new_mask.shape == self.pixelmask.shape:
            self.pixelmask = np.array(new_mask, dtype="u2")
        else:
            print("Error: Mask has the wrong shape")

    def set_pixel_mask_2D(self, new_mask):
        """
        Function to set the mask in the data_cube

        :param new_mask: mask to be set
        :type new_mask: intarray
        :return: data_cube with mask
        :rtype: DataCube
        """
        if new_mask.shape[0] == self.shape[0] and new_mask.shape[1] == self.shape[1]:
            self.pixelmask = new_mask.T.flatten()
        else:
            print("Error: Mask has the wrong shape")

    def set_pixel_mask_idx(self, idx, new_mask):
        """
        Function to set the mask in a specific pixel

        :param idx:  idx of the target pixel in the data_cube
        :type idx: int
        :param new_mask: mask to be set
        :type new_mask: intarray
        :return: data_cube with mask
        :rtype: DataCube
        """
        index = np.argmin(abs(idx - self.idxs))
        self.pixelmask[index] = new_mask

    def add_metadata(self, key, value):
        """
        Function to include the metadata in the data_cube (NOT TESTED)

        :param key: attribute of the dictionary
        :type key: string
        :param value: value of the attribute
        :type value: string
        :return: dictionary of the metadata
        :rtype: dict
        """
        self.cubemetadata[key] = value

    def get_pixel_mask_1D(self):
        """
        Function to obtain the mask in 1D

        :return: mask
        :rtype: intarray
        """
        return self.pixelmask

    def get_pixel_mask_2D(self):
        """
        Function to obtain the mask in 2D (image shape)

        :return: mask
        :rtype: intarray
        """
        mask = np.zeros(self.shape)
        for i in range(len(self.coord)):
            mask[self.coord[i][0], self.coord[i][1]] = self.pixelmask[i]
        return mask

    def get_intensities_at_mz(self, mz):
        """
        Function to read the intensity in each spectra for a given mz value to create
        the single ion images, for instance. The intensity is a single reading in each spectra

        :param mz: mz value of interest
        :type mz: float
        :return: Array with the intensity values at the mz location
        :rtype: 2D array
        """
        intensities = np.zeros(self.shape)
        for i in range(len(self.coord)):
            intensities[self.coord[i][0], self.coord[i][1]] = self.data[i].get_intensity_at_mz(mz)
        return intensities

    def get_intensities_between_mzs(self, mzlow, mzhigh):
        """
        Function to obtain the intensities between two values in each spectra to create
        single ion images, for instance. The intensity is the sum between these two values.

        :param mzlow: left value of the interval
        :type mzlow: float
        :param mzhigh: right value of the interval
        :type mzhigh: float
        :return: Array with the intensity values between the interval limits
        :rtype: 2D array
        """
        intensities = np.zeros(self.shape)
        for i in range(len(self.coord)):
            if self.data[i] is not None:
                intensities[self.coord[i][0], self.coord[i][1]] = self.data[i].get_intensity_between_mzs(mzlow, mzhigh)
            else:
                intensities[self.coord[i][0], self.coord[i][1]] = np.nan
        return intensities

    def get_offsets_at_mz(self, mz):
        """
        Function to obtain the offset in each pixel of the data_cube for a given mz reference value

        :param mz: centroid of the peak of interest
        :type mz: array or list
        :return: array with the offset values
        :rtype: 2D array
        """
        offsets = np.zeros(self.shape)
        for i in range(len(self.coord)):
            if self.data[1] is not None:
                offsets[self.coord[i][0], self.coord[i][1]] = self.data[i].get_offset(mz)
            else:
                offsets[self.coord[i][0], self.coord[i][1]] = np.nan
        return offsets
