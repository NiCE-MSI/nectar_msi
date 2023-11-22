"""nectar_msi - nectar_msi (NoisE CorrecTion AlgoRithm) is a python package for noise correction
in Q-ToF MSI instruments."""

from nectar_msi.data_formats.data_cube import DataCube
from nectar_msi.data_formats.data_spectrum import DataSpectrum
from nectar_msi.data_processing.readers import Readers
from nectar_msi.data_processing.savers import Savers
from nectar_msi.data_processing.data_operations import DataOperations
from nectar_msi.noise_correction.noise_determination import NoiseDetermination
from nectar_msi.noise_correction.noise_correction import NoiseCorrection
from nectar_msi.noise_correction.chemical_noise import ChemicalNoise
from nectar_msi.noise_correction.background_noise import BackgroundNoise
from nectar_msi.database_matching.database_matching import DatabaseMatching
from nectar_msi.plotting.plotting import Plotting
from nectar_msi.peak_picking.peak_picking import PeakPicking

__author__ = "Ariadna Gonzalez-Fernandez <ariadna.gonzalez@npl.co.uk>"
__all__ = []

from ._version import __version__
