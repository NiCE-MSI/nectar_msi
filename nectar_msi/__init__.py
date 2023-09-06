"""nectar - nectar (NoisE CorrecTion AlgoRithm) is a python package for noise correction in Q-ToF MSI instruments."""

from nectar.data_formats.data_cube import DataCube
from nectar.data_formats.data_spectrum import DataSpectrum
from nectar.data_processing.readers import Readers
from nectar.data_processing.savers import Savers
from nectar.data_processing.data_operations import DataOperations
from nectar.noise_correction.noise_determination import NoiseDetermination
from nectar.noise_correction.noise_correction import NoiseCorrection
from nectar.noise_correction.chemical_noise import ChemicalNoise
from nectar.noise_correction.background_noise import BackgroundNoise
from nectar.database_matching.database_matching import DatabaseMatching
from nectar.plotting.plotting import Plotting
from nectar.peak_picking.peak_picking import PeakPicking

__author__ = "Ariadna Gonzalez-Fernandez <ariadna.gonzalez@npl.co.uk>"
__all__ = []

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
