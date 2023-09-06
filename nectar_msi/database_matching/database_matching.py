"""Class to do the database matching with the HMDB database.
It takes into account the presence of isotopes."""

"""___Built-In Modules___"""

"""___Third-Party Modules___"""
import numpy as np
import pandas as pd

"""___Authorship___"""
__author__ = "Ariadna Gonzalez"
__email__ = "ariadna.gonzalez@npl.co.uk"


class DatabaseMatching:
    def __init__(self):
        pass

    def database_matching(self, list_of_compounds, polarity, modality, path_outputs, ppm=30.0):
        """
        Function to obtain the list of all peaks that match with the HMDB database. The
        information in the output table will be:
        ["hmbdb_id","name","chemical_formula", "monoisotopic_mass","monoisotopic_mass_HMDB", "polarity",
        "mode","adduct", "meas mz","peak intensity","n_lines","n_isotopes",
        "theoretical mass", "ppm", "ppm_abs","kingdom", "super_class","class"]

        :param list_of_compounds: list of peaks
        :type list_of_compounds: pandas DataFrame
        :param polarity: polarity in which the data was acquire
        :type polarity: str
        :param modality: modality in which the data was acquire
        :type modality: str
        :param path_outputs: path to save output table
        :type path_outputs: str
        :param ppm: Absolut value uncertainty on the m/z centroid to cross-match with HMDB
        :type ppm: float
        :return: table with all possible matches
        :rtype: pandas DataFrame
        """

        path_libraries = "C:\\Users\\ag12\\PycharmProjects\\nectar\\HMDB files\\"

        '''Possible adducts for each polarity'''
        if polarity == "positive":
            adducts = {"[M+H]": "1.00727645", "[M+Na]": "22.9892207", "[M+K]": "38.963158", "[M-OH]": "-17.00273965",
                       "[M+H30]": "19.01838971", "[M+NH4]": "18.033823"}  # subtracted electron_mass
        if polarity == "negative":
            adducts = {"[M-H3O]": "-19.01838971", "[M-H]": "-1.007276", "[M+OH]": "17.00273965",
                       "[M+Cl]": "34.969402"}  # added electron_mass electron_mass = 0.00054857990924

        '''initialise final table and read HMDB internal library (this library was updated on 2023)'''
        hmdb_compounds = np.empty((0, 18))
        for keys in adducts.keys():
            library = np.load(path_libraries + "hmdb_isotopes_library_" + polarity + "_" + keys + ".npy",
                              allow_pickle=True)
            print("Finding molecules in the spectrum " + keys + "...")
            for i in range(len(library)):
                hmdb_id = library[i][0]
                name = library[i][1]  # name
                name_form = library[i][2]  # chemical formula
                monoisotopic_mass_hmdb = library[i][3]
                kingdom = library[i][4]
                superclass = library[i][5]
                classs = library[i][6]
                isopeak_location = library[i][7]  # centroid of the isotope
                peak_probability = library[i][8]  # probability of the isotope
                monoisotopic_mass = library[i][9]  # - float(adducts.get(keys))

                AllLinesObserved = True
                threshold = (0.10 * peak_probability[0])  # Probability threshold for the intensities
                for j in range(len(peak_probability)):
                    if peak_probability[j] > threshold:
                        linefound = False

                        for ii in range(len(list_of_compounds["meas mz"])):
                            if (abs(((isopeak_location[j] - float(list_of_compounds.iloc[ii, 1])) /
                                     float(list_of_compounds.iloc[ii, 1])) * 10**6) < ppm):
                                linefound = True
                        if linefound == False:
                            AllLinesObserved = False

                if AllLinesObserved == True:
                    counter = 1
                    for jj in range(len(peak_probability)):
                        if peak_probability[jj] > threshold:
                            linefound = False
                            for ii in range(len(list_of_compounds["meas mz"])):
                                if (abs(((isopeak_location[jj] - float(list_of_compounds.iloc[ii, 1])) /
                                         float(list_of_compounds.iloc[ii, 1]) * 10**6)) < ppm):

                                    ppm_dif = ((isopeak_location[jj] - float(list_of_compounds.iloc[ii, 1])) /
                                               float(list_of_compounds.iloc[ii, 1])) * 10**6
                                    ppm_abs = abs(((isopeak_location[jj] - float(list_of_compounds.iloc[ii, 1])) /
                                                   float(list_of_compounds.iloc[ii, 1]) * 10**6))
                                    linefound = True

                                    hmdb_compounds = np.append(hmdb_compounds, np.array([[
                                                    hmdb_id,
                                                    name,
                                                    name_form,
                                                    monoisotopic_mass,
                                                    monoisotopic_mass_hmdb,
                                                    polarity,
                                                    modality,
                                                    str(keys),
                                                    list_of_compounds.iloc[ii, 1],
                                                    list_of_compounds.iloc[ii, 5],
                                                    counter,
                                                    jj,
                                                    isopeak_location[jj],
                                                    ppm_dif,
                                                    ppm_abs,
                                                    str(kingdom),
                                                    str(superclass),
                                                    str(classs)]]), axis=0)
                                    counter += 1  # Gives me the number of detected isotopes

        hmdb_present_compounds = pd.DataFrame(hmdb_compounds, columns=[
                "hmbdb_id",
                "name",
                "name_form",
                "monoisotopic_mass",
                "monoisotopic_mass_HMDB",
                "polarity",
                "mode",
                "adduct",
                "meas mz",
                "peak intensity",
                "n_lines",
                "n_isotopes",
                "theoretical mass",
                "ppm",
                "ppm_abs",
                "kingdom",
                "super_class",
                "class"])

        hmdb_present_compounds.to_csv(path_outputs + "HMDB_compounds_of_interest.csv", index=False)
        print("! List of peaks saved in: " + path_outputs + "HMDB_compounds_of_interest.csv")

        return

