import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
import pandas as pd



def LoadProjectorSpectrum(filename):
    data = np.loadtxt(filename, skiprows=17, dtype=float)
    Wavelength_nm = data[:, 0]
    Intensity = data[:, 1]
    Dwl = Wavelength_nm[1:] - Wavelength_nm[0:-1]
    tmp = np.concatenate([Dwl[0:1], Dwl[:]])
    Dwl = tmp
    ScaleFactor = Dwl / Dwl[0]
    Intensity = Intensity / ScaleFactor
    Intensity = Intensity / np.sum(Intensity)

    #plt.figure(301)
    #plt.plot(Wavelength_nm, Intensity)

    #plt.title(" Projector spectral intensity")
    #plt.xlabel("wavelength [nm]")
    #plt.ylabel("I")
    #plt.show()

    return Wavelength_nm, Intensity


def load_photopic_function(filename):
    #columns = ['Wavelength', 'photopic', 'Scotopic']
    data = pd.read_excel(filename, header=None)# names=columns)


    tmp = data.to_numpy()
    Wavelength_photopic_nm = tmp[:, 0]
    photopic_response = tmp[:, 1]
    photopic_response = photopic_response / np.max(photopic_response)

    #plt.figure(305)
    #plt.title(" Photopic response")
    #plt.xlabel("wavelength [nm]")

    #plt.plot(Wavelength_photopic_nm, photopic_response)

    #plt.show()

    return Wavelength_photopic_nm, photopic_response
