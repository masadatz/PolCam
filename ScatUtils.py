# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from LoadSizeDistribution import *

#!pip install miepython
try:
    import miepython

except ModuleNotFoundError:
    print('miepython not installed. To install, uncomment and run the cell above.')
    print('Once installation is successful, rerun this cell again.')




def N2V_distribution(Radii, N_Distribition):
    V_Distribition = N_Distribition
    for  i in range(len(Radii)):
      volume = (4*np.pi/3) * Radii[i]**3
      V_Distribition[i] = N_Distribition[i] * volume

    V_Distribition = V_Distribition / np.sum(V_Distribition)
    return Radii, V_Distribition


def VisRange2Sigma(VisRange_m):
    Sigma_1_over_m = 3.912 / VisRange_m
    return Sigma_1_over_m

def Sigma2VisRange(Sigma_1_over_m):
    VisRange_m   = 3.912 / Sigma_1_over_m
    return VisRange_m

def Visibilty2OpticalDepth(VisRange_m, Length_m):
      Sigma_1_over_m = 3.912 / VisRange_m          # Ref. : https://en.wikipedia.org/wiki/Visibility x_\text{V} = \frac{3.912}{b_\text{ext}}
      transmittance = np.exp(-Sigma_1_over_m*Length_m)
      OpticalDepth = -np.log(transmittance)        # OpticalDepth = -log(Trans) REf. : https://en.wikipedia.org/wiki/Optical_depth
      return OpticalDepth

def LWC2TotalVDist(LWC_gr_cm3, V_Distribition):
    WaterDensity_gr_cm3 = 0.99802  # gr/cm3
    TotalVDist = V_Distribition * (LWC_gr_cm3 / WaterDensity_gr_cm3)
    TotalVDist = TotalVDist

    return TotalVDist

def LWC2Visibility(C_ext, Cloud_LWC_gr_cm3, Radii_micron, V_disrib):

    print(f'Not active yet')
    WaterDensity_gr_cm3 = 0.99802  # gr/cm3
    Norm_C_ext = np.zeros(np.length(Radii_micron))   # distribution normalised extinction cross section

    for i in range(len(Radii)):
        Norm_C_ext[i] = V_disrib[i] * C_ext[i]
        Sigma_1_over_m = np.sum(Norm_C_ext) * (Cloud_LWC_gr_cm3 / WaterDensity_gr_cm3)*1E6

        VisRange_m = Sigma2VisRange(Sigma_1_over_m)
    return VisRange_m

def MieCalc(Wavelength, Radii):
    # import the Segelstein data
    h2o = np.genfromtxt('http://omlc.org/spectra/water/data/segelstein81_index.txt', delimiter='\t', skip_header=4)
    h2o_lam = h2o[:, 0]
    h2o_mre = h2o[:, 1]
    h2o_mim = h2o[:, 2]

    # plot it
    plt.figure(3)
    plt.plot(h2o_lam, h2o_mre)
    plt.plot(h2o_lam, h2o_mim)
    plt.xlim((1, 15))
    plt.ylim((0, 1.8))
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Refractive Index')
    plt.annotate(r'$m_\mathrm{re}$', xy=(3.4, 1.5))
    plt.annotate(r'$m_\mathrm{im}$', xy=(3.4, 0.2))

    plt.title('Complex Refractive Index of Water')

    plt.show(block=False)


    #x = np.linspace(0.1, 100, 300)
    #refWavelength = [0.200, 0.225, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525,
    #                 0.550, 0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725, 0.750, 0.775, 0.800, 0.825, 0.850, 0.875,
    #                 0.900, 0.925, 0.950, 0.975, 1.0,   1.2,   1.4,   1.6,   1.8,   2.0]
    #ref_n         = [1.396, 1.373, 1.362, 1.354, 1.349, 1.346, 1.343, 1.341, 1.339, 1.338, 1.337, 1.336, 1.335, 1.334,
    #                 1.333, 1.333, 1.332, 1.332, 1.331,	1.331, 1.331, 1.330, 1.330, 1.330, 1.329, 1.329, 1.329, 1.328,
    #                 1.328, 1.328, 1.327, 1.327, 1.327, 1.324, 1.321, 1.317, 1.312, 1.306]

    #ref_k = [1.1e−7, 4.9e−8 , 3.35e−8, 2.35e−8, 1.6e−8, 1.08e−8, 6.5e−9, 3.5e−9, 1.86e−9, 	1.3e−9, 1.02e−9, 9.35×10−10, 1.00e−9,
    # 1.32e−9, 1.96e−9, 3.60e−9, 1.09e−8, 1.39e−8, 1.64e−8, 2.23e−8, 3.35e−8, 9.15e−8, 1.56e−7, 1.48e−7, 1.25e−7, 1.82e−7, 2.93e−7,
    # 3.91e−7, 4.86e−7, 1.06e−6, 2.93e−6, 3.48e−6, 	2.89e−6, 9.89e−6, 1.38e−4, 8.55e−5, 1.15e−4, 1.1e−3

    x = (2 * np.pi * Radii) / Wavelength     # size parameter in vacuume https://miepython.readthedocs.io/en/latest/01_basics.html
    ref_n_wl = np.interp(Wavelength, h2o_lam, h2o_mre)
    ref_k_wl = np.interp(Wavelength, h2o_lam, h2o_mim)
    qext, qsca, qback, g = miepython.mie(ref_n_wl - 1.0j * ref_k_wl, x)  # https://miepython.readthedocs.io/en/latest/
    # cross_section_area = np.pi * radius ** 2
    #sca_cross_section = qsca * cross_secton_area
    #abs_cross_section = (qext - qsca) * cross_section_area


    plt.figure(4)
    plt.plot(Radii, qext, color='red', label=str(ref_n_wl))

    plt.title("Water droplets Qext")
    plt.xlabel("Droplet Radius [microns]")
    plt.ylabel("Qext")
    #plt.show(block=True)
    plt.show(block=False)

    theta = np.linspace(-180, 180, 1800)
    mu = np.cos(theta / 180 * np.pi)
    s1, s2 = miepython.mie_S1_S2(ref_n_wl - 1.0j * ref_k_wl, x[100], mu)
    scat = 5 * (abs(s1) ** 2 + abs(s2) ** 2) / 2  # unpolarized scattered light
    plt.figure(5)
    plt.polar(theta/180 * np.pi, np.log10(scat))


    plt.show()

    return qext, qsca, qback, theta, scat

#----------- Start of test code ---------------
print('Test Cloud droplets distribution functions')

DEBUG = 0
if DEBUG == 1:

    # try distribution functions conversions
    shape, scale = 3., 3.  # mean=4, std=2*sqrt(2)
    Radii = np.linspace(0.1, 40.0, num=200)
    N_Distribition = Radii**(shape-1)*(np.exp(-Radii/scale) / (sps.gamma(shape)*scale**shape)) # create a gamma  distribution
    N_Distribition = N_Distribition / np.sum(N_Distribition)

else:
    Radii, N_Distribition = LoadSizeDistribution()

plt.figure(1)
plt.plot(Radii, N_Distribition)
plt.title(" 'Green' cloud droplets Size distribution")
plt.xlabel("Droplet Radius [microns]")
plt.ylabel("Number")


Radii, V_Distribition = N2V_distribution(Radii, N_Distribition)
plt.plot(Radii, V_Distribition )
plt.legend(['Number Dist.', 'Volume Dist.'])
plt.xlabel('Radius [\mu  m]')
plt.ylabel('Probability')
plt.suptitle('Cloud droplets distribution')
plt.show(block=False)


Sigma_1_over_m = VisRange2Sigma(30) # 30 meter visibility
VisRange = Sigma2VisRange(Sigma_1_over_m) # we should receive back 30 meter visibility

LWC_gr_cm3 = 0.3*1E-6  # gr/m^3
TotalVDist = LWC2TotalVDist(LWC_gr_cm3, V_Distribition)
plt.figure(2)
plt.plot(Radii, TotalVDist )
plt.legend(['Number Dist.','Volume Dist.'])
plt.xlabel('Radius [\mu  m]')
plt.ylabel('Total Number')
plt.show(block=False)

# try visibility to optical depth function
OD = Visibilty2OpticalDepth(30, 30)

# Run Mie Calculations
Wavelength = 0.55 # microns
MieCalc(Wavelength, Radii)