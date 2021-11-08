# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

#!pip install miepython
try:
    import miepython

except ModuleNotFoundError:
    print('miepython not installed. To install, uncomment and run the cell above.')
    print('Once installation is successful, rerun this cell again.')

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Check Cloud droplets distribution')

# create a gamma  distribution
Radii = np.linspace(0.1, 40.0, num=200)
shape, scale = 3., 3.  # mean=4, std=2*sqrt(2)
N_Distribition = Radii**(shape-1)*(np.exp(-Radii/scale) / (sps.gamma(shape)*scale**shape))
N_Distribition = N_Distribition / np.sum(N_Distribition)
plt.plot(Radii, N_Distribition )

Radii, V_Distribition = N2V_distribution(Radii, N_Distribition)
plt.plot(Radii, V_Distribition )
plt.legend(['Number Dist.','Volume Dist.'])
plt.xlabel('Radius [\mu  m]')
plt.ylabel('Probability')
plt.suptitle('Cloud droplets distribution')
plt.show()


Sigma_1_over_m = VisRange2Sigma(30) # 30 meter visibility
VisRange = Sigma2VisRange(Sigma_1_over_m) # we should receive back 30 meter visibility

LWC_gr_cm3 = 0.3*1E-6  # gr/m^3
TotalVDist = LWC2TotalVDist(LWC_gr_cm3, V_Distribition)
plt.plot(Radii, TotalVDist )
plt.legend(['Number Dist.','Volume Dist.'])
plt.xlabel('Radius [\mu  m]')
plt.ylabel('Total Number')
plt.show()