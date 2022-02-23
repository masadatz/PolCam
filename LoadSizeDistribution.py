import numpy as np
import matplotlib.pyplot as plt


def LoadSizeDistribution(filename):

    data = np.loadtxt(filename, dtype=float)
    Radii = data[:, 0]
    Numpar = data[:, 1]
    #print(Numpar)

    #plt.figure
    #plt.plot(Radii, Numpar)

    #plt.title(" 'Green' cloud droplets Size distribution")
    #plt.xlabel("Droplet Radius [microns]")
    #plt.ylabel("Number")
    #plt.show()

    return Radii, Numpar