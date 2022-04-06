import numpy as np
import matplotlib.pyplot as plt


def LoadSizeDistribution(filename):

    data = np.loadtxt(filename, dtype=float)
    Radii = data[:, 0]
    Numpar = data[:, 1]
    DR = Radii[1:] - Radii[0:-1]
    DR = np.concatenate([DR[0:1], DR[:]])
    ScaleFactor = DR / DR[0]
    Numpar = Numpar/ScaleFactor
    Numpar = Numpar / np.sum(Numpar)
    #print(Numpar)

    #plt.figure
    #plt.plot(Radii, Numpar)

    #plt.title(" 'Green' cloud droplets Size distribution")
    #plt.xlabel("Droplet Radius [microns]")
    #plt.ylabel("Number")
    #plt.show()

    return Radii, Numpar