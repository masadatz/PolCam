from roipoly import RoiPoly
from matplotlib import pyplot as plt
import cv2
import numpy as np
import logging


def put_the_black_masks():
    for roi in lis_rois:
        mask = roi.get_mask(img[:, :, 0])
        img[mask] = 0

def display_rois(ind):
    for i in range (ind):
        lis_rois[i].display_roi()


lis_rois = []

img = cv2.imread('26_09_44_03_456833_101935.png')

cont = True

while cont:
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    display_rois(len(lis_rois))
    plt.title("left click: line segment         right click or double click: close region")
    plt.show(block = False) #maybe we don't need that line?
    roi = RoiPoly(color='r', fig=fig)
    lis_rois.append(roi)

    cont = input("proceed? 'y' for Yes, anything else for No.")
    print (cont)
    cont = True if cont =='y' else False
    print(cont)

put_the_black_masks()

#show finished pic
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
plt.title("finished?!")
#fig.savefig('to.png')
cv2.imwrite('to.png', img=img)
plt.show()




