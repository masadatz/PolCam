from roipoly import RoiPoly
from matplotlib import pyplot as plt
import cv2
import numpy as np
import logging


# Create image
img = cv2.imread('24_14_09_56_065849_101933.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img)

# Show the image
fig = plt.figure()
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
plt.title("left click: line segment         right click or double click: close region")
plt.show(block=False)

# Let user draw first ROI
roi1 = RoiPoly(color='r', fig=fig)
#print(len(roi1))
#fig.savefig('to.png')

# Show the image with the first ROI
fig = plt.figure()
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
roi1.display_roi()
plt.title('draw second ROI')
plt.show(block=False)

# Let user draw second ROI
roi2 = RoiPoly(color='b', fig=fig)

 # Show the image with both ROIs and their mean values
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
for roi in [roi1, roi2]:
    roi.display_roi()
    #roi.display_mean(img)
plt.title('The two ROIs')
plt.show()

mask1 = roi1.get_mask(img[:,:,0])
print(img[:,:,0])
img[mask1]=0
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
plt.title("blup")
plt.show()


# # Show ROI masks
# plt.imshow(roi1.get_mask(img) + roi2.get_mask(img),
#            interpolation='nearest', cmap="Greys")
# plt.title('ROI masks of the two ROIs')
# plt.show()


