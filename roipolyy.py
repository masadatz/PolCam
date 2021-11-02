from roipoly import RoiPoly
from matplotlib import pyplot as plt
import cv2
import numpy as np
import logging

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Create image
img = np.ones((100, 100)) * range(0, 100)

# Show the image

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.imshow(img)
ax.set_title("left click: line segment         right click or double click: close region")
my_roi = RoiPoly(fig=fig, ax=ax, color='r', close_fig=True)  # draw new ROI in red color
print(my_roi)
# fig = plt.figure()
# plt.imshow(img, interpolation='nearest', cmap="Greys")
# plt.colorbar()
# plt.title("left click: line segment         right click or double click: close region")
# # plt.show(block=False)
#
# # Let user draw first ROI
# roi1 = RoiPoly(color='r', fig=fig)

# Show the image with the first ROI
# fig = plt.figure()
# plt.imshow(img, interpolation='nearest', cmap="Greys")
# plt.colorbar()
# roi1.display_roi()
# plt.title('draw second ROI')
# plt.show(block=False)
#
# # Let user draw second ROI
# roi2 = RoiPoly(color='b', fig=fig)
#
# # Show the image with both ROIs and their mean values
# plt.imshow(img, interpolation='nearest', cmap="Greys")
# plt.colorbar()
# for roi in [roi1, roi2]:
#     roi.display_roi()
#     roi.display_mean(img)
# plt.title('The two ROIs')
# plt.show()
#
# # Show ROI masks
# plt.imshow(roi1.get_mask(img) + roi2.get_mask(img),
#            interpolation='nearest', cmap="Greys")
# plt.title('ROI masks of the two ROIs')
plt.show()


# image = cv2.imread('24_14_09_56_065849_101933.png')
# fig = plt.figure()
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(image)
# plt.show(block=False)
#
# my_roi = RoiPoly(color='r', fig=fig) # draw new ROI in red color
#
# print('yay')
# #my_roi.display_roi()