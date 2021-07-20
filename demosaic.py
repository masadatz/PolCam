import cv2
import polanalyser as pa
from PIL import Image
# Based on https://github.com/elerac/polanalyser
img_raw = cv2.imread("spoons.png", 0)

img_demosaiced = pa.demosaicing(img_raw)

img_0, img_45, img_90, img_135 = cv2.split(img_demosaiced)
Image.fromarray(img_0).show()
Image.fromarray(img_45).show()
Image.fromarray(img_90).show()
Image.fromarray(img_135).show()
