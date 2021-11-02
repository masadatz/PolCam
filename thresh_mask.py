import numpy as np
import cv2

img = cv2.imread('24_14_09_56_065849_101933.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
_, th1 = cv2.threshold(gray, 50,255, cv2.THRESH_TOZERO) #less than 47- to black
cv2.imshow('th1',th1)
_, thn = cv2.threshold(th1, 85, 0 , cv2.THRESH_TOZERO_INV) # above 85 - to black
cv2.imshow('thn',thn)
cv2.waitKey(0)

cv2.destroyAllWindows()