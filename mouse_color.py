import numpy as np
import cv2


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,',' ,y,' color:',str(gray[y][x]))


img = cv2.imread('24_14_09_56_065849_101933.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray, print(type(gray)), print(len(gray)), print(len(gray[0])))
#print(img, print(type(img)))

cv2.imshow('image', gray)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()