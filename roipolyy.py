from roipoly import RoiPoly
from matplotlib import pyplot as plt
import cv2
import numpy as np
import logging
import os.path


def put_the_black_masks():
    ind=0
    for roi in lis_rois:
        mask = roi.get_mask(img[:, :, 0])
        img[mask] = 0
        if ind==0:
            img[~mask]=255
        ind+=1

def display_rois(ind):
    for i in range (ind):
        lis_rois[i].display_roi()


lis_rois = []
name_of_file = '26_13_04_32_394234_192900073.png'
img = cv2.imread(name_of_file)

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
#img= cv2.threshold(img, 0, 255 , cv2.THRESH_TOZERO_INV) # above 0 - to white
#cv2.imshow('img', img)
#cv2.waitKey(0)
save_path = 'C:\\Users\\noaraifler\\נועה - עבודה\\ניסוי במכון הביולוגי\\מסיכות ל10 תמונות\\G13\\מסיכות'
os.chdir(save_path)
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.colorbar()
plt.title("finished?!")
#fig.savefig('to.png')
cv2.imwrite(name_of_file, img=img)
plt.show()



