from roipoly import RoiPoly
from matplotlib import pyplot as plt
import cv2
import numpy as np
import logging
import os.path
import os
import re

'''
for now it's hardcoded... We need to know all the details about when and how
we ran the script in order to change it.
'''

geo_calib = 'G13'
name_file = '26_13_04_32_394234_192900073.png'

def find_ID_from_name(name):
    idi= re.findall("([0-9]{5,}).png", name)
    print(idi)

def find_mask_from_calib_and_id(name_calib, idi):
    path1 ='C:\\Users\\noaraifler\\נועה - עבודה\\ניסוי במכון הביולוגי\\מסיכות ל10 תמונות\\G14\\מסיכות'
    path2 = 'C:\\Users\\noaraifler\\נועה - עבודה\\ניסוי במכון הביולוגי\\מסיכות ל10 תמונות\\G13\\מסיכות'

    right_path = path1 if name_calib=='G14' else path2
    all_names=os.listdir(right_path)
    os.chdir(right_path)
    print(all_names)
    right_name = ''
    for name in all_names:
        if re.search(str(idi),name):
            right_name = name
            break
    mask=cv2.imread(right_name)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


find_mask_from_calib_and_id(geo_calib, '192900073')