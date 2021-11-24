import cv2
import glob
import os
import re

# All files ending with .png
#files=glob.glob('C:\\Users\\noaraifler\\נועה - עבודה\\ניסוי במכון הביולוגי\\תמונות מה-USB של הניסוי\\*\\*\\*\\*.png')

'''
    For the given path, get the List of all files in the directory tree 
'''


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

#first, we are sorting by the cameras's indexes.
def dividie_names_per_ind_cam(files):
    for i in range(len(files)):
        idi = re.findall("([0-9]{5,}).png", files[i])[0]
        cameras[idi].append(files[i])

#second, from one camera (each),we divide per day:
def divide_per_day(files_per_cam):
    pass



files=getListOfFiles('C:\\Users\\noaraifler\\נועה - עבודה\\ניסוי במכון הביולוגי\\תמונות מה-USB של הניסוי')
print(files[0], len (files))
ids= ['101933', '101935', '101934', '101936', '192900073']
#cameras = [i for i in range(5)]
cameras={'101933':[], '101935':[], '101934':[], '101936':[], '192900073':[]}
print(cameras)

