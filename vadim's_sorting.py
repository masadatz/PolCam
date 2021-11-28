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
def divide_per_day(files_per_cam, idi):
    for file in files_per_cam:
        basename = os.path.basename(file)
        day = basename[0:2]
        sortCamAndDay[idi][day].append(file)

#finally, for each day and camera, sort by the time we took the pic.
def sorting_in_milsec():
    new_a = []
    for i in range (len(ids)):
        for day in lis.keys():
            new_a.clear()
            for basename in sortCamAndDay[ids[i]][day]:
                try:
                    only_time = re.findall(day+'_(.*)'+'_'+ids[i], basename)[0]
                    new_a.append(only_time)
                except:
                    continue

            sortt = sorted(new_a) # the array for each day (with only times), sorted by time
            for j in range(len(sortt)):
                name = day+'_'+sortt[j]+'_'+ids[i]+'.png'
                path =find_name_in_full_path(name,sortCamAndDay[ids[i]][day])
                sortt[j]=path
            sortCamAndDay[ids[i]][day]=sortt



def find_name_in_full_path(name,paths):
    for path in paths:
        if name in path:
            return path


def copy_to_diff_path():
    for i in range (len(ids)):
        for day in lis.keys():
            for path in sortCamAndDay[ids[i]][day]:
                pic = cv2.imread(path)
                basename = os.path.basename(path)
                cv2.imwrite(basename,pic)


init_path = 'C:\\Users\\noaraifler\\נועה - עבודה\\ניסוי במכון הביולוגי\\תמונות מה-USB של הניסוי'
files=getListOfFiles(init_path)
#print(files[0], len (files))
ids= ['101933', '101935', '101934', '101936', '192900073']
#cameras = [i for i in range(5)]
cameras={'101933':[], '101935':[], '101934':[], '101936':[], '192900073':[]}
sortCamAndDay = {'101933':[], '101935':[], '101934':[], '101936':[], '192900073':[]}
dividie_names_per_ind_cam(files)

lis = {'20': [], '24':[], '25':[],'26':[]}
for idi in sortCamAndDay.keys():
    sortCamAndDay[idi] = lis.copy()

print(sortCamAndDay)

for idi in cameras.keys():
    divide_per_day(cameras[idi], idi)

print(sortCamAndDay[ids[1]]['26'])

sorting_in_milsec()
print(sortCamAndDay)



