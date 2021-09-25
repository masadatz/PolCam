
'''
decide for a "main" camera, (1) - according to it the global coordinate would be.
a stereo calibration with 1-2, 1-3, 1-4, 1-5.
outputs: E and F matrix for each dual. those matrix would help us convert a
point in some camera coordinate, to the "global" coordinate, determined by camera 1.

but first, we do calibration for each camera separately.
'''
import numpy as np
import cv2 as cv
import glob

'''
make equal length of vectors from 2 differente cameras.
'''

def equaly_length(ind_other):
    list0=all_img_points_success[0]
    list_other=all_img_points_success[ind_other]
    new0=[]
    new_other=[]
    for i in range (len(list0)):
        if list0[i]==False: continue
        if list_other[i]==False: continue
        new0.append(list0[i])
        new_other.append(list_other[i])

    len_min = len(new0)
    return len_min,new0, new_other

'''
change background of vadim's shdom images to white
'''

def convert_bk_to_white(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([0, 0, 0], dtype=np.uint8)
    mask = cv.inRange(img, lower_white, upper_white)  # could also use threshold
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
    mask = cv.bitwise_not(mask)  # invert mask

    # load background (could be an image too)
    bk = np.full(img.shape, 255, dtype=np.uint8)  # black bk

    # get masked foreground
    fg_masked = cv.bitwise_and(img, img, mask=mask)

    # get masked background, mask must be inverted
    mask = cv.bitwise_not(mask)
    bk_masked = cv.bitwise_and(bk, bk, mask=mask)

    # combine masked foreground and masked background
    final = cv.bitwise_or(fg_masked, bk_masked)
    return final



'''
calibraton for one camera
'''
def single_calibration(ind_cam, images_per_cam, change_background=False):
    all_img_points.append([])
    all_img_points_success.append([])
    objpoints2.append([])
    for fname in images_per_cam:
        img = cv.imread(fname)
        if change_background:
            img = convert_bk_to_white(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('img', gray)
        cv.waitKey(10)
        if ind_cam==0 and len(gray0)==0:
            gray0.append(gray)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)  # cv2.findChessboardCorners(image, patternSize, flags)
        #print(corners)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            objpoints2[ind_cam].append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)  # cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria)
            print(type(corners2))
            print(corners2)
            all_img_points[ind_cam].append(corners2) #maybe all_img_points[ind_cam].append(corners2)
            all_img_points_success[ind_cam].append(True)
            #print(all_img_points)
            # Draw and display the corners
            img = cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(10)
        else:
            all_img_points_success[ind_cam].append(False)
         #   objpoints2[ind_cam].append(objp)
    print(all_img_points[ind_cam])
    list_calibration = ()
    if all_img_points[ind_cam]!=[]:
        list_calibration= cv.calibrateCamera(objpoints2[ind_cam], all_img_points[ind_cam], gray.shape[::-1], None,
                                                       None)  # (objectPoints, imagePoints, imageSize) #retL, cameraMatrixL, distL, rvecsL, tvecsL
        success[ind_cam]=True
        print(type(list_calibration))
        height, width, channels = img.shape
        list_calibration,roi = cv.getOptimalNewCameraMatrix(list_calibration[1], list_calibration[2], (width, height), 1, (width, height))


    return list_calibration


def stereo_calib(ind_per,imgpoints0, imgpoints_other, objpoints_now):
     flags = 0
     flags |= cv.CALIB_FIX_INTRINSIC
     # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
     # Hence intrinsic parameters are the same
     criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
     # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
     stereo_information.append([])
     stereo_information[ind_per-1]= cv.stereoCalibrate(
         objpoints_now, imgpoints0, imgpoints_other, dict_calib_per_cam[0][1], dict_calib_per_cam[0][2], dict_calib_per_cam[1][1], dict_calib_per_cam[1][1], gray0[0].shape[::-1],
         criteria_stereo, flags)



num_of_cams = 5
################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
chessboardSize = (9,6) #how many: left- columns, right - rows.
frameSize = (640,480)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * 30 #change according to the length of each squre

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space, will do it a list of lists.
objpoints2=[]
all_img_points=[]# 2d points in image plane, a list of lists for all cameras.
all_img_points_success=[] #to each frame : succeded to get corners? (=imgpoints?)
success = [False]*num_of_cams
dict_calib_per_cam = dict(zip(range(num_of_cams),range(num_of_cams))) #dict{ 0:[retL, cameraMatrixL, distL, rvecsL, tvecsL],...} , would be update in the for
#dict_shapes_img = dict(zip(range(num_of_cams),range(num_of_cams))) #dict { 0:[heightL, widthL, channelsL]...}

'''
calibration for each camera, and saving parameters in dict.
'''
gray0=[]
for ind_cam in range(num_of_cams):
    images_per_cam = sorted(glob.glob('geometric_calib_images_2/camera'+str(ind_cam+1)+'/*.png'))
    dict_calib_per_cam[ind_cam]=single_calibration(ind_cam=ind_cam,images_per_cam=images_per_cam, change_background=True)
    print('next!')

cv.destroyAllWindows()



'''
now, stereo!
'''

stereo_information = [] #the len would be 4 for each per, list of list: [ [retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix],[]... ]

for ind_cam in range(1,num_of_cams):
    if success[ind_cam]:
        len_min,new0 , new_other = equaly_length(ind_cam)
        objpoints_now = [objp]*len_min
        stereo_calib(ind_per=ind_cam,imgpoints0=new0, imgpoints_other=new_other, objpoints_now=objpoints_now)

'''
stereo rectification
'''

rectifyScale= 1
for ind_cam in range(1,num_of_cams):
    if success[ind_cam]:
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(stereo_information[ind_cam-1][1], stereo_information[ind_cam-1][2],
                                                                                   stereo_information[ind_cam-1][3], stereo_information[ind_cam-1][4],
                                                                                   gray0[0].shape[::-1], stereo_information[ind_cam-1][5], stereo_information[ind_cam-1][6],
                                                                                   rectifyScale, (0, 0))

        stereoMap0 = cv.initUndistortRectifyMap(stereo_information[ind_cam-1][1], stereo_information[ind_cam-1][2], rectL, projMatrixL, gray0[0].shape[::-1], cv.CV_16SC2)
        stereoMap_other = cv.initUndistortRectifyMap(stereo_information[ind_cam-1][3], stereo_information[ind_cam-1][4], rectR, projMatrixR, gray0[0].shape[::-1], cv.CV_16SC2)

        print("Saving parameters!")
        cv_file = cv.FileStorage('stereoMap_per_'+str(ind_cam)+'.xml', cv.FILE_STORAGE_WRITE)

        cv_file.write('stereoMapL_x', stereoMap0[0])
        cv_file.write('stereoMapL_y', stereoMap0[1])
        cv_file.write('stereoMapR_x', stereoMap_other[0])
        cv_file.write('stereoMapR_y', stereoMap_other[1])

cv_file.release()