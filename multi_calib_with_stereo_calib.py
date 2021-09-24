
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
calibraton for one camera
'''
def single_calibration(ind_cam, images_per_cam):
    all_img_points.append([])
    for fname in images_per_cam:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if ind_cam==0 and len(gray0)==0:
            gray0.append(gray)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)  # cv2.findChessboardCorners(image, patternSize, flags)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)  # cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria)

            all_img_points[ind_cam]=corners2
            # Draw and display the corners
            img = cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(1000)

    list_calibration= cv.calibrateCamera(objpoints, all_img_points[ind_cam], gray.shape[::-1], None,
                                                       None)  # (objectPoints, imagePoints, imageSize) #retL, cameraMatrixL, distL, rvecsL, tvecsL
    height, width, channels = img.shape
    list_calibration[1],roi = cv.getOptimalNewCameraMatrix(list_calibration[1], list_calibration[2], (width, height), 1, (width, height))


    return list_calibration


def stereo_calib(ind_per,imgpoints0, imgpoints_other):
     flags = 0
     flags |= cv.CALIB_FIX_INTRINSIC
     # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
     # Hence intrinsic parameters are the same
     criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
     # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
     stereo_information.append([])
     stereo_information[ind_per-1]= cv.stereoCalibrate(
         objpoints, imgpoints0, imgpoints_other, dict_calib_per_cam[0][1], dict_calib_per_cam[0][2], dict_calib_per_cam[1][1], dict_calib_per_cam[1][1], gray0[0].shape[::-1],
         criteria_stereo, flags)



num_of_cams = 5
################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
chessboardSize = (9,6)
frameSize = (640,480)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = objp * 30 #change according to the length of each squre

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
all_img_points=[]# 2d points in image plane, a list of list for all cameras.

dict_calib_per_cam = dict(zip(range(num_of_cams),range(num_of_cams))) #dict{ 0:[retL, cameraMatrixL, distL, rvecsL, tvecsL],...} , would be update in the for
#dict_shapes_img = dict(zip(range(num_of_cams),range(num_of_cams))) #dict { 0:[heightL, widthL, channelsL]...}

'''
calibration for each camera, and saving parameters in dict.
'''
gray0=[]
for ind_cam in num_of_cams:
    images_per_cam = sorted(glob.glob('images_all_cameras/camera'+str(ind_cam)+'/*.png'))
    dict_calib_per_cam[ind_cam]=single_calibration(ind_cam=ind_cam,images_per_cam=images_per_cam)


cv.destroyAllWindows()


'''
now, stereo!
'''

stereo_information = [] #the len would be 4 for each per, list of list: [ [retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix],[]... ]

for ind_cam in range(1,num_of_cams):
    stereo_calib(ind_per=ind_cam,imgpoints0=all_img_points[0], imgpoints_other=all_img_points[ind_cam])

'''
stereo rectification
'''

rectifyScale= 1
for ind_cam in range(1,num_of_cams):
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