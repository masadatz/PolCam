
import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,6) #7 points in a coloumn, 6 in a row
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria, for after use...

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = [] # 3d point in real world space

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []# 2d points in image plane.

# Defining the world coordinates for 3D points
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
# Extracting path of individual image stored in a given directory
print ('objp')
print (objp)
images = glob.glob('*.jpg') #not all!!!

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None) #cv2.findChessboardCorners(image, patternSize, flags)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) #cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria)

        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

h,w = img.shape[:2]
print ('objpoints')
print (objpoints)

print ('imgpoints')
print (imgpoints)
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
#returns: canera matrix (K), distortion coefficient (k1 k2 p1 p2 k3), R, t
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None) # (objectPoints, imagePoints, imageSize)

print("Camera matrix : \n") #k - intrinsic matrix
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")#Rs (array of 3 elements) from each image in a big array
print(rvecs)
print("tvecs : \n") #ts (array of 3 elements) from each image in a big array
print(tvecs)



img = cv2.imread('left12.jpg')
h,  w = img.shape[:2]
print (h,w)
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
print ('roi',roi)
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print ("total error: ", mean_error/len(objpoints))