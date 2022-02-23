from __future__ import division
import os
from os.path import join
import numpy as np
import os
import glob
from datetime import datetime
import cv2

import json
# Enthought library imports
import scipy.io as sio
import pandas as pd
from collections import OrderedDict

import matplotlib.pyplot as plt
#from plyfile import PlyData, PlyElement

#from shdom.CloudCT_Utils import *


# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
#  ------------ Functions ------------------------------:
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------

def ParseTXT_FromAgisoft(txtFile):  
    
    """
    Yaw, Pitch, Roll definition:
    * Yaw is the rotation around the Z axis.
    * Pitch is the rotation around the Y axis.
    * Roll is the rotation around the X axis.
 
    """
    GeometricDataEstimated = pd.read_csv(txtFile,skiprows=1, index_col=None)
    # get estimated coordinates after SFM
    view_names = GeometricDataEstimated['#Label'].values.tolist()
    GeometricDataEstimated = GeometricDataEstimated.set_index('#Label')
    # get rig of the 0,1,2... indexing in the first column.
    MULTI_VIEW_PARAMS = OrderedDict()
    
    for index, name in enumerate(view_names):
        
        one_view_params = {
            'est_x' : np.array(GeometricDataEstimated['X_est'][index]),
            'est_y' : np.array(GeometricDataEstimated['Y_est'][index]),
            'est_z' : np.array(GeometricDataEstimated['Z_est'][index]),
            'est_yaw' : np.array(GeometricDataEstimated['Yaw_est'][index]),
            'est_pitch' : np.array(GeometricDataEstimated['Pitch_est'][index]),
            'est_roll' : np.array(GeometricDataEstimated['Roll_est'][index])   
            }
        
        MULTI_VIEW_PARAMS[name] = one_view_params
    
    return MULTI_VIEW_PARAMS

def intrinsic(img, CALIB_DAY, cam_id):
    dir = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/calibration params/geometric_params"
    intrinsic_calib_xml_file = dir+'/{}_intrinsic_{}_adjusted_by_agisoft.xml'.format(CALIB_DAY,cam_id)

    # LOAD DISTORTION COEFFICIENS FROM INTRINSIC CALIBRATION:
    try:
        from xml.etree import ElementTree
        tree = ElementTree.parse(intrinsic_calib_xml_file)
    except IOError:
        print('unable to read ' + intrinsic_calib_xml_file)

    root = tree.getroot()  # Element 'document'

    f = float(root.find('f').text)
    width = int(root.find('width').text)
    height = int(root.find('height').text)
    cx = float(root.find('cx').text) + (0.5 * width - 0.5)
    cy = float(root.find('cy').text) + (0.5 * height - 0.5)
    k1 = float(root.find('k1').text)
    k2 = float(root.find('k2').text)
    k3 = float(root.find('k3').text)
    fy = f
    fx = f

    camera_matrix = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    Distortion_coefficients = np.array([k1, k2, 0, 0, k3])
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, Distortion_coefficients, None, camera_matrix,
                                             (width, height), 5)
    corrected_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return corrected_img

def get_camera_position(CALIB_DAY,cam_id):
    # READ TEXT file from Agisoft :
    dir = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/calibration/calibration params/geometric_params"
    DAYx_txtFile = dir+'/{}_calibration_5_cameras_python_calib_cameras_adjusted_intrinsic_calib.txt'.format(
        CALIB_DAY)

    # -------------------------------------------------------
    # -----------------LOAD RECONSTRUCTED CAMERAS GOEMETRY--:
    # -------------------------------------------------------
    setup_cameras_params = ParseTXT_FromAgisoft(DAYx_txtFile)

    # SORT both dicts since the order is important:
    sorted_list = sorted(setup_cameras_params.items(), key=lambda x: x[0].split('_')[-1].split('.')[0])
    setup_cameras_params = OrderedDict(sorted_list)

    names = list(setup_cameras_params.keys())
    # FIND 3D calibration parameters of only cam_id:
    k = [i for i in names if (cam_id in i)]
    cam_index = names.index(k[0])
    this_camera_params = setup_cameras_params[names[cam_index]]

    x = this_camera_params['est_x']
    y = this_camera_params['est_y']
    z = this_camera_params['est_z']

    yaw = this_camera_params['est_yaw']  # Yaw is the rotation around the Z axis.
    pitch = this_camera_params['est_pitch']  # Pitch is the rotation around the Y axis.
    roll = this_camera_params['est_roll']  # Roll is the rotation around the X axis.

    return x,y,z,yaw,pitch,roll
    # -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
#  ------------ MAIN      ------------------------------:
# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------


def main():
    
    """
    TO LOAD original image and remove distortions:
    
    """
    # example image:
    # 
    image_to_load = r'C:\Users\masadatz\Google Drive\CloudCT\svs_vistek\Data_From_Experiment\B\B11_20000\1\24_14_09_56_212847_101936.png'
    # imitate - method which detects to what calibration day (DAY1, DAY2 or DAY3) match this image.
    # assume it is of DAY1
    CALIB_DAY = 'DAY1'
    
    # get the camera ID:
    cam_id = image_to_load.split('_')[-1].split('.')[0]
    
    # NOA - VERY IMPORTANT THAT THE IMAGES ARE AFTER POLARIMETRIC CORRECTIONS.
    img = cv2.imread(image_to_load)

    img_int_correction = intrinsic(img, CALIB_DAY, cam_id)

    plt.imshow(img)
    plt.show()
    plt.imshow(img_int_correction)
    plt.show()

    x,y,z,yaw,pitch,roll = get_camera_position(CALIB_DAY, cam_id)

    print(x,y,z,yaw,pitch,roll)
    plt.show()
    

if __name__ == '__main__':
    main()
    