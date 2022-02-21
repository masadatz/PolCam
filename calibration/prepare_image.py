import cv2
import polanalyser as pa
from PIL import Image
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import copy
import RadCal
import PolCal
import GeoCal


def find_geocal_day(dir):
    DAY1 = ['A11']
    DAY2 = ['A12','A2','A3','B1','B2','B3','B4','B5','B6','C1','C2','C3']
    DAY3 = ['D1']
    if any(x in dir for x in DAY1):
        CALIB_DAY = 'DAY1'
    elif any(x in dir for x in DAY2):
        CALIB_DAY = 'DAY2'
    elif any(x in dir for x in DAY3):
        CALIB_DAY = 'DAY3'
    else:
        print('CALIB_DAY not found')
    return CALIB_DAY

def main():

    image_dir = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/Data_From_Experiment/B/B15_40000/9/24_15_27_38_496342_192900073.png"
    cam_id = image_dir.split('_')[-1].split('.')[0]
    exposure = int(image_dir.split('/')[-3].split('_')[1])
    bit = [12 if 'A' in image_dir else 8]
    CALIB_DAY = find_geocal_day(image_dir)
    #load original image
    #image = np.load(image_dir)
    img = cv2.imread(image_dir)
    image = np.squeeze(np.array(img[..., 0]))
    #flat field and dark current corrections
    ff_image = RadCal.ff_correct(image,cam_id,exposure)

    #correct demosaiced normalized Stokes vector
    [h, v] = ff_image.shape
    X_mat = PolCal.load_xmat(cam_id)
    [n,hx,vx] = X_mat.shape
    clip = int((h-hx)/2)
    ff_image_clip =ff_image[clip:h-clip,clip:v-clip]
    images_demosaiced = pa.demosaicing(ff_image_clip)
    img_0, img_45, img_90, img_135 = cv2.split(images_demosaiced)

    Stokes= pa.calcLinearStokes(np.moveaxis(np.array([img_0, img_45, img_90, img_135]), 0, -1),
                                 np.deg2rad([0, 45, 90, 135]))
    Stokes_norm,Intensity  = PolCal.norm_stokes(Stokes)

    Stokes_cal = PolCal.Cal(Stokes_norm, X_mat)

    #convert to radiation units
    #TODO check RadCal
    Abs_rad = RadCal.gray2rad(Stokes_cal, cam_id, exposure, bit)
    Stokes_rad =  np.repeat(Intensity[:, :, np.newaxis], 3, axis=2)*Abs_rad

    plt.imshow(Stokes_rad[...,0])
    plt.colorbar()
    plt.show()

    #correct distortion and
    Stokes_dist_correction = GeoCal.intrinsic(Stokes_rad, CALIB_DAY, cam_id)
    plt.imshow(Stokes_dist_correction[...,0])
    plt.colorbar()
    plt.show()

    x, y, z, yaw, pitch, roll = GeoCal.get_camera_position(CALIB_DAY, cam_id)

    #add mask TODO





if __name__ == '__main__':
    main()