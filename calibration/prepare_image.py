import cv2
import polanalyser as pa
from PIL import Image
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import copy
import os.path
import RadCal
import PolCal
import GeoCal
from PIL import Image
import sys
sys.path.append('../')

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



def prepare_image(image_dir1):
    dir = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/Data_From_Experiment"
    image_dir = dir + image_dir1
    cam_id = image_dir.split('_')[-1].split('.')[0]
    exposure = int(image_dir.split('/')[-3].split('_')[1])
    bit = [12 if 'A' in image_dir else 8]
    CALIB_DAY = find_geocal_day(image_dir)

    # load original image

    extension = os.path.splitext(image_dir)[1]
    if extension == '.png':
        img = cv2.imread(image_dir)
        image = np.squeeze(np.array(img[..., 0])).astype(float)
    elif extension == '.npy':
        image = np.load(image_dir)

    if cam_id == '192900073':
        image_type = 'COLOR_PolarRGB'

        # flat field and dark current corrections
        ff_image = RadCal.ff_correct(image, cam_id, exposure)

        # correct demosaiced normalized Stokes vectors
        [h, v] = ff_image.shape
        X_mat = PolCal.load_xmat(cam_id)
        [n, hx, vx] = X_mat.shape
        clip = int((h - hx) / 2)
        ff_image_clip = ff_image[clip:h - clip, clip:v - clip]

        images_demosaiced = pa.demosaicing((ff_image_clip.astype(np.uint8)), code=image_type)
        Stokes_final = np.zeros([h, v, 3, 3])
        DoLP = np.zeros([h, v, 3])
        for i in range(3):
            img_0, img_45, img_90, img_135 = cv2.split(images_demosaiced[:, :, i, :])

            Stokes = pa.calcLinearStokes(np.moveaxis(np.array([img_0, img_45, img_90, img_135]), 0, -1),
                                         np.deg2rad([0, 45, 90, 135]))
            Stokes_norm, Intensity = PolCal.norm_stokes(Stokes)

            Stokes_cal = PolCal.Cal(Stokes_norm, X_mat)

            # convert to radiation units
            # TODO check RadCal
            Abs_rad = RadCal.gray2rad(Stokes_cal, cam_id, exposure, bit)
            Stokes_rad = np.repeat(Intensity[:, :, np.newaxis], 3, axis=2) * Abs_rad

            # pad back to full image size
            npad = ((clip, clip), (clip, clip), (0, 0))
            Stokes_rad_full = np.pad(Stokes_rad, pad_width=npad, mode='constant', constant_values=0)

            # correct distortion get camera position
            Stokes_dist_correction = GeoCal.intrinsic(Stokes_rad_full, CALIB_DAY, cam_id)

            # Add mask
            mask_dir = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/Data_From_Experiment/masks/{CALIB_DAY}/{cam_id}.png"
            img = cv2.imread(mask_dir)
            mask = np.squeeze(np.array(img[..., 0])).astype(float)
            Stokes_final[:, :, i, :] = Stokes_dist_correction * np.repeat(mask[:, :, np.newaxis], 3, axis=2)


    else:
        image_type = 'COLOR_PolarMono'
        # load original image

        extension = os.path.splitext(image_dir)[1]
        if extension == '.png':
            img = cv2.imread(image_dir)
            image = np.squeeze(np.array(img[..., 0])).astype(float)
        elif extension == '.npy':
            image = np.load(image_dir)

        # flat field and dark current corrections
        ff_image = RadCal.ff_correct(image, cam_id, exposure)

        # correct demosaiced normalized Stokes vector
        [h, v] = ff_image.shape
        X_mat = PolCal.load_xmat(cam_id)
        [n, hx, vx] = X_mat.shape
        clip = int((h - hx) / 2)
        ff_image_clip = ff_image[clip:h - clip, clip:v - clip]

        images_demosaiced = pa.demosaicing((ff_image_clip), code=image_type)
        img_0, img_45, img_90, img_135 = cv2.split(images_demosaiced)

        Stokes = pa.calcLinearStokes(np.moveaxis(np.array([img_0, img_45, img_90, img_135]), 0, -1),
                                     np.deg2rad([0, 45, 90, 135]))
        Stokes_norm, Intensity = PolCal.norm_stokes(Stokes)

        Stokes_cal = PolCal.Cal(Stokes_norm, X_mat)

        # convert to radiation units
        # TODO check RadCal
        Abs_rad = RadCal.gray2rad(Stokes_cal, cam_id, exposure, bit)
        Stokes_rad = np.repeat(Intensity[:, :, np.newaxis], 3, axis=2) * Abs_rad

        # pad back to full image size
        npad = ((clip, clip), (clip, clip), (0, 0))
        Stokes_rad_full = np.pad(Stokes_rad, pad_width=npad, mode='constant', constant_values=0)

        # correct distortion get camera position
        Stokes_dist_correction = GeoCal.intrinsic(Stokes_rad_full, CALIB_DAY, cam_id)

        # add mask
        mask_dir = f"C:/Users/masadatz/Google Drive/CloudCT/svs_vistek/Data_From_Experiment/masks/{CALIB_DAY}/{cam_id}.png"
        img = cv2.imread(mask_dir)
        mask = np.squeeze(np.array(img[..., 0])).astype(float)
        Stokes_final =  Stokes_dist_correction*np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    return Stokes_final


def main():

    image_dir =  f"/B/B42_25000/10/26_10_45_50_764219_192900073.png"
    Stokes_final = prepare_image(image_dir)
    DoLP = pa.cvtStokesToDoLP(Stokes_final)
    DoLP[DoLP > 1] = 0
    plt.imshow(DoLP)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()