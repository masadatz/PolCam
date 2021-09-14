import copy
from datetime import datetime
from PIL import Image
import time
import numpy as np
# Based on - https://github.com/genicam/harvesters
from harvesters.core import Harvester
from params import GIGE_CTI, USB3_CTI
import os
import pathlib
import cv2
import polanalyser as pa
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
from harvesters.util.pfnc import mono_location_formats, \
    rgb_formats, bgr_formats, \
    rgba_formats, bgra_formats



class Imager:
    def __init__(self):
        self.h = Harvester()
        self.h.add_file(USB3_CTI)
        self.h.add_file(GIGE_CTI)
        self.h.update()
        self.num_devices = len(self.h.device_info_list)
        print(f"Found {self.num_devices} devices")  # see all cameras
        if self.num_devices == 0:
            print(
                "No devices found. If the camera is connected - call imager.clear_all() or restart the kernel and try again.")
        self.serial_ids = [info.serial_number for info in self.h.device_info_list]
        print(f"Devices IDs - {self.serial_ids}")  # see all cameras
        self.cams = [self.h.create_image_acquirer(serial_number=_id) for _id in self.serial_ids]

    def get_camera_params(self):
        """
        List of possible parameters to adjust"""
        print(dir(self.cams[0].remote_device.node_map))  # see all parameters

    def update_params(self, ExposureTime=None, PixelFormat=None, AcquisitionFrameRate=None):
        for ia in self.cams:
            if ExposureTime is not None:
                ia.remote_device.node_map.ExposureTime.value = ExposureTime  # for example, change exposure time
            if PixelFormat is not None:
                ia.remote_device.node_map.PixelFormat.value = PixelFormat  # .symbolics instead of .value for options
            if AcquisitionFrameRate is not None:
                ia.remote_device.node_map.AcquisitionFrameRate.value = AcquisitionFrameRate

    def get_temperature(self):
        for ia in self.cams:
            print(ia.remote_device.node_map.DeviceTemperature.value)

    def _start_acquisitions(self):
        for ia in self.cams:
            ia.start_acquisition(run_in_background=True)  # Start capturing images # TODO is this flag necessary?

    def _stop_acquisitions(self):
        for ia in self.cams:
            ia.stop_acquisition()  # stop capturing images

    def show_images(self):
        for img, _, _ in self.images:
            img.show()

    def save_images(self):
        for img, img_time, cam_id in self.images:
            img.save(f'{img_time.strftime("%Y_%m_%d_%H%M%S")}_{cam_id}.jpeg')

    def get_images(self, show_images, save_images, run_indx=0):
        images_with_times = []
        raw_images = []
        metadata = []
        for cam_id, ia in zip(self.serial_ids, self.cams):
            cur_time = datetime.now()

            # acquire an image
            with ia.fetch_buffer() as buffer:
                component = buffer.payload.components[0]
                # print(f"{run_indx} - {cur_time} - {cam_id} captured {component.data_format} image ")
                if component.data_format == 'Mono12Packed':
                    data = component.data >> 4
                else:
                    data = component.data
                _2d = data.reshape(component.height, component.width)
                raw_image = copy.deepcopy(_2d)
                # img = Image.fromarray(raw_image)
                # images_with_times.append((img, cur_time, cam_id))
                raw_images.append(raw_image)
                metadata.append((run_indx, cur_time, cam_id))
        self.images = images_with_times
        if show_images:
            self.show_images()
        if save_images:
            self.save_images()

        return raw_images, metadata

    def capture_sequence(self, num_frames, sleep_seconds):
        all_raw_images = []
        all_meta_data = []
        MARGIN = 0.0001  # 10us
        frame_num = 1
        start_time = time.time()
        next_pic = start_time + sleep_seconds
        # arr = np.empty((num_frames, self.num_devices, 2048, 2448), dtype='uint8')
        # time.sleep(0.5)
            # arr[frame_num] = np.array(raw_images)
        while frame_num<=num_frames:
            tm = time.time()
            if (tm + MARGIN) > next_pic:
                frame_num += 1  # integer counter
                next_pic = start_time + frame_num * sleep_seconds
                raw_images, metadata = self.get_images(show_images=False, save_images=False, run_indx=frame_num-1)
                all_raw_images.append(raw_images)
                all_meta_data.extend(metadata)

            maxi=max((next_pic - tm) * 0.95, MARGIN)
            # if maxi==MARGIN: print('margin!')
            time.sleep(maxi)  # sleep for 90% of the remaining time

        return all_raw_images, all_meta_data

    """
    MARGIN = 0.001 #10us
DELAY = 0.2
num = 1
start_time = time.time()
next_pic = start_time + DELAY
while True:
    tm = time.time()
    if (tm + MARGIN) > next_pic:
        num += 1 # integer counter
        next_pic = start_time + num * DELAY
        print('take picture at {:.6f}'.format(tm))
    time.sleep((next_pic - tm) * 0.9) # sleep for 90% of the remaining time

    """


    def clear_all(self):
        for ia in self.cams:
            ia.destroy()
        self.h.reset()

    def show_splitted_images(self, img_0, img_45, img_90, img_135):
        Image.fromarray(img_0).show()
        Image.fromarray(img_45).show()
        Image.fromarray(img_90).show()
        Image.fromarray(img_135).show()

    def save_selected_fig(self, fig, i, title, block):
        plt.figure(i)
        im = plt.imshow(fig)
        plt.colorbar(im)
        datacursor()
        plt.title(title)
        plt.axis('off')
        plt.savefig(title+'.png')
      #  Image.fromarray(fig).show()
        plt.show(block=block) #double??? what does it do????



    def splitter(self):

        # Based on https://github.com/elerac/polanalyser
        init_image = self.capture_sequence(num_frames=1, sleep_seconds=0.2)
        init_image = np.squeeze(np.array(init_image[0]))
        img_demosaiced = pa.demosaicing(init_image)
        img_0, img_45, img_90, img_135 = cv2.split(img_demosaiced)

        #self.show_splitted_images(img_0, img_45, img_90, img_135)

        Stokes = pa.calcLinearStokes([img_0, img_45, img_90, img_135], [0, 45, 90, 135])
        I = Stokes[:, :, 0]
        print(I)
        DoLP = pa.cvtStokesToDoLP(Stokes)
        AoLP = (180 / np.pi) * pa.cvtStokesToAoLP(Stokes) - 90

        arr = [I / np.max(I),DoLP,AoLP]
        titles = ['I','DoLP','AoLP']
        blocks= [True,False,True]


        for i in range(1,4):
            self.save_selected_fig(arr[i-1], i, titles[i-1], blocks[i-1])


    def video_in_action(self): #video of AolP,DOLP, Intensity. (to change photos?)
        while True:
            for ia in self.cams:
                with ia.fetch_buffer() as buffer:
                    # Work with the Buffer object. It consists of everything you need.
                    _1d = buffer.payload.components[0].data
                    # The buffer will automatically be queued.

                    payload = buffer.payload
                    component = payload.components[0]
                    width = component.width
                    height = component.height
                    data_format = component.data_format

                    # Reshape the image so that it can be drawn on the VisPy canvas:
                    if data_format in mono_location_formats:
                        content = component.data.reshape(height, width)  # the 2d photo, and if I want 3d?
                    else:
                        print('not in mono location format. check docs.')

                    # Display the resulting frame
                    img_demosaiced = pa.demosaicing(content)
                    # Calculate the Stokes vector per-pixel
                    angles = np.deg2rad([0, 45, 90, 135])
                    img_stokes = pa.calcStokes(img_demosaiced, angles)
                    # Convert the Stokes vector to Intensity, DoLP and AoLP
                    img_intensity = pa.cvtStokesToIntensity(img_stokes)
                    img_DoLP = pa.cvtStokesToDoLP(img_stokes)
                    img_AoLP = pa.cvtStokesToAoLP(img_stokes)*(180 / np.pi)-90
                    img_AoLP_cmapped = pa.applyColorToAoLP(img_AoLP)
                    cv2.imshow('intensity', img_intensity)
                    cv2.imshow('DolP', img_DoLP)
                    cv2.imshow('img_AoLP_cmapped', img_AoLP_cmapped)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()


    def full_action(self): #all the neceacerry orders for full action of the video
        self._start_acquisitions()
        self.video_in_action()
        self._stop_acquisitions()
        self.clear_all()

    #create folders to the images of the geometric calibraion.
    def create_folders_geo_calib(self):
        path = pathlib.Path().resolve()
        dir = 'geometric_calib_images'
        path_dir = os.path.join(path,dir)

        try:
            os.makedirs(path_dir)
            print('created successfully')
        except:
            print('already exists')

        for i in range(self.num_devices):
            mid_dir = 'camera' + str(i)
            path_mid_dir = os.path.join(path_dir, mid_dir)
            try:
                os.makedirs(path_mid_dir)
                print('created successfully')
            except:
                print('already exists')

    #accquiring_and_saving_photos to the geometric calibration
    def accquiring_and_saving_photos(self, num_frames,time_sleep):
        self.create_folders_geo_calib()
        cur_path = pathlib.Path().resolve()
        path = os.path.join(cur_path,'geometric_calib_images')
        #print(path)
        all_raw_imgs, meta_data =self.capture_sequence(num_frames,time_sleep)
        for i in range(self.num_devices):
            for j in range(num_frames):
                dir = '\camera' + str(i)
                full = str(path)+dir+'\Frame'+str(j)+'.png'
                cv2.imwrite(full, all_raw_imgs[j][i])
                print('saving...')

    #full action to get photos to geo
    def full_acuire_to_geo_calib(self):
        self._start_acquisitions()
        self.accquiring_and_saving_photos(num_frames=3,time_sleep=6)
        self._stop_acquisitions()
        self.clear_all()


if __name__ == '__main__':


    imager = Imager()
    imager._start_acquisitions()
    # raw,meta=imager.capture_sequence(3,0.2)
    # cv2.imshow('img',raw[0][0])
    # cv2.waitKey(0)
    imager.accquiring_and_saving_photos(num_frames=3,time_sleep=0.2)
    imager._stop_acquisitions()
    imager.clear_all()
    #imager.full_acuire_to_geo_calib()





