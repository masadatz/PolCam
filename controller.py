import copy
from datetime import datetime
from PIL import Image

# Based on - https://github.com/genicam/harvesters
from harvesters.core import Harvester


class Imager:
    def __init__(self):
        self.h = Harvester()
        self.h.add_file(r'C:\\Program Files\\SVS-VISTEK GmbH\\SVCam Kit\\TLUsb\\bin\\sv_u3v_tl_x64.cti')
        self.h.add_file(r'C:\Program Files\SVS-VISTEK GmbH\SVCam Kit\TLGigE\bin\\sv_gev_tl_x64.cti')
        self.h.update()
        num_devices = len(self.h.device_info_list)
        print(f"Found {num_devices} devices")  # see all cameras
        if num_devices == 0:
            print("No devices found. If the camera is connected - call imager.clear_all() or restart the kernel and try again.")
        self.serial_ids = [info.serial_number for info in self.h.device_info_list]
        print(f"Devices IDs - {self.serial_ids}")  # see all cameras
        self.cams = [self.h.create_image_acquirer(serial_number=_id) for _id in self.serial_ids]

    def get_camera_params(self):
        """
        List of possible parameters to adjust"""
        print(dir(self.cams[0].remote_device.node_map))  # see all parameters

    def update_params(self, ExposureTime, PixelFormat, AcquisitionFrameRate):
        for ia in self.cams:
            ia.remote_device.node_map.ExposureTime.value = ExposureTime  # for example, change exposure time
            ia.remote_device.node_map.PixelFormat.value = PixelFormat  # .symbolics instead of .value for options
            ia.remote_device.node_map.AcquisitionFrameRate.value = AcquisitionFrameRate

    def get_temperature(self):
        for ia in self.cams:
            print(ia.remote_device.node_map.DeviceTemperature.value)

    def _start_acquisitions(self):
        for ia in self.cams:
            ia.start_acquisition(run_in_background=True)  # Start capturing images

    def _stop_acquisitions(self):
        for ia in self.cams:
            ia.stop_acquisition()  # stop capturing images

    def show_images(self):
        for img, img_time, cam_id in self.images:
            img.show()
            img.save(f'{cam_id}_{img_time.strftime("%Y_%m_%d_%H%M%S")}.jpeg')

    def get_images(self, show_images):
        images_with_times = []
        for cam_id, ia in zip(self.serial_ids, self.cams):
            # acquire and save an image
            cur_time = datetime.now()

            with ia.fetch_buffer() as buffer:
                component = buffer.payload.components[0]
                print(f"cam {cam_id} captured image of format {component.data_format} at {cur_time}")
                if component.data_format == 'Mono12Packed':
                    data = component.data >> 4
                else:
                    data = component.data
                _2d = component.data.reshape(component.height, component.width)

                img = Image.fromarray(copy.deepcopy(_2d))
                images_with_times.append((img, cur_time, cam_id))

        self.images = images_with_times
        if show_images:
            self.show_images()

    def clear_all(self):
        for ia in self.cams:
            ia.destroy()
        self.h.reset()
