#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Based on - https://github.com/genicam/harvesters

from harvesters.core import Harvester
from PIL import Image
from datetime import datetime 
import time

with Harvester() as h:
    h.add_file(r'C:\\Program Files\\SVS-VISTEK GmbH\\SVCam Kit\\TLUsb\\bin\\sv_u3v_tl_x64.cti')
    h.update() 
    with h.create_image_acquirer()  as ia:
        ia.start_acquisition()
        with ia.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
        _2d = component.data.reshape(component.height, component.width)

        if _2d.mean() != 0:
            img = Image.fromarray(_2d)
            img.show()
            img.save(f'my_image_{cur_time.strftime("%Y_%m_%d_%H%M%S")}.png')
        else:
            print("Black image :(")
        ia.stop_acquisition()

    
    
    
class Imager:
    def __init__(self):
        self.h = Harvester()
        self.h.add_file(r'C:\\Program Files\\SVS-VISTEK GmbH\\SVCam Kit\\TLUsb\\bin\\sv_u3v_tl_x64.cti') 
        self.h.add_file(r'C:\Program Files\SVS-VISTEK GmbH\SVCam Kit\TLGigE\bin\\sv_gev_tl_x64.cti') 
        self.h.update()
        num_devices = len(self.h.device_info_list)
        print(f"Found {num_devices} devices") # see all cameras
        if num_devices == 0:
            print("No devices found. If the camera is connected - call imager.clear_all() or restart the kernel and try again.")
        self.serial_ids = [info.serial_number for info in self.h.device_info_list]
        print(f"Devices IDs - {self.serial_ids}") # see all cameras
        self.cams = [self.h.create_image_acquirer(serial_number=_id) for _id in self.serial_ids]
        
    def get_camera_params(self):
        """
        List of possible parameters to adjust"""
        print(dir(self.cams[0].remote_device.node_map)) # see all parameters

    def update_params(self, ExposureTime, PixelFormat):
        for ia in self.cams:
            ia.remote_device.node_map.ExposureTime.value = ExposureTime # for example, change exposure time
            ia.remote_device.node_map.PixelFormat.value = PixelFormat # .symbolics instead of .value for options
    
    def get_temperature(self):
        for ia in self.cams:
            print(ia.remote_device.node_map.DeviceTemperature.value)
            
    def _start_acquisitions(self):
        for ia in self.cams:
            ia.start_acquisition() # Start capturing images

    def _stop_acquisitions(self):
        for ia in self.cams:
            ia.stop_acquisition() # stop capturing images
            
    def process_components_to_images(self, components):
        images = []
        for component in components:

            _2d = data.reshape(component.height, component.width)
            print(_2d.min(), _2d.max())
            images.append(Image.fromarray(_2d))
        return images

            
    def get_images(self, show_images):
        
        for ia in self.cams:
            # acquire and save an image
            try:
                device_id = ia.remote_device.node_map.DeviceID.value
            except AttributeError:
                device_id = ia.remote_device.node_map.DeviceUserID.value
            cur_time = datetime.now()
            
            with ia.fetch_buffer() as buffer:
                component = buffer.payload.components[0]
                print(f"cam {device_id} captured image of format {component.data_format} at {cur_time}")
#             if component.data_format == 'Mono12Packed':
#                 data = component.data >> 4
#             else:
#                 data = component.data
                print(ia.remote_device.node_map.ExposureTime.value)
                _2d = component.data.reshape(component.height, component.width)
            if _2d.mean() == 0:
                time.sleep(1)
                cur_time = datetime.now()
                with ia.fetch_buffer(timeout=1) as buffer:
                    component = buffer.payload.components[0]
                    print(f"cam {device_id} captured image of format {component.data_format} at {cur_time}")
                    _2d = component.data.reshape(component.height, component.width)

                print(_2d.mean())

            img = Image.fromarray(_2d)
            if _2d.mean() != 0:
                img.show()
                img.save(f'my_image_{cur_time.strftime("%Y_%m_%d_%H%M%S")}.jpeg')
            else:
                print("Black image :(")
    def clear_all(self):
        for ia in self.cams:
            ia.destroy()
        self.h.reset()
    
                

    
            
    # TODO 

imager = Imager()


# In[2]:


imager._start_acquisitions()
time.sleep(2)

# In[3]:


#imager.update_params(ExposureTime=20000, PixelFormat='Mono8')
imager.get_images(show_images=True)


# In[12]:


imager._stop_acquisitions()
imager.clear_all()


# In[20]:

"""

ia = imager.cams[0]

cur_time = datetime.now()
with ia.fetch_buffer() as buffer:
    component = buffer.payload.components[0]

_2d = component.data.reshape(component.height, component.width)
if _2d.mean() != 0:
    img.show()
    img.save(f'my_image_{cur_time.strftime("%Y_%m_%d_%H%M%S")}.jpeg')
else:
    print("Black image :(")

"""
# In[ ]:





# In[ ]:




