
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