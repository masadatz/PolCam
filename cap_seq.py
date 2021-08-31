

from controller import Imager
import pandas as pd
import matplotlib.pyplot as plt

def diff_time_pic(num_cam, meta_data): #measuring the maximus difference in time between picturing the photos in the same frame from diffrent cameras.
    max=0
    min=24*60*60*60
    for i in range (num_cam):
        tup_time=meta_data[i][1]
        in_sec=tup_time.hour*60*60+tup_time.minute*60+tup_time.second+tup_time.microsecond*pow(10,-6) #3-hours, 4-min, 5-sec, 6-milionit
        max=[in_sec if in_sec>max else max][0]
        min=[in_sec if in_sec<min else min][0]
    return max-min


imager = Imager()

PIXELFORMATS = {'8':'Mono8','12':'Mono12Packed'}
PIXELFORMAT = PIXELFORMATS['8']
imager.update_params(ExposureTime=20000, PixelFormat=PIXELFORMAT, AcquisitionFrameRate = 50)
imager._start_acquisitions()

raw_images, meta_data = imager.capture_sequence(num_frames=100, sleep_seconds=0.2)

imager._stop_acquisitions()
imager.clear_all()

#print(meta_data)

num_frames=100
diff=[]

#print(meta_data[0:2], type(meta_data))
num_of_times = [num_frames/imager.num_devices if num_frames%imager.num_devices==0 else int(num_frames/imager.num_devices)+1][0]
print(num_of_times)
for i in range (num_frames):

    diffi=diff_time_pic(imager.num_devices, meta_data[i*imager.num_devices:i*imager.num_devices+imager.num_devices])
    print('index',i, 'diffi',diffi)
    diff.append(diffi)

print('diff')
print(diff)
print(len(diff))

print('max in diff:' ,max(diff), 'min in diff', min(diff))

df = pd.DataFrame(diff, columns=['diffrences'])
df['diffrences'].plot(kind='hist', bins=20)
plt.title('max diff = '+ str(round(max(diff),4))) #
plt.show()
#hist= df.hist(bins=10)