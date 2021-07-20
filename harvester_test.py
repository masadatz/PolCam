import time
from harvesters.core import Harvester
from PIL import Image
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