import numpy as np
import cv2
from scipy.stats import gaussian_kde
import time
import utilis 
import video_stabilize
import background_subtraction
import matting
import new_bg_sub
import test
import new_matting
import new_matting_v2
import tracking

import json

dic ={}

ID1 = 203200480
ID2 = 320521461
start_all = time.time()
'''
#video_stabilize.stabalize_video('Inputs\INPUT.avi','Outputs\stabilized_{}_{}.avi'.format(ID1,ID2))
end_stabalize = time.time()

time_to_stable_in_min= np.round((end_stabalize- start_all)/60)
print('time to finshe stabalize took {} minutes'.format(time_to_stable_in_min))

#background_subtraction.background_subtraction('Outputs\stabilized_{}_{}.avi'.format(ID1,ID2))
new_bg_sub.background_subtraction('Outputs\stabilized_{}_{}.avi'.format(ID1,ID2))
end_bg = time.time()

time_to_bg_in_min= np.round((end_bg- start_all)/60)
print('time to finshe bg sub took {} minutes'.format(time_to_bg_in_min))

new_matting_v2.matting('Outputs\stabilized_203200480_320521461.avi','Outputs\_binary_203200480_320521461.avi','Inputs\\background.jpg')
end_mat = time.time()
time_to_mat_in_min= np.round((end_mat- start_all)/60)
print('time to finshe matting took {} minutes'.format(time_to_mat_in_min))
'''
tracking.tracking('Outputs\matt_{}_{}.avi'.format(ID1,ID2),'Outputs\_alpha_{}_{}.avi'.format(ID1,ID2))
end_track = time.time()
time_to_track_in_min= np.round((end_track- start_all)/60)
print('time to finshe tracking took {} minutes'.format(time_to_track_in_min))


### NOW FOR THE LOGGING
#dic.update({'time_to_stabilize':time_to_stable_in_min})
#dic.update({'time_to_binary':time_to_bg_in_min})
#dic.update({'time_to_alpha':time_to_mat_in_min})
#dic.update({'time_to_matted':time_to_mat_in_min})
#dic.update({'time_to_Output':time_to_mat_in_min})
#with open("Outputs\Timing.json", "w") as outfile:
#    json.dump(dic, outfile)

