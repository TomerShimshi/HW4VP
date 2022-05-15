import numpy as np
import cv2
from scipy.stats import gaussian_kde

import utilis 
import video_stabilize
import background_subtraction
import trashV2

ID1 = 203200480
ID2 = 320521461

#video_stabilize.stabalize_video('Inputs\INPUT.avi','Outputs\stabilized_{}_{}.avi'.format(ID1,ID2))

background_subtraction.background_subtraction('Outputs\stabilized_{}_{}.avi'.format(ID1,ID2))
#trashV2.background_subtraction('Outputs\stabilized_{}_{}.avi'.format(ID1,ID2))