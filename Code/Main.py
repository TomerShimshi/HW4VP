import numpy as np
import cv2
from scipy.stats import gaussian_kde

import utilis 
import video_stabilize
import background_subtraction
import matting

ID1 = 203200480
ID2 = 320521461

#video_stabilize.stabalize_video('Inputs\INPUT.avi','Outputs\stabilized_{}_{}.avi'.format(ID1,ID2))

background_subtraction.background_subtraction('Outputs\stabilized_{}_{}.avi'.format(ID1,ID2))
#matting.matting('Outputs\stabilized_{}_{}.avi'.format(ID1,ID2),'Outputs\_binary_203200480_320521461.avi','Inputs\\background.jpg')
