import numpy as np
import cv2
from scipy.stats import gaussian_kde

import utilis 
import video_stabilize

ID1 = 203200480
ID2 = 320521461

video_stabilize.stabalize_video('Inputs\INPUT.avi','Outputs\{}_{}_stabilize.avi'.format(ID1,ID2))