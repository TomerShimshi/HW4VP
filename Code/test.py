import logging
import cv2
import numpy as np
import tqdm

from scipy.stats import gaussian_kde

import constants

import utilis

ID1 = 203200480
ID2 = 320521461

my_logger = logging.getLogger('MyLogger')

def background_subtraction(input_video_path):
    my_logger.info('Starting Background Subtraction')
    cap = cv2.VideoCapture(input_video_path)
    parameters = utilis.get_video_parameters(cap)
    h,w = parameters["height"],parameters['width']
    #load the frames
    frames_bgr =  utilis.load_video(cap,wanted_colors='bgr')
    frames_gray = utilis.color_to_gray(frames_bgr)
    
    n_frames = len(frames_bgr)
    #create the backround subtractor
    num_iter = 8
    fgbg = cv2.createBackgroundSubtractorKNN(history=num_iter*n_frames,detectShadows=False,dist2Threshold =170.0)
    mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
   
    print('started studing frames history')
    pbar = tqdm.tqdm(total=num_iter*n_frames)
    frames_list =[]
    for frame_idx, frame in enumerate(frames_bgr):
        #frame_hsv= frame[:,:,1:]
        upper_mask= frame.copy()
        '''
        we give a diffrent number than zero in order to
        niglact this pixwls while calculating the KDE
        '''
    
        upper_mask[:h//3,:]=0
        upper_mask[2*(h//3):,:]=0
        frames_list.append(upper_mask)
    

    utilis.write_video('Outputs\_test_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=frames_list,isColor=True)