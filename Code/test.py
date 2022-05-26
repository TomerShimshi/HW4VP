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

def matting (input_video_path, BW_mask_path,bg_path):
    my_logger.info('Starting Background Subtraction')
    cap = cv2.VideoCapture(input_video_path)
    cap_mask = cv2.VideoCapture(BW_mask_path)
    parameters = utilis.get_video_parameters(cap)
    h,w = parameters["height"],parameters['width']
    #load the frames
    frames_bgr =  utilis.load_video(cap,wanted_colors='bgr')
    frames_yuv = utilis.load_video(cap,wanted_colors='hsv')
    frames_mask = utilis.load_video(cap_mask,wanted_colors='gray')
    new_bg = cv2.imread(bg_path)
    new_bg = cv2.resize(new_bg,(w,h))
    n_frames = len(frames_bgr)
    matted_frames_list= []
    alpha_frame_list = []
   
    print('started studing frames history')
    pbar = tqdm.tqdm(total=n_frames)
    frames_list =[]

    for frame_idx, frame in enumerate(frames_bgr):
        luma_frame,_,_ = cv2.split(frames_yuv[frame_idx])
        mask = frames_mask[frame_idx]
        mask = (mask>150).astype(np.uint8)
        mask= cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel=np.ones((7,7)),iterations=1)
        mask_idx = np.where(mask==1)
        y_mean,x_mean = (np.mean(mask_idx[0]).astype(int),np.mean(mask_idx[1]).astype(int))

        small_mask = mask[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        temp = np.zeros(frame.shape).astype(np.uint8)
        temp2 = np.max(temp)
        shuff = frame [max(0, y_mean - constants.SMALL_WINDOW_H  // 2):min(h, y_mean + constants.SMALL_WINDOW_H // 2),
            max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2), : ]

        temp[max(0, y_mean - constants.SMALL_WINDOW_H  // 2):min(h, y_mean + constants.SMALL_WINDOW_H // 2),
            max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2), : ] = shuff
        temp2 = np.max(temp)
        frames_list.append(temp.astype(np.uint8))
        #$$$%%% HERE IS THE TEST FOR THE MASK
        temp3 = np.zeros(mask.shape).astype(np.uint8)
        temp2 = np.max(temp)
        shuff2 = mask [max(0, y_mean - constants.SMALL_WINDOW_H  // 2):min(h, y_mean + constants.SMALL_WINDOW_H // 2),
            max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2) ]

        temp3[max(0, y_mean - constants.SMALL_WINDOW_H  // 2):min(h, y_mean + constants.SMALL_WINDOW_H // 2),
            max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2) ] = shuff2
        
        
        matted_frames_list.append((temp3*255).astype(np.uint8))
        pbar.update(1)
    utilis.write_video('Outputs\_test_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=matted_frames_list,isColor=False)

    '''
    for frame_idx, frame in enumerate(frames_bgr):
        #frame_hsv= frame[:,:,1:]
        upper_mask= frame.copy()
        
        we give a diffrent number than zero in order to
        niglact this pixwls while calculating the KDE
        
    
        upper_mask[:h//3,:]=0
        upper_mask[2*(h//3):,:]=0
        frames_list.append(upper_mask)
    

    utilis.write_video('Outputs\_test_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=frames_list,isColor=True)
    '''