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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
       
        mask_idx = np.where(mask==1)
        y_mean,x_mean = (np.mean(mask_idx[0]).astype(int),np.mean(mask_idx[1]).astype(int))
        small_luma_frame = luma_frame[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        small_bgr_frame = frame[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        small_mask = mask[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        if frame_idx%30==0:
            t=1

        new_mask = mask.copy()
        new_mask[max(0, y_mean - 2*(constants.SMALL_WINDOW_H//3)  ):min(h, y_mean +constants.SMALL_WINDOW_H//3 )
        ,max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2) ] =1
        bg= 1-new_mask
        #small_fg_mask = cv2.dilate(small_mask,kernel=kernel,iterations=constants.DIAL_N_ITER)#cv2.morphologyEx(small_mask,cv2.cv2.MORPH_OPEN, kernel=kernel,iterations=constants.ERODE_N_ITER)
        small_fg_mask = bg[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2)
        ,max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]#1-temp_mask
        temp_mask =small_fg_mask#np.zeros(mask.shape)
        #temp_mask[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)] = small_fg_mask
        idx = np.where(temp_mask ==1)
        temp3 = np.zeros(frame.shape)
        temp3[idx]=frame[idx]
        #frames_list.append(otzi.astype(np.uint8))
        matted_frames_list.append((temp3).astype(np.uint8))
        
        pbar.update(1)
    utilis.write_video('Outputs\_test_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=matted_frames_list,isColor=True)

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




    just_ones = np.ones(mask.shape)
        just_zeros = np.zeros(mask.shape)
        shuff = just_zeros[max(0, y_mean - 2*(constants.SMALL_WINDOW_H//3)  ):min(h, y_mean +constants.SMALL_WINDOW_H//3 ),
                        max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2) ]
        just_ones[max(0, y_mean - 2*(constants.SMALL_WINDOW_H//3)  ):min(h, y_mean +constants.SMALL_WINDOW_H//3 )
        ,max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2) ] =shuff
    '''