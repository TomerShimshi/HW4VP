from audioop import reverse
import logging
import cv2
import numpy as np
import tqdm

from scipy.stats import gaussian_kde

import constants

import utilis

my_logger = logging.getLogger('MyLogger')

def background_subtraction(input_video_path):
    my_logger.info('Starting Background Subtraction')
    cap = cv2.VideoCapture(input_video_path)
    parameters = utilis.get_video_parameters(cap)
    frames_bgr =  utilis.load_video(cap,wanted_colors='bgr')
    frames_hsv = utilis.load_video(cap,wanted_colors='hsv')
    n_frames = len(frames_bgr)
    fgbg = cv2.createBackgroundSubtractorKNN(history=600,detectShadows=False,dist2Threshold =800.0)
    mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
    print('started studing frames history')
    pbar = tqdm.tqdm(total=12*n_frames)
    for i in range(12):
        for frame_idx, frame in enumerate(frames_hsv):
            frame_hsv= frame[:,:,2]
            fg_mask = fgbg.apply(frame_hsv)
            fg_mask = (fg_mask>200).astype(np.uint8)
            mask_list[frame_idx]= fg_mask
            pbar.update(1)
    print('finished studying video history')
    fg_colors, bg_colors = None,None
    fg_shoes_colors,bg_shoes_colors = None,None
    person_and_blue_mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
    '''now we start to collect the coloros in order to learn the shoes snd body gaussian KDE '''
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr):
        blue_fram,_,_ = cv2.split(frame)
        mask= mask_list[frame_idx]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel=kernel)
        mask = cv2.medianBlur(mask,ksize=5)
        _,contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key= cv2.contourArea,reverse = True)
        person_mask = np.zeros(mask.shape)
        cv2.fillPoly(person_mask,pts=contours[0],color=1)
        blue_mask = (blue_fram<constants.BLUE_MASK_T).astype(np.uint8)
        person_and_blue_mask = (blue_mask*person_mask).astype(np.uint8)
        fg_indices = utilis.choose_randome_indecis(person_and_blue_mask,20,True)
        bg_indices = utilis.choose_randome_indecis(person_and_blue_mask,20,False)
        #$$$$$$$$$$$$ Mybe need to find ccolors for the shoes $$$$$$$$$$$$
        person_and_blue_mask_list[frame_idx] = person_and_blue_mask
        if fg_colors is None:
            fg_colors = frame[fg_indices[:,0],fg_indices[:,1]]
            bg_colors = frame[bg_indices[:,0],bg_indices[:,1]]
        
        else:
            fg_colors = np.concatenate((fg_colors, frame[fg_indices[:,0], fg_indices[:,1]]))
            bg_colors =np.concatenate((bg_colors, frame[bg_indices[:,0],bg_indices[:,1]] ))
        fg_pdf = utilis.estimate_pdf(dataset_valus= fg_colors,bw_method=constants.BW_MEDIUM)
        bg_pdf = utilis.estimate_pdf(dataset_valus= bg_colors,bw_method=constants.BW_MEDIUM)
        fg_pdf_memo, bg_pdf_memo= dict(),dict()
        or_mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))







