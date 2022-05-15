from audioop import reverse
import logging
from tkinter import X
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
    frames_hsv = utilis.load_video(cap,wanted_colors='hsv')
    
    n_frames = len(frames_bgr)
    #create the backround subtractor
    fgbg = cv2.createBackgroundSubtractorKNN(history=600,detectShadows=False,dist2Threshold =90.0)
    mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
    print('started studing frames history')
    pbar = tqdm.tqdm(total=12*n_frames)
    for i in range(12):
        for frame_idx, frame in enumerate(frames_hsv):
            frame_hsv= frame[:,:,1:]
            fg_mask = fgbg.apply(frame_hsv)
            #stuff for debug
            temp = np.max(fg_mask)
            if frame_idx %50 ==0:
                t=1
            fg_mask = (fg_mask>200).astype(np.uint8)
            mask_list[frame_idx]= fg_mask
            pbar.update(1)
    print('finished studying video history')
    fg_colors, bg_colors = None,None
    fg_shoes_colors,bg_shoes_colors = None,None
    person_and_blue_mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
    
    print('start collect color KDE')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr):
        blue_fram,_,_ = cv2.split(frame)
        mask= mask_list[frame_idx]
        temp =np.max(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)) 
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel=kernel,iterations=2).astype(np.uint8)
        mask = cv2.medianBlur(mask,ksize=7)
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        #contours.sort(key= cv2.contourArea,reverse = True)
        person_mask = np.zeros(mask.shape)
        #cv2.fillPoly(person_mask,pts=[contours[0]],color=1)
        cv2.drawContours(person_mask, [max(contours, key = cv2.contourArea)], -1, color=(1, 1, 1), thickness=cv2.FILLED)
        temp =np.max(person_mask)
        #person_mask = 
        blue_mask = (blue_fram<constants.BLUE_MASK_T).astype(np.uint8)
        temp =np.max(blue_mask)
        person_and_blue_mask =(blue_mask*person_mask).astype(np.uint8)
        temp =np.max(person_and_blue_mask)
        fg_indices = utilis.choose_randome_indecis(person_and_blue_mask,22,True)
        bg_indices = utilis.choose_randome_indecis(person_and_blue_mask,22,False)
        #$$$$$$$$$$$$ Mybe need to find ccolors for the shoes $$$$$$$$$$$$
        person_and_blue_mask_list[frame_idx] = person_and_blue_mask
        temp =np.max(person_and_blue_mask)
        if fg_colors is None:
            fg_colors = frame[fg_indices[:,0],fg_indices[:,1]]
            bg_colors = frame[bg_indices[:,0],bg_indices[:,1]]
        
        else:
            fg_colors = np.concatenate((fg_colors, frame[fg_indices[:,0], fg_indices[:,1]]))
            bg_colors =np.concatenate((bg_colors, frame[bg_indices[:,0],bg_indices[:,1]] ))
        pbar.update(1)
    fg_pdf = utilis.estimate_pdf(dataset_valus= fg_colors,bw_method=constants.BW_MEDIUM)
    bg_pdf = utilis.estimate_pdf(dataset_valus= bg_colors,bw_method=constants.BW_MEDIUM)
    fg_pdf_memo, bg_pdf_memo= dict(),dict()
    or_mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))

    #filtering using the KDE
    print('start the KDE filtering')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:]):
        person_and_blue_mask = person_and_blue_mask_list[frame_idx]
        person_and_blue_mask_indecis = np.where(person_and_blue_mask ==1)
        y_mean,x_mean = (np.mean(person_and_blue_mask_indecis[0]).astype(int),np.mean(person_and_blue_mask_indecis[1]).astype(int))
        small_frame_bgr = frame[max(0, y_mean - constants.WINDOW_H // 2):min(h, y_mean + constants.WINDOW_H // 2),
                          max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2),  :]
        small_person_and_blue_mask = person_and_blue_mask[
                                     max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        small_person_and_blue_mask_idx = np.where(small_person_and_blue_mask == 1)
        small_prop_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_fg_prob_stacked = np.fromiter(map(lambda elem:utilis.check_if_in_dic(fg_pdf_memo,elem,fg_pdf),map(tuple,small_frame_bgr[small_person_and_blue_mask_idx])),
        dtype= float)
        small_bg_prob_stacked = np.fromiter(map(lambda elem:utilis.check_if_in_dic(bg_pdf_memo,elem,bg_pdf),map(tuple,small_frame_bgr[small_person_and_blue_mask_idx])),
        dtype= float)
        small_probs_fg_bigger_bg_mask= np.zeros(small_person_and_blue_mask.shape)
        small_probs_fg_bigger_bg_mask[small_person_and_blue_mask_idx]= (small_fg_prob_stacked>small_bg_prob_stacked).astype(np.uint8)
        
        
        small_or_mask = np.zeros(small_probs_fg_bigger_bg_mask.shape)
        small_or_mask = small_probs_fg_bigger_bg_mask
        or_mask = np.zeros(person_and_blue_mask.shape)
        or_mask [max(0,y_mean-constants.WINDOW_H//2):min(h,y_mean+constants.WINDOW_H//2),max(0,x_mean- constants.WINDOW_W//2):min(w,x_mean+constants.WINDOW_W//2)]=small_or_mask
        or_mask_list[frame_idx]=or_mask
        pbar.update(1)

        #final proccseing

    print('final proccseing')
    final_masks_list, final_frames_list = [], []
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr):
        or_mask = or_mask_list[frame_idx]
        #or_mask =person_and_blue_mask_list[frame_idx] #mask_list[frame_idx]
        final_mask = np.copy(or_mask).astype(np.uint8)
        temp= np.max(final_mask)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cv2.contourArea, reverse=True)
        final_contour_mask = np.zeros(final_mask.shape)
        try:
            #cv2.fillPoly(final_contour_mask, pts=[contours[0]], color=1)
            cv2.drawContours(final_contour_mask, [max(contours, key = cv2.contourArea)], -1, color=(1, 1, 1), thickness=cv2.FILLED)
        except:
            pass
        final_mask = (final_contour_mask * final_mask).astype(np.uint8)
        #final_mask [final_contour_mask == 0] = 0#(final_contour_mask * final_mask).astype(np.uint8)
        temp_2 = np.max(final_mask)
        final_masks_list.append(final_mask)
        temp =np.max( utilis.use_mask_on_frame(frame=frame,mask=final_mask))
        final_frames_list.append(utilis.use_mask_on_frame(frame=frame,mask=final_mask))
        pbar.update(1)

        #or_mas
        
        

    utilis.write_video('Outputs\extracted_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=final_frames_list,isColor=True)




        


            








