import logging
from pickletools import uint8
import cv2
import numpy as np
import tqdm

import constants

import utilis

ID1 = 203200480
ID2 = 320521461

EPSILON = 10**-30
my_logger = logging.getLogger('MyLogger')
def tracking (input_video_path, Alpha_mask_path):
    my_logger.info('Starting Background Subtraction')
    cap = cv2.VideoCapture(input_video_path)
    cap_mask = cv2.VideoCapture(Alpha_mask_path)
    parameters = utilis.get_video_parameters(cap)
    h,w = parameters["height"],parameters['width']
    #load the frames
    frames_bgr =  utilis.load_video(cap,wanted_colors='bgr')
    frames_mask = utilis.load_video(cap_mask,wanted_colors='gray')
    
    
    n_frames = len(frames_bgr)
    tracked_frames_list= []
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    print('start mattin creation')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:]):
        mask = frames_mask[frame_idx]
        mask = (mask>150).astype(np.uint8)

        #close a small rectangle over the binary image
        
        

        mask_x_axis = np.where(mask ==1)[1]
        mask_left_idx =np.min(mask_x_axis)
        mask_right_idx = np.max(mask_x_axis)
        mask_y_axis = np.where(mask ==1)[0]
        mask_top_idx = np.min(mask_y_axis)
        mask_bottom_idx = np.max(mask_y_axis)

        #add small offset to look at a larger image

        mask_left_idx= max(0,mask_left_idx)
        mask_right_idx = min(w,mask_right_idx)
        mask_bottom_idx = min(h,mask_bottom_idx )
        mask_top_idx = max(0,mask_top_idx)

        start_point = (mask_left_idx,mask_top_idx)

        end_point= (mask_right_idx,mask_bottom_idx)
        frame =cv2.rectangle(frame, start_point, end_point, (255,0,0), thickness=2) 
        tracked_frames_list.append(frame)
        pbar.update(1)

    utilis.write_video('Outputs\Output_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=tracked_frames_list,isColor=True)



        