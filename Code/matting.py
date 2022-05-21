import logging
import cv2
import numpy as np
import tqdm

import GeodisTK
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
    
    
    print('start mattin creation')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr):
        luma_frame,_,_ = cv2.split(frames_yuv[frame_idx])
        mask = frames_mask[frame_idx]
        mask = (mask>150).astype(np.uint8)

        #close a small rectangle over the binary image

        OFFSET = 30

        mask_x_axis = np.where(mask ==1)[1]
        mask_left_idx =np.min(mask_x_axis)
        mask_right_idx = np.max(mask_x_axis)
        mask_y_axis = np.where(mask ==1)[0]
        mask_top_idx = np.min(mask_y_axis)
        mask_bottom_idx = np.max(mask_y_axis)

        #add small offset to look at a larger image

        mask_left_idx= max(0,mask_left_idx-OFFSET)
        mask_right_idx = min(w,mask_right_idx+OFFSET)
        mask_bottom_idx = min(h,mask_bottom_idx + OFFSET)
        mask_top_idx = max(0,mask_top_idx+OFFSET)

        #resize the image to look just at a small window aroud the person

        small_luma_frame = luma_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_bgr_frame = frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_mask = mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_new_bg = new_bg[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]

        #inorder to recive a trimap we calculate the erode image as fg
        fg_mask = cv2.erode(mask,kernel=np.ones((7,7)),iterations=constants.ERODE_N_ITER)
        small_fg_mask = fg_mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        #for more documantation GoTo: https://github.com/taigw/GeodisTK/blob/master/demo2d.py
        small_fg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_fg_mask,1.0,constants.GEO_N_ITER)

        #inorder to recive a trimap we calculate the dialate image as bg the trimap
        #will be the diff between them

        bg_mask = cv2.dilate(mask,kernel=np.ones((3,3)),iterations=constants.DIAL_N_ITER)
        bg_mask = 1-bg_mask
        small_bg_mask = bg_mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        #for more documantation GoTo: https://github.com/taigw/GeodisTK/blob/master/demo2d.py
        small_bg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_bg_mask,1.0,constants.GEO_N_ITER)

        #now we build the trimap zone

        small_fg_dist_map = small_fg_dist_map/(small_fg_dist_map+small_bg_dist_map)
        small_bg_dist_map = 1-small_fg_dist_map
        small_trimap_dist_map = (np.abs(small_bg_dist_map-small_fg_dist_map<constants.EPSILON_SMALL_BAND))
        small_trimap_dist_map_idx = np.where(small_trimap_dist_map==1)

        #NOW WE WANT TO BUILD THE KDE FOR THE BG AND FG TO CALC THE PRIOR FOR ALPHA

        fg_idx = utilis.choose_randome_indecis(small_fg_dist_map,250)
        bg_idx = utilis.choose_randome_indecis(small_bg_dist_map,250)
        fg_pdf = utilis.matting_estimate_pdf(dataset_valus=small_bgr_frame,bw_method=1,idx= fg_idx )
        bg_pdf = utilis.matting_estimate_pdf(dataset_valus=small_bgr_frame,bw_method=1,idx= bg_idx )

        small_fg_probs = fg_pdf(small_bgr_frame[small_trimap_dist_map_idx])
        small_bg_probs = bg_pdf(small_bgr_frame[small_trimap_dist_map_idx])

        #NOW we want to find Alpha
        w_fg = np.power(small_fg_dist_map[small_trimap_dist_map_idx],-2)*small_fg_probs
        w_bg = np.power(small_bg_dist_map[small_trimap_dist_map_idx],-2)*small_bg_probs
        alpha = w_fg/(w_fg+w_bg)
        small_alpha = np.copy(small_fg_mask).astype(np.float)
        small_alpha[small_trimap_dist_map_idx]= alpha
        
        #now we implement the alpha 
        small_mated_frame = small_bgr_frame*small_alpha[:, :, np.newaxis]+small_new_bg*(1-small_alpha[:, :, np.newaxis])  #cv2.addWeighted(small_bgr_frame,small_alpha,small_new_bg,1-small_alpha,gamma=0)

        #resize to original frame size

        matted_frame = np.copy(new_bg)
        matted_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx] = small_mated_frame
        matted_frames_list.append(matted_frame)     

       


         



        pbar.update(1)
        

        
        
    
    utilis.write_video('Outputs\matt_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=matted_frames_list,isColor=True)

