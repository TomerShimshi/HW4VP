import logging
from pickletools import uint8
import cv2
import numpy as np
import tqdm

import GeodisTK
from scipy.stats import gaussian_kde

import constants

import utilis

ID1 = 203200480
ID2 = 320521461

EPSILON = 10**-10
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
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    print('start mattin creation')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:]):
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
        mask_top_idx = max(0,mask_top_idx-OFFSET)
        
        #resize the image to look just at a small window aroud the person
        mask_idx = np.where(mask==1)
        #y_mean,x_mean = (np.mean(mask_idx[0]).astype(int),np.mean(mask_idx[1]).astype(int))
        
        small_luma_frame = luma_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_bgr_frame = frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_mask = mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_new_bg = new_bg[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        '''
        small_luma_frame = luma_frame[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        small_bgr_frame = frame[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        small_mask = mask[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        small_new_bg = new_bg[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                             max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        '''

        #inorder to recive a trimap we calculate the erode image as fg
        '''
        now we try somthing new

        

        small_mask_idx = np.where(small_mask == 1)
        shuff = mask[max(0, y_mean - 2*(constants.SMALL_WINDOW_H//3)  ):min(h, y_mean +constants.SMALL_WINDOW_H//3 ),
                        max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2) ]
        small_fg_mask = np.zeros(mask.shape).astype(np.uint8)
        #small_fg_mask = cv2.erode(small_mask,kernel=np.ones((7,7)),iterations=constants.ERODE_N_ITER)
        small_fg_mask [max(0, y_mean - 2*(constants.SMALL_WINDOW_H//3)  ):min(h, y_mean +constants.SMALL_WINDOW_H//3 ),
                        max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2) ]= shuff #shuff [max(0, y_mean - constants.SMALL_WINDOW_H  // 2):min(h, y_mean + constants.SMALL_WINDOW_H // 2),
                                                                                                                           #max(0, x_mean - constants.SMALL_WINDOW_W // 2):min(w, x_mean + constants.SMALL_WINDOW_W // 2) ]
        
        small_fg_mask =small_fg_mask[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)] 
        temp2 = np.max(small_fg_mask)
        if temp2 == 0:
            t=1
        '''

        small_fg_mask =cv2.erode(small_mask,kernel=kernel,iterations=constants.ERODE_N_ITER)#cv2.morphologyEx(small_mask,cv2.cv2.MORPH_OPEN, kernel=kernel,iterations=constants.ERODE_N_ITER)
        #cv2.erode(small_mask,kernel=np.ones((7,7)),iterations=constants.ERODE_N_ITER)
       
        #small_fg_mask = fg_mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        #for more documantation GoTo: https://github.com/taigw/GeodisTK/blob/master/demo2d.py
        small_fg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_fg_mask,1.0,constants.GEO_N_ITER)

        #inorder to recive a trimap we calculate the dialate image as bg the trimap
        #will be the diff between them

        #small_bg_mask = cv2.dilate(small_mask,kernel=np.ones((3,3)),iterations=constants.DIAL_N_ITER)
        temp_mask =small_mask.copy()
        temp_mask =cv2.dilate(temp_mask, kernel=kernel,iterations=constants.DIAL_N_ITER)
        small_bg_mask = 1-temp_mask
        #small_bg_mask = bg_mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        #for more documantation GoTo: https://github.com/taigw/GeodisTK/blob/master/demo2d.py
        small_bg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_bg_mask,1.0,constants.GEO_N_ITER)

        #now we build the trimap zone

        small_fg_dist_map = small_fg_dist_map/(small_fg_dist_map+small_bg_dist_map+EPSILON)
        small_bg_dist_map = 1-small_fg_dist_map
        small_trimap_dist_map = (np.abs(small_bg_dist_map-small_fg_dist_map<constants.EPSILON_SMALL_BAND))
        small_trimap_dist_map_idx = np.where(small_trimap_dist_map==1)

        small_accepted_fg_mask = (small_fg_dist_map<small_bg_dist_map-constants.EPSILON_SMALL_BAND).astype(np.uint8)
        small_accepted_bg_mask = (small_bg_dist_map>=small_fg_dist_map-constants.EPSILON_SMALL_BAND).astype(np.uint8)
        temp = np.count_nonzero(small_accepted_fg_mask)
        if temp<150:
            t=1
        temp = np.count_nonzero(small_accepted_bg_mask)
        if temp<150:
            t=1
        #NOW WE WANT TO BUILD THE KDE FOR THE BG AND FG TO CALC THE PRIOR FOR ALPHA

        fg_idx = utilis.choose_randome_indecis(small_accepted_fg_mask,220)
        bg_idx = utilis.choose_randome_indecis(small_accepted_bg_mask,220)
        fg_pdf = utilis.matting_estimate_pdf(dataset_valus=small_bgr_frame,bw_method=constants.BW_MATTING,idx= fg_idx )
        bg_pdf = utilis.matting_estimate_pdf(dataset_valus=small_bgr_frame,bw_method=constants.BW_MATTING,idx= bg_idx )

        small_fg_probs = fg_pdf(small_bgr_frame[small_trimap_dist_map_idx])
        small_bg_probs = bg_pdf(small_bgr_frame[small_trimap_dist_map_idx])

        #NOW we want to find Alpha
        w_fg =small_fg_probs/ (EPSILON+np.power(small_fg_dist_map[small_trimap_dist_map_idx],constants.R))
        w_bg = small_bg_probs /(EPSILON+np.power(small_bg_dist_map[small_trimap_dist_map_idx],constants.R))
        alpha = w_fg/(w_fg+w_bg+EPSILON)
        
        #small_fg_mask_test =cv2.morphologyEx(small_mask,cv2.cv2.MORPH_OPEN, kernel=np.ones((11,11)),iterations=constants.ERODE_N_ITER) #cv2.erode(small_mask,kernel=np.ones((11,11)),iterations=constants.ERODE_N_ITER)
        #small_alpha = np.copy(small_fg_mask).astype(np.float)
        small_alpha = np.copy(small_accepted_fg_mask).astype(np.float)#np.copy(small_fg_mask).astype(np.float)
        small_alpha[small_trimap_dist_map_idx]= np.maximum(alpha,small_fg_mask[small_trimap_dist_map_idx])

        
        #now we implement the alpha 
        #small_mated_frame = small_bgr_frame*small_alpha[:, :, np.newaxis]+small_new_bg*(1-small_alpha[:, :, np.newaxis])  #cv2.addWeighted(small_bgr_frame,small_alpha,small_new_bg,1-small_alpha,gamma=0)
        small_mated_frame = utilis.use_mask_on_frame(frame=small_bgr_frame,mask=small_alpha) + utilis.use_mask_on_frame(frame=small_new_bg,mask=1-small_alpha)  #small_bgr_frame*small_alpha[:, :, np.newaxis]+small_new_bg*(1-small_alpha[:, :, np.newaxis])
        #resize to original frame size

        matted_frame = np.copy(new_bg)
        '''
        matted_frame[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)] = small_mated_frame
        '''
        matted_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx] = small_mated_frame
        matted_frames_list.append(matted_frame)  

        alpha_frame = np.zeros(mask.shape) 
        #alpha_frame[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
        #             max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]  = small_alpha
        alpha_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx] = small_alpha
        alpha_frame= (alpha_frame*255).astype(np.uint8)
        alpha_frame_list.append(alpha_frame)   

       


         



        pbar.update(1)
        

        
        
    
    utilis.write_video('Outputs\matt_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=matted_frames_list,isColor=True)

    utilis.write_video('Outputs\_alpha_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=alpha_frame_list,isColor=False)

