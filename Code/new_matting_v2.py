
from pickletools import uint8
import time
import cv2
import numpy as np
import tqdm

import GeodisTK
from scipy.stats import gaussian_kde

import constants

import utilis

ID1 = 203200480
ID2 = 320521461

EPSILON = 10**-30

def matting (input_video_path, BW_mask_path,bg_path):
    start = time.time()
   
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
        blue_fram,_,_ = cv2.split(frame)
        mask = frames_mask[frame_idx]
        mask = (mask>150).astype(np.uint8)

        #close a small rectangle over the binary image
        
        OFFSET = 120

        mask_x_axis = np.where(mask ==1)[1]
        mask_left_idx =np.min(mask_x_axis)
        mask_right_idx = np.max(mask_x_axis)
        mask_y_axis = np.where(mask ==1)[0]
        mask_top_idx = np.min(mask_y_axis)
        mask_bottom_idx = np.max(mask_y_axis)

        #add small offset to look at a larger image

        mask_left_idx= max(0,mask_left_idx-OFFSET//2)
        mask_right_idx = min(w,mask_right_idx+OFFSET//2)
        mask_bottom_idx = min(h,mask_bottom_idx + OFFSET//2)
        #INCREASE MOSTTLY AROUD THE HEAD EREA
        mask_top_idx = max(0,mask_top_idx-OFFSET)
        
        #resize the image to look just at a small window aroud the person        
        small_luma_frame = luma_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_luma_frame = np.float32(small_luma_frame)
        small_bgr_frame = frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_mask = mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_new_bg = new_bg[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_blue_frame = blue_fram[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        

        #inorder to recive a trimap we calculate the erode image as fg
        

        fg_mask =cv2.erode(mask,kernel=kernel,iterations=constants.ERODE_N_ITER)#cv2.morphologyEx(small_mask,cv2.cv2.MORPH_OPEN, kernel=kernel,iterations=constants.ERODE_N_ITER)
        small_fg_mask = fg_mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx].astype(np.uint8)
        
        #WE FIRST CALCULATE THE DIST MAP WITH THE SMALL FG MASK INSTED OF THE PROB MAP
        #WE TRIED ALEX METHOD AND ADDED THE CODE TO THE REPORT BUT WE GOT BAD RESULTS
        #for more documantation GoTo: https://github.com/taigw/GeodisTK/blob/master/demo2d.py
        small_fg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_fg_mask,1.0,constants.GEO_N_ITER)

        #WE USE 1-THE DIALATE MASK IN ORDER TO LOOK AT THE BG
        bg_mask =cv2.dilate(mask, kernel=kernel,iterations=constants.DIAL_N_ITER)
        #try to solve the vanishing head problem
        bg_mask[:constants.FACE_HIGHT, :] =cv2.dilate(bg_mask[:constants.FACE_HIGHT, :], kernel=kernel,iterations=constants.DIAL_N_ITER)
        bg_mask = 1-bg_mask
        small_bg_mask = bg_mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
       
        #for more documantation GoTo: https://github.com/taigw/GeodisTK/blob/master/demo2d.py
        small_bg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_bg_mask,1.0,constants.GEO_N_ITER)

        ### WE ADD EPSILON TO AVIOD ZERO DIV
        small_bg_dist_map[small_bg_dist_map==0]=constants.EPSILON
        small_fg_dist_map[small_fg_dist_map==0]=constants.EPSILON
        denominator= np.add(small_fg_dist_map,small_bg_dist_map)
        #now we build the trimap zone

        small_fg_dist_map = small_fg_dist_map/denominator
        small_bg_dist_map = small_bg_dist_map/denominator
        small_trimap_dist_map = (np.abs(small_bg_dist_map-small_fg_dist_map)<constants.EPSILON_SMALL_BAND)
        small_trimap_dist_map_idx = np.where(small_trimap_dist_map==1)

        small_accepted_fg_mask = (small_fg_dist_map<small_bg_dist_map).astype(np.uint8)
        small_accepted_bg_mask = (small_bg_dist_map>small_fg_dist_map).astype(np.uint8)
      
        #NOW WE WANT TO BUILD THE KDE FOR THE BG AND FG TO CALC THE PRIOR FOR ALPHA

        fg_idx = utilis.choose_randome_indecis(small_accepted_fg_mask,1520)
        bg_idx = utilis.choose_randome_indecis(small_accepted_bg_mask,1520)
        grid = np.linspace(0,2900,2901)
        fg_pdf = utilis.matting_estimate_pdf_test(dataset_valus=small_blue_frame,bw_method=constants.BW_MATTING,idx= fg_idx ,grid=grid)
        bg_pdf = utilis.matting_estimate_pdf_test(dataset_valus=small_blue_frame,bw_method=constants.BW_MATTING,idx= bg_idx,grid=grid )
        ### WE ADD EPSILON TO AVIOD ZERO DIV
        fg_pdf[fg_pdf==0]=EPSILON
        bg_pdf[bg_pdf==0]= EPSILON
        denominator = np.add(fg_pdf,bg_pdf)
        #WE DIVIDE THE pdf's IN ORDER TO NORMALIZE THEMe
        fg_pdf = (fg_pdf)/(denominator)
        bg_pdf = (bg_pdf)/(denominator)
        

        #NOW WE CALCULATE THE THE PROBS FOR THE BG AND FG
        PB = bg_pdf[small_blue_frame]
        PF = fg_pdf[small_blue_frame]

        #NOW we want to find Alpha
        w_fg =PF[small_trimap_dist_map_idx]* (np.power(small_fg_dist_map[small_trimap_dist_map_idx],-constants.R))
        w_bg = PB[small_trimap_dist_map_idx]*(np.power(small_bg_dist_map[small_trimap_dist_map_idx],-constants.R))
        alpha = w_fg/(w_fg+w_bg)

        #$$$$$%%%%%test$$$$$%%%%%%%%
        #alpha = (alpha/(alpha.max()))*1.0
       
        small_alpha =np.copy(small_accepted_fg_mask).astype(np.float)#np.copy(small_fg_mask).astype(np.float)
        small_alpha[small_trimap_dist_map_idx]= alpha #np.maximum(alpha,small_accepted_fg_mask[small_trimap_dist_map_idx])
        
        
        #now we implement the alpha 
        
        small_mated_frame = utilis.use_mask_on_frame(frame=small_bgr_frame,mask=small_alpha) + utilis.use_mask_on_frame(frame=small_new_bg,mask=1-small_alpha)  #small_bgr_frame*small_alpha[:, :, np.newaxis]+small_new_bg*(1-small_alpha[:, :, np.newaxis])
        #resize to original frame size

        matted_frame = np.copy(new_bg)
       
        matted_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx] = small_mated_frame
        matted_frames_list.append(matted_frame)  

        alpha_frame = np.zeros(frame.shape) 
        alpha_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx] = 1
        alpha_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx] =utilis.use_mask_on_frame(frame=alpha_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx],mask=small_alpha).astype(np.uint8) #small_alpha
        alpha_frame= (alpha_frame*255).astype(np.uint8)
        alpha_frame_list.append(alpha_frame)   

       


         



        pbar.update(1)
        

        
        
    
    utilis.write_video('Outputs/matted_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=matted_frames_list,isColor=True)

    utilis.write_video('Outputs/alpha_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=alpha_frame_list,isColor=True)
    end = time.time()
    print('matting took {}'.format(end-start))

