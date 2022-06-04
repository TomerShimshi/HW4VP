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

EPSILON = 10**-30
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
    for frame_idx, frame in enumerate(frames_bgr[:5]):
        luma_frame,_,_ = cv2.split(frames_yuv[frame_idx])
        blue_fram,_,_ = cv2.split(frame)
        mask = frames_mask[frame_idx]
        mask = (mask>150).astype(np.uint8)

        #close a small rectangle over the binary image
        
        OFFSET = 50

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
       
        #y_mean,x_mean = (np.mean(mask_idx[0]).astype(int),np.mean(mask_idx[1]).astype(int))
        
        small_luma_frame = luma_frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_bgr_frame = frame[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_mask = mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        small_blue_frame = blue_fram[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        

        #inorder to recive a trimap we calculate the erode image as fg
      

        small_fg_mask =cv2.erode(small_mask,kernel=kernel,iterations=constants.ERODE_N_ITER)#cv2.morphologyEx(small_mask,cv2.cv2.MORPH_OPEN, kernel=kernel,iterations=constants.ERODE_N_ITER)
       
        #temp_mask =small_mask.copy()
        temp_mask =cv2.dilate(small_mask, kernel=kernel,iterations=constants.DIAL_N_ITER)
        small_bg_mask = 1-temp_mask
        
        #small_bg_mask = bg_mask[mask_top_idx:mask_bottom_idx,mask_left_idx:mask_right_idx]
        #for more documantation GoTo: https://github.com/taigw/GeodisTK/blob/master/demo2d.py


        #small_bg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_bg_mask,1.0,constants.GEO_N_ITER)


        #NOW WE WANT TO BUILD THE KDE FOR THE BG AND FG TO CALC THE PRIOR FOR ALPHA
        idx =  np.where(small_blue_frame >-1)
        grid = np.linspace(0,1900,1901)
        fg_idx = utilis.choose_randome_indecis(small_fg_mask,1450)
        bg_idx =  utilis.choose_randome_indecis(small_bg_mask,1450)
        fg_pdf = utilis.matting_estimate_pdf_test(dataset_valus=small_luma_frame,bw_method=constants.BW_MATTING,idx= fg_idx ,grid=grid)
        bg_pdf = utilis.matting_estimate_pdf_test(dataset_valus=small_luma_frame,bw_method=constants.BW_MATTING,idx= bg_idx,grid=grid )
        denominator = np.add(fg_pdf,bg_pdf)

        fg_pdf = fg_pdf/denominator
        bg_pdf = bg_pdf/denominator

        PB = bg_pdf[small_luma_frame]
        PF = fg_pdf[small_luma_frame]
        #PF= np.where(PF>constants.Min_Prob,255,0)
        #PB= 255-PF
        small_fg_probs_for_dist =cv2.normalize(src=PF, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) #np.rint(255*small_bg_probs/(np.max(small_bg_probs) - np.min(small_bg_probs)),dtype= uint8)
        small_bg_probs_for_dist =cv2.normalize(src=PB, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        small_bg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_bg_probs_for_dist,1.0,constants.GEO_N_ITER)
        small_fg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_fg_probs_for_dist,1.0,constants.GEO_N_ITER)

        small_bg_dist_map = ((small_bg_dist_map+EPSILON)/(small_fg_dist_map+small_fg_dist_map+EPSILON))*255#(small_bg_dist_map /np.max(small_bg_dist_map))*255 #((small_bg_dist_map+EPSILON)/(small_fg_dist_map+small_fg_dist_map+EPSILON))*255
        small_bg_dist_map= np.where(small_bg_dist_map>255,255,small_bg_dist_map)
        small_fg_dist_map = 255-small_bg_dist_map #(small_fg_dist_map/np.max(small_fg_dist_map))*255


        #PF= ((PF/np.max(PF)))
        #PF = np.where(PF>constants.Min_Prob,1,0)
        #PB= 1- PF
        
        
        
        
       
        small_bg_probs = bg_pdf[small_luma_frame[:,:]]*255
        small_fg_probs = fg_pdf[small_luma_frame]*255


        #small_bg_probs = small_bg_probs/np.sum(small_bg_probs)
        #small_fg_probs = small_fg_probs/np.sum(small_fg_probs)

        #small_bg_probs = (small_bg_probs/(small_bg_probs.max()- small_bg_probs.min()))*255.0+small_bg_probs.min()
        #small_fg_probs = (small_fg_probs/(small_fg_probs.max()- small_fg_probs.min()))*255.0+small_fg_probs.min()

        #small_fg_probs=small_fg_probs/(small_fg_probs+small_bg_probs)
        #small_bg_probs = 1.0-small_fg_probs
        
        
        #small_fg_probs=small_fg_probs/(small_fg_probs+small_bg_probs)
        #small_bg_probs = 1.0-small_fg_probs+EPSILON
        '''
        small_fg_probs_for_dist= np.where(small_fg_probs>constants.Min_Prob,1,0)

        small_fg_probs = cv2.resize(small_fg_probs,(small_luma_frame.shape[1],small_luma_frame.shape[0]))
        small_bg_probs = cv2.resize(small_bg_probs,(small_luma_frame.shape[1],small_luma_frame.shape[0]))
        
        small_bg_probs_for_dist=1.0-small_fg_probs_for_dist
        small_luma_frame = small_luma_frame.astype(np.float32)
        dest = np.zeros(small_bg_probs.shape)
        dest2 = np.zeros(small_bg_probs.shape)
        dest2 =cv2.normalize(src=small_fg_probs, dst=dest2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) #np.rint(255*small_bg_probs/(np.max(small_bg_probs) - np.min(small_bg_probs)),dtype= uint8)
        dest =cv2.normalize(src=small_bg_probs, dst=dest, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        
        #shuff = np.copy(temp).astype(uint8)
        #temp=temp.astype(uint8)
        
        
        #now we buikd the dist map


        #small_bg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_bg_probs_for_dist,1.0,constants.GEO_N_ITER)
        #small_fg_dist_map = GeodisTK.geodesic2d_raster_scan(small_luma_frame,small_fg_probs_for_dist,1.0,constants.GEO_N_ITER)
        '''

        #temp_mask[max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)] = small_fg_mask
        #idx_fg = np.where(dest2 >-1)
        #idx_bg = np.where(dest2 >-1)
        temp_fg = np.zeros(mask.shape)
        temp_bg = np.zeros(mask.shape)
        temp_fg[idx]=small_fg_dist_map[idx]
        temp_bg[idx]=small_bg_dist_map[idx]
        #frames_list.append(otzi.astype(np.uint8))
        matted_frames_list.append((temp_fg).astype(np.uint8))
        alpha_frame_list.append((temp_bg).astype(np.uint8))
        
        pbar.update(1)
    utilis.write_video('Outputs\_test_fg_probs_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=matted_frames_list,isColor=False)
    utilis.write_video('Outputs\_test_bg_probs_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=alpha_frame_list,isColor=False)

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