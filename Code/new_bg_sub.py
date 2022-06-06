   

import logging
from sndhdr import whathdr
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
    num_iter = 10
    fgbg = cv2.createBackgroundSubtractorKNN(history=num_iter*n_frames,detectShadows=False,dist2Threshold =30)#45)#90.0)
    mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
   
    print('started studing frames history')
    pbar = tqdm.tqdm(total=num_iter*n_frames)
    for i in range(num_iter):
        for frame_idx, frame in enumerate(frames_gray[:]):
            #frame_hsv= frame[:,:,1:]
            fg_mask = fgbg.apply(frame)
            #if i == num_iter-1:
            #    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            #    #fg_mask[constants.SHOES_HIGHT:,:] = cv2.erode(fg_mask,kernel,iterations =1)
            #    fg_mask[constants.SHOES_HIGHT:,:] = cv2.dilate(fg_mask[constants.SHOES_HIGHT:,:],kernel,iterations =2)


            fg_mask = (fg_mask>200).astype(np.uint8)

            mask_list[frame_idx]= fg_mask
            pbar.update(1)
    print('finished studying video history')
    fg_colors, bg_colors = None,None
    middle_fg_colors, middle_bg_colors = None,None
    fg_shoes_colors,bg_shoes_colors = None,None
    person_and_blue_mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
    
    print('start collect color KDE')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:]):
        blue_fram,_,_ = cv2.split(frame)
        mask= mask_list[frame_idx].astype(np.uint8)
        temp =np.max(mask)
        mask = cv2.medianBlur(mask,ksize=7)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel=kernel,iterations=1).astype(np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel=kernel,iterations=2).astype(np.uint8)
        #mask = cv2.bilateralFilter(mask,d=15,sigmaColor=85,sigmaSpace=85) #cv2.bilateralFilter(mask,d=15,sigmaColor=85,sigmaSpace=85) #cv2.medianBlur(mask,ksize=7)
        #mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel=kernel,iterations=1).astype(np.uint8)
        
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        person_mask = np.zeros(mask.shape)
        cv2.drawContours(person_mask, [max(contours, key = cv2.contourArea)], -1, color=(1, 1, 1), thickness=cv2.FILLED)
        temp =np.max(person_mask)
        
        blue_mask = (blue_fram<constants.BLUE_MASK_T).astype(np.uint8)
        temp =np.max(blue_mask)
        person_and_blue_mask =(blue_mask*person_mask).astype(np.uint8)
        temp =np.max(person_and_blue_mask)
        #$$$%% now we split the masks for upper middle and lower

        #$$$$$%% UPPER PART
        upper_mask= person_and_blue_mask.copy()
        upper_mask = cv2.morphologyEx(upper_mask,cv2.MORPH_CLOSE,kernel=kernel,iterations=1).astype(np.uint8)
        upper_mask = cv2.morphologyEx(upper_mask,cv2.MORPH_CLOSE,kernel=np.ones((20, 1)),iterations=1).astype(np.uint8)
        '''
        we give a diffrent number than zero in order to
        niglact this pixwls while calculating the KDE
        '''
        OFFSET = 30
        upper_mask[h//2:,:]=2
        
        temp =np.max(upper_mask)
        fg_indices = utilis.choose_randome_indecis(upper_mask,82,True)
        bg_indices = utilis.choose_randome_indecis(upper_mask,82,False)

        ###@@###$%%%%% Middle part
        middle_mask= person_and_blue_mask.copy()
        middle_mask[:h//3-OFFSET//2,:]=2
        middle_mask[2*(h//3)+OFFSET//2:,:]=2
        fg_indices_middle = utilis.choose_randome_indecis(middle_mask,42,True)
        bg_indices_middle = utilis.choose_randome_indecis(middle_mask,42,False)

        #$$$$$$$$$$$$ now the last lower part $$$$$$$$$$$$
        shoes_mask = person_and_blue_mask.copy()
        shoes_mask[:2*(h//3)+OFFSET,:]=2
        fg_shoes_indices = utilis.choose_randome_indecis(shoes_mask,42,True)
        bg_shoes_indices = utilis.choose_randome_indecis(shoes_mask,42,False)
        person_and_blue_mask_list[frame_idx] = person_and_blue_mask
        temp =np.max(person_and_blue_mask)
        if fg_colors is None:
            fg_colors = frame[fg_indices[:,0],fg_indices[:,1]]
            bg_colors = frame[bg_indices[:,0],bg_indices[:,1]]
            middle_fg_colors = frame[fg_indices_middle[:,0],fg_indices_middle[:,1]]
            middle_bg_colors = frame[bg_indices_middle[:,0],bg_indices_middle[:,1]]
            fg_shoes_colors = frame[fg_shoes_indices[:,0],fg_shoes_indices[:,1]]
            bg_shoes_colors = frame[bg_shoes_indices[:,0],bg_shoes_indices[:,1]]
        
        else:
            fg_colors = np.concatenate((fg_colors, frame[fg_indices[:,0], fg_indices[:,1]]))
            bg_colors =np.concatenate((bg_colors, frame[bg_indices[:,0],bg_indices[:,1]] ))
            middle_fg_colors = np.concatenate((middle_fg_colors, frame[fg_indices_middle[:,0], fg_indices_middle[:,1]]))
            middle_bg_colors =np.concatenate((middle_bg_colors, frame[bg_indices_middle[:,0],bg_indices_middle[:,1]] ))
            fg_shoes_colors = np.concatenate((fg_shoes_colors, frame[fg_shoes_indices[:,0], fg_shoes_indices[:,1]]))
            bg_shoes_colors =np.concatenate((bg_shoes_colors, frame[bg_shoes_indices[:,0],bg_shoes_indices[:,1]] ))
        pbar.update(1)
    print('start building the pdf')
    fg_pdf = utilis.estimate_pdf(dataset_valus= fg_colors,bw_method=constants.BW_NARROW)
    bg_pdf = utilis.estimate_pdf(dataset_valus= bg_colors,bw_method=constants.BW_NARROW)
    fg_middle_pdf = utilis.estimate_pdf(dataset_valus= middle_fg_colors,bw_method=constants.BW_MEDIUM)
    bg_middle_pdf = utilis.estimate_pdf(dataset_valus= middle_bg_colors,bw_method=constants.BW_MEDIUM)
    fg_shoes_pdf = utilis.estimate_pdf(dataset_valus= fg_shoes_colors,bw_method=constants.BW_MEDIUM)
    bg_shoes_pdf = utilis.estimate_pdf(dataset_valus= bg_shoes_colors,bw_method=constants.BW_MEDIUM)
    fg_pdf_memo, bg_pdf_memo= dict(),dict()
    fg_middle_pdf_memo, bg_middle_pdf_memo= dict(),dict()
    fg_shoes_pdf_memo, bg_shoes_pdf_memo= dict(),dict()

    or_mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))

    #filtering using the KDE
    print('start the KDE filtering')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:]):
        person_and_blue_mask = person_and_blue_mask_list[frame_idx]
        person_and_blue_mask_indecis = np.where(person_and_blue_mask ==1)
        #### NOW we split from upper lower and middle
        
        y_mean,x_mean = (np.mean(person_and_blue_mask_indecis[0]).astype(int),np.mean(person_and_blue_mask_indecis[1]).astype(int))
        small_frame_bgr = frame[max(0, y_mean - constants.WINDOW_H // 2):min(h, y_mean + constants.WINDOW_H // 2),
                          max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2),  : ]
        small_person_and_blue_mask= person_and_blue_mask[
                                     max(0, y_mean - constants.WINDOW_H  // 2):min(h, y_mean + constants.WINDOW_H // 2),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        upper_mask= small_person_and_blue_mask.copy()
        
        ## NEED TO ADD SMALLER WINDOW
        upper_mask[h//3:, : ]=0
        small_person_and_blue_mask_upper_idx = np.where(upper_mask == 1)
        
        small_fg_prob_stacked_upper = np.fromiter(map(lambda elem:utilis.check_if_in_dic(fg_pdf_memo,elem,fg_pdf),map(tuple,small_frame_bgr[small_person_and_blue_mask_upper_idx])),
        dtype= float)
        small_bg_prob_stacked_upper = np.fromiter(map(lambda elem:utilis.check_if_in_dic(bg_pdf_memo,elem,bg_pdf),map(tuple,small_frame_bgr[small_person_and_blue_mask_upper_idx])),
        dtype= float)
        small_probs_fg_bigger_bg_mask_upper= np.zeros(small_person_and_blue_mask.shape)

        small_fg_bigger_bg_mask_upper= (small_fg_prob_stacked_upper/(small_bg_prob_stacked_upper+small_fg_prob_stacked_upper))
        if frame_idx <50:
            small_probs_fg_bigger_bg_mask_upper[small_person_and_blue_mask_upper_idx]=(small_fg_bigger_bg_mask_upper>0.55).astype(np.uint8) # used to ne 67
        else:
            small_probs_fg_bigger_bg_mask_upper[small_person_and_blue_mask_upper_idx]=(small_fg_bigger_bg_mask_upper>0.95).astype(np.uint8) #used to 97

        #### now for the middle
        
        middle_mask= small_person_and_blue_mask.copy()
        middle_mask[:h//3,:]=0
        middle_mask[2*(h//3):,:]=0
        small_person_and_blue_mask_upper_idx = np.where(middle_mask == 1)
        
        small_fg_prob_stacked_middle = np.fromiter(map(lambda elem:utilis.check_if_in_dic(fg_middle_pdf_memo,elem,fg_middle_pdf),map(tuple,small_frame_bgr[small_person_and_blue_mask_upper_idx])),
        dtype= float)
        small_bg_prob_stacked_middle = np.fromiter(map(lambda elem:utilis.check_if_in_dic(bg_middle_pdf_memo,elem,bg_middle_pdf),map(tuple,small_frame_bgr[small_person_and_blue_mask_upper_idx])),
        dtype= float)
        small_probs_fg_bigger_bg_mask_middle= np.zeros(small_person_and_blue_mask.shape)

        small_probs_fg_bigger_bg_middle= (small_fg_prob_stacked_middle/(small_fg_prob_stacked_middle+small_bg_prob_stacked_middle))
        if frame_idx<50:
            small_probs_fg_bigger_bg_mask_middle[small_person_and_blue_mask_upper_idx]=(small_probs_fg_bigger_bg_middle>0.5).astype(np.uint8)
        else:
            small_probs_fg_bigger_bg_mask_middle[small_person_and_blue_mask_upper_idx]=(small_probs_fg_bigger_bg_middle>0.7).astype(np.uint8)
        #### from here its the lower

        lower_mask= small_person_and_blue_mask.copy()
        lower_mask[:2*(h//3),:]=0
        small_person_and_blue_mask_upper_idx = np.where(lower_mask == 1)
        
        small_fg_prob_stacked_lower = np.fromiter(map(lambda elem:utilis.check_if_in_dic(fg_shoes_pdf_memo,elem,fg_shoes_pdf),map(tuple,small_frame_bgr[small_person_and_blue_mask_upper_idx])),
        dtype= float)
        small_bg_prob_stacked_lower = np.fromiter(map(lambda elem:utilis.check_if_in_dic(bg_shoes_pdf_memo,elem,bg_shoes_pdf),map(tuple,small_frame_bgr[small_person_and_blue_mask_upper_idx])),
        dtype= float)
        small_probs_fg_bigger_bg_mask_lower= np.zeros(small_person_and_blue_mask.shape)

        small_probs_fg_bigger_bg_lower= (small_fg_prob_stacked_lower/(small_bg_prob_stacked_lower+small_fg_prob_stacked_lower))
        small_probs_fg_bigger_bg_mask_lower[small_person_and_blue_mask_upper_idx]=(small_probs_fg_bigger_bg_lower>0.92).astype(np.uint8)

        small_probs_fg_bigger_bg_mask = small_probs_fg_bigger_bg_mask_lower+small_probs_fg_bigger_bg_mask_upper+small_probs_fg_bigger_bg_mask_middle
        #small_probs_fg_bigger_bg_mask[small_person_and_blue_mask_idx]= (small_fg_prob_stacked>small_bg_prob_stacked*1.1).astype(np.uint8)
        
        ##$$$$ Now we do the same for the shoes
        #here we try somthinmg new 

        '''
        shoes_mask = person_and_blue_mask[constants.SHOES_HIGHT:min(h, y_mean + constants.WINDOW_H // 2),:]
        shoes_idx = np.where(shoes_mask == 1)
        y_mean_shoes,x_mean_shoes = (np.mean(shoes_idx[0]).astype(int),np.mean(shoes_idx[1]).astype(int))
        
       

    

        small_white_mask = np.copy(small_probs_fg_bigger_bg_mask)
        small_white_mask[:-270,:]=1
        small_prob_fg_bigger_bg_mask_idx = np.where(small_white_mask==0)
        small_shoes_probs_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_shoes_fg_prob_stacked = np.fromiter(map(lambda elem:utilis.check_if_in_dic(fg_shoes_pdf_memo,elem,fg_shoes_pdf)
            ,map(tuple,small_frame_bgr[small_prob_fg_bigger_bg_mask_idx])),dtype= float)
        small_shoes_bg_prob_stacked = np.fromiter(map(lambda elem:utilis.check_if_in_dic(bg_shoes_pdf_memo,elem,bg_shoes_pdf)
            ,map(tuple,small_frame_bgr[small_prob_fg_bigger_bg_mask_idx])), dtype= float)
        #shoes_fg_beats_shoes_bg_mask= (small_shoes_fg_prob_stacked/(small_shoes_bg_prob_stacked+small_shoes_fg_prob_stacked)).astype(np.uint8)
        shoes_fg_beats_shoes_bg_mask= (small_shoes_fg_prob_stacked/(small_shoes_bg_prob_stacked+small_shoes_fg_prob_stacked))
        shoes_fg_beats_shoes_bg_mask=(shoes_fg_beats_shoes_bg_mask>0.7).astype(np.uint8)
        small_shoes_probs_fg_bigger_bg_mask[small_prob_fg_bigger_bg_mask_idx] = shoes_fg_beats_shoes_bg_mask
        shoes_idx = np.where(small_shoes_probs_fg_bigger_bg_mask == 1)
        '''

        ### here i changed this
        shoes_idx = np.where(small_probs_fg_bigger_bg_mask_lower == 1)
        y_mean_shoes,x_mean_shoes = (np.mean(shoes_idx[0]).astype(int),np.mean(shoes_idx[1]).astype(int))
        
                
        small_or_mask = small_probs_fg_bigger_bg_mask #np.zeros(small_probs_fg_bigger_bg_mask.shape)

        #small_or_mask = small_probs_fg_bigger_bg_mask

        #%%%% we removed this look at old func for ref
        #small_or_mask[:y_mean_shoes,:]= small_probs_fg_bigger_bg_mask[:y_mean_shoes]
        #small_or_mask[y_mean_shoes:,:]= np.maximum(small_probs_fg_bigger_bg_mask[y_mean_shoes:,:],small_probs_fg_bigger_bg_mask_lower[y_mean_shoes:,:])#small_probs_fg_bigger_bg_mask[:y_mean_shoes]
        y_offset= 30
        #small_or_mask[y_mean_shoes - y_offset:, :] = cv2.morphologyEx(small_or_mask[y_mean_shoes - y_offset:, :],
        #                                                             cv2.MORPH_CLOSE, np.ones((1, 20)),iterations=3)
        kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        kernel_close =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        small_or_mask[:constants.FACE_HIGHT, :] = cv2.morphologyEx(small_or_mask[:constants.FACE_HIGHT, :],cv2.MORPH_OPEN,kernel=kernel,iterations=1).astype(np.uint8)
        #small_or_mask[:constants.FACE_HIGHT, :] = cv2.morphologyEx(small_or_mask[:constants.FACE_HIGHT, :],cv2.MORPH_CLOSE,kernel=np.ones((10,1)),iterations=2)
        small_or_mask[:constants.FACE_HIGHT, :] = cv2.morphologyEx(small_or_mask[:constants.FACE_HIGHT, :],cv2.MORPH_CLOSE,kernel=kernel_close,iterations=2)
        if y_mean_shoes>0:
            small_or_mask[y_mean_shoes - y_offset:, :] = cv2.morphologyEx(small_or_mask[y_mean_shoes - y_offset:, :],
                                                                         cv2.MORPH_CLOSE, kernel=np.ones((1,20)))
            small_or_mask[y_mean_shoes - y_offset:, :] = cv2.morphologyEx(small_or_mask[y_mean_shoes - y_offset:, :],
                                                                         cv2.MORPH_CLOSE, kernel=np.ones((20,1)))
        #if frame_idx<70:
        #    small_or_mask = cv2.morphologyEx(small_or_mask,cv2.MORPH_CLOSE, kernel=kernel_close,iterations=2)
        or_mask = np.zeros(person_and_blue_mask.shape)
        or_mask [max(0,y_mean-constants.WINDOW_H//2):min(h,y_mean+constants.WINDOW_H//2),max(0,x_mean- constants.WINDOW_W//2):min(w,x_mean+constants.WINDOW_W//2)]=small_or_mask
        or_mask_list[frame_idx]=or_mask
        pbar.update(1)

        #final proccseing

    print('final proccseing')
    final_masks_list, final_frames_list = [], []
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:]):
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
        temp_2 = utilis.scale_fig_0_to_255(final_mask)
        final_masks_list.append(utilis.scale_fig_0_to_255(final_mask))
        temp =np.max( utilis.use_mask_on_frame(frame=frame,mask=final_mask))
        final_frames_list.append(utilis.use_mask_on_frame(frame=frame,mask=final_mask))
        pbar.update(1)

        #or_mas
        
        

    utilis.write_video('Outputs\extracted_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=final_frames_list,isColor=True)
    utilis.write_video('Outputs\_binary_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=final_masks_list,isColor=False)




        


            







