from audioop import reverse
from cmath import nan
from this import d
import numpy as np
import cv2
from scipy.stats import gaussian_kde
import numpy as np
import logging
import json
from sklearn import utils
import tqdm
import os

import utilis
import constants

ID1 = 203200480
ID2 = 320521461

my_logger = logging.getLogger('MyLogger')

def background_subtraction(input_video_path):
    my_logger.info('Starting background_subtraction')
    cap = cv2.VideoCapture(input_video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    parameters = utilis.get_video_parameters(cap)
    frames = utilis.load_video(cap)
    gray_frames = utilis.color_to_gray(frames=frames)
    (col,row) = gray_frames[0].shape[:2]
    var = np.ones((col,row),np.uint8)
    #var[:col,:row] = 50
    #var = np.var(frames[:50],axis=0).astype(np.uint8)#gray_frames[:190],axis=0).astype(np.uint8)
    mean = np.mean(gray_frames[:50],axis=0).astype(np.uint8)#np.ones((col,row),np.uint8) ##gray_frames[:190],axis=0).astype(np.uint8)
    count =0
    fgbg = cv2.createBackgroundSubtractorKNN(history=600,detectShadows=False,dist2Threshold =800.0)#cv2.createBackgroundSubtractorMOG2(history=600,varThreshold=20)#
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG2()
    subtracted_frames =[]
    #background = np.median(frames[:constants.comperd_frames] ,axis=0)
    #background = cv2.GaussianBlur(background,(5,5),sigmaX=constants.bluer_sigma).astype(np.uint8)
    print('started bg subtraction')
    os.chdir('test')
    #frames = frames[0:50]
    pbar = tqdm.tqdm(total=n_frames-1)
    '''
    now we try a diffrent approach
    '''
    #background = np.median(frames[0:constants.comperd_frames] ,axis=0)
    #background = background.astype(np.uint8)
    #for frame_idx in (n_frames-1, -1, -1):
    #    fgmask = fgbg.apply(frames[frame_idx])
    prev_mean=[0,0]
    for frame_idx, frame in enumerate( frames[:]):
        path = 'frame_num{}.jpg'.format(frame_idx)
        fgmask = fgbg.apply(frame)
        num_non_zeros = np.count_nonzero(fgmask,   keepdims=False)
        if num_non_zeros > 2003600:
           value=  cv2.absdiff(gray_frames[frame_idx],mean)
           fgmask = np.where(value>=200,gray_frames[frame_idx],0)
        num_non_zeros = np.count_nonzero(fgmask,   keepdims=False)
        kernel = np.asarray([[0,0, 0, 4, 0, 0,0],
        [0,0, 2, 3, 2, 0,0],
        [0,1, 1, 3, 1, 1,0],
        [1,1, 1, 3, 1, 1,1],
        [0,1, 1, 3, 1, 1,0],
        [0,0, 2, 3, 2, 0,0],
        [0,0, 0, 4, 0, 0,0]]).astype(np.uint8) #cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #np.ones((5,5),np.uint8)
        kernel =np.ones((11,11),np.uint8)
        kernel_dial = np.asarray([[0,0, 0, 1, 0, 0,0],
        [0,0, 0, 1, 0, 0,0],
        [0,0, 1, 1, 1, 0,0],
        [0,0, 1, 1, 1, 0,0],
        [0,0, 1, 1, 1, 0,0],
        [0,0, 1, 1, 1, 0,0],
        [0,0, 0, 1, 0, 0,0]]).astype(np.uint8)
        kernel_leg = np.asarray([[0,0, 0, 3, 0, 0,0],
        [0,0, 0, 3, 0, 0,0],
        [0,0, 0, 3, 0, 0,0],
        [0,0, 0, 3, 0, 0,0],
        [0,0, 0, 3, 0, 0,0],
        [0,0, 0, 3, 0, 0,0],
        [0,0, 0, 3, 0, 0,0]]).astype(np.uint8)
        erode = cv2.erode(fgmask,kernel,iterations =5)#forground,kernel,borderType=cv2.BORDER_REFLECT,iterations =constants.Num_Of_Iter)
        
        idx = np.where(erode>0)
        temp = [(idx[0][i],idx[1][i]) for i in range (len(idx[0])) ]

        temp= np.asarray(temp)
        if frame_idx %112==0:
            remp =1
        mean_location = temp.mean(0)
        #[point[0],point[1]]
        try:
            shuff = [[point[0],point[1]] for point in temp if ((abs(point[0] - mean_location[0])>50 or abs(point[1] - mean_location[1]))>50 or  ((abs(point[0] - prev_mean[0])> constants.Mean_Threshold +10 or abs(point[1] - prev_mean[1]))>50+30.0)) and point[1] <1080 ]#np.linalg.norm(point - mean)<100]
            shuff= np.asarray(shuff)
        except:
            

            shuff = [[point[0],point[1]] for point in temp if ((abs(point[0] - mean_location[0])>50 or abs(point[1] - mean_location[1]))>50) and point[1] <1080]#np.linalg.norm(point - mean)<100]
            shuff= np.asarray(shuff)

        #temp = shuff.shape()
        if len(shuff>0):
            erode[shuff] = 0
            
        #print(idx)
        erode = cv2.dilate(erode, kernel_dial, iterations=int(5))
        num_non_zeros = np.count_nonzero(erode,   keepdims=False)
        #if num_non_zeros< 50:
        #    erode = cv2.erode(fgmask,kernel,iterations =int(constants.Num_Of_Iter*0.75))#forground,kernel,borderType=cv2.BORDER_REFLECT,iterations =constants.Num_Of_Iter)
        #    erode = cv2.dilate(erode, kernel_dial, iterations=int(constants.Num_Of_Iter*4))
        erode[600:,:]= cv2.dilate(erode[600:,:], kernel_leg, iterations=int(5))
        num_non_zeros = np.count_nonzero(erode,   keepdims=False)
       
        maskd_frame = frame.copy()
        maskd_frame[erode == 0] = 0
        cv2.imwrite(path, maskd_frame)
        prev_mean=shuff.mean(0)
        
        #maskd_frame[fgmask< constants.Diff_Threshold] =0

        subtracted_frames.append(maskd_frame)
        #prev_frame =blured_frame#(prev_frame*0.5 +frame*0.5).astype(np.uint8)
        pbar.update(1)
    utilis.release_video(cap)
    utilis.write_video('Outputs\extracted_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=subtracted_frames,isColor=True)
    print('finished bg subtraction')


    '''
    this is optional stuff
    
    diff = cv2.subtract(frame,prev_frame)
        diff2 = cv2.subtract(prev_frame,frame)
        diff= diff+diff2
        diff[abs(diff)<constants.Diff_Threshold]=0
        gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray[np.abs(gray) < 10] = 0
        fgmask = gray.astype(np.uint8)
        fgmask[fgmask>0]=255
        #invert the mask
        fgmask_inv = cv2.bitwise_not(fgmask)
        #use the masks to extract the relevant parts from FG and BG
        fgimg = cv2.bitwise_and(frame,frame,mask = fgmask)
        bgimg = cv2.bitwise_and(bg,bg,mask = fgmask).astype(np.uint8)
        dst = cv2.add(bgimg,fgimg)
    
    '''


    ### $$$$ALL THE LEGACY IS HERE #########$$$$$

    '''
    
    fgmask = fgbg.apply(frame)
        #fgmask = fgmask.astype(np.uint8)
        #find the diff between the 2 frames
        
       
        try:
            #prev_frame= 0.5*frames[frame_idx+constants.comperd_frames] + 0.5*frames[frame_idx+constants.comperd_frames+1]
            #prev_frame += float((1/constants.comperd_frames))*frames[frame_idx+i]
            background = np.mean(frames[frame_idx:frame_idx+constants.comperd_frames] ,axis=0)
            #background = frames[frame_idx+constants.comperd_frames]
        except:
            #prev_frame= 0.5*frames[frame_idx-constants.comperd_frames] +0.5*frames[frame_idx-constants.comperd_frames-1]
            #prev_frame += (1/constants.comperd_frames)*frames[frame_idx-i]
            background = np.mean((frames[frame_idx-constants.comperd_frames:frame_idx+constants.comperd_frames]) ,axis=0)
            #background = frames[frame_idx-constants.comperd_frames]
        #prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) 
        
        
        #background = cv2.GaussianBlur(background,(5,5),sigmaX=constants.bluer_sigma).astype(np.uint8)
        blured_frame = cv2.GaussianBlur(frame,(5,5),sigmaX=constants.bluer_sigma)
        #blured_frame = cv2.cvtColor(blured_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(blured_frame,background)
        #diff2 = cv2.subtract(prev_frame,blured_frame)
        #diff= abs(diff+diff2)
        #used to be minus
        #diff[abs(diff)>constants.Diff_Threshold]=0
        gray_diff = diff #cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        maskd_frame = frame.copy()
        
        #mybe add here a foor loop to go over the pixels
        for i in range(h):
            for j in range(w):
                temp1= fgmask[i][j]< constants.filter_Threshold
                temp2= gray_diff[i][j]>constants.Diff_Threshold and gray_diff[i][j]<constants.upper_Threshold
                
                #if (temp1   ):
                #    maskd_frame[i][j]=0
                if (  temp2  ):
                    maskd_frame[i][j]=0
        
        #maskd_frame[(fgmask< constants.Diff_Threshold).any() and (diff> constants.Diff_Threshold).any()] =0
        temp1 = [fgmask< constants.filter_Threshold][0]
        temp2 = np.where(gray_diff<constants.Diff_Threshold)  
        #temp3 = np.where(temp2 == temp1,True,False)
        #temp3[fgmask< constants.filter_Threshold] = False #= np.where(temp3 == temp1,True,False)
        #maskd_frame[fgmask< constants.Diff_Threshold] =0
        maskd_frame[ temp2] =0
        #maskd_frame[ temp1] =0
        #maskd_frame[diff>constants.Diff_Threshold]
        #temp = np.sum(maskd_frame>0)
        #maskd_frame[diff<constants.Diff_Threshold] =0
        #temp = np.sum(maskd_frame>0)
        
        #cv2.imshow('maskd_frame', maskd_frame)
    
  



  ############### NEW TRASH

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
    frames_gray = utilis.load_video(cap,wanted_colors='gray')
    
    n_frames = len(frames_bgr)
    #create the backround subtractor
    fgbg = cv2.createBackgroundSubtractorKNN(history=800,detectShadows=False,dist2Threshold =90.0)
    mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
    num_iter = 5
    num_frames_cut =205
    print('started studing frames history')
    pbar = tqdm.tqdm(total=num_iter*n_frames)
    for i in range(num_iter):
        for frame_idx, frame in enumerate(frames_gray[:num_frames_cut]):
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
    fg_face_colors, bg_face_colors = None,None
    fg_shoes_colors,bg_shoes_colors = None,None
    person_and_blue_mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))
    
    print('start collect color KDE')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:num_frames_cut]):
        blue_fram,_,_ = cv2.split(frame)
        mask= mask_list[frame_idx]
        temp =np.max(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)) 
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel=kernel,iterations=6).astype(np.uint8)
        mask = cv2.medianBlur(mask,ksize=7) #cv2.bilateralFilter(mask,d=15,sigmaColor=85,sigmaSpace=85) #cv2.medianBlur(mask,ksize=7)
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        person_mask = np.zeros(mask.shape)
        cv2.drawContours(person_mask, [max(contours, key = cv2.contourArea)], -1, color=(1, 1, 1), thickness=cv2.FILLED)
        temp =np.max(person_mask)
        #person_mask = 
        blue_mask = (blue_fram<constants.BLUE_MASK_T).astype(np.uint8)
        temp =np.max(blue_mask)
        person_and_blue_mask =(blue_mask*person_mask).astype(np.uint8)
        temp =np.max(person_and_blue_mask)
        fg_indices = utilis.choose_randome_indecis(person_and_blue_mask,22,True)
        bg_indices = utilis.choose_randome_indecis(person_and_blue_mask,80,False)
        #$$$$$$$$$$$$ Mybe need to find ccolors for the shoes $$$$$$$$$$$$
        shoes_mask = person_and_blue_mask.copy()
        shoes_mask[:constants.SHOES_HIGHT,:]=0
        fg_shoes_indices = utilis.choose_randome_indecis(shoes_mask,22,True)
        bg_shoes_indices = utilis.choose_randome_indecis(shoes_mask,80,False)
        person_and_blue_mask_list[frame_idx] = person_and_blue_mask

        #$$$$$$$$$%%%%%%% NOW WE ADD FACE MASK#######$$$$$$$$%%%%%%%%
        face_mask =  person_and_blue_mask.copy()
        face_mask[constants.FACE_HIGHT:,:]=0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_OPEN, kernel,iterations=1)
        face_mask_idx = np.where(face_mask == 1)
        y_mean_face, x_mean_face = int(np.mean(face_mask_idx[0])), int(np.mean(face_mask_idx[1]))
        ### minimize the face mask window to compute the face 
        ##$$%$ Bg better
        #small_face_mask =  np.zeros((frame.shape))
        #first we copy again since the face mask has changed
        #face_mask =  person_and_blue_mask.copy()
        small_face_mask= face_mask[max(0, y_mean_face - constants.FACE_WINDOW_H // 2):min(h, y_mean_face +  constants.FACE_WINDOW_H // 2),
                                  max(0, x_mean_face - constants.FACE_WINDOW_W // 2):min(w, x_mean_face +  constants.FACE_WINDOW_W // 2)]
        fg_face_indices = utilis.choose_randome_indecis(small_face_mask,22,True)
        bg_face_indices = utilis.choose_randome_indecis(small_face_mask,80,False)
        temp =np.max(person_and_blue_mask)
        if fg_colors is None:
            fg_colors = frame[fg_indices[:,0],fg_indices[:,1]]
            bg_colors = frame[bg_indices[:,0],bg_indices[:,1]]
            fg_face_colors = frame[fg_face_indices[:,0],fg_face_indices[:,1]]
            bg_face_colors = frame[bg_face_indices[:,0],bg_face_indices[:,1]]
            fg_shoes_colors = frame[fg_shoes_indices[:,0],fg_shoes_indices[:,1]]
            bg_shoes_colors = frame[bg_shoes_indices[:,0],bg_shoes_indices[:,1]]
        
        else:
            fg_colors = np.concatenate((fg_colors, frame[fg_indices[:,0], fg_indices[:,1]]))
            bg_colors =np.concatenate((bg_colors, frame[bg_indices[:,0],bg_indices[:,1]] ))
            fg_face_colors = np.concatenate((fg_face_colors, frame[fg_face_indices[:,0], fg_face_indices[:,1]]))
            bg_face_colors =np.concatenate((bg_face_colors, frame[bg_face_indices[:,0],bg_face_indices[:,1]] ))
            fg_shoes_colors = np.concatenate((fg_shoes_colors, frame[fg_shoes_indices[:,0], fg_shoes_indices[:,1]]))
            bg_shoes_colors =np.concatenate((bg_shoes_colors, frame[bg_shoes_indices[:,0],bg_shoes_indices[:,1]] ))
        pbar.update(1)
    fg_pdf = utilis.estimate_pdf(dataset_valus= fg_colors,bw_method=constants.BW_MEDIUM)
    bg_pdf = utilis.estimate_pdf(dataset_valus= bg_colors,bw_method=constants.BW_MEDIUM)
    fg_face_pdf = utilis.estimate_pdf(dataset_valus= fg_face_colors,bw_method=constants.BW_MEDIUM)
    bg_face_pdf = utilis.estimate_pdf(dataset_valus= bg_face_colors,bw_method=constants.BW_MEDIUM)
    fg_shoes_pdf = utilis.estimate_pdf(dataset_valus= fg_shoes_colors,bw_method=constants.BW_MEDIUM)
    bg_shoes_pdf = utilis.estimate_pdf(dataset_valus= bg_shoes_colors,bw_method=constants.BW_MEDIUM)
    fg_pdf_memo, bg_pdf_memo= dict(),dict()
    fg_face_pdf_memo, bg_face_pdf_memo= dict(),dict()
    fg_shoes_pdf_memo, bg_shoes_pdf_memo= dict(),dict()

    or_mask_list = np.zeros((n_frames,parameters["height"],parameters['width']))

    #filtering using the KDE
    print('start the KDE filtering')
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:num_frames_cut]):
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
        small_probs_fg_bigger_bg_mask[small_person_and_blue_mask_idx]= (small_fg_prob_stacked>small_bg_prob_stacked*1.1).astype(np.uint8)
        
        ##$$$$ Now we do the same for the shoes
        #here we try somthinmg new 
        shoes_mask = person_and_blue_mask[constants.SHOES_HIGHT:min(h, y_mean + constants.WINDOW_H // 2),:]
        shoes_idx = np.where(shoes_mask == 1)
        y_mean_shoes,x_mean_shoes = (np.mean(shoes_idx[0]).astype(int),np.mean(shoes_idx[1]).astype(int))
        
       
        small_shoes_frame_bgr = frame[max(0, y_mean_shoes - constants.WINDOW_H  // 2):min(h,  y_mean + constants.WINDOW_H // 2 ),
                                     max(0, x_mean_shoes - constants.WINDOW_W // 2):min(w, x_mean_shoes + constants.WINDOW_W // 2)]
        small_shoes_mask = person_and_blue_mask[
                                     max(0, y_mean_shoes - constants.WINDOW_H  // 2):min(h,  y_mean + constants.WINDOW_H // 2 ),
                                     max(0, x_mean - constants.WINDOW_W // 2):min(w, x_mean + constants.WINDOW_W // 2)]
        small_shoes_mask_idx = np.where(small_shoes_mask == 1)
        
        small_shoes_frame_bgr = small_frame_bgr[y_mean_shoes:,:]
        small_shoes_mask = person_and_blue_mask[y_mean_shoes:,:]
        small_shoes_mask_idx = np.where(small_shoes_mask == 0)

        ###### DELETE THE ABOVE

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
        shoes_fg_beats_shoes_bg_mask=(shoes_fg_beats_shoes_bg_mask>0.65).astype(np.uint8)
        small_shoes_probs_fg_bigger_bg_mask[small_prob_fg_bigger_bg_mask_idx] = shoes_fg_beats_shoes_bg_mask
        shoes_idx = np.where(small_shoes_probs_fg_bigger_bg_mask == 1)
        y_mean_shoes,x_mean_shoes = (np.mean(shoes_idx[0]).astype(int),np.mean(shoes_idx[1]).astype(int))


        ### NOW WE USE THE FACE KDE #$$$$$$$$$$$%%%%%%%
        small_white_mask_face = np.copy(small_probs_fg_bigger_bg_mask)
        small_white_mask_face[constants.FACE_HIGHT:,:]=0
        small_prob_fg_bigger_bg_mask_face_idx = np.where(small_white_mask_face==1)
        small_face_probs_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_face_fg_prob_stacked = np.fromiter(map(lambda elem:utilis.check_if_in_dic(fg_face_pdf_memo,elem,fg_face_pdf)
            ,map(tuple,small_frame_bgr[small_prob_fg_bigger_bg_mask_face_idx])),dtype= float)
        small_face_bg_prob_stacked = np.fromiter(map(lambda elem:utilis.check_if_in_dic(bg_face_pdf_memo,elem,bg_face_pdf)
            ,map(tuple,small_frame_bgr[small_prob_fg_bigger_bg_mask_face_idx])), dtype= float)
        #shoes_fg_beats_shoes_bg_mask= (small_shoes_fg_prob_stacked/(small_shoes_bg_prob_stacked+small_shoes_fg_prob_stacked)).astype(np.uint8)
        #face_fg_beats_face_bg_mask= (small_face_fg_prob_stacked/(small_face_bg_prob_stacked+small_face_fg_prob_stacked))
        face_fg_beats_face_bg_mask=(small_face_fg_prob_stacked>small_face_bg_prob_stacked).astype(np.uint8)
        small_face_probs_fg_bigger_bg_mask[small_prob_fg_bigger_bg_mask_face_idx] = face_fg_beats_face_bg_mask
        face_idx = np.where(small_face_probs_fg_bigger_bg_mask == 1)
        y_mean_face,x_mean_shoes = (np.mean(face_idx[0]).astype(int),np.mean(face_idx[1]).astype(int))


        
        small_shoes_fg_prob_stacked = np.fromiter(map(lambda elem:utilis.check_if_in_dic(fg_shoes_pdf_memo,elem,fg_shoes_pdf),map(tuple,small_frame_bgr[small_shoes_mask_idx])),
        dtype= float)
        small_shoes_bg_prob_stacked = np.fromiter(map(lambda elem:utilis.check_if_in_dic(bg_shoes_pdf_memo,elem,bg_shoes_pdf),map(tuple,small_frame_bgr[small_shoes_mask_idx])),
        dtype= float)
        #temp =np.asarray(small_shoes_mask_idx[0])
        #small_shoes_mask_idx[0] = np.clip(small_shoes_mask_idx, 0, small_person_and_blue_mask.shape[0])#, out=a)
        small_shoes_probs_fg_bigger_bg_mask= np.zeros(small_person_and_blue_mask.shape)
        small_shoes_probs_fg_bigger_bg_mask[small_shoes_mask_idx]= (small_shoes_fg_prob_stacked>small_shoes_bg_prob_stacked).astype(np.uint8)
        

        
        small_or_mask = np.zeros(small_probs_fg_bigger_bg_mask.shape)

        #small_or_mask = small_probs_fg_bigger_bg_mask
        small_or_mask[:constants.FACE_HIGHT,:]=np.minimum(small_face_probs_fg_bigger_bg_mask[:constants.FACE_HIGHT,:],small_probs_fg_bigger_bg_mask[:constants.FACE_HIGHT,:] )
        small_or_mask[constants.FACE_HIGHT:y_mean_shoes,:]=np.maximum(small_face_probs_fg_bigger_bg_mask[constants.FACE_HIGHT:y_mean_shoes,:],
                                                   small_probs_fg_bigger_bg_mask[constants.FACE_HIGHT:y_mean_shoes,:] )
        #small_or_mask[:y_mean_shoes,:]= small_probs_fg_bigger_bg_mask[:y_mean_shoes]
        small_or_mask[y_mean_shoes:,:]= np.maximum(small_probs_fg_bigger_bg_mask[y_mean_shoes:,:],small_shoes_probs_fg_bigger_bg_mask[y_mean_shoes:,:])#small_probs_fg_bigger_bg_mask[:y_mean_shoes]
        y_offset= 30
        #small_or_mask[y_mean_shoes - y_offset:, :] = cv2.morphologyEx(small_or_mask[y_mean_shoes - y_offset:, :],
        #                                                             cv2.MORPH_CLOSE, np.ones((1, 20)),iterations=3)
        kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        small_or_mask[y_mean_shoes - y_offset:, :] = cv2.morphologyEx(small_or_mask[y_mean_shoes - y_offset:, :],
                                                                     cv2.MORPH_CLOSE, kernel=np.ones((1,20)))
        small_or_mask[:y_mean_face + y_offset, :] = cv2.morphologyEx(small_or_mask[:y_mean_face + y_offset, :],
                                                                     cv2.MORPH_CLOSE, kernel=np.ones((1,20)),iterations=3)
        small_or_mask[:y_mean_face + y_offset, :] = cv2.morphologyEx(small_or_mask[:y_mean_face + y_offset, :],
                                                                     cv2.MORPH_CLOSE, kernel=np.ones((20,1)),iterations=3 )
        small_or_mask = cv2.morphologyEx(small_or_mask,cv2.MORPH_CLOSE, kernel=kernel,iterations=2)
        or_mask = np.zeros(person_and_blue_mask.shape)
        or_mask [max(0,y_mean-constants.WINDOW_H//2):min(h,y_mean+constants.WINDOW_H//2),max(0,x_mean- constants.WINDOW_W//2):min(w,x_mean+constants.WINDOW_W//2)]=small_or_mask
        or_mask_list[frame_idx]=or_mask
        pbar.update(1)

        #final proccseing

    print('final proccseing')
    final_masks_list, final_frames_list = [], []
    pbar = tqdm.tqdm(total=n_frames)
    for frame_idx, frame in enumerate(frames_bgr[:num_frames_cut]):
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




        


            








'''