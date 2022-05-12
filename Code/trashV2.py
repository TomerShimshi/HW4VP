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
           fgmask = np.where(value>=constants.filter_Threshold,gray_frames[frame_idx],0)
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
        erode = cv2.erode(fgmask,kernel,iterations =constants.Num_Of_Iter)#forground,kernel,borderType=cv2.BORDER_REFLECT,iterations =constants.Num_Of_Iter)
        
        idx = np.where(erode>0)
        temp = [(idx[0][i],idx[1][i]) for i in range (len(idx[0])) ]

        temp= np.asarray(temp)
        if frame_idx %112==0:
            remp =1
        mean_location = temp.mean(0)
        #[point[0],point[1]]
        try:
            shuff = [[point[0],point[1]] for point in temp if ((abs(point[0] - mean_location[0])>constants.Mean_Threshold or abs(point[1] - mean_location[1]))>constants.Diff_Threshold or  ((abs(point[0] - prev_mean[0])> constants.Mean_Threshold +10 or abs(point[1] - prev_mean[1]))>constants.Diff_Threshold+30.0)) and point[1] <1080 ]#np.linalg.norm(point - mean)<100]
            shuff= np.asarray(shuff)
        except:
            

            shuff = [[point[0],point[1]] for point in temp if ((abs(point[0] - mean_location[0])>constants.Mean_Threshold or abs(point[1] - mean_location[1]))>constants.Diff_Threshold) and point[1] <1080]#np.linalg.norm(point - mean)<100]
            shuff= np.asarray(shuff)

        #temp = shuff.shape()
        if len(shuff>0):
            erode[shuff] = 0
            
        #print(idx)
        erode = cv2.dilate(erode, kernel_dial, iterations=int(constants.Num_Of_Iter*1))
        num_non_zeros = np.count_nonzero(erode,   keepdims=False)
        #if num_non_zeros< 50:
        #    erode = cv2.erode(fgmask,kernel,iterations =int(constants.Num_Of_Iter*0.75))#forground,kernel,borderType=cv2.BORDER_REFLECT,iterations =constants.Num_Of_Iter)
        #    erode = cv2.dilate(erode, kernel_dial, iterations=int(constants.Num_Of_Iter*4))
        erode[600:,:]= cv2.dilate(erode[600:,:], kernel_leg, iterations=int(constants.Num_Of_Iter*1))
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
    utilis.write_video('Outputs\extracted_{}_{}_num_of_iter{}_trshehold_{}_alpha_{}_mean_thresh{}_var_thresh{}.avi'.format(ID1,ID2,constants.Num_Of_Iter,constants.Diff_Threshold,constants.alpha,constants.Mean_Threshold,constants.Var_Threshold),parameters=parameters,frames=subtracted_frames,isColor=True)
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
    
    '''