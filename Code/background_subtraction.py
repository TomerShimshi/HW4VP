from this import d
import numpy as np
import cv2
from scipy.stats import gaussian_kde
import numpy as np
import logging
import json
from sklearn import utils
import tqdm

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
    h, w = frames[0].shape[0],frames[0].shape[1]
    fgbg = cv2.createBackgroundSubtractorKNN(history=200,detectShadows=False,dist2Threshold =400)
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG2()
    subtracted_frames =[]
    #prev_frame= frames[0]
    #prev_frame= frames[constants.comperd_frames]
    #prev_frame = cv2.GaussianBlur(prev_frame,(5,5),sigmaX=3)
    print('started bg subtraction')
    pbar = tqdm.tqdm(total=n_frames-1)
    bg=255*np.ones((frames[0].shape))
    for frame_idx, frame in enumerate( frames[:]):
        fgmask = fgbg.apply(frame)
        #fgmask = fgmask.astype(np.uint8)
        #find the diff between the 2 frames
        try:
            prev_frame= frames[frame_idx+constants.comperd_frames]
        except:
            prev_frame= frames[frame_idx-constants.comperd_frames]
        prev_frame = cv2.GaussianBlur(prev_frame,(5,5),sigmaX=0)
        blured_frame = cv2.GaussianBlur(frame,(5,5),sigmaX=0)
        diff = cv2.subtract(blured_frame,prev_frame)
        diff2 = cv2.subtract(prev_frame,blured_frame)
        diff= abs(diff+diff2)
        diff[abs(diff)<constants.Diff_Threshold]=0
        gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        maskd_frame = frame.copy()

        #mybe add here a foor loop to go over the pixels
        #for i in range(h):
        #    for j in range(w):
        #        temp1= fgmask[i][j]
        #        temp2= diff[i][j]>constants.Diff_Threshold
        #        temp2= True in( temp2)
        #        if (fgmask[i][j] <constants.Diff_Threshold and  temp2  ):
        #            maskd_frame[i][j]=0

        #maskd_frame[(fgmask< constants.Diff_Threshold).any() and (diff> constants.Diff_Threshold).any()] =0
        maskd_frame[fgmask< constants.Diff_Threshold] =0
        #temp = np.sum(maskd_frame>0)
        maskd_frame[diff> 0] =0
        #temp = np.sum(maskd_frame>0)
        
        #cv2.imshow('maskd_frame', maskd_frame)
        subtracted_frames.append(maskd_frame)
        prev_frame =blured_frame#(prev_frame*0.5 +frame*0.5).astype(np.uint8)
        pbar.update(1)
    utilis.release_video(cap)
    utilis.write_video('Outputs\extracted_{}_{}.avi'.format(ID1,ID2),parameters=parameters,frames=subtracted_frames,isColor=True)


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