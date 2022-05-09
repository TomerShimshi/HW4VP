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
#BASED ON:
#https://github.com/rajan9519/Background-subtraction/blob/master/Gausian%20Mixure%20Model.py
def background_subtraction(input_video_path):

    my_logger.info('Starting background_subtraction')
    cap = cv2.VideoCapture(input_video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    parameters = utilis.get_video_parameters(cap)
    frames = utilis.load_video(cap)
    gray_frames = utilis.color_to_gray(frames=frames)
    subtracted_frames =[]
    row,col = gray_frames[0].shape
    # initialising mean,variance,omega and omega by sigma
    mean = np.zeros([3,row,col],np.float64)
    mean[1,:,:] = gray_frames[0]

    variance = np.zeros([3,row,col],np.float64)
    variance[:,:,:] = 400

    omega = np.zeros([3,row,col],np.float64)
    omega[0,:,:],omega[1,:,:],omega[2,:,:] = 0,0,1

    omega_by_sigma = np.zeros([3,row,col],np.float64)

    # initialising foreground and background
    foreground = np.zeros([row,col],np.uint8)
    background = np.zeros([row,col],np.uint8)

    #initialising T and alpha
    alpha = 0.3
    T = 0.75

    # converting data type of integers 0 and 255 to uint8 type
    a = np.uint8([255])
    b = np.uint8([0])

    print('started bg subtraction')
    #frames = frames[0:50]
    pbar = tqdm.tqdm(total=n_frames-1)
    '''
    now we try a diffrent approach
    '''
    #background = np.median(frames[0:constants.comperd_frames] ,axis=0)
    #background = background.astype(np.uint8)
    
    for frame_idx, frame in enumerate( frames[:60]):

        frame_gray = gray_frames[frame_idx]
    
        # converting data type of frame_gray so that different operation with it can be performed
        frame_gray = frame_gray.astype(np.float64)

        # Because variance becomes negative after some time because of norm_pdf function so we are converting those indices 
        # values which are near zero to some higher values according to their preferences
        variance[0][np.where(variance[0]<1)] = 10
        variance[1][np.where(variance[1]<1)] = 5
        variance[2][np.where(variance[2]<1)] = 1

        #calulating standard deviation
        sigma1 = np.sqrt(variance[0])
        sigma2 = np.sqrt(variance[1])
        sigma3 = np.sqrt(variance[2])

        # getting values for the inequality test to get indexes of fitting indexes
        compare_val_1 = cv2.absdiff(frame_gray,mean[0])
        compare_val_2 = cv2.absdiff(frame_gray,mean[1])
        compare_val_3 = cv2.absdiff(frame_gray,mean[2])

        value1 = 2.5 * sigma1
        value2 = 2.5 * sigma2
        value3 = 2.5 * sigma3

        # finding those indexes where values of T are less than most probable gaussian and those where sum of most probale
        # and medium probable is greater than T and most probable is less than T
        fore_index1 = np.where(omega[2]>T)
        fore_index2 = np.where(((omega[2]+omega[1])>T) & (omega[2]<T))

        # Finding those indices where a particular pixel values fits at least one of the gaussian
        gauss_fit_index1 = np.where(compare_val_1 <= value1)
        gauss_not_fit_index1 = np.where(compare_val_1 > value1)

        gauss_fit_index2 = np.where(compare_val_2 <= value2)
        gauss_not_fit_index2 = np.where(compare_val_2 > value2)

        gauss_fit_index3 = np.where(compare_val_3 <= value3)
        gauss_not_fit_index3 = np.where(compare_val_3 > value3)

        #finding common indices for those indices which satisfy line 70 and 80
        temp = np.zeros([row, col])
        temp[fore_index1] = 1
        temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
        index3 = np.where(temp == 2)

        # finding com
        temp = np.zeros([row,col])
        temp[fore_index2] = 1
        index = np.where((compare_val_3<=value3)|(compare_val_2<=value2))
        temp[index] = temp[index]+1
        index2 = np.where(temp==2)

        match_index = np.zeros([row,col])
        match_index[gauss_fit_index1] = 1
        match_index[gauss_fit_index2] = 1
        match_index[gauss_fit_index3] = 1
        not_match_index = np.where(match_index == 0)

        #updating variance and mean value of the matched indices of all three gaussians
        rho = alpha * utilis.norm_pdf(frame_gray[gauss_fit_index1], mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])
        constant = rho * ((frame_gray[gauss_fit_index1] - mean[0][gauss_fit_index1]) ** 2)
        mean[0][gauss_fit_index1] = (1 - rho) * mean[0][gauss_fit_index1] + rho * frame_gray[gauss_fit_index1]
        variance[0][gauss_fit_index1] = (1 - rho) * variance[0][gauss_fit_index1] + constant
        omega[0][gauss_fit_index1] = (1 - alpha) * omega[0][gauss_fit_index1] + alpha
        omega[0][gauss_not_fit_index1] = (1 - alpha) * omega[0][gauss_not_fit_index1]

        rho = alpha * utilis.norm_pdf(frame_gray[gauss_fit_index2], mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])
        constant = rho * ((frame_gray[gauss_fit_index2] - mean[1][gauss_fit_index2]) ** 2)
        mean[1][gauss_fit_index2] = (1 - rho) * mean[1][gauss_fit_index2] + rho * frame_gray[gauss_fit_index2]
        variance[1][gauss_fit_index2] = (1 - rho) * variance[1][gauss_fit_index2] + rho * constant
        omega[1][gauss_fit_index2] = (1 - alpha) * omega[1][gauss_fit_index2] + alpha
        omega[1][gauss_not_fit_index2] = (1 - alpha) * omega[1][gauss_not_fit_index2]

        rho = alpha * utilis.norm_pdf(frame_gray[gauss_fit_index3], mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])
        constant = rho * ((frame_gray[gauss_fit_index3] - mean[2][gauss_fit_index3]) ** 2)
        mean[2][gauss_fit_index3] = (1 - rho) * mean[2][gauss_fit_index3] + rho * frame_gray[gauss_fit_index3]
        variance[2][gauss_fit_index3] = (1 - rho) * variance[2][gauss_fit_index3] + constant
        omega[2][gauss_fit_index3] = (1 - alpha) * omega[2][gauss_fit_index3] + alpha
        omega[2][gauss_not_fit_index3] = (1 - alpha) * omega[2][gauss_not_fit_index3]

        # updating least probable gaussian for those pixel values which do not match any of the gaussian
        mean[0][not_match_index] = frame_gray[not_match_index]
        variance[0][not_match_index] = 200
        omega[0][not_match_index] = 0.1

        # normalise omega
        sum = np.sum(omega,axis=0)
        omega = omega/sum

        #finding omega by sigma for ordering of the gaussian
        omega_by_sigma[0] = omega[0] / sigma1
        omega_by_sigma[1] = omega[1] / sigma2
        omega_by_sigma[2] = omega[2] / sigma3

        # getting index order for sorting omega by sigma
        index = np.argsort(omega_by_sigma,axis=0)

        # from that index(line 139) sorting mean,variance and omega
        mean = np.take_along_axis(mean,index,axis=0)
        variance = np.take_along_axis(variance,index,axis=0)
        omega = np.take_along_axis(omega,index,axis=0)

        # converting data type of frame_gray so that we can use it to perform operations for displaying the image
        frame_gray = frame_gray.astype(np.uint8)

        # getting background from the index2 and index3
        background[index2] = frame_gray[index2]
        background[index3] = frame_gray[index3]
        fg = cv2.subtract(frame_gray,background)
        maskd_frame = frame.copy()
        maskd_frame[fg<=1.0]=0
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
    
    '''