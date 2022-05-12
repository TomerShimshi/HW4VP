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


    '''
        alpha= constants.alpha

        new_mean = (1-alpha)*mean + gray_frames[frame_idx] #alpha*frame#gray_frames[frame_idx]       
        new_mean = new_mean.astype(np.uint8)
        
        new_var = (alpha)*(cv2.subtract(gray_frames[frame_idx],mean)**2) + (1-alpha)*(var)#frame,mean)**2) + #(1-alpha)*(var)#gray_frames[frame_idx],mean)**2) + (1-alpha)*(var)

        value  = cv2.absdiff(gray_frames[frame_idx],mean)#frame,mean)#gray_frames[frame_idx],mean)
        value = value /np.sqrt(var)
        
       
        
        mean = np.where(value < constants.Mean_Threshold,new_mean,mean)
        var = np.where(value < constants.Var_Threshold,new_var,var)
        a = np.uint8([255])
        b = np.uint8([0])
        Threshold= constants.Diff_Threshold
        if frame_idx <= n_frames//3:
            Threshold-=2
        else:
           Threshold+=2 
        
        #value = value[:,:,0]+value[:,:,1]+value[:,:,2]
        background =np.where(value < Threshold,gray_frames[frame_idx],0)
        forground = np.where(value>=Threshold,gray_frames[frame_idx],b)
        #cv2.imshow('background',background)  
           
        kernel = np.asarray([[0, 0, 1, 0, 0],
       [0,0, 1, 1, 1, 0,0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0]]).astype(np.uint8) #cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #np.ones((5,5),np.uint8)
    ''' 

import logging
import cv2
import numpy as np

from constants import (
    BW_MEDIUM,
    SHOES_HEIGHT,
    SHOULDERS_HEIGHT,
    LEGS_HEIGHT,
    BW_NARROW,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    BLUE_MASK_THR,
    FACE_WINDOW_HEIGHT,
    FACE_WINDOW_WIDTH
)
from utils import (
    get_video_files,
    load_entire_video,
    apply_mask_on_color_frame,
    write_video,
    release_video_files,
    disk_kernel,
    choose_indices_for_foreground,
    choose_indices_for_background,
    new_estimate_pdf,
    check_in_dict, scale_matrix_0_to_255
)

my_logger = logging.getLogger('MyLogger')


def background_subtraction(input_video_path):
    my_logger.info('Starting Background Subtraction')
    # Read input video
    cap, w, h, fps = get_video_files(path=input_video_path)
    # Get frame count
    frames_bgr = load_entire_video(cap, color_space='bgr')
    frames_hsv = load_entire_video(cap, color_space='hsv')
    n_frames = len(frames_bgr)

    backSub = cv2.createBackgroundSubtractorKNN()
    mask_list = np.zeros((n_frames, h, w)).astype(np.uint8)
    print(f"[BS] - BackgroundSubtractorKNN Studying Frames history")
    for j in range(8):
        print(f"[BS] - BackgroundSubtractorKNN {j + 1} / 8 pass")
        for index_frame, frame in enumerate(frames_hsv):
            frame_sv = frame[:, :, 1:]
            fgMask = backSub.apply(frame_sv)
            fgMask = (fgMask > 200).astype(np.uint8)
            mask_list[index_frame] = fgMask
    print(f"[BS] - BackgroundSubtractorKNN Finished")

    omega_f_colors, omega_b_colors = None, None
    omega_f_shoes_colors, omega_b_shoes_colors = None, None
    person_and_blue_mask_list = np.zeros((n_frames, h, w))
    '''Collecting colors for building body & shoes KDEs'''
    for frame_index, frame in enumerate(frames_bgr):
        print(f"[BS] - Collecting colors for building body & shoes KDEs , Frame: {frame_index + 1} / {n_frames}")
        blue_frame, _, _ = cv2.split(frame)
        mask_for_frame = mask_list[frame_index].astype(np.uint8)
        mask_for_frame = cv2.morphologyEx(mask_for_frame, cv2.MORPH_CLOSE, disk_kernel(6))
        mask_for_frame = cv2.medianBlur(mask_for_frame, 7)
        _, contours, _ = cv2.findContours(mask_for_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        person_mask = np.zeros(mask_for_frame.shape)
        cv2.fillPoly(person_mask, pts=[contours[0]], color=1)
        blue_mask = (blue_frame < BLUE_MASK_THR).astype(np.uint8)
        person_and_blue_mask = (person_mask * blue_mask).astype(np.uint8)
        omega_f_indices = choose_indices_for_foreground(person_and_blue_mask, 20)
        omega_b_indices = choose_indices_for_background(person_and_blue_mask, 20)
        '''Collect colors for shoes'''
        shoes_mask = np.copy(person_and_blue_mask)
        shoes_mask[:SHOES_HEIGHT, :] = 0
        omega_f_shoes_indices = choose_indices_for_foreground(shoes_mask, 20)
        shoes_mask = np.copy(person_and_blue_mask)
        shoes_mask[:SHOES_HEIGHT - 120, :] = 1
        omega_b_shoes_indices = choose_indices_for_background(shoes_mask, 20)
        person_and_blue_mask_list[frame_index] = person_and_blue_mask
        if omega_f_colors is None:
            omega_f_colors = frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :]
            omega_b_colors = frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :]
            omega_f_shoes_colors = frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :]
            omega_b_shoes_colors = frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :]
        else:
            omega_f_colors = np.concatenate((omega_f_colors, frame[omega_f_indices[:, 0], omega_f_indices[:, 1], :]))
            omega_b_colors = np.concatenate((omega_b_colors, frame[omega_b_indices[:, 0], omega_b_indices[:, 1], :]))
            omega_f_shoes_colors = np.concatenate(
                (omega_f_shoes_colors, frame[omega_f_shoes_indices[:, 0], omega_f_shoes_indices[:, 1], :]))
            omega_b_shoes_colors = np.concatenate(
                (omega_b_shoes_colors, frame[omega_b_shoes_indices[:, 0], omega_b_shoes_indices[:, 1], :]))

    foreground_pdf = new_estimate_pdf(omega_values=omega_f_colors, bw_method=BW_MEDIUM)
    background_pdf = new_estimate_pdf(omega_values=omega_b_colors, bw_method=BW_MEDIUM)
    foreground_shoes_pdf = new_estimate_pdf(omega_values=omega_f_shoes_colors, bw_method=BW_MEDIUM)
    background_shoes_pdf = new_estimate_pdf(omega_values=omega_b_shoes_colors, bw_method=BW_MEDIUM)

    foreground_pdf_memoization, background_pdf_memoization = dict(), dict()
    foreground_shoes_pdf_memoization, background_shoes_pdf_memoization = dict(), dict()
    or_mask_list = np.zeros((n_frames, h, w))
    '''Filtering with KDEs general body parts & shoes'''
    for frame_index, frame in enumerate(frames_bgr):
        print(f"[BS] - Filtering with KDEs general body parts & shoes , Frame: {frame_index + 1} / {n_frames}")
        person_and_blue_mask = person_and_blue_mask_list[frame_index]
        person_and_blue_mask_indices = np.where(person_and_blue_mask == 1)
        y_mean, x_mean = int(np.mean(person_and_blue_mask_indices[0])), int(np.mean(person_and_blue_mask_indices[1]))
        small_frame_bgr = frame[max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
                          max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2),
                          :]
        small_person_and_blue_mask = person_and_blue_mask[
                                     max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
                                     max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2)]

        small_person_and_blue_mask_indices = np.where(small_person_and_blue_mask == 1)
        small_probs_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(foreground_pdf_memoization, elem, foreground_pdf),
                map(tuple, small_frame_bgr[small_person_and_blue_mask_indices])), dtype=float)
        small_background_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(background_pdf_memoization, elem, background_pdf),
                map(tuple, small_frame_bgr[small_person_and_blue_mask_indices])), dtype=float)
        small_probs_fg_bigger_bg_mask[small_person_and_blue_mask_indices] = (
                small_foreground_probabilities_stacked > small_background_probabilities_stacked).astype(np.uint8)

        '''Shoes restoration'''
        smaller_upper_white_mask = np.copy(small_probs_fg_bigger_bg_mask)
        smaller_upper_white_mask[:-270, :] = 1
        small_probs_fg_bigger_bg_mask_black_indices = np.where(smaller_upper_white_mask == 0)
        small_probs_shoes_fg_bigger_bg_mask = np.zeros(small_person_and_blue_mask.shape)
        small_shoes_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(foreground_shoes_pdf_memoization, elem, foreground_shoes_pdf),
                map(tuple, small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices])), dtype=float)

        small_shoes_background_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(background_shoes_pdf_memoization, elem, background_shoes_pdf),
                map(tuple, small_frame_bgr[small_probs_fg_bigger_bg_mask_black_indices])), dtype=float)

        shoes_fg_shoes_bg_ratio = small_shoes_foreground_probabilities_stacked / (
                small_shoes_foreground_probabilities_stacked + small_shoes_background_probabilities_stacked)
        shoes_fg_beats_shoes_bg_mask = (shoes_fg_shoes_bg_ratio > 0.75).astype(np.uint8)
        small_probs_shoes_fg_bigger_bg_mask[small_probs_fg_bigger_bg_mask_black_indices] = shoes_fg_beats_shoes_bg_mask
        small_probs_shoes_fg_bigger_bg_mask_indices = np.where(small_probs_shoes_fg_bigger_bg_mask == 1)
        y_shoes_mean, x_shoes_mean = int(np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[0])), int(
            np.mean(small_probs_shoes_fg_bigger_bg_mask_indices[1]))
        small_or_mask = np.zeros(small_probs_fg_bigger_bg_mask.shape)
        small_or_mask[:y_shoes_mean, :] = small_probs_fg_bigger_bg_mask[:y_shoes_mean, :]
        small_or_mask[y_shoes_mean:, :] = np.maximum(small_probs_fg_bigger_bg_mask[y_shoes_mean:, :],
                                                     small_probs_shoes_fg_bigger_bg_mask[y_shoes_mean:, :]).astype(
            np.uint8)

        DELTA_Y = 30
        small_or_mask[y_shoes_mean - DELTA_Y:, :] = cv2.morphologyEx(small_or_mask[y_shoes_mean - DELTA_Y:, :],
                                                                     cv2.MORPH_CLOSE, np.ones((1, 20)))
        small_or_mask[y_shoes_mean - DELTA_Y:, :] = cv2.morphologyEx(small_or_mask[y_shoes_mean - DELTA_Y:, :],
                                                                     cv2.MORPH_CLOSE, disk_kernel(20))

        or_mask = np.zeros(person_and_blue_mask.shape)
        or_mask[max(0, y_mean - WINDOW_HEIGHT // 2):min(h, y_mean + WINDOW_HEIGHT // 2),
        max(0, x_mean - WINDOW_WIDTH // 2):min(w, x_mean + WINDOW_WIDTH // 2)] = small_or_mask
        or_mask_list[frame_index] = or_mask

    omega_f_face_colors, omega_b_face_colors = None, None
    '''Collecting colors for building face KDE'''
    for frame_index, frame in enumerate(frames_bgr):
        print(f"[BS] - Collecting colors for building face KDE , Frame: {frame_index + 1} / {n_frames}")
        or_mask = or_mask_list[frame_index]
        face_mask = np.copy(or_mask)
        face_mask[SHOULDERS_HEIGHT:, :] = 0
        face_mask_indices = np.where(face_mask == 1)
        y_mean, x_mean = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))

        small_frame_bgr = frame[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2),
                          :]
        face_mask = np.copy(or_mask)  # Restore face mask again
        small_face_mask = face_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)]

        small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, np.ones((20, 1), np.uint8))
        small_face_mask = cv2.morphologyEx(small_face_mask, cv2.MORPH_OPEN, np.ones((1, 20), np.uint8))

        omega_f_face_indices = choose_indices_for_foreground(small_face_mask, 20)
        omega_b_face_indices = choose_indices_for_background(small_face_mask, 20)
        if omega_f_face_colors is None:
            omega_f_face_colors = small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :]
            omega_b_face_colors = small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :]
        else:
            omega_f_face_colors = np.concatenate(
                (omega_f_face_colors, small_frame_bgr[omega_f_face_indices[:, 0], omega_f_face_indices[:, 1], :]))
            omega_b_face_colors = np.concatenate(
                (omega_b_face_colors, small_frame_bgr[omega_b_face_indices[:, 0], omega_b_face_indices[:, 1], :]))

    foreground_face_pdf = new_estimate_pdf(omega_values=omega_f_face_colors, bw_method=BW_NARROW)
    background_face_pdf = new_estimate_pdf(omega_values=omega_b_face_colors, bw_method=BW_NARROW)
    foreground_face_pdf_memoization, background_face_pdf_memoization = dict(), dict()
    final_masks_list, final_frames_list = [], []
    '''Final Processing of BS (applying face KDE)'''
    for frame_index, frame in enumerate(frames_bgr):
        print(f"[BS] - Final Processing of BS (applying face KDE) , Frame: {frame_index + 1} / {n_frames}")
        or_mask = or_mask_list[frame_index]
        face_mask = np.copy(or_mask)
        face_mask[SHOULDERS_HEIGHT:, :] = 0
        face_mask_indices = np.where(face_mask == 1)
        y_mean, x_mean = int(np.mean(face_mask_indices[0])), int(np.mean(face_mask_indices[1]))

        small_frame_bgr = frame[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2),
                          :]
        face_mask = np.copy(or_mask)  # Restore face mask again
        small_face_mask = face_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
                          max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)]

        small_frame_bgr_stacked = small_frame_bgr.reshape((-1, 3))

        small_face_foreground_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(foreground_face_pdf_memoization, elem, foreground_face_pdf),
                map(tuple, small_frame_bgr_stacked)), dtype=float)
        small_face_background_probabilities_stacked = np.fromiter(
            map(lambda elem: check_in_dict(background_face_pdf_memoization, elem, background_face_pdf),
                map(tuple, small_frame_bgr_stacked)), dtype=float)

        small_face_foreground_probabilities = small_face_foreground_probabilities_stacked.reshape(small_face_mask.shape)
        small_face_background_probabilities = small_face_background_probabilities_stacked.reshape(small_face_mask.shape)
        small_probs_face_fg_bigger_face_bg_mask = (
                small_face_foreground_probabilities > small_face_background_probabilities).astype(np.uint8)
        small_probs_face_fg_bigger_face_bg_mask_laplacian = cv2.Laplacian(small_probs_face_fg_bigger_face_bg_mask,
                                                                          cv2.CV_32F)
        small_probs_face_fg_bigger_face_bg_mask_laplacian = np.abs(small_probs_face_fg_bigger_face_bg_mask_laplacian)
        small_probs_face_fg_bigger_face_bg_mask = np.maximum(
            small_probs_face_fg_bigger_face_bg_mask - small_probs_face_fg_bigger_face_bg_mask_laplacian, 0)
        small_probs_face_fg_bigger_face_bg_mask[np.where(small_probs_face_fg_bigger_face_bg_mask > 1)] = 0
        small_probs_face_fg_bigger_face_bg_mask = small_probs_face_fg_bigger_face_bg_mask.astype(np.uint8)

        _, contours, _ = cv2.findContours(small_probs_face_fg_bigger_face_bg_mask, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        small_contour_mask = np.zeros(small_probs_face_fg_bigger_face_bg_mask.shape, dtype=np.uint8)
        cv2.fillPoly(small_contour_mask, pts=[contours[0]], color=1)

        small_contour_mask = cv2.morphologyEx(small_contour_mask, cv2.MORPH_CLOSE, disk_kernel(12))
        small_contour_mask = cv2.dilate(small_contour_mask, disk_kernel(3), iterations=1).astype(np.uint8)
        small_contour_mask[-50:, :] = small_face_mask[-50:, :]

        final_mask = np.copy(or_mask).astype(np.uint8)
        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):min(h, y_mean + FACE_WINDOW_HEIGHT // 2),
        max(0, x_mean - FACE_WINDOW_WIDTH // 2):min(w, x_mean + FACE_WINDOW_WIDTH // 2)] = small_contour_mask

        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :] = cv2.morphologyEx(
            final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :], cv2.MORPH_OPEN,
            np.ones((6, 1), np.uint8))
        final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :] = cv2.morphologyEx(
            final_mask[max(0, y_mean - FACE_WINDOW_HEIGHT // 2):LEGS_HEIGHT, :], cv2.MORPH_OPEN,
            np.ones((1, 6), np.uint8))

        _, contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        final_contour_mask = np.zeros(final_mask.shape)
        cv2.fillPoly(final_contour_mask, pts=[contours[0]], color=1)
        final_mask = (final_contour_mask * final_mask).astype(np.uint8)
        final_masks_list.append(scale_matrix_0_to_255(final_mask))
        final_frames_list.append(apply_mask_on_color_frame(frame=frame, mask=final_mask))

    write_video(output_path='../Outputs/extracted.avi', frames=final_frames_list, fps=fps, out_size=(w, h), is_color=True)
    write_video(output_path='../Outputs/binary.avi', frames=final_masks_list, fps=fps, out_size=(w, h), is_color=False)
    print('~~~~~~~~~~~ [BS] FINISHED! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ binary.avi has been created! ~~~~~~~~~~~')
    print('~~~~~~~~~~~ extracted.avi has been created! ~~~~~~~~~~~')
    my_logger.info('Finished Background Subtraction')

    release_video_files(cap)


