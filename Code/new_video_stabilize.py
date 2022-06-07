import numpy as np
import cv2
from scipy.stats import gaussian_kde
import numpy as np
import json
from sklearn import utils
import tqdm

import utilis
import constants

# log the procces


'''
the following function recives as input unsablazed video input addres and the adress for the ouput file
and saves the stabilazed video in the output path
mostly based on :https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
'''
def stabalize_video(input_video_path,output_video_path):
    
    cap = cv2.VideoCapture(input_video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    parameters = utilis.get_video_parameters(cap)
    frames = utilis.load_video(cap)
    h,w = parameters["height"],parameters['width']
    
    
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) 
    transforms = np.zeros((n_frames, 3), np.float32) 
    #create detector
    detector = cv2.ORB_create()
    print('started trasformation calc')
    pbar = tqdm.tqdm(total=n_frames-1)
    for frame_idx, frame in enumerate(frames[1:]):
     
        
            # converting to gray-scale 
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            #extract features in 2 consecutive frames:
            prev_gray_keypoints, prev_gray_descriptor = detector.detectAndCompute(prev_gray, None)
            curr_gray_keypoints, curr_gray_descriptor = detector.detectAndCompute(curr_gray, None)
            #apply personolized bf matcherfor 3 extra points!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
            match_list = bf.match(prev_gray_descriptor, curr_gray_descriptor)
            # The matches with shorter distance are the ones we want.
            matches = sorted(match_list, key = lambda x : x.distance)
            
            #Find transformation matrix
            prev_pts = np.float32([prev_gray_keypoints[m.queryIdx].pt for m in matches])
            curr_pts =  np.float32([curr_gray_keypoints[m.trainIdx].pt for m in matches])
            warp_mat, inliers	=	cv2.estimateAffinePartial2D( prev_pts, curr_pts, method = cv2.RANSAC, ransacReprojThreshold = 3, maxIters = 2000,confidence = 0.99, refineIters = 10)	

            
            # Extract traslation https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
            dx = warp_mat[0,2]
            dy = warp_mat[1,2]
            # Extract rotation angle
            da = np.arctan2(warp_mat[1,0], warp_mat[0,0])
            # Store transformation
            transforms[frame_idx] = [dx,dy,da]
            prev_gray = curr_gray

            pbar.update(1)
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)  
    smoothed_trajectory = utilis.smooth(trajectory,wanted_radius=constants.SMOOTH_RADIUS) 

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference
    #now we build the stabalized video
    stabalize_video_frames = [frames[0]]
    print('started applaying trasformation')
    pbar = tqdm.tqdm(total=n_frames-1)
    for frame_idx, frame in enumerate(frames[:-1]):
         # Extract transformations from the new transformation array
         dx = transforms_smooth[frame_idx,0]
         dy = transforms_smooth[frame_idx,1]
         da = transforms_smooth[frame_idx,2]
         # Reconstruct transformation matrix accordingly to new values
         m = np.zeros((2,3), np.float32)
         m[0,0] = np.cos(da)
         m[0,1] = -np.sin(da)
         m[1,0] = np.sin(da)
         m[1,1] = np.cos(da)
         m[0,2] = dx
         m[1,2] = dy
         # Apply affine wrapping to the given frame
         frame_stabalized = cv2.warpAffine(frame, m, (w,h))
         #fix the borders
         frame_stabalized = utilis.fixBorder(frame_stabalized)

         stabalize_video_frames.append(frame_stabalized)
         #transforms_list[frame_idx] = transform_matrix
         pbar.update(1)
    utilis.release_video(cap)
    utilis.write_video(output_video_path,parameters=parameters,frames=stabalize_video_frames,isColor=True)
    print('Finished video stabilization')
   


       


