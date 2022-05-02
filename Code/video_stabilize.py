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

# log the procces
my_logger = logging.getLogger('MyLogger')

'''
the following function recives as input unsablazed video input addres and the adress for the ouput file
and saves the stabilazed video in the output path
mostly based on :https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
'''
def stabalize_video(input_video_path,output_video_path):
    my_logger.info('Starting Video Stabilization')
    cap = cv2.VideoCapture(input_video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    parameters = utilis.get_video_parameters(cap)
    frames = utilis.load_video(cap)
    
    
    
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) 
    transforms = np.zeros((n_frames - 1, 9), np.float32)
    transforms_list = np.zeros((n_frames - 1, 3, 3), np.float32)
    print('started trasformation calc')
    pbar = tqdm.tqdm(total=n_frames-1)
    for frame_idx, frame in enumerate(frames[1:]):
     
        
            # converting to gray-scale 
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            prev_pts = cv2.goodFeaturesToTrack(prev_gray
                                                , maxCorners=constants.MAX_CORNERS
                                                ,qualityLevel=constants.QUALITY_LEVEL
                                                , minDistance=constants.MIN_DISTANCE
                                                ,blockSize=constants.BLOCK_SIZE)

            # Calculate optical flow (i.e. track feature points)

            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            # Sanity check

            assert prev_pts.shape == curr_pts.shape
            
            # Filter only valid points
            
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            
            # Find transformation matrix
            transform_matrix, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 10.0)

            transforms[frame_idx]= transform_matrix.flatten()
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
         transform_matrix = transforms_smooth[frame_idx].reshape((3, 3))
         #warp the image
         frame_stabalized = cv2.warpPerspective(frame,transform_matrix,(parameters['width'], parameters['height']))
         #fix the borders
         frame_stabalized = utilis.fixBorder(frame_stabalized)

         stabalize_video_frames.append(frame_stabalized)
         #transforms_list[frame_idx] = transform_matrix
         pbar.update(1)
    utilis.release_video(cap)
    utilis.write_video(output_video_path,parameters=parameters,frames=stabalize_video_frames,isColor=True)
    print('Finished video stabilization')
    my_logger.info('Finished Video Stabilization')


       


