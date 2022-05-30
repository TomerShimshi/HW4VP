from pickletools import uint8
import numpy as np
import cv2
from scipy.stats import gaussian_kde



def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object. The input video's VideoCapture.
    Returns:
        parameters: dict. A dictionary of parameters names to their values.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    parameters = {"fourcc": fourcc, "fps": fps, "height": height, "width": width}
    return parameters

def write_video(path,parameters,frames,isColor =True):
    '''
    the following method recives a frame array a path and parameters and write 
    the video to the wanted path
    '''
    out = cv2.VideoWriter(path ,cv2.VideoWriter_fourcc(*'XVID'),parameters['fps'],(parameters['width'], parameters['height']), isColor=isColor)
    for frame in frames:
        out.write(frame)
    out.release()

def load_video(cap,wanted_colors = 'bgr'):
    '''
    Get an OpenCV capture object and return all of its frames.
    Args:
        capture: cv2.VideoCapture object. The input video's VideoCapture.
    Returns:
        frames : all the frames in the capture in the wanted format

    '''
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames= []
    for i in range (n_frames):
        ret,frame = cap.read()
        if not ret:
            break
        if wanted_colors == 'bgr':
            frames.append(frame)
        elif wanted_colors == 'yuv':
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV))
        elif wanted_colors == 'gray':
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames) 

def color_to_gray(frames):
        gray_frames = []
        for frame in frames:
            temp1= cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            gray_frames.append(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY))
        return np.asarray(gray_frames)


def release_video(cap):
    '''
    recives a video captire and release it
    ''' 
    cv2.destroyAllWindows() 
    cap.release()

'''
the three next helper funcs are based on these artical https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/
'''
def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  # Define the filter
  f = np.ones(window_size)/window_size
  # Add padding to the boundaries
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  '''Fix padding manually'''
  for i in range(radius):
        curve_pad[i] = curve_pad[radius] - curve_pad[i]

  for i in range(len(curve_pad) - 1, len(curve_pad) - 1 - radius, -1):
        curve_pad[i] = curve_pad[len(curve_pad) - radius - 1] - curve_pad[i]
  # Apply convolution
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  # Remove padding
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed

def smooth(trajectory,wanted_radius=5):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(smoothed_trajectory.shape[1]):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=wanted_radius)

  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame


def choose_randome_indecis(mask,num_of_indecis, find_fg = True):
    if find_fg:
        indices = np.where(mask==1)
    else:
        indices = np.where(mask==0)
    if len(indices[0])==0:
        return np.column_stack((indices[0],indices[1]))
    idx_chosed = np.random.choice(len(indices[0]),size=num_of_indecis)
    return np.column_stack((indices[0][idx_chosed],indices[1][idx_chosed]))

def estimate_pdf (dataset_valus, bw_method):
    pdf = gaussian_kde(dataset=dataset_valus.T,bw_method=bw_method)
    return lambda x: pdf(x.T)

def matting_estimate_pdf (dataset_valus, bw_method, idx):
    wanted_dataset_valus= dataset_valus[idx[:,0],idx[:,1],:]
    #wanted_dataset_valus= dataset_valus[idx]
    pdf = gaussian_kde(dataset=wanted_dataset_valus.T,bw_method=bw_method)
    return lambda x: pdf(x.T)

def  check_if_in_dic(dic,element,func):
    if element in dic:
        return dic[element]
    dic[element] = func(np.asanyarray(element))[0]
    return dic[element]

def scale_fig_0_to_255(input_martix):
    if type(input_martix) == np.bool:
        input_martix =  np.uint8(input_martix)
    #input_martix = input_martix.astype(uint8)
    scaled = 255*(input_martix-np.min(input_martix))/np.ptp(input_martix)
    return np.uint8(scaled)


def use_mask_on_frame(frame,mask):
    masked_frame = np.copy(frame)
    masked_frame[:,:,0]= masked_frame[:,:,0]*mask
    masked_frame[:,:,1]= masked_frame[:,:,1]*mask
    masked_frame[:,:,2]= masked_frame[:,:,2]*mask
    return masked_frame

def convert_mak_to_image(mask):
    if type(mask) == np.bool:
        mask =  np.uint8(mask)
    #mask= mask.astype(uint8)
    scaled_mask = 255*(mask-np.min(mask)/np.ptp(mask))
    return np.uint8(scaled_mask)


    ### taken from https://stackoverflow.com/questions/57294609/how-to-generate-a-trimap-image

def generate_trimap(mask,eroision_iter=6,dilate_iter=8):
    
    #mask[mask==1] = 255
    d_kernel = np.ones((3,3))
    erode  = cv2.erode(mask,d_kernel,iterations=eroision_iter)
    dilate = cv2.dilate(mask,d_kernel,iterations=dilate_iter)
    unknown1 = cv2.bitwise_xor(erode,mask)
    unknown2 = cv2.bitwise_xor(dilate,mask)
    unknowns = cv2.add(unknown1,unknown2)
    unknowns[unknowns==255]=127
    trimap = cv2.add(mask,unknowns)
    # cv2.imwrite("mask.png",mask)
    # cv2.imwrite("dilate.png",dilate)
    # cv2.imwrite("tri.png",trimap)
    labels = trimap.copy()
    labels[trimap==127]=1 #unknown
    labels[trimap==255]=2 #foreground
    #cv2.imwrite(mask_path,labels)
    return labels








    