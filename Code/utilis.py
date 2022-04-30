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
            frames.append(cv2.cvtColor(ret, cv2.COLOR_BGR2YUV))
        elif wanted_colors == 'bw':
            frames.append(cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY))
        else:
            frames.append(cv2.cvtColor(ret, cv2.COLOR_BGR2HSV))
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames) 


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

    