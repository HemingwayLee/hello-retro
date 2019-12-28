import cv2
import numpy as np
from config import *
from collections import deque

def preprocess(frame):
    # TODO:
    # chop
    # rescale
    # x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
    # x_t1 = x_t1 / 255.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (int(gray.shape[1]/RESIZE_FACTOR), int(gray.shape[0]/RESIZE_FACTOR)), interpolation=cv2.INTER_AREA)
    # print(gray.shape)

    return resized

def stack_frames(stacked_frames, observation, is_new_episode, shape, STACK_SIZE):
    frame = preprocess(observation)
    
    if is_new_episode:
        stacked_frames = deque([np.zeros(shape, dtype=np.int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=2) 
    return stacked_state, stacked_frames
