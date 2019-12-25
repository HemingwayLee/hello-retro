import retro
import cv2
import sys
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

def preprocess(frame):
    # TODO:
    # resize
    # chop
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    return gray

def stack_frames(stacked_frames, observation, is_new_episode, shape, stack_size):
    frame = preprocess(observation)
    
    if is_new_episode:
        stacked_frames = deque([np.zeros(shape, dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

def build_model(action_size, shape, learning_rate=1e-4):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(action_size))

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    
    return model

def train_network(mode):
    env = retro.make(game='Airstriker-Genesis')
    print(f"The size of our frame is: {env.observation_space}")
    print(f"The action size is : {env.action_space.n}")

    possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
    print(f"possible actions:\n{possible_actions}")

    observation = env.reset()
    rows, cols = observation.shape[0], observation.shape[1]
    
    stack_size = 4
    stacked_frames = deque([np.zeros((rows, cols), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
    stack_frames(stacked_frames, observation, True, (rows, cols), stack_size)

    model = build_model(env.action_space.n, (rows, cols, stack_size))

    # while True:
    #     observation, rew, done, info = env.step(env.action_space.sample())
    #     env.render()
    #     if done:
    #         observation = env.reset()
    # env.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        train_network(sys.argv[1])
    else:
        print("Incorrect arguments")
