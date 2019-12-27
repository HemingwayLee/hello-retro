import retro
import cv2
import sys
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

class Memory():
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        index = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]

def instantiate_memory(memory, pretrain_length, env, rows, cols, stack_size, possible_actions):
    for i in range(pretrain_length):
        if i == 0:
            state = env.reset() 
            stacked_frames = deque([np.zeros((rows, cols), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)   
            state, stacked_frames = stack_frames(stacked_frames, state, True, (rows, cols), stack_size)

        choice = random.randint(1,len(possible_actions))-1
        action = possible_actions[choice]
        next_state, reward, done, _ = env.step(action)
        
        env.render()
        
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, (rows, cols), stack_size)
        
        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True, (rows, cols), stack_size)
        else:
            memory.add((state, action, reward, next_state, done))
            state = next_state

    return memory

def preprocess(frame):
    # TODO:
    # resize
    # chop
    # rescale
    # x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
    # x_t1 = x_t1 / 255.0
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
    stacked_state, stacked_frames = stack_frames(stacked_frames, observation, True, (rows, cols), stack_size)

    model = build_model(env.action_space.n, (rows, cols, stack_size))

    if mode == 'run':
        print ("Now we load weight")
        print ("Weight load successfully")    
    else:
        print("training mode")

    print("stacked state shape:")
    print(stacked_state.shape)
    stacked_state = stacked_state.reshape(1, stacked_state.shape[0], stacked_state.shape[1], stacked_state.shape[2]) 
    print(stacked_state.shape)

    batch_size = 64
    memory = Memory(memory_size=1000000)
    memory = instantiate_memory(memory, batch_size, env, rows, cols, stack_size, possible_actions)
    print(memory.buffer)

    # t = 0
    # FRAME_PER_ACTION = 1
    # while True:
    #     loss = 0
    #     Q_sa = 0
    #     action_index = 0
    #     r_t = 0
    #     a_t = np.zeros([env.action_space.n])
    #     # Choose an action epsilon greedy
    #     if t % FRAME_PER_ACTION == 0:
    #         if random.random() <= epsilon:
    #             print("----------Random Action----------")
    #             action_index = random.randrange(env.action_space.n)
    #             a_t[action_index] = 1
    #         else:
    #             q = model.predict(s_t) # input a stack of 4 images, get the prediction
    #             max_Q = np.argmax(q)
    #             action_index = max_Q
    #             a_t[max_Q] = 1

    #     # We reduced the epsilon gradually
    #     if epsilon > FINAL_EPSILON and t > OBSERVE:
    #         epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    #     observation, reword, done, info = env.step(env.action_space.sample())
    #     env.render()
    #     if done:
    #         observation = env.reset()

    #     #run the selected action and observed next state and reward
    #     # x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
    #     # x_t1 = skimage.color.rgb2gray(x_t1_colored)
    #     # x_t1 = skimage.transform.resize(x_t1,(80,80))
    #     # x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
    #     # x_t1 = x_t1 / 255.0
    #     # x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
    #     # s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

    #     # store the transition in D
    #     D.append((s_t, action_index, r_t, s_t1, terminal))
    #     if len(D) > REPLAY_MEMORY:
    #         D.popleft()

    #     #only train if done observing
    #     if t > OBSERVE:
    #         #sample a minibatch to train on
    #         minibatch = random.sample(D, BATCH)

    #         #Now we do the experience replay
    #         state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
    #         state_t = np.concatenate(state_t)
    #         state_t1 = np.concatenate(state_t1)
    #         targets = model.predict(state_t)
    #         Q_sa = model.predict(state_t1)
    #         targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(Q_sa, axis=1)*np.invert(terminal)

    #         loss += model.train_on_batch(state_t, targets)

    #     # s_t = s_t1
    #     t = t + 1

    #     # save progress every 10000 iterations
    #     # if t % 1000 == 0:
    #     #     print("Now we save model")
    #     #     model.save_weights("model.h5", overwrite=True)
    #     #     with open("model.json", "w") as outfile:
    #     #         json.dump(model.to_json(), outfile)

    #     # print info
    #     state = ""
    #     if t <= OBSERVE:
    #         state = "observe"
    #     elif t > OBSERVE and t <= OBSERVE + EXPLORE:
    #         state = "explore"
    #     else:
    #         state = "train"

    #     print("TIMESTEP", t, "/ STATE", state, \
    #         "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
    #         "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    # env.close()
    # print("Episode finished!")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        train_network(sys.argv[1])
    else:
        print("Incorrect arguments")
