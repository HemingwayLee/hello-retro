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

def instantiate_memory(pretrain_length, env, rows, cols, stack_size, possible_actions):
    memory = Memory(memory_size=1000000)
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
            next_state = np.zeros((rows, cols, 3), dtype=np.uint8) # data type need to be correct
            memory.add((state, action, reward, next_state, done))
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True, (rows, cols), stack_size)
        else:
            # print("not done")
            memory.add((state, action, reward, next_state, done))
            state = next_state

    return memory

def preprocess(frame):
    # TODO:
    # chop
    # rescale
    # x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
    # x_t1 = x_t1 / 255.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (int(gray.shape[1]/4), int(gray.shape[0]/4)), interpolation=cv2.INTER_AREA)
    # print(gray.shape)

    return resized

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
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(action_size))

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    
    return model

def predict_action(model, explore_start, explore_stop, decay_rate, decay_step, state, possible_actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        choice = random.randint(1,len(possible_actions))-1
        action = possible_actions[choice]
    else:
        # Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        # Take the biggest Q value (= the best action)
        Qs = model.predict(state.reshape(1, state.shape[0], state.shape[1], state.shape[2]))
        choice = np.argmax(Qs)
        action = possible_actions[choice]
       
    return action, explore_probability

def train_network(mode):
    env = retro.make(game='Airstriker-Genesis')
    print(f"The size of our frame is: {env.observation_space}")
    print(f"The action size is : {env.action_space.n}")

    possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
    print(f"possible actions:\n{possible_actions}")

    observation = env.reset()
    rows, cols = int(observation.shape[0]/4), int(observation.shape[1]/4)
    print(rows, cols)
    
    stack_size = 4
    stacked_frames = deque([np.zeros((rows, cols), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
    # stacked_state, stacked_frames = stack_frames(stacked_frames, observation, True, (rows, cols), stack_size)
    _, stacked_frames = stack_frames(stacked_frames, observation, True, (rows, cols), stack_size)

    model = build_model(env.action_space.n, (rows, cols, stack_size))

    # if mode == 'run':
    #     print ("Now we load weight")
    #     print ("Weight load successfully")    
    # else:
    #     print("training mode")

    # print("stacked state shape:")
    # print(stacked_state.shape)
    # stacked_state = stacked_state.reshape(1, stacked_state.shape[0], stacked_state.shape[1], stacked_state.shape[2]) 
    # print(stacked_state.shape)

    explore_start = 1.0
    explore_stop = 0.01 
    decay_rate = 0.00001

    gamma = 0.9
    max_steps = 50000
    total_episodes = 50
    batch_size = 64
    memory = instantiate_memory(batch_size, env, rows, cols, stack_size, possible_actions)
    
    if mode == 'train':
        decay_step = 0
        
        for episode in range(50):
            loss = 0
            step = 0
            episode_rewards = []
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True, (rows, cols), stack_size)
            
            while step < max_steps:
                print(step)
                step += 1
                decay_step +=1
                action, explore_probability = predict_action(model, explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
                next_state, reward, done, _ = env.step(action)
                
                # if True: #always render
                #     env.render()
                
                episode_rewards.append(reward)
                
                if done:
                    next_state = np.zeros((rows, cols, 3), dtype=np.uint8) # data type need to be correct
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, (rows, cols), stack_size)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    print(f'Episode: {episode}, Total reward: {total_reward}, Explore P: {explore_probability}, Training Loss {loss}')

                    # rewards_list.append((episode, total_reward))
                    memory.add((state, action, reward, next_state, done))
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, (rows, cols), stack_size)
                    memory.add((state, action, reward, next_state, done))
                    state = next_state
                    
                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)

                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []
                Qs_next_state = [model.predict(nsmb.reshape(1, nsmb.shape[0], nsmb.shape[1], nsmb.shape[2])) for nsmb in next_states_mb]
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    if terminal:
                        target_Qs_batch.append([a * rewards_mb[i] for a in actions_mb[i]])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append([a * target for a in actions_mb[i]])

                targets_mb = np.array([each for each in target_Qs_batch])
                loss += model.train_on_batch(states_mb, targets_mb)

            # if episode % 5 == 0:
            #     save_path = saver.save(sess, "./models/model.ckpt")
            #     print("Model Saved")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        train_network(sys.argv[1])
    else:
        print("Incorrect arguments")
