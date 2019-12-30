import retro
import sys
import random
import json
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from config import *
from memory import Memory
from helper import stack_frames

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

def predict_action(model, decay_step, state, possible_actions):
    # EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    exp_exp_tradeoff = np.random.rand()
    explore_probability = EXPLORE_STOP + (EXPLORE_START - EXPLORE_STOP) * np.exp(-DECAY_RATE * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        choice = random.randint(1,len(possible_actions))-1
        action = possible_actions[choice]
    else:
        Qs = model.predict(state.reshape(1, state.shape[0], state.shape[1], state.shape[2]))
        choice = np.argmax(Qs)
        action = possible_actions[choice]
       
    return action, explore_probability

def do_training(model, memory):
    batch = memory.sample(BATCH_SIZE)

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
            target = rewards_mb[i] + GAMMA * np.max(Qs_next_state[i])
            target_Qs_batch.append([a * target for a in actions_mb[i]])

    targets_mb = np.array([each for each in target_Qs_batch])
    return model.train_on_batch(states_mb, targets_mb)

# def is_done(prev, curr):
#     if prev == -1:
#         return False
#
#     if prev == curr:
#         return False
#    
#     return True

def train(env, model, rows, cols, possible_actions):
    memory = Memory(memory_size=1000000)
    memory.instantiate_memory(BATCH_SIZE, env, rows, cols, STACK_SIZE, possible_actions)
    
    decay_step = 0
    for episode in range(TOTAL_EPISODES):
        loss = 0
        step = 0
        prev_lives = -1
        episode_rewards = []

        state = env.reset()
        stacked_frames = deque([np.zeros((rows, cols), dtype=np.int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)
        state, stacked_frames = stack_frames(stacked_frames, state, True, (rows, cols), STACK_SIZE)
        
        while step < MAX_STEPS:
            # print(step)
            step += 1
            decay_step +=1
            action, explore_probability = predict_action(model, decay_step, state, possible_actions)
            
            ori_action = convert_back_possible_actions(action)
            # print(env.get_action_meaning(ori_action))

            # restart everytime we die
            # next_state, reward, _, info = env.step(ori_action)
            # done = is_done(prev_lives, info['lives'])
            # prev_lives = info['lives']
            
            next_state, reward, done, _ = env.step(ori_action)
            episode_rewards.append(reward * 25)
            # env.render()
            
            if done:
                # next_state = np.zeros((rows, cols, 3), dtype=np.uint8) # data type need to be correct
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, (rows, cols), STACK_SIZE)
                step = MAX_STEPS
                total_reward = np.sum(episode_rewards)
                print(f'Episode: {episode}, Total reward: {total_reward}, Explore P: {explore_probability}, Training Loss {loss}')

                # rewards_list.append((episode, total_reward))
                memory.add((state, action, reward, next_state, done))
            else:
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, (rows, cols), STACK_SIZE)
                memory.add((state, action, reward, next_state, done))
                state = next_state
                
            loss += do_training(model, memory)

        if episode % 25 == 0:
            print("Save the model now")
            model.save_weights(f"model.{episode}.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

def run(env, model, rows, cols, possible_actions, filename="model.150.h5"):
    model.load_weights(filename)
    
    state = env.reset()
    stacked_frames = deque([np.zeros((rows, cols), dtype=np.int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)
    state, stacked_frames = stack_frames(stacked_frames, state, True, (rows, cols), STACK_SIZE)
    
    total_rewards = 0
    while True:
        Qs = model.predict(state.reshape(1, state.shape[0], state.shape[1], state.shape[2]))
        choice = np.argmax(Qs)
        action = possible_actions[choice]
        
        next_state, reward, done, _ = env.step(convert_back_possible_actions(action))
        env.render()

        total_rewards += reward

        if done:
            print(f"The Score is {total_rewards}")
            break
            
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, (rows, cols), STACK_SIZE)
        state = next_state

    env.close()

def convert_back_possible_actions(valid_action):
    if (valid_action == np.array([1,0,0])).all():
        return np.array([1,0,0,0,0,0,0,0])
    elif (valid_action == np.array([0,1,0])).all():
        return np.array([0,0,0,0,0,0,1,0])
    elif (valid_action == np.array([0,0,1])).all():
        return np.array([0,0,0,0,0,0,0,1])
    else:
        print(valid_action)
        print("unexpected error!!!!")
        return np.array([0,0,0,0,0,0,0,0])

# NOTE: This is Breakout dependent
def get_valid_actions():
    valid_actions = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])
    
    return valid_actions

def main(mode):
    env = retro.make(game='Breakout-Atari2600')
    # env = retro.make(game='Airstriker-Genesis')
    print(f"The size of our frame is: {env.observation_space}")
    print(f"The action size is : {env.action_space.n}")

    possible_actions = get_valid_actions()
    # possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
    print(f"possible actions:\n{possible_actions}")

    observation = env.reset()
    rows, cols = int(observation.shape[0]/RESIZE_FACTOR), int(observation.shape[1]/RESIZE_FACTOR)
    print(f"rows: {rows}, cols: {cols}")

    model = build_model(len(possible_actions), (rows, cols, STACK_SIZE))
    
    if mode == 'train':
        train(env, model, rows, cols, possible_actions)
    elif mode == 'run':
        run(env, model, rows, cols, possible_actions)
    else:
        print("Incorrect mode")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Incorrect argument count")
