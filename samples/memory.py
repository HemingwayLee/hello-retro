import random
import numpy as np
from collections import deque
from helper import stack_frames

class Memory():
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        index = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]

    def instantiate_memory(self, pretrain_length, env, rows, cols, stack_size, possible_actions):
        state = env.reset() 
        stacked_frames = deque([np.zeros((rows, cols), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)   
        state, stacked_frames = stack_frames(stacked_frames, state, True, (rows, cols), stack_size)
        
        for i in range(pretrain_length):
            choice = random.randint(1,len(possible_actions))-1
            action = possible_actions[choice]
            next_state, reward, done, _ = env.step(action)
            # env.render()
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, (rows, cols), stack_size)
            
            if done:
                next_state = np.zeros((rows, cols, 3), dtype=np.uint8) # data type need to be correct
                self.add((state, action, reward, next_state, done))
                state = env.reset()
                state, stacked_frames = stack_frames(stacked_frames, state, True, (rows, cols), stack_size)
            else:
                self.add((state, action, reward, next_state, done))
                state = next_state
