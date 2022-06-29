import numpy as np
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.reward_ones = []
        self.reward_zeros = []
        self.idx = 0
    
    def push(self, state, action, reward, next_state, done):
            
        self.buffer.append((state, action, reward, next_state, done))
        if reward > 0:
          self.reward_ones.append(self.idx)
        else:
          self.reward_zeros.append(self.idx)
        self.idx += 1

    
    def sample(self, batch_size,ratio_zero_to_one=-1):
        if ratio_zero_to_one == -1:
          idx = np.random.randint(0,len(self.buffer), batch_size)
        else:
          n_zeros = int(ratio_zero_to_one * batch_size)
          n_ones = batch_size - n_zeros
          if not len(self.reward_zeros):
            n_ones = batch_size
            n_zeros = 0
          if not len(self.reward_ones):
            n_zeros = batch_size
            n_ones = 0

          if n_zeros:
            idx_zeros = np.random.choice(self.reward_zeros, n_zeros)
          else:
            idx_zeros = np.array([])
          if n_ones:
            idx_ones = np.random.choice(self.reward_ones, n_ones)
          else:
            idx_ones = np.array([])
          idx = np.concatenate([idx_ones,idx_zeros],0)
        
        idx = idx.astype(np.int).ravel()
        state, action, reward, next_state, done = zip(*np.array(self.buffer)[idx])
        return np.stack(state), action, reward, np.stack(next_state), done
    
    def __len__(self):
        return len(self.buffer)