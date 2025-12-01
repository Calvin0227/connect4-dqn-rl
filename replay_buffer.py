import random
import numpy as np
from collections import deque

# Data structure that stores all experiences (we will use these experiences, which are just tuples of features, as training data for DQN)
class ReplayBuffer:
    # construct a replay buffer (double sided queue = deque) with a maximum capacity of 50,000 experiences
    def __init__(self, capacity=50000): 
        self.buffer = deque(maxlen=capacity)

    # store experience tuples in the replay buffer
    def push(self, state, action, reward, next_state, done):
        # Convert to numpy arrays
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))

    # randomly sample batch_size amount of experiences from the buffer and store in batch
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert all attributes to numpy arrays
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)