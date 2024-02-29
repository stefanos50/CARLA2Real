import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, next_state, reward, done):
        experience = (state, action, next_state, reward, done)
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        return np.stack(state), np.stack(action), np.stack(next_state), np.stack(reward), np.stack(done)

