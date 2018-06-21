import numpy as np
import random
from collections import deque


class Replay_buffer(object):
    def __init__(self, memory_size,random_seed = 1):
        self.memory_size = memory_size
        self.buffer = deque()
        random.seed(random_seed)
        self.counter = 0

    def store_transition(self,s, a, r, t, s_,):
        # new experience is always stored in the right side
        experience = (s, a, r, t, s_)
        if self.counter < self.memory_size:
            self.buffer.append(experience)
            self.counter += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):

        if self.counter < batch_size:
            batch = random.sample(self.buffer, self.counter)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([sample[0] for sample in batch])
        a_batch = np.array([sample[1] for sample in batch])
        r_batch = np.array([sample[2] for sample in batch])
        t_batch = np.array([sample[3] for sample in batch])
        s2_batch = np.array([sample[4] for sample in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        # clear the buffer
        self.buffer.clear()
        self.counter = 0

    def size(self):
        # get how many transitions have been seen
        return self.counter
