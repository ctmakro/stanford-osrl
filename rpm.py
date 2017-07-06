# from collections import deque
import numpy as np
import random

# replay buffer per http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
class rpm(object):
    #replay memory
    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, obj):
        if self.size() >= self.buffer_size:
            # self.buffer.popleft()
            # self.buffer = self.buffer[1:]
            self.buffer.pop(0)
        self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''

        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        item_count = len(batch[0])
        res = []
        for i in range(item_count):
            # k = np.array([item[i] for item in batch])
            k = np.stack((item[i] for item in batch),axis=0)
            # if len(k.shape)==1: k = k.reshape(k.shape+(1,))
            if len(k.shape)==1: k.shape+=(1,)
            res.append(k)
        return res
