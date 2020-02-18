import google.protobuf
import random
import numpy as np
import ctypes as ct
import threading
import sys
import time

sys.path.append("..")

import global_variables as G
import samples_pb2 as pbfmt


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.num_samples = 0
        self.num_read, self.num_write = 0, 0

        # asynchronous settings
        self.readcntlock = threading.Lock()
        self.writecntlock = threading.Lock()
        self.rwlock = threading.Lock()
        self.writelock = threading.Lock()
        self.readcnt = 0

        # initialize shared memory
        self.shmbuf = np.zeros((G.MAX_SHM_BYTES), dtype=np.float32)
        if not self.shmbuf.flags['C_CONTIGUOUS']:
            self.shmbuf = np.ascontiguousarray(self.shmbuf, dtype=np.float32)
        c_shmbuf = self.shmbuf.ctypes.data_as(ct.c_void_p)
        c_shm_bytes = ct.c_int(G.MAX_SHM_BYTES)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, G.DLLNAME), '.')
        self.dll.init_shm(c_shmbuf, c_shm_bytes)
        

    def push(self, pbdata):
        # higher priority than func sample
        self.writelock.acquire()
        self.rwlock.acquire()

        # deserializing pbdata and write to buffer
        episode = pbfmt.Episode()
        episode.ParseFromString(pbdata)
        for sample in episode.samples:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            state = np.frombuffer(sample.state, dtype=np.float32).reshape(G.STATE_SHAPE)
            action = np.frombuffer(sample.action, dtype=np.float32).reshape(G.ACTION_SHAPE)
            self.buffer[self.position] = (state, action, sample.reward_sum)
            self.position = (self.position + 1) % self.capacity
        self.num_samples += len(episode.samples)
        self.num_write += 1

        self.rwlock.release()
        self.writelock.release()


    def sample(self, batch_size):
        # only applicable when more than half of the poll is filled
        if len(self.buffer) < self.capacity // 2:
            time.sleep(5)
        # lower priority than func push
        self.writelock.acquire()
        self.readcntlock.acquire()
        if self.readcnt == 0:
            self.rwlock.acquire()
        self.readcnt += 1
        self.readcntlock.release()
        self.writelock.release()

        batch = random.sample(self.buffer, batch_size)
        state, action, reward_sum = map(np.stack, zip(*batch))

        self.readcntlock.acquire()
        self.readcnt -= 1
        if self.readcnt == 0:
            self.rwlock.release()
        self.num_read += 1
        self.readcntlock.release()
        return state, action, reward_sum


    def statistic(self):
        return self.num_samples, self.num_read, self.num_write


    def __len__(self):
        return len(self.buffer)