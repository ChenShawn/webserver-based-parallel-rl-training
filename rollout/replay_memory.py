import google.protobuf
import random
import numpy as np
from functools import reduce
import ctypes as ct
import threading
import logging
import sys
import os
import time

sys.path.append("..")

import global_variables as G
import samples_pb2 as pbfmt


logfmt = '[%(levelname)s][%(asctime)s][%(filename)s][%(funcName)s][%(lineno)d] %(message)s'
logging.basicConfig(filename='./logs/replay_memory.log', level=logging.ERROR, format=logfmt)
logger = logging.getLogger(__name__)


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

        # init logs for debugging
        logline = 'state_shape={} action_shape={} max_shm_bytes={} max_shm_bytes_rounded={}'
        logline = logline.format(G.STATE_SHAPE, G.ACTION_SHAPE, G.MAX_SHM_BYTES, G.MAX_SHM_BYTES_ROUNDED)
        logger.debug(logline)

        # initialize shared memory
        self.shmbuf = np.zeros((G.MAX_SHM_BYTES_ROUNDED // 4), dtype=np.float32)
        if not self.shmbuf.flags['C_CONTIGUOUS']:
            self.shmbuf = np.ascontiguousarray(self.shmbuf, dtype=np.float32)
        # addinfo[0] = shmid, addinfo[1] = offset to c_shmbuf (bytes)
        self.addinfo = np.zeros((5), dtype=np.int32)
        c_addinfo = self.addinfo.ctypes.data_as(ct.c_void_p)
        c_shmbuf = self.shmbuf.ctypes.data_as(ct.c_void_p)
        c_shm_bytes = ct.c_int(G.MAX_SHM_BYTES)
        c_shmkey = ct.c_int(G.SHMKEY)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, G.REPLAY_DLLNAME), '.')
        self.dll.init_shm(c_shmbuf, c_shm_bytes, c_shmkey, c_addinfo)
        # addinfo[2] = STATE_DIMS, addinfo[3] = ACTION_DIMS, addinfo[4] = BATCH_SIZE
        self.addinfo[2] = reduce(lambda x, y: x * y, G.STATE_SHAPE)
        self.addinfo[3] = reduce(lambda x, y: x * y, G.ACTION_SHAPE)
        self.addinfo[4] = G.BATCH_SIZE
        logger.debug('shmid={} offset={}'.format(self.addinfo[0], self.addinfo[1]))
        

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
            self.position = int((self.position + 1) % self.capacity)
        self.num_samples += len(episode.samples)
        self.num_write += 1

        self.rwlock.release()
        self.writelock.release()


    def sample(self, batch_size):
        # TODO: support any batch size passed via GET
        # lower priority than func push
        self.writelock.acquire()
        self.readcntlock.acquire()
        if self.readcnt == 0:
            self.rwlock.acquire()
        self.readcnt += 1
        self.readcntlock.release()
        self.writelock.release()

        # read samples from Python list buffer and map it to fill shm
        batch = random.sample(self.buffer, batch_size)
        state, action, reward_sum = map(np.stack, zip(*batch))
        state = np.ascontiguousarray(state.flatten(), dtype=np.float32)
        action = np.ascontiguousarray(action.flatten(), dtype=np.float32)
        reward_sum = np.ascontiguousarray(reward_sum.flatten(), dtype=np.float32)
        c_state = state.ctypes.data_as(ct.c_void_p)
        c_action = action.ctypes.data_as(ct.c_void_p)
        c_reward_sum = reward_sum.ctypes.data_as(ct.c_void_p)
        c_shmbuf = self.shmbuf.ctypes.data_as(ct.c_void_p)
        c_addinfo = self.addinfo.ctypes.data_as(ct.c_void_p)
        self.dll.fill_batch_shm(c_shmbuf, c_addinfo, c_state, c_action, c_reward_sum)

        self.readcntlock.acquire()
        self.readcnt -= 1
        if self.readcnt == 0:
            self.rwlock.release()
        self.num_read += 1
        self.readcntlock.release()
        return True


    def statistic(self):
        return self.num_samples, self.num_read, self.num_write

    def close(self):
        """close
        call `shmdt` and `shm`
        """
        c_shmbuf = self.shmbuf.ctypes.data_as(ct.c_void_p)
        c_addinfo = self.addinfo.ctypes.data_as(ct.c_void_p)
        self.dll.close_shm(c_shmbuf, c_addinfo)

    def __len__(self):
        return len(self.buffer)

    # def __del__(self):
    #     self.close()



if __name__ == '__main__':
    # ONLY FOR DEBUG
    replay = ReplayMemory(capacity=G.REPLAY_CAPACITY)
    
    replay.close()

    print(' [*] Done!!')