import numpy as np
import ctypes as ct
import requests
import logging
import grequests
from functools import reduce
import os, sys
sys.path.append("..")

import global_variables as G

logfmt = '[%(levelname)s][%(asctime)s][%(filename)s][%(funcName)s][%(lineno)d] %(message)s'
logging.basicConfig(filename='./logs/network_dataset.log', level=logging.ERROR, format=logfmt)
logger = logging.getLogger(__name__)


def get_shminfo(remote_addr):
    """get_shminfo
    return: shmid, offset
    """
    remote_url = remote_addr + '/shminfo'
    ret = requests.get(remote_url)
    if ret.status_code == 200:
        shminfo = ret.json()
    else:
        errinfo = 'curl ({}) response error: {}'.format(remote_url, str(ret))
        logger.error(errinfo)
        raise RuntimeError(errinfo)
    return shminfo


class NetworkDataset(object):
    def __init__(self):
        self.remote_url_list = [addr + '/read' for addr in G.MEMPOOL_SERVER_LIST]
        self.remote_shminfo_list = [get_shminfo(addr) for addr in G.MEMPOOL_SERVER_LIST]
        self.buffer_list, self.addinfo_list = [], []

        # initialize shared memory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.dll = np.ctypeslib.load_library(os.path.join(BASE_DIR, G.TRAINER_DLLNAME), '.')
        for url in self.remote_url_list:
            shmbuf = np.zeros((G.MAX_SHM_BYTES_ROUNDED // 4), dtype=np.float32)
            if not shmbuf.flags['C_CONTIGUOUS']:
                shmbuf = np.ascontiguousarray(shmbuf, dtype=np.float32)
            self.buffer_list.append(shmbuf)
            c_shmbuf = shmbuf.ctypes.data_as(ct.c_void_p)
            c_shmkey = ct.c_int(G.SHMKEY)
            c_shm_bytes = ct.c_int(G.MAX_SHM_BYTES)
            # the same structure with that defined in replay_memory.py
            addinfo = np.zeros((5), dtype=np.int32)
            if not addinfo.flags['C_CONTIGUOUS']:
                addinfo = np.ascontiguousarray(addinfo, dtype=np.float32)
            c_addinfo = addinfo.ctypes.data_as(ct.c_void_p)
            self.dll.init_shm(c_shmbuf, c_shm_bytes, c_shmkey, c_addinfo)

            addinfo[2] = reduce(lambda x, y: x * y, G.STATE_SHAPE)
            addinfo[3] = reduce(lambda x, y: x * y, G.ACTION_SHAPE)
            addinfo[4] = G.BATCH_SIZE
            self.addinfo_list.append(addinfo)
            logger.debug('remote={} shmid={} offset={}'.format(url, addinfo[0], addinfo[1]))
        logger.info('network_dataset initialized!')


    def _read_shm(self, shmbuf, addinfo):
        # prepare s, a, r buffers independently
        states = np.zeros((G.STATE_SIZE * G.BATCH_SIZE), dtype=np.float32)
        states = np.ascontiguousarray(states, dtype=states.dtype)
        c_states = states.ctypes.data_as(ct.c_void_p)
        actions = np.zeros((G.ACTION_SIZE * G.BATCH_SIZE), dtype=np.float32)
        actions = np.ascontiguousarray(actions, dtype=actions.dtype)
        c_actions = actions.ctypes.data_as(ct.c_void_p)
        rewards = np.zeros((G.BATCH_SIZE), dtype=np.float32)
        rewards = np.ascontiguousarray(rewards, dtype=rewards.dtype)
        c_rewards = rewards.ctypes.data_as(ct.c_void_p)

        c_shmbuf = shmbuf.ctypes.data_as(ct.c_void_p)
        c_addinfo = addinfo.ctypes.data_as(ct.c_void_p)
        self.dll.read_batch_shm(c_shmbuf, c_addinfo, c_states, c_actions, c_rewards)
        states = np.reshape(states, [G.BATCH_SIZE] + list(G.STATE_SHAPE))
        actions = np.reshape(actions, [G.BATCH_SIZE] + list(G.ACTION_SHAPE))
        return states, actions, rewards


    def get_batch_data(self):
        # use grequests to support large number of mempool server requests
        req_list = [grequests.get(url) for url in self.remote_url_list]
        ret_list = grequests.map(req_list)
        state_list, action_list, reward_list = [], [], []
        for ret, buffer, addinfo in zip(ret_list, self.buffer_list, self.addinfo_list):
            if ret.status_code != 200:
                continue
            state, action, reward_sum = self._read_shm(buffer, addinfo)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward_sum)
        state = np.concatenate(state_list, axis=0)
        action = np.concatenate(action_list, axis=0)
        reward_sum = np.concatenate(reward_list, axis=0)
        return state, action, reward_sum



if __name__ == '__main__':
    # unit test: this is to test if the data passed through shm are correct
    network_dataset = NetworkDataset()
    s, a, r = network_dataset.get_batch_data()
    print('state: ', s[0: 2, :], s.shape)
    print('action: ', a[0: 2, :], a.shape)
    print('reward: ', r[:10])