import gym
import requests
import numpy as np
import sys, os
import random
import time
import torch
import hashlib
from collections import namedtuple
import ctypes as ct

from models import sac

def md5sum(filename):
    """md5sum
    Equivalent to the `md5sum` cmd in Linux shell
    return empty str if file does not exist
    """
    if not os.path.exists(filename):
        return ''
    with open(filename, 'rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


def get_pointer_from_array(arr):
    arr = arr.flatten().astype(np.float32)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr.ctypes.data_as(ct.c_void_p)


class Worker(object):
    def __init__(self, args, index=0):
        self.index = index
        self.env = gym.make(args.envname)
        self.args = args
        self.num_samples, self.num_send = 0, 0
        self.actor_md5sum = ''
        ArgList = namedtuple('ArgList', 'lr hidden_size alpha policy')
        self.agent = sac.SAC(
            num_inputs=args.state_size, 
            action_space=self.env.action_space, 
            args=ArgList(
                lr=args.lr, 
                hidden_size=args.hidden_size, 
                alpha=args.alpha, 
                policy=args.policy
            )
        )
        print(f' [*] Model SAC {args.policy} initialized!!')
        # initialize dynamic libraries
        basedir = os.path.dirname(os.path.abspath(__file__))
        self.dll = np.ctypeslib.load_library(os.path.join(basedir, args.dllname), '.')
        self.dll.init_channel(
            get_pointer_from_array(np.fromstring(args.remote_addr, dtype=np.uint8)),
            ct.c_int(args.timeout_ms),
            ct.c_int(args.max_retry)
        )
        print('Worker {} dll initialized!!'.format(index))


    def get_episode_data(self, reward_shaping=lambda x: float(x)):
        """get_episode_data
            Get an episode of samples
            return: type numpy.ndarray [s, a, r, s_next, mask]
        """
        s = self.env.reset()
        done, epilen = False, 0
        states, actions, rewards, next_states, masks = [], [], [], [], []
        while not done:
            # self.env.render()
            if os.path.exists(self.args.actor_name):
                # if both ckpt files exist the models should have been loaded
                a = self.agent.select_action(s)
            else:
                a = self.env.action_space.sample()
            s_next, r, done, info = self.env.step(a)
            mask = 1 if epilen == self.args.max_episode_len else float(not done)
            states.append(s[None, :])
            actions.append(a[None, :])
            rewards.append(reward_shaping(r))
            next_states.append(s_next[None, :])
            masks.append(float(mask))
            epilen += 1
            if epilen >= self.args.max_episode_len or done:
                break
            s = s_next
        logger.info('epilen={} reward_sum={}'.format(len(states), np.array(rewards).sum()))
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.array(rewards, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        return states, actions, rewards, next_states, masks


    def check_reload(self):
        """check_reload"""
        actor_md5sum = md5sum(self.args.actor_name)
        if actor_md5sum != self.actor_md5sum:
            self.agent.load_model(actor_path=self.args.actor_name)
            self.actor_md5sum = actor_md5sum
            logger.info('new models confirmed and reload is now finished!')
        elif self.num_send % self.args.req_models_interval == 0:
            # NOTE: large files can cause long delay
            # asynchronously requests mempool server for lastest models
            self.dll.download_model_files()
        return True


    def send_data(self):
        self.check_reload()
        states, actions, rewards, next_states, masks = self.get_episode_data()
        self.dll.send_rpc_request(
            ct.c_int(states.shape[0]),
            ct.c_int(self.args.state_size),
            ct.c_int(self.args.action_size),
            get_pointer_from_array(states),
            get_pointer_from_array(actions),
            get_pointer_from_array(rewards),
            get_pointer_from_array(next_states),
            get_pointer_from_array(masks)
        )
        self.num_samples += states.shape[0]
        return True


    def run(self):
        # TODO: support dynamic maintainance of mempool server
        while True:
            self.send_data()
            self.num_send += 1
        return True


    def close(self):
        self.dll.close_channel()



if __name__ == '__main__':
    worker = Worker()
    worker.run()
    worker.close()