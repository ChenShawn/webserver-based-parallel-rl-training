import gym
import requests
import google.protobuf
import numpy as np
import sys, os
import random
import time
import logging
sys.path.append("..")

import global_variables as G
import samples_pb2 as pbfmt


logfmt = '[%(levelname)s][%(asctime)s][%(filename)s][%(funcName)s][%(lineno)d] %(message)s'
logging.basicConfig(filename='./logs/workers.log', level=logging.DEBUG, format=logfmt)
logger = logging.getLogger(__name__)


class Worker(object):
    def __init__(self):
        self.env = gym.make(G.ENV_NAME)
        self.send_list = [addr + '/save' for addr in G.MEMPOOL_SERVER_LIST]
        self.max_episode_len = G.MAX_EPISODE_LEN
        self.gamma = G.GAMMA


    def get_random_episode_data(self, reward_option=None):
        """get_random_episode_data
            Get a episode of samples where the actions are uniformly sampled
            reward_option: 'GAE' or anything else, by default equivalent to GAE where lambda==1
            return: [(s_0, a_0, r_0), (s_1, a_1, r_1), ...]
        """
        s = self.env.reset()
        done = False
        epilen = 0
        states, actions, rewards = [], [], []
        while not done:
            # self.env.render()
            a = self.env.action_space.sample()
            s_next, r, done, info = self.env.step(a)
            states.append(s.astype(np.float32))
            actions.append(a.astype(np.float32))
            rewards.append(self.reward_shaping(r))
            epilen += 1
            if epilen > self.max_episode_len or done:
                break
            s = s_next
        if reward_option == 'GAE':
            # TODO: implement GAE to reduce variance
            raise NotImplementedError('GAE on developing...')
        else:
            reward_sum_list = []
            rval = 0.0
            for r in rewards[::-1]:
                rval = self.gamma * rval + r
                reward_sum_list.append(rval)
            reward_sum_list = reward_sum_list[::-1]
        return states, actions, reward_sum_list


    def get_episode_data(self):
        return []


    def reward_shaping(self, r):
        """reward_shaping
            no reward shaping by default
        """
        return r


    def run(self, random_mode=False):
        # while True:
        for _ in range(20):
            if random_mode:
                states, actions, rewards = self.get_random_episode_data()
            else:
                states, actions, rewards = self.get_episode_data()
            episode = pbfmt.Episode()
            for s, a, r in zip(states, actions, rewards):
                sample = episode.samples.add()
                sample.state = s.tobytes()
                sample.action = a.tobytes()
                sample.reward_sum = r
            pbdata = episode.SerializeToString()
            remote = random.choice(self.send_list)
            ret = requests.post(remote, data=pbdata)
            if ret.status_code != 200:
                logger.error("remote server response exception")
            time.sleep(0.5)


if __name__ == '__main__':
    worker = Worker()
    worker.run(random_mode=True)

    # s, a, r = worker.get_random_episode_data()
    # print(s, a, r)