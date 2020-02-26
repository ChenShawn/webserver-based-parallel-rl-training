import gym
import requests
import google.protobuf
import numpy as np
import sys, os
import random
import time
import logging
import torch
import hashlib
from functools import partial, reduce
from collections import namedtuple
import socket
import threading
sys.path.append("..")

import global_variables as G
import samples_pb2 as pbfmt
from models import sac


logfmt = '[%(levelname)s][%(asctime)s][%(filename)s][%(funcName)s][%(lineno)d] %(message)s'
logging.basicConfig(filename='./logs/workers.log', level=logging.INFO, format=logfmt)
logger = logging.getLogger(__name__)
device = torch.device('cpu')


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


class DownloadThread(threading.Thread):
    """DownloadThread
    Models saving are running asynchronously to avoid long IO wait and delay
    TODO: Requesting with stream=True avoid loading large file into memory at once.
    However, when calling iter_content the remote file may have been modified,
    resulting in empty or error model files.
    """
    def __init__(self):
        super(DownloadThread, self).__init__()
        self.flag_success = False

    def run(self):
        url = random.choice(G.MEMPOOL_SERVER_LIST) + '/send_model'
        ret = requests.get(url + '/sac_actor.pth', stream=True)
        if ret.status_code == 200:
            with open(G.ACTOR_FILENAME, 'wb') as fd:
                for chunk in ret.iter_content(chunk_size=128):
                    if chunk:
                        fd.write(chunk)
        else:
            self.flag_success = False
        self.flag_success = True


class Worker(object):
    def __init__(self):
        self.env = gym.make(G.ENV_NAME)
        self.send_list = [addr + '/save' for addr in G.MEMPOOL_SERVER_LIST]
        self.max_episode_len = G.MAX_EPISODE_LEN
        self.gamma = G.GAMMA
        self.num_samples, self.num_send, self.num_failed = 0, 0, 0
        self.actor_md5sum = ''
        ArgList = namedtuple('ArgList', 'lr hidden_size alpha policy')
        arglist = ArgList(lr=G.LR, hidden_size=G.HIDDEN_SIZE, alpha=G.ALPHA, policy=G.POLICY_TYPE)
        state_size = reduce(lambda x, y: x * y, G.STATE_SHAPE)
        self.agent = sac.SAC(num_inputs=state_size, action_space=self.env.action_space, args=arglist)
        logger.info('Worker initialized in {}'.format(socket.gethostbyname(socket.gethostname())))


    def get_episode_data(self, reward_shaping=lambda x: float(x)):
        """get_episode_data
            Get an episode of samples
            return: [(s_0, a_0, r_0, s_1, m_0), (s_1, a_1, r_1, s_2, m_1), ...]
        """
        s = self.env.reset()
        done, epilen = False, 0
        states, actions, rewards, next_states, masks = [], [], [], [], []
        while not done:
            # self.env.render()
            if os.path.exists(G.ACTOR_FILENAME):
                # if both ckpt files exist the models should have been loaded
                a = self.agent.select_action(s)
            else:
                a = self.env.action_space.sample()
            s_next, r, done, info = self.env.step(a)
            mask = 1 if epilen == self.max_episode_len else float(not done)
            states.append(s.astype(np.float32))
            actions.append(a.astype(np.float32))
            rewards.append(reward_shaping(r))
            next_states.append(s_next.astype(np.float32))
            masks.append(float(mask))
            epilen += 1
            if epilen >= self.max_episode_len or done:
                break
            s = s_next
        logger.info('epilen={} reward_sum={}'.format(len(states), np.array(rewards).sum()))
        return states, actions, rewards, next_states, masks


    def check_reload(self):
        """check_reload"""
        actor_md5sum = md5sum(G.ACTOR_FILENAME)
        if actor_md5sum != self.actor_md5sum:
            self.agent.load_model(actor_path=G.ACTOR_FILENAME)
            self.actor_md5sum = actor_md5sum
            logger.info('new models confirmed and reload is now finished!')
        elif self.num_send % G.REQ_MODELS_INTERVAL == 0:
            # NOTE: large files can cause long delay
            # asynchronously requests mempool server for lastest models
            download_thread = DownloadThread()
            download_thread.start()
            return download_thread.flag_success
        return True


    def send_data(self):
        self.check_reload()
        states, actions, rewards, next_states, masks = self.get_episode_data()
        episode = pbfmt.Episode()
        for s, a, r, s_next, m in zip(states, actions, rewards, next_states, masks):
            sample = episode.samples.add()
            # TODO: ONLY FOR DEBUG!!!
            # sample.state = np.arange(s.shape[0], dtype=np.float32).tobytes()
            # sample.action = np.arange(a.shape[0], dtype=np.float32).tobytes()
            # sample.reward_sum = 3.1415926
            # sample.next_state = np.arange(s_next.shape[0], dtype=np.float32).tobytes()
            # sample.mask = 1.0
            sample.state = s.tobytes()
            sample.action = a.tobytes()
            sample.reward = r
            sample.next_state = s_next.tobytes()
            sample.mask = m
        pbdata = episode.SerializeToString()
        remote = random.choice(self.send_list)
        ret = requests.post(remote, data=pbdata)
        if ret.status_code != 200:
            logger.error("remote response error: " + str(ret))
            self.num_failed += 1
            return False
        self.num_samples += len(rewards)
        return True


    def run(self):
        # TODO: support dynamic maintainance of mempool server
        while True:
            # if self.num_failed > 20:
            #     logger.error('error time reaches maximum threshold')
            #     break
            # try:
            #     self.send_data()
            #     self.num_send += 1
            # except Exception as err:
            #     logger.error('Error: ' + str(err))
            #     self.num_failed += 1
            #     continue
            self.send_data()
            self.num_send += 1
        exit(-1)



if __name__ == '__main__':
    worker = Worker()
    worker.run()
