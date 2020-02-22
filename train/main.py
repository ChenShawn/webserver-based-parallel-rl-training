import numpy as np
import random
import torch
import tensorflow as tf
import argparse
import os, sys
import gym
import logging
import requests
sys.path.append("..")

import global_variables as G
from models import sac
from network_dataset import NetworkDataset


def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('-e', '--env', default=G.ENV_NAME, type=str, help='gym env')
    parser.add_argument('-p', '--policy', default=G.POLICY_TYPE, type=str, help='Gaussian|anything else')
    parser.add_argument('--gamma', type=float, default=G.GAMMA, help='discount reward')
    parser.add_argument('--tau', type=float, default=G.TAU, help='target smoothing coefficient (τ)')
    parser.add_argument('--lr', type=float, default=G.LR, help='learning rate')
    parser.add_argument('--alpha', type=float, default=G.ALPHA, help='Temperature')
    parser.add_argument('--seed', type=int, default=G.SEED, help='global random seed')
    parser.add_argument('--num-steps', type=int, default=G.NUM_STEPS, help='maximum number of steps')
    parser.add_argument('--hidden-size', type=int, default=G.HIDDEN_SIZE, help='hidden size')
    parser.add_argument('--max-episode-len', type=int, default=G.MAX_EPISODE_LEN, help='max episode len')
    parser.add_argument('--target-update-interval', type=int, default=G.TARGET_UPDATE_INTERVAL, help='Q-target update per steps')
    parser.add_argument('--save-interval', type=int, default=G.SAVE_INTERVAL, help='Q-target update per steps')
    parser.add_argument('--log-interval', type=int, default=200, help='interval to write tensorboard')
    parser.add_argument('--replay-size', type=int, default=G.REPLAY_CAPACITY, help='size of replay buffer')
    return parser.parse_args()


def set_global_seeds(seed):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)


def main():
    args = get_arguments()
    env = gym.make(args.env)
    agent = sac.SAC(num_inputs=G.STATE_SIZE, action_space=env.action_space, args=args)
    env.close()
    dataset = NetworkDataset()
    set_global_seeds(args.seed)

    logname  = './logs/{}.log'.format(args.env)
    logfmt = '%(levelname)s - %(asctime)s - %(message)s'
    logging.basicConfig(filename=logname, level=logging.DEBUG, format=logfmt)
    logger = logging.getLogger(__name__)
    writer = tf.summary.FileWriter('./tensorboard/')

    for total_it in range(args.num_steps):
        qf1_loss, qf2_loss, policy_loss, reward_mean = agent.update_parameters(dataset, total_it)
        if total_it % args.log_interval == 0:
            sumstr = tf.Summary(value=[
                tf.Summary.Value(tag='loss/qloss_1', simple_value=qf1_loss),
                tf.Summary.Value(tag='loss/qloss_2', simple_value=qf2_loss),
                tf.Summary.Value(tag='loss/pi_loss', simple_value=policy_loss),
                tf.Summary.Value(tag='agent/reward', simple_value=reward_mean),
            ])
            writer.add_summary(sumstr, global_step=total_it)
        if total_it % args.save_interval == 0:
            agent.save_model(args.env)
    logger.info('training finished!')
    # close shm
    req_list = [grequests.get(url + '/close') for url in G.MEMPOOL_SERVER_LIST]
    ret_list = grequests.map(req_list)
    for url, ret in zip(G.MEMPOOL_SERVER_LIST, ret_list):
        logger.info(f'shm exit status: {url} - {ret.status_code}')
    logger.info(' [*] All finished! Ready to exit!')


if __name__ == '__main__':
    main()