import gym
import pybullet_envs
from PIL import Image
import argparse
import numpy as np
import torch
import copy
import os
from sklearn.preprocessing import normalize as Normalize

from models import sac

def parse_arguments():
    parser = argparse.ArgumentParser("TESTING")
    parser.add_argument('-p', "--policy", type=str, default='Gaussian', help="Gaussian/Deterministic")
    parser.add_argument('-e', "--env", type=str, default="Pendulum-v0", help="env name")
    parser.add_argument('-n', "--n-episodes", type=int, default=10, help="number of episodes")
    parser.add_argument('-m', "--relative-mass", type=float, default=1.0, help="relative-mass")
    parser.add_argument("--noise-scale", type=float, default=0.0, help="relative-mass")
    parser.add_argument("--train-seed", type=int, default=123456, help="random seed for training")
    parser.add_argument("--test-seed", type=int, default=1, help="random seed for testing")
    parser.add_argument("--render", action="store_true", default=False)
    # RESERVED FIELD: DO NOT PASS VALUE TO THESE PARAMS FROM CMD
    parser.add_argument("--hidden-size", type=int, default=256, help="reserved field")
    parser.add_argument("--tau", type=float, default=0.1, help="reserved field")
    parser.add_argument("--gamma", type=float, default=0.99, help="reserved field")
    parser.add_argument("--alpha", type=float, default=0.2, help="reserved field")
    parser.add_argument("--lr", type=float, default=0.001, help="reserved field")
    parser.add_argument("--target-update-interval", type=int, default=1, help="reserved field")
    parser.add_argument("--automatic-entropy-tuning", action='store_true', default=False, help="reserved field")

    return parser.parse_args()


def gen_envs(arglist):
    env = gym.make(arglist.env)
    if 'model' in dir(env.env):
        # For mujoco envs
        ori_mass = copy.deepcopy(env.env.model.body_mass.copy())
        for idx in range(len(ori_mass)):
            env.env.model.body_mass[idx] = ori_mass[idx] * arglist.relative_mass
    elif 'world' in dir(env.env):
        # For some of the classic control envs
        env.env.world.gravity *= arglist.relative_mass
    return env



def test(arglist):
    env_name = arglist.env
    train_seed = arglist.train_seed
    test_seed = arglist.test_seed
    n_episodes = arglist.n_episodes
    render = arglist.render
    max_timesteps = 1001
    
    #env = gym.make(env_name)
    env = gen_envs(arglist)

    # Set random seed
    env.seed(test_seed)
    torch.manual_seed(test_seed)
    np.random.seed(test_seed)

    # load pretrained RL models
    agent = sac.SAC(env.observation_space.shape[0], env.action_space, arglist)
    agent.load_model(env_name)
    
    total_reward_list = []
    for ep in range(1, n_episodes+1):
        ep_reward = 0.0
        state = env.reset()
        for t in range(max_timesteps):
            noise = np.random.normal(0.0, 1.0, size=state.shape)
            noise = np.clip(noise, -1.0, 1.0)
            adv_state = state + arglist.noise_scale * noise
            action = agent.select_action(adv_state, eval=True)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break
            
        #print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward_list.append(ep_reward)
        ep_reward = 0.0
    env.close()
    return total_reward_list


if __name__ == '__main__':
    args = parse_arguments()

    reward_list = test(args)

    reward_array = np.array(reward_list, dtype=np.float32)
    reward_mean = reward_array.mean()
    reward_half_std = reward_array.std() / 2.0
    loginfo = 'policy={} env={} train_seed={} test_seed={} relative_mass={} noise_scale={} result={}±{}'
    print(loginfo.format(args.policy, args.env, args.train_seed, args.test_seed, args.relative_mass, args.noise_scale, reward_mean, reward_half_std))

