import gym
import gc
from functools import reduce


# environmental parameters
# ENV_NAME = 'LunarLanderContinuous-v2'
ENV_NAME = 'Pendulum-v0'
_env = gym.make(ENV_NAME)
STATE_SHAPE = _env.observation_space.shape
ACTION_SHAPE = _env.action_space.shape
STATE_SIZE = reduce(lambda x, y: x * y, STATE_SHAPE)
ACTION_SIZE = reduce(lambda x, y: x * y, ACTION_SHAPE)
ACTION_RANGE = [_env.action_space.low[0], _env.action_space.high[0]]
del _env
gc.collect()


# training
# NOTE: note that the true batch_size used in training is BATCH_SIZE * len(MEMPOOL_SERVER_LIST)
BATCH_SIZE = 256
GAMMA = 0.99
POLICY_TYPE = 'Gaussian'
HIDDEN_SIZE = 256
LR = 0.0001
TAU = 0.005
ALPHA = 0.2
NUM_STEPS = 5000001
SEED = 123456
TARGET_UPDATE_INTERVAL = 1
POLICY_UPDATE_INTERVAL = 2
SAVE_INTERVAL = 50


# replay memory
# NOTE: The following line will be modified by start.sh in the beginning
SERVER_PORT_LIST = [20000]
REPLAY_CAPACITY = 1e+6
# NOTE: compute the smallest size that can contain a batch of samples and two more pages (4096)
_num_float32 = STATE_SIZE * 2 + ACTION_SIZE + 2
MAX_SHM_BYTES = 4 * BATCH_SIZE * _num_float32
MAX_SHM_BYTES_ROUNDED = MAX_SHM_BYTES
if MAX_SHM_BYTES_ROUNDED % 4096 != 0:
    # rounded up to multiple times of pgsize
    MAX_SHM_BYTES_ROUNDED = MAX_SHM_BYTES + 2 * 4096 + (4096 - (MAX_SHM_BYTES % 4096))

REPLAY_DLLNAME = 'libshm.so'
# NOTE: run `ipcs -m` to check whether conflict shmid exists
SHMKEY = 654321


# workers
# NOTE: replace 'localhost' with the IP address of training machine
MEMPOOL_SERVER_LIST = ['http://172.20.144.234:{}'.format(port) for port in SERVER_PORT_LIST]
MAX_EPISODE_LEN = 1000
TRAINER_DLLNAME = 'libtrain.so'
ACTOR_FILENAME = './train/sac_actor.pth'
CRITIC_FILENAME = './train/sac_critic.pth'
REQ_MODELS_INTERVAL = 10


if __name__ == '__main__':
    print('env_name:', ENV_NAME)
    print('shapes:', STATE_SHAPE, ACTION_SHAPE)
    print('action_range', ACTION_RANGE)
    
