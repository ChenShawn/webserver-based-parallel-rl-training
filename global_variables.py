import gym
import gc
from functools import reduce


# environmental parameters
ENV_NAME = 'Pendulum-v0'
_env = gym.make(ENV_NAME)
STATE_SHAPE = _env.observation_space.shape
ACTION_SHAPE = _env.action_space.shape
ACTION_RANGE = [_env.action_space.low[0], _env.action_space.high[0]]
del _env
gc.collect()


# training
BATCH_SIZE = 256
GAMMA = 0.99


# replay memory
SERVER_PORT = 20000
REPLAY_CAPACITY = 1e+6
# NOTE: compute the smallest size that can contain a batch of samples and two more pages (4096)
_num_float32 = reduce(lambda x, y: x * y, STATE_SHAPE) + reduce(lambda x, y: x * y, ACTION_SHAPE) + 1
MAX_SHM_BYTES = 4 * BATCH_SIZE * _num_float32
MAX_SHM_BYTES_ROUNDED = MAX_SHM_BYTES
if MAX_SHM_BYTES_ROUNDED % 4096 != 0:
    # rounded up to multiple times of pgsize
    MAX_SHM_BYTES_ROUNDED = MAX_SHM_BYTES + 2 * 4096 + (4096 - (MAX_SHM_BYTES % 4096))

DLLNAME = 'libshm.so'
# NOTE: run `ipcs -m` to check whether conflict shmid exists
SHMKEY = 654321


# workers
MEMPOOL_SERVER_LIST = [
    'http://localhost:{}'.format(SERVER_PORT),
]
MAX_EPISODE_LEN = 1000


if __name__ == '__main__':
    print('env_name:', ENV_NAME)
    print('shapes:', STATE_SHAPE, ACTION_SHAPE)
    print('action_range', ACTION_RANGE)
    