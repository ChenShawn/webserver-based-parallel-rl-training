import gym
import gc


# environmental parameters
ENV_NAME = 'Pendulum-v0'
_env = gym.make(ENV_NAME)
STATE_SHAPE = _env.observation_space.shape
ACTION_SHAPE = _env.action_space.shape
ACTION_RANGE = [_env.action_space.low[0], _env.action_space.high[0]]
del _env
gc.collect()

# replay memory
REPLAY_CAPACITY = 1e+6
MAX_SHM_BYTES = 65532
DLLNAME = 'shmlib.so'

# training
BATCH_SIZE = 512


if __name__ == '__main__':
    print('env_name:', ENV_NAME)
    print('shapes:', STATE_SHAPE, ACTION_SHAPE)
    print('action_range', ACTION_RANGE)
    