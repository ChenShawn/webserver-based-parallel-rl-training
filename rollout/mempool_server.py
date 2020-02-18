from flask import Flask
from flask import request
import argparse
import sys
sys.path.append("..")

import global_variables as G
from replay_memory import ReplayMemory

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/save', methods=['POST'])
def save_pbdata_samples():
    global replay_memory
    pbdata = request.data
    replay_memory.push(pbdata)
    return 200


@app.route('/read', methods=['GET'])
def sample_batch():
    """sample_batch
    NOTE: should NOT use http to transmit samples due to its low efficiency!
    The mempool server and trainer must be activated on the same machine.
    By calling this interface the mempool server just refills the shmbuf,
    so that the Trainer can directly read shared memory to get a batch of samples.
    """
    global replay_memory
    state, action, reward_sum = replay_memory.sample()


@app.route('/stats', methods=['GET'])
def get_stat_info():
    return ''


@app.route('./shmid', methods=['GET'])
def get_shmid():
    global replay_memory
    return replay_memory.shmid

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mempool_server')
    parser.add_argument('-p', '--port', default=20000, type=int, help='port')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args = parser.parse_args()

    replay_memory = ReplayMemory(capacity=1e+6)

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)