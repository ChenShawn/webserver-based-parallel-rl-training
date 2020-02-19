from flask import Flask
from flask import request, jsonify
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
    return '200'


@app.route('/read', methods=['GET'])
def sample_batch():
    """sample_batch
    NOTE: should NOT use http to transmit samples due to its low efficiency!
    The mempool server and trainer must be activated on the same machine.
    By calling this interface the mempool server just refills the shmbuf,
    so that the Trainer can directly read shared memory to get a batch of samples.
    """
    global replay_memory
    if replay_memory.sample(G.BATCH_SIZE):
        return '200'
    else:
        return '404'


@app.route('/stats', methods=['GET'])
def get_stat_info():
    global replay_memory
    jsoninfo = {
        'n_samples': replay_memory.num_samples, 
        'num_read': replay_memory.num_read,
        'num_write': replay_memory.num_write,
    }
    return jsonify(jsoninfo)


@app.route('/shminfo', methods=['GET'])
def get_shminfo():
    global replay_memory
    info = {
        'shmid': int(replay_memory.addinfo[0]), 
        'offset': int(replay_memory.addinfo[1])
    }
    return jsonify(info)


@app.route('/close', methods=['GET'])
def server_prepare_close():
    """server_prepare_close
    Only be called when training is finished
    mempool server start to release and delete its shared memory
    """
    global replay_memory
    try:
        replay_memory.close()
    except Exception as err:
        return '404'
    return '200'

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mempool_server')
    parser.add_argument('-p', '--port', default=G.SERVER_PORT, type=int, help='port')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args = parser.parse_args()

    replay_memory = ReplayMemory(capacity=1e+6)

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)