# webserver-based-parallel-rl-training

A **simplified** scalable distributed RL training framework, supporting asynchronous RL training with arbitrary number of rollout workers and replay memory servers.

Implemented with:
- Python Flask server (to be replaced by [brpc](https://github.com/apache/incubator-brpc))
- [protobuf Python interface](https://github.com/protocolbuffers/protobuf)
- Linux shared memory mechanism
- [PyTorch](https://github.com/pytorch/pytorch)
- [OpenAI gym](https://github.com/openai/gym)

Still on developing...

## Usage

Compile C++ brpc version of rollout server:
```bash
cd rollout/cpp/ && make brpc
```
This will create a folder named `./rollout/cpp/bld/`, with the binary executable `mempool_server` in it.

Start training:
```bash
# NOTE: Use `bash` to activate the scripts instead of `sh`, which is a link of dash and may have potential bugs
bash start.sh
```

`start.sh` will create several folders under the current working directory, with their names starting with either `running_rollout_` or `running_worker_`, which correspond to mempool server processes and worker processes, respectively.

Terminate training:
```bash
bash kill.sh
```

Clean all temporary files:
```bash
bash clean.sh
```
**Warning:** this will delete all log files and models checkpoints at once, back-up if necessary.

## Parameters

Global variables are set in two files:
- `distributed.config`: number of worker processes in training, ports of the replay memory servers
- `global_variables.py`: all other relative variables in training

My preliminary experimental results on a CPU machine show that , when the number of mempool processes and worker processes reaches 1:4, the writing speed and reading speed of the memory pool can be roughly balanced.

## More information (in Chinese)

See [Web Server Based Parallel RL Training Framework](https://chenshawn.github.io/2020/02/18/web-based-parallel-rl/#more)
