import argparse
import os

from rpc import async_rpc_worker

def parse_arguments():
    parser = argparse.ArgumentParser(description='worker')

    # global configuration
    parser.add_argument('--remote-addr', default='localhost:20000', type=str)
    parser.add_argument('--dllname', default='client.so', type=str)
    parser.add_argument('--actor-name', default='./train/sac_actor.pth', type=str)
    parser.add_argument('--req-models-interval', default=10, type=int, help='10')
    parser.add_argument('--timeout-ms', default=500, type=int, help='500')
    parser.add_argument('--max-retry', default=5, type=int, help='3')

    # environmental variables
    parser.add_argument('--envname', default='Pendulum-v0', type=str, help='Pendulum')
    parser.add_argument('--state-size', default=3, type=int, help='3')
    parser.add_argument('--action-size', default=1, type=int, help='1')
    parser.add_argument('--max-episode-len', default=1000, type=int, help='1000')

    # model configuration
    parser.add_argument('--policy', default='Gaussian', type=str, help='Gaussian')
    parser.add_argument('--lr', default=1e-3, type=float, help='1e-3')
    parser.add_argument('--alpha', default=0.1, type=float, help='0.1')
    parser.add_argument('--hidden-size', default=256, type=int, help='256')
    return parser.parse_args()


def main():
    args = parse_arguments()
    worker = async_rpc_worker.Worker(args)
    worker.run()


if __name__ == '__main__':
    main()