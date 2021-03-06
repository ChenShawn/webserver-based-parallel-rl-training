set -x

# step 0: generate configurations and replace variables in global_variables
cd ./rollout/cpp/ && make && cd -
cd ./train/cpp/ && make && cd -
source ./distributed.config
mempool_ports=`echo "${mempool_ports_raw}" | tr "," "\n"`
sed -i "s/^SERVER_PORT_LIST.*=.*/SERVER_PORT_LIST = [${mempool_ports_raw}]/g" ./global_variables.py

# step 1: activate mempool servers
for port in ${mempool_ports}
do
    cp -r ./rollout ./running_rollout_${port}
    cd ./running_rollout_${port}
    nohup python mempool_server.py -p ${port} >./logs/mempool_server_${port}.log 2>&1 &
    cd -
done

# step 2: activate workers (if workers run on different machines you have to do this mannualy)
worker_ending_index=`expr ${worker_num} - 1`
for idx in `seq 0 ${worker_ending_index}`
do
    cp -r ./workers ./running_worker_${idx}
    cd ./running_worker_${idx}
    nohup python worker.py >./logs/run.log 2>&1 &
    cd -
done

# waiting for some time to let the workers fill all mempools
sleep 30
# step 3: start training
cd ./train
nohup python main.py >./logs/run.log 2>&1 &
nohup tensorboard --logdir ./tensorboard >/dev/null 2>&1 &
cd -

echo " [*] all processes started to run, good luck for a good convergence :)"
