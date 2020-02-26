set -x

read -r -p "Warning: this script will rm all running results and log files in this dir. Confirm? [Y/n] " confirm

rm ./rollout/logs/*
rm ./workers/logs/*
rm ./train/logs/*
rm ./train/tensorboard/*
rm ./train/nohup.out
#rm -rf ./train/train/*

source ./distributed.config
worker_ending_index=`expr ${worker_num} - 1`
for idx in `seq 0 ${worker_ending_index}`
do
    rm -rf ./running_worker_${idx}
done

source ./distributed.config
mempool_ports=`echo "${mempool_ports_raw}" | tr "," "\n"`
for port in ${mempool_ports}
do
    rm -rf ./running_rollout_${port}
done

