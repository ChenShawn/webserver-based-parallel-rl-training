set -x

processes=`ps -ef | grep "python main.py" | awk '{ print $2 }'`
for proc in ${processes}
do
    kill -9 ${proc}
done

processes=`ps -ef | grep "worker.py" | awk '{ print $2 }'`
for proc in ${processes}
do
    kill -9 ${proc}
done

# remove all shm before killing mempool servers
source ./distributed.config
mempool_ports=`echo "${mempool_ports_raw}" | awk -F, '{ print $1 }'`
for port in ${mempool_ports}
do
    ret=`curl http://localhost:${port}/close`
    echo "curl http://localhost/close    ${ret}"
done

processes=`ps -ef | grep "mempool_server.py" | awk '{ print $2 }'`
for proc in ${processes}
do
    kill -9 ${proc}
done

processes=`ps -ef | grep "tensorboard" | awk '{ print $2 }'`
for proc in ${processes}
do
    kill -9 ${proc}
done
