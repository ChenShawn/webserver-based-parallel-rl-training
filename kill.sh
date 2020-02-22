set -x

for addr in ${mempool_server_list}
do
    ret=`curl http://${addr}/close`
    echo "curl http://${addr}/close    ${ret}"
done
