set -x

# step 1: activate mempool servers

# step 2: activate workers (if workers run on different machines u have to do this mannualy)

# waiting for some time to let the workers fill all mempools
sleep 20

# step 3: start training

# activate send_model
cd ./train
nohup sh send_models.sh >./logs/send_models.log 2>&1 &

echo " [*] all processes started to run, good luck for a good convergence :)"
