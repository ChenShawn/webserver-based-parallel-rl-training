set -x

read -r -p "Warning: this script will rm all running results and log files in this dir. Confirm? [Y/n] " confirm
if [ ${confirm} == "n" -o ${confirm} == "N" ]; then
    exit 0
fi

rm ./rollout/logs/*
rm ./train/logs/*
rm ./train/tensorboard/*

filedirs=`ls running_*`
for file in ${filedirs}
do
    rm -rf ${file}
done
