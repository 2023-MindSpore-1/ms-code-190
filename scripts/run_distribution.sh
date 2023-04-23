
#!/bin/bash
ROOT_PATH=$(pwd)

export RANK_TABLE_FILE=$(realpath $1)
export RANK_SIZE=2
export DEVICE_NUM=2
#RANK_SIZE=$2
#export RANK_SIZE=$2
#test_dist_8pcs()
#{
#    export RANK_TABLE_FILE=${ROOT_PATH}/scripts/rank_table_8pcs.json
#    export DEVICE_NUM=8
#}
#test_dist_2pcs()
#{
#    export RANK_TABLE_FILE=${ROOT_PATH}/scripts/rank_table_2pcs.json
#    export DEVICE_NUM=2
#}
#test_dist_${RANK_SIZE}pcs

START_ID=0

for((i=0;i<${RANK_SIZE};i++))
do
    export DEVICE_ID=$((i+START_ID))
    export RANK_ID=$i
    echo "start training for device $DEVICE_ID, rank is $RANK_ID"
    rm -rf train$i
    mkdir train$i
    cd train$i
    env > env.log
    cp ../*.py .
    cp -r ../elmo .
    python train.py > train.log 2>&1 &
    cd ..
done
