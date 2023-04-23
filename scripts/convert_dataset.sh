CUR_DIR=`pwd`
DATA_PATH=${CUR_DIR}/dataset
# train dataset
python ${CUR_DIR}/elmo/data/reader.py  \
    --vocab_path="${DATA_PATH}/vocab-2016-09-10.txt" \
    --options_path="${DATA_PATH}/options.json"\
    --input_file="${DATA_PATH}/training-monolingual.tokenized.shuffled/*" \
    --output_file="${DATA_PATH}/train.mindrecord"