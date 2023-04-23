import json
import os
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Model, context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
import mindspore.communication.management as D
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig
from elmo.data.reader import create_elmo_dataset
from elmo.model import LanguageModel
from ElmoTrainOne import ElmoTrainOnestepWithLoss, ElmoTrainOnestepWithLossAscend
from elmo.utils.util import LossCallBack
from mindspore.profiler import Profiler
from mindspore.train.serialization import load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser(description="elmo")
parser.add_argument('--data_url', default='../dataset/train/', help='Location of data.')
parser.add_argument('--eval_url', default='../dataset/small.mindrecord', help='Location of data.')
parser.add_argument('--train_url', default='../ckpt', help='Location of training outputs.')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.2)')
parser.add_argument('--epoch_num', type=int, default=2, help='epoch_num, default is 1')
parser.add_argument('--sink_size', type=int, default=100, help='Sink size for every iteration, default is 100')
parser.add_argument('--do_train', type=str, default='true', choices=["true","false"], help='enable train')
parser.add_argument('--do_eval', type=str, default='false', choices=["true","false"], help='enable eval')
args = parser.parse_args()

def run_pretrain():
    set_seed(0)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(save_graphs=True, save_graphs_path='irs')
    device_id = int(os.getenv('DEVICE_ID', 0))
    device_num = int(os.getenv('RANK_SIZE', 1))
    rank_id = int(os.getenv('RANK_ID', 0))
    if args.device_target == 'Ascend':
        print(f'device_id={device_id}, rank_id={rank_id}, device_num={device_num}', flush=True)
        context.set_context(device_id=device_id)
        # profiler = Profiler()
        if device_num > 1:
            D.init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    elif args.device_target == "GPU":
        if device_num > 1:
            D.init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                gradients_mean=True)

    options_file = '/home/taoqiuyu/ELMo/dataset/options.json'
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    if args.do_train.lower() == 'true':
        lm = LanguageModel(options=options, training=True)
       
        if device_num > 1:
            dataset = create_elmo_dataset(batch_size=options['batch_size'], data_file_path=args.data_url,
                                  num_shards=device_num, shard_id=rank_id)
        do_train(dataset, lm) 

    if args.do_eval.lower() == 'true':
        options_eval = options
        options_eval['unroll_steps'] = 1
        options_eval['char_cnn']['n_characters'] = 262
        lm = LanguageModel(options=options_eval, training=False)
        dataset = create_elmo_dataset(batch_size=256, data_file_path=args.eval_url)
        do_eval()

def do_train(dataset=None, network=None):
    steps_per_epoch = dataset.get_dataset_size()
    callback_size = args.sink_size
    actual_epoch_num = int(args.epoch_num * steps_per_epoch / callback_size)

    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="elmo", directory=args.train_url, config=config_ck)
    callback = [LossCallBack(1), TimeMonitor(1), ckpoint_cb]

    opt = nn.Adagrad(lm.trainable_params(), learning_rate=args.lr)
    update_scale_cell = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
    train_one_step = ElmoTrainOnestepWithLoss(lm, opt, update_scale_cell)

    model = Model(train_one_step)
    model.train(actual_epoch_num, dataset, callbacks=callback, dataset_sink_mode=True, sink_size=args.sink_size)
    # profiler.analyse()

def do_eval(dataset=None, network=None, load_checkpoint_path="", options="", batch_size=256):
    if load_checkpoint_path == "":
        raise ValueError("")
    lm = netword(options=options, training=False)    
    lm.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)
    model = Model(network)
    callback = [LossCallBack(1), TimeMonitor(1)]
   
    total_loss = []
    for  batch_no, data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        tokens_characters, tokens_characters_reverse, next_token_id, next_token_id_reverse = input_data
        time_begin = time.time()
        loss = model.predict(tokens_characters, tokens_characters_reverse, next_token_id, next_token_id_reverse)
        total_loss.append(loss.asnumpy())
        time_end = time.time()
        evaluate_times.append(time_end - time_begin)
    avg_loss = np.mean(total_loss)
    avg_preplexity = np.exp(avg_loss)
    print("==============================================================")
    print("(w/o first and last) elapsed time: {}, per step time : {}".format(
        sum(evaluate_times), sum(evaluate_times)/len(evaluate_times)))
    print("avg_loss:{}, avg_preplexity:{}".format(avg_loss, avg_preplexity))
    print("==============================================================")
if __name__=='__main__':
    run_pretrain()
