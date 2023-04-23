# elmo_mindspore

## 目录

[TOC]


## 模型架构

## 概述

ELMo（Embeddings from Language Model）模型出自NAACL2018会议上的论文[“Deep contextualized word representations”](http://arxiv.org/abs/1802.05365)，论文中提出了一个新的词表征方法。常用的词嵌入方法如：Word2Vec、GloVe等忽略了词的上下文信息。ELMo同时对（1）单词在不同语境下的使用（如语法和语义）；（2）这些用法如何在不同的语言环境中变化(例如，建模一词多义)。该模型是在大型文本语料库上进行预训练的。模型建立在biLMs（双向语言模型）基础上，可以很容易地添加到现有的模型中，并显著改善一系列具有挑战性的自然语言处理问题(包括问答、文本蕴涵和情感分析)的技术水平。

## 模型架构

## 数据集

Elmo模型的训练和评估需要准备：

- 词汇表
- 训练数据集
- 测试数据集
- 配置文件

1. ELMo 模型训练数据来自 [1 billion word benchmark](http://www.statmt.org/lm-benchmark/)，训练集及测试集可以通过运行`download_dataset.sh`进行下载。

```shell
sh script/download_dataset.sh
```

2. 词汇表中每个token一行，词汇表的前三行必须为`<S>``</S>``<UNK>`，词汇表文件下载：[vocab-2016-09-10.txt](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt)。
3. 创建配置文件，如[options.json](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_tf_checkpoint/options.json) ，可根据需求修改超参。



## 环境要求

- 硬件（Ascend处理器）
    - 准备Ascend处理器搭建硬件环境。如需试用昇腾处理器，请发送[申请表](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)至ascend@huawei.com，申请通过后，即可获得资源。
- 框架版本
    - [MindSpore1.1+](https://gitee.com/mindspore/mindspore)
    - python 3.7.5
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 快速入门

### 在Ascend上运行

1. 将训练数据转换为mindrecord格式

```shell
# 运行以下脚本
bash script/convert_dataset.sh
```

2. 运行`train.py` 脚本进行模型训练

```shell
# for example
python train.py --epoch_num=10
```

*暂不支持静态图模型及分布式训练*

## 脚本说明

### 脚本和样例代码

```shell
.
└─ELMo
  ├─README.md
  ├─elmo
  	├─data								
  		├─__init__.py					
  		├─dataset.py				    # 数据预处理
  		├─fields.py
  		├─instance.py
  		├─reader.py						# 数据转换
  		├─token_indexer.py
  		├─tokenizer.py
  		├─vocabulary.py					# 数据预处理
  	├─modules							
  		├─__init__.py 
  		├─embedding.py					# char encoder实现脚本
  		├─highway.py					# highway层实现脚本
  		├─loss.py						# 损失层实现脚本
  		├─lstm.py						# biLM层实现脚本
  		├─scalar_mix.py                 
  		├─time_distributed.py
  	├─nn
  		├─__init__.py
  		├─layers.py                      # 参数初始化脚本
  		├─rnn.py						 # GRU实现脚本
  		├─rnn_cell_warpper.py		     # GRU实现脚本
  		├─rnn_cells.py					 # GRU实现脚本
  	├─ops
  		├─sampled_softmax_loss.py		# 损失函数实现
  	├─utils
  		├─config.py
  		├─model.py						# 网络骨干编码
  ├─scripts
    ├─convert_dataset.sh             # 数据集预处理shell脚本
    ├─download_dataset.sh            # 下载数据集shell脚本
    ├─rank_table_2pcs.json           # 多卡配置文件
    ├─rank_table_8pcs.json           # 多卡配置文件
    └─run_distribution.sh            # Ascend上多机ELMo任务shell脚本
  ├─tests							 # 单元测试
  	├─test_char_encoder.py
  	├─test_data.py
  	├─test_elmo_lstm.py  	
  	├─test_highway.py
  	├─test_lm.py
  	├─test_rnn.py
    ├─test_rnn_cell.py
    ├─test_rnn_cell_warpper.py
	├─test_sampled_softmax_loss.py
  ├─.gitignore  
  ├─ElmoTrainOne.py  			# TrainOnestep脚本
  └─train.py                    # ELMo模型的训练脚本
```

## 脚本参数
```shell
用法：reader.py      [--vocab_path VOCAB_PATH]
                    [--options_path OPTIONS_PATH]
                    [--input_file INPUT_FILE]
                    [--output_file OUTPUT_FILE]

选项：
    --vocab_path                 ELMo模型训练的词汇表
    --options_path               配置文件的路径
    --input_file                 原始数据集路径
    --output_file                保存生成mindRecord格式数据的路径

用法：train.py     [--data_url DATA_URL]
          		  [--train_url TRAIN_URL] 
          		  [--device_target DEVICE_TARGET]
          		  [--lr, LR]  [--epoch_num EPOCH_NUM]
                                
选项:
    --data_url                      用于保存训练数据的mindRecord文件，如train.mindrecord
    --train_url                     保存生成微调检查点的路径
    --device_target                 代码实现设备，可选项为Ascend或CPU。默认为Ascend
    --lr                            学习率
    --epoch_num                     训练轮次总数
    
   
```