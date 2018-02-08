#!/usr/lib/env python
# coding:utf-8
"""
  description:
    该模块将所有的功能模块进行了整合，实现了语音情绪识别系统。
    整合的模块包含：
      1. 不同数据库的统一处理
      2. 将数据库转换为语谱图
      3. 对语谱图进行切分
      4. 数据源格式统一处理
      5. 进行数据模型的训练和识别
"""
import json
from .data_helper import *
from .wave_data_preprocess import *
from .run_all_model import *
import logging
from .time_w import time_w
from .basic_model import basic_model
from .run_cnn_rnn_svm import *

"""
配置日志模式
"""
def configure_logger():
    logger = logging.getLogger("rnn_embedding")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("../SVM-CNN-Experiment-Result/{}_embedding.log".format(str(int(time.time()))))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s-%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("{:=>80}".format(""))
    return logger

"""
  数据预处理,输入参数，将原始数据库生成格式化名字后wav,png,split_png
"""
def data_preprocess():

    """load data preprocess params"""
    with open("preprocess_paramters.json") as p:
        preprocess_parameters_arr = json.loads(p.read())
    logger.info(preprocess_parameters_arr)
    logger.info( "数据预处理参数读取完成，要处理{}组预处理参数".format(len(preprocess_parameters_arr)))
    n = 1
    for preprocess_parameters in preprocess_parameters_arr:
        with time_w("数据第{}部分预处理完成".format(n)) as t:
            # 将原始数据库生成格式化名字后wav, png, split_png
            raw_data_preprocess(preprocess_parameters=preprocess_parameters,refresh=True)
        n = n + 1
    logger.info("数据预处理全部完成,共计处理完成{}项".format(n-1))
    logger.info("{:=>80}".format(""))

"""
  读取数据进行训练
"""
def reading_and_training():
    """
    {
    "n_steps": 128,
    "n_input": 128,
    "n_units": 128,
    "n_classes": 6,
    "batch_size": 100,
    "n_epochs": 1,
    "learning_rate": 0.001,
    "display_step": 1,
    "run_mode": "/cpu:0",
    "split_png_data": "/Users/jw/Desktop/audio_data/1484131952_256_0.5/split_png_data/CASIA",
    "cell_type": "LSTM",
    "n_layers":1,
    "keep_prob": 0.9,
    "n_weights": 128
    }
    """

    """load params"""
    with open("paramters.json") as p:
        params_array = json.loads(p.read())
    logger.info(params_array)
    logger.info("模型训练参数读取完成,共有参数{}组".format(len(params_array)))
    n = 1

    """
    实现 mfcc 窗口大小和预加重系数的生成判断实验
    preemp = [x/10 for x in range(1, 10, 1)]
    winlen = [x/1000 for x in range(10,55,5)]
    params = params_array[0]
    for i in range(len(preemp)):
        preemp_tmp = preemp[i]
        for j in range(len(winlen)):
            winlen_tmp = winlen[j]
            params["preemph"] = preemp_tmp
            params["winlen"] = winlen_tmp
            with time_w("svm - preemp:{} winlen: {} 训练完成".format(preemp_tmp,winlen))as t:
                run_cnn_svm_eml_model(params)
    """

    for params in params_array:
        logger.info("第{}组参数训开始".format(n))
        logger.info("params:{}".format(params))
        # with time_w("cnn_svm训练完成")as t:
            # run_cnn_svm_eml_model(params)
        with time_w("cnn_rnn_svm训练完成")as t:
            # run_cnn(params)
            run_rnn(params)
            # run_cnn_rnn(params)
        #
        # """load data sets"""
        # with time_w("模型输入数据读取完成,") as t:
        #     data_helper = read_data_sets(params["split_png_data"], num_class=params["n_classes"],one_hot=True)
        #
        # """initialize model and train and save the model"""
        # with time_w("第{}组参数模型训练完成".format(n)) as t:
        #     basic_model(params).train(data_helper)
        n = n + 1
    logger.info("全部参数已经训练完成")

if __name__ == '__main__':

    logger = configure_logger()
    # with time_w("数据预处理") as t:
    #     data_preprocess()
    with time_w("数据训练") as t:
        reading_and_training()