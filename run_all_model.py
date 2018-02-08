#coding=utf-8
# ==============================================================================
# Copyright David Macedo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import pandas as pd
# from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
import time
from math import floor
import logging

from .input_data import *
from .elm import GenELMClassifier
from .random_layer import MLPRandomLayer

logger = logging.getLogger("rnn_embedding.run_all_model")

"""
"NUMBER_OF_FEATURES" : 128,
        "BATCH_SIZE" : 100,
        "NUMBER_OF_EPOCHS" : 5,
        "NUMBER_OF_EXPERIMENTS" : 5,
        "IMAGES_WIDTH" : 256,
        "IMAGES_HEIGHT" : 256,
        "EMOTION_CLASS" : 7,
        "split_png_dir":"../audio_data/1484131952_512_0.7/split_png_data/Emo"
"""
"""
 cnn cnn-svm svm 层数是固定的，但是每层的大小和迭代次数是可变的，参数输入在 parameters.json 文件中
"""
def run_cnn_svm_eml_model(params):

    NUMBER_OF_FEATURES = params['NUMBER_OF_FEATURES']
    BATCH_SIZE =  params['BATCH_SIZE']
    NUMBER_OF_EPOCHS =  params['NUMBER_OF_EPOCHS']
    NUMBER_OF_EXPERIMENTS =  params['NUMBER_OF_EXPERIMENTS']
    IMAGES_WIDTH =  params['IMAGES_WIDTH']
    IMAGES_HEIGHT =  params['IMAGES_HEIGHT']
    EMOTION_CLASS =  params['EMOTION_CLASS']
    DATA_TYPES = params['DATA_TYPES']
    mnist = read_data_sets(params['split_png_dir'], num_class=EMOTION_CLASS, data_type=DATA_TYPES, one_hot=True, logfbank_width=IMAGES_WIDTH, params=params)
    BATCHES_IN_EPOCH = len(mnist.train.images) // BATCH_SIZE

    converter = np.array([x for x in range(EMOTION_CLASS)])
    svm_results = {
        "LK-SVM-ACCU":0, "GK-SVM-ACCU":0, "SOFTMAX-ACCU":0,"LK-SVM-TIME":0, "GK-SVM-TIME":0,"SOFTMAX-TIME":0,
        "LK-SVM-0-ACCU": 0, "LK-SVM-1-ACCU": 0, "LK-SVM-2-ACCU": 0, "LK-SVM-3-ACCU": 0, "LK-SVM-4-ACCU": 0,"LK-SVM-5-ACCU": 0, "LK-SVM-6-ACCU": 0,
        "GK-SVM-0-ACCU": 0, "GK-SVM-1-ACCU": 0, "GK-SVM-2-ACCU": 0, "GK-SVM-3-ACCU": 0, "GK-SVM-4-ACCU": 0,"GK-SVM-5-ACCU": 0, "GK-SVM-6-ACCU": 0,
        "SOFTMAX-0-ACCU": 0, "SOFTMAX-1-ACCU": 0, "SOFTMAX-2-ACCU": 0, "SOFTMAX-3-ACCU": 0, "SOFTMAX-4-ACCU": 0,"SOFTMAX-5-ACCU": 0, "SOFTMAX-6-ACCU": 0,
    }
    experiment_results = {
        "1024HL-ELM-ACCU":0, "4096HL-ELM-ACCU":0, "ConvNet-ACCU":0, "ConvNetSVM-ACCU":0,
        "1024HL-ELM-TIME":0, "4096HL-ELM-TIME":0, "ConvNet-TIME":0, "ConvNetSVM-TIME":0,
        "ConvNet-0-ACCU": 0, "ConvNet-1-ACCU": 0, "ConvNet-2-ACCU": 0, "ConvNet-3-ACCU": 0, "ConvNet-4-ACCU": 0,
        "ConvNet-5-ACCU": 0, "ConvNet-6-ACCU": 0,
        "ConvNetSVM-0-ACCU": 0, "ConvNetSVM-1-ACCU": 0, "ConvNetSVM-2-ACCU": 0, "ConvNetSVM-3-ACCU": 0, "ConvNetSVM-4-ACCU": 0,
        "ConvNetSVM-5-ACCU": 0, "ConvNetSVM-6-ACCU": 0,
    }
    TRAIN_SIZE = BATCHES_IN_EPOCH * BATCH_SIZE

    train_features = np.zeros((TRAIN_SIZE, IMAGES_HEIGHT * IMAGES_WIDTH), dtype=float)
    train_labels = np.zeros(TRAIN_SIZE, dtype=int)

    train_features_cnn = np.zeros((TRAIN_SIZE, NUMBER_OF_FEATURES), dtype=float)
    train_labels_cnn = np.zeros(TRAIN_SIZE, dtype=int)

    def split_features_labels_by_classes(features, labels, n_class):
        features_group={}
        labels_group={}
        for i in range(n_class):
            for j in range(features.shape[0]):
                features_tmp = features_group.get("{}".format(i))
                if features_tmp is None:
                    features_group.setdefault("{}".format(i),[])
                else:
                    features_tmp.append(features[j])
                labels_tmp = labels_group.get("{}".format(i))
                if labels_tmp is None:
                    labels_group.setdefault("{}".format(i), [])
                else:
                    labels_group.append(labels[j])
        return features_group,labels_group


    def print_debug(ndarrayinput, stringinput):
        print(("\n"+stringinput))
        print((ndarrayinput.shape))
        print((type(ndarrayinput)))
        print((np.mean(ndarrayinput)))
        print(ndarrayinput)

    # 产生一个均值分布初始化的tensor变量，[filter_height, filter_width, in_channels, out_channels]
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 创建一个新的变量，shape (out_channels,)
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 进行卷积,填充的方式是'SAME' 卷积前后的维度大小是相同的，移动的步长为1
    def conv2d(x, W):
        """
        :param x: [batch, in_height, in_width, in_channels]
        :param W: [filter_height, filter_width, in_channels, out_channels]
        :return:
        """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        """
          Args:
        value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
          type `tf.float32`.
        ksize: A list of ints that has length >= 4.  The size of the window for
          each dimension of the input tensor.
        strides: A list of ints that has length >= 4.  The stride of the sliding
          window for each dimension of the input tensor.
        padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def SVM(krnl):
        # logger.info("\n###############################\n {}  Kernel SVM Train/Test\n###############################".format(krnl))

        for i in range(BATCHES_IN_EPOCH):
            train_batch = mnist.train.next_batch(BATCH_SIZE)
            features_batch = train_batch[0]
            labels_batch = train_batch[1]
            for j in range(BATCH_SIZE):
                for k in range(IMAGES_HEIGHT*IMAGES_WIDTH):
                    train_features[BATCH_SIZE * i + j, k] = features_batch[j, k]
                train_labels[BATCH_SIZE * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

        # print_debug(train_features, "train_features")
        # print_debug(train_labels, "train_labels")


        initial_time = time.time()

        clf = svm.SVC(kernel=krnl)
        """
               支持向量机模型：模型的输入是
               features:shape (n_samples, n_features)
               labels: shape(n_samples,)，label是分类值
           """
        clf.fit(train_features, train_labels)

        training_time = time.time()-initial_time
        # logger.info("Training Time = {}".format(training_time))

        test_features = mnist.test.images
        test_labels = np.zeros(len(mnist.test.images), dtype=int)
        for j in range(len(mnist.test.images)):
            test_labels[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

            # print_debug(test_features, "test_features")
            # print_debug(test_labels, "test_labels")
        all_accuracy = []
        accuracy = clf.score(test_features, test_labels)
        test_time = time.time() - (training_time + initial_time)
        # logger.info("Test Time = {}".format(test_time))
        # logger.info("{} SVM accuracy ={}".format(krnl, accuracy))

        for i in range(EMOTION_CLASS):
            test = mnist.all_test[i]
            test_features = test.images
            test_labels = np.zeros(len(test.images), dtype=int)
            for j in range(len(test.images)):
                test_labels[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

                # print_debug(test_features, "test_features")
                # print_debug(test_labels, "test_labels")
            accuracy_tmp = clf.score(test_features, test_labels)
            test_time = time.time() - (training_time + initial_time)
            # logger.info("class: {}Test Time = {}".format(x,test_time))
            # logger.info("{} class:{} SVM accuracy ={}".format(krnl, x, accuracy))
            all_accuracy.append(accuracy_tmp)

        all_accuracy.append(accuracy)
        return accuracy, training_time, all_accuracy

    def ELM(nodes):
        # print("\n#########################\n", nodes, "Hidden Layer Nodes ELM Train/Test\n#########################")

        for i in range(BATCHES_IN_EPOCH):
            train_batch = mnist.train.next_batch(BATCH_SIZE)
            features_batch = train_batch[0]
            labels_batch = train_batch[1]
            for j in range(BATCH_SIZE):
                for k in range(IMAGES_HEIGHT * IMAGES_WIDTH):
                    train_features[BATCH_SIZE * i + j, k] = features_batch[j, k]
                train_labels[BATCH_SIZE * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

                #    print_debug(train_features, "train_features")
                #    print_debug(train_labels, "train_labels")

        initial_time = time.time()

        srhl_tanh = MLPRandomLayer(n_hidden=nodes, activation_func="tanh")
        clf = GenELMClassifier(hidden_layer=srhl_tanh)
        clf.fit(train_features, train_labels)
        training_time = time.time() - initial_time
        # print("\nTraining Time = ", training_time)

        test_labels = np.zeros(len(mnist.test.images), dtype=int)
        for j in range(len(mnist.test.images)):
            test_labels[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

            #    print_debug(test_features, "test_features")
            #    print_debug(test_labels, "test_labels")

        test_features = mnist.test.images
        accuracy = clf.score(test_features, test_labels)
        #    test_time = time.time() - (training_time + initial_time)
        #    print("\nTest Time = ", test_time)

        # print("\n", nodes, "hidden layer nodes ELM accuracy =", accuracy)
        """
        分别针对不同的心情计算准确率
        """
        all_accuracy = []
        for i in range(EMOTION_CLASS):
            test = mnist.all_test[i]
            test_labels = np.zeros(len(test.images), dtype=int)
            for j in range(len(test.images)):
                test_labels[j] = np.sum(np.multiply(converter, test.labels[j, :]))

                #    print_debug(test_features, "test_features")
                #    print_debug(test_labels, "test_labels")

            test_features = test.images
            accuracy_tmp = clf.score(test_features, test_labels)
            all_accuracy.append(accuracy_tmp)

        all_accuracy.append(accuracy)
        return accuracy, training_time, all_accuracy

    def ConvNet(number_of_training_epochs):
        # logger.info("\n#########################\nConvNet Train/Test\n#########################")
        initial_time = time.time()

        """
            显示中间操作的维度
        """
        # batch_shape = mnist.train.next_batch(BATCH_SIZE)
        # feed_dic = {x: batch_shape[0], y_: batch_shape[1], keep_prob: 1.0}
        # logger.info("h_conv1 {}".format(h_conv1.eval(feed_dict=feed_dic).shape))
        # logger.info("h_pool1. {}".format(h_pool1.eval(feed_dict=feed_dic).shape))
        # logger.info("h_conv2 {}".format( h_conv2.eval(feed_dict=feed_dic).shape))
        # logger.info("h_pool2 {}".format( h_pool2.eval(feed_dict=feed_dic).shape))
        # logger.info("h_pool2_flat {}".format(h_pool2_flat.eval(feed_dict=feed_dic).shape))
        # logger.info("h_fc1 {}".format(h_fc1.eval(feed_dict=feed_dic).shape))
        # logger.info("y_conv {}".format( y_conv.eval(feed_dict=feed_dic).shape))
        # logger.info("y_ {}".format( y_.shape))

        for i in range(number_of_training_epochs * BATCHES_IN_EPOCH):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if i%BATCHES_IN_EPOCH == 0:
                train_accuracy = model_accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                # logger.info("Epoch {},Training Accuracy {} ".format(int(i/BATCHES_IN_EPOCH), train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        training_time = time.time()-initial_time
        # print(("\nTraining Time = ", training_time))

        accuracy = model_accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    #    test_time = time.time() - (training_time + initial_time)
    #    print("\nTest Time = ", test_time)

        # print(("\nConvNet accuracy =", accuracy))

        """
             分别针对不同的心情计算准确率
             """
        all_accuracy = []
        for i in range(EMOTION_CLASS):
            test = mnist.all_test[i]
            accuracy_tmp = model_accuracy.eval(feed_dict={x: test.images, y_: test.labels, keep_prob: 1.0})
            all_accuracy.append(accuracy_tmp)
        all_accuracy.append(accuracy)

        return accuracy, training_time, all_accuracy

    def ConvNetSVM():
        # print("\n#########################\nConvNetSVM Train/Test\n#########################")
        initial_time = time.time()

        for i in range(BATCHES_IN_EPOCH):
            train_batch = mnist.train.next_batch(BATCH_SIZE)
            features_batch = h_fc1.eval(feed_dict={x: train_batch[0]})
            # print(features_batch)
            labels_batch = train_batch[1]
            for j in range(BATCH_SIZE):
                for k in range(NUMBER_OF_FEATURES):
                    train_features_cnn[BATCH_SIZE * i + j, k] = features_batch[j, k]
                train_labels_cnn[BATCH_SIZE * i + j] = np.sum(np.multiply(converter, labels_batch[j, :]))

    #    print_debug(train_features_cnn, "train_features_cnn")
    #    print_debug(train_labels_cnn, "train_labels_cnn")

        clf = svm.SVC()
        clf.fit(train_features_cnn, train_labels_cnn)
        training_time = time.time()-initial_time
        # print(("\nTraining Time = ", training_time))

        test_labels_cnn = np.zeros(len(mnist.test.images), dtype=int)
        test_features_cnn = h_fc1.eval(feed_dict={x: mnist.test.images})
        for j in range(len(mnist.test.images)):
            test_labels_cnn[j] = np.sum(np.multiply(converter, mnist.test.labels[j, :]))

            #    print_debug(test_features_cnn, "test_features_cnn")
            #    print_debug(test_labels_cnn, "train_labels_cnn")

        accuracy = clf.score(test_features_cnn, test_labels_cnn)
    #    test_time = time.time() - (training_time + initial_time)
    #    print("\nTest Time = ", test_time)

        # print(("\nConvNetSVM accuracy =", accuracy))
        """
            分别针对不同的心情计算准确率
                    """
        all_accuracy = []
        for i in range(EMOTION_CLASS):
            test = mnist.all_test[i]
            test_labels_cnn = np.zeros(len(test.images), dtype=int)
            test_features_cnn = h_fc1.eval(feed_dict={x: test.images})
            for j in range(len(test.images)):
                test_labels_cnn[j] = np.sum(np.multiply(converter, test.labels[j, :]))

                #    print_debug(test_features_cnn, "test_features_cnn")
                #    print_debug(test_labels_cnn, "train_labels_cnn")

            accuracy_tmp = clf.score(test_features_cnn, test_labels_cnn)
            all_accuracy.append(accuracy_tmp)

        all_accuracy.append(accuracy)
        return accuracy, training_time, all_accuracy


    def softmax(number_of_training_epochs):
        initial_time = time.time()

        for i in range(number_of_training_epochs * BATCHES_IN_EPOCH):
            batch = mnist.train.next_batch(BATCH_SIZE)
            # if i % BATCHES_IN_EPOCH == 0:
                # train_accuracy = model_accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                # logger.info("Epoch {},Training Accuracy {} ".format(int(i/BATCHES_IN_EPOCH), train_accuracy))
            y_softmax_train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
        training_time = time.time() - initial_time
        # print(("\nTraining Time = ", training_time))

        accuracy = y_softmax_model_accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

        all_accuracy = []
        for i in range(EMOTION_CLASS):
            test = mnist.all_test[i]
            accuracy_tmp = y_softmax_model_accuracy.eval(feed_dict={x: test.images, y_: test.labels, keep_prob: 1.0})
            all_accuracy.append(accuracy_tmp)
        all_accuracy.append(accuracy)

        return accuracy, training_time, all_accuracy
# print("\n#########################\nStarting\n#########################\n")


    sess = tf.InteractiveSession()


    # print("\n#########################\nBuilding ConvNet\n#########################")

    x = tf.placeholder(tf.float32, shape=[None, IMAGES_HEIGHT*IMAGES_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, EMOTION_CLASS])

    x_image = tf.reshape(x, [-1,IMAGES_HEIGHT,IMAGES_WIDTH,1])
    """
       softmax 单层
       """
    w_softmax = weight_variable([IMAGES_WIDTH * IMAGES_WIDTH, EMOTION_CLASS])
    b_softmax = bias_variable([EMOTION_CLASS])

    y_softmax = tf.nn.softmax(tf.matmul(x, w_softmax) + b_softmax)

    y_softmax_cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_softmax), reduction_indices=[1]))
    y_softmax_train_step = tf.train.AdamOptimizer(1e-4).minimize(y_softmax_cross_entropy)

    y_softmax_correct_prediction = tf.equal(tf.argmax(y_softmax, 1), tf.argmax(y_, 1))
    y_softmax_model_accuracy = tf.reduce_mean(tf.cast(y_softmax_correct_prediction, tf.float32))



    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, floor(IMAGES_HEIGHT / 4)  * floor(IMAGES_WIDTH / 4) * 32])

    W_fc1 = weight_variable([floor(IMAGES_HEIGHT / 4)  * floor(IMAGES_WIDTH / 4)  * 32, NUMBER_OF_FEATURES])
    b_fc1 = bias_variable([NUMBER_OF_FEATURES])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([NUMBER_OF_FEATURES, EMOTION_CLASS])
    b_fc2 = bias_variable([EMOTION_CLASS])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    model_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # print("\n#########################\nExecuting Experiments\n#########################")

    dataframe_svm = pd.DataFrame()
    dataframe_results = pd.DataFrame()


    svm_results["LK-SVM-ACCU"], svm_results["LK-SVM-TIME"],all_accu_lr = SVM("linear")
    for i in range(EMOTION_CLASS):
        svm_results["LK-SVM-{}-ACCU".format(i)] = all_accu_lr[i]

    svm_results["GK-SVM-ACCU"], svm_results["GK-SVM-TIME"],all_accu_rbf = SVM("rbf")
    for i in range(EMOTION_CLASS):
        svm_results["GK-SVM-{}-ACCU".format(i)] = all_accu_rbf[i]

    sess.run(tf.initialize_all_variables())
    svm_results["SOFTMAX-ACCU"], svm_results["SOFTMAX-TIME"], all_accu_softmax = softmax(number_of_training_epochs=NUMBER_OF_EPOCHS)
    for i in range(EMOTION_CLASS):
        svm_results["SOFTMAX-{}-ACCU".format(i)] = all_accu_softmax[i]

    dataframe_svm = dataframe_svm.append(svm_results, ignore_index=True)
    title = ["LK-SVM-ACCU", "GK-SVM-ACCU","SOFTMAX-ACCU","LK-SVM-TIME", "GK-SVM-TIME","SOFTMAX-TIME"]
    for i in range(EMOTION_CLASS):
        title.append("LK-SVM-{}-ACCU".format(i))
        title.append("GK-SVM-{}-ACCU".format(i))
        title.append("SOFTMAX-{}-ACCU".format(i))
    dataframe_svm = dataframe_svm[title]




    # for index in range(NUMBER_OF_EXPERIMENTS):
    #     # print(("\n#########################\nExperiment", index+1, "of", NUMBER_OF_EXPERIMENTS, "\n#########################"))
    #     # experiment_results["1024HL-ELM-ACCU"], experiment_results["1024HL-ELM-TIME"] = ELM(1024)
    #     # experiment_results["4096HL-ELM-ACCU"], experiment_results["4096HL-ELM-TIME"] = ELM(4096)
    #     sess.run(tf.initialize_all_variables())
    #     experiment_results["ConvNet-ACCU"], experiment_results["ConvNet-TIME"],conv_all_accuracy = ConvNet(NUMBER_OF_EPOCHS)
    #     for i in range(EMOTION_CLASS):
    #         experiment_results["ConvNet-{}-ACCU".format(i)] = conv_all_accuracy[i]
    #
    #     experiment_results["ConvNetSVM-ACCU"], experiment_results["ConvNetSVM-TIME"],conv_svm_all_accuarcy = ConvNetSVM()
    #     for i in range(EMOTION_CLASS):
    #         experiment_results["ConvNetSVM-{}-ACCU".format(i)] = conv_svm_all_accuarcy[i]
    #     dataframe_results = dataframe_results.append(experiment_results, ignore_index=True)
    #
    # # dataframe_results = dataframe_results[["1024HL-ELM-ACCU", "4096HL-ELM-ACCU", "ConvNet-ACCU", "ConvNetSVM-ACCU",
    # #                        "1024HL-ELM-TIME", "4096HL-ELM-TIME", "ConvNet-TIME", "ConvNetSVM-TIME",]]
    # # dataframe_results = dataframe_results[["ConvNet-ACCU","ConvNetSVM-ACCU","ConvNet-TIME", "ConvNetSVM-TIME",]]
    # result_title = ["ConvNet-ACCU","ConvNetSVM-ACCU","ConvNet-TIME", "ConvNetSVM-TIME"]
    # for i in range(EMOTION_CLASS):
    #     result_title.append("ConvNet-{}-ACCU".format(i))
    #     result_title.append("ConvNetSVM-{}-ACCU".format(i))
    # dataframe_results = dataframe_results[result_title]
    # dataframe_results = dataframe_results[["ConvNet-ACCU", "ConvNet-TIME",]]
    # dataframe_results = dataframe_results[["ConvNetSVM-ACCU", "ConvNet-TIME", ]]
    logger.info("\n#########################\nPrinting Results\n#########################\n")
    logger.info("\n{}\n".format(params))
    logger.info("\n{}".format(dataframe_svm))
    # logger.info("\n{}\n".format(dataframe_results))
    logger.info("\n{}".format(dataframe_svm.describe(include='all')))

    logger.info("\n#########################\nStoping\n#########################\n")

    sess.close()


"""

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
mpl.use('pdf')

dataframe_results = dataframe_results[["1024HL-ELM-ACCU", "4096HL-ELM-ACCU", "ConvNet-ACCU", "ConvNetSVM-ACCU",]]

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
#plt.tight_layout()

width = 3.487
height = width / 1.618

fig=plt.figure()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

ax = dataframe_results.plot.box(figsize=(width, height))
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_title("Title")

plt.savefig("df_global.pdf")
"""