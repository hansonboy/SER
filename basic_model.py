#!/usr/lib/env python
# coding:utf-8
"""
  description:
    实现了深度学习模型的定义及训练
"""
import os
import os.path
import time
import logging
import numpy as np
import tensorflow as tf
from .layer_normalized_lstm_cell import LayerNormalizedLSTMCell
from .custom_cell import CustomCell
from tensorflow.contrib import rnn

logger = logging.getLogger("rnn_embedding.basic_model")
"""
summary:
  basic_model 是一个框架，提供训练模型的数据参数就会自动生成结果，并且将结果保存到文件中，同时生成图的
  tensorboard 信息，后续的继承需要重写相应的方法
"""


class basic_model(object):
    """
    {
      "n_layers":2,
      "n_steps": 128,
      "n_input": 128,
      "n_units": 128,
      "n_classes": 6,
      "batch_size": 100,
      "learning_rate": 0.001,
      "display_step": 20,
      "run_mode":"/cpu:0",
      "split_png_data": "/Users/jw/Desktop/audio_data/1483530666/split_png_data/CASIA"
      "cell_type":"GRU",
      "n_weights":128
    }
    """

    def cell(self):
        cell = 0
        return cell

    def __init__(self, params):
        """
        :param params:是一个字典，包含num_steps,state_size,batch_size,num_classes,learning_rate
        """

        self.params = params
        n_steps = params["n_steps"]
        n_input = params["n_input"]
        n_units = params["n_units"]
        n_classes = params["n_classes"]
        batch_size = params["batch_size"]
        # "n_steps": 128,
        # "n_input": 128,
        # "n_units": 128,
        # "n_classes": 6,
        # "batch_size": 100,
        # "n_epochs": 50,
        # "learning_rate": 0.0003,
        # "display_step": 1,
        # "run_mode": "/cpu:0",
        # "split_png_data": "/Users/jw/Desktop/audio_data/1484131952_256_0.5/split_png_data/CASIA"

        tf.reset_default_graph()
        with tf.get_default_graph().as_default():
            with tf.name_scope("placeholder"):
                self.x = tf.placeholder("float", [None, n_steps*n_input],
                                        name="x")
                self.input = tf.reshape(self.x, [-1, n_steps, n_input])
                self.y = tf.placeholder("float", [None, n_classes], name="y")
                self.keep_prob = tf.placeholder(tf.float32)

            with tf.variable_scope("softmax"):
                weights = tf.Variable(tf.random_normal([n_units, n_classes]),
                                      name='weights')
                biases = tf.Variable(tf.random_normal([n_classes]), name='biases')

            # x = tf.transpose(self.x, [1, 0, 2])
            # x = tf.reshape(x, [-1, n_input])
            # x = tf.split(0, n_steps, x)
            sequence_length = np.zeros([batch_size], dtype=int)
            sequence_length += n_steps

            state_size = self.params["n_units"]
            num_layers = self.params["n_layers"]
            cell_type = self.params["cell_type"]
            num_weights_for_custom_cell = self.params.get("n_weights")

            if cell_type == 'Custom':
                cell = CustomCell(state_size, num_weights_for_custom_cell)
                cell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.LSTMCell(state_size, state_is_tuple=True),
                                                            input_keep_prob=self.keep_prob)
                                         for _ in range(num_layers)])
            elif cell_type == 'GRU':
                cell = rnn.GRUCell(state_size)
            elif cell_type == 'LSTM':
                cell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.LSTMCell(state_size, state_is_tuple=True),
                                                            input_keep_prob=self.keep_prob)
                                         for _ in range(num_layers)])
            elif cell_type == 'LN_LSTM':
                cell = LayerNormalizedLSTMCell(state_size)
            else:
                cell = rnn.BasicRNNCell(state_size)

            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            self.init_state = cell.zero_state(batch_size, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.input, dtype=tf.float32,
                                                          initial_state=self.init_state,
                                                          sequence_length=sequence_length)
            # outputs's shape [batch_size, time_step, state_size]
            outputs = tf.transpose(outputs, [1, 0, 2])

            pred = tf.matmul(outputs[-1], weights) + biases
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']) \
                .minimize(self.cost)

            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            tf.summary.scalar("cost", self.cost)
            tf.summary.scalar("accuracy", self.accuracy)
            self.merge_summary_op = tf.summary.merge_all()

            logger.info("模型构建完毕")

    """train the network"""
    def train(self, data_helper):
        display_step = self.params["display_step"]
        n_steps = self.params["n_steps"]
        n_input = self.params["n_input"]
        batch_size = self.params["batch_size"]
        batches_in_one_epoch = len(data_helper.train.images)//batch_size
        with tf.Session() as sess:
            """initial variables"""
            sess.run(tf.global_variables_initializer())

            """create a directory for saving model files"""
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            logs_train = os.path.join(out_dir, "logs_train")
            logs_test = os.path.join(out_dir, "logs_test")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            """define writer"""
            writer_train = tf.summary.FileWriter(logs_train, graph=tf.get_default_graph())
            writer_test = tf.summary.FileWriter(logs_test, graph=tf.get_default_graph())

            """save the model into checkpoints"""
            saver = tf.train.Saver(tf.global_variables())

            for i in range(self.params["n_epochs"] * batches_in_one_epoch):
                logger.info("epoch {} 开始".format(i//batches_in_one_epoch))

                batch = data_helper.train.next_batch(self.params["batch_size"])
                self.optimizer.run(feed_dict={
                                                self.x: batch[0],
                                                self.y: batch[1],
                                                self.keep_prob: 0.5
                                               })
                if i % display_step == 0:
                    train_loss, train_acc, train_merge_summary = sess.run([self.cost,
                                                                          self.accuracy,
                                                                          self.merge_summary_op],
                                                                          feed_dict={self.x: batch[0],
                                                                                     self.y: batch[1],
                                                                                     self.keep_prob: 1.0
                                                                                     }
                                                                          )
                    """collect the results"""
                    logger.info("Iter:{:0>6} Loss:{:.6f} Accuracy:{:.6f}"
                                .format(i, train_loss, train_acc))
                    """save the graph and tensors into tensorboard"""
                    writer_train.add_summary(train_merge_summary, global_step=i)

                    total_test_loss = 0
                    total_test_accuracy = 0
                    for i in range(len(data_helper.test.images)//batch_size):

                        test_loss, test_acc, test_merge_summary = sess.run([self.cost, self.accuracy,
                                                                              self.merge_summary_op],
                                                                              feed_dict={self.x: data_helper.test.images.next,
                                                                                         self.y: data_helper.test.labels,
                                                                                         self.keep_prob: 1.0
                                                                                         }
                                                                           )

                    """collect the results"""
                    logger.info("Iter:{:0>6} Loss:{:.6f} Accuracy:{:.6f}"
                                .format(i, test_loss, test_acc))
                    """save the graph and tensors into tensorboard"""
                    writer_test.add_summary(train_merge_summary, global_step=i)

            saver.save(sess, checkpoint_prefix)


if __name__ == '__main__':
    # sess = tf.Session()

    # x = np.array([[1,2,3],[4,5,6]])
    # data = tf.Variable(x)
    # d = tf.tile(data, [1,10])
    # tf.reshape(d, [2,10,3])
    # print tf.shape(d)
    # sess.run(tf.initialize_all_variables())
    # print sess.run(d)
    print([1 for _ in range(3)])
    x = [[1, 2, 3], [4, 5, 6]]
    print(x[-1])
