#!/usr/lib/env python
#coding:utf-8
import tensorflow as tf
from tensorflow.python.ops.tensor_array_ops import TensorArray as ta
from .input_data import *
from sklearn import svm
logger = logging.getLogger("rnn_embedding.run_cnn_rnn_svm")
# 添加全连接层
def add_fc_layer(inputs, in_size, out_size, activation_function=None):
    """
    入参：
        in_size：上一层神经元的个数。
        out_size：本层神经元的个数。
        input：None个长度为in_size一维向量（None行in_size列，shape(None,in_size)），即用一行来存储一个特征值输入。

    返回值：
        输入一个长度为in_size的一维向量，则返回一个长度为out_size的向量。
        输入batch_size个长度为in_size的一维向量，则返回batch_size个长度为out_size的向量。
    """
    # 保存上一层到本层之间的连接的权重
    weight = tf.Variable(tf.random_normal([in_size, out_size]))  # in_size行out_size列，shape:(in_size,out_size)
    # 保存本层的所有神经元的偏置
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 1行out_size列，shape:(1,out_size)
    XW_plus_b = tf.matmul(inputs, weight) + biases
    if activation_function is None:
        outputs = XW_plus_b
    else:
        outputs = activation_function(XW_plus_b)
    return outputs

def add_cnn_layers_updated_lunwen1(inputs, batch_norm=True):
    """
    入参：
        inputs：每张图片的大小是spectrum_height,spectrum_width。
                    shape:(batch_size,time_step,spectrum_height,spectrum_width)
    返回值：
        shape:(batch_size,time_step,feature_size)
    """

    def conv2d(x, W, b, strides=1):
        """
        x的shape：[batch, in_height, in_width, in_channels]
        W：就是filter，它的shape：[filter_height, filter_width, in_channels, out_channels]
        """
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, filter=W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, ksize=3, stride=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

    shapeList = inputs.get_shape().as_list()
    logger.info(
        "cnn_input：[batch,time,H,W] = {}".format(shapeList)
    )


    time_step = shapeList[1]
    spectrum_height = shapeList[2]
    spectrum_width = shapeList[3]

    weights = {
        # 9x9 conv, 1 input, 32 outputs
        'filter1': tf.Variable(tf.random_normal([7, 7, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'filter2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'filter3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'filter4': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'filter5': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    }

    biases = {
        'filter1': tf.Variable(tf.random_normal([32])),
        'filter2': tf.Variable(tf.random_normal([64])),
        'filter3': tf.Variable(tf.random_normal([128])),
        'filter4': tf.Variable(tf.random_normal([256])),
        'filter5': tf.Variable(tf.random_normal([512])),
    }

    logger.info(
        "params: [batch*time,H,W,out_channels]"
    )

    # 开始定义CNN
    in_x = tf.reshape(inputs, shape=[-1, spectrum_height, spectrum_width, 1])
    # Convolution Layer
    conv1 = conv2d(in_x, weights['filter1'], biases['filter1'])
    logger.info(
        'conv1: {}'.format(conv1.get_shape().as_list())
    )


    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    logger.info(
        'pooling1: {}'.format(conv1.get_shape().as_list())
    )

    if batch_norm:
        conv1 = batch_normalization(conv1, biases['filter1'].get_shape().as_list()[0])

    # Convolution Layer
    conv2 = conv2d(conv1, weights['filter2'], biases['filter2'])
    logger.info(
        'conv2: {}'.format(conv2.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    logger.info(
        'pooling2: {}'.format(conv2.get_shape().as_list())
    )

    if batch_norm:
        conv2 = batch_normalization(conv2, biases['filter2'].get_shape().as_list()[0])

    # Convolution Layer
    conv3 = conv2d(conv2, weights['filter3'], biases['filter3'])
    logger.info(
        'conv3: {}'.format(conv3.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3)
    logger.info(
        'pooling3: {}'.format(conv3.get_shape().as_list())
    )

    if batch_norm:
        conv3 = batch_normalization(conv3, biases['filter3'].get_shape().as_list()[0])

    # Convolution Layer
    conv4 = conv2d(conv3, weights['filter4'], biases['filter4'])
    logger.info(
        'conv4: {}'.format(conv4.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4)
    logger.info(
        'pooling4: {}'.format(conv4.get_shape().as_list())
    )

    if batch_norm:
        conv4 = batch_normalization(conv4, biases['filter4'].get_shape().as_list()[0])

    # Convolution Layer
    conv5 = conv2d(conv4, weights['filter5'], biases['filter5'])
    logger.info(
        'conv5: {}'.format(conv5.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5)
    logger.info(
        'pooling5: {}'.format(conv5.get_shape().as_list())
    )

    if batch_norm:
        conv5 = batch_normalization(conv5, biases['filter5'].get_shape().as_list()[0])


    # conv6.get_shape()是(bath_size*time_step,feature_height,feature_width,out_channels]
    conv_shape_list = conv5.get_shape().as_list()
    feature_height = conv_shape_list[1]
    feature_width = conv_shape_list[2]
    out_channels = conv_shape_list[3]
    # 计算这张图片的feature总数
    total_feature_size = feature_height * feature_width * out_channels
    logger.info(
    "total_feature_size: {}".format(total_feature_size))
    # 返回值shape:(batch_size,time_step,feature_size)
    result = tf.reshape(conv5, [-1, time_step, total_feature_size])

    return result


def add_cnn_layers_updated(inputs, batch_norm=True):
    """
    入参：
        inputs：每张图片的大小是spectrum_height,spectrum_width。
                    shape:(batch_size,time_step,spectrum_height,spectrum_width)
    返回值：
        shape:(batch_size,time_step,feature_size)
    """

    def conv2d(x, W, b, strides=2):
        """
        x的shape：[batch, in_height, in_width, in_channels]
        W：就是filter，它的shape：[filter_height, filter_width, in_channels, out_channels]
        """
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, filter=W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, ksize=4, stride=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

    shapeList = inputs.get_shape().as_list()
    logger.info(
        "cnn_input：[batch,time,H,W] ={}".format(shapeList)
    )


    time_step = shapeList[1]
    spectrum_height = shapeList[2]
    spectrum_width = shapeList[3]

    weights = {
        # 9x9 conv, 1 input, 32 outputs
        'filter1': tf.Variable(tf.random_normal([4, 4, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'filter2': tf.Variable(tf.random_normal([4, 4, 32, 64])),
    }

    biases = {
        'filter1': tf.Variable(tf.random_normal([32])),
        'filter2': tf.Variable(tf.random_normal([64])),
    }

    logger.info(
        "params: [batch*time,H,W,out_channels]"
    )

    # 开始定义CNN
    in_x = tf.reshape(inputs, shape=[-1, spectrum_height, spectrum_width, 1])
    # Convolution Layer
    conv1 = conv2d(in_x, weights['filter1'], biases['filter1'])
    logger.info(
        'conv1: {}'.format(conv1.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    logger.info(
        'pooling1: {} '.format( conv1.get_shape().as_list())
    )

    if batch_norm:
        conv1 = batch_normalization(conv1, biases['filter1'].get_shape().as_list()[0])

    # Convolution Layer
    conv2 = conv2d(conv1, weights['filter2'], biases['filter2'])
    logger.info(
        'conv2: {}'.format(conv2.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    logger.info(
        'pooling2: {}'.format(conv2.get_shape().as_list())
    )

    if batch_norm:
        conv2 = batch_normalization(conv2, biases['filter2'].get_shape().as_list()[0])

    # conv2.get_shape()是(bath_size*time_step,feature_height,feature_width,out_channels]
    conv_shape_list = conv2.get_shape().as_list()
    feature_height = conv_shape_list[1]
    feature_width = conv_shape_list[2]
    out_channels = conv_shape_list[3]
    # 计算这张图片的feature总数
    total_feature_size = feature_height * feature_width * out_channels
    logger.info(
        "total_feature_size {}".format(total_feature_size)
    )

    # 返回值shape:(batch_size,time_step,feature_size)
    result = tf.reshape(conv2, [-1, time_step, total_feature_size])

    return result


# 添加多层CNN
def add_cnn_layers_updated_v2(inputs, batch_norm=True):
    """
    入参：
        inputs：每张图片的大小是spectrum_height,spectrum_width。
                    shape:(batch_size,time_step,spectrum_height,spectrum_width)
    返回值：
        shape:(batch_size,time_step,feature_size)
    """

    def conv2d(x, W, b, strides=1):
        """
        x的shape：[batch, in_height, in_width, in_channels]
        W：就是filter，它的shape：[filter_height, filter_width, in_channels, out_channels]
        """
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, filter=W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, ksize=3, stride=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

    shapeList = inputs.get_shape().as_list()
    logger.info(
        "cnn_input：[batch,time,H,W] = {}".format(shapeList)
    )


    time_step = shapeList[1]
    spectrum_height = shapeList[2]
    spectrum_width = shapeList[3]

    weights = {
        # 9x9 conv, 1 input, 32 outputs
        'filter1': tf.Variable(tf.random_normal([7, 7, 1, 16])),
        # 5x5 conv, 32 inputs, 64 outputs
        'filter2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
        'filter3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        'filter4': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'filter5': tf.Variable(tf.random_normal([3, 3, 128, 128])),
        'filter6': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    }

    biases = {
        'filter1': tf.Variable(tf.random_normal([16])),
        'filter2': tf.Variable(tf.random_normal([32])),
        'filter3': tf.Variable(tf.random_normal([64])),
        'filter4': tf.Variable(tf.random_normal([128])),
        'filter5': tf.Variable(tf.random_normal([128])),
        'filter6': tf.Variable(tf.random_normal([256])),
    }

    logger.info(
        "params: [batch*time,H,W,out_channels]"
    )

    # 开始定义CNN
    in_x = tf.reshape(inputs, shape=[-1, spectrum_height, spectrum_width, 1])
    # Convolution Layer
    conv1 = conv2d(in_x, weights['filter1'], biases['filter1'])
    logger.info(
        'conv1: {}'.format(conv1.get_shape().as_list())
    )


    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    logger.info(
        'pooling1: {}'.format(conv1.get_shape().as_list())
    )

    if batch_norm:
        conv1 = batch_normalization(conv1, biases['filter1'].get_shape().as_list()[0])

    # Convolution Layer
    conv2 = conv2d(conv1, weights['filter2'], biases['filter2'])
    logger.info(
        'conv2: {}'.format(conv2.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    logger.info(
        'pooling2: {}'.format(conv2.get_shape().as_list())
    )

    if batch_norm:
        conv2 = batch_normalization(conv2, biases['filter2'].get_shape().as_list()[0])

    # Convolution Layer
    conv3 = conv2d(conv2, weights['filter3'], biases['filter3'])
    logger.info(
        'conv3: {}'.format(conv3.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3)
    logger.info(
        'pooling3: {}'.format(conv3.get_shape().as_list())
    )

    if batch_norm:
        conv3 = batch_normalization(conv3, biases['filter3'].get_shape().as_list()[0])

    # Convolution Layer
    conv4 = conv2d(conv3, weights['filter4'], biases['filter4'])
    logger.info(
        'conv4: {}'.format(conv4.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4)
    logger.info(
        'pooling4: {}'.format(conv4.get_shape().as_list())
    )

    if batch_norm:
        conv4 = batch_normalization(conv4, biases['filter4'].get_shape().as_list()[0])

    # Convolution Layer
    conv5 = conv2d(conv4, weights['filter5'], biases['filter5'])
    logger.info(
        'conv5: {}'.format(conv5.get_shape().as_list())
    )

    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5)
    logger.info(
        'pooling5: {}'.format(conv5.get_shape().as_list())
    )

    if batch_norm:
        conv5 = batch_normalization(conv5, biases['filter5'].get_shape().as_list()[0])

    # Convolution Layer
    conv6 = conv2d(conv5, weights['filter6'], biases['filter6'])
    logger.info(
    'conv6: {}'.format(conv6.get_shape().as_list()))
    # Max Pooling (down-sampling)
    conv6 = maxpool2d(conv6)
    logger.info(
    'pooling6: {}'.format(conv6.get_shape().as_list()))
    if batch_norm:
        conv6 = batch_normalization(conv6, biases['filter6'].get_shape().as_list()[0])

    # conv6.get_shape()是(bath_size*time_step,feature_height,feature_width,out_channels]
    conv_shape_list = conv6.get_shape().as_list()
    feature_height = conv_shape_list[1]
    feature_width = conv_shape_list[2]
    out_channels = conv_shape_list[3]
    # 计算这张图片的feature总数
    total_feature_size = feature_height * feature_width * out_channels
    logger.info(
    "total_feature_size: {}".format(total_feature_size))
    # 返回值shape:(batch_size,time_step,feature_size)
    result = tf.reshape(conv6, [-1, time_step, total_feature_size])

    return result



def batch_normalization(Wx_plus_b, out_size):
    # Batch Normalize
    fc_mean, fc_var = tf.nn.moments(
        Wx_plus_b,
        axes=list(range(len(Wx_plus_b.get_shape()) - 1)),
        # [0,1,2],   # the dimension you wanna normalize, here [0] for batch
        # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
    )
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
    return Wx_plus_b


def test_add_cnn_layers():
    """
    size: 655360
    add_cnn_layers() inputs shape: [10, 2, 256, 128]
    conv_shape_list: [20, 28, 12, 64]
    feature_height 28
    feature_width 12
    out_channels 64
    total_feature_size 21504
    add_cnn_layers返回结果的shape为： [10, 2, 21504]

    """
    batch_size = 1
    time_step = 1
    spectrum_height = 256
    spectrum_width = 100
    size = batch_size * time_step * spectrum_height * spectrum_width

    logger.info(
    "size:{}".format(size))
    import numpy as np

    raw = np.arange(size)
    x = np.reshape(raw, (batch_size, time_step, spectrum_height, spectrum_width))

    tensor = tf.constant(x, tf.float32)
    result = add_cnn_layers_updated_v2(tensor)
    logger.info(
    "add_cnn_layers返回结果的shape为：{}".format(result.get_shape().as_list()))


# 添加RNN层

def add_dynamic_rnn_layer(inputs, out_size, batch_size, Xt_size, time_step, num_layer=1, keep_prob=0.5):
    """
    设网络在t时刻的输入为Xt，Xt是一个n维向量。
    s个时刻的输入X组成一整个序列，也就是[X0,...,Xs−1,Xs,]，s为time step。

    入参：
        batch_size:
        out_size：RNN层自身的神经元的个数。
        Xt_size：X网络在t时刻的输入为Xt，Xt是一个n维向量。Xt_size等于n。
        inputs：这里input代表一次输入序列。不是t时刻的Xt，而是time_step个Xt组成的输入序列。flatted为一维向量。
                    如果有batch_size个这样的序列，则有batch_size个这样的序列。
                    shape:(batch_size,Xt_size*time_step)。
    返回值：
        一共有time_step个时刻，这里返回的是最后一个时刻的该RNN输出。shape为(batch_size, out_size)

    疑问？！
        这里一个sequence对应一个label，而非一个Xt对应一个label。这里的time_step个Xt的label值是相同的。
        即组成这个sequence的每个Xt的label都相同。

        既然功能是添加一层网络，那么输入到该层的所有input表征的是同一个类别。即一个（x，y）样本点。

    """
    # Reshaping to (batch_size, time_step, Xt_size)
    inputs = tf.reshape(inputs, [-1, time_step, Xt_size])

    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(out_size, state_is_tuple=True,forget_bias=1.0),
                                                input_keep_prob=keep_prob)
                             for _ in range(num_layer)])
    cell =  tf.contrib.rnn.DropoutWrapper(cell,  input_keep_prob=keep_prob)


    sequence_length = np.zeros([batch_size], dtype=int)
    sequence_length += time_step
    print(sequence_length, time_step)
    init_state = cell.zero_state(batch_size, tf.float32)
    print(inputs.shape)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state, dtype=tf.float32, time_major=False,sequence_length=sequence_length)
    """
    rnn_outputs的shape是:[batch_size, time_step, out_size]
    tf.transpose(rnn_outputs, [1, 0, 2])后rnn_outputs的shape变为：(time_step,batch_size, out_size)
    然后再tf.unpack()，得到一个长度为time_step的list，其中的每个元素的shape为(batch_size, out_size)
    """
    # rnn_outputs = tf.unpack(tf.transpose(rnn_outputs, [1, 0, 2]))

    # 返回最后一个time_step的输出,是一个 tensor
    return tf.transpose(rnn_outputs, [1, 0, 2])[-1]


# 构建网络



def build_cnn_net(time_step=10,
                          spectrum_hight=256,  # 代表每列语谱图的高度
                          spectrum_width=128,  # 语谱图的宽度
                          num_classes=10,
                          batch_size=32,
                          ):
    """
    参数：
        spectrum_hight：代表每列语谱图的高度
        spectrum_width：语谱图的宽度
            一张spectrum_hight*spectrum_width的语谱图为一个t时刻的输入Xt。
            time_step个连续这样的语谱图作为一个sequence输入到RNN，输出一个label值。


    网络结构：

    input层         num_of_units：spectrum_hight*spectrum_width     即将spectrum_hight*spectrum_width的图像看做一个特征输入。
        |
    CNN网络
        |
     fc层             num_of_units：fc_in_num_units
        |
    fc层          num_of_units：rnn_out_size
        |
    output层      num_of_units：num_classes
    """

    # 输入层神经元的个数
    input_size = spectrum_hight * spectrum_width

    x_placeholder = tf.placeholder(tf.float32, [None, time_step, input_size], name='input_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, num_classes], name='labels_placeholder')

    # 学习率,可以在训练过程中动态变化
    learning_rate = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

    # 输入层
    # 先reshape为(batch_size,time_step,spectrum_height,spectrum_width)
    X = tf.reshape(x_placeholder, [-1, time_step, spectrum_hight, spectrum_width])

    # CNN网络
    # X = add_cnn_layers_updated_v2(X, True)

    X = add_cnn_layers_updated_lunwen1(X, True)
    # X shape:(batch_size,time_step,feature_size)，当spectrum_width=128时，feature_size=2048
    cnn_out_size = X.get_shape().as_list()[2]  # 获取feature_size的值
    X = tf.reshape(X, [-1, cnn_out_size])
    # X shape:(batch_size*time_step,cnn_out_size)
    logger.info(
    "CNN输出特征大小：{}".format(cnn_out_size)
    )

    # 下面的步骤中，将time_step与batch_size绑在一起。
    # fc1层
    fc1_size = 4096  # fc层的神经元的个数
    fc1 = add_fc_layer(X, cnn_out_size, fc1_size, tf.nn.relu)
    fc1 = tf.nn.dropout(fc1, keep_prob=0.9)
    # fc1的shape (batch_size*time_step,fc1_size)

    logger.info(
        "fc1层大小：{}".format(fc1_size)
    )

    # fc2
    fc2_size = 1024  # fc层的神经元的个数
    fc2 = add_fc_layer(fc1, fc1_size, fc2_size, tf.nn.relu)
    # fc2 = tf.nn.dropout(fc1, keep_prob=0.9)
    # fc2的shape (batch_size*time_step,fc2_size)
    logger.info(
        "fc2层大小：{}".format(fc2_size)
    )


    # output层
    logits = add_fc_layer(fc2, fc2_size, num_classes)
    logger.info(
        "output层大小：{}".format(num_classes)
    )


    predictions = tf.nn.softmax(logits)

    # 平均的cost
    # tf.nn.softmax_cross_entropy_with_logits(logits, y_placeholder)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_placeholder))
    # 在TIMIT数据集上使用GradientDescentOptimizer效果不好，数据量太小
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # prediction_labels和real_labels都是一个numpy.ndarray，shape: (batch_size,)
    # 每一个值是一个下标，指示取值最大的值所在的下标。简单点就是预测的标签值。
    prediction_labels = tf.argmax(predictions, axis=1)
    real_labels = tf.argmax(y_placeholder, axis=1)

    # correct_prediction是一个numpy.ndarray，shape: (batch_size,)，指示那些预测对了。
    correct_prediction = tf.equal(prediction_labels, real_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 一个Batch中预测正确的次数
    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        keep_prob_placeholder=keep_prob_placeholder,
        learning_rate=learning_rate,
        x_placeholder=x_placeholder,
        y_placeholder=y_placeholder,
        optimize=optimize,
        logits=logits,
        prediction_labels=prediction_labels,
        real_labels=real_labels,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        cost=cost,
        accuracy=accuracy,
        fc2=fc2
    )



def build_rnn_net(time_step=1,
                        spectrum_hight=128,  # 代表每列语谱图的高度
                        spectrum_width=128,  # 语谱图的宽度
                        num_classes=7,
                        rnn_out_size=128,  # RNN层的内部的神经元的个数
                        rnn_layers = 2,
                        batch_size=16,
                        ):
    """
    同build_net_v4，只是cnn的输出直接给rnn，不经过fc层。

    参数：
        spectrum_hight：代表每列语谱图的高度
        spectrum_width：语谱图的宽度
            一张spectrum_hight*spectrum_width的语谱图为一个t时刻的输入Xt。
            time_step个连续这样的语谱图作为一个sequence输入到RNN，输出一个label值。


    网络结构：

    input层         num_of_units：spectrum_hight*spectrum_width     即将spectrum_hight*spectrum_width的图像看做一个特征输入。
        |
    CNN网络
        |
    RNN层          num_of_units：rnn_out_size
        |
    output层      num_of_units：num_classes
    """

    # 输入层神经元的个数
    input_size = spectrum_hight * spectrum_width

    x_placeholder = tf.placeholder(tf.float32, [None, time_step, input_size], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, num_classes], name='y_placeholder')

    # 学习率,可以在训练过程中动态变化
    learning_rate = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

    # 输入层
    # 先reshape为(batch_size,time_step,spectrum_height,spectrum_width)
    X = tf.reshape(x_placeholder, [-1, time_step, spectrum_hight, spectrum_width])

    # RNN层
    rnn_output = add_dynamic_rnn_layer(X, rnn_out_size, batch_size, spectrum_hight*spectrum_width, time_step,
                                       num_layer=rnn_layers, keep_prob=keep_prob_placeholder)
    logger.info("RNN output: {}".format(rnn_output.get_shape().as_list()))
    logger.info(
        "RNN num_of_units: {}".format(rnn_out_size)
    )


    # output层
    logits = add_fc_layer(rnn_output, rnn_output.get_shape().as_list()[1], num_classes)

    predictions = tf.nn.softmax(logits, name="out_softmax")

    # 平均的cost
    # tf.nn.softmax_cross_entropy_with_logits(logits, y_placeholder)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels= y_placeholder))
    # 在TIMIT数据集上使用GradientDescentOptimizer效果不好，数据量太小
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # prediction_labels和real_labels都是一个numpy.ndarray，shape: (batch_size,)
    # 每一个值是一个下标，指示取值最大的值所在的下标。简单点就是预测的标签值。
    prediction_labels = tf.argmax(predictions, axis=1)
    real_labels = tf.argmax(y_placeholder, axis=1)

    # correct_prediction是一个numpy.ndarray，shape: (batch_size,)，指示那些预测对了。
    correct_prediction = tf.equal(prediction_labels, real_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 一个Batch中预测正确的次数
    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        keep_prob_placeholder=keep_prob_placeholder,
        learning_rate=learning_rate,
        x_placeholder=x_placeholder,
        y_placeholder=y_placeholder,
        optimize=optimize,
        logits=logits,
        prediction_labels=prediction_labels,
        real_labels=real_labels,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        cost=cost,
        accuracy=accuracy,
    )


def build_cnn_rnn_net(time_step=10,
                          spectrum_hight=256,  # 代表每列语谱图的高度
                          spectrum_width=128,  # 语谱图的宽度
                          num_classes=10,
                          rnn_out_size=128,
                          rnn_layers = 2,
                          batch_size=16,
                          ):
    """
    参数：
        spectrum_hight：代表每列语谱图的高度
        spectrum_width：语谱图的宽度
            一张spectrum_hight*spectrum_width的语谱图为一个t时刻的输入Xt。
            time_step个连续这样的语谱图作为一个sequence输入到RNN，输出一个label值。


    网络结构：

    input层         num_of_units：spectrum_hight*spectrum_width     即将spectrum_hight*spectrum_width的图像看做一个特征输入。
        |
    CNN网络
        |
    RNN             num_of_units：fc_in_num_units
        |
    output层      num_of_units：num_classes
    """

    # 输入层神经元的个数
    input_size = spectrum_hight * spectrum_width

    x_placeholder = tf.placeholder(tf.float32, [None, time_step, input_size], name='input_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, num_classes], name='labels_placeholder')

    # 学习率,可以在训练过程中动态变化
    learning_rate = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

    # 输入层
    # 先reshape为(batch_size,time_step,spectrum_height,spectrum_width)
    X = tf.reshape(x_placeholder, [-1, time_step, spectrum_hight, spectrum_width])

    # CNN网络

    X = add_cnn_layers_updated_lunwen1(X)

    # cnn_out_size = X.get_shape().as_list()[2]  # 获取feature_size的值
    # X = tf.reshape(X, [-1, cnn_out_size])
    # # X shape:(batch_size*time_step,cnn_out_size)
    #
    cnn_out_H_plus_Channels = X.get_shape().as_list()[2]  # 获取feature_size的值
    cnn_out_W = X.get_shape().as_list()[1]  # 获取time_step
    logger.info(
    "cnn_out_H_plus_Channels: {}".format(cnn_out_H_plus_Channels)
    )
    logger.info(
    "cnn_out_W: {}".format(cnn_out_W)
    )
    X = tf.reshape(X, [-1, cnn_out_H_plus_Channels * cnn_out_W])

    # RNN层
    rnn_output = add_dynamic_rnn_layer(X, rnn_out_size, batch_size, cnn_out_H_plus_Channels, cnn_out_W,
                                       num_layer=rnn_layers, keep_prob=keep_prob_placeholder)
    logger.info(
    "RNN num_of_units: {}".format( rnn_out_size)
    )
    # output层
    logits = add_fc_layer(rnn_output, rnn_output.get_shape().as_list()[1], num_classes)
    logger.info(
        "output层大小：{}".format(num_classes)
    )


    predictions = tf.nn.softmax(logits)

    # 平均的cost
    # tf.nn.softmax_cross_entropy_with_logits(logits, y_placeholder)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_placeholder))
    # 在TIMIT数据集上使用GradientDescentOptimizer效果不好，数据量太小
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # prediction_labels和real_labels都是一个numpy.ndarray，shape: (batch_size,)
    # 每一个值是一个下标，指示取值最大的值所在的下标。简单点就是预测的标签值。
    prediction_labels = tf.argmax(predictions, axis=1)
    real_labels = tf.argmax(y_placeholder, axis=1)

    # correct_prediction是一个numpy.ndarray，shape: (batch_size,)，指示那些预测对了。
    correct_prediction = tf.equal(prediction_labels, real_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 一个Batch中预测正确的次数
    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        keep_prob_placeholder=keep_prob_placeholder,
        learning_rate=learning_rate,
        x_placeholder=x_placeholder,
        y_placeholder=y_placeholder,
        optimize=optimize,
        logits=logits,
        prediction_labels=prediction_labels,
        real_labels=real_labels,
        correct_prediction=correct_prediction,
        correct_times_in_batch=correct_times_in_batch,
        cost=cost,
        accuracy=accuracy,
    )




def train_cnn_rnn_svm_tensorboard(graph,
                             dataset,
                             time_step,
                             spectrum_hight,
                             spectrum_width,
                             batch_size,
                             num_epochs,
                             init_learning_rate=0.001,
                             tensorboard_dir="tensorboard",
                             model_file_dir=None, ):
    # tensorboard相关
    cost_tensorboard_placeholder = tf.placeholder(tf.float32, 1)
    accuracy_tensorboard_placeholder = tf.placeholder(tf.float32, 1)


    # 必须要有一个histogram_summary
    #     tf.histogram_summary("prediction",graph['logits'])
    # tf.scalar_summary('cost', cost_tensorboard_placeholder[0])    #已过时
    # tf.scalar_summary('accuracy', accuracy_tensorboard_placeholder[0])
    # 参考：http://stackoverflow.com/questions/41066244/tensorflow-module-object-has-no-attribute-scalar-summary
    tf.summary.scalar('cost', cost_tensorboard_placeholder[0])
    tf.summary.scalar('accuracy', accuracy_tensorboard_placeholder[0])

    # merged = tf.merge_all_summaries() #已过时
    merged = tf.summary.merge_all()

    with tf.Session() as sess:

        # depreacted
        #         train_writer = tf.train.SummaryWriter(tensorboard_dir+"train", sess.graph)     #已过时
        #         test_writer = tf.train.SummaryWriter(tensorboard_dir+"test", sess.graph)
        # 参考:http://stackoverflow.com/questions/41482913/module-object-has-no-attribute-summarywriter
        time_prefix = str(int(time.time()))
        train_writer = tf.summary.FileWriter(tensorboard_dir + "/" + time_prefix + "/train", sess.graph)
        test_writer = tf.summary.FileWriter(tensorboard_dir + "/" + time_prefix + "/test", sess.graph)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        # 从模型读读取session继续训练
        #         checkpoint_state = tf.train.get_checkpoint_state(model_file_dir)
        #         if checkpoint_state and checkpoint_state.model_checkpoint_path:
        #             saver.restore(sess, checkpoint_state.model_checkpoint_path)
        #             logger.info "从模型文件中读取session继续训练..."


        logger.info(
            "batch size: {}".format(batch_size)
        )

        for epoch_index in range(num_epochs):

            # 训练阶段
            # 开始训练
            for (batch_xs, batch_ys) in dataset.train.mini_batches(batch_size):
                batch_xs = batch_xs.reshape([batch_size, time_step, spectrum_width * spectrum_hight])

                sess.run([graph['optimize']], feed_dict={
                    graph['x_placeholder']: batch_xs,
                    graph['y_placeholder']: batch_ys,
                    graph['keep_prob_placeholder']: 0.9
                })

            # 测试阶段
            epoch_delta = 2
            if epoch_index % epoch_delta == 0:


                if epoch_index % 10 == 0:
                    # 计算学习率
                    learnrate = init_learning_rate * (0.99 ** epoch_index)
                    sess.run(tf.assign(graph['learning_rate'], learnrate))

                ##################### test on train set
                # 记录训练集中有多少个batch
                total_batches_in_train_set = 0
                # 记录在训练集中预测正确的次数
                total_correct_times_in_train_set = 0
                # 记录在训练集中的总cost
                total_cost_in_train_set = 0.
                for (train_batch_xs, train_batch_ys) in dataset.train.mini_batches(batch_size):
                    train_batch_xs = train_batch_xs.reshape([batch_size, time_step, spectrum_width * spectrum_hight])
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        graph['keep_prob_placeholder']: 1.0
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        graph['keep_prob_placeholder']: 1.0
                    })

                    total_batches_in_train_set += 1
                    total_correct_times_in_train_set += return_correct_times_in_batch
                    total_cost_in_train_set += (mean_cost_in_batch * batch_size)

                    # tensorboard相关
                    train_acy_tensor = total_correct_times_in_train_set / float(
                        total_batches_in_train_set * batch_size) * 100.0
                    train_mean_cost = total_cost_in_train_set / float(total_batches_in_train_set * batch_size)
                    train_result = sess.run(merged, feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        graph['keep_prob_placeholder']: 1.0,
                        cost_tensorboard_placeholder: [train_mean_cost],
                        accuracy_tensorboard_placeholder: [train_acy_tensor]
                    })

                logger.info(
                    "Epoch - {}, train_mean_cost: {}, train_acy_tensor: {}".format(epoch_index,train_mean_cost,
                    train_acy_tensor)
                )

                train_writer.add_summary(train_result, epoch_index)

                ##################### test on test set
                # 记录测试集中有多少个batch
                total_batches_in_test_set = 0
                # 记录在测试集中预测正确的次数
                total_correct_times_in_test_set = 0
                # 记录在测试集中的总cost
                total_cost_in_test_set = 0.
                for (test_batch_xs, test_batch_ys) in dataset.test.mini_batches(batch_size):
                    test_batch_xs = test_batch_xs.reshape([batch_size, time_step, spectrum_width * spectrum_hight])
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x_placeholder']: test_batch_xs,
                        graph['y_placeholder']: test_batch_ys,
                        graph['keep_prob_placeholder']: 1.0
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x_placeholder']: test_batch_xs,
                        graph['y_placeholder']: test_batch_ys,
                        graph['keep_prob_placeholder']: 1.0
                    })

                    total_batches_in_test_set += 1
                    total_correct_times_in_test_set += return_correct_times_in_batch
                    total_cost_in_test_set += (mean_cost_in_batch * batch_size)

                    # tensorboard相关
                    test_acy_tensor = total_correct_times_in_test_set / float(
                        total_batches_in_test_set * batch_size) * 100.0
                    test_mean_cost = total_cost_in_test_set / float(total_batches_in_test_set * batch_size)
                    test_result = sess.run(merged, feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        graph['keep_prob_placeholder']: 1.0,
                        cost_tensorboard_placeholder: [test_mean_cost],
                        accuracy_tensorboard_placeholder: [test_acy_tensor]
                    })

                logger.info(
                "Epoch -{} test_mean_cost: {}, test_acy_tensor:{}".format(epoch_index, test_mean_cost, test_acy_tensor)
                )

                test_writer.add_summary(test_result, epoch_index)

                ### summary and logger.info
                acy_on_test = total_correct_times_in_test_set / float(total_batches_in_test_set * batch_size)
                acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * batch_size)
                logger.info(
                    'Epoch - {:2d} , acy_on_test:{:6.2f}%({}/{}),loss_on_test:{:6.2f}, acy_on_train:{:6.2f}%({}/{}),loss_on_train:{:6.2f}'.
                    format(epoch_index, acy_on_test * 100.0, total_correct_times_in_test_set,
                           total_batches_in_test_set * batch_size, total_cost_in_test_set, acy_on_train * 100.0,
                           total_correct_times_in_train_set, total_batches_in_train_set * batch_size,
                           total_cost_in_train_set))

                logger.info(
                    "Epoch -{}, learning rate: {}".format(epoch_index, learnrate)
                )

        # SVM 训练阶段
        converter = np.array([x for x in range(test_batch_ys.shape[1])])
        train_rnn_features = []
        train_rnn_labels = []
        for (batch_xs, batch_ys) in dataset.train.mini_batches(batch_size):
            batch_xs = batch_xs.reshape([batch_size, time_step, spectrum_width * spectrum_hight])
            # 默认 time_step == 1时候是一个 cnn 网络，不含 RNN,否则后端是一个 RNN
            features_batch = graph["logits"].eval(feed_dict={graph['x_placeholder']: batch_xs,
                                                                     graph['keep_prob_placeholder']: 0.9})
            train_rnn_features.append(np.reshape(features_batch,[-1]))
            train_rnn_labels.append(np.sum(np.multiply(converter, batch_ys),axis=1))
        train_rnn_features = np.reshape(train_rnn_features,[len(dataset.train.images)//batch_size*batch_size, np.shape(features_batch)[1]])
        train_rnn_labels = np.reshape(train_rnn_labels,[len(dataset.train.images)//batch_size*batch_size,])
        np.savetxt(str(time.time()) + "_train_features.csv",train_rnn_features, delimiter=',')
        np.savetxt(str(time.time()) + "_train_labels.csv", train_rnn_labels, delimiter=',')
        logger.info("Epoch-{} svm features: {}".format(epoch_index, np.shape(train_rnn_features)))
        logger.info("Epoch-{} svm labels: {}".format(epoch_index, np.shape(train_rnn_labels)))
        clf = svm.SVC()
        clf.fit(train_rnn_features, train_rnn_labels)

        # SVM 测试阶段
        test_rnn_features = []
        test_rnn_labels = []
        for (batch_xs, batch_ys) in dataset.test.mini_batches(batch_size):
            batch_xs = batch_xs.reshape([batch_size, time_step, spectrum_width * spectrum_hight])
            features_batch = graph["logits"].eval(feed_dict={graph['x_placeholder']: batch_xs,
                                                                 graph['keep_prob_placeholder']: 1.0})
            test_rnn_features.append(np.reshape(features_batch, [-1]))
            test_rnn_labels.append(np.sum(np.multiply(converter, batch_ys), axis=1))
        test_rnn_features = np.reshape(train_rnn_features, [len(dataset.train.images) // batch_size * batch_size,
                                                            np.shape(features_batch)[1]])
        test_rnn_labels = np.reshape(train_rnn_labels, [len(dataset.train.images) // batch_size * batch_size, ])

        accuracy = clf.score(test_rnn_features, test_rnn_labels)
        np.savetxt(str(time.time()) + "_test_features.csv",test_rnn_features, delimiter=',')
        np.savetxt(str(time.time()) + "_test_labels.csv", test_rnn_labels, delimiter=',')
        logger.info("Epoch-{} svm accuracy: {}".format(epoch_index, accuracy))


def test_svm():
    batch_size = 10
    output_size = 128
    features_arr = []
    labels_arr = []
    converter = [1,2]

    features_1 = [[1,2,3],[4,5,6]]
    features_2 = [[1, 2, 3], [4, 5, 6]]
    features_arr.append(np.reshape(features_1, [-1]))
    features_arr.append(np.reshape(features_2, [-1]))
    features_arr = np.reshape(features_arr, [4,3])

    labels_1 = [[0,1],[1,0]]
    labels_2 = [[0, 1], [1, 0]]
    labels_arr.append(np.sum(np.multiply(converter, labels_1), axis=1))
    labels_arr.append(np.sum(np.multiply(converter, labels_2), axis=1))
    labels_arr = np.reshape(labels_arr, [4,])
    logger.info(features_arr,labels_arr)


def run_cnn(params):
    """
    time_step 设置为1即可
    """

    spectrum_width = params["spectrum_width"]
    spectrum_hight = params["spectrum_hight"]
    num_classes = params["num_classes"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    time_step = spectrum_hight // spectrum_width

    # spectr_cols = spectrum_width * time_step, 注意语谱图的宽度应该这样定义：方便输入到网络中
    # spectr_rows = spectrum_hight 语谱图的宽度

    dataset = read_data_sets(params["split_png_data"],
                             num_class=num_classes, one_hot=True, params=params)
    tf.reset_default_graph()
    g = build_cnn_net(time_step, spectrum_hight, spectrum_width, num_classes, batch_size=batch_size)
    train_cnn_rnn_svm_tensorboard(g, dataset, time_step, spectrum_hight, spectrum_width,
                          batch_size, num_epochs, tensorboard_dir="tensorboard_cnn")


def run_rnn(params):
    """

    :param params:
    注意这里的 time_step*spectrum_width = 语谱图的宽度


    :return:
    """
    # time_step = 128
    # spectrum_width = 1
    # spectrum_hight = 128
    # num_classes = 7
    # batch_size = 32
    # num_epochs = 15
    # rnn_out_size = 128
    # rnn_layers = 3


    spectrum_width = params["spectrum_width"]
    spectrum_hight = params["spectrum_hight"]
    num_classes = params["num_classes"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    rnn_out_size = params["rnn_out_size"]
    rnn_layers = params["rnn_layers"]
    time_step = spectrum_hight // spectrum_width

    # spectr_cols = spectrum_width * time_step, 注意语谱图的宽度应该这样定义：方便输入到网络中
    # spectr_rows = spectrum_hight 语谱图的宽度
    dataset = read_data_sets(params["split_png_data"],
                                        num_class=num_classes, one_hot=True, params=params)
    tf.reset_default_graph()
    g = build_rnn_net(time_step, spectrum_hight, spectrum_width, num_classes, rnn_out_size,rnn_layers, batch_size)

    # train_cnn_rnn_tensorboard(g, dataset, time_step, spectrum_hight, spectrum_width,
    #                          batch_size, num_epochs, tensorboard_dir="tensorboard_cnn_rnn")

    train_cnn_rnn_svm_tensorboard(g, dataset, time_step, spectrum_hight, spectrum_width,
                                  batch_size, num_epochs, tensorboard_dir="tensorboard_rnn")


def run_cnn_rnn(params):

    # "time_step":2
    # "spectrum_width":64,
    # "spectrum_hight": 128,
    # "num_classes": 7,
    # "batch_size": 32,
    # "num_epochs": 1,
    # "rnn_out_size": 128,

    spectrum_width = params["spectrum_width"]
    spectrum_hight = params["spectrum_hight"]
    num_classes = params["num_classes"]
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    rnn_out_size = params["rnn_out_size"]
    rnn_layers = params["rnn_layers"]
    time_step = spectrum_hight // spectrum_width

    # spectr_cols = spectrum_width * time_step, 注意语谱图的宽度应该这样定义：方便输入到网络中
    # spectr_rows = spectrum_hight 语谱图的宽度
    dataset = read_data_sets(params["split_png_data"],
                                        num_class=num_classes, one_hot=True, params=params)
    tf.reset_default_graph()
    g = build_cnn_rnn_net(time_step, spectrum_hight, spectrum_width, num_classes, rnn_out_size=rnn_out_size, rnn_layers=rnn_layers,batch_size=batch_size)

    # train_cnn_rnn_tensorboard(g, dataset, time_step, spectrum_hight, spectrum_width,
    #                          batch_size, num_epochs, tensorboard_dir="tensorboard_cnn_rnn")

    train_cnn_rnn_svm_tensorboard(g, dataset, time_step, spectrum_hight, spectrum_width,
                                  batch_size, num_epochs, tensorboard_dir="tensorboard_cnn_rnn")




if __name__ == '__main__':
    test_svm()