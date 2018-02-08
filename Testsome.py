import tensorflow as tf
import numpy as np


def SimulateData(batch_size=10, Pic_size=None, channel=3, labels=10):
    y = np.random.randint(labels,size=(batch_size,1))
    y_data = np.zeros((batch_size,labels),np.float32)
    for i in range(batch_size):
        y_data[i,y[i]]=1

    return np.random.rand(batch_size, Pic_size[0], Pic_size[1], channel), y_data


def conv2d_layer(input,in_channel, out_channel, filter_height=3, filter_weight=3, padding='VALID', name='Conv'):
    '''
    :param input: The input of layer with size [batch_size, height, weight, input_channel]
    :param out_channel: The number of feature map in this layer
    :param filter_height:
    :param filter_weight:
    :param padding:
    :param name: the name scope of the layer
    :return: the out put tensor of this layer with size [batch_size, **, **, out_channel]
    '''
    with tf.name_scope(name):
        with tf.variable_scope('weight'):
            filter_shape = [filter_height, filter_weight, in_channel, out_channel]
            filter = tf.Variable(initial_value=tf.truncated_normal(filter_shape))

            # filter = tf.get_variable(name='filter',
            #                          shape=filter_shape,
            #                          initializer=tf.truncated_normal_initializer())
            # suit for summary.image change the filter shape from [height, weight, in_channel, out_channel] to [batch, height, width, channels] channel = 1.
            image = tf.reshape(filter,[in_channel*out_channel,filter_height, filter_weight,1])
            tf.summary.image(name=name+'/weight/filter-image', tensor=image)

            baise = tf.Variable(initial_value=tf.constant(value=0.1,shape=[out_channel]))
            # baise = tf.get_variable(name='baise',
            #                        shape= out_channel,
            #                        initializer=tf.constant_initializer(value=0.1))
            tf.summary.histogram(name=name+'/weight/baise',values=baise)
            # defult setting of tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)

            output = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding = padding)
            output = tf.nn.relu(output + baise)

    return output

def max_pool(input, ksize=2, strides=2, padding='SAME', name='max_pool'):
    with tf.name_scope(name):
        output = tf.nn.max_pool(input, ksize=[1,ksize,ksize,1],
                          strides=[1,strides,strides,1],padding=padding)
    return output


def Dens(input,input_size, output_size, name='FC'):
    '''
    :param input: [batch_size, input_size]
    :param output_size: Just output size
    :return: 2-D tensor with size [batch_size, output_size]
    '''
    with tf.name_scope(name):
        with tf.variable_scope('weight'):
            FC_weight = tf.get_variable(name='FC_weight',
                                        shape=[input_size,output_size],
                                        initializer=tf.truncated_normal_initializer())
            tf.summary.histogram(name=name+'/weight/FC-weight',values=FC_weight)
            FC_baise = tf.get_variable(name='FC_baise',
                                       shape= [output_size],
                                       initializer=tf.constant_initializer(value=0.1))
            tf.summary.histogram(name=name+'/weight/FC-baise',values=FC_baise)

            output = tf.nn.relu(tf.matmul(input, FC_weight) + FC_baise)

    return output

# def main():
#     Inputs, Labels = SimulateData(Pic_size=[28,28])
#     log_dir = 'F:\\Program Files\Python\TestInception\log'
#
#     with tf.name_scope('Input'):
#         input = tf.placeholder(tf.float32,[None,28,28,3])
#         label = tf.placeholder(tf.float32,[None,10])
#
#     net = conv2d_layer(input,out_channel=10,name='Conv26x26')
#     net = max_pool(net,name='max_pool_13x13')
#     net = conv2d_layer(net,out_channel=8,filter_height=4,filter_weight=4,name='Conv10x10')
#     net = max_pool(net,name='max_pool_5x5')
#
#     net_flatten = tf.reshape(net, [-1, 5*5*8])
#     net = Dens(net_flatten, output_size=10, name='FC')
#
#     output = tf.nn.softmax(net)
#     loss = tf.reduce_sum(output-label)
#
#     tf.summary.scalar(name='loss', tensor=loss)
#
#     train = tf.train.AdamOptimizer(0.001).minimize(loss)
#     merge = tf.summary.merge_all()
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         write = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())
#         for j in range(100):
#             _,summary = sess.run([train,merge],
#                             feed_dict={input:Inputs, label:Labels})
#             write.add_summary(summary,global_step=j)


if __name__ == '__main__':
    tf.reset_default_graph()
    Inputs, Labels = SimulateData(Pic_size=[28,28], channel=3, labels=10)
    log_dir = 'log'

    with tf.name_scope('Input'):
        input = tf.placeholder(tf.float32,[None,28,28,3])
        label = tf.placeholder(tf.float32,[None,10])

    net = conv2d_layer(input,in_channel=3,out_channel=10,filter_height=3,filter_weight=3,name='Conv26x26')
    net = max_pool(net,name='max_pool_13x13')
    net = conv2d_layer(net,in_channel=10,out_channel=8,filter_height=4,filter_weight=4,name='Conv10x10')
    net = max_pool(net,name='max_pool_5x5')

    print(tf.shape(net))
    net_flatten = tf.reshape(net, [-1, 5*5*8])
    print(tf.shape(net_flatten))
    net = Dens(net_flatten,input_size=5*5*8, output_size=10, name='FC')

    output = tf.nn.softmax(net)
    with tf.name_scope('loss'):
        loss = tf.reduce_sum(output-label)
        tf.summary.scalar('loss',loss)

    merge = tf.summary.merge_all()

    with tf.name_scope('train'):
        train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        write = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())
        for j in range(100):
            _,summary = sess.run([train,merge],
                            feed_dict={input:Inputs, label:Labels})
            write.add_summary(summary,global_step=j)
            #print(j)