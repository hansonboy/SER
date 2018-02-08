#!/usr/lib/env python
# coding:utf-8
"""
  description:
    模拟mnist tensorflow 的官方例子，构造模型输入类
"""
import os
import os.path
import numpy as np
import time
import logging
from .path_helper import *
import tensorflow as tf
from .png_data_preprocess import *
from .mfcc import *
import re

logger = logging.getLogger("rnn_embedding.input_data")

class DataSet(object):
    def __init__(self, images, labels, fake_data=False, one_hot=False,
                 dtype=tf.float32):
        """Construct p20170103 DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        images = np.array(images)
        labels = np.array(labels)
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            if dtype == tf.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def batches_in_one_epoch(self, batch_size):
        batch_len = self.num_examples / batch_size
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self.labels[perm]
        for i in range(int(batch_len)):
            yield (self._images[i * batch_size:(i + 1) * batch_size], self._labels[i * batch_size:(i + 1) * batch_size])

    def batches_in_epochs(self, epochs, batch_size):
        for i in range(epochs):
            yield self.batches_in_one_epoch(batch_size)

    # 最后一个不足一个 batch 就返回去掉最后一个的部分
    def mini_batches(self, mini_batch_size):
        """
          return: list of tuple(x,y)
        """
        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

        n = self.images.shape[0]

        mini_batches = [(self._images[k:k + mini_batch_size], self._labels[k:k + mini_batch_size])
                        for k in range(0, n, mini_batch_size)]

        if len(mini_batches[-1]) != mini_batch_size:
            return mini_batches[:-1]
        else:
            return mini_batches

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]




def get_value_by_key(dict, key, default):
    if dict.get(key) is None:
        return default
    else:
        return dict.get(key)


def test_get_value_by_key():
     dict = {}
     print(get_value_by_key(dict,"1",1))


def read_data_sets(train_dir, data_type="spectrum", num_class=7, fake_data=False, one_hot=False, dtype=tf.float32, logfbank_width=80, params={}):

    class DataSets(object):
        pass

    data_sets = DataSets()
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

        data_sets.train = fake()
        data_sets.validation = fake()
        data_sets.test = fake()
        return data_sets
    images = []
    labels = []

    if data_type == "spectrum":
        images, labels = extract_images_labels(train_dir, False, num_class)
    if data_type == "mfcc":
        images, labels = extract_mfcc_labels(train_dir, logfbank_width, num_class, one_hot=False, params=params)
    if len(images) != len(labels):
        raise ValueError("images's dimission doesn't match with labels")
    print(np.shape(images))
    train_images = images[:int(len(images) * 0.8)]
    train_labels = labels[:int(len(labels) * 0.8)]
    validation_images = images[int(len(images) * 0.6):int(len(images) * 0.8)]
    validation_labels = labels[int(len(labels) * 0.6):int(len(labels) * 0.8)]
    test_images = images[int(len(images) * 0.8):]
    test_labels = labels[int(len(labels) * 0.8):]

    if one_hot:
        train_labels = dense_to_one_hot(np.array(train_labels) - 1,num_class)
        validation_labels = dense_to_one_hot(np.array(validation_labels) - 1, num_class)
        data_sets.test = DataSet(test_images, dense_to_one_hot(np.array(test_labels) - 1, num_class), dtype=dtype)
    else:
        data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
    data_sets.validation = DataSet(validation_images, validation_labels, dtype=dtype)


    data_sets.all_test = []
    test_images_group, test_labels_group = split_features_labels_by_classes(test_images, test_labels, num_class)
    for x in range(num_class):
        test_images_tmp = test_images_group["{}".format(x + 1)]
        test_labels_tmp = test_labels_group["{}".format(x + 1)]
        if one_hot:
            test_labels_tmp = dense_to_one_hot(np.array(test_labels_tmp) - 1, num_class)
        dataset = DataSet(test_images_tmp, test_labels_tmp,dtype=dtype)
        data_sets.all_test.append(dataset)

    logger.info(
        "train--image's shape: {}, labels' shape: {}"
            .format(data_sets.train.images.shape, data_sets.train.labels.shape))
    logger.info(
        "validation--image's shape: {}, labels' shape: {}"
            .format(data_sets.validation.images.shape, data_sets.validation.labels.shape))
    logger.info(
        "test--image's shape: {}, labels' shape: {}"
            .format(data_sets.test.images.shape, data_sets.test.labels.shape))
    logger.info(
        "test-categories's shape:{}".format(data_sets.all_test)
    )
    return data_sets



def extract_images_labels(image_path, one_hot=False, num_class=7):
    images = []
    labels = []
    for images_file in remove_file_in_dirs(os.listdir(image_path), ):
        labels.append(int(images_file.split(".png")[0].split("_")[2]))
        images.append(read_image(os.path.join(image_path, images_file)))
    logger.debug("抽取images: {} labels: {}".format(np.shape(images), np.shape(labels)))
    if one_hot:
        labels = dense_to_one_hot(np.array(labels) - 1, num_class)
    return images, labels


def dense_to_one_hot(labels_dense, num_classes=7):
    """Convert class labels from scalars to one-hot vectors."""
    labels_dense = np.array(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

"""
    抽取 mfcc -labels
"""
# signal,samplerate=16000,winlen=0.025,winstep=0.01,
          # nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97
def extract_mfcc_labels(wav_dir, logfbank_width, num_class, one_hot=False, params={}):
    winlen = get_value_by_key(params,"winlen", 0.025)
    winstep = get_value_by_key(params,"winstep", 0.01)
    preemph = get_value_by_key(params,"preemph", 0.97)
    nfft = get_value_by_key(params,"nfft", 512)

    logger.info(wav_dir)
    files = os.listdir(wav_dir)
    remove_file_in_dirs(files,".DS_Store")
    logfbank_arr = []
    labels_arr=[]
    # 要求 logfbank_width 必须能够被3整除才行
    out_size = logfbank_width//3
    for file in files:
        if re.match(r'.*wav',file) is not None:
            full_path = os.path.join(wav_dir, file)
            (rate, sig) = wav.read(full_path)
            mfcc_feat = mfcc(sig, rate, nfilt=out_size, winlen=winlen, winstep=winstep, preemph=preemph, nfft=nfft,
                       numcep=out_size)
            d_mfcc_feat = delta(mfcc_feat,1)
            d_d_mfcc_feat = delta(mfcc_feat,2)
            mfcc_feat_bank = np.column_stack((np.column_stack((mfcc_feat, d_mfcc_feat)), d_d_mfcc_feat))
            logfbank_arr.append(mfcc_feat_bank)
            labels_arr.append(int(file.split(".png")[0].split("_")[2]))
    if one_hot:
        labels_arr = dense_to_one_hot(np.array(labels_arr) - 1, num_class)
    logger.info("{} 已经生成了{}个 MFCC".format(wav_dir, len(logfbank_arr)))
    count_MFCC(wav_dir,logfbank_arr)
    return crop_mfcc_80_80(logfbank_arr,labels_arr,width=params["IMAGES_WIDTH"],height=params["IMAGES_HEIGHT"])

def count_MFCC(wav_dir,logfbank_arr):
    colNums = []
    for logfbank in logfbank_arr:
        image_shape = np.shape(logfbank)
        colNums.append(image_shape[0])
    logger.info("已经分析了{0:s}-{1:d}个 MFCC, 尺寸最大的是 {2:d} * {3:d}, 尺寸最小的是 {4:d} * {5:d}".format(wav_dir,
                                                                                               len(logfbank_arr),
                                                                                               max(colNums),
                                                                                               image_shape[1],
                                                                                               min(colNums),
                                                                                               image_shape[1]))
def crop_mfcc_80_80(logfbank_arr, labels_arr, width=80, height=80):
    logfbank_split_arr = []
    labels_split_arr = []
    for i in range(len(logfbank_arr)):
        logfbank_value = logfbank_arr[i]
        iter = 0
        while (iter+1)*height <= logfbank_value.shape[0]:
            start = iter * height
            end = (iter + 1) * height
            iter = iter+1
            logfbank_split_arr.append(logfbank_value[start:end,:])
            labels_split_arr.append(labels_arr[i])
    logger.info("裁剪统一 MFCC 长度为 {}*{},共计{}组".format(width, height, len(labels_split_arr)))
    return logfbank_split_arr,labels_split_arr

"""
将给定的 featrues， labels 按照给定的 n_class 进行分开
"""
def split_features_labels_by_classes(features, labels, n_class):
    features_group = {}
    labels_group = {}
    for j in range(np.array(features).shape[0]):
        features_tmp = features_group.get("{}".format(labels[j]))
        if features_tmp is None:
            features_group.setdefault("{}".format(labels[j]), [features[j]])
        else:
            features_tmp.append(features[j])

        labels_tmp = labels_group.get("{}".format(labels[j]))
        if labels_tmp is None:
            labels_group.setdefault("{}".format(labels[j]), [labels[j]])
        else:
            labels_tmp.append(labels[j])
    return features_group, labels_group



def test_split_features_labels_by_classes():
    featrues = [
        [1,2,3,4],
        [2,3,4,5],
        [3,4,5,6],
        [1, 3, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ]
    labels = [
        1,2,3,1,2,3
    ]
    featrues_group , labels_grop = split_features_labels_by_classes(featrues, labels, 3)
    print(featrues_group, labels_grop)

def test_read_minist_spectrum():
    data_sets = read_data_sets("/Users/jw/Documents/MyProject/PythonWorkSpace/cnn_rnn_audio_emotion_recognition/audio_data/1484131952_256_0.5/split_png_data/Emo",
                           num_class=7, data_type="spectrum", one_hot=True,
                           logfbank_width=80)
    all_test = data_sets.all_test
    sum = 0
    for x in range(7):
        print(all_test[x].images.shape, all_test[x].labels.shape)
        sum += all_test[x].labels.shape[0]
        print(all_test[x].images[0], all_test[x].labels[0])
    print(sum)
    print(data_sets.test.images[0],data_sets.test.labels[0])
    print(data_sets.test.images.shape, data_sets.test.labels.shape)
def test_read_minis_mfcc():
    data_sets = read_data_sets(
        "../wave_data/Emo",
        num_class=7, data_type="mfcc", one_hot=True,
        logfbank_width=80)
    all_test = data_sets.all_test
    sum = 0
    for x in range(7):
        print(all_test[x].images.shape, all_test[x].labels.shape)
        sum += all_test[x].labels.shape[0]
        print(all_test[x].images[0], all_test[x].labels[0])
    print(sum)
    print(data_sets.test.images[0], data_sets.test.labels[0])
    print(data_sets.test.images.shape, data_sets.test.labels.shape)

if __name__ == '__main__':
    # split_png_dir = "../audio_data/1484131952_256_0.5/split_png_data/"
    # CASIA_split_png_dir = os.path.join(split_png_dir, "CASIA")
    # Emo_split_png_dir = os.path.join(split_png_dir, "Emo")
    # SAVEE_split_png_dir = os.path.join(split_png_dir, "SAVEE")
    # t = time.time()
    # # images,labels =  extract_images_labels(CASIA_split_png_dir,6)
    # mnist = read_data_sets(CASIA_split_png_dir, 7)
    # print((mnist.test.images.shape,mnist.test.labels.shape))
    # print(("耗费时间：%s s " % str(time.time() - t)))
    # test_split_features_labels_by_classes()
    # test_read_minist_spectrum()
    # test_read_minis_mfcc()
    # test_get_value_by_key()
    a = []
    a.append(int("1_1_6.png".split(".png")[0].split("_")[2]))
    print(a)

"""
运行结果如下： 第一次
抽取images: (4870, 128, 128) labels: (4870,)
train:
images's shape: (2922, 128, 128) labels'shape (2922, 7)
validation:
images's shape: (974, 128, 128) labels'shape (974, 7)
test:
images's shape: (974, 128, 128) labels'shape (974, 7)
耗费时间：4.59748911858 s

第二次
抽取images: (4870, 128, 128) labels: (4870,)
train:
images's shape: (2922, 16384) labels'shape (2922, 7)
validation:
images's shape: (974, 16384) labels'shape (974, 7)
test:
images's shape: (974, 16384) labels'shape (974, 7)
耗费时间：7.26304101944 s

第三次
/Users/jw/Desktop/audio_data/1483530666/split_png_data/CASIA
抽取images: (4870, 128, 128) labels: (4870,)
train:
images's shape: (2922, 16384) labels'shape (2922, 7)
validation:
images's shape: (974, 16384) labels'shape (974, 7)
test:
images's shape: (974, 16384) labels'shape (974, 7)
耗费时间：4.10564899445 s

最近一次
test.images.shape :(326, 16384)  test.labels.shape(326, 7)
耗费时间：1.776453971862793 s

"""
