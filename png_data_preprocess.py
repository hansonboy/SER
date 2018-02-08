#!/usr/lib/env python
#coding:utf-8
"""
  description:
    实现统计数据库中包含的语谱图大小数目功能
"""
import os
import re
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("rnn_embedding.png_data_preprocess")

def read_image(image_path):
    im = Image.open(image_path).convert("L")
    res = np.array(im)
    im.close()
    return res


def save_image(image_data, image_path):
    im = Image.fromarray(image_data.astype(np.uint8),mode="L")
    im.save(image_path)
    im.close()

def is_png(image_path):
    return re.match(r".*\.png$",image_path) is not None

"""
sumary:
  统计Image data 的维度
"""
def count_png(png_dir):
    dirs = os.listdir(png_dir)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")

    for dir in dirs:
        absolute_dir = os.path.join(png_dir, dir)
        colNums = []
        if os.path.isdir(absolute_dir):
            for file in os.listdir(absolute_dir):
                source_file = os.path.join(absolute_dir, file)
                image_shape = np.shape(read_image(source_file))
                colNums.append(image_shape[1])
        logger.info("已经分析了{0:s} - {1:d}张图片, 尺寸最大的是 {2:d} * {3:d}, 尺寸最小的是 {4:d} * {5:d}".format(dir,
                                                                                         len(os.listdir(absolute_dir)),
                                                                                         max(colNums),
                                                                                         image_shape[0],
                                                                                         min(colNums),
                                                                                         image_shape[0]))


"""
运行结果：
已经分析了CASIA - 1200张图片, 尺寸最大的是 761 * 256, 尺寸最小的是 143 * 256
已经分析了Emo - 535张图片, 尺寸最大的是 1389 * 256, 尺寸最小的是 185 * 256
已经分析了SAVEE - 480张图片, 尺寸最大的是 3051 * 256, 尺寸最小的是 693 * 256
"""

if __name__ == '__main__':
    png_dir = "/Users/jw/Desktop/audio_data/png_data"
    count_png(png_dir)


