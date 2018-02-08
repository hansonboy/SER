#!/usr/lib/env python
#coding=utf-8
"""
  description:
    实现了数据的统一预处理：
        数据库重新命名、统一格式
        音频转换为语谱图
        进行语谱图统计
        切分语谱图为固定大小的图片
"""
import os
import time
import os.path
import numpy as np
from pprint import pprint
import logging
from .png_data_preprocess import *
from .wave_data_preprocess import *
from .database_preprocess import *
from pprint import pprint

logger = logging.getLogger("rnn_embedding.data_helper")
def test():
    im = [[0, 255, 0, 255 ],[0, 0, 0 ,255],[255, 0, 0, 0]]
    im = np.array(im)
    print((im.shape))
    save_image(im, "new.png")
    save_image(im.transpose(), "newT.png")
    crop_one_png("new.png","./n",2,2)

def crop_one_png(old_png_path, new_png_path, width, height):
    im = read_image(old_png_path)
    save_image(im, "new.png")
    iwidth, iheight = np.shape(im)
    try:
        if width * height > iwidth * iheight:
            raise ValueError("width * height > num of pixle")
    except ValueError:
        logger.debug("width * height > num of pixle:oldPath:{} newPath:{}".format(old_png_path,new_png_path))
        return 0
    num =  int( iwidth * iheight * 1.0 / (width * height))
    im = im.transpose().reshape([-1])[: num * (width * height)].reshape([-1,height,width])
    for i in range(im.shape[0]):
        save_image(np.array(im[i]).transpose(),new_png_path.split(".png")[0] + "_" + str(i) + ".png")
    return num

"""
  移动切分，有重叠
  移动的个数为 strides
  old_png_path 是旧图片的路径
  new_png_path 是新图片的路径（含有原来的名称，新的切分图片名称会在new_png_path_1_.png）
"""
def crop_one_png_with_strides(old_png_path, new_png_path, width, height, stride):
    im = read_image(old_png_path)
    iheight, iwidth = np.shape(im)
    if iwidth <= width:
        ims = np.zeros([iheight,width])
        for i in range(iwidth):
            for j in range(iheight):
                ims[j][i] = im[j][i]

        save_image(np.array(ims), new_png_path)
        return 1


    try:
        if width * height > iwidth * iheight:
            raise ValueError("width * height > num of pixle")
    except ValueError:
        logger.debug("width * height > num of pixle:oldPath:{} newPath:{}".format(old_png_path,new_png_path))
        return 0
    num = (iwidth - width)//stride + 1
    start = 0
    end = 0;
    for i in range(num):
        start = i*stride
        end = start + width
        save_image(np.array(im[:,start:end]), new_png_path.split(".png")[0] + "_" + str(i) + ".png")
    if end < iwidth:
        end = iwidth -1
        start = end - width
        save_image(np.array(im[:, start:end]), new_png_path.split(".png")[0] + "_" + str(num) + ".png")
    return num


def test_crop_one_png_with_strides():
    old_png_path = "1_1_1_0.png"
    new_png_path = "log/1_1_1_0.png"
    crop_one_png_with_strides(old_png_path,new_png_path,128,128,128)

def crop_all_png(png_dir,output_dir, width=128, height=128, strides=0):
    dirs = os.listdir(png_dir)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    sum = 0
    for dir in dirs:
        absolute_dir = os.path.join(png_dir, dir)
        out_dir = os.path.join(output_dir, dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if os.path.isdir(absolute_dir):
            fileDirs = os.listdir(absolute_dir)
            if ".DS_Store" in fileDirs:
                fileDirs.remove(".DS_Store")
            for file in fileDirs:
                source_file = os.path.join(absolute_dir, file)
                output_file = os.path.join(out_dir, file)
                # sum = sum + crop_one_png(source_file,output_file,width,height)
                sum = sum + crop_one_png_with_strides(source_file, output_file, width, height, strides)

        logger.info("{}共切分成了{}张图片".format(dir,str(sum)))


def raw_data_preprocess(preprocess_parameters, refresh=False):
    if refresh:
        # 取参数数据
        binsize = preprocess_parameters["binsize"]
        overlapFac = preprocess_parameters["overlapFac"]
        alpha = preprocess_parameters["alpha"]
        split_png_width = preprocess_parameters["split_png_width"]
        split_png_height = preprocess_parameters["split_png_height"]
        strides = preprocess_parameters["strides"]

        # 构造生成路径
        root_dir = str(preprocess_parameters["root_dir"])
        raw_wav_data_dir = os.path.join(root_dir, "raw_wave_data")
        root_dir_suffix = "{}_{}_{}_{}".format(str(int(time.time())),binsize,overlapFac,strides)
        root_dir = os.path.join(root_dir, root_dir_suffix)
        wave_data_dir = os.path.join(root_dir, "wave_data")
        png_dir = os.path.join(root_dir, "png_data")
        split_png_dir = os.path.join(root_dir, "split_png_data")
        params_file_path = os.path.join(root_dir, "parameters.txt")

        # 保存数据处理参数
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        else:
            logger.error("root_dir 失败")

        with open(params_file_path, "w") as p:
            pprint(dict(preprocess_parameters), p)
        logger.info("数据处理参数保存完成")

        # 统一不同数据库的数据格式
        format_database(raw_wav_data_dir, wave_data_dir)
        logger.info("数据库格式统一完成")

        # 将原始的wave 数据生成png 数据
        raw_wave_2_png(wave_data_dir, png_dir, binsize=binsize, overlapFac=overlapFac, alpha=alpha)
        logger.info("原始音频生成语谱图完成")

        # 统计png 数据
        count_png(png_dir)
        logger.info("音频统计完成")

        # 将大图片统一裁剪成小的相同大小的图片
        crop_all_png(png_dir, split_png_dir, width=split_png_width, height=split_png_height,strides=strides)
        logger.info("语谱图统一裁剪完成")


if __name__ == '__main__':
    test_crop_one_png_with_strides()