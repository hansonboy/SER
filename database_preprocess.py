#!/usr/lib/env python
# coding=utf-8
"""
sumary:
  将不同数据库的格式进行统一：database_name/actor_text_class.wav
  class: 1 anger 2 boredom 3 disgust 4anxiety/fear 5 happiness 6 sadness 7 Neutral
  example: 1_1_1 表示1号演员第一段文本语音情绪为1 anger
args:
  databaseDir: 数据库根路径
  outputDir: 格式化后的根路径
"""

import os.path
import re
import shutil
from .path_helper import *
import logging

logger = logging.getLogger("rnn_embedding.data_preprocess")

def format_database_casia(database_dir, out_dir=None):
    EMOTION_CLASS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
    outdirs = {}
    dirs = os.listdir(database_dir)
    remove_file_in_dirs(".DS_Store","相同的文本201--250.txt")
    for dir in dirs:
        absolute_dir = os.path.join(database_dir, dir)
        if os.path.isdir(absolute_dir):
            for path in os.listdir(absolute_dir):
                if path in EMOTION_CLASS:
                    for i in range(201, 251, 1):
                        source_dir = os.path.join(absolute_dir, path)
                        filename = '%d.wav' % i
                        source_dir = os.path.join(source_dir, filename)
                        if out_dir:
                            outfilename = '%d_%d_%d.wav' % (
                            (dirs.index(dir) + 1), (i - 200), (EMOTION_CLASS.index(path) + 1))
                            output_dir = os.path.join(out_dir, outfilename)
                            outdirs[source_dir] = output_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for (key, value) in list(outdirs.items()):
        shutil.copyfile(key, value)

    logger.info("CASIA 已经整理生成%d条数据" % (len(outdirs)))

"""
--------------------------------------
Speakers
--------------------------------------
'DC', 'JE', 'JK' and 'KL' are four male speakers recorded for the SAVEE database


--------------------------------------
Audio data
--------------------------------------
Audio files consist of audio WAV files sampled at 44.1 kHz

There are 15 sentences for each of the 7 emotion categories.
The initial letter(s) of the file name represents the emotion class, and the following digits represent the sentence number.
The letters 'a', 'd', 'f', 'h', 'n', 'sa' and 'su' represent 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise' emotion classes respectively.
E.g., 'd03.wav' is the 3rd disgust sentence.

15*7*4 = 420

这里的中性有30个句子 所以总共有 15*4 + 420 = 480 个句子
"""


def format_database_savee(database_dir, out_dir=None):
    EMOTION_CLASS1 = ["a", "f", "h", "n", "sa", "su", "d"]
    outdirs = {}
    dirs = os.listdir(database_dir)
    remove_file_in_dirs(dirs,".DS_Store","Info.txt")
    rec = re.compile(r"([a-z]*)(\d*)\.wav")
    for dir in dirs:
        absolute_dir = os.path.join(database_dir, dir)
        if os.path.isdir(absolute_dir):
            for path in os.listdir(absolute_dir):
                source_dir = os.path.join(absolute_dir, path)
                m = rec.search(path)
                if m is not None:
                    classes = m.group(1)
                    text = m.group(2)
                    filename = "%d_%d_%d.wav" % (dirs.index(dir) + 1, int(text), EMOTION_CLASS1.index(classes) + 1)
                    dest_dir = os.path.join(out_dir, filename)
                    outdirs[source_dir] = dest_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for (key, value) in list(outdirs.items()):
        shutil.copyfile(key, value)

    logger.info("SAVEE 已经整理生成%d条数据" % (len(outdirs)))


def format_database_emo(database_dir, out_dir=None):
    EMOTION_CLASS = ["W", "A", "F", "N", "T", "L", "E"]
    PERSON_ID = ["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    CODE_ID = ["a01", "a02", "a04", "a05", "a07", "b01", "b02", "b03", "b09", "b10"]

    outdirs = {}
    rec = re.compile(r"(.{2})(.{3})(.{1})(.*)")
    file_dir = os.listdir(database_dir)
    remove_file_in_dirs(file_dir, ".DS_Store")
    for file in file_dir:
        m = rec.match(file)
        if m is not None:
            person_id = m.group(1)
            code_id = m.group(2)
            emotion_class = m.group(3)
            version = m.group(4)
            newfilename = "%d_%d_%d_%s" % (PERSON_ID.index(person_id) + 1,
                                           CODE_ID.index(code_id) + 1,
                                           EMOTION_CLASS.index(emotion_class) + 1,
                                           version)
            source_dir = os.path.join(database_dir, file)
            outdirs[source_dir] = os.path.join(out_dir, newfilename)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for (key, value) in list(outdirs.items()):
        shutil.copyfile(key, value)

    logger.info( "Emo 已经整理生成%d条数据" % (len(outdirs)))


"""
sumary:
  将database_dir 处的音频数据库统一命名，然后放置到output_dir处
"""


def format_database(database_dir, output_dir):
    dirs = os.listdir(database_dir)
    remove_file_in_dirs(dirs, ".DS_Store")
    if os.path.exists(output_dir):
        return
    for dir in dirs:
        source_dir = os.path.join(database_dir, dir)
        out_dir = os.path.join(output_dir, dir)
        if dir == "CASIA":
            format_database_casia(source_dir, out_dir)
        if dir == "SAVEE":
            format_database_savee(source_dir, out_dir)
        if dir == "Emo":
            format_database_emo(source_dir, out_dir)


if __name__ == "__main__":
    format_database("/Users/jw/Desktop/audio_data/raw_wave_data", "/Users/jw/Desktop/audio_data/wave_data")

