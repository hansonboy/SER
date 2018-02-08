#!/usr/lib/env python
#coding:utf-8
"""
description:
  实现将音频数据转化为语谱图的功能
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os
import shutil
import logging
import pprint
import re

logger = logging.getLogger("rnn_embedding.wave_data_preprocess")
""" short time fourier transform of audio signal """

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(cols, int(frameSize)),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()

    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """

def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)  # ** factor
    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(
        [x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0 for x in scale])
    scale *= (freqbins - 1) / max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] //= totw[i]

    return newspec, freqs


""" plot spectrogram"""

def plotstft(audiopath, binsize=2**8, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1,overlapFac = 0.8):
    samplerate, samples = wav.read(audiopath)
    if len(samples.shape) > 1:
        samples = samples[:, channel]
    s = stft(samples, binsize,overlapFac)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    # print "ims.shape", ims.shape
    ims = np.transpose(ims)
    ims = ims[0:128,:]  # 人说话的频率在30Hz 到 3400Hz

    image = Image.fromarray(ims)
    image = image.convert('L')
    if not os.path.exists(os.path.split(name)[0]):
        os.makedirs(os.path.split(name)[0])
    image.save(name)


def raw_wave_2_png(raw_wav_data_dir, output_dir, binsize=2**8, overlapFac=0.8, alpha=1):
    logger.info(raw_wav_data_dir)
    dirs = os.listdir(raw_wav_data_dir)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    for dir in dirs:
        absolute_dir = os.path.join(raw_wav_data_dir, dir)
        if os.path.isdir(absolute_dir):
            for file in os.listdir(absolute_dir):
                source_file = os.path.join(absolute_dir, file)
                dest_file = os.path.join(output_dir,dir,file.split(".wav")[0] + ".png")
                plotstft(source_file, name=dest_file,binsize=binsize,overlapFac=overlapFac,alpha=alpha)
        logger.info("{} 已经生成了{}张图片".format(dir, len(os.listdir(absolute_dir))))


def testPlotstft():
    wav_file = "./bM0C26012.wav"
    plotstft(wav_file,name=wav_file.split(".wav")[0] + ".png",binsize=2**9,overlapFac=0.5)

def test_raw_wave_2_png():
    params = [
        {'alpha': 1,
         'binsize': 256,
         'overlapFac': 0.5,
         "wave_data_dir": "/Users/jw/Desktop/audio_data_test/wave_data",
         "dest_dir":"/Users/jw/Desktop/audio_data_test/png_data_256_0.5"
         },
        {'alpha': 1,
         'binsize': 256,
         'overlapFac': 0.8,
         "wave_data_dir": "/Users/jw/Desktop/audio_data_test/wave_data",
         "dest_dir": "/Users/jw/Desktop/audio_data_test/png_data_256_0.8"
         },
        {'alpha': 1,
         'binsize': 512,
         'overlapFac': 0.5,
         "wave_data_dir": "/Users/jw/Desktop/audio_data_test/wave_data",
         "dest_dir": "/Users/jw/Desktop/audio_data_test/png_data_512_0.5"
         }
    ]
    logging.info("读取参数完毕，共需要处理{}组参数".format(len(params)))
    n = 1
    for param in params:
        wav_data_dir = param["wave_data_dir"]
        dest_dir = param["dest_dir"]
        binsize =param["binsize"]
        overlapFac = param["overlapFac"]
        logger.info("binsize{} overlapFac{}".format(binsize, overlapFac))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with open(os.path.join(dest_dir,"params.txt"),"w") as p:
            pprint.pprint(param, p)
        raw_wave_2_png(wav_data_dir, dest_dir,binsize=binsize,overlapFac=overlapFac)
        logging.info("第{}组参数处理完成".format(n))
        n = n + 1


def count_wave_data(wave_data_dir):
    wav_result = {}
    for file in os.listdir(wave_data_dir):
        if re.match(r".*\.wav$",file):
            audiopath  = os.path.join(wave_data_dir, file)
            samplerate, samples = wav.read(audiopath)
            time = np.shape(samples)[0] // samplerate
            if wav_result.get("{}".format(time)) is None:
                wav_result["{}".format(time)] = 1
            else:
                wav_result["{}".format(time)] = wav_result["{}".format(time)] + 1
    result = []
    for i in range(len(wav_result) + 1):
        sum = wav_result.get("{}".format(i))
        if sum is  None:
            result.append(0)
        else :
            result.append(sum)
    return result




def test_count_wave_data():
    casia_count = count_wave_data("/Users/jw/Documents/MyProject/PythonWorkSpace/cnn_rnn_audio_emotion_recognition/audio_data/1488984665_512_0.5_64/wave_data/CASIA")
    emo_count = count_wave_data(
        "/Users/jw/Documents/MyProject/PythonWorkSpace/cnn_rnn_audio_emotion_recognition/audio_data/1488984665_512_0.5_64/wave_data/Emo")
    savee_count = count_wave_data(
        "/Users/jw/Documents/MyProject/PythonWorkSpace/cnn_rnn_audio_emotion_recognition/audio_data/1488984665_512_0.5_64/wave_data/SAVEE")
    print(casia_count, emo_count, savee_count)
    plt.plot(casia_count,'r-x',label='casia')
    plt.plot(emo_count, 'g-^', label='emo')
    plt.plot(savee_count, 'b-o', label='savee')
    plt.legend()
    plt.xlabel('time/s')
    plt.ylabel('num')
    plt.title('Speech database\'s length distribution')
    plt.show()


if __name__ == '__main__':
    # test_raw_wave_2_png()
    test_count_wave_data()




