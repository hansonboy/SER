#coding:utf-8
"""
description:
  增加计时功能
"""
import time
import logging
logger = logging.getLogger("rnn_embedding")
class time_w(object):
    __unitfactor = {'min': 1.0 / 60.0,
                    's': 1,
                    'ms': 1000,
                    'us': 1000000}
    def __init__(self,messgae, unit='s', precision=4):
        self.start = None
        self.end = None
        self.total = 0
        self.unit = unit
        self.precision = precision
        self.message = messgae

    def __enter__(self):
        if self.unit not in time_w.__unitfactor:
            raise KeyError('Unsupported time unit.')
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.total = (self.end - self.start) * time_w.__unitfactor[self.unit]
        self.total = round(self.total, self.precision)
        logger.info(self)

    def __str__(self):
        return '{}运行时间: {}{}'.format(self.message, self.total, self.unit)

if __name__ == '__main__':
    with time_w("测试") as t:
        print((1))