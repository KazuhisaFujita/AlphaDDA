#---------------------------------------
#Since : 2019/04/24
#Update: 2019/07/25
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np

class RingBuffer:
    def __init__(self, buf_size):
        self.size = buf_size
        self.buf = []
        for i in range(self.size):
            self.buf.append([])
        self.start = 0
        self.end = 0

    def add(self, el):
        self.buf[self.end] = el
        self.end = (self.end + 1) % self.size
        if self.end == self.start:
            self.start = (self.start + 1) % self.size

    def Get_buffer(self):
        array = []
        for i in range(self.size):
            buf_num = (self.end - i) % self.size
            array.append(self.buf[buf_num])
        return array

    def Get_buffer_start_end(self):
        array = []
        for i in range(self.size):
            buf_num = (self.start + i) % self.size
            if self.buf[buf_num] == []:
                return array
            array.append(self.buf[buf_num])
        return array


    def get(self):
        val = self.buf[self.start]
        self.start =(self.start + 1) % self.size
        return val
