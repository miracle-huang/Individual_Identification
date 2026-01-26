#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   load_utils.py
@Time    :   2024/08/14 22:19:43
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   供data_load调用的工具函数
'''

# 标准库导入
import os
import sys
import copy
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# 第三方库导入
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

# 本地模块导入

def divide_data_base_on_time(data, divide_time = 22):
    '''
    根据时间划分数据，默认为22s
    data - 用于划分的数据，来源于单个subject的单个video，默认22s
    divide_time - 每块数据的长度

    返回：
    chunk_list - chunk指一块数据，默认22s，将110s的数据分成5份
    '''
    sample_rate = 200  # 目前的采样率为200 Hz
    chunk_size = divide_time*sample_rate # 分成的每个数据块有多长

    chunk_list = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    return chunk_list

def window_segmentation(data, window_size = 2, sliding_size = 10):
    '''
    划分窗口

    入参：
    data - 一段时间内的数据
    window_size - 窗口大小，单位为s
    sliding_size - 滑动长度，单位为ms

    返回：
    segmented_signals - 已经切分好窗口的数据
    '''
    sliding_size = sliding_size/1000 # 把毫秒换算成秒
    window_data_size = int(200 * window_size)
    sliding_step = int(200 * sliding_size)

    segmented_signals = []
    
    start = 0
    while start + window_data_size <= len(data):
        end = start + window_data_size
        signal_ = copy.deepcopy(data.iloc[start:end])
        signal_ = signal_
        if len(signal_) == window_data_size:
            segmented_signals.append(signal_)
        start += sliding_step
    
    return segmented_signals

def one_hot_encoding(y_data):
    '''
    对lable进行独热编码
    '''
    classes = np.unique(y_data)
    num_classes = len(classes)
    y_data_sort = []
    for y in y_data:
        index = np.argmax(classes == y)
        y_data_sort.append(index)
    y_data_sort = np.array(y_data_sort)
    y_data_sort = to_categorical(y_data_sort, num_classes) 
    return y_data_sort, num_classes

def data_settle(data, signal_type):
    '''
    将数据整理成适合进行神经网络训练的形式
    '''
    x_ = []
    y_all = []
    y_ = []
    for i in data:
        temp_signal = i[signal_type].values
        temp_sub = i['sub'].values
        x_.append(temp_signal)
        y_all.append(temp_sub)
    x_ = np.array(x_)
    for i in y_all:
        y_.append(i[0])
    y_ = np.array(y_)
    y_, num_classes = one_hot_encoding(y_)
    return x_, y_, num_classes

