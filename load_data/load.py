#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   data_load.py
@Time    :   2024/08/14 22:49:29
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   用于加载数据的函数
'''
# 标准库导入
import os
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# 第三方库导入
import pandas as pd
import numpy as np

# 本地模块导入
import load_data.load_utils as utils
import config

def load_video(subject_list, video_list, **kwargs):
    '''
    将数据划分时间段并划分窗口

    subject_list - 参与实验的subject列表
    video_list - 需要处理的video列表

    **kwargs可能包含参数:
    window_size: 窗口大小, 单位为s
    sliding_size: 滑动距离, 单位为ms
    divide_time: 分割时间
    '''
    video_chunk_dic = {} # 按照video来存储chunk，一个chunk包含了所有subject该chunk的数据
    for video_name in video_list:
        subject_data_list = []
        for subject in subject_list:
            folder_name = f"data_{video_name}_downsampled"
            file_name = f"sub{subject}_{video_name}_downsampled.csv"
            file_path = os.path.join('data', folder_name, file_name)

            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"Warning: File '{file_path}' does not exist.")
                continue
                
            # 读取 CSV 文件
            data = pd.read_csv(file_path)

            data_chunk_list = None
            if 'divide_time' in kwargs:
                data_chunk_list = utils.divide_data_base_on_time(data, kwargs['divide_time']) # 将数据分割成时间段，这里会得到一个二维数组
            else:
                data_chunk_list = utils.divide_data_base_on_time(data)

            subject_data = []
            for chunk in data_chunk_list:
                chunk.loc[:, 'sub'] = subject
                data_windows = None
                if {'window_size', 'sliding_size'} <= kwargs.keys():
                    data_windows = utils.window_segmentation(chunk, kwargs['window_size'], kwargs['sliding_size'])
                else:
                    data_windows = utils.window_segmentation(chunk)
                subject_data.append(data_windows)

            subject_data_list.append(subject_data)
        
        for subject_data in subject_data_list:
            for index, data in enumerate(subject_data):
                video_chunk_dic.setdefault(f'{index + 1}', []).extend(data)
    return video_chunk_dic

def load_all_data(subject_list, video_list, **kwargs):
    '''
    将每个chunk的数据处理为适合神经网络训练的形式

    subject_list - 参与实验的subject列表
    video_list - 需要处理的video列表

    chunk_for_deeplearning_dic - 每个chunk为一个元组, 其中:
    0: x
    1: y
    2: num_classes
    '''
    print('subject: ', subject_list)
    print('video: ', video_list)
    print(kwargs)
    
    video_chunk_dic = load_video(subject_list, video_list, **kwargs)
    chunk_for_deeplearning_dic = {} # 存储已经被整理成适合神经网络训练的形式的数据
    for chunk_num, chunk_data in video_chunk_dic.items():
        after_settled_data = utils.data_settle(chunk_data, 'ecg')
        chunk_for_deeplearning_dic[chunk_num] = after_settled_data
    
    return chunk_for_deeplearning_dic
    

if __name__ == "__main__":
    load_all_data(config.subject_amount_10, ['amusing-1'], window_size = 4, sliding_size = 100)
    