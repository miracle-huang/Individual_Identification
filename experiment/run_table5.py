#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   run_table5.py
@Time    :   2024/08/18 12:39:37
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   Identification in Different Test Time Periods (Table 5)
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
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from openpyxl import Workbook

# 本地模块导入
import config
from load_data.load import load_all_data
from model.cnn_model import cnn_model
from model.train import run
from model.train_model_and_save import train_and_save
from model.use_model_2_test import model_2_test

WINDOW_SIZE = 2 # 已经定了窗口为2s
SLIDING_SIZE = 10 # 已经定了

def save_model(train_data, window_size, folder_name):
    print('begin training:', folder_name)
    save_folder = f'model/save_model/table_5/{folder_name}/'
    if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    for i in range(1, 6):
        print('cross validation ', i)
        validation = i
        test = (i % 5) + 1  # 当 i=5 时，(5 % 5) + 1 = 1，test 为 1
        train = [j for j in range(1, 6) if j != validation and j != test]
            
        x_train = np.concatenate((train_data[f'{train[0]}'][0], train_data[f'{train[1]}'][0], train_data[f'{train[2]}'][0]))
        y_train = np.concatenate((train_data[f'{train[0]}'][1], train_data[f'{train[1]}'][1], train_data[f'{train[2]}'][1]))
        x_val = train_data[f'{validation}'][0]
        y_val = train_data[f'{validation}'][1]

        data_to_model = {
            'x_train' : x_train,
            'y_train' : y_train,
            'x_val' : x_val,
            'y_val' : y_val,
        }
        model = cnn_model(window_size, len(config.subject_amount_30))

        save_address = save_folder + f'/cross_validation{i}.h5'

        print('save_address: ', save_address)

        train_and_save(model, data_to_model, save_address)

def save_model_table5():
    video_list_1 = [['amusing-1'], ['boring-1'], ['relaxed-1'], ['scary-1']]
    video_list_2 = [['amusing-2'], ['boring-2'], ['relaxed-2'], ['scary-2']]
    window_size = 2
    sliding_size = 10
    test_time_list = [3, 4, 5, 10, 20]

    video_list_1d_1 = [item for sublist in video_list_1 for item in sublist]
    video_list_1d_2 = [item for sublist in video_list_2 for item in sublist]
    print('train1:')
    chunk_dict_train_1 = load_all_data(
                config.subject_amount_30, video_list_1d_1, window_size = window_size, sliding_size = sliding_size)
    print('train2:')
    chunk_dict_train_2 = load_all_data(
                config.subject_amount_30, video_list_1d_2, window_size = window_size, sliding_size = sliding_size)

    # 用编号1的video训练，保存训练模型
    save_model(chunk_dict_train_1, window_size, 'model_from_all_number1_video_30_subjects')
    # 用编号2的video训练，保存训练模型
    save_model(chunk_dict_train_2, window_size, 'model_from_all_number2_video_30_subjects')

def test_use_model(test_video, model_video_number, chunk_time):
    '''
    test_video - 用于测试的video, 为list
    model_video_number - 用哪个model, 1或者2
    chunk_time - 每个chunk的时间有多长
    '''
    print('test video: ', test_video[0])
    print()

    chunk_dict_test = load_all_data(
        config.subject_amount_30, test_video, window_size = WINDOW_SIZE, sliding_size = SLIDING_SIZE, divide_time = chunk_time)
    
    chunk_num = len(chunk_dict_test) # 总共能分多少个时间块

    # 每个cross当中有多少个时间块，chunk_time大于22s直接=1
    if chunk_time > 22:
        chunk_in_cross = 1
    else:
        chunk_in_cross = chunk_num // 5

    save_folder = f'result/table5/model_from_all_number{model_video_number}_video/test_time{chunk_time}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    wb_accuracy = Workbook()
    ws_accuracy = wb_accuracy.active
    ws_accuracy.append(['cross_validation', 'accuracy'])
    xlsx_accuracy_name = os.path.join(save_folder, f'test_{test_video[0]}.xlsx')

    for i in range(1, 4): 
        # 循环五个cross的model，使用30s数据时只循环三次
        print(f'cross validation {i}')
        print()

        model_address = f'model/save_model/table_5/model_from_all_number{model_video_number}_video_30_subjects/cross_validation{i}.h5'

        save_matrix_folder = f'result/table5/model_from_all_number{model_video_number}_video/matrix/{test_video[0]}/'
        if not os.path.exists(save_matrix_folder):
            os.makedirs(save_matrix_folder)
        xlsx_matrix_name = os.path.join(save_matrix_folder, f'matrix_in_cross_validation{i}.xlsx')
        wb_matrix = Workbook()
        ws_matrix = wb_matrix.active
        ws_matrix.append(['chunk_no', 'accuracy', 'matrix'])

        accuracy_in_cross_list = []
        for j in range(1, chunk_in_cross + 1):
            # 循环每个cross当中的时间块
            x_test = chunk_dict_test[f'{i*j}'][0]
            y_test = chunk_dict_test[f'{i*j}'][1]

            data_to_test = {
                'x_test' : x_test,
                'y_test' : y_test,
            }
            
            accuracy, matrix = model_2_test(data_to_test, model_address)
            accuracy_in_cross_list.append(accuracy)
            ws_matrix.append([f'{j}', f'{accuracy}', f'{matrix}'])
        wb_matrix.save(xlsx_matrix_name)
            
        accuracy_of_cross = sum(accuracy_in_cross_list) / len(accuracy_in_cross_list)
        accuracy_of_cross = round(accuracy_of_cross * 100, 2)

        ws_accuracy.append([f'{i}', f'{accuracy_of_cross}'])
    wb_accuracy.save(xlsx_accuracy_name)

if __name__ == "__main__":
    # for test_time in config.test_time_list:
    #     for test_video in config.video_list_1:
    #         test_use_model(test_video, 2, test_time)
    #     for test_video in config.video_list_2:
    #         test_use_model(test_video, 1, test_time)
        # for test_video in config.video_list_1:
        #     test_use_model(test_video, 2, 30)
        # for test_video in config.video_list_2:
        #     test_use_model(test_video, 1, 30)
    for test_video in config.video_list_2:
        test_use_model(test_video, 1, 3)