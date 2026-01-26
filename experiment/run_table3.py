#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   run_table3.py
@Time    :   2024/08/16 17:58:47
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   Identification Modeled by Multi Emotion States (Table 3)
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

def run_table3():
    '''
    根据table 1的结果, 选择表现最好的组合window size=4s, sliding size=10ms
    '''
    video_list_1 = [['amusing-1'], ['boring-1'], ['relaxed-1'], ['scary-1']]
    video_list_2 = [['amusing-2'], ['boring-2'], ['relaxed-2'], ['scary-2']]
    window_size = 2
    sliding_size = 10

    # 用编号1的video训练，编号2的video测试
    # for video in video_list_2:
    #     video_list_1d = [item for sublist in video_list_1 for item in sublist]
    #     print('train:')
    #     chunk_dict_train = load_all_data(
    #                 config.subject_amount_10, video_list_1d, window_size = window_size, sliding_size = sliding_size)
    #     print('test:')
    #     chunk_dict_test = load_all_data(
    #         config.subject_amount_10, video, window_size = window_size, sliding_size = sliding_size)
        
    #     save_folder = f'result/table3/number1train_number2test/{video[0]}/'
    #     if not os.path.exists(save_folder):
    #         os.makedirs(save_folder)
    #     xlsx_name = os.path.join(save_folder, f'window_size{window_size}_sliding_size{sliding_size}.xlsx')
    #     wb = Workbook()
    #     ws = wb.active
    #     ws.append(['cross_validation', 'accuracy', 'matrix'])
        
    #     for i in range(1, 6):
    #         print('cross validation ', i)
    #         validation = i
    #         test = (i % 5) + 1  # 当 i=5 时，(5 % 5) + 1 = 1，test 为 1
    #         train = [j for j in range(1, 6) if j != validation and j != test]
                
    #         x_train = np.concatenate((chunk_dict_train[f'{train[0]}'][0], chunk_dict_train[f'{train[1]}'][0], chunk_dict_train[f'{train[2]}'][0]))
    #         y_train = np.concatenate((chunk_dict_train[f'{train[0]}'][1], chunk_dict_train[f'{train[1]}'][1], chunk_dict_train[f'{train[2]}'][1]))
    #         x_val = chunk_dict_train[f'{validation}'][0]
    #         y_val = chunk_dict_train[f'{validation}'][1]
    #         x_test = chunk_dict_test[f'{test}'][0]
    #         y_test = chunk_dict_test[f'{test}'][1]

    #         data_to_model = {
    #             'x_train' : x_train,
    #             'y_train' : y_train,
    #             'x_val' : x_val,
    #             'y_val' : y_val,
    #             'x_test' : x_test,
    #             'y_test' : y_test,
    #         }
    #         model = cnn_model(window_size, len(config.subject_amount_10))
    #         accuracy, matrix = run(model, data_to_model)
    #         ws.append([f'{i}', f'{accuracy}', f'{matrix}'])
    #     wb.save(xlsx_name)

    # 用编号2的video训练，编号1的video测试
    for video in video_list_1:
        video_list_1d = [item for sublist in video_list_2 for item in sublist]
        print('train:')
        chunk_dict_train = load_all_data(
                    config.subject_amount_10, video_list_1d, window_size = window_size, sliding_size = sliding_size)
        print('test:')
        chunk_dict_test = load_all_data(
            config.subject_amount_10, video, window_size = window_size, sliding_size = sliding_size)
        
        save_folder = f'result/table3/number2train_number1test/{video[0]}/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        xlsx_name = os.path.join(save_folder, f'window_size{window_size}_sliding_size{sliding_size}.xlsx')
        wb = Workbook()
        ws = wb.active
        ws.append(['cross_validation', 'accuracy', 'matrix'])
        
        for i in range(1, 6):
            print('cross validation ', i)
            validation = i
            test = (i % 5) + 1  # 当 i=5 时，(5 % 5) + 1 = 1，test 为 1
            train = [j for j in range(1, 6) if j != validation and j != test]
                
            x_train = np.concatenate((chunk_dict_train[f'{train[0]}'][0], chunk_dict_train[f'{train[1]}'][0], chunk_dict_train[f'{train[2]}'][0]))
            y_train = np.concatenate((chunk_dict_train[f'{train[0]}'][1], chunk_dict_train[f'{train[1]}'][1], chunk_dict_train[f'{train[2]}'][1]))
            x_val = chunk_dict_train[f'{validation}'][0]
            y_val = chunk_dict_train[f'{validation}'][1]
            x_test = chunk_dict_test[f'{test}'][0]
            y_test = chunk_dict_test[f'{test}'][1]

            data_to_model = {
                'x_train' : x_train,
                'y_train' : y_train,
                'x_val' : x_val,
                'y_val' : y_val,
                'x_test' : x_test,
                'y_test' : y_test,
            }
            model = cnn_model(window_size, len(config.subject_amount_10))
            accuracy, matrix = run(model, data_to_model)
            ws.append([f'{i}', f'{accuracy}', f'{matrix}'])
        wb.save(xlsx_name)
            