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

def run_table2():
    video_list_1 = [['amusing-1'], ['boring-1'], ['relaxed-1'], ['scary-1']]
    video_list_2 = [['amusing-2'], ['boring-2'], ['relaxed-2'], ['scary-2']]
    window_size_list = [2]
    sliding_size_list = [10]
    for video_1 in video_list_1:
        for video_2 in video_list_2:
            for window_size in window_size_list:
                for sliding_size in sliding_size_list:
                    save_folder = f'result/table2/{video_2[0]}_{video_1[0]}/'
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    xlsx_name = os.path.join(save_folder, f'window_size{window_size}_sliding_size{sliding_size}.xlsx')
                    wb = Workbook()
                    ws = wb.active
                    ws.append(['cross_validation', 'accuracy', 'matrix'])

                    chunk_dict_1 = load_all_data(
                    config.subject_amount_10, video_2, window_size=window_size, sliding_size=sliding_size)

                    chunk_dict_2 = load_all_data(
                    config.subject_amount_10, video_1, window_size=window_size, sliding_size=sliding_size)

                    for i in range(1, 6):
                        print('cross validation ', i)

                        validation = i
                        test = (i % 5) + 1  # 当 i=5 时，(5 % 5) + 1 = 1，test 为 1
                        train = [j for j in range(1, 6) if j != validation and j != test]
                            
                        x_train = np.concatenate((chunk_dict_1[f'{train[0]}'][0], chunk_dict_1[f'{train[1]}'][0], chunk_dict_1[f'{train[2]}'][0]))
                        y_train = np.concatenate((chunk_dict_1[f'{train[0]}'][1], chunk_dict_1[f'{train[1]}'][1], chunk_dict_1[f'{train[2]}'][1]))
                        x_val = chunk_dict_1[f'{validation}'][0]
                        y_val = chunk_dict_1[f'{validation}'][1]

                        x_test = chunk_dict_2[f'{test}'][0]
                        y_test = chunk_dict_2[f'{test}'][1]

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