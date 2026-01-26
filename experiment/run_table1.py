#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   run_table1.py
@Time    :   2024/08/14 22:32:16
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   Identification under same state and period (Table 1)
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

def run_table1():
    video_list = [['amusing-1'], ['boring-1'], ['relaxed-1'], ['scary-1']]
    # video_list = [['boring-1']]
    window_size_list = [2, 4, 6, 8]
    # window_size_list = [4]
    sliding_size_list = [10, 50, 100]
    # sliding_size_list = [10, 50]
    subject_amount_list = [config.subject_amount_10]

    for video in video_list:
        for window_size in window_size_list:
            for sliding_size in sliding_size_list:
                for subject_amount in subject_amount_list:
                    save_folder = f'result/table1/without_randomseed/{video[0]}/'
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    xlsx_name = os.path.join(save_folder, f'window_size{window_size}_sliding_size{sliding_size}_subject_{len(subject_amount)}.xlsx')
                    wb = Workbook()
                    ws = wb.active
                    ws.append(['cross_validation', 'accuracy', 'matrix'])

                    chunk_dict = load_all_data(
                        subject_amount, video, window_size = window_size, sliding_size = sliding_size)

                    for i in range(1, 6):
                        print('cross validation ', i)

                        validation = i
                        test = (i % 5) + 1  # 当 i=5 时，(5 % 5) + 1 = 1，test 为 1
                        train = [j for j in range(1, 6) if j != validation and j != test]
                            
                        x_train = np.concatenate((chunk_dict[f'{train[0]}'][0], chunk_dict[f'{train[1]}'][0], chunk_dict[f'{train[2]}'][0]))
                        y_train = np.concatenate((chunk_dict[f'{train[0]}'][1], chunk_dict[f'{train[1]}'][1], chunk_dict[f'{train[2]}'][1]))
                        x_val = chunk_dict[f'{validation}'][0]
                        y_val = chunk_dict[f'{validation}'][1]
                        x_test = chunk_dict[f'{test}'][0]
                        y_test = chunk_dict[f'{test}'][1]

                        data_to_model = {
                            'x_train' : x_train,
                            'y_train' : y_train,
                            'x_val' : x_val,
                            'y_val' : y_val,
                            'x_test' : x_test,
                            'y_test' : y_test,
                        }
                        model = cnn_model(window_size, len(subject_amount))
                        accuracy, matrix = run(model, data_to_model)
                        ws.append([f'{i}', f'{accuracy}', f'{matrix}'])
                    wb.save(xlsx_name)
