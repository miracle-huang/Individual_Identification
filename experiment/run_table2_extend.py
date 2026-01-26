#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_table2_extend.py
@Time    :   2024/08/19 17:12:07
@Author  :   Zhiying Huang 
@Email   :   zhiying.huang.4g@stu.hosei.ac.jp
@description   :   使用另外的两组人数来跑table 2, 判断人数是否会对实验结果有影响
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

def table2_extend_train_test(train_video_list, test_video_list):
    subject_list_dict = {
        'subject_10to20' : list(range(11, 21)),
        'subject_20to30' : list(range(21, 31))
    }
    for subject_key, subject_value in subject_list_dict.items():
        for train_video in train_video_list:
            for test_video in test_video_list:
                train_video_number = train_video[0].split('-')[1]
                save_folder = f'result/table2/extend/{subject_key}/train_video{train_video_number}'
                model_save_folder = f'model/save_model/table2/extend/{subject_key}/train_video{train_video_number}/{train_video[0]}_train_{test_video[0]}'
                print('save_folder:', save_folder)
                print('model_save_folder:', model_save_folder)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                if not os.path.exists(model_save_folder):
                    os.makedirs(model_save_folder)

                xlsx_name = os.path.join(save_folder, f'{train_video[0]}_train_{test_video[0]}_test.xlsx')
                wb = Workbook()
                ws = wb.active
                ws.append(['cross_validation', 'accuracy', 'matrix'])
                chunk_dict_train = load_all_data(
                    subject_value, train_video, window_size = WINDOW_SIZE, sliding_size = SLIDING_SIZE)
                chunk_dict_test = load_all_data(
                    subject_value, test_video, window_size = WINDOW_SIZE, sliding_size = SLIDING_SIZE)

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
                    model = cnn_model(WINDOW_SIZE, len(subject_value))

                    save_address = model_save_folder + f'/cross_validation{i}.h5'
                    train_and_save(model, data_to_model, save_address) # 保存模型
                    accuracy, matrix = model_2_test(data_to_model, save_address) # 用保存好的模型test
                    ws.append([f'{i}', f'{accuracy}', f'{matrix}'])
                wb.save(xlsx_name)

def run_table2_extend():
    video_list_1 = [['amusing-1'], ['boring-1'], ['relaxed-1'], ['scary-1']]
    video_list_2 = [['amusing-2'], ['boring-2'], ['relaxed-2'], ['scary-2']]

    table2_extend_train_test(train_video_list = video_list_1, test_video_list = video_list_2)
    table2_extend_train_test(train_video_list = video_list_2, test_video_list = video_list_1)

if __name__ == "__main__":
    run_table2_extend()