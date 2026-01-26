#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   use_model_2_test.py
@Time    :   2024/08/19 20:33:32
@Author  :   Zhiying Huang 
@Email   :   zhiying.huang.4g@stu.hosei.ac.jp
@description   :   使用已保存的模型进行测试
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


# 本地模块导入
import config

def model_2_test(data_to_model, save_address):
    '''
    data_to_model - 用于给模型测试的数据, 是一个dict
    save_address - 模型保存的地址
    '''

    # 加载模型
    model = tf.keras.models.load_model(save_address)

    true_labels = []
    predicted_labels = []

    # 使用model.predict方法对测试数据data_to_model['x_test']进行预测，获取预测结果predictions。
    predictions = model.predict(data_to_model['x_test'])
    # 使用np.argmax函数从data_to_model['y_test']中获取真实标签，沿轴1取最大值的索引（即类别）。
    true_labels = np.argmax(data_to_model['y_test'], axis=1)
    # 使用np.argmax函数从predictions中获取预测标签，沿轴1取最大值的索引（即类别）。
    predicted_labels = np.argmax(predictions, axis=1)

    confusion = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(confusion)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    return accuracy, confusion