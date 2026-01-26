#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/08/15 15:49:46
@Author  :   Zhiying Huang 
@Email   :   zhiying.huang.4g@stu.hosei.ac.jp
@description   :   用于训练模型的函数
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

def run(model, data_to_model):
    # 学习率衰减函数
    def lr_scheduler(epoch, lr):
        decay_rate = 0.5  # 根据需要调整衰减率
        decay_step = 5   # 每5个epoch衰减一次
        if epoch % decay_step == 0 and epoch:      # 如果当前epoch是衰减步长的倍数且epoch不为0
            return lr * decay_rate       # 返回衰减后的学习率
        return lr      # 否则返回原始学习率

    '''
    创建学习率衰减和早停的回调函数
    在训练过程中监控和控制模型的行为，如在每个训练周期（epoch）或批次（batch）结束时执行特定操作。
    EarlyStopping：在验证指标不再提升时提前停止训练，防止过拟合。LearningRateScheduler：根据预定义的计划调整学习率。
    '''
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),             # 使用前面定义的lr_scheduler函数，并在每次epoch结束后输出学习率
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=16, verbose=1)   # 如果验证损失在16个epoch内没有改善，则停止训练。
    ]

    model.fit(
        data_to_model['x_train'],
        data_to_model['y_train'],
        batch_size = config.batch_size,
        epochs = config.epochs,
        validation_data=(data_to_model['x_val'], data_to_model['y_val']),
        callbacks = callbacks
    )


    # 在模型训练完成后保存模型
    # model_save_path = 'model_save/model_train_30.h5'
    # model.save(model_save_path)
    # print(f'Model saved at {model_save_path}')


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
        
