'''
cnn模型
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import math


def cnn_model(WINDOWSIZE, num_classes):

    # np.random.seed(random_seed)
    # tf.random.set_seed(random_seed)

    model = keras.Sequential([
        
        # Conv1D：一维卷积层。32：卷积核的数量，即输出的维度。6：卷积核的大小。activation="relu"：使用 ReLU 激活函数。padding="same"：填充方式使输出的长度与输入的长度相同。
        # input_shape=(int(200 * WINDOWSIZE), 1)：输入形状。假设输入信号的长度是 200 * WINDOWSIZE，并且是单通道数据。
        Conv1D(32, 6, activation="relu", padding="same", input_shape=(int(200 * WINDOWSIZE), 1)),  
        # MaxPool1D：一维最大池化层。pool_size=2：池化窗口大小。strides=2：步长。池化窗口每次移动两个位置。
        MaxPool1D(pool_size=2, strides=2),
        Conv1D(64, 6, activation="relu", padding="same"),
        MaxPool1D(pool_size=2, strides=2),
        Flatten(),
        # Dense：全连接层。64：输出维度。activation="relu"：使用 ReLU 激活函数。
        Dense(64, activation="relu"),
        # Dropout：Dropout 正则化层。0.5：Dropout 率。每次训练时随机丢弃 50% 的神经元，以防止过拟合。
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.5),
        # Dense：全连接层。num_classes：输出维度，对应分类的类别数。activation="softmax"：使用 Softmax 激活函数，输出一个概率分布。
        Dense(num_classes, activation="softmax")
    ])
    # optimizer="Adam"：使用 Adam 优化器。loss="categorical_crossentropy"：使用分类交叉熵损失函数。metrics=["accuracy"]：评估指标为准确率。
    model.compile(optimizer="Adam", 
                loss="categorical_crossentropy", 
                metrics=["accuracy"]) 
    # 输出模型摘要
    model.summary()

    return model