'''
lstm模型
'''
import time
import pandas as pd
import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tqdm import tqdm

import tensorflow_addons as tfa

import math

def lstm_model(WINDOWSIZE, num_classes, random_seed):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    model = keras.Sequential([
            LSTM(64, return_sequences=True, input_shape=(int(15.5*WINDOWSIZE), 1)),
            LSTM(64),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
    ])

    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                loss="categorical_crossentropy", 
                metrics=["accuracy"]) 
#     model.compile(optimizer=tfa.optimizers.AdamW(weight_decay=0.01), loss="categorical_crossentropy", metrics=["accuracy"])

    
    model.summary()
    return model