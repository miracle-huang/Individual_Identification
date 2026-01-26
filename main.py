#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/08/14 22:24:15
@Author  :   HuangZhiying 
@Email    :   zhiying.huang.4g@stu.hosei.ac.jp
@Description    :   主函数，程序的最初入口
'''

# 标准库导入
import time
import random
import os

# 第三方库导入
import tensorflow as tf
import numpy as np

# 本地模块导入
from experiment.run_table1 import run_table1
from experiment.run_table2_extend import run_table2_extend
from experiment.run_table3 import run_table3
from experiment.run_table4 import run_table4
import experiment.run_table5 as rt5

# 禁用多线程和多进程
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

# 确保设置生效
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 打印当前设置的线程数
inter_op_threads = tf.config.threading.get_inter_op_parallelism_threads()
intra_op_threads = tf.config.threading.get_intra_op_parallelism_threads()
print("Inter-op threads:", inter_op_threads)
print("Intra-op threads:", intra_op_threads)

# 控制GPU随机性
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# 本次研究不重复随机种子，因此只设置一个随机种子 3047
# 来源于文章 torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision
# 仅作为彩蛋，并无实际意义
os.environ['PYTHONHASHSEED'] = str(3047)
tf.random.set_seed(3047)
np.random.seed(3047)
random.seed(3047)

if __name__ == "__main__":
    start_time = time.time()

    rt5.save_model_table5()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time} 秒")