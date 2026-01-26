import pandas as pd
import glob
import os

# 指定文件夹路径
folder_path = 'result/table1/without_randomseed/scary-1'

# 获取文件夹下所有的xlsx文件
xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

# 存储每个文件的平均 accuracy
accuracy_averages = {}

for file in xlsx_files:
    # 读取 Excel 文件
    df = pd.read_excel(file)
    
    # 检查是否存在 'accuracy' 列
    if 'accuracy' in df.columns:
        # 计算 accuracy 列的平均值
        accuracy_avg = df['accuracy'].mean()
        # 将平均值乘以100，并保留两位小数
        accuracy_avg = round(accuracy_avg * 100, 2)
        # 获取文件名（不包括路径）
        file_name = os.path.basename(file)
        # 将结果存储在字典中
        accuracy_averages[file_name] = accuracy_avg
    else:
        print(f"文件 {file} 不包含 'accuracy' 列")

# 打印每个文件的平均 accuracy
for file_name, accuracy_avg in accuracy_averages.items():
    print(f"文件 {file_name} 的 accuracy 平均值为: {accuracy_avg}%")
    print("")
