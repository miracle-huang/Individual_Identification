import glob
import pandas as pd
import os

# 指定文件夹路径
folder_path = 'result/table2/extend/subject_21to30/train_video2'

# 使用glob模块获取文件夹下所有的xlsx文件路径
file_paths = glob.glob(f"{folder_path}/*.xlsx")

# 创建一个字典来存储平均accuracy，以train和test为行列坐标
accuracy_matrix = {}

# 遍历每个文件路径
for file in file_paths:
    # 提取文件名并去掉扩展名
    file_name = os.path.basename(file).replace('.xlsx', '')
    
    # 分割文件名，提取其中的两部分
    parts = file_name.split('_')
    part1 = parts[0] + '_' + parts[1]  # 获取amusing-1_train
    part2 = parts[2] + '_' + parts[3]  # 获取amusing-2_test
    
    # 读取xlsx文件
    df = pd.read_excel(file)
    
    # 计算accuracy列的平均值
    if 'accuracy' in df.columns:
        average_accuracy = round(df['accuracy'].mean() * 100, 2)
        
        # 将结果存入字典，以 part1 为行坐标，part2 为列坐标
        if part2 not in accuracy_matrix:
            accuracy_matrix[part2] = {}
        accuracy_matrix[part2][part1] = average_accuracy
    else:
        print(f"文件 {file} 中没有 'accuracy' 列。")

# 将字典转换为DataFrame，以part1作为行索引，以part2作为列索引
df_accuracy = pd.DataFrame.from_dict(accuracy_matrix, orient='index')

# 将结果输出到一个新的Excel文件
output_path = 'result/table2/extend/subject_21to30/train_video_2_result.xlsx'
df_accuracy.to_excel(output_path)

print(f"结果已保存到 {output_path}")