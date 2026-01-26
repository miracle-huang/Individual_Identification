import pandas as pd

# 读取两张表的数据 (假设分别存储在两个Excel文件中)
table1 = pd.read_excel('result/table2/extend/subject_21to30/train_video_1_result.xlsx', index_col=0)
table2 = pd.read_excel('result/table2/extend/subject_21to30/train_video_2_result.xlsx', index_col=0)

# 将两个表格的数据存储到矩阵中
table1_matrix = table1.to_numpy()
table2_matrix = table2.to_numpy()

# 计算两张表两两对应数据的平均值
average_matrix = (table1_matrix + table2_matrix) / 2

# 新表格的行标签和列标签
cols = ['amusing_train', 'boring_train', 'relaxed_train', 'scary_train']
rows = ['amusing_test', 'boring_test', 'relaxed_test', 'scary_test']

# 创建一个新的DataFrame保存结果
average_df = pd.DataFrame(average_matrix, index=rows, columns=cols)

# 将结果输出到一个新的Excel文件
output_path = 'result/table2/extend/subject_21to30/sub21to30.xlsx'
average_df.to_excel(output_path)

print(f"平均值表格已保存到 {output_path}")
