from sklearn.model_selection import train_test_split
import pandas as pd 
# 读取csv文件
file = pd.read_csv(r'E:\bert_email\output.csv', encoding_errors='ignore')

# 划分数据集
train, test = train_test_split(file, test_size=0.2, random_state=42)

# 查看训练集和测试集
print(train.head())
print(test.head())
# 保存训练集和测试集为.csv文件
train.to_csv(r'E:\bert_email\train.csv', index=False)
test.to_csv(r'E:\bert_email\test.csv', index=False)