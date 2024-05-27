import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('data/train.csv')
'''print(df_train.head())
print(df_train.shape)'''
#print(df_train.head())
# 观察数据是否平衡
'''print(df_train.rating.value_counts())
print(df_train.rating.value_counts() / df_train.shape[0] * 100)'''

'''print(train['rating'].value_counts())
print(train['rating'].value_counts() / train.shape[0] * 100)
plt.figure(figsize=(8, 4))
sns.countplot(x='rating', data=train)
plt.show()
#text = str(train['rating'][item])
#print(text.duplicated().sum())
#print(text)'''
# 查看重复数据
print(df_train.duplicated().sum())
print(df_train[df_train.duplicated()==True])
print(df_train[df_train.duplicated()==True].sort_index())
# 删除重复数据
index = df_train[df_train.duplicated() == True].index
df_train.drop(index, axis=0, inplace=True)
df_train.reset_index(inplace=True, drop=True)
print(df_train.duplicated().sum())
print(df_train)
#df_train.to_csv('data/clean.csv')
df = df_train
'''duplicated_rows = df_train[df_train.duplicated(keep=False)]
first_duplicate_index = duplicated_rows.index[0]
print(first_duplicate_index)'''

# 我们还需要关心的重复数据是text一样但是label不一样的数据。
'''print(df_train['comments'].duplicated().sum())
print(df_train[df_train['comments'].duplicated() == True])
# 查看例子
print(df_train[df_train['comments'] == df_train.iloc[856]['comments']])
print(df_train[df_train['comments'] == df_train.iloc[3096]['comments']])'''

# 去掉text一样但是label不一样的数据
index2 = df[df.comments.duplicated() == True].index
'''print(df.comments.duplicated())
print(df.comments.duplicated().sum())'''
df.drop(index2, axis=0, inplace=True)
df.reset_index(inplace=True, drop=True)
# 检查
print(df['comments'].duplicated().sum())  # 0
print(df)
df.to_csv('data/clean_train.csv')

df_test = pd.read_csv('data/test.csv')
index = df_test[df_test.duplicated() == True].index
df_test.drop(index, axis=0, inplace=True)
df_test.reset_index(inplace=True, drop=True)
df2 = df_test
index3 = df2[df2.comments.duplicated() == True].index
df2.drop(index3, axis=0, inplace=True)
df2.reset_index(inplace=True, drop=True)
df2.to_csv('data/clean_test.csv')