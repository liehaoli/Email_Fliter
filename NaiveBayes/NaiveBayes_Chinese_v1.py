# 代码改进点：
# 数据预处理：过滤掉长度小于1的词汇，因为它们可能是无意义的单字。
# 特征提取：调整TfidfVectorizer的参数，如max_df、min_df和ngram_range。
# 模型调优：使用GridSearchCV来调节alpha参数以找到最优模型。
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 加载训练集和测试集数据
train_data = pd.read_csv('/home/u2021213473/jupyterlab/Email_filter/Naive Bayes/data/clean_train.csv')
test_data = pd.read_csv('/home/u2021213473/jupyterlab/Email_filter/Naive Bayes/data/clean_test.csv')

# 假设停用词文件为stopwords.txt，每行一个停用词
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().strip().split('\n'))
    return stopwords

stopwords = load_stopwords('/home/u2021213473/jupyterlab/Email_filter/Naive Bayes/data/stop/stopword.txt')

def preprocess_text(text):
    words = jieba.cut(text)
    return ' '.join([word for word in words if word not in stopwords and len(word) > 1])

# 预处理文本数据
train_data['processed_comments'] = train_data['comments'].apply(preprocess_text)
test_data['processed_comments'] = test_data['comments'].apply(preprocess_text)

# 提取文本和标签
X_train = train_data['processed_comments']
y_train = train_data['rating']
X_test = test_data['processed_comments']
y_test = test_data['rating']

# 文本特征提取
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 初始化并训练分类器
nb_classifier = MultinomialNB()

# 使用GridSearchCV调优超参数
param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0]}
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_vec, y_train)

# 获取最佳模型
best_nb_classifier = grid_search.best_estimator_

# 预测并评估
y_pred = best_nb_classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# 使用 zero_division 参数计算分类报告
report = classification_report(y_test, y_pred, zero_division=0)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# 保存模型
joblib.dump(best_nb_classifier, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
