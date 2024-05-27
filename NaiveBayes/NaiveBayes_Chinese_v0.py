import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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
    return ' '.join([word for word in words if word not in stopwords])

# 预处理文本数据
train_data['processed_comments'] = train_data['comments'].apply(preprocess_text)
test_data['processed_comments'] = test_data['comments'].apply(preprocess_text)

# 提取文本和标签
X_train = train_data['processed_comments']
y_train = train_data['rating']
X_test = test_data['processed_comments']
y_test = test_data['rating']

# 文本特征提取
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 初始化并训练分类器
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# 预测并评估
y_pred = nb_classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# 保存模型
joblib.dump(nb_classifier, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
