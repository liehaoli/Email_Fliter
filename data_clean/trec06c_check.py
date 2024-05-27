import pandas as pd 
file = pd.read_csv(r'E:\bert_email\train.csv', encoding_errors='ignore')
print(file.head())