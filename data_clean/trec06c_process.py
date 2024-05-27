import re
# 通过index文件获取所有文件路径及标签值
def get_Label():
    mail_index = open(r"E:\bert_email\trec06c/full/index", "r", encoding="gb2312", errors='ignore')
    index_list = [t for t in mail_index]
    index_split = [x.split() for x in index_list if len(x.split()) == 2]  # 分割了标记和路径
    path_list = [y[1].replace('..', './trec06c') for y in index_split]
    label_list = [ 1 if y[0] == "spam" else 0 for y in index_split] #1：垃圾邮件；0：正常邮件
    return path_list, label_list
 
# 根据路径打开文件 并提取每个邮件中的文本
def get_Text(path):
    mail = open(path, "r", encoding="gb2312", errors='ignore')
    TextList = [text for text in mail]
    XindexList = [TextList.index(i) for i in TextList if re.match("[a-zA-Z0-9]", i)]
    text = ''.join(TextList[max(XindexList) + 1:]) #去除邮件头
    text = re.sub('\s+','',re.sub("\u3000","", re.sub("\n", "",text)))  #去空格分隔符及一些特殊字符
    return text
 
# 获取path、label列表
path_list, label_list = get_Label()
# 获取所有文本
content_list = [get_Text(Path) for Path in path_list]

import csv

# 将content_list和label_list组合成一个新的列表
combined_list = list(zip(content_list, label_list))

# 打开你想要写入的文件，并创建一个csv.writer对象
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # 写入列名
    writer.writerow(['content', 'label'])

    # 写入数据
    for item in combined_list:
        writer.writerow(item)