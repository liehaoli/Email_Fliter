# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : predict.py
# @Project: Douban_Bert
# @CreateTime : 2022/3/13 上午12:08:22
# @Version：V 0.1
'''
模型训练和评估
'''
import numpy as np
from torch import nn
import time
import os
import torch
import logging
import torch.nn.functional as F

from torch.nn.functional import one_hot
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, BertModel
from torch.utils.data import DataLoader
from transformers.utils.notebook import format_time
from data_process import InputDataSet, read_data
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLSTM(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型,生成embedding层
        self.embedding = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        # 去掉移至gpu
        #冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # 生成下游RNN层以及全连接层
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=self.drop)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # 使用CrossEntropyLoss作为损失函数时，不需要激活。因为实际上CrossEntropyLoss将softmax-log-NLLLoss一并实现的。

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        # Access the first element of the tuple returned by the embedding
        out, (h_n, _) = self.lstm(embedded)
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.fc(output)
        return output



    # def save_pretrained(self, save_path):
    #     encoder_path = os.path.join(save_path, "encoder")
    #     dense_path = os.path.join(save_path, "project_layers")
    #     if not os.path.exists(encoder_path):
    #         os.mkdir(encoder_path)
    #     if not os.path.exists(dense_path):
    #         os.mkdir(dense_path)
    #     # save LSTM
    #     torch.save({'state_dict': self._encoder.state_dict()}, os.path.join(encoder_path, "Bi-LSTM.pth.tar"))
    #     # save project_layers
    #     torch.save({'state_dict': self._project_layer.state_dict()}, os.path.join(dense_path, "dense.pth.tar"))


def train(batch_size, EPOCHS):
    # model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=5)使用bert-base-chinese预训练模型

    model = BiLSTM(drop=0.3, hidden_dim=384, output_dim=5)

    # model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=5)#使用hfl/chinese-roberta-wwm-ext预训练模型

    train = read_data('data/clean_train.csv')
    val = read_data('data/clean_test.csv')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')

    train_dataset = InputDataSet(train, tokenizer, 64)
    val_dataset = InputDataSet(val, tokenizer, 64)

    train_dataloader = DataLoader(train_dataset, batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    total_steps = len(train_dataloader) * EPOCHS  # len(dataset)*epochs / batchsize
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    total_t0 = time.time()

    log = log_creater(output_dir='./cache/logs/')

    log.info("   Train batch size = {}".format(batch_size))
    log.info("   Total steps = {}".format(total_steps))
    log.info("   Training Start!")

    train_loss = []
    test_loss = []
    test_acc = []

    for epoch in range(EPOCHS):
        total_train_loss = 0
        t0 = time.time()
        model.to(device)
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()

            outputs = model.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            #one_hot_labels = one_hot(labels, num_classes=5)
            # 将one_hot_labels类型转换成float
            #one_hot_labels = one_hot_labels.to(dtype=torch.float)
            #labels = labels.to(dtype=torch.float)
            # 选择损失函数
            loss = nn.CrossEntropyLoss()(outputs, labels)
            #print(loss)
            total_train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_time = format_time(time.time() - t0)

        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch + 1, EPOCHS, avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')

        model.eval()  # 这里调用了eval方法
        avg_val_loss, avg_val_acc = evaluate(model, val_dataloader)
        val_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f}===='.format(epoch + 1, EPOCHS, avg_val_loss,
                                                                                       avg_val_acc))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('')

        if epoch == EPOCHS - 1:  # 保存模型

            output_dir = "./cache/"  # 定义保存路径
            '''
            第一种保存方法，不推荐这样写
            '''
            model_to_save = model.module if hasattr(model, 'module') else model
            # 如果使用预定义的名称保存，则可以使用`from_pretrained`加载
            output_model_file = os.path.join(output_dir, 'Bi-LSTM.bin')
            # output_config_file = os.path.join(output_dir, 'config.json')

            torch.save(model_to_save.state_dict(), output_model_file)
            # model_to_save.config.to_json_file(output_config_file)
            # tokenizer.save_vocabulary(output_dir)
            print('Model Saved!')

            # model_to_save = model.module if hasattr(model, 'module') else model
            # model_to_save.save_pretrained('cache')
            # print('Model Saved!')

        # 将数据保存到列表
        train_loss.append(avg_train_loss)
        test_loss.append(avg_val_loss)
        test_acc.append(avg_val_acc)

    log.info('')
    log.info('   Training Completed!')
    print('Total training took {:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))
    # 简单可视化
    x1 = range(0, EPOCHS)
    y1 = train_loss
    plt.plot(x1, y1)
    plt.title("train loss of chinese-bert-wwm")
    plt.xlabel("epoches")
    plt.ylabel("train loss")
    plt.savefig('./cache/wwm_1.png')
    plt.close()
    # plt.show()

    x2 = range(0, EPOCHS)
    y2 = test_loss
    plt.plot(x2, y2)
    plt.title("test loss of chinese-bert-wwm")
    plt.xlabel("epoches")
    plt.ylabel("test loss")
    plt.savefig('./cache/wwm_2.png')
    plt.close()
    # plt.show()

    x3 = range(0, EPOCHS)
    y3 = test_acc
    plt.plot(x3, y3)
    plt.title("test acc of chinese-bert-wwm")
    plt.xlabel("epoches")
    plt.ylabel("test acc")
    plt.savefig('./cache/wwm_3.png')
    plt.close()
    # plt.show()


def evaluate(model, val_dataloader):
    total_val_loss = 0
    corrects = []
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = torch.argmax(outputs, dim=1)
        ## 把每个batch预测的准确率加入到一个list中
        ## 在加入之前，preds和labels变成cpu的格式
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to('cpu').numpy()
        #print('outputs.logits type:', outputs.shape)
        #print('preds type:', preds.shape)
        corrects.append((preds == labels_ids).mean())  ## [0.8,0.7,0.9]
        ## 返回loss
        #one_hot_labels = one_hot(labels, num_classes=5)
        # 将one_hot_labels类型转换成float
        #one_hot_labels = one_hot_labels.to(dtype=torch.float)

        # 选择损失函数
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        ## 把每个batch的loss加入 total_val_loss
        ## 总共有len(val_dataloader)个batch
        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_acc = np.mean(corrects)

    return avg_val_loss, avg_val_acc


# 训练日志，复用性很高的代码
def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


if __name__ == '__main__':
    train(batch_size=32, EPOCHS=20)
