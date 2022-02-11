'''
功能：通过名词识别事件类模型训练
@author:宋杨
更新日期：2020-3-6
'''
import glob
import logging
from gensim.models import word2vec
import numpy as np
from numba import jit
import os
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import math
import time
from pyltp import NamedEntityRecognizer
from pyltp import Postagger
from pyltp import Segmentor

@jit
def probability_densitify(v):
    v = v - np.min(v) + 0.01
    return v/np.sum(v)

@jit
def softmax(v):
    v = v - torch.min(v) + 0.01
    return v/torch.sum(v)

def sen_vec_average(s):
    '''
    将所有单词的词向量相加求平均值，得到的向量即为句子的向量
    :param s:
    :return:
    '''
    words = s.split(" ")
    v = np.zeros(size)
    for word in words:
        try:
            v += model[word]
        except KeyError as ke:
            continue
    v /= len(words)
    return v

def sen_vec_compress(s):
    '''
    将所有单词的词向量相连并压缩，得到的向量即为句子的向量
    :param s:
    :return:
    '''
    words = s.split(" ")
    v = np.zeros(size)
    for i in range(size):
        sum = 0
        for word in words:
            try:
                sum += model[word][i]
            except KeyError as ke:
                continue
        v[i] = sum/len(words)
    return probability_densitify(v)

def readlines2vec(verb):
    return [sen_vec_compress(verb[line]) for line in range(len(verb))]

def readlines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]

def words2vec(words):
    return sen_vec_average(words)

def random_training_pair():
    category = random.choice(all_cate)#随机选择一种语言
    vec = random.choice(vec_cate[category])
    category_index = all_cate.index(category)
    return category, vec, category_index

def category_from_output(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0]
    return all_cate[category_i],category_i

@jit
def easy_ds(v1, v2):
    v1, v2 = (v1-np.min(v1)+1), (v2-np.min(v2)+1)
    v1, v2 = v1/np.sum(v1), v2/np.sum(v2)
    sum1 = sum2 = sum3 = 0
    for i in range(size):
        sum1 += v1[i]*math.log(v1[i], 2)
        sum2 += v2[i]*math.log(v2[i], 2)
        sum3 += (v1[i]+v2[i])*math.log((v1[i]+v2[i]), 2)
    return ((sum3-2)**2)*4/(sum1*sum2)

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

class my_softmax(nn.Module):
    def __init__(self, dim=None):
        super(my_softmax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return softmax(input)

def mylog(a, b):
    if a > 0:
        return math.log(a, b)
    else:
        return 0
def plt_vec(vec):
    records = np.zeros(size)
    for i in range(len(vec)):
        records += vec[i]
    records /= len(vec)
    plt.plot(records)
    plt.xlabel('')
    plt.ylabel('向量')
    plt.legend()
    plt.show()
    return
class mycost1(nn.Module):
    def __init__(self):
        super(mycost1, self).__init__()

    def forward(self, v1, v2):
        v1 = v1 - torch.min(v1)
        v1 /= torch.sum(v1)
        return torch.sum((v1-1/len_cate)*(v1-1/len_cate))

class mycost2(nn.Module):
    def __init__(self):
        super(mycost2, self).__init__()

    def forward(self, v1, v2):
        v1 = v1 + 0.01 - torch.min(v1)
        v1 /= torch.sum(v1)
        return torch.sum(v2 * (1/v1))

    #def backward(self):
        #return -torch.log2(v1+v2)-1

'''
class LSTMNetwork(nn.Module):#没写完
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTMNetwork, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, output_size)
        self.lstm = nn.LSTM(self.n_layers, self.hidden_size)
        self.fc = nn.Linear(input_size, output_size)
        self.logsoftmax = my_softmax()

    def forward(self, input, hidden=None):
        embedded = nn.Embedding(input)
        embedded = embedded.view(1, 1, -1)
'''
def one_hot(y):
    '''
    l = len_cate
    t = torch.FloatTensor(np.zeros(l))
    for i in range(l):
        if i == float(y[0]):
            t[i] = 1
    '''
    t = torch.LongTensor(np.zeros(len_cate))
    for i in range(len_cate):
        if i == float(y[0]):
            t[i] = 1
    #t = torch.reshape(t, (len(t), 1))
    return t

def adjust_learning_rate(optimizer, epoch):
    lr_t = 0.001
    lr_t = lr_t * (0.3 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_t

def train_model():
    #开始训练LSTM网络
    epochs = 210
    learning_rate = 0.001
    model = nn.Sequential(
        nn.LSTM(size, len_cate * 8),
        nn.ReLU(),
        nn.LSTM(len_cate * 8, len_cate),
        my_softmax(),
    )
    lstm = nn.LSTM(input_size=size, hidden_size=len_cate, num_layers=4, bidirectional=False)
    optimLSTM = torch.optim.Adam(lstm.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    records = []
    losses = []
    acc = []
    start = time.time()
    criterion1 = mycost1()
    criterion2 = mycost2()
    num = n_right = 0
    #开始训练
    for i in range(all_line_num * epochs):
        category, x, y = random_training_pair()
        x = Variable(torch.FloatTensor(x).view(1, 1, -1), requires_grad=True)
        y = Variable(torch.LongTensor(np.array([y])))
        optimLSTM.zero_grad()
        output, (hn, cn) = lstm(x)
        label = Variable(one_hot(y))
        guess, guess_i = category_from_output(output)
        num += 1
        if guess == category:
            correct = '√'
            n_right += 1
            loss1 = criterion1(output, label)  # 计算损失
            losses.append(loss1.data.numpy())
            loss1.backward()  # 反向传播
            optimLSTM.step()
        else:
            correct = '× (%s)' % category
            loss2 = criterion2(output, label)  # 计算损失
            losses.append(loss2.data.numpy())
            loss2.backward()  # 反向传播
            optimLSTM.step()
        if num == 100:
            trainging_process = 100 * (i + 1) / (all_line_num * epochs)
            trainging_process = '%.2f' % trainging_process
            print('第{}轮，训练损失:{:.8f}，训练进度：{}%，({})，预测事件类别：{}，正确？{} ,正确率：{}%' \
                  .format(int(i / all_line_num) + 1, np.mean(losses), float(trainging_process), time_since(start), \
                    guess, correct, n_right))
            records.append(np.mean(losses))
            acc.append(n_right / 100)
            num = n_right = 0

        # 绘制误差和正确率曲线
    plt.plot(records, label='Train Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(acc, label='Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    #保存模型
    # 保存模型
    state = {'model': lstm.state_dict(), 'optimizer': optimLSTM.state_dict(), 'epoch': epochs}
    torch.save(lstm.state_dict(), 'event_n_ana.mdl')

def t_model():
    #开始训练LSTM网络
    epochs = 90
    lstm = nn.LSTM(input_size=size, hidden_size=len_cate, num_layers=4, bidirectional=False)
    lstm.load_state_dict(torch.load('event_n_ana.mdl'))
    start = time.time()
    num = n_right = 0
    #开始训练
    for i in range(all_line_num * epochs):
        category, x, y = random_training_pair()
        x = Variable(torch.FloatTensor(x).view(1, 1, -1), requires_grad=True)
        output, (hn, cn) = lstm(x)
        guess, guess_i = category_from_output(output)
        num += 1
        if guess == category:
            n_right += 1
    print('第{}轮，({})，正确率：{:.2f}%' \
            .format(int(i / all_line_num) + 1,   time_since(start), \
             100 * n_right/num))

if __name__ == "__main__" :
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 使用gensim中的word2vec模块
    size = 256
    model = word2vec.Word2Vec.load("model/word2vec_baike")
    LTP_DATA_DIR = "../ltp_data_v3.4.0/"  # ltp模型目录的路径，根据实际情况修改
    cws_model_path = os.path.join(LTP_DATA_DIR,
                                  'cws.model')  # 分词模型路径，模型名称为`cws.model`
    pos_model_path = os.path.join(LTP_DATA_DIR,
                                  'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    ner_model_path = os.path.join(LTP_DATA_DIR,
                                  'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`
    data_path = 'data'
    all_filenames = glob.glob(data_path + '/*.txt')
    segmentor = Segmentor()  # 初始化分词实例
    segmentor.load_with_lexicon(cws_model_path, 'dict')  # 加载分词模型，以及自定义词典
    recognizer = NamedEntityRecognizer()  # 初始化命名实体识别实例
    recognizer.load(ner_model_path)  # 加载模型
    postagger = Postagger()  # 初始化词性标注实例
    postagger.load(pos_model_path)  # 加载模型
    # 构建向量字典
    vec_cate = {}
    all_cate = []
    for filename in all_filenames:
        verb = []
        #取出每个文件的文件名
        categories = filename.split('\\')[-1].split('.')[0]
        #将事件类型名称加入到列表中
        all_cate.append(categories)
        for line in readlines(filename):
            sen = ''
            seg_list = segmentor.segment(line)  # 分词
            seg_list = list(seg_list)  # 返回值并不是list类型，因此需要转换为list
            # LTP不能很好地处理回车，因此需要去除回车给分词带来的干扰。
            # LTP也不能很好地处理数字，可能把一串数字分成好几个单词，因此需要连接可能拆开的数字
            postags = postagger.postag(seg_list)
            netags = recognizer.recognize(seg_list, postags)  # 命名实体识别
            for post, word, netag in zip(postags, seg_list, netags):
                if post[0] == 'n' and netag == 'O':  # 词性标注
                     sen = sen + word + " "
            verb.append(sen)
        try:
            vec_cate[categories] = readlines2vec(verb)
        except KeyError:
            continue
    segmentor.release()
    postagger.release()
    recognizer.release()
    len_cate = len(all_cate)
    all_line_num = 0
    for key in vec_cate:
        all_line_num += len(vec_cate[key])
    print('###step finish###')
    train_model()
    print('###step finish###')
    t_model()
    print('###step finish###')


