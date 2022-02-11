'''
功能：事件类型综合识别
@author:宋杨
更新日期：2020-3-6
'''

import glob
import logging
from gensim.models import word2vec
import numpy as np
from numba import jit
import os
import re,string
import torch
import torch.nn as nn
import torch.optim
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
def normalize(v):
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

def tensor_norm(tens):
    tens1 = tens - torch.min(tens)
    tens2 = tens1 / torch.sum(tens1)
    return tens2

def readlines2vec(lines):
    return [sen_vec_compress(lines[line]) for line in range(len(lines))]

def readlines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]

def words2vec(words):
    return sen_vec_average(words)

def category_from_output(output):
    predict = []
    topv, topi = torch.topk(output.view(len_cate), len_cate, sorted=True)
    for i in range(len_cate):
        value = float(topv[i])
        category_i = topi[i]
        predict.append([value, all_cate[category_i]])
    return predict

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

def mylog(a):
    if a > 0:
        return math.log2(a)
    elif a < 0:
        return math.log2(-a)
    else:
        return 0

def wordpredict(word_vec, word_cha):
    lstm = nn.LSTM(input_size=size, hidden_size=len_cate, num_layers=4, bidirectional=False)
    if word_cha == 'n':
        lstm.load_state_dict(torch.load('event_n_ana.mdl'))
    else:
        lstm.load_state_dict(torch.load('event_v_ana.mdl'))
    #开始训练
    x = word_vec
    x = torch.FloatTensor(x).view(1, 1, -1)
    output, (hn, cn) = lstm(x)
    guess = category_from_output(output)
    return guess

def linepredict(line1, line2, line3, type = True, weight = [1, 1, 1]):
    print('名词权重：{:.2f}，句子权重：{:.2f}，动词权重{:.2f}'.format(weight[0], weight[1], weight[2]))
    if type:
        lstm1 = nn.LSTM(input_size=size, hidden_size=len_cate, num_layers=4, bidirectional=False)
        lstm1.load_state_dict(torch.load('event_n_ana.mdl'))
        lstm2 = nn.LSTM(input_size=size, hidden_size=len_cate, num_layers=4, bidirectional=False)
        lstm2.load_state_dict(torch.load('event_s_ana.mdl'))
        lstm3 = nn.LSTM(input_size=size, hidden_size=len_cate, num_layers=4, bidirectional=False)
        lstm3.load_state_dict(torch.load('event_v_ana.mdl'))
        x1 = torch.FloatTensor(line1).view(1, 1, -1)
        output1, (hn, cn) = lstm1(x1)
        x2 = torch.FloatTensor(line2).view(1, 1, -1)
        output2, (hn, cn) = lstm2(x2)
        x3 = torch.FloatTensor(line3).view(1, 1, -1)
        output3, (hn, cn) = lstm3(x3)
        output11 = tensor_norm(output1)
        output22 = tensor_norm(output2)
        output33 = tensor_norm(output3)
        guess = category_from_output(weight[0]*output11+weight[1]*output22+weight[2]*output33)
        return guess
    else:
        lstm = nn.LSTM(input_size=size, hidden_size=len_cate, num_layers=4, bidirectional=False)
        lstm.load_state_dict(torch.load('event_s_ana.mdl'))
        x = torch.FloatTensor(line1).view(1, 1, -1)
        output, (hn, cn) = lstm(x)
        output = tensor_norm(output)
        guess = category_from_output(output)
        return guess
    #开始训练

def word_predict(word):
    try:
        post = postagger.postag([word])
        netag = recognizer.recognize([word], post)  # 命名实体识别
        try:
            if post[0][0] == 'n' and netag[0] == 'O':  # 词性标注
                guess = wordpredict(probability_densitify(model[word]), 'n')
            elif post[0][0] == 'v':
                guess = wordpredict(probability_densitify(model[word]), 'v')
            else:
                guess = ['无法判断']
            return guess
        except IndexError as ie:
            return [ie]
    except KeyError as ke:
        return [ke]

def sentence_predict(s, weight):
    try:
        punctuations = '。，《》；‘’“”【】'
        line = re.sub('[%s]' % re.escape(string.punctuation), ' ', s)  # 去除标点、特殊字母
        line = re.sub('[%s]' % re.escape(punctuations), ' ', line)
        seg_list = list(segmentor.segment(line))  # 分词
        postags = postagger.postag(seg_list)
        netags = recognizer.recognize(seg_list, postags)  # 命名实体识别
        nsent = ''
        vsent = ''
        try:
            sum1 = sum2 = sum3 = 0
            line_vec = sen_vec_compress(line)
            for postag, word, netag in zip(postags, seg_list, netags):
                if postag[0] == 'n' and netag == 'O':  # 词性标注
                    nsent = nsent + word + ' '
                    sum1 += weight[0]
                elif postag[0] == 'v':
                    vsent = vsent + word + ' '
                    sum3 += weight[2]
                else:
                    sum2 += weight[1]
            weight = [sum1, sum2, sum3]
            if nsent != '' or vsent != '':
                guess = linepredict(sen_vec_compress(nsent), line_vec, sen_vec_compress(vsent), True, normalize(weight))
            else:
                guess = linepredict(line_vec, line_vec, line_vec, False, normalize(weight))
            return guess
        except IndexError as ie:
            return [0, ie]
    except KeyError as ke:
        return [0, ke]

def para_predict():
    return

def file_predict(filename):
    return

def path_predict(path):
    data_path = 'path'
    all_filenames = glob.glob(data_path + '/*.txt')
    # 构建向量字典
    vec_cate = {}
    all_cate = []
    for filename in all_filenames:
        verb = []
        # 取出每个文件的文件名
        categories = filename.split('\\')[-1].split('.')[0]
        # 将事件类型名称加入到列表中
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
    return

if __name__ == "__main__" :
    weight = [0.5, 0.1, 0.4]  # 权重参数，经验设定，由句子中各要素长度和初始权重决定，分别表示一般名词，句子和动词的权重
    all_cate = ['交通事故', '地震', '恐怖袭击', '火灾', '食物中毒']
    len_cate = len(all_cate)
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 使用gensim中的word2vec模块
    model = word2vec.Word2Vec.load("model/word2vec_baike")
    size = 256 #由model决定
    LTP_DATA_DIR = "../ltp_data_v3.4.0/"  # ltp模型目录的路径，根据实际情况修改
    cws_model_path = os.path.join(LTP_DATA_DIR,
                                  'cws.model')  # 分词模型路径，模型名称为`cws.model`
    pos_model_path = os.path.join(LTP_DATA_DIR,
                                  'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    ner_model_path = os.path.join(LTP_DATA_DIR,
                                  'ner.model')  # 命名实体识别模型路径，模型名称为`ner.model`
    segmentor = Segmentor()  # 初始化分词实例
    segmentor.load_with_lexicon(cws_model_path, 'dict')  # 加载分词模型，以及自定义词典
    recognizer = NamedEntityRecognizer()  # 初始化命名实体识别实例
    recognizer.load(ner_model_path)  # 加载模型
    postagger = Postagger()  # 初始化词性标注实例
    postagger.load(pos_model_path)  # 加载模型
    recognizer = NamedEntityRecognizer()  # 初始化命名实体识别实例
    recognizer.load(ner_model_path)  # 加载模型
    #初始化完毕
    #################事件类型预测
    while True:
        line = input('\n请输入一句话,输入0退出:')
        try:
            preguess = sentence_predict(line, weight)
        except:
            print('Error!')
            continue
        for i in range(len_cate):
            if preguess[i][0] > float(1/len_cate):
                print('事件类型：{}，概率：{:.2f}%' .format(preguess[i][1], 100 * preguess[i][0]))
        print('最有可能事件类型:{}'.format(preguess[0][1]))
        if line == '0':
            break
    #################选择不同功能
    #释放模型
    segmentor.release()
    postagger.release()
    recognizer.release()

