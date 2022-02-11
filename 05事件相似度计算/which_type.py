import logging
from gensim.models import word2vec
import numpy as np
import math
from numba import jit
import os
import re

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 使用gensim中的word2vec模块
size = 256
model = word2vec.Word2Vec.load("model/word2vec_baike")

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

@jit
def easy_ds(s1, s2):
    v1, v2 = sen_vec_average(s1), sen_vec_average(s2)
    v1, v2 = (v1-np.min(v1)+0.01), (v2-np.min(v2)+0.01)
    v1, v2 = v1/np.sum(v1), v2/np.sum(v2)
    sum1 = sum2 = sum3 = 0
    for i in range(size):
        sum1 += v1[i]*math.log(v1[i], 2)
        sum2 += v2[i]*math.log(v2[i], 2)
        sum3 += (v1[i]+v2[i])*math.log((v1[i]+v2[i]), 2)
    return (sum1*sum2)/((sum3-2)**2)*4

def cut_sentence(s):
    # 标点符号和特殊字母
    space_punctuation = '''，。、�；!！.,.：:（）()"“”《》·丶．/・-'''
    # 对原始数据进行预处理
    line = re.sub(u"（.*?）", "", s)  # 去除括号内注释
    line = re.sub("[%s]+" % space_punctuation, " ", line)  # 去除标点、特殊字母
    return line

def eds_sentence(s):
    path = "data"
    files = os.listdir(path)
    result = []
    name = []
    for file in files:
        with open(path+'/'+file, "r", encoding="utf-8") as f:
            contents = f.readlines()
            arr = np.zeros(len(contents))
            for i in range(len(contents)):
                # 使用矩阵存储所有案件之间的相似度
                arr[i] = easy_ds(contents[i].strip(), s.strip())
            result.append(np.mean(arr))
            name.append(file)
        f.close()
    return name[np.argsort(arr)[-1]][7:-4]

if __name__ == "__main__" :
    print("请输入句子，完成后按回车键")
    s = input()
    print("您输入句子的事件类别为：", eds_sentence(cut_sentence(s)))
