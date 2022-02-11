import logging
from gensim.models import word2vec
import numpy as np
from scipy import linalg
import math
import os

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

# 使用gensim中的word2vec模块
size = 256
#model = word2vec.Word2Vec.load("Word2vec.w2v")
model = word2vec.Word2Vec.load("Word2vec.w2v")
org_path = "data"
labels = os.listdir(org_path)

# req_count = 5
# for key in model.wv.similar_by_word('被告人', topn=100):
#     if len(key[0]) == 3:
#         req_count -= 1
#         print(key[0], key[1])
#         if req_count == 0:
#             break


def sentence_vector(s):
    '''
    将所有单词的词向量相加求平均值，得到的向量即为句子的向量
    :param s:
    :return:
    '''
    words = s.split(" ")
    v = np.zeros(size)
    for word in words:
        v += model[word]
    v /= len(words)
    return v

def mysentence_vectors(s):
    words = s.split(" ")
    v = np.zeros(size)
    for i in range(size):
        sum = 0
        for word in words:
            sum += model[word][i]
        v[i] = sum/len(words)
    return v

def vector_similarity(s1, s2):
    '''
    计算两个句子之间的相似度:将两个向量的夹角余弦值作为其相似度
    :param s1:
    :param s2:
    :return:
    '''
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))

def jizuobiao(s):
    v = np.zeros(size+1)
    sum = 0
    for i in range(size):
        sum += s[i]**2
    v[0] = sum**(1/2)
    for i in range(size-1):
        x, y = s[i], s[i+1]
        if x > 0 and y >= 0:
            v[i+1] = math.atan(y/x)/(math.pi*2)
        elif x > 0 and y < 0:
            v[i + 1] = 1 - math.atan(-y/x)/(math.pi * 2)
        elif x < 0 and y >= 0:
            v[i+1] = 0.5-(math.atan(y/(-x)))/(math.pi*2)
        elif x < 0 and y < 0:
            v[i + 1] = 0.75-(math.atan((-y) / (-x))) / (math.pi * 2)
        elif y > 0:
            v[i + 1] = 0.25
        elif y < 0:
            v[i + 1] = 0.75
        else:
            v[i + 1] = 0
    x, y = s[size-1], s[0]
    if x > 0 and y >= 0:
        v[size] = math.atan(y / x) / (math.pi * 2)
    elif x > 0 and y < 0:
        v[size] = 1 - math.atan(-y / x) / (math.pi * 2)
    elif x < 0 and y >= 0:
        v[size] = 0.5 - (math.atan(y / (-x))) / (math.pi * 2)
    elif x < 0 and y < 0:
        v[size] = 0.75 - (math.atan((-y) / (-x))) / (math.pi * 2)
    elif y > 0:
        v[size] = 0.25
    elif y < 0:
        v[size] = 0.75
    else:
        v[size] = 0
    return v

def standard_vector(v):
    sum = 0
    for i in range(size):
        v[i] = v[i]/10+0.5
    for i in range(size):
        sum += v[i]
    for i in range(size):
        v[i] = v[i]/sum
    return v

def vector_gather(v1, v2):
    v = np.zeros(size)
    for i in range(size):
        v[i] = (v1[i]+v2[i])/2
    return v

def ent(v):
    sum = 0
    for i in range(size):
        sum += v[i+1]*math.log(v[i+1],math.e)
    return sum*v[0]

def dent(v,num):
    sum = 0
    for i in range((int)(size/num)):
        sum += v[i]*math.log(v[i],math.e)
    return sum

def ds(v1, v2, v3, num):
    tmp1 = tmp2 = tmp3 = np.zeros(num)
    sum = 0
    for i in range((int)(size/num)):
        sum1 = sum2 = sum3 = 0
        for j in range(num):
            tmp1[j] = v1[i*num+j+1]
            sum1 += v1[i*num+j+1]
            print(sum1)
            tmp2[j] = v2[i * num + j + 1]
            sum2 += v2[i * num + j + 1]
            tmp3[j] = v3[i * num + j + 1]
            sum3 += v3[i * num + j + 1]
        sum += (((v3[0]*dent(tmp3,num))**2)*(sum3**2))/(v1[0]*dent(v1,num)*sum1*v2[0]*dent(v2,num)*sum2)
    return sum*size/num

def mylog(d):
    d = 1/d
    if d > 0:
        return math.log(d,math.e)
    elif d < 0:
        return math.log(-d,math.e)
    else:
        return 0

def sim(v1,v2,v3):
    sum1 = sum2 = sum3 =0
    for i in range(size):
        sum1 += v1[i]*mylog(v1[i])
        sum2 += v2[i]*mylog(v2[i])
        sum3 += v3[i]*mylog(v3[i])
    return (sum1*sum2)/(sum3**2)

def vector_ds(s1, s2):
    v1, v2 = mysentence_vectors(s1), mysentence_vectors(s2)
    v3 = vector_gather(v1, v2)
    sv1, sv2, sv3 = standard_vector(v1), standard_vector(v2), standard_vector(v3) #概率密度化
    #print(sv1)
    return sim(sv1, sv2, sv3)


'''
with open("案件.txt", "r", encoding="utf-8") as f:
    contents = f.readlines()
    matrix = np.zeros((len(contents), len(contents)))
    for i in range(len(contents)):
        for j in range(len(contents)):
            # 使用矩阵存储所有案件之间的相似度
            matrix[i][j] = vector_similarity(
                contents[i].strip(), contents[j].strip())

    f1 = open("result.txt", "w", encoding="utf-8")
    for j in range(len(contents)):
        # 获取最为相似的案件
        # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
        index = np.argsort(matrix[j])[-2]

        f1.writelines("案件" + str(j + 1) + ":" + '\t')
        f1.writelines(contents[j])
        f1.writelines("案件" + str(index + 1) + ":" + '\t')
        f1.writelines(contents[index])
        f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')
'''

with open("test.txt", "r", encoding="utf-8") as f:
    contents = f.readlines()
    matrix = np.zeros((len(contents), len(contents)))
    for i in range(len(contents)):
        for j in range(len(contents)):
            # 使用矩阵存储所有案件之间的相似度
            matrix[i][j] = vector_ds(contents[i].strip(), contents[j].strip())
    f1 = open("dsresult3.txt", "w", encoding="utf-8")
    for j in range(len(contents)):
        # 获取最为相似的案件
        # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
        index = np.argsort(matrix[j])[-2]

        f1.writelines("案件" + str(j + 1) + ":" + '\t')
        f1.writelines(contents[j])
        f1.writelines("案件" + str(index + 1) + ":" + '\t')
        f1.writelines(contents[index])
        f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')
    print('finish')