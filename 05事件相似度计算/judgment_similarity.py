import logging
from gensim.models import word2vec
import numpy as np
from scipy import linalg
import math
from numba import jit
import os
import profile

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
    return v

@jit
def cos_sim(s1, s2):
    '''
    计算两个句子之间的相似度:将两个向量的夹角余弦值作为其相似度
    :param s1:
    :param s2:
    :return:
    '''
    v1, v2 = sen_vec_average(s1), sen_vec_average(s2)
    return np.dot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))

@jit
def easy_ds(s1, s2):
    v1, v2 = sen_vec_average(s1), sen_vec_average(s2)
    v1, v2 = (v1-np.min(v1)+1), (v2-np.min(v2)+1)
    v1, v2 = v1/np.sum(v1), v2/np.sum(v2)
    sum1 = sum2 = sum3 = 0
    for i in range(size):
        sum1 += v1[i]*math.log(v1[i], 2)
        sum2 += v2[i]*math.log(v2[i], 2)
        sum3 += (v1[i]+v2[i])*math.log((v1[i]+v2[i]), 2)
    return (sum1*sum2)/((sum3-2)**2)*4

@jit
def probability_densitify(v):
    v = v - np.min(v) + 0.01
    return v/np.sum(v)

@jit
def vector_gather(v1, v2):
    v = np.zeros(size)
    for i in range(size):
        v[i] = (v1[i]+v2[i])/2
    return v

@jit
def mylog(d):
    if d > 0:
        return math.log(d, 2)
    elif d < 0:
        return math.log(-d, 2)
    else:
        return 0

@jit
def ent(v):
    sum = 0
    for i in range(size):
        sum += v[i]*mylog(v[i])
    return sum

@jit
def cent(v,num):
    sum = 0
    for i in range(int(size/num)):
        sum += v[i]*mylog(v[i])
    return sum

@jit
def conditional_diff_sim(v1, v2, v3, num):
    num = int(num)
    tmp1 = tmp2 = tmp3 = np.zeros(num)
    sum = 0
    for i in range(int(size/num)):
        sum1 = sum2 = sum3 = 0
        for j in range(num):
            tmp1[j] = v1[i*num+j]
            sum1 += v1[i*num+j]
            tmp2[j] = v2[i * num + j]
            sum2 += v2[i * num + j]
            tmp3[j] = v3[i * num + j]
            sum3 += v3[i * num + j]
        sum += (cent(v1, num)*sum1*cent(v2, num)*sum2)/(((cent(tmp3, num))**2)*(sum3**2))
    return sum/num

@jit
def diff_sim(v1,v2,v3):
    sum1 = sum2 = sum3 =0
    for i in range(size):
        sum1 += v1[i]*math.log(v1[i], 2)
        sum2 += v2[i]*math.log(v2[i], 2)
        sum3 += v3[i]*math.log(v3[i], 2)
    return (sum3**2)/(sum1*sum2)

@jit
def kl_sim(v1, v2):
    sum = 0
    for i in range(size):
        sum += v1[i]*mylog(v1[i]/v2[i])
    return abs(sum)

@jit
def vector_cds(s1, s2):
    v1, v2 = sen_vec_average(s1), sen_vec_average(s2)
    v3 = vector_gather(v1, v2)
    sv1, sv2, sv3 = probability_densitify(v1), probability_densitify(v2), probability_densitify(v3) #概率密度化
    return conditional_diff_sim(sv1, sv2, sv3, size/16)

@jit
def vector_ds(s1, s2):
    v1, v2 = sen_vec_average(s1), sen_vec_average(s2)
    v3 = vector_gather(v1, v2)
    sv1, sv2, sv3 = probability_densitify(v1), probability_densitify(v2), probability_densitify(v3) #概率密度化
    return diff_sim(sv1, sv2, sv3)

@jit
def vector_dsc(s1, s2):
    v1, v2 = sen_vec_compress(s1), sen_vec_compress(s2)
    v3 = vector_gather(v1, v2)
    sv1, sv2, sv3 = probability_densitify(v1), probability_densitify(v2), probability_densitify(v3) #概率密度化
    return diff_sim(sv1, sv2, sv3)

@jit
def vector_kl(s1, s2):
    v1, v2 = sen_vec_average(s1), sen_vec_average(s2)
    sv1, sv2 = probability_densitify(v1), probability_densitify(v2) #概率密度化
    return kl_sim(sv1, sv2)

@jit
def vector_cos(s1, s2):
    return cos_sim(s1, s2)

def ds_result():
    with open("test.txt", "r", encoding="utf-8") as f:
        contents = f.readlines()
        matrix = np.zeros((len(contents), len(contents)))
        for i in range(len(contents)):
            for j in range(len(contents)):
                # 使用矩阵存储所有案件之间的相似度
                matrix[i][j] = vector_ds(contents[i].strip(), contents[j].strip())
        f1 = open("result_ds.txt", "w", encoding="utf-8")
        for j in range(len(contents)):
            # 获取最为相似的案件
            # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
            index = np.argsort(matrix[j])[-2]
            f1.writelines("事件" + str(j + 1) + ":" + '\t')
            f1.writelines(contents[j])
            f1.writelines("事件" + str(index + 1) + ":" + '\t')
            f1.writelines(contents[index])
            f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')
        f.close()

def eds_result():
    with open("test.txt", "r", encoding="utf-8") as f:
        contents = f.readlines()
        matrix = np.zeros((len(contents), len(contents)))
        for i in range(len(contents)):
            for j in range(len(contents)):
                # 使用矩阵存储所有案件之间的相似度
                matrix[i][j] = easy_ds(contents[i].strip(), contents[j].strip())
        f1 = open("result_eds.txt", "w", encoding="utf-8")
        for j in range(len(contents)):
            # 获取最为相似的案件
            # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
            index = np.argsort(matrix[j])[-2]
            f1.writelines("事件" + str(j + 1) + ":" + '\t')
            f1.writelines(contents[j])
            f1.writelines("事件" + str(index + 1) + ":" + '\t')
            f1.writelines(contents[index])
            f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')
        f.close()

def dsc_result():
    with open("test.txt", "r", encoding="utf-8") as f:
        contents = f.readlines()
        matrix = np.zeros((len(contents), len(contents)))
        for i in range(len(contents)):
            for j in range(len(contents)):
                # 使用矩阵存储所有案件之间的相似度
                matrix[i][j] = vector_dsc(contents[i].strip(), contents[j].strip())
        f1 = open("result_dsc.txt", "w", encoding="utf-8")
        for j in range(len(contents)):
            # 获取最为相似的案件
            # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
            index = np.argsort(matrix[j])[-2]
            f1.writelines("事件" + str(j + 1) + ":" + '\t')
            f1.writelines(contents[j])
            f1.writelines("事件" + str(index + 1) + ":" + '\t')
            f1.writelines(contents[index])
            f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')
        f.close()

def cds_result():
    with open("test.txt", "r", encoding="utf-8") as f:
        contents = f.readlines()
        matrix = np.zeros((len(contents), len(contents)))
        for i in range(len(contents)):
            for j in range(len(contents)):
                # 使用矩阵存储所有案件之间的相似度
                matrix[i][j] = vector_cds(contents[i].strip(), contents[j].strip())
        f1 = open("result_cds.txt", "w", encoding="utf-8")
        for j in range(len(contents)):
            # 获取最为相似的案件
            # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
            index = np.argsort(matrix[j])[-2]
            f1.writelines("事件" + str(j + 1) + ":" + '\t')
            f1.writelines(contents[j])
            f1.writelines("事件" + str(index + 1) + ":" + '\t')
            f1.writelines(contents[index])
            f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')
        f.close()

def kl_result():
    with open("test.txt", "r", encoding="utf-8") as f:
        contents = f.readlines()
        matrix = np.zeros((len(contents), len(contents)))
        for i in range(len(contents)):
            for j in range(len(contents)):
                # 使用矩阵存储所有案件之间的相似度
                matrix[i][j] = vector_kl(contents[i].strip(), contents[j].strip())
        f1 = open("result_kl.txt", "w", encoding="utf-8")
        for j in range(len(contents)):
            # 获取最为相似的案件
            # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
            index = np.argsort(matrix[j])[2]
            f1.writelines("事件" + str(j + 1) + ":" + '\t')
            f1.writelines(contents[j])
            f1.writelines("事件" + str(index + 1) + ":" + '\t')
            f1.writelines(contents[index])
            f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')
        f.close()

def cos_result():
    with open("test.txt", "r", encoding="utf-8") as f:
        contents = f.readlines()
        matrix = np.zeros((len(contents), len(contents)))
        for i in range(len(contents)):
            for j in range(len(contents)):
                # 使用矩阵存储所有案件之间的相似度
                matrix[i][j] = vector_cos(contents[i].strip(), contents[j].strip())
        f1 = open("result_cos.txt", "w", encoding="utf-8")
        for j in range(len(contents)):
            # 获取最为相似的案件
            # 注意：每个案件与自己的相似度为1，因此获取的是相似度第二大的案件
            index = np.argsort(matrix[j])[-2]
            f1.writelines("事件" + str(j + 1) + ":" + '\t')
            f1.writelines(contents[j])
            f1.writelines("事件" + str(index + 1) + ":" + '\t')
            f1.writelines(contents[index])
            f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')
        f.close()

if __name__ == "__main__" :
    #profile.run("ds_result()")
    profile.run("eds_result()")
    #profile.run("dsc_result()")
    #profile.run("cds_result()")
    #profile.run("kl_result()")
    #profile.run("cos_result()")
