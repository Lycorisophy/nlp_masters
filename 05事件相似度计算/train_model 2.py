#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Author:Zhang Shiwei

import logging
from gensim.models import word2vec
import os
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

# 使用gensim中的word2vec模块
'''
size = 100
sentences = word2vec.LineSentence('训练案件.txt')
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=5, size=size)
model.train(sentences, total_examples=model.corpus_count, epochs=10)
model.save("model/Word2vec.w2v")
# req_count = 5
# for key in model.wv.similar_by_word('被告人', topn=100):
#     if len(key[0]) == 3:
#         req_count -= 1
#         print(key[0], key[1])
#         if req_count == 0:
#             break
'''

filedir = "data"
filenames = os.listdir(filedir)
f=open('data.txt', 'w', encoding='utf-8')
i=0
#先遍历文件名
for filename in filenames:
    i += 1
    if i>0:
        filepath = filedir+'/'+filename
        for line in open(filepath, 'r', encoding='utf-8'):
            f.writelines(line)
#关闭文件
f.close()
'''
model = word2vec.Word2Vec.load("model/word2vec_wx")
sentence = word2vec.LineSentence(filedir+'/result.txt')
#tte = model.corpus_count + 1  # total_examples参数更新
model.train(sentence, total_examples=model.corpus_count, epochs=model.iter)
model.save("model/Word2vec.w2v")
print("============over============")
'''


'''with open("案件.txt", "r", encoding="utf-8") as f:
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
        f1.writelines("相似度： " + str(matrix[j][index]) + '\n\n')'''

