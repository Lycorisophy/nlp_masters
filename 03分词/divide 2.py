# -*- coding: utf-8 -*-
from pyltp import Segmentor
import os

LTP_DATA_DIR = "ltp_data_v3.4.0/"  # ltp模型目录的路径，根据实际情况修改
cws_model_path = os.path.join(LTP_DATA_DIR,
                              'cws.model')  # 分词模型路径，模型名称为`cws.model`
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(cws_model_path, os.path.join(LTP_DATA_DIR,
                              'dict'))  # 加载自定义字典

org_path = "../01数据预处理/data"
divided_path = "data"
labels = os.listdir(org_path)
for label in labels:
    f1 = open(org_path+'/'+label, "r", encoding="utf-8")
    f2 = open(divided_path+'/divide_'+label, "w", encoding="utf-8")
    texts = f1.readlines()
    for text in texts:
        words = segmentor.segment(text.strip())  # 分词
        words_list = list(words)
        for word in words_list:
            f2.write(word + ' ')
        f2.write('\n')
    f2.close()
    f1.close()
segmentor.release()  # 释放模型