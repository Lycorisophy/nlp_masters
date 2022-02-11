import logging
from gensim.models import word2vec
#import numpy as np
#import os
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
# 使用gensim中的word2vec模块
model = word2vec.Word2Vec.load("model/word2vec_baike")
sentence = word2vec.LineSentence('train.txt')
#print(str(model.corpus_count)+","+str(model.iter))
model.train(sentence, total_examples=model.corpus_count+1, epochs=model.iter)
model.save("model/Word2vec_SongYang.w2v")
print("============over============")

