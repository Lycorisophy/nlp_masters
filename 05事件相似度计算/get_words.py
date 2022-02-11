from gensim.models import word2vec
import numpy as np

def remove_duplicate_elements(l):
    new_list = []
    for i in l:
        if i not in new_list:
            new_list.append(i)
    return new_list

model = word2vec.Word2Vec.load("model/word2vec_baike")
vs = []
with open("train.txt", "r", encoding="utf-8") as f:
    contents = f.readlines()
    matrix = np.zeros((len(contents), len(contents)))
    for i in range(len(contents)):
        for j in range(len(contents)):
            s = contents[i].strip()
            words = s.split(" ")
            for word in words:
                try:
                    model[word]
                except KeyError as ke:
                    vs.append(str(ke)[7:-20])
nvs = remove_duplicate_elements(vs)
with open("words.txt", "w", encoding="utf-8") as fv:
    i = 0
    while i < len(nvs):
        for j in range(10):
            try:
                fv.writelines(nvs[i])
            except IndexError as IE:
                print(IE)
            finally:
                fv.writelines(' ')
                i += 1
        fv.writelines('\n')
print('finish')
