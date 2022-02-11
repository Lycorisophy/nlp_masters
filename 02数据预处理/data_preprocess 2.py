import re
import os

# 标点符号和特殊字母
space_punctuation = '''，。、:；!！.,.（）()"“”《》'''
# 对原始数据进行预处理
org_path = "../00数据获取/data"
preprocessed_path = "data"
labels = os.listdir(org_path)
for label in labels:
    f1 = open(org_path+'/'+label, "r", encoding='utf-8')
    f2 = open(preprocessed_path+'/'+label, "w", encoding='utf-8')
    for line in f1.readlines():
        line1 = re.sub(u"（.*?）", "", line)  # 去除括号内注释
        line2 = re.sub("[%s]+" % space_punctuation, " ", line1)  # 去除标点、特殊字母
        f2.writelines(line2)
    f1.close()
    f2.close()


