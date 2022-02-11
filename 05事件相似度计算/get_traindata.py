import os
filedir = "data"
filenames = os.listdir(filedir)
f=open('train.txt', 'w', encoding='utf-8')
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