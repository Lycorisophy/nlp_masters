import os
filedir = "data"
filenames = os.listdir(filedir)
f=open('data.txt', 'w', encoding='utf-8')
#先遍历文件名
for filename in filenames:
        filepath = filedir+'/'+filename
        for line in open(filepath, 'r', encoding='utf-8'):
            f.writelines(line)
        f.writelines('\n')
#关闭文件
f.close()
print("########step1 over#########")
with open("data.txt", "r", encoding="utf-8") as f1:
    data = f1.readlines()
    num = 1
    file_name = '元数据/' + str(num) + '.txt'
    for i in range(len(data)):
        f2 = open(file_name, "a", encoding='utf-8')
        if data[i] == '\n':
            num += 1
            file_name = '元数据/' + str(num) + '.txt'
        else:
            f2.writelines(data[i])
print("########step2 over#########")