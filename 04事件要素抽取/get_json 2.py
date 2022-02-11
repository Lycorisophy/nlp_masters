import json
import os

def remove_duplicate_elements(l):
    new_list = []
    for i in l:
        if i not in new_list:
            new_list.append(i)
    return new_list

# 将属于同一事件要素的词语合并
def func(file_name):
    words = []
    element_type = []
    with open(file_name, "r", encoding='utf-8') as f1:
        contents = f1.readlines()
        new_contents = []
        # 将文本转换成list，方便后续处理
        for content in contents:
            new_contents.append(content.strip("\n").split(" "))

        for index, content in enumerate(new_contents):
            #print(content)
            if "S" in content[-1]:
                # 处理由一个单词组成的事件要素
                words.append(content[0])
                element_type.append(content[-1])

            elif "B" in content[-1]:
                # 处理由多个单词组成的事件要素
                words.append(content[0])
                element_type.append(content[-1])
                j = index
                while "I" in new_contents[j][-1] or "E" in new_contents[j][-1]:
                    words[-1] = words[-1] + new_contents[j][0]
                    j += 1
                    if j == len(new_contents):
                        break
            elif content[1] == 'v':
                words.append(content[0])
                element_type.append(content[1])
        SI = []
        H = []
        V = []

        for i in range(len(element_type)):
            #print(element_type[i])
            if element_type[i][-1] == "s" or element_type[i][-1] == "i":
                SI.append(words[i])
            elif element_type[i][-1] == "h":
                if len(words[i]) == 1:
                    H.append(words[i]+'姓人士')
                elif words[i] in fuxin:
                    H.append(words[i]+'氏')
                else:
                    H.append(words[i])
            elif element_type[i][-1] == "v":
                V.append(words[i])
        # 整理抽取结果
        result = dict()
        result["地点"] = remove_duplicate_elements(SI)
        result["姓名"] = remove_duplicate_elements(H)
        result["动作"] = remove_duplicate_elements(V)

        #   打印出完整的事件要素
        #for key, value in result.items():
            #print(key, value)
    return result
fuxin = ['司马', '上官', '欧阳', '夏侯', '诸葛', '西门', '左丘', '梁丘', '百里', '拓跋', '尉迟', '端木', '皇甫']
dd = dict()
filedir = "元数据"
filenames = os.listdir(filedir)
i = 1
for filename in filenames:
    file_name = "元数据/" + str(i) + ".txt"
    dd[i] = func(file_name)
    i += 1
with open("事件要素统计.json", "w", encoding='utf-8') as f:
    json.dump(dd, f, indent=2, ensure_ascii=False)
f.close()
print("#############over##################")