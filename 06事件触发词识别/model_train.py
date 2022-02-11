from pyltp import Segmentor, SentenceSplitter, Postagger, Parser
from bs4 import BeautifulSoup
import json
import os
#from denoter_tools import statistics_denoters, get_denoters_index

def statistics_denoters(train_sets, segmentor, postagger, parser):
    statistics_postagger = {}
    statistics_parser = {}
    statistics_single_words = {}
    for train_text in train_sets:
        with open('data/train/' + train_text, 'r', encoding='utf-8') as f:
            xml = f.read()
        soup = BeautifulSoup(xml, 'lxml')
        # print(train_text)
        for denoter in soup.find_all('denoter'):
            denoter.string = '#' + denoter.string + '#'

        content = soup.content.get_text(strip=True).replace(' ', '')
        sents = SentenceSplitter.split(content)  # 分句

        for x in range(len(sents)):
            sentence = sents[x]
            denoters_range = []
            denoter_range = []
            while '#' in sentence:
                if len(denoter_range) == 0:
                    denoter_range.append(sentence.index('#'))
                elif len(denoter_range) == 1:
                    denoter_range.append(sentence.index('#') - 1)
                    denoters_range.append(denoter_range)
                    denoter_range = []
                sentence = sentence.replace('#', '', 1)

            words = list(segmentor.segment(sentence))
            postags = postagger.postag(words)  # 词性标注
            arcs = parser.parse(words, postags)  # 句法分析
            heads = [arc.head for arc in arcs]
            relations = [arc.relation for arc in arcs]

            denoters_index = []
            denoters_index_tmp = []
            denoter_index = []
            denoter_index_tmp = []
            denoter_tmp = []
            words_length = 0
            for index, word in enumerate(words):
                words_length += len(word)
                if len(denoters_range) > 0:
                    if words_length - 1 >= denoters_range[0][0]:
                        denoter_index_tmp.append(index)
                    if words_length - 1 >= denoters_range[0][1]:
                        denoters_index_tmp.append(denoter_index_tmp)
                        denoter_index_tmp = []
                        denoters_range.pop(0)
            #             print(denoters_index_tmp)

            for denoter_index_tmp in denoters_index_tmp:
                if len(denoter_index_tmp) == 1:
                    denoters_index.extend(denoter_index_tmp)
                    if postags[denoter_index_tmp[0]] not in statistics_postagger:
                        statistics_postagger[postags[denoter_index_tmp[0]]] = 1
                    else:
                        statistics_postagger[postags[denoter_index_tmp[0]]] += 1

                    if relations[denoter_index_tmp[0]] not in statistics_parser:
                        statistics_parser[relations[denoter_index_tmp[0]]] = 1
                    else:
                        statistics_parser[relations[denoter_index_tmp[0]]] += 1
                    if words[denoter_index_tmp[0]] not in statistics_single_words:
                        statistics_single_words[words[denoter_index_tmp[0]]] = 1
                    else:
                        statistics_single_words[words[denoter_index_tmp[0]]] += 1
    return statistics_postagger, statistics_parser, statistics_single_words


def get_denoters_index(words, denoters_range, postags, relations, statistics_postagger, statistics_parser,
                       statistics_single_words):
    denoters_index = []
    denoters_index_tmp = []
    denoter_index_tmp = {}
    words_length = 0
    for index, word in enumerate(words):
        words_length += len(word)
        if len(denoters_range) > 0:
            if words_length - 1 >= denoters_range[0][0]:
                a, b, c = 0, 0, 0
                if postags[index] in statistics_postagger:
                    a = statistics_postagger[postags[index]]
                if relations[index] in statistics_parser:
                    b = statistics_parser[relations[index]]
                if words[index] in statistics_single_words:
                    c = statistics_single_words[words[index]]
                denoter_index_tmp[index] = (a + b + c)
            if words_length - 1 >= denoters_range[0][1]:
                denoters_index_tmp.append(denoter_index_tmp)
                denoter_index_tmp = {}
                denoters_range.pop(0)

    for denoter_index_tmp in denoters_index_tmp:
        if len(denoter_index_tmp) == 1:
            denoters_index.append([item[0] for item in denoter_index_tmp.items()][0])
        else:
            # print(words)
            # print(denoter_index_tmp)
            denoter_index_tmp_sorted = sorted(denoter_index_tmp.items(), key=lambda item: item[1])
            # print(denoter_index_tmp_sorted)
            denoters_index.append(denoter_index_tmp_sorted[-1][0])
            # print(100*'#')
    return denoters_index


train_sets = os.listdir('data/train/')
# train_sets = ['伊拉克3月恐怖袭击和暴力冲突致死592人.xml']
results_all = {}
LTP_DATA_DIR = "ltp_data_v3.4.0/"  # ltp模型目录的路径，根据实际情况修改
cws_model_path = os.path.join(LTP_DATA_DIR,
                              'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR,
                              'pos.model')  # 词性标注模型路径，模型名称为`cws.model`
parser_model_path = os.path.join(LTP_DATA_DIR,
                              'parser.model')  # parser模型路径，模型名称为`cws.model`
try:
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    parser = Parser()  # 初始化实例
    parser.load(parser_model_path)  # 加载模型

    results = {}
    statistics_postagger, statistics_parser, statistics_single_words = statistics_denoters(train_sets, segmentor,
                                                                                           postagger, parser)
    print(sorted(statistics_postagger.items(), key=lambda item: item[1]))
    print(sorted(statistics_parser.items(), key=lambda item: item[1]))
    sum_postagger = sum([item[1] for item in statistics_postagger.items()])
    sum_parser = sum([item[1] for item in statistics_parser.items()])
    for item in statistics_postagger.items():
        statistics_postagger[item[0]] = item[1] / sum_postagger
    for item in statistics_parser.items():
        statistics_parser[item[0]] = item[1] / sum_parser
    print(sorted(statistics_postagger.items(), key=lambda item: item[1]))
    print(sorted(statistics_parser.items(), key=lambda item: item[1]))

    for train_text in train_sets:
        print('正在训练文本：', train_text)
        with open('data/train/' + train_text, 'r', encoding='utf-8') as f:
            xml = f.read()
        soup = BeautifulSoup(xml, 'lxml')
        #         print(soup.content.get_text(strip=True))
        for denoter in soup.find_all('denoter'):
            denoter.string = '#' + denoter.string + '#'

        content = soup.content.get_text(strip=True).replace(' ', '')  # .replace('，','。')
        sents = SentenceSplitter.split(content)  # 分句

        for x in range(len(sents)):
            sentence = sents[x]
            denoters_range = []
            denoter_range = []
            while '#' in sentence:
                if len(denoter_range) == 0:
                    denoter_range.append(sentence.index('#'))
                elif len(denoter_range) == 1:
                    denoter_range.append(sentence.index('#') - 1)
                    denoters_range.append(denoter_range)
                    denoter_range = []
                sentence = sentence.replace('#', '', 1)
            #             print(denoters_range)

            words = list(segmentor.segment(sentence))
            postags = list(postagger.postag(words))  # 词性标注
            arcs = parser.parse(words, postags)  # 句法分析
            heads = [arc.head for arc in arcs]
            relations = [arc.relation for arc in arcs]
            #             print(words)
            #             print(postags)
            #             print(heads)
            #             print(relations)
            denoters_index = get_denoters_index(words, denoters_range, postags,
                                                relations, statistics_postagger, statistics_parser,
                                                statistics_single_words)
            #             print(denoters_index)
            for word_index, word_value in enumerate(words):
                if relations[word_index] != 'WP':
                    all_related_nodes = []
                    parent_node = []
                    child_nodes = []
                    other_nodes = []
                    if heads[word_index] != 0:
                        parent_node.append(words[heads[word_index] - 1])
                    for head_index, head_value in enumerate(heads):
                        if head_value == (word_index + 1) and relations[head_index] != 'WP':
                            child_nodes.append(words[head_index])
                    if word_index - 1 >= 0 and relations[word_index - 1] != 'WP':
                        other_nodes.append(words[word_index - 1])
                    if word_index + 2 <= len(words) and relations[word_index + 1] != 'WP':
                        other_nodes.append(words[word_index + 1])
                    all_related_nodes = list(set(child_nodes + other_nodes + parent_node))
                    if word_index in denoters_index:
                        if word_value not in results:
                            results[word_value] = [{}, {}, {}, {}, {}, {}, []]
                        for node in all_related_nodes:
                            if node not in results[word_value][0]:
                                results[word_value][0][node] = 1
                            else:
                                results[word_value][0][node] += 1
                        if relations[word_index] not in results[word_value][1]:
                            results[word_value][1][relations[word_index]] = 1
                        else:
                            results[word_value][1][relations[word_index]] += 1
                        if postags[word_index] not in results[word_value][2]:
                            results[word_value][2][postags[word_index]] = 1
                        else:
                            results[word_value][2][postags[word_index]] += 1

                    else:
                        if word_value not in results:
                            results[word_value] = [{}, {}, {}, {}, {}, {}, []]
                        for node in all_related_nodes:
                            if node not in results[word_value][3]:
                                results[word_value][3][node] = 1
                            else:
                                results[word_value][3][node] += 1
                        if relations[word_index] not in results[word_value][4]:
                            results[word_value][4][relations[word_index]] = 1
                        else:
                            results[word_value][4][relations[word_index]] += 1
                        if postags[word_index] not in results[word_value][5]:
                            results[word_value][5][postags[word_index]] = 1
                        else:
                            results[word_value][5][postags[word_index]] += 1

        print('已训练完成文本：', train_text)


finally:
    segmentor.release()  # 释放模型
    postagger.release()  # 释放模型
    parser.release()  # 释放模型

# 修剪结果集
results_reduce = []
for result in results.items():
    if len(result[1][1]) == 0:
        results_reduce.append(result[0])
for result_reduce in results_reduce:
    results.pop(result_reduce)

for result in results.items():
    for index in range(6):
        result[1][6].append(sum([item[1] for item in result[1][index].items()]))

results_all = {'results': results, 'statistics_postagger': statistics_postagger, 'statistics_parser': statistics_parser,
               'statistics_single_words': statistics_single_words}
with open('result/event_extraction_contrast.txt', 'w', encoding='utf-8') as f:
    f.write(json.dumps(results_all, ensure_ascii=False,indent=2))

print('训练结束！！！')