from pyltp import Segmentor
from pyltp import SentenceSplitter
from pyltp import Postagger
from pyltp import Parser
from bs4 import BeautifulSoup
import json
import os
#from denoter_tools import statistics_denoters, get_denoters_index
import math
from pyltp import SentenceSplitter
from bs4 import BeautifulSoup


def statistics_denoters(train_sets, segmentor, postagger, parser):
    statistics_postagger = {}
    statistics_parser = {}
    statistics_single_words = {}
    for train_text in train_sets:
        with open('data/train3/' + train_text, 'r', encoding='utf-8') as f:
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

def fun(x):
    return x / (1 + x)


#     return (1-math.exp(-x))/(1+math.exp(-x))
#     return math.log(1+math.exp(x))
#     return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
test_sets = os.listdir('data/test/')
# test_sets = ['世界杯险遇恐怖袭击警方发现犯罪组织欲炸桥.xml']
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

    with open('result/event_extraction_contrast.txt', 'r', encoding='utf-8') as f:
        results_all = json.loads(f.read())
    results = results_all['results']
    statistics_postagger = results_all['statistics_postagger']
    statistics_parser = results_all['statistics_parser']
    statistics_single_words = results_all['statistics_single_words']

    total_event_count = 0
    total_event_find_count = 0
    total_event_find_right_count = 0

    for test_text in test_sets:
        print('正在测试文本：', test_text)
        with open('data/test/' + test_text, 'r', encoding='utf-8') as f:
            xml = f.read()
        soup = BeautifulSoup(xml, 'lxml')
        event_count = len(soup.find_all('denoter'))
        event_find_count = 0
        event_find_right_count = 0
        text_with_denoter = ''
        #         print(soup.content.get_text(strip=True))
        for denoter in soup.find_all('denoter'):
            denoter.string = '#' + denoter.string + '#'
        content = soup.content.get_text(strip=True).replace(' ', '')  # .replace('，','。')
        #         print(content)
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
            denoters_index = get_denoters_index(words, denoters_range, postags,
                                                relations, statistics_postagger, statistics_parser,
                                                statistics_single_words)
            candidates = set()
            #             print(words)
            #             print(denoters_index)
            for word_index, word_value in enumerate(words):
                if word_value in results and relations[word_index] != 'WP':
                    positive_num, negative_num = 0, 0
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

                    for node in all_related_nodes:
                        if node in results[word_value][0]:
                            positive_num += fun(results[word_value][0][node])
                        #                             if results[word_value][0][node] > 1:
                        #                                 positive_num += 1.5
                        #                             else:
                        #                                 positive_num += 1
                        # #                             print(node,results[word_value][0][node],'=>',word_value)
                        if node in results[word_value][3]:
                            negative_num += fun(results[word_value][3][node])
                    #                             if results[word_value][3][node] > 1:
                    #                                 negative_num += 1.5
                    #                             else:
                    #                                 negative_num += 1
                    #                             print(node,results[word_value][3][node],'!=>',word_value)

                    if relations[word_index] in results[word_value][1]:
                        positive_num += fun(results[word_value][1][relations[word_index]])
                    #                         if results[word_value][1][relations[word_index]] > 1:
                    #                             positive_num += 1.5
                    #                         else:
                    #                             positive_num += 1
                    # #                         print(relations[word_index],results[word_value][1][relations[word_index]],'<rel>',word_value)
                    if relations[word_index] in results[word_value][4]:
                        negative_num += fun(results[word_value][4][relations[word_index]])
                    #                         if results[word_value][4][relations[word_index]] > 1:
                    #                             negative_num += 1.5
                    #                         else:
                    #                             negative_num += 1
                    #                         print(relations[word_index],results[word_value][4][relations[word_index]],'!<rel>',word_value)
                    if postags[word_index] in results[word_value][2]:
                        positive_num += fun(results[word_value][2][postags[word_index]])
                    #                         if results[word_value][2][postags[word_index]] > 1:
                    #                             positive_num += 1.5
                    #                         else:
                    #                             positive_num += 1
                    if postags[word_index] in results[word_value][5]:
                        negative_num += fun(results[word_value][5][postags[word_index]])
                    #                         if results[word_value][5][postags[word_index]] > 1:
                    #                             negative_num += 1.5
                    #                         else:
                    #                             negative_num += 1

                    #                     candidates.add(word_index)
                    if positive_num - negative_num >= 0:
                        candidates.add(word_index)
                        print(word_value, positive_num, negative_num, 'positive_num - negative_num:',
                              positive_num - negative_num)
                    else:
                        print('NOT' + word_value, positive_num, negative_num, 'positive_num - negative_num:',
                              positive_num - negative_num)
            event_find_count += len(candidates)
            denoters_index_not_find = list(set(denoters_index) - set(candidates))
            for index in denoters_index_not_find:
                words[index] = '\033[1;33m' + words[index] + '\033[0m'
            for candidate in candidates:
                if candidate in denoters_index:
                    event_find_right_count += 1
                    words[candidate] = '\033[1;32m' + words[candidate] + '\033[0m'
                else:
                    words[candidate] = '\033[1;31m' + words[candidate] + '\033[0m'
            text_with_denoter += ''.join(words)
        print(text_with_denoter)
        total_event_count += event_count
        total_event_find_count += event_find_count
        total_event_find_right_count += event_find_right_count

        #         print('本文档查准率 Precision：',event_find_right_count/event_find_count)
        #         print('本文档查全率 Recall：',event_find_right_count/event_count)
        #         print('本文档F1值 ：',(2*pow(event_find_right_count,2))/((event_find_right_count*event_find_count)+(event_find_right_count*event_count)))

        #         print('到目前为止总的查准率 Precision：',total_event_find_right_count/total_event_find_count)
        #         print('到目前为止总的查全率 Recall：',total_event_find_right_count/total_event_count)
        #         print('到目前为止总的F1值 ：',(2*pow(total_event_find_right_count,2))/((total_event_find_right_count*total_event_find_count)+(total_event_find_right_count*total_event_count)))

        print('已训练完成文本：', test_text, '\n\n')
finally:
    segmentor.release()  # 释放模型
    postagger.release()  # 释放模型
    parser.release()  # 释放模型

if total_event_find_count != 0:
    print('自动标注识别正确的触发词数：', total_event_find_right_count)
    print('自动标注识别的触发词数：', total_event_find_count)
    print('人工标注的触发词数：', total_event_count)
    print('查准率 Precision：', total_event_find_right_count / total_event_find_count)
    print('查全率 Recall：', total_event_find_right_count / total_event_count)
    print('F1值 ：', (2 * pow(total_event_find_right_count, 2)) / ((total_event_find_right_count * total_event_find_count) + (
            total_event_find_right_count * total_event_count)))

