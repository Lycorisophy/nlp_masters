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
        #print(train_text)
        for denoter in soup.find_all('denoter'):
            denoter.string = '#' + denoter.string + '#'
            
        content = soup.content.get_text(strip=True).replace(' ','')
        sents = SentenceSplitter.split(content)  # 分句
        
        for x in range(len(sents)):
            sentence = sents[x]
            denoters_range = []
            denoter_range = []
            while '#' in sentence:
                if len(denoter_range) == 0:
                    denoter_range.append(sentence.index('#'))
                elif len(denoter_range) == 1:
                    denoter_range.append(sentence.index('#')-1)
                    denoters_range.append(denoter_range)
                    denoter_range = []
                sentence = sentence.replace('#','',1)
            
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
            for index,word in enumerate(words):
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

def get_denoters_index(words, denoters_range, postags, relations, statistics_postagger, statistics_parser, statistics_single_words):
    denoters_index = []
    denoters_index_tmp = []
    denoter_index_tmp = {}
    words_length = 0
    for index,word in enumerate(words):
        words_length += len(word)
        if len(denoters_range) > 0:
            if words_length - 1 >= denoters_range[0][0]:
                a,b,c=0,0,0
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
            #print(words)
            #print(denoter_index_tmp)
            denoter_index_tmp_sorted = sorted(denoter_index_tmp.items(), key=lambda item: item[1])
            #print(denoter_index_tmp_sorted)
            denoters_index.append(denoter_index_tmp_sorted[-1][0])
            #print(100*'#')
    return denoters_index
