import re
import os
from bs4 import BeautifulSoup
from pyltp import SentenceSplitter

def xml2txt(path):
    labels = os.listdir(path)
    for label in labels:
        files = os.listdir(path+'/'+label)
        f1 = open("data/"+label+".txt", 'w', encoding='utf-8')
        for file in files:
            f = open(path+'/'+label+'/'+file, 'r', encoding='utf-8')
            xml = f.read()
            f.close()
            soup = BeautifulSoup(xml, 'lxml')
            content = soup.content.get_text(strip=True).replace(' ', '')
            f1.writelines(SentenceSplitter.split(content))
            f1.writelines('\n')
        f1.close()
    return

path = "CEC语料库2.0/CEC"
xml2txt(path)
print("finish")
