import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import collections
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import sent_tokenize, word_tokenize, regexp_tokenize, FreqDist


report = {}

def read_json_file():
    path = '../data/' + '/cleaned/data_cleand.json'
    data = pd.read_json(path)
    return data

def read_text_file(path, label):
    
    content = {}

    for file_name, key in zip(glob.glob(path), label):
        print(file_name, key)
        
        text_file = open(file_name, "r", encoding="utf8")
        
        content[key] = text_file.read().splitlines()
        
    return content

def data_unit(data, label):
    counts = {}
    
    for key in label:
        counts[key] = len(data[data['label'] == int(key)])

    return counts

def words_count(words, is_unique):
    
    counts = {}
    
    for key, words_list in words.items():
        if is_unique : words_list = set(words_list)
        counts[key] = len(words_list)

    return counts

def find_common_words(words, label):

    commen = set([value for value in words[label[0]] if value in words[label[1]]])
    return len(commen), commen

def most_freq_uncommon(words, common_words, thereshold=10):
    
    words_list_report = {}
    most_freq_words_result = {}
    for key, words_list in words.items():
        most_freq_words_list = {}
        most_freq_words = collections.Counter(words_list).most_common(len(words_list))
        
        most_freq_words_result[key] = most_freq_words
        
        for w in most_freq_words :
            if w[0] not in common_words: 
                most_freq_words_list[w[0]] = w[1]
            
            if len(most_freq_words_list) == thereshold: break
        
        words_list_report[key] = most_freq_words_list

    return words_list_report, most_freq_words_result

def calculate_RNF(common_words, counts, accum_sum, i, j, thereshold=10):
    score = {}
    for w in set(common_words):
        x = counts[i][w]/accum_sum[i]
        y = counts[j][w]/accum_sum[j]
        z = x/y
        score[w] = z
    
    result = {}
    for key, value in sorted(score.items(), key=lambda item: item[1], reverse=True)[:thereshold]:
        result[key] = value

    return result

def relative_normalize_frequency(words, common_words, thereshold=10, label = ["0", "1"]):

    counts = {}
    accum_sum = {}
    for key, words_list in words.items():
        counts[key] = Counter(words_list)
        accum_sum[key] = 0
        for x in set(words_list):
            accum_sum[key]+=counts['0'][x]

    report = {}
    report['0'] = calculate_RNF(common_words, counts, accum_sum, '0', '1', thereshold)
    report['1'] = calculate_RNF(common_words, counts, accum_sum, '1', '0', thereshold=10,)
    return report

def compute_tfidf(clean_data, label, thereshold=10):
    
    documentA = ' '.join(clean_data[clean_data['label'] == 0]['selftext_clean'].tolist())
    documentB = ' '.join(clean_data[clean_data['label'] == 1]['selftext_clean'].tolist())
    
    vectorizer = TfidfVectorizer()
    # vectors = vectorizer.fit_transform([' '.join(words['0']), ' '.join(words['1'])])
    vectors = vectorizer.fit_transform([documentA, documentB])
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    
    df = pd.DataFrame(denselist, columns=feature_names)

    report = {}
    df = df.T
    for l in label:
        l = int(l)
        df = df.sort_values(by = l, ascending = 0)
        x = df.head(thereshold)[l]
        report[l] =x.to_dict() 
        
    return report

def histogram(most_freq_words_result):

    print(most_freq_words_result)
    for key, value in most_freq_words_result.items():
        print(value)

        plt.hist(value)
        plt.show()

def main():
    label = ['0', '1']

    clean_data = read_json_file()
    words = read_text_file(path = '../data/word_broken/*.txt', label=label)
    # sentences = read_text_file(path = '../data/sentence_broken/*.txt')

    report['Number of data units'] = data_unit(data=clean_data, label=label)
    
    #TODO
    report['Sectences count'] = "ToDO"

    report['words count'] = words_count(words, is_unique=False)

    report['unique words count'] = words_count(words, is_unique=True)

    report['common words count'], common_words = find_common_words(words, label)
    
    report['uncommon words count'] = report['unique words count'][label[0]] + report['unique words count'][label[1]] - report['common words count']
    
    report['most frequence uncommon words'], most_freq_words_result = most_freq_uncommon(words, common_words)

    report['Relative Normalize Frequency'] = relative_normalize_frequency(words, common_words)

    report['TF IDF'] = compute_tfidf(clean_data, label)

    histogram(most_freq_words_result)

    with open("analysis-report.json", "w", encoding="utf8") as json_file :
        json.dump(report, json_file, ensure_ascii=False, indent=4)

    
    
    # for key, value in words.items():
    #     fdist = FreqDist(commen_words)

    #     wc = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(fdist)

    #     plt.figure(figsize=(12,10))
    #     plt.imshow(wc, interpolation="bilinear")
    #     plt.axis("off")
    #     plt.savefig(key + 'result.png')


if __name__ == '__main__':
    main()
