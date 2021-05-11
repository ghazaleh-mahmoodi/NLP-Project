import re
import glob
import pandas as pd
import matplotlib.pyplot as plt


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
def compute_tfidf(word_0, word_1):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))
    tfidf = tfidf_vectorizer.fit_transform(word_0)
    idf = tfidf_vectorizer.idf_
    
    ii = sorted(zip(idf, tfidf_vectorizer.get_feature_names()))
    j = 0
    for x, y in ii:
        print(x, y)
        if j == 10 : break


def data_preprocessing(data):
    all_sentences = []
    clean_data = []
    all_words = []

    #1.filling Nans
    print("train.isnull().sum()", data.isnull().sum())
    data.fillna("IS NULL", inplace=True)

    #find word
    tokenizer = RegexpTokenizer(r'(\w+)')
    lemmatizer = WordNetLemmatizer()

    for i in range(len(data)):
        #2.set all alphabet to lower case
        text_to_lower = data[i].lower()
        
        #3.remove hyper link
        text_to_lower = re.sub(r"https?://\S+", "", text_to_lower)
        
        #4.extract sentences
        all_sentences.append(sentence_broken(text_to_lower))
        
        #5.remove number
        text_to_lower = re.sub(r"\b[0-9]+\b\s*", "", text_to_lower)
        
        #6.remove PUNCTUATION
        tokenized_text = tokenizer.tokenize(text_to_lower) 
        
        words_lemmatizer = [lemmatizer.lemmatize(i) for i in tokenized_text]
        
        words_without_stop = [i for i in words_lemmatizer if i not in stopwords.words("english")]
        
        #5.extract words
        all_words.extend(words_without_stop)

        clean_text = " ".join(word for word in words_without_stop if not word.isdigit() and word.isalpha())
        
        clean_data.append(clean_text)

    return clean_data, all_sentences, all_words


def sentence_broken(input_row):
    
    punkt_params = PunktParameters()
    punkt_params.abbrev_types = set(['Mr', 'Mrs', 'LLC'])
    tokenizer = PunktSentenceTokenizer(punkt_params)
    tokens = tokenizer.tokenize(input_row)

    sentences = []
    for t in tokens:
        if t != "" : sentences.append(t)
    
    return sentences

def main():
    
    print("data cleaning start :")
    cleand_data = pd.DataFrame()
    for label in ['0', '1']:
        word_per_label = []
        sentences_per_label = []
        
        path = '../data/' + '/original/' + label + '/*'

        for file_name in glob.glob(path):
            print(file_name)
            data = pd.read_json(file_name)
            data['label'] = label
            data["selftext_clean"], all_sentences, all_words = data_preprocessing(data["selftext"])
            
            word_per_label.extend(all_words)
            sentences_per_label.extend(all_sentences)
            
            data = data[["label", "title","selftext_clean", "author", "score","url", "selftext"]]
            dest_path = file_name.replace('original', 'cleaned')
            data.to_json(dest_path, indent=4)
            cleand_data = cleand_data.append(data, ignore_index=True)
        
        output_path = '../data/' + '/sentence_broken/' + label+"_sentences.txt"
        with open(output_path, "w", encoding="utf8") as f:
            f.write("\n".join(str(sen) for item in sentences_per_label for sen in item))


        output_path = '../data/' + '/word_broken/' + label +" _words.txt"
        with open(output_path, "w", encoding="utf8") as f:
            f.write("\n".join([str(x) for x in word_per_label]) + '\n')


        print("len(set(word_per_label))", len(set(word_per_label)))
    
    path = '../data/' + '/cleaned/data_cleand.json'
    # cleand_data = pd.DataFrame(cleand_data)
    cleand_data.to_json(path, indent=4)

if __name__ == '__main__':
    main()
