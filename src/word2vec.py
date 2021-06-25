from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from numpy.core.fromnumeric import shape
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk import word_tokenize
import seaborn as sns
import pandas as pd  
import numpy as np
import logging
from time import time

logging.basicConfig(filename='../logs/word2vec.log',  level=logging.DEBUG)
labels = {2 : "ALL", 1 : "happiness", 0 : "depression"}

def train_word_word2vec(source_sentences, des_path, vector_size=64, window=3, min_count=20):
    
    sents = [word_tokenize(s) for s in source_sentences]
    model = Word2Vec(sentences=sents, vector_size=vector_size, window=window, min_count=min_count, epochs=25)   
    model.save("{}.model".format(des_path))

    t = time()
    model.build_vocab(sents, progress_per=1000)
    logging.info('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    
    t = time()
    model.train(sents, total_examples=model.corpus_count, epochs=30, report_delay=1)
    logging.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    word_vectors = model.wv
    word_vectors.save("{}.wordvectors".format(des_path))   

def tsne_scatter_plot(model, word):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    list_names = [t[0] for t in model.wv.most_similar(positive=[word], topn=30)][10:]
    
    arrays = np.empty((0, 64), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=20).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.savefig(f'../reports/_word_vectors_common_{word}.png')
    
def plot_similar(model_0,model_1,word):
    arrays = np.empty((0, 64), dtype='f')
    word_labels = [word,word]
    color_list  = ['green','red']

    similars_0 = model_0.most_similar([word])
    similars_1 = model_1.most_similar([word])
    
    print(similars_0)
    print(shape(similars_1))

    arrays = np.append(arrays, model_0.__getitem__([word]), axis=0)
    arrays = np.append(arrays, model_1.__getitem__([word]), axis=0)

    for wrd in similars_0:
        wrd_vector = model_0.__getitem__([wrd[0]])
        word_labels.append(wrd[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    for wrd in similars_1:
        wrd_vector = model_1.__getitem__([wrd[0]])
        word_labels.append(wrd[0])
        color_list.append('orange')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=2).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization')
    plt.savefig(f'../reports/word_vectors_common_{word}.png')

def save_word2vec_model():
    
    path = '../data/' + '/cleaned/data_cleand.json'
    df = pd.read_json(path)

    for label_code, label_name in labels.items(): 
        
        source_sentences = df.copy()
        
        if label_code != 2 : 
            source_sentences = df[df.label == label_code]
        
        source_sentences = source_sentences['selftext_clean'].to_list()
        
        des_path="../models/word2vec/{}.word2vec".format(label_name)
        train_word_word2vec(source_sentences, des_path)
        
        logging.info("save {}.model".format(des_path))

def bias_experimnt(source = "All"):
    # pass
    model = Word2Vec.load('../models/word2vec/{}.word2vec.model'.format(source))
    # print(model.wv.most_similar('psychology', topn=15))
    print(model.wv.most_similar(positive=['woman', 'sad'], negative=['man']))
    print(model.wv.most_similar(positive=['man', 'sad'], negative=['woman']))

def intersection_word_vector_experiment():
    pass

# def CosineDistance(model, word, target_list, count):
#     cosine_dict ={}
#     word_list = []
#     a = model[word]

#     for item in target_list:
#         if item != word:
#             b = model [item]
#             cos_sim = dot(a, b)/(norm(a)*norm(b))
#             # cos_sim = distance.cosine(a, b)
#             cosine_dict[item] = cos_sim

#     dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
#     for item in dist_sort:
#         word_list.append((item[0], item[1]))

#     return word_list[: count]

def main():

    save_word2vec_model()
    # bias_experimnt()
    w2v_model_0 = KeyedVectors.load("../models/word2vec/depression.word2vec.model", mmap='r')
    w2v_model_1 = KeyedVectors.load("../models/word2vec/happiness.word2vec.wordvectors", mmap='r')
    tsne_scatter_plot(w2v_model_0, "man")
    
    w2v_model_0 = KeyedVectors.load("../models/word2vec/depression.word2vec.wordvectors", mmap='r')
    w2v_model_1 = KeyedVectors.load("../models/word2vec/happiness.word2vec.wordvectors", mmap='r')
    plot_similar(w2v_model_0, w2v_model_1, word='happiness')


if __name__ == '__main__':
    main()
