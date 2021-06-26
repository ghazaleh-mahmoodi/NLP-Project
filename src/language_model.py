# import tensorflow as tf
from pickle import load, dump
import pandas as pd
from numpy import array
from nltk import word_tokenize
from keras.models import load_model, Sequential
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Bidirectional, Embedding, Dense
import dataframe_image as dfi
import logging

logging.basicConfig(filename='../logs/language_model----------.log',  level=logging.DEBUG)
labels = {1 : "happiness", 0 : "depression"}

def prepare_data(label_code = 0):
    
    path = '../data/' + '/cleaned/data_cleand.json'
    df = pd.read_json(path)
    df = df[df.label == label_code]
    df = df['selftext_clean']
    
    sequences = list()

    for row_data in df:

        tokens = word_tokenize(row_data)
        
        # organize into sequences of tokens
        length = 10 + 1
        for i in range(length, len(tokens)):
            # select sequence of tokens
            seq = tokens[i-length:i]
            # convert into a line
            line = ' '.join(seq)
            # store
            sequences.append(line)
           
    logging.info('total sequences: {}'.format(len(sequences)))
    
    return sequences

def train_language_model(data, label, model_path, tokenizer_path):
    
    logging.info('start training lanhuage model for {} class'.format(labels[label]))

    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # separate into input and output
    sequences = array(sequences)
    print(sequences)

    X, y = sequences[:,:-1], sequences[:,-1]
    print(X)
    print(y)
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]
    
    print(vocab_size, seq_length)
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(Bidirectional(LSTM( 100, dropout=0.3)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    logging.info(model.summary())
    
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # fit model
    model.fit(X, y, batch_size=128, epochs=100)
    
    # save the model to file
    model.save(model_path)
    logging.info('save {}'.format(model_path))
    
    # save the tokenizer
    dump(tokenizer, open(tokenizer_path, 'wb'))
    logging.info('save {}'.format(tokenizer_path))

def generate_sequence(model, tokenizer, seq_length, seed_text, n_words):
    
    result = []
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)

    return ' '.join(result)

def main():
    
    generated_sen = pd.DataFrame()

    for label_code, label_name  in labels.items():
        
        data = prepare_data()
        
        model_path = '../models/language_model/{}_language_model.h5'.format(label_name)
        tokenizer_path = '../models/language_model/{}_tokenizer.pkl'.format(label_name)
        
        # train_language_model(data, label_code, model_path, tokenizer_path)


        # load the model
        model = load_model(model_path)

        # load the tokenizer
        tokenizer = load(open(tokenizer_path, 'rb'))
        
        print(model_path)
        print(tokenizer_path)
        
        seed_text = "i feel so "
        print("seed_text : " + '\n' + seed_text)

        seq_length = 10
        # generate new text
        generated = generate_sequence(model, tokenizer, seq_length, seed_text, 10)
        print("generated : ", generated)
        
        new_row = pd.Series(data={'class':label_name, 'input_sen':seed_text, 'generate_sequence':generated}, name='x')
        generated_sen = generated_sen.append(new_row, ignore_index=False)

    generated_sen = pd.DataFrame(generated_sen)
    dfi.export(generated_sen, '../reports/dataframe.png')

if __name__ == '__main__':
    main()
