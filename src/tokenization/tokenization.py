import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import train_test_split, ShuffleSplit
import dataframe_image as dfi
from statistics import mean

import logging

logging.basicConfig(filename='../../logs/tokenization.log',  level=logging.DEBUG)

def experiment():
    
    path = '../../data/cleaned/data_cleand.json'
    data = pd.read_json(path)['selftext_clean'].to_numpy()

    train_path = "train_data.txt"
    
    list_vocab_size = [60, 500, 2000, 5000, 10067]
    
    result = []
    str_result = ""
    
    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=22)

    for vocab_size in list_vocab_size:
        re = [] 
        re.append(vocab_size)  
        iteration = 1
        for train_index, test_index in rs.split(data):
            
            train_data = data[train_index]
            test_data = data[test_index]

            with open(train_path, "w", encoding="utf8") as f:
                f.write("\n".join(train_data))

            source = 'working_dir/tokenization_{}_vocabsize__iteration_{}.model'.format(vocab_size, iteration)
            
            spm.SentencePieceTrainer.train(input=train_path, model_prefix=source, vocab_size=vocab_size, unk_id=3, model_type='word')     # train the spm model
            sp = spm.SentencePieceProcessor()                                                               # create an instance; this saves .model and .vocab files 

            sp.load('{}.model'.format(source))
            
            unk_count, token_count = 0, 0

            for row_data in test_data:
                subword_tokens = sp.encode_as_ids(row_data)
                # subword_tokens = sp.encode_as_pieces(row_data)
                token_count += len(subword_tokens)
                unk_count += subword_tokens.count(3) #unk count

            percent = round((unk_count/token_count*100), 2)
            re.append(percent)

            logging.info("in iteration {} vocab_size {} : unk token count {} , percent {}".format(iteration, vocab_size, unk_count, percent))
            print("in iteration {}  {} :  {}".format(iteration, vocab_size, unk_count))
            iteration += 1
            str_result += "in iteration {} vocab_size {} : unk token count {} , percent {}\n".format(iteration, vocab_size, unk_count, percent)

        result.append(re)
        avg = mean(re[1:])
        re.append(avg)
        str_result += "Average vocab_size {} is {} \n \n".format(vocab_size, avg)

    #final report
    with open('../../reports/tokenization.txt', "w", encoding="utf8") as f:
        f.write(str_result)
    
    df = pd.DataFrame(result, columns=['token count','1','2','3','4','5','average unk token percent'])
    
    print(df.head())
    dfi.export(df, '../../reports/tokenization.png')

def best_tokenization():
    path = '../../data/cleaned/data_cleand.json'
    data = pd.read_json(path)['selftext_clean'].to_numpy()

    train_path = "train_data.txt"
    train_data, _ = train_test_split(data, test_size=0.2)
    
    with open(train_path, "w", encoding="utf8") as f:
                f.write("\n".join(train_data))

    source = '../../models/tokenization/tokenization_best_model'
    
    spm.SentencePieceTrainer.train(input=train_path, model_prefix=source, vocab_size=10067, unk_id=3, model_type='word')     # train the spm model
    sp = spm.SentencePieceProcessor()                                                               # create an instance; this saves .model and .vocab files 

    sp.load('{}.model'.format(source))

def main():
    experiment()
    best_tokenization()
        
        
if __name__ == '__main__':
    main()
