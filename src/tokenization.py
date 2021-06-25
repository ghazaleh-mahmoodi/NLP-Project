import glob
import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import ShuffleSplit

import logging

logging.basicConfig(filename='../logs/tokenization.log',  level=logging.DEBUG)

def experiment():
    
    path = '../data/' + '/cleaned/data_cleand.json'
    data = pd.read_json(path)['selftext_clean'].to_numpy()

    rs = ShuffleSplit(n_splits=5, test_size=0.2, train_size=None)
    
    list_vocab_size = [60, 500, 1000, 2000, 5000]
    for vocab_size in list_vocab_size:
        iteration = 1
        for train_index, test_index in rs.split(data):
            train_data, test_data = data[train_index], data[test_index]
            
            train_data = list(filter(None, train_data))
            test_data = list(filter(None, test_data))

            train_path = "working_dir/tokenization/train_data{}.txt".format(str(iteration))
            with open(train_path, "w", encoding="utf8") as f:
                f.write("\n".join(train_data))

            
            source = '../models/tokenization/tokenization_{}_vocabsize__iteration_{}.model'.format(vocab_size, iteration)

            spm.SentencePieceTrainer.train(input=train_path, model_prefix=source, vocab_size=vocab_size)     # train the spm model
            sp = spm.SentencePieceProcessor()                                                               # create an instance; this saves .model and .vocab files 

            sp.load('{}.model'.format(source))
            
            unk = 0

            out_data = []
            for row_data in test_data:
                subword_tokens = sp.encode_as_ids(row_data)
                unk += subword_tokens.count(0)
                
                # subword_tokens = sp.encode_as_pieces(row_data)
                # unk += subword_tokens.count('<unk>')
                out_data.append(subword_tokens)
        
            # output_path = ".working_dor/tokemization/{}encode_as_pieces.txt".format(str(iteration))
            # with open(output_path, "w", encoding="utf8") as f:
            #     f.write("\n".join([str(x) for x in out_data]))

            logging.info("in iteration {} vocab_size {} : unk token count {}".format(iteration, vocab_size, unk))
            print("in iteration {}  {} :  {}".format(iteration, vocab_size, unk))
            iteration += 1

def best_tokenization():
    pass

def main():
    experiment()
    best_tokenization()
        
        
if __name__ == '__main__':
    main()
