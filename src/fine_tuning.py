#Preparing the datasets
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from plot_dict import plot_dict
from time import time
import pandas as pd
import argparse
import logging
import torch
import math
import re


argp = argparse.ArgumentParser()
argp.add_argument('model_type',
    help="finue-tuning GPT2 language model or Bert classification",
    choices=["GPT2", "Bert"])
args = argp.parse_args()

if args.model_type == 'Bert':
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)   
    model = BertForSequenceClassification.from_pretrained(model_name)
    logging.basicConfig(filename='../logs/fine_tuning_bert.log',  level=logging.INFO)

if args.model_type == 'GPT2':
    model_name = 'distilgpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    logging.basicConfig(filename='../logs/fine_tuning_GPT2.log',  level=logging.INFO)

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    logging.info('There are %d GPU(s) available.' % torch.cuda.device_count())

else:
    print('No GPU available, using the CPU instead.')
    logging.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#########################################################################################

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def read_dataset(label= 2, test_size=0.2):
    path = '../data/cleaned/data_cleand.json'
    dataset = pd.read_json(path)
    dataset = dataset.sample(frac = 1)
    
    if label != 2:
        dataset = dataset[dataset.label == label]

    print(dataset.head())
    documents = dataset['selftext_clean'].to_list()
    labels = dataset['label'].to_list()

    # split into training & testing a return data as well as label names
    return train_test_split(documents, labels, test_size=test_size)

def bert_understanding():
    # Get all of the model's parameters as a list of tuples.
    #just to understand bert model
    params = list(model.named_parameters())

    logging.info('The BERT model has {:} different named parameters.\n'.format(len(params)))

    logging.info('==== Embedding Layer ====\n')

    for p in params[0:5]:
        logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    logging.info('\n==== First Transformer ====\n')

    for p in params[5:21]:
        logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    logging.info('\n==== Output Layer ====\n')

    for p in params[-4:]:
        logging.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train_evaluate_bert_model(train_dataset, eval_dataset):
    
    logging.info('Start Trainig')
    # training_args = TrainingArguments("test_trainer")
    
    training_args = TrainingArguments(
        output_dir='../models/bert_classification_lm',          # output directory
        evaluation_strategy="epoch",     # evaluate each `logging_steps`
        num_train_epochs=4,              # total number of training epochs
        per_device_train_batch_size=4,   # batch size per device during training
        per_device_eval_batch_size=4,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='../logs/bert_classification_lm ',           # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=200,               # log & save weights each logging_steps
        save_total_limit=1,              # limit the total amount of checkpoints. Deletes the older checkpoints.
        
    )

    t = time()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    logging.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    
    logging.info('Start Evaluation')
    
    t = time()
    
    trainer.evaluate()
    
    logging.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    
    # trainer.save_model()
    model_path = "../models/bert_classification_lm"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    plot(trainer, training_args, 'Bert')

def get_bert_classification_prediction(text,model, tokenizer, max_length=512):
    logging.info("input text => " + text)
    # prepare our text into tokenized sequence
    target_names = {1 : "happiness", 0 : "depression"}
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    
    logging.info("predicted lable for input text => " + str(target_names[probs.argmax().item()]))
    
def bert_predict_text_class():
    
    # reload our model/tokenizer. Optional, only usable when in Python
    model_path = "../models/bert_classification_lm"
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to("cuda")
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    #depression
    text = "I feel like my depression took my entire teen years from me. I spent all my free time in high school crying and self harming at home while everyone else had friends and hobbies and went to parties."
    get_bert_classification_prediction(text, model, tokenizer)
    
    #happiness
    text = "Hello people of reddit, I'm in a dilemma. I feel like I need to reinvent myself. I'm very happy with my gf (not a problem), my family is ok, sometimes we have problems but you got to deal with it sometimes. "
    get_bert_classification_prediction(text, model, tokenizer)
    
    #happiness
    text = "I have some of those friends but some of them take life a joke and it gets annoying.\n\nI feel like I need to improve my current situation. I want a more mature friend group, I want to move out of my parents house but I work in a minimum wage job. I'm in major which I feel is bs unless I go to law school (political science, I chose it because I'm terrible at math and I love the political realm). a part of me wants to open up a small business but I don't know where to start. I'm stuck in a rut, I was very consistent with working out but for the past months I messed up my wrist, had my wisdom teeth pulled out, and got sick so it hasn't exactly been the best situation lol.\n\nin conclusion, I'm a 19 yr old man trying figure out my place in this world. I want to open a small business to support me and my future family needs. I'm practical so I don't care about \"luxury\" everyone breaks their backs for(I try to practice minimalism) . I want to feel healthy and I want to have a group of friends that are mature and don't fuck around too much and actually respect me for trying to improve ( they roast me for trying to stay completely sober and trying to sleep earlier etc.) . I want to move due to the fact that sometimes I feel like I'm not home due to all arguing I have to go through. I want challenge, I want to be independent."
    get_bert_classification_prediction(text, model, tokenizer)
    
    #depression
    text = " I didnt even move out from my parents house because I was too anxious to stay in a dorm. My ex used to tell me about how much fun they had in high school and how they were in a bunch of clubs and how they had the most fun in college staying in dorms with all their friends and doing fun, dumb stuff. It always destroys me to think about how I missed out on all of that because of something I couldn't control. Now I am 21 and just graduated college and Iam lost. I have no friends, no partner, no job, me and my family are on rough terms and I just got diagnosed with anorexia. It is just crushing me and idk what to do anymore. And I\u2019m afraid to tell anyone because I don\u2019t want to be sent back to a psych ward since I have no money to pay for it again. I\u2019m just so broken right now. Can anyone relate to this crippling regret of missing out on what were supposed to be the best years of your life?"
    get_bert_classification_prediction(text, model, tokenizer)

#################################################################################################
def build_text_files(data_json, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for texts in data_json:
        summary = str(texts).strip()
        summary = re.sub(r"\s", " ", summary)
        data += summary + "  "
    f.write(data)

def load_dataset_GPT2(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

def train_evaluate_GPT2_model(train_dataset, eval_dataset, data_collator, label_name):
    
    logging.info('Start Trainig')
    # training_args = TrainingArguments("test_trainer")
    
    training_args = TrainingArguments(
        output_dir='../models/{}GPT2_lm'.format(label_name),          # output directory
        evaluation_strategy="epoch",     # evaluate each `logging_steps`
        num_train_epochs=4,              # total number of training epochs
        per_device_train_batch_size=4,   # batch size per device during training
        per_device_eval_batch_size=4,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='../logs/{}GPT2_lm'.format(label_name),           # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=200,               # log & save weights each logging_steps
        save_total_limit=1,              # limit the total amount of checkpoints. Deletes the older checkpoints.
        
    )

    t = time()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    logging.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    
    logging.info('Start Evaluation')
    
    t = time()
    
    trainer.evaluate()
    
    logging.info('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    
    trainer.save_model()

    plot(trainer, training_args, 'GPT2', label_name)

def text_generation_GPT2(label):
    text_generatior = pipeline('text-generation',
                                 model='../models/{}GPT2_lm'.format(label), 
                                 tokenizer='distilgpt2')
                                #  config={'max_length':50})
    
    #happiness
    input_text = "Hi, we have created a success forum for"
    logging.info(text_generatior(input_text)[0]['generated_text'])
    
    #depresion
    input_text = "I'm so depressed. I have nothing to live"
    logging.info(text_generatior(input_text)[0]['generated_text'])
    # text_generatior()


def plot(trainer, training_args, model_, label=''):
    # Keep track of train and evaluate loss.
    loss_history = {'train_loss':[], 'eval_loss':[]}

    # Keep track of train and evaluate perplexity.
    # This is a metric useful to track for language models.
    perplexity_history = {'train_perplexity':[], 'eval_perplexity':[]}

    # Loop through each log history.
    for log_history in trainer.state.log_history:

        if 'loss' in log_history.keys():
            # Deal with trianing loss.
            loss_history['train_loss'].append(log_history['loss'])
            perplexity_history['train_perplexity'].append(math.exp(log_history['loss']))
        
        
        elif 'eval_loss' in log_history.keys():
            # Deal with eval loss.
            loss_history['eval_loss'].append(log_history['eval_loss'])
            perplexity_history['eval_perplexity'].append(math.exp(log_history['eval_loss']))
    
    try:
        if args.model_type == 'Bert':
            # Plot Losses.
            plot_dict(loss_history, start_step=training_args.logging_steps, 
                    step_size=training_args.logging_steps, use_title='Loss', 
                    use_xlabel='Train Steps', use_ylabel='Values', magnify=2, path='../reports/loss_history_{}.png'.format(model_))
    except:
        print("there is errror with plot")

    if args.model_type == 'GPT2':
        plt.style.use('ggplot')

        # Place legend best position.
        # Plot Losses.
        plt.plot(loss_history['train_loss'])
        plt.plot(loss_history['eval_loss'])
        plt.title('Loss'.format(label))
        plt.xlabel('value')
        plt.ylabel('step')
        plt.savefig('../reports/loss_history_{}_{}.png'.format(model_, label))
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.cla()
        plt.clf()

def main():
    
    if args.model_type == 'Bert':
        bert_understanding()

        max_length = 512
        logging.info('Start prepare data')
        
        t = time()
        (train_texts, valid_texts, train_labels, valid_labels) = read_dataset()

        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
        valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
        
        # convert our tokenized data into a torch Dataset
        train_dataset = NewsGroupsDataset(train_encodings, train_labels)
        eval_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
        logging.info('Time to prepare : {} mins'.format(round((time() - t) / 60, 2)))
        
        train_evaluate_bert_model(train_dataset, eval_dataset)
        
        bert_predict_text_class()
    
    if args.model_type == 'GPT2':
        labels = {0 : "depression",1 : "happiness"}
        
        for label_code, label_name  in labels.items():
            
            max_length = 128

            logging.info('Start prepare data')
            
            t = time()
            
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            train_path = '../data/GPT2/train_dataset.txt'
            test_path = '../data/GPT2/test_dataset.txt'
            
            (train, test, _, _) = read_dataset(label_code)

            build_text_files(train, train_path)
            build_text_files(test, test_path)

            train_dataset, eval_dataset, data_collator = load_dataset_GPT2(train_path,test_path,tokenizer)

            logging.info('Time to prepare : {} mins'.format(round((time() - t) / 60, 2)))

            train_evaluate_GPT2_model(train_dataset, eval_dataset, data_collator, label_name)

            text_generation_GPT2(label_name)

if __name__ == '__main__':
    main()

