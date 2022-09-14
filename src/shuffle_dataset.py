import os
import numpy as np
import pandas as pd
import torch
import pickle
from datasets import load_dataset
import math
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")
import random
import argparse

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

def preparedata(args):
    '''
    create unshuffled and shuffled dataset
    '''
    if args.dataset == "xsum":
        dataset = load_dataset(args.dataset)
    else:
        raise ValueError("Undefined dataset")

    train_dataset, val_dataset, test_dataset = get_dataset(dataset, "train", args), get_dataset(dataset, "validation", args), get_dataset(dataset, "test", args)
    save_csv(train_dataset,'train'), save_csv(val_dataset,'val'), save_csv(test_dataset,'test')
    train_dataset_shuff, val_dataset_shuff,  test_dataset_shuff = shuffle_sentence(train_dataset), shuffle_sentence(val_dataset), shuffle_sentence(test_dataset)
    save_csv_shuffled(train_dataset_shuff,'train', args), save_csv_shuffled(val_dataset_shuff,'val', args), save_csv_shuffled(test_dataset_shuff,'test', args)

def get_dataset(dataset, mode, args):
    '''
    create dataset of approx 512-token document
    specifically, 485-512 token document
    '''
    data = dataset[mode]
    path =  f"../datalength/{mode}_info.csv"
    df = pd.read_csv(path)
    
    if args.orig_source_length == 512:
        cond1 = df["doc len"] >= math.floor(0.95*args.orig_source_length-1) #485, 972
    else:
        raise ValueError("undefined...")
        
    cond2 = df["doc len"] <= args.orig_source_length #512, 1024
    cond3 = df["sum len"] <= args.max_target_length

    index = df['index'][cond1 & cond2 & cond3]
    
    print(f"Filtered {mode} Dataset: {len(data[index]['id'])}")
    
    return data[index]

def shuffle_sentence(dataset):
    '''
    shuffle order of sentences in documents
    '''
    
    shuffled_sent_doc = []
    shuffled_sent_ids = []
    summary = []
    for i in range(len(dataset['document'])):
        document = dataset['document'][i]
        sent_text = sent_tokenize(document)
        random.shuffle(sent_text)
        shuffed_document = ' '.join(sent_text)
        
        shuffled_sent_doc.append(shuffed_document)
        shuffled_sent_ids.append(dataset['id'][i] )
        summary.append(dataset['summary'][i])
    
    return (shuffled_sent_ids, shuffled_sent_doc, summary)

def save_csv(dataset, mode):
    '''
    save shuffled dataset as csv
    '''
    df = pd.DataFrame(zip(dataset['id'], dataset['document'], dataset['summary']),
               columns =['id', 'document', 'summary'])
    
    df.to_csv(f'../dataset/{mode}.csv')
    print(f"{mode} saved")    
    
def save_csv_shuffled(dataset, mode, args):
    '''
    save unshuffled dataset as csv
    '''
    df = pd.DataFrame(zip(dataset[0], dataset[1], dataset[2]),
               columns =['id', 'document', 'summary'])
    if not os.path.exists('../dataset'):
        os.makedirs('../dataset')
    df.to_csv(f'../dataset/{mode}_shuffled_seed{args.seed}.csv')
    print(f"{mode} saved")
    
# config = dict(
#     data = "xsum",    
#     orig_source_length = 512,
#     max_target_length = 36,
#     seed = 42,
# )
        
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="specify dataset", type=str)
parser.add_argument("--orig_source_length", help="length of source document", type=int)
parser.add_argument("--max_target_length", help="maximum of the reference summary", type=int)
parser.add_argument("--seed", help="specify seed", type=int)
args = parser.parse_args()


if __name__ == "__main__": 
    
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    preparedata(args)
     