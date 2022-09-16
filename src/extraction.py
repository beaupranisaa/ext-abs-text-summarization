# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pickle

from datasets import load_dataset
from extraction_dataset import Dataset
import math
import argparse

from transformers import T5Tokenizer

os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

def preparedata(config, args):
    
    train_dataset, val_dataset, test_dataset = get_dataset("train", config, args), get_dataset("val", config, args), get_dataset("test", config, args)
    training_loader, val_loader,  test_loader = get_loader(train_dataset, config), get_loader(val_dataset, config), get_loader(test_dataset, config)
    
    return training_loader, val_loader,  test_loader

def get_dataset( mode, config, args):
    if args.shuffle == True:
        print("SHUFFLED")
        path =  f"../dataset/{mode}_shuffled_seed{args.seed}.csv"
    else:
        path =  f"../dataset/{mode}.csv"
        
    df = pd.read_csv(path)
    
    ## preprocessed_data.rename(columns = {'Sample ids':'id'}, inplace = True)
    # total_row = df.shape[0]
    # df = df[1408:]
    # df.set_index(np.arange(0, total_row-1408), inplace=True)
    
    
    config['source_text'] = 'document'
    config['target_text'] = 'summary'
    
    print(f"Filtered {mode} Dataset: {len(df['id'])}")
    
    tokenizer = T5Tokenizer.from_pretrained(config['model'])

    # Creating the Training and Validation dataset for further creation of Dataloader
    data_set = Dataset(
        df,
        tokenizer,
        config['model'],
        config['max_source_length'],
        config['source_text'],
        config['target_text'],
        args.approach,
    )
    
    return data_set

def get_loader(dataset, config): 
    loader_params = {
        "batch_size": config['batch_size'],
        "shuffle": False,
        "num_workers": 0,
    }   
    loader = DataLoader(dataset, **loader_params)
    
    print(f"Loader Length: {len(loader)}")
    return loader


def process(loader, config, args, mode):
    results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Shortened Document length": []} 
    
    if args.approach in ['head-only','tail-only','head+tail0.2', 'head+tail0.5']:
        if args.shuffle == True:
            path = f'''../extracted/truncation/{args.approach}/shuffled/{mode}_seed{args.seed}/'''
        else:
            path = f'''../extracted/truncation/{args.approach}/unshuffled/{mode}'''
    else:
        if args.shuffle == True:
            path = f'''../extracted/extractive/{args.approach}/shuffled/{mode}_seed{args.seed}/'''
        else: 
            path = f'''../extracted/extractive/{args.approach}/unshuffled/{mode}'''
            
    if not os.path.exists(path):
        os.makedirs(path)

    for _, data in enumerate(loader, 0):
        # _ = _ + 11
        results['Sample ids'].extend(data['ids'].tolist())
        results['Document'].extend(data['source_text'])
        results['Shortened Document'].extend(data['shortened_source_text'])
        results['Summary'].extend(data["target_text"])
        results['Document length'].extend(data['source_len'].tolist())
        results["Shortened Document length"].extend(data['source_text_short_len'].tolist())
        
        print("STEP: ", _,"/",len(loader))
        final_df = pd.DataFrame(results)
        final_df.to_csv(os.path.join(path, f"""step{_}.csv"""""))   
        print("SAVE TO CSV FINISHED")
        results = {"Sample ids": [], "Document": [], "Shortened Document": [], "Summary": [], "Document length": [] , "Shortened Document length": []}                  

config = dict(
    model = "t5-small",  
    batch_size = 128,
    orig_source_length = 512,
    max_source_length = 254, # 512, 373, 323, 273
    seed = 42,
)

parser = argparse.ArgumentParser()
parser.add_argument("--approach", help="specify extraction approach", type=str)
parser.add_argument("--shuffle", default=False, help="select unshuffled or shuffled", type=bool) 
parser.add_argument("--seed", help="specify seed of shuffled dataset", type=int) 
args = parser.parse_args()

if __name__ == "__main__": 
    torch.manual_seed(config['seed'])  # pytorch random seed
    np.random.seed(config['seed'])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    train_loader, val_loader, test_loader = preparedata(config, args)
    process(test_loader, config, args, "test_set")
    process(val_loader, config, args, "val_set")
    process(train_loader, config, args, "train_set")
     