# from transformers import T5Tokenizer,
import math
from torch.utils.data import Dataset
import torch
import pandas as pd
from tokenizers import decoders
import re
import sys   
from extraction_approach import *
import numpy as np

class Dataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    ideally, loading it into the dataloader to pass it to the T5 directly.
    But here we save the result as csv as our extracted document
    """
    def __init__(
        self, dataframe, tokenizer, model_name, max_source_len, source_text, target_text, approach = "luhn", mode = None, 
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            source_text (str): column name of source text
            target_text (str): column name of target text
            approach (str): extraction approach
        """
        
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.data = dataframe
        self.max_source_len = max_source_len
        self.approach = approach
        self.source_text = self.data[source_text]
        self.target_text = self.data[target_text]
        
        self.ids = self.data['id']

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the ids, source_text, shortened_source_text, target_text, source_len, source_text_short_len"""
        
        ids = self.ids[index]
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        
        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())
            
        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length = 1024, #self.max_source_len
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        source_ids = source["input_ids"].squeeze()
        source_len = torch.count_nonzero(source_ids.to(dtype=torch.long))
        
        if self.approach in ["head-only", "tail-only", "head+tail0.2", "head+tail0.5"]:
            approach = TokenLevel(self.source_text[index], source_ids, source_len, self.max_source_len)
        else:
            approach = SentenceLevel(self.source_text[index], source_ids, source_len, self.max_source_len)
            
        source_text_short, source_text_short_len = approach.shorten(self.approach)
        
        return {
            "ids": ids,
            "source_text": source_text,
            "shortened_source_text": source_text_short,
            "target_text": target_text,
            "source_len": source_len,
            "source_text_short_len": source_text_short_len,
        }

