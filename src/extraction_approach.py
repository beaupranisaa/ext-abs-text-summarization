import math
import re
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
nltk.download('stopwords')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from extractive.lsa import *
from extractive.luhn import *
from extractive.textrank import *

# from summarizer import Summarizer #bertbase
# from summarizer.sbert import SBertSummarizer

# from sklearn.cluster import KMeans
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial import distance
import numpy as np
import numpy.matlib

# from sklearn.feature_extraction.text import TfidfVectorizer

# from classic_extractive import *

import transformers
from transformers import T5Tokenizer, BertTokenizer
import warnings
warnings.filterwarnings('ignore')

import os
# os.environ['TRANSFORMERS_CACHE'] = '../../../cache/huggingface/transformers'


# tokenizer.from_pretrained(os.path.join(config.output_dir, 'checkpoints/current_model'))
class TokenLevel:
    def __init__(self, source_text, source_ids, source_len, max_source_len):
        self.source_text = source_text
        self.source_ids = source_ids
        self.source_len = source_len
        self.end_eos = int(torch.where(self.source_ids == 1)[0])
        self.max_source_len = max_source_len
        
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
    def _get_fulltext(self, source_ids, source_len):
        source_ids_short = source_ids[0:source_len] # the first two tokens are "summarize" and ":"
        shortened_source_text = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in source_ids_short]
        return shortened_source_text, source_len

    def _get_head(self, source_ids, max_source_len):
        source_ids_short = source_ids[0:max_source_len]
        shortened_source_text = self.tokenizer.decode(source_ids_short, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return shortened_source_text, torch.count_nonzero(source_ids_short.to(dtype=torch.long))
    
    def _get_tail(self, source_ids, end_eos, max_source_len):
        source_ids_short = source_ids[end_eos-max_source_len+1:end_eos+1] 
        shortened_source_text = self.tokenizer.decode(source_ids_short, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return shortened_source_text, torch.count_nonzero(source_ids_short.to(dtype=torch.long))
    
    def _get_headtail(self, source_ids, end_eos, max_source_len, head_ratio):
        head_ids = source_ids[0:math.floor(head_ratio*max_source_len)+1]
        tail_ids = source_ids[end_eos - math.floor((1-head_ratio)*max_source_len)+1:end_eos+1]
        source_ids_short = torch.cat((head_ids, tail_ids),0)
        shortened_source_text = self.tokenizer.decode(source_ids_short, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return shortened_source_text, torch.count_nonzero(source_ids_short.to(dtype=torch.long))

    def shorten(self, approach):
        if "full-text" in approach:
            source_text_short, source_text_short_len = self._get_fulltext(self.source_ids, self.source_len)
        
        elif "head-only" in approach:
            source_text_short, source_text_short_len = self._get_head(self.source_ids, self.max_source_len)
        
        elif "tail-only" in approach:
            source_text_short, source_text_short_len = self._get_tail(self.source_ids, self.end_eos, self.max_source_len)
        
        elif "head+tail" in approach:
            head_ratio = float(re.findall(r"\d*\.\d", approach)[0])
            source_text_short, source_text_short_len = self._get_headtail(self.source_ids, self.end_eos, self.max_source_len, head_ratio)
        else:
            raise ValueError("Undefined approach ...") 
        
        return source_text_short, source_text_short_len    
    
class SentenceLevel:
    def __init__(self, source_text, source_ids, source_len, max_source_len):
        self.source_text = source_text
        self.source_ids = source_ids
        self.source_len = source_len
        self.max_source_len = max_source_len

        self.parser, self.sentence_count = self._parser(self.source_text)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
    def _parser(self, source_text):
        paragraph_split = sent_tokenize(source_text)
        sentences = [i for i in paragraph_split]
        sentence_count = len(sentences)
        parser = PlaintextParser.from_string(source_text,Tokenizer("english"))        
        return parser, sentence_count
    
    def _get_luhn(self):  
        summarizer = LuhnSummarizer()
        summary, summary_len = self._get_summary(summarizer, self.sentence_count, self.parser)
        return  summary, summary_len
    
    def _get_lsa(self):
        summarizer = LsaSummarizer()
        summary, summary_len = self._get_summary(summarizer, self.sentence_count, self.parser)
        return  summary, summary_len
    
    def _get_textrank(self):
        summarizer = TextRankSummarizer()
        summary, summary_len = self._get_summary(summarizer, self.sentence_count, self.parser)
        return  summary, summary_len

    def _get_kmeansbert(self, source_text):
        summarizer = KmeansBERTSummarizer()
        summary, summary_len = self._get_summary(summarizer, self.sentence_count)
        return  summary, summary_len        
    
    def _get_summary(self, summarizer, sentence_count, parser= None):    
        sum_candidates = []
        for i in range(sentence_count):
            if self.approach == "kmeansbert":
                summary = summarizer(self.source_text,i)
                full_summary = ' '.join([sentence for sentence in summary])
            else:
                summary = summarizer(parser.document,i)
                full_summary = ' '.join([sentence._text for sentence in summary])
            sum_candidates.append(full_summary)
        source = self.tokenizer.batch_encode_plus(sum_candidates, max_length = 1024, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", ) 
        token_count = torch.count_nonzero(source['input_ids'], axis = 1)
        idx = (token_count == min(token_count, key=lambda x:abs(x-self.max_source_len))).nonzero().flatten()
        return sum_candidates[idx], token_count[idx][0]

    def shorten(self, approach):
        self.approach = approach
        
        if self.approach == "luhn":
            source_text_short, source_text_short_len = self._get_luhn()
            
        elif self.approach == "lsa":
            source_text_short, source_text_short_len = self._get_lsa()     
            
        elif self.approach == "textrank":
            source_text_short, source_text_short_len = self._get_textrank()

#         elif self.approach == "bertbased":
#             source_text_short, source_text_short_len = self._get_bertbased(self.source_text, self.source_ids, self.max_source_len)
        
        elif self.approach == "kmeansbert":
            source_text_short, source_text_short_len = self._get_kmeansbert(self.source_text, self.source_ids, self.max_source_len)
        
#         elif self.approach == "bertbased_full":
#             source_text_short, source_text_short_len = self._get_bertbased_full(self.source_text, self.source_ids, self.max_source_len)
            
#         elif self.approach == "bertbased_setk":
#             source_text_short, source_text_short_len = self._get_bertbased_setk(self.source_text, self.source_ids, self.max_source_len)            
            
        else:
            raise ValueError("Undefined strategy ...") 
        
        return source_text_short, source_text_short_len    
    
    def _get_bertbased(self, source_text, source_ids, max_source_len, mode):
        # try:
        os.environ['TRANSFORMERS_CACHE'] = '../../../cache/huggingface/transformers'
        paragraph_split = sent_tokenize(source_text)
        sentences = [i for i in paragraph_split]

        # Want BERT instead of distilBERT? Uncomment the following line:
        model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer

        # tokenizer = BertTokenizer.from_pretrained("./cache/bert-base-uncased")
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

        source = tokenizer.batch_encode_plus(
            sentences,
            max_length = 512, #self.max_source_len
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        with torch.no_grad():
            hidden_states = model(source_ids, attention_mask=source_mask)

        last_hidden_states = hidden_states[0]
        sentence_features = last_hidden_states[:,0,:].detach().numpy() #[cls]

        Sum_of_squared_distances = [KMeans(n_clusters=k, random_state=42, init = 'k-means++').fit(sentence_features).inertia_ for k in range(1,len(paragraph_split))]

        elbow = self._get_elbow_point(Sum_of_squared_distances)
        # print("ELBOW: ", elbow)

        number_extract = elbow
        print("Elbow: ", number_extract)

        # print("Performing KMeans.............")
        kmeans = KMeans(n_clusters=number_extract, 
                        random_state=42, init = 'k-means++').fit(sentence_features)
        label = kmeans.fit_predict(sentence_features)
        cluster_center = kmeans.cluster_centers_

        # args = _find_closest_args(sentence_features, cluster_center)
        indices = self._find_closest_args(sentence_features, cluster_center)
        # indices = np.sort(list(args.values()))

        topic_answer = [paragraph_split[i] for i in indices]

        full_summary = ' '.join([sentence for sentence in topic_answer])

        # print(full_summary)

        # tokenizerT5 = T5Tokenizer.from_pretrained("t5-small")
        tokenizerT5 = T5Tokenizer.from_pretrained("../model/512_256_head-only/checkpoints/best_model")
        source_new = tokenizerT5.batch_encode_plus(topic_answer, max_length = 2048, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )
        token_count = torch.count_nonzero(source_new['input_ids']) 

        return full_summary, token_count #, elbow
        # except:
        #     return source_text, self.source_len #, 0        

    def _get_bertbased_setk(self, source_text, source_ids, max_source_len, mode):
        try:
            print("setk")
            paragraph_split = sent_tokenize(source_text)
            sentences = [i for i in paragraph_split]

            # Want BERT instead of distilBERT? Uncomment the following line:
            model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')

            # Load pretrained model/tokenizer

            # tokenizer = BertTokenizer.from_pretrained("./cache/bert-base-uncased")
            # os.environ['TRANSFORMERS_CACHE'] = '../../../cache/huggingface/transformers'

            tokenizer = tokenizer_class.from_pretrained('./cache/bert-base-uncased/')
            # model = model_class.from_pretrained( os.path.join('../../../cache/huggingface/transformers', pretrained_weights))

            # tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            model = model_class.from_pretrained(pretrained_weights)

            source = tokenizer.batch_encode_plus(
                sentences,
                max_length = 512, #self.max_source_len
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            source_ids = source["input_ids"].squeeze()
            source_mask = source["attention_mask"].squeeze()

            with torch.no_grad():
                hidden_states = model(source_ids, attention_mask=source_mask)

            last_hidden_states = hidden_states[0]
            sentence_features = last_hidden_states[:,0,:].detach().numpy() #[cls]

            elbow = 5

            # Sum_of_squared_distances = [KMeans(n_clusters=k, random_state=42, init = 'k-means++').fit(sentence_features).inertia_ for k in range(1,len(paragraph_split))]

            # elbow = self._get_elbow_point(Sum_of_squared_distances)
            # print("ELBOW: ", elbow)

            number_extract = elbow
            print("Elbow: ", number_extract)

            # print("Performing KMeans.............")
            kmeans = KMeans(n_clusters=number_extract, 
                            random_state=42, init = 'k-means++').fit(sentence_features)
            label = kmeans.fit_predict(sentence_features)
            cluster_center = kmeans.cluster_centers_

            # args = _find_closest_args(sentence_features, cluster_center)
            indices = self._find_closest_args(sentence_features, cluster_center)
            # indices = np.sort(list(args.values()))

            topic_answer = [paragraph_split[i] for i in indices]

            full_summary = ' '.join([sentence for sentence in topic_answer])

            # print(full_summary)

            # tokenizerT5 = T5Tokenizer.from_pretrained("t5-small")
            tokenizerT5 = T5Tokenizer.from_pretrained("../model/512_256_head-only/checkpoints/best_model")
            source_new = tokenizerT5.batch_encode_plus(topic_answer, max_length = 2048, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )
            token_count = torch.count_nonzero(source_new['input_ids']) 

            return full_summary, token_count #, elbow
        except:
            return source_text, self.source_len
    
    def _get_bertbased_full(self,source_text, source_ids, max_source_len, mode):
        try:
            if mode == "combo": 
                sentence_count = source_text.count(".") + 1
            else:
                paragraph_split = sent_tokenize(source_text)
                sentences = [i for i in paragraph_split]
                sentence_count = len(sentences)
                # sentence_count = source_text.count("\n") + 1 # count number of sentences
            token_count = torch.count_nonzero(source_ids)
            n_token_per_sentence = token_count/sentence_count
            n_sum_sentence = int(math.floor(max_source_len/n_token_per_sentence))
            summarizer_bert = Summarizer()
            tokenizer = T5Tokenizer.from_pretrained("../model/512_256_head-only/checkpoints/best_model")
            sum_candidates = []
    #         print("HI: ",n_sum_sentence)
            for i in range(n_sum_sentence-1, n_sum_sentence +2):
                summary = summarizer_bert(source_text, num_sentences = i)
                full_summary = ''.join(summary)
                sum_candidates.append(full_summary)
            source = tokenizer.batch_encode_plus(sum_candidates, max_length = 512, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors="pt", )   # change to 1024 from 512     
            token_count = torch.count_nonzero(source['input_ids'], axis = 1)   
            idx = torch.argmin(token_count % self.max_source_len)
            return sum_candidates[idx], token_count[idx]
        except:
            print("oops")
            return source_text, self.source_len