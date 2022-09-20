import numpy as np
from sklearn.cluster import KMeans
from typing import List


from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from tqdm import tqdm
from numpy import ndarray
from typing import List
from nltk import sent_tokenize

class BertParent(object):

    model_handler = {'bert': BertModel,}

    token_handler = {'bert': BertTokenizer}

    size_handler = {'base': {'bert': 'bert-base-uncased',},
                    'large': {'bert': 'bert-large-uncased',}}

    vector_handler = {'base': {'bert': 768,},
                      'large': {'bert': 1024,}}

    def __init__(self, model_type: str, size: str):
        self.model = self.model_handler[model_type].from_pretrained(self.size_handler[size][model_type])
        self.tokenizer = self.token_handler[model_type].from_pretrained(self.size_handler[size][model_type])
        self.vector_size = self.vector_handler[size][model_type]
        self.model_type = model_type
        self.model.eval()

    def tokenize_input(self, text) -> torch.tensor:
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens])

    def extract_embeddings(self, text: str, use_hidden=True, squeeze=False) -> ndarray:
        tokens_tensor = self.tokenize_input(text)
        hidden_states, pooled = self.model(tokens_tensor)
        if use_hidden:
            pooled = hidden_states[-2].mean(dim=1)
        if squeeze:
            return pooled.detach().numpy().squeeze()
        return pooled

    def create_matrix(self, content: List[str], use_hidden=False) -> ndarray:
        train_vec = np.zeros((len(content), self.vector_size))
        for i, t in tqdm(enumerate(content)):
            train_vec[i] = self.extract_embeddings(t, use_hidden).data.numpy()
        return train_vec


class ClusterFeatures(object):

    def __init__(self, features, algorithm='kmeans', pca_k=None):
        if pca_k:
            self.features = PCA(n_components=pca_k).fit_transform(features)
        else:
            self.features = features
        self.algorithm = algorithm
        self.pca_k = pca_k

    def __get_model(self, k):
        return KMeans(n_clusters=k)

    def __get_centroids(self, model):
        return model.cluster_centers_

    def __find_closest_args(self, centroids):
        centroid_min = 1e7
        cur_arg = -1
        args = {}
        used_idx = []
        for j, centroid in enumerate(centroids):
            for i, feature in enumerate(self.features):
                value = np.sum(np.abs(feature - centroid))
                if value < centroid_min and i not in used_idx:
                    cur_arg = i
                    centroid_min= value
            used_idx.append(cur_arg)
            args[j] = cur_arg
            centroid_min = 1e7
            cur_arg = -1
        return args

    def cluster(self, sentences_count = 1):
        # k = 1 if sentences_count * len(self.features) < 1 else int(len(self.features) * sentences_count)
        k = 1 if sentences_count < 1 else int(sentences_count)
        # print("k: ",k)
        model = self.__get_model(k).fit(self.features)
        centroids = self.__get_centroids(model)
        cluster_args = self.__find_closest_args(centroids)
        # print(cluster_args)
        sorted_values = sorted(cluster_args.values())
        # print(sorted_values)
        return sorted_values

    # def create_plots(self, k=4, plot_location='./kbert.png', title = ''):
    #     if self.pca_k != 2:
    #         raise RuntimeError("Must be dimension of 2")
    #     model = self.__get_model(k)
    #     model.fit(self.features)
    #     y = model.predict(self.features)
    #     plt.title(title)
    #     plt.scatter(self.features[:, 0], self.features[:, 1], c=y, s=50, cmap='viridis')
    #     centers = model.cluster_centers_
    #     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    #     plt.savefig(plot_location)


class KmeansBERTSummarizer(object):

    def __init__(self):
        self.bert_model = BertParent('bert', 'large')

    def __vote(self, arg_list):
        # print("arg_list: ",arg_list)
        all_tally = {}
        for args in arg_list:
            for arg in args:
                if arg in all_tally:
                    all_tally[arg] += 1
                else:
                    all_tally[arg] = 1
        # print("all_tally: ",all_tally)
        to_return = {k: v for k, v in all_tally.items() if v > 1}
        # print("TO_RETURN: ",to_return)
        return to_return

    def __call__(self, source_text, sentences_count = 1):
        
        paragraph_split = sent_tokenize(source_text)
        sentences = [i for i in paragraph_split]
        
        self.bert_hidden = self.bert_model.create_matrix(sentences, True)
        # print(self.bert_hidden.shape)
        
        # bc_non_hidden_args = ClusterFeatures(self.bert_non_hidden).cluster(sentences_count)
        bc_hidden_args = ClusterFeatures(self.bert_hidden).cluster(sentences_count)
        # print("bc_hidden_args: ",bc_hidden_args)
        
        # votes = self.__vote([bc_non_hidden_args, bc_hidden_args])

        # sorted_keys = sorted(bc_hidden_args.keys())
        sorted_keys = bc_hidden_args
        # print("SORTED KEYS: ",sorted_keys)
        if sorted_keys[0] != 0:
            sorted_keys.insert(0, 0)
        res = []
        for key in sorted_keys:
            res.append(key)
        
        summary = []
        for j in res:
            summary.append(sentences[j]) 
        # print(summary)
        return summary
    