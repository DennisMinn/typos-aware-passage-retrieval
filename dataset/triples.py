import torch

from transformers import AutoTokenizer
from base.base_dataset import BaseDataset

class TriplesDataset(BaseDataset):
    def __init__(self, dataframe, tokenizer, query_maxlen, passage_maxlen, transformations = None):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.query_maxlen = query_maxlen
        self.passage_maxlen = passage_maxlen
        
        self.queries = dataframe['query'].values
        self.pos_passages = dataframe['positive_passage'].values
        self.neg_passages = dataframe['negative_passage'].values
        
        # TODO: apply textattack transformations
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        query = self.queries[index]
        pos = self.pos_passages[index]
        neg = self.neg_passages[index]
        
        inputs_query = self.tokenizer.encode_plus(query,
                                                  truncation = True,
                                                  add_special_tokens = True,
                                                  max_length = self.query_maxlen,
                                                  padding = 'max_length')
        
        inputs_pos = self.tokenizer.encode_plus(pos,
                                                truncation = True,
                                                add_special_tokens = True,
                                                max_length = self.passage_maxlen,
                                                padding = 'max_length')
        
        inputs_neg = self.tokenizer.encode_plus(neg,
                                                truncation = True,
                                                add_special_tokens = True,
                                                max_length = self.passage_maxlen,
                                                padding = 'max_length')
        
        query_ids, query_mask = inputs_query['input_ids'], inputs_query['attention_mask']
        pos_ids, pos_mask = inputs_pos['input_ids'], inputs_pos['attention_mask']
        neg_ids, neg_mask = inputs_neg['input_ids'], inputs_neg['attention_mask']
        
        return {
            'query_ids': torch.tensor(query_ids, dtype = torch.long),
            'query_mask': torch.tensor(query_mask, dtype = torch.long),
            'pos_ids': torch.tensor(pos_ids, dtype = torch.long),
            'pos_mask': torch.tensor(pos_mask, dtype = torch.long),
            'neg_ids': torch.tensor(neg_ids, dtype = torch.long),
            'neg_mask': torch.tensor(neg_mask, dtype = torch.long)
        }


