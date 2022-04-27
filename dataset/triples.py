import torch

from transformers import AutoTokenizer
from base.base_dataset import BaseDataset

class TriplesDataset(BaseDataset):
    def __init__(self, collection, queries, qidpidtriples,
                 tokenizer, query_maxlen, passage_maxlen,
                 transformations = None):

        self.collection = collection
        self.queries = queries
        self.qidpidtriples = qidpidtriples

        self.tokenizer = tokenizer
        self.query_maxlen = query_maxlen
        self.passage_maxlen = passage_maxlen
        
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

        # TODO: apply textattack transformations
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        query = self.queries[self.qidpidtriples['qid'][index]]
        pos = self.passages[self.qidpidtriples['pos_pid'][index]]
        neg = self.passages[self.qidpidtriples['neg_pid'][index]]
        
        inputs_query = self.tokenizer.encode_plus(query,
                                                  truncation = True,
                                                  add_special_tokens = False,
                                                  max_length = self.query_maxlen,
                                                  padding = 'max_length')
        
        inputs_pos = self.tokenizer.encode_plus(pos,
                                                truncation = True,
                                                add_special_tokens = False,
                                                max_length = self.passage_maxlen,
                                                padding = 'max_length')
        
        inputs_neg = self.tokenizer.encode_plus(neg,
                                                truncation = True,
                                                add_special_tokens = False,
                                                max_length = self.passage_maxlen,
                                                padding = 'max_length')
        
        cls_id = self.cls_id
        sep_id = self.sep_id
        query_ids, query_mask = inputs_query['input_ids'], inputs_query['attention_mask']
        pos_ids, pos_mask = inputs_pos['input_ids'], inputs_pos['attention_mask']
        neg_ids, neg_mask = inputs_neg['input_ids'], inputs_neg['attention_mask']
        label = [1]

        return {
            'cls_id': cls_id,
            'sep_id': sep_id,
            'query_ids': query_ids,
            'query_mask': query_mask,
            'pos_ids': pos_ids,
            'pos_mask': pos_mask,
            'neg_ids': neg_ids,
            'neg_mask': neg_mask,
            'target': label,
        }


