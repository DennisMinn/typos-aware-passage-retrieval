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
        qid, pos_pid, neg_pid = qidpidtriples.items()
        query = self.queries[qid]
        pos = self.passages[pos_pid]
        neg = self.passages[neg_pid]
        
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
            'cls_id': torch.tensor(cls_id, dtype=torch.long),
            'sep_id': torch.tensor(sep_id, dtype=torch.long),
            'query_ids': torch.tensor(query_ids, dtype=torch.long),
            'query_mask': torch.tensor(query_mask, dtype=torch.long),
            'pos_ids': torch.tensor(pos_ids, dtype=torch.long),
            'pos_mask': torch.tensor(pos_mask, dtype=torch.long),
            'neg_ids': torch.tensor(neg_ids, dtype=torch.long),
            'neg_mask': torch.tensor(neg_mask, dtype=torch.long),
            'target': torch.tensor(label, dtype=torch.long),
        }


