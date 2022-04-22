from transformers import AutoModel
from base.base_model import BaseModel

import torch
import torch.nn as nn

class CrossEncoder(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids = input_ids,
                           attention_mask = attention_mask).last_hidden_state[:, 0, :]

        output = self.fc(out)
        return output
    
    @staticmethod
    def format(batch):
        query_ids, query_mask = batch['query_ids'], batch['query_mask']
        pos_ids, pos_mask = batch['pos_ids'], batch['pos_mask']
        neg_ids, neg_mask = batch['neg_ids'], batch['neg_mask']
        cls_id, sep_id = batch['cls_id'], batch['sep_id']

        query_pos_ids, query_pos_mask = CrossEncoder._format(cls_id,
                                                sep_id,
                                                query_ids,
                                                query_mask,
                                                pos_ids,
                                                pos_mask,)

        query_neg_ids, query_neg_mask = CrossEncoder._format(cls_id,
                                                sep_id,
                                                query_ids,
                                                query_mask,
                                                neg_ids,
                                                neg_mask,)
        return {
            'query_pos_ids': torch.tensor(query_pos_ids, dtype = torch.long),
            'query_pos_mask': torch.tensor(query_pos_mask, dtype = torch.long),
            'query_neg_ids': torch.tensor(query_neg_ids, dtype = torch.long),
            'query_neg_mask': torch.tensor(query_neg_mask, dtype = torch.long),
        }        

    @staticmethod
    def _format(cls_id, sep_id, query_ids, query_mask, passage_ids, passage_mask):
        batch_size = query_ids.shape[0]
        seq_len = 1 + query_ids.shape[1] + 1 + passage_ids.shape[1]
        
        seq_ids, seq_mask = torch.zeros(batch_size, seq_len), torch.zeros(batch_size, seq_len)
        
        seq_ids[:, 0] += cls_id 
        seq_ids[:, 1:query_ids.shape[1]+1] += query_ids
        seq_ids[:, query_ids.shape[1]+1] += sep_id
        seq_ids[:, query_ids.shape[1]+2:] +=  passage_ids       
        
        seq_mask[:, 0] += 1
        seq_mask[:, 1:query_mask.shape[1]+1] += query_mask
        seq_mask[:, query_mask.shape[1]+1] += 1
        seq_mask[:, query_mask.shape[1]+2:] += passage_mask 
    
        return (seq_ids, seq_mask)
