from base.base_model import BaseModel
import torch

class CrossEncoder(BaseModel):
    def __init__(self, model_name):
        pass

    def forward():
        pass
    
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
            'query_pos_ids': query_pos_ids,
            'query_pos_mask': query_pos_mask,
            'query_neg_ids': query_neg_ids,
            'query_neg_mask': query_neg_mask,
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
