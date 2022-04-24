import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from base.base_model import BaseModel

class CrossEncoder(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config['model_name'])
        self.encoder = AutoModel.from_config(self.model_config)
        self.fc = nn.Linear(self.model_config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        out = out.last_hidden_state[:, 0, :]
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        cls_id, sep_id = batch['cls_id'], batch['sep_id']
        query_ids, query_mask = batch['query_ids'], batch['query_mask']
        pos_ids, pos_mask = batch['pos_ids'], batch['pos_mask']
        neg_ids, neg_mask = batch['neg_ids'], batch['neg_mask']
        targets = batch['target']

        pos_input = CrossEncoder.triples_format(cls_id, sep_id,
                                         query_ids, query_mask,
                                         pos_ids, pos_mask)

        neg_input = CrossEncoder.triples_format(cls_id, sep_id,
                                         query_ids, query_mask,
                                         neg_ids, neg_mask)

        pos_score = self.forward(pos_input['input_ids'],
                                 pos_input['attention_mask'])

        neg_score = self.forward(neg_input['input_ids'],
                                 pos_input['attention_mask'])

        loss = F.margin_ranking_loss(pos_score, neg_score, targets)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pass

    @staticmethod
    def format(cls_id, sep_id, query_ids, query_mask, passage_ids, passage_mask):
        # did Reranker pad query then added sep token query+0s+sep+pass or query+sep+0s+pass
        n, q, p = query_ids.shape[0], query_ids.shape[1], passage_ids.shape[1]
        input_ids = torch.zeros(n, 1+q+1+p, dtype=torch.long) 
        attention_mask = torch.zeros(n, 1+q+1+p, dtype=torch.long)

        input_ids[:, 0] += cls_id
        input_ids[:, 1:q+1] += query_ids
        input_ids[:, q+1] += sep_id
        input_ids[:, q+2:] += passage_ids

        attention_mask[:, 0] += 1
        attention_mask[:, 1:q+1] += query_mask
        attention_mask[:, q+1] += 1
        attention_mask[:, q+2:] += passage_mask

        return {'input_ids': input_ids, 'attention_mask': attention_mask}
