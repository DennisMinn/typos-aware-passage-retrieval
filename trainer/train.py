from base.base_trainer import BaseTrainer
class Trainer(BaseTrainer):
    def _train(self, batch):
        query_pos_ids, query_pos_mask = batch['query_pos_ids'], batch['query_pos_mask']
        query_neg_ids, query_neg_mask = batch['query_neg_ids'], batch['query_neg_mask']
        labels = batch['labels']
        
        pos_output = self.model.forward(query_pos_ids, query_pos_mask)
        neg_output = self.model.forward(query_neg_ids, query_neg_mask)
        
        loss = self.criterion(pos_output, neg_output, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        return loss.item()
