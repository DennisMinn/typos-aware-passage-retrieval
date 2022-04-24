import os
import tqdm

import torch
import pytorch_lightning as pl

from abc import abstractmethod

class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, input_ids, attention_mask):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Training logic for one batch
        :return: dict containing loss and log
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Training logic for one batch in validation set
        """
        raise NotImplementedError


    #TODO implement lr scheduler
    #TODO add tensorboard log
    def configure_optimizers(self):
        """
        initializes optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    #def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #    pass
