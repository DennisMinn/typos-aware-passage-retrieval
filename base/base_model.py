import os
import tqdm

import pytorch_lightning as pl
from abc import abstractmethod

from dataset.triples import TriplesDataset

class BaseModel(pl.LightningModule):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
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

 
    def configure_optimizers(self):
        """
        initializes optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])
        return optimizer
