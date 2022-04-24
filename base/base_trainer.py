import torch
from abc import abstractmethod
# from logger import TensorboardWriter

class BaseTrainer:
    '''
    Base class for all training
    '''
    def __init__(self, config, model, criterion, optimizer, metric, scheduler = None):
        '''
        TODO: implement initialization
        '''
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.scheduler = scheduler

    @abstractmethod
    def _train(self, batch):
        '''
        Training logic for one batch
        '''
        raise NotImplementedError

    def train(self, data_loader):
        '''
        TODO: implement training logic
        '''
        total_loss = 0.0
        for epoch in range(config['n_epochs']):
            for batch in data_loader:
                loss = self._train(batch)
                total_loss += loss

        return total_loss

    def _save_checkpoint(self, epoch, save_best = False):
        '''
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        
        TODO: implement checkpoint
        '''
        pass

    def _resume_checkpoint(self, checkpoint_path):
        '''
        :param checkpoint_path: checkpoint path to be resumed

        TODO: implement checkpoint resume
        '''
        pass
