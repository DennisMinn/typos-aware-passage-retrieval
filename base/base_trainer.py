import torch
from abc import abstractmethod
# from logger import TensorboardWriter

class BaseTrainer:
    '''
    Base class for all training
    '''
    def __init__(self, model, criterion, metrics, optimizer, scheduler, config):
        '''
        TODO: implement initialization
        '''
        pass

    @abstractmethod
    def _train(self, step):
        '''
        Training logic for one batch
        '''
        raise NotImplementedError

    def train(self):
        '''
        TODO: implement training logic
        '''
        pass

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
