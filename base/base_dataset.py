import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from abc import abstractmethod

class BaseDataset(Dataset):
    '''
    Base class for all datasets
    '''
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def textattack_augment():
        '''
        Applies textattack augmentation on numpy.ndarray
        '''
        pass

