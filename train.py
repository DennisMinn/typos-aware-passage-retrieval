# Utilities
import time
import pandas as pd
import tqdm as tqdm

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

# Huggingface
from transformers import AutoTokenizer, AutoModel, AdamW

# Repository 
from utils import *
from dataset.triples import TriplesDataset
from model.cross_encoder import CrossEncoder

if __name__ == "__main__":

    CONFIG = {
        'model_name': 'bert-base-uncased',
        'query_maxlen': 64,
        'passage_maxlen': 128,
        'batch_size': 8,
        'epochs': 16,
        'learning_rate': 2e-5,
    }

    print('Reading file')
    pidqidtriples = read_qidpidtriples('data/triples.train.small.tsv')
    collection = read_collection('data/collection.tsv')
    queries = read_queries('data/queries.tsv')

    print('Setting up')
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    triples_dataset = TriplesDataset(collection, queries, pidqidtriples, 
                                     tokenizer, CONFIG['query_maxlen'], CONFIG['passage_maxlen'])
    triples_dataloader = DataLoader(triples_dataset, batch_size=CONFIG['batch_size'])

    print('Training')
    trainer = pl.Trainer(fast_dev_run=False, gpus=1)
    model = CrossEncoder(**CONFIG)

    trainer.fit(model=model, train_dataloaders=triples_dataloader)
