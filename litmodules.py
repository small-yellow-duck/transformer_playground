import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader

from pytorch_lightning.core.lightning import LightningModule


def makeplot(fig, samples):
    #fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(sample.reshape(28, 28), cmap='Greys_r')
        #plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        plt.pause(0.01)

    #fig.canvas.draw()


class Dataseq(Dataset):
    """dataset."""

    def __init__(self, df):
        """
        Args:
            data: pandas dataframe
        """
        self.df = df
        #self.use_idx = use_idx

    def __len__(self):
        return self.df.shape[0]
        #return len(self.use_idx)

    def __getitem__(self, idx):
        #x = load_and_resample(self.train_df.loc[idx, 'filename'])

        # x = self.df.loc[self.use_idx[idx]][1:].values.reshape(1, 28, 28)
        # x = x/255.0
        # label = self.df.loc[self.use_idx[idx]][0]

        x = self.df.loc[idx][1:].values.reshape(1, 28, 28)
        x = x/255.0
        label = self.df.loc[idx][0]


        return torch.tensor(x).float(), torch.tensor(label)


class EDModule(LightningModule):
    def __init__(self, df, size=(28, 28), seed=0, batch_size=8):
        super().__init__()
        self.df = df
        self.size = size
        self.seed = seed
        self.batch_size = batch_size

        np.random.seed(self.seed)

        plt.close('all')
        self.fig = plt.figure(figsize=(4, 4))

        self.ds = Dataseq(df)

        #self.model = None #self.getmodel()

        #split train-val by row in data.csv
        folds, fold_groups = self.train_val_split()
        fold = 0

        self.train_idx = np.array(list(itertools.chain.from_iterable([folds[i] for i in fold_groups[fold]['train']])))
        self.val_idx = np.array(list(itertools.chain.from_iterable([folds[i] for i in fold_groups[fold]['val']])))
        #self.val_idx = self.val_idx[0:self.batch_size]


        self.train_ds = Subset(self.ds, self.train_idx)
        self.val_ds = Subset(self.ds, self.val_idx)

        print(f'train ds size: {self.train_ds.__len__()}')


    def train_val_split(self):
        data_idx = np.arange(self.df.shape[0])

        np.random.shuffle(data_idx)
        n_folds = 5
        fold_size = 1.0 * self.df.shape[0] / n_folds
        folds = [data_idx[int(i * fold_size):int((i + 1) * fold_size)] for i in range(6)]

        fold_groups = {}
        fold_groups[0] = {'train': [0, 1, 2, 3], 'val': [4]}
        fold_groups[1] = {'train': [1, 2, 3, 4], 'val': [0]}
        fold_groups[2] = {'train': [0, 2, 3, 4], 'val': [1]}
        fold_groups[3] = {'train': [0, 1, 3, 4], 'val': [2]}
        fold_groups[4] = {'train': [0, 1, 2, 4], 'val': [3]}

        return folds, fold_groups


    def collate_fn(self, batch):
        return batch


    def prepare_data(self):
        None


    def train_dataloader(self):
        kwargs = {'num_workers': 8, 'pin_memory': False} #if self.device == 'cuda' else {}
        print('train dataloader')
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, **kwargs)
        #return DataLoader(self.train_ds, batch_size=64, shuffle=True, **kwargs)

    def val_dataloader(self):
        kwargs = {'num_workers': 8, 'pin_memory': False} #if self.device == 'cuda' else {}
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, **kwargs)
        #return DataLoader(self.val_ds, batch_size=64, shuffle=False, **kwargs)