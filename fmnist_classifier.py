import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import importlib as imp



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


import models
#import models_conv as models
import litmodules

device = 'cuda'
seed = 1



class Classifier(litmodules.EDModule):
    def __init__(self, df, size=(28, 28), d_model=64, d_ff=64, seed=0, batch_size=8, dropout=0.0, n_outputs=10):

        super().__init__(df, size=size, seed=seed, batch_size=batch_size)
        self.loss = nn.BCELoss()  # nn.L1Loss() #
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.d_model = d_model
        self.d_ff = d_ff
        self.model = self.getmodel()
        print(self.n_outputs)


    def getmodel(self):
        return models.Classifier(size=self.size, d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout, n_outputs=self.n_outputs)

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)

        y_onehot = torch.zeros_like(pred)
        y_onehot.scatter_(1, y.reshape(-1, 1), 1)

        loss = self.loss(pred, y_onehot)

        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self.model(x)

        y_onehot = torch.zeros_like(pred)
        y_onehot.scatter_(1, y.reshape(-1, 1), 1)

        return {'val_loss': self.loss(pred, y_onehot)}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #avg_prec = torch.stack([x['val_prec'] for x in outputs]).mean()

        metrics = {}
        metrics['avg_loss'] = avg_loss
        #metrics['avg_prec'] = avg_prec
        #print(f'val_avg_prec {avg_prec}')

        logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': logs}


    def validation_end(self, outputs):
        # OPTIONAL

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)




def main(df, seed=1):


    '''
    df = pd.read_csv('fashion-mnist_train.csv')
    ld = LitData(df, seed=seed)
    x, y = next(iter(model.train_dataloader()))
    h = model.model.enc(x.float(), None)
    p = model.model.dec(h)

    preds = model.model(x)
    plt.imshow(preds[1, 0].detach().cpu().numpy())
    '''

    pl.seed_everything(seed)
    #logger = pl_loggers.TensorBoardLogger('logs/')
    logger = pl_loggers.csv_logs.CSVLogger('logs_csv/', name='pairwise_add_emb_'+str(seed), version=None)


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.getcwd(), #'../working'
        save_top_k=2,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
    )

    trainer_args = {
        'gpus':1,
        'checkpoint_callback' : checkpoint_callback,
        'max_epochs' : 25,
        'limit_train_batches': 0.05,
        'logger': logger,
    }

    model = Classifier(df, d_model=64, d_ff=64, seed=seed, batch_size=16, dropout=0.0, n_outputs=10)

    #trainer = pl.Trainer(gpus=1, checkpoint_callback=checkpoint_callback, max_epochs= 70, limit_train_batches= 0.05)
    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model)


if __name__ == "__main__":
    df = pd.read_csv('fashion-mnist_train.csv')
    main(df, seed=1)
    main(df, seed=42)

    plt.close('all')