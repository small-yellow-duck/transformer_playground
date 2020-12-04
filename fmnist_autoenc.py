import numpy as np
import os
import pandas as pd

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



class Autoencoder(litmodules.EDModule):
    def __init__(self, df, size=(28, 28), seed=0, batch_size=8):
        super().__init__(df, size=size, seed=seed, batch_size=batch_size)

        self.loss = nn.MSELoss()  # nn.L1Loss() #
        self.model = models.Autoencoder(size=self.size)

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch

        latent_dim = self.model.enc(x)
        pred = self.model.dec(latent_dim)

        return {'loss': self.loss(pred, x)}


    def validation_step(self, batch, batch_idx):
        x, y = batch

        latent_dim = self.model.enc(x)
        pred = self.model.dec(latent_dim)

        if batch_idx == 0:
            litmodules.makeplot(self.fig, torch.cat([x[0:8], pred[0:8]], dim=0).detach().cpu().numpy())

        #return {'val_loss': self.loss(pred, x), 'val_prec':torch.mean((y>0.5)*pred)}
        return {'val_loss': self.loss(pred, x)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #avg_prec = torch.stack([x['val_prec'] for x in outputs]).mean()

        metrics = {}
        metrics['avg_loss'] = avg_loss
        #metrics['avg_prec'] = avg_prec
        #print(f'val_avg_prec {avg_prec}')

        return {'avg_val_loss': avg_loss}


    def validation_end(self, outputs):
        # OPTIONAL

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)




def main(df):


    '''
    df = pd.read_csv('fashion-mnist_train.csv')
    ld = LitData(df, seed=seed)
    x, y = next(iter(model.train_dataloader()))
    h = model.model.enc(x.float(), None)
    p = model.model.dec(h)

    preds = model.model(x)
    plt.imshow(preds[1, 0].detach().cpu().numpy())
    '''

    seed = 42
    pl.seed_everything(seed)
    tb_logger = pl_loggers.TensorBoardLogger('logs/')


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
        'max_epochs' : 70,
        'limit_train_batches': 0.125,
        'logger': tb_logger
    }

    model = Autoencoder(df, seed=seed)

    #trainer = pl.Trainer(gpus=1, checkpoint_callback=checkpoint_callback, max_epochs= 70, limit_train_batches= 0.05)
    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model)


if __name__ == "__main__":
    df = pd.read_csv('fashion-mnist_train.csv')
    main(df)