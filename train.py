import torch
from model import Net
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from dataset import MedicalDataModule
from pytorch_lightning.loggers import WandbLogger
import numpy as np


torch.manual_seed(111)
np.random.RandomState(111)


net = Net(val_window_infer_batch_size=32)
dm = MedicalDataModule('subset_imgs', 'subset_masks', imgs_num=2, samples_per_img=16)

root_dir = ''
wandb_logger = WandbLogger()

trainer = Trainer(
    accelerator='gpu', 
    devices=[0],
    max_epochs=20000,
    callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
            ),
            EarlyStopping(monitor="val_loss", mode="min", patience=50)
        ],
    default_root_dir=root_dir,
    logger=wandb_logger,
)

trainer.fit(net, datamodule=dm)