from monai.networks.nets import UNet
from pytorch_lightning import LightningModule
import torch
from dice_coef import dice_coef
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.losses import DiceLoss
from torch import nn
import numpy as np


class Net(LightningModule):
    def __init__(self, val_window_infer_batch_size):
        super().__init__()

        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(8, 16, 32),
            strides=(2, 2),
            num_res_units=2,
        )

        self.dice_loss = DiceLoss()
        self.bce_logits_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))
        self.epoch_loss_values = []
        self.post_pred = AsDiscrete(threshold=0.5)
        self.best_val_dice = 0
        self.metric_values = []
        self.val_window_infer_batch_size = val_window_infer_batch_size
        self.save_hyperparameters()


    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-2, weight_decay=1e-3
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)


        sigmoid_outputs = torch.sigmoid(output)
        loss = self.dice_loss(sigmoid_outputs, labels) + 0.3 * self.bce_logits_loss(output, labels)

        preds = self.post_pred(sigmoid_outputs)
        dice = dice_coef(preds, labels)

        tensorboard_logs = {"train_loss": loss.item(), "dice_coef": dice}
        return {"loss": loss, "dice": dice, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_loss = avg_loss.detach().cpu().numpy()
        self.log("train_loss", float(avg_loss))

        dice_mean = torch.stack([x["dice"] for x in outputs]).mean()
        self.log("train_dice", dice_mean)

        self.epoch_loss_values.append(avg_loss)

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        roi_size = (48, 96, 96)
        outputs = sliding_window_inference(
            inputs=images,
            roi_size=roi_size,
            sw_batch_size=self.val_window_infer_batch_size,
            predictor=self.forward,
            overlap=0.75,
            mode = 'gaussian'
        )

        sigmoid_outputs = torch.sigmoid(outputs)
        loss = self.dice_loss(sigmoid_outputs, labels) + 0.3 * self.bce_logits_loss(outputs, labels)

        preds = self.post_pred(sigmoid_outputs)
        dice = dice_coef(preds, labels)
        pred_labels_pos_ratio = preds.sum()/labels.sum()
        intersection_ratio = ((preds == labels) & (labels == 1)).sum()/labels.sum()

        return {"val_loss": loss, 
                "val_number": len(outputs), 
                "dice": dice, 
                'pred_labels_pos_ratio':pred_labels_pos_ratio, 
                'intersection_ratio':intersection_ratio}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_loss = torch.tensor(val_loss / num_items)

        mean_pred_labels_pos_ratio = np.mean([o['pred_labels_pos_ratio'].cpu().numpy() for o in outputs])
        mean_intersection_ratio = np.mean([o['intersection_ratio'].cpu().numpy() for o in outputs])
        mean_val_dice = np.mean([o['dice'].cpu().numpy() for o in outputs])

        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        self.log("val_loss", mean_val_loss)
        self.log("val_dice", mean_val_dice)
        self.log('pred_labels_pos_ratio', mean_pred_labels_pos_ratio)
        self.log('val_intersection_ratio', mean_intersection_ratio)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current val dice: {mean_val_dice.item():.4f}"
            f"\nbest val dice: {self.best_val_dice.item():.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        return {"log": tensorboard_logs}
