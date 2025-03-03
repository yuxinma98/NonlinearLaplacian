import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import wandb
import matplotlib.pyplot as plt

from learning_from_data.model import CustomLaplacian
from learning_from_data.data import (
    PlantedSubmatrixDataset,
    NonnegativePCADataset,
    PlantedCliqueDataset,
)


def train(config):
    pl.seed_everything(config["training_seed"])
    model = NNTrainingModule(config)
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        mode="max",
        monitor="val_acc",
    )
    if config["logger"]:
        logger = WandbLogger(
            project=config["project"],
            name=config["name"],
            log_model=config["log_checkpoint"],
            save_dir=config["log_dir"],
        )
        logger.watch(model, log=config["log_model"], log_freq=50)
    trainer = pl.Trainer(
        callbacks=[model_checkpoint],  # , early_stop_callback],
        devices=1,
        max_epochs=config["max_epochs"],
        logger=logger if config["logger"] else None,
        enable_progress_bar=True,
    )
    trainer.fit(model)
    if config["logger"]:
        logger.experiment.unwatch(model)
    trainer.test(model, verbose=True, ckpt_path="best")
    wandb.finish()


class NNTrainingModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)  # log hyperparameters in wandb
        self.params = params
        self.model = CustomLaplacian(params["model"])
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")

    def prepare_data(self):
        if self.params["task"] == "planted_submatrix":
            self.dataset = PlantedSubmatrixDataset(
                N=self.params["N"], n=self.params["n"], beta=self.params["beta"]
            )
            self.large_dataset = PlantedSubmatrixDataset(
                N=self.params["test_N"], n=self.params["test_n"], beta=self.params["beta"]
            )
        elif self.params["task"] == "nonnegative_pca":
            self.dataset = NonnegativePCADataset(
                N=self.params["N"], n=self.params["n"], beta=self.params["beta"]
            )
            self.large_dataset = NonnegativePCADataset(
                N=self.params["test_N"], n=self.params["test_n"], beta=self.params["beta"]
            )
        elif self.params["task"] == "planted_clique":
            self.dataset = PlantedCliqueDataset(
                N=self.params["N"], n=self.params["n"], beta=self.params["beta"]
            )
            self.large_dataset = PlantedCliqueDataset(
                N=self.params["test_N"], n=self.params["test_n"], beta=self.params["beta"]
            )

    def setup(self, stage):
        """Train, val, test split."""
        N = self.params["N"]
        self.n_test = int(self.params["test_fraction"] * N)
        self.n_val = int(self.params["val_fraction"] * N)
        self.n_train = N - self.n_val - self.n_test
        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            [self.n_train, self.n_val, self.n_test],
            generator=torch.Generator().manual_seed(self.params["data_seed"]),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.params["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.params["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_data,
            batch_size=self.params["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        large_test_loader = DataLoader(
            self.large_dataset,
            batch_size=self.params["batch_size"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return [test_loader, large_test_loader]

    def forward(self, data):
        return self.model(data)

    def predict(self, data):
        return self.model(data) > 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.params["lr_patience"]
        )
        scheduler = {
            "scheduler": sch,
            "monitor": "val_loss",
            "frequency": 1,
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss, acc = self._compute_loss_and_metrics(batch, mode="train")
        self.log_dict({"train_loss": loss, "train_acc": acc}, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._compute_loss_and_metrics(batch, mode="val")
        self.log_dict({"val_loss": loss, "val_acc": acc}, batch_size=len(batch))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, acc, precision, recall = self._compute_loss_and_metrics(batch, mode="test")
        if dataloader_idx == 0:
            self.log_dict(
                {
                    "test_loss": loss,
                    "test_acc": acc,
                    "test_precision": precision,
                    "test_recall": recall,
                },
                batch_size=len(batch),
            )
        elif dataloader_idx == 1:
            self.log_dict(
                {
                    "large_test_loss": loss,
                    "large_test_acc": acc,
                    "large_test_precision": precision,
                    "large_test_recall": recall,
                },
                batch_size=len(batch),
            )

    def on_test_end(self):
        """plot learned sigma function."""
        self.model.eval()
        x = torch.range(-10, 10, 0.1).unsqueeze(1).to(self.device).float()
        y = self.model.mlp(x).detach()
        fname = self.params["log_dir"] / "learned_sigma.png"
        plt.plot(x.cpu().numpy(), y.cpu().numpy())
        plt.savefig(fname)
        plt.close()
        if self.params.get("logger", True):
            logger = self.logger
            logger.log_image(key="learned_sigma", images=[str(fname)])

    def _compute_loss_and_metrics(self, batch, mode="train"):
        p = self.forward(batch[0])
        preds = p > 0
        loss = self.bce_logit(p, batch[1].float())
        acc = self.accuracy(preds, batch[1])

        if mode == "test":
            precision = self.precision(preds, batch[1])
            recall = self.recall(preds, batch[1])
            return loss, acc, precision, recall
        return loss, acc
