import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import wandb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from learning_from_data.model import CustomLaplacian
from learning_from_data.data import (
    PlantedSubmatrixDataset,
    PlantedCliqueDataset,
)


class DotProductLoss(nn.Module):
    def __init__(self):
        super(DotProductLoss, self).__init__()

    def forward(self, pred, y):
        y = y.double()  # N x n
        pred = pred.double()  # N x n
        dot = torch.bmm(pred.unsqueeze(dim=1), y.unsqueeze(dim=2))  # N x 1 x 1
        square = torch.pow(dot, 2)
        loss = 1 - square
        return loss.mean()


def train(config):
    pl.seed_everything(config["training_seed"])
    model = NNTrainingModule(config)
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        mode="min" if config["task"] == "nonnegative_pca_recovery" else "max",
        monitor="val_loss" if config["task"] == "nonnegative_pca_recovery" else "val_acc",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=100, mode="min")
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
        self.model = CustomLaplacian(
            **params["model"],
            eigenvector=True if params["task"] == "nonnegative_pca_recovery" else False
        )
        if params["task"] == "nonnegative_pca_recovery":
            self.loss = DotProductLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
            self.accuracy = torchmetrics.Accuracy(task="binary")
            self.precision = torchmetrics.Precision(task="binary")
            self.recall = torchmetrics.Recall(task="binary")

    def prepare_data(self):
        dataset_dict = {
            "planted_submatrix": PlantedSubmatrixDataset,
            "nonnegative_pca": NonnegativePCADataset,
            "nonnegative_pca_recovery": NonnegativePCARecoveryDataset,
            "planted_clique": PlantedCliqueDataset,
        }
        dataset = dataset_dict[self.params["task"]]
        self.dataset = dataset(N=self.params["N"], n=self.params["n"], beta=self.params["beta"])
        self.large_train_dataset = dataset(
            N=self.params["test_N"],
            n=self.params["test_n"],
            beta=self.params["beta"],
            suffix="_train",
        )
        self.large_test_dataset = dataset(
            N=self.params["test_N"],
            n=self.params["test_n"],
            beta=self.params["beta"],
            suffix="_test",
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
        large_train_loader = DataLoader(
            self.large_train_dataset,
            batch_size=20,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        large_test_loader = DataLoader(
            self.large_test_dataset,
            batch_size=20,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return [test_loader, large_train_loader, large_test_loader]

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
        if self.params["task"] == "nonnegative_pca_recovery":
            y_pred = self.forward(batch[0])
            loss = self.loss(y_pred, batch[1])
            self.log_dict({"train_loss": loss}, batch_size=len(batch))
        else:
            loss, acc = self._compute_loss_and_metrics(batch)
            self.log_dict({"train_loss": loss, "train_acc": acc}, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        if self.params["task"] == "nonnegative_pca_recovery":
            y_pred = self.forward(batch[0])
            loss = self.loss(y_pred, batch[1])
            self.log_dict({"val_loss": loss}, batch_size=len(batch))
        else:
            loss, acc = self._compute_loss_and_metrics(batch)
            self.log_dict({"val_loss": loss, "val_acc": acc}, batch_size=len(batch))

    def on_test_start(self):
        self.large_train_out, self.large_train_target = [], []
        self.large_test_out, self.large_test_target = [], []
        return super().on_test_start()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.params["task"] == "nonnegative_pca_recovery":
            y_pred = self.forward(batch[0])
            loss = self.loss(y_pred, batch[1])
            self.log_dict({"test_loss": loss}, batch_size=len(batch))
        else:
            if dataloader_idx == 0:
                loss, acc = self._compute_loss_and_metrics(batch)
                self.log_dict({"test_loss": loss, "test_acc": acc}, batch_size=len(batch))
            elif dataloader_idx == 1:
                self.large_train_out.append(self.model.compute_top_eig(batch[0]).reshape(-1, 1))
                self.large_train_target.append(batch[1])
            elif dataloader_idx == 2:
                self.large_test_out.append(self.model.compute_top_eig(batch[0]).reshape(-1, 1))
                self.large_test_target.append(batch[1])

    def on_test_end(self):
        # plot learned sigma function
        x = torch.arange(-10, 10, 0.1).unsqueeze(1).to(self.device).float()
        y = self.model.mlp(x).detach()
        fname = self.params["log_dir"] / "learned_sigma.png"
        plt.plot(x.cpu().numpy(), y.cpu().numpy())
        plt.savefig(fname)
        plt.close()
        if self.params.get("logger", True):
            logger = self.logger
            logger.log_image(key="learned_sigma", images=[str(fname)])

        if self.params["task"] == "nonnegative_pca_recovery":
            return
        # logistic regression on largest eigenvalue to learn the threshold
        train_x = torch.cat(self.large_train_out, dim=0).cpu().numpy()
        train_y = torch.cat(self.large_train_target, dim=0).cpu().numpy()
        test_x = torch.cat(self.large_test_out, dim=0).cpu().numpy()
        test_y = torch.cat(self.large_test_target, dim=0).cpu().numpy()
        model = LogisticRegression()
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        accuracy = accuracy_score(test_y, y_pred)
        if self.params.get("logger", True):
            logger = self.logger
            logger.log_metrics({"large_test_acc": accuracy})

    def _compute_loss_and_metrics(self, batch):
        p = self.forward(batch[0])
        preds = p > 0
        loss = self.loss(p, batch[1].float())
        acc = self.accuracy(preds, batch[1])
        return loss, acc
