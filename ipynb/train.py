import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from degree_diag_model import pointwise_degree_diag_model

class GNNTrainingModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params) # log hyperparameters in wandb
        model_name = params["model"].get("name", "GNN")
        model_dict = {
            # "GNN": GNN,
            # "degree_diag_model": degree_diag_model,
            "pointwise_degree_diag_model": pointwise_degree_diag_model,
        }
        self.model = model_dict[model_name](params)
        self.bce_logit = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.params = params

    def forward(self, data):
        A, p = self.model(data)
        p = p.squeeze(dim=-1)
        preds = (p > 0).float()
        return A, p, preds

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
        self.log_dict({"train_loss": loss, 
                       "train_acc": acc},
                       batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._compute_loss_and_metrics(batch, mode="val")
        self.log_dict({"val_loss": loss, 
                       "val_acc": acc},
                       batch_size=len(batch))

    def _plot_hist(self, key):
        n,k = self.params["n_range"][0], self.params["k_range"][0]
        G = nx.erdos_renyi_graph(n, self.params["noise_dist"]["p"])
        adj_matrix = torch.tensor(nx.adjacency_matrix(G).todense())
        A, p = self.model(adj_matrix.unsqueeze(0).float().to('cuda'))
        D1 = torch.linalg.eigvals(A).real.squeeze().detach().cpu().numpy()
        G = nx.erdos_renyi_graph(n, self.params["noise_dist"]["p"])
        vertices = np.random.choice(np.arange(n), k, replace=False)
        for index, u in enumerate(vertices):
            for v in vertices[index + 1 :]:
                G.add_edge(u, v)
        adj_matrix = torch.tensor(nx.adjacency_matrix(G).todense()).unsqueeze(0).to('cuda')
        A, p = self.model(adj_matrix)
        D2 = torch.linalg.eigvals(A).real.squeeze().detach().cpu().numpy()

        plt.figure()
        range = (min(D1.min(), D2.min()), max(D1.max(), D2.max()))
        plt.hist(D1, label='ER graph', bins = 20, range=range)
        plt.hist(D2, label='Planted graph', bins=20, alpha=0.5, range=range)
        plt.title('Histogram of Eigenvalues')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.params["log_dir"], "histogram.png"))
        plt.clf()
        if self.params.get("logger", True):
            logger = self.logger
            logger.log_image(key=key, images = [os.path.join(self.params["log_dir"], "histogram.png")])

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.params.get("log_hist", False) and self.current_epoch % 50 == 0:
            self._plot_hist("train_histogram")

    def test_step(self, batch, batch_idx):
        loss, acc, precision, recall = self._compute_loss_and_metrics(batch, mode="test")
        self.log_dict({"test_loss": loss,
                       "test_acc": acc,
                       "test_precision": precision,
                       "test_recall": recall},
                        batch_size=len(batch))

    def on_test_start(self):
        super().on_test_start()
        if self.params.get("log_hist", False):
            self._plot_hist("test_histogram")        

    def _compute_loss_and_metrics(self, batch, mode="train"):
        _, x, preds = self.forward(batch[0])
        loss = self.bce_logit(x, batch[1].float())
        acc = self.accuracy(preds, batch[1])

        if mode == "test":
            precision = self.precision(preds, batch[1])
            recall = self.recall(preds, batch[1])
            return loss, acc, precision, recall

        return loss, acc
