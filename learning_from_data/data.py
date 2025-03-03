import torch
import numpy as np
from torch.utils.data import Dataset


class PlantedSubmatrixDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.N, self.n = self.params["N"], self.params["n"]
        self.k = int(np.sqrt(self.n) * self.params["beta"])
        self.generate_data()

    def generate_data(self):
        # generate noise matrix
        A = torch.randn(self.N, self.n, self.n)
        A = A.tril(diagonal=0) + A.tril(diagonal=-1).transpose(-2, -1)
        # labels: plant submatrix or not
        y = torch.bernoulli(0.5 * torch.ones(self.N))
        planted = (y == 1).nonzero().squeeze()
        if planted.numel() > 0:
            clique_vertices = torch.multinomial(
                torch.ones(planted.numel(), self.n), self.k, replacement=False
            )  # planted.numel() x k
            A[
                planted.reshape(-1, 1, 1),
                clique_vertices.unsqueeze(2),
                clique_vertices.unsqueeze(1),
            ] += 1
        self.A, self.y = A / np.sqrt(self.n), y

    def __len__(self):
        return self.params["N"]

    def __getitem__(self, idx):
        return self.A[idx], self.y[idx]


class NonnegativePCADataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.N, self.n, self.beta = self.params["N"], self.params["n"], self.params["beta"]
        self.generate_data()

    def generate_data(self):
        # generate noise matrix
        A = torch.randn(self.N, self.n, self.n)
        A = A.tril(diagonal=0) + A.tril(diagonal=-1).transpose(-2, -1)
        # labels: plant signal or not
        y = torch.bernoulli(0.5 * torch.ones(self.N))
        planted = (y == 1).nonzero().squeeze()
        if planted.numel() > 0:
            x = torch.randn(planted.numel(), self.n)
            x = torch.abs(x) / torch.norm(x, dim=-1, keepdim=True)
            A[planted] += self.beta * x.unsqueeze(2) @ x.unsqueeze(1)
        self.A, self.y = A / np.sqrt(self.n), y

    def __len__(self):
        return self.params["N"]

    def __getitem__(self, idx):
        return self.A[idx], self.y[idx]


class PlantedCliqueDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.N, self.n = self.params["N"], self.params["n"]
        self.k = int(np.sqrt(self.n) * self.params["beta"])
        self.generate_data()

    def generate_data(self):
        # generate ER(0.5) graphs
        A = torch.bernoulli(0.5 * torch.ones(self.N, self.n, self.n))
        A = A.tril(diagonal=-1) + A.tril(diagonal=-1).transpose(-2, -1)
        # labels: plant clique or not
        y = torch.bernoulli(0.5 * torch.ones(self.N))
        planted = (y == 1).nonzero().squeeze()
        if planted.numel() > 0:
            clique_vertices = torch.multinomial(
                torch.ones(planted.numel(), self.n), self.k, replacement=False
            )  # planted.numel() x k
            A[
                planted.reshape(-1, 1, 1),
                clique_vertices.unsqueeze(2),
                clique_vertices.unsqueeze(1),
            ] += 1
            A = A - torch.diag(torch.diag(A))
        self.A = A * 2 - torch.ones(self.n, self.n) + torch.eye(self.n)
        self.A, self.y = A / np.sqrt(self.n), y

    def __len__(self):
        return self.params["N"]

    def __getitem__(self, idx):
        return self.A[idx], self.y[idx]
