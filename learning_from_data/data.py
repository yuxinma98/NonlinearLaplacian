import torch
import numpy as np
from torch.utils.data import Dataset
from learning_from_data import data_dir

class PlantedSubmatrixDataset(Dataset):

    def __init__(self, N, n, beta, suffix=""):
        self.N, self.n, self.beta = N, n, beta
        self.k = int(np.sqrt(n) * beta)
        self.fname = data_dir / f"planted_submatrix_N={N}_n={n}_beta={beta}{suffix}.pt"
        try:
            self.A, self.y = torch.load(self.fname)
        except FileNotFoundError:
            self.generate_data()

    def generate_data(self):
        # generate noise matrix
        A = torch.randn(self.N, self.n, self.n)
        A = (A + A.transpose(-2, -1)) / np.sqrt(2)
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
        torch.save((self.A, self.y), self.fname)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.A[idx], self.y[idx]


class NonnegativePCADataset(Dataset):

    def __init__(self, N, n, beta, suffix=""):
        self.N, self.n, self.beta = N, n, beta
        self.k = int(np.sqrt(n) * beta)
        self.fname = data_dir / f"nonnegative_PCA_N={N}_n={n}_beta={beta}{suffix}.pt"
        try:
            self.A, self.y = torch.load(self.fname)
        except FileNotFoundError:
            self.generate_data()

    def generate_data(self):
        # generate noise matrix
        A = torch.randn(self.N, self.n, self.n)
        A = (A + A.transpose(-2, -1)) / np.sqrt(2)
        # labels: plant signal or not
        y = torch.bernoulli(0.5 * torch.ones(self.N))
        planted = (y == 1).nonzero().squeeze()
        if planted.numel() > 0:
            x = torch.randn(planted.numel(), self.n)
            x = torch.abs(x) / torch.norm(x, dim=-1, keepdim=True)
            A[planted] += self.beta * np.sqrt(self.n) * x.unsqueeze(2) @ x.unsqueeze(1)
        self.A, self.y = A / np.sqrt(self.n), y
        torch.save((self.A, self.y), self.fname)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.A[idx], self.y[idx]


class NonnegativePCARecoveryDataset(Dataset):

    def __init__(self, N, n, beta, suffix=""):
        self.N, self.n, self.beta = N, n, beta
        self.k = int(np.sqrt(n) * beta)
        self.fname = data_dir / f"nonnegative_PCA_recovery_N={N}_n={n}_beta={beta}{suffix}.pt"
        try:
            self.A, self.y = torch.load(self.fname)
        except FileNotFoundError:
            self.generate_data()

    def generate_data(self):
        # generate noise matrix
        A = torch.randn(self.N, self.n, self.n)
        A = (A + A.transpose(-2, -1)) / np.sqrt(2)
        x = torch.randn(self.N, self.n)
        x = torch.abs(x) / torch.norm(x, dim=-1, keepdim=True)
        A += self.beta * np.sqrt(self.n) * x.unsqueeze(2) @ x.unsqueeze(1)
        self.A, self.y = A / np.sqrt(self.n), x
        torch.save((self.A, self.y), self.fname)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.A[idx], self.y[idx]


class PlantedCliqueDataset(Dataset):

    def __init__(self, N, n, beta, suffix=""):
        self.N, self.n, self.beta = N, n, beta
        self.k = int(np.sqrt(n) * beta)
        self.fname = data_dir / f"planted_clique_N={N}_n={n}_beta={beta}{suffix}.pt"
        try:
            self.A, self.y = torch.load(self.fname)
        except FileNotFoundError:
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
            ] = 1
            A[:, torch.arange(self.n), torch.arange(self.n)] = 0
        self.A = (
            A * 2 - torch.ones(self.n, self.n) + torch.eye(self.n)
        ) /np.sqrt(self.n) # form signed adjacency matrix
        self.y = y
        torch.save((self.A, self.y), self.fname)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.A[idx], self.y[idx]
