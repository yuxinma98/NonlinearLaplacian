import torch
import torch.nn as nn
import torch.functional as F


class MLP(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        hidden_channels: int,
        nonlinearity: str = "relu",
    ) -> None:
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        if nonlinearity == "relu":
            nonlinearity_layer = nn.ReLU()
        elif nonlinearity == "tanh":
            nonlinearity_layer = nn.Tanh()

        assert num_layers > 1
        layers = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(num_layers - 2):
            layers.append(nonlinearity_layer)
            layers.append(nn.Linear(hidden_channels, hidden_channels))
        layers.append(nonlinearity_layer)
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (*, in_channels)

        Returns:
            torch.Tensor: Shape (*, out_channels)
        """
        return self.mlp(x)


class CustomLaplacian(nn.Module):

    def __init__(self, model, hidden_channels, num_layers, eigenvector=False):
        super(CustomLaplacian, self).__init__()
        self.eigenvector = eigenvector
        if model == "relu":
            self.mlp = nn.Sequential(
                MLP(
                    in_channels=1,
                    out_channels=hidden_channels,
                    num_layers=num_layers // 2,
                    hidden_channels=hidden_channels,
                    nonlinearity="relu",
                ),
                nn.Tanh(),
                MLP(
                    in_channels=hidden_channels,
                    out_channels=1,
                    num_layers=num_layers - num_layers // 2,
                    hidden_channels=hidden_channels,
                    nonlinearity="relu",
                ),
            )
        elif model == "tanh":
            self.mlp = nn.Sequential(
                MLP(
                    in_channels=1,
                    out_channels=1,
                    num_layers=num_layers,
                    hidden_channels=hidden_channels,
                    nonlinearity="tanh",
                )
            )
        self.head = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): N x n x n

        Returns:
            torch.Tensor: N x 1
        """

        if self.eigenvector:
            return self.compute_top_eig(inputs)
        else:
            lambda_max = self.compute_top_eig(inputs)
            return self.head(lambda_max.unsqueeze(-1)).squeeze(-1)

    def compute_top_eig(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): N x n x n

        Returns:
            torch.Tensor: N x n or N
        """
        d = torch.sum(inputs, dim=-1).unsqueeze(-1)  # N x n x 1
        d = self.mlp(d).squeeze(dim=-1)
        L = torch.diag_embed(d) + inputs  # N x n x n
        if self.eigenvector:
            evals, evecs = torch.linalg.eigh(L)
            return evecs[:, :, -1]
        else:
            evals = torch.linalg.eigvalsh(L)  # N x n
            return evals[:, -1]
