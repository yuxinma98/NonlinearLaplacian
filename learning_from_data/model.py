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
    def __init__(self, params):
        super(CustomLaplacian, self).__init__()
        hidden_channels = params["hidden_channels"]
        num_layers = params["num_layers"]
        if params["model"] == "relu":
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
        elif params["model"] == "tanh":
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
        n = inputs.size(-1)
        degree = torch.sum(inputs, dim=-1).unsqueeze(-1)  # N x n x 1
        for layer in self.mlp:
            x = layer(degree)
        x = torch.diag_embed(x.squeeze(dim=-1))  # N x n x n
        x = x + inputs  # N x n x n
        D = torch.linalg.eigvalsh(x)  # N x n
        lambda_max, _ = torch.max(D, dim=-1)  # N
        return self.head(lambda_max.unsqueeze(-1)).squeeze(-1)
