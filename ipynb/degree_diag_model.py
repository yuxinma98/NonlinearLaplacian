import numpy as np
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
        for _ in range(num_layers-2):
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

# class DeepSet_equivariant_layer(nn.Module):
#     def __init__(self, in_channels:int, out_channels:int, hidden_channels:int, num_layers:int) -> None:
#         super(DeepSet_equivariant_layer, self).__init__()


#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             inputs (torch.Tensor): Shape (N, n, in_channels)

#         Returns:
#             torch.Tensor: Shape (N, n, out_channels)
#         """
#         n = inputs.size(1)
#         local_transformed = self.local_mlp(inputs) #N x n x hidden_channels
#         mean_feature = torch.mean(local_transformed, dim=-2) #N x hidden_channels
#         mean_feature = self.message_mlp(mean_feature) #N x in_channels
#         mean_feature = mean_feature.unsqueeze(dim=1).expand(-1,n,-1) #N x n x hidden_channels
#         feature = torch.cat([inputs, mean_feature], dim=-1) #N x n x (hidden_channels+in_channels)
#         return self.global_mlp(feature) #N x n x out_channels


class degree_diag_model(nn.Module):
    def __init__(self, params):
        super(degree_diag_model, self).__init__()
        self.params = params["model"]
        hidden_channels = self.params["hidden_channels"]
        num_layers = self.params["num_layers"]
        self.local_mlp = MLP(in_channels=1, 
                            out_channels=hidden_channels, 
                            num_layers=num_layers, 
                            hidden_channels=hidden_channels)
        self.message_mlp = MLP(in_channels=hidden_channels,
                               out_channels=1,
                               num_layers=num_layers,
                               hidden_channels=hidden_channels)
        self.global_mlp = MLP(in_channels=2,
                        out_channels=1,
                        num_layers=num_layers,
                        hidden_channels=hidden_channels)

        self.c = nn.Parameter(nn.init.uniform_(torch.empty(1), a=0, b=1))
        self.head = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): N x n x n

        Returns:
            torch.Tensor: N x 1
        """
        n = inputs.size(-1)
        inputs = (inputs*2 - torch.ones(n,n).to(inputs) + torch.eye(n).to(inputs)) / np.sqrt(n)
        degree = torch.sum(inputs, dim=-1).unsqueeze(-1) #N x n x 1

        local_transformed = self.local_mlp(degree) #N x n x hidden_channels
        if self.params["aggr"] == "mean":
            aggr_feature = torch.mean(local_transformed, dim=-2)  # N x hidden_channels
        elif self.params["aggr"] == "max":
            aggr_feature, _ = torch.max(local_transformed, dim=-2)
        elif self.params["aggr"] == "sqrt":
            aggr_feature = torch.sum(local_transformed, dim=-2) / np.sqrt(n)
        aggr_feature = self.message_mlp(aggr_feature)  # N x 1
        aggr_feature = aggr_feature.unsqueeze(dim=1).expand(-1, n, -1)  # N x n x 1
        x = torch.cat([degree, aggr_feature], dim=-1)  # N x n x 2
        x = nn.Tanh()(x)
        x = self.global_mlp(x) #N x n x 1
        x = torch.diag_embed(x.squeeze(dim=-1)) # N x n x n
        x = x + inputs * self.c
        A = x.clone()
        D = torch.linalg.eigvalsh(x) # N x n
        lambda_max, _ = torch.max(D, dim = -1) # N
        return A, self.head(lambda_max.unsqueeze(-1))


class pointwise_degree_diag_model(nn.Module):
    def __init__(self, params):
        super(pointwise_degree_diag_model, self).__init__()
        self.params = params["model"]
        hidden_channels = self.params["hidden_channels"]
        num_layers = self.params["num_layers"]
        self.mlp1 = MLP(
            in_channels=1,
            out_channels=1,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
        )
        self.mlp2 = MLP(
            in_channels=1,
            out_channels=1,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
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
        inputs = (inputs * 2 - torch.ones(n, n).to(inputs) + torch.eye(n).to(inputs)) / np.sqrt(n)
        degree = torch.sum(inputs, dim=-1).unsqueeze(-1)  # N x n x 1

        x = self.mlp1(degree)  # N x n x 1
        x = nn.Tanh()(x)
        x = self.mlp2(x)  # N x n x 1
        x = torch.diag_embed(x.squeeze(dim=-1))  # N x n x n
        x = x + inputs  # N x n x n
        A = x.clone()
        D = torch.linalg.eigvalsh(x)  # N x n
        lambda_max, _ = torch.max(D, dim=-1)  # N
        return A, self.head(lambda_max.unsqueeze(-1))
