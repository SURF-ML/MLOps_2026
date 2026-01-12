from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        # TODO: Build the MLP architecture
        # If you are up to the task, explore other architectures or model types
        # Hint: Flatten -> [Linear -> ReLU -> Dropout] * N_layers -> Linear
        c, h, w = input_shape
        in_features = int(c * h * w)

        layers = []
        prev = in_features
        for width in hidden_units:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=float(dropout_rate)))
            prev = int(width)

        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        x = x.view(x.size(0), -1)
        return self.net(x)

