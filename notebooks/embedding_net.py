import torch
import torch.nn as nn

class resnet_block(nn.Module):
    def __init__(self, n_channels, activation):
        super().__init__()
        self.activation = activation
        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_channels),
            self.activation,
            nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_channels),
        )

    def forward(self, x):
        identity = x
        out = self.layers(x)
        out += identity
        out = self.activation(out)
        return out

class EMBEDDING_NET(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.activation = nn.ELU()
        n_channels = 64
        self.conv_layers = nn.Sequential(
            nn.Conv2d(16, n_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(n_channels),
            self.activation,
            resnet_block(n_channels, self.activation),
            resnet_block(n_channels, self.activation),
            resnet_block(n_channels, self.activation),
            nn.AvgPool2d(2),
            nn.Conv2d(n_channels, n_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(n_channels),
            self.activation,
            resnet_block(n_channels, self.activation),
            resnet_block(n_channels, self.activation),
            resnet_block(n_channels, self.activation),
            nn.AvgPool2d(4),
            nn.Flatten(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(n_channels*2*2, n_channels),
            self.activation,
            nn.Linear(n_channels, dim),
            self.activation,
        )


    def forward(self, x):
        x = torch.log(1+x)
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x