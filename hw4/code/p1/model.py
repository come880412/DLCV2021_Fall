import torch.nn as nn

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )
        self.MLP = nn.Linear(1600, 256)
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.MLP(x)

        return x

class parametric_func(nn.Module):

    def __init__(self, n_way):
        super().__init__()
        self.distance = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, n_way)
        )
        
    
    def forward(self, x):
        x = self.distance(x)
        return x