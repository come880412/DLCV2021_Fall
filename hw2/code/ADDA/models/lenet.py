"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn


class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [3 x 28 x 28]
            # output [48 x 12 x 12]
            nn.Conv2d(3, 48, kernel_size=5),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # 2nd conv layer
            # input [48 x 12 x 12]
            # output [64 x 4 x 4]
            nn.Conv2d(48, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(64 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 64 * 4 * 4))
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.classfier = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(250,10),
        )

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = self.classfier(feat)
        return out
